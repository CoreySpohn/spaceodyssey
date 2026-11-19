"""Value and spend: what each star is worth to the planner at a given price of time.

A utility scores what the planner believes about one star (``LogScore``, ``Count``),
read as two beliefs: the forecast's and the verdict's, each from its own replay
(``planning_beliefs``). ``frontier`` turns a menu of actions into the reward of each:
the utility it is expected to add, averaged over the outcome cells the forecast
expects, with up to ``depth - 1`` further actions at the same star after it, each
charged at the price of time.

The planner prices exposure time at ``lam`` (reward per day). Each star offers a padded
menu of options, each with a reward and a cost in days, and always the null option
(observe nothing, reward and cost zero). At price ``lam`` a star is worth its best net
reward,

    V_i(lam) = max(0, max_m mask_m (r_m - lam c_m)),

a convex, nonincreasing, piecewise-linear function of ``lam``. Its negative slope is the
time the star would spend at that price. Axis names: ``S`` stars, ``M`` menu entries,
``Y`` outcome cells, ``Am`` the marks of a pool.
"""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PyTree

from spaceodyssey._numerics import EXACT, guarded_log, safe_ratio
from spaceodyssey.belief import Belief, count_pmf, mark_weights, pool_rate
from spaceodyssey.claim import Claim
from spaceodyssey.information import (
    expected_log_score,
    outcome_prob,
    require_point_draws,
    update_by_cell,
)
from spaceodyssey.types import AbstractObservationModel, Action, Atoms, Pool, as_index


class Frontier(eqx.Module):
    """One padded menu of options per star.

    Menus are padded to a shared ``M`` with ``mask``, so stars and worlds with shorter
    menus reuse the same compiled program. Padded entries are ignored; their reward and
    cost should still be finite. The null option is implied and is not stored.

    An option is a plan: a first action, and the follow-ups a lookahead takes after it
    (``frontier``). Its ``reward`` is the utility the plan is expected to add and its
    ``cost_d`` the time it is expected to use: the first action's cost plus the
    expected cost of the follow-ups, both at the price the plan was built at. So the
    dual and the spend see the time a plan really uses, and ``reward - lam cost_d`` is
    the plan's net value at that price and its tangent in the price elsewhere.

    Attributes:
        reward: reward of each option, shape ``(S, M)``.
        cost_d: expected time of each option in days, shape ``(S, M)``.
        mask: true where the option exists, shape ``(S, M)``.
    """

    reward: Float[Array, "S M"] = eqx.field(converter=jnp.asarray)
    cost_d: Float[Array, "S M"] = eqx.field(converter=jnp.asarray)
    mask: Bool[Array, "S M"] = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Check that the three arrays share one ``(S, M)`` shape."""
        if not self.reward.shape == self.cost_d.shape == self.mask.shape:
            raise ValueError(
                "reward, cost_d and mask must share one shape, got "
                f"{self.reward.shape}, {self.cost_d.shape} and {self.mask.shape}"
            )
        if self.reward.ndim != 2:
            raise ValueError(f"a frontier is (S, M), got shape {self.reward.shape}")


def _values(lam: Float[Array, ""], f: Frontier) -> Float[Array, " S"]:
    """Per-star value at price ``lam``, with the null option as column zero."""
    net = jnp.where(f.mask, f.reward - lam * f.cost_d, -jnp.inf)
    null = jnp.zeros((*net.shape[:-1], 1), net.dtype)
    return jnp.max(jnp.concatenate([null, net], axis=-1), axis=-1)


def value_and_spend(
    lam: Float[Array, ""], f: Frontier
) -> tuple[Float[Array, " S"], Float[Array, " S"]]:
    """Per-star value at price ``lam`` and the time each star spends there.

    The spend is ``-dV_i/dlam``, taken as one forward tangent of the value itself
    (``jax.jvp``), so the two cannot drift apart. Away from a kink it is the cost of the
    option the star chooses, follow-ups included. At a kink, where several options (the
    null among them) tie for the best net reward, ``max`` differentiates by averaging,
    so the spend is the mean cost of the tied options: one point of the
    subdifferential, not a proof that the budget clears (Boyd and Vandenberghe, Convex
    Optimization, sec. 3.2.3 and 5.1, for the pointwise maximum and the dual
    function).

    Args:
        lam: the price of time, reward per day, ``lam >= 0``.
        f: the frontier.

    Returns:
        The value of each star and its spend in days, each of shape ``(S,)``.
    """
    lam = jnp.asarray(lam, jnp.result_type(f.reward.dtype, f.cost_d.dtype, float))
    values, slope = jax.jvp(lambda x: _values(x, f), (lam,), (jnp.ones_like(lam),))
    return values, -slope


class AbstractUtility(eqx.Module):
    """What the planner's beliefs about one star are worth to it, a scalar.

    The reward of an action is the utility it is expected to add (``frontier``), so a
    utility matters only up to a constant.
    """

    @abc.abstractmethod
    def __call__(
        self, forecast: Belief, verdict: Belief, claim: Claim, star
    ) -> Float[Array, ""]:
        """The utility of the forecast's and the verdict's beliefs about one star.

        Args:
            forecast: the forecast's belief, one role (``R = 1``).
            verdict: the verdict's belief, one role per member.
            claim: an unconditional claim.
            star: a pytree of scalar leaves describing the star.

        Returns:
            The utility, a scalar.
        """


class LogScore(AbstractUtility):
    """The verdict's log score on the claim-true count, expected under the forecast.

    ``U = sum_n P_F(N = n) ln P_gamma(N = n)``, with ``N`` the number of claim-true
    planets at the star (``count_pmf``), ``F`` the forecast and ``gamma`` the verdict
    member, each read off its own belief. Planning reads one member (a declared
    deviation: a minimum over members would not be a proper score), so the verdict's
    belief has ``R = 1``.

    One action's expected gain in ``U`` is the verdict gain
    ``E_F[ln p_gamma(N | Y) - ln p_gamma(N)]`` over its outcome cells ``Y``: averaged
    over the forecast's cells, the forecast's posterior counts average back to its
    prior count. It equals the forecast's mutual information of ``N`` and ``Y`` only
    when forecast and verdict agree.

    Attributes:
        n_max: the count at which ``count_pmf`` lumps the tail, static.
    """

    n_max: int = eqx.field(static=True)

    def __call__(self, forecast, verdict, claim, star):
        """The forecast-expected verdict log score of the count.

        Raises:
            ValueError: if the verdict's belief does not have exactly one member.
        """
        pmf = count_pmf(verdict, claim, star, self.n_max)
        if pmf.shape[0] != 1:
            raise ValueError(
                f"LogScore plans with one verdict member, not {pmf.shape[0]}"
            )
        expected = count_pmf(forecast, claim, star, self.n_max)[0]
        return expected_log_score(pmf[0], expected)


class Count(AbstractUtility):
    """Expected number of claim-true planets found since time zero, under the forecast.

    ``U = rate0 - rate``, the forecast's claim-true pool rate (``pool_rate``) in the
    pool at time zero less the rate in the forecast belief's pool. A planet found is
    one no longer in the pool, so a second look at a star counts only what the first
    look missed: ``thin`` has already taken the found planets out of the pool. The
    verdict is not read.

    Attributes:
        pool0: the pool at time zero, after the grant, its row 0 the forecast's; held
            beside the beliefs, not in them.
    """

    pool0: Pool

    def __call__(self, forecast, verdict, claim, star):
        """The claim-true rate at time zero less the rate now, for the forecast."""
        now = pool_rate(forecast.pool, claim, star)[0]
        return pool_rate(self.pool0, claim, star)[0] - now


class Menu(eqx.Module):
    """The options each star offers, padded to a shared length ``M``.

    Entry ``m`` is one observing mode at every star (``mode[m]``), with its own
    exposure at each star, so the menus of all stars share one structure and a star
    with fewer options masks the rest. An entry's model is its mode's, read from the
    design's one table (``mode_models``). Its cost is its exposure: no mode charges an
    overhead yet. A masked entry is still evaluated: keep its exposure finite.

    Availability by stage: each star is at an integer stage of its follow-up chain
    (0 before anything is found), and an entry is offered at a star only once the star
    has reached the entry's ``requires_stage`` (a spectrum, which needs a planet to
    point at, requires the stage a detection reaches). At the root a star is at its
    ``stage``; at a hypothetical step of the lookahead it is at the highest stage an
    outcome on the path to it reached: an outcome other than cell 0 reaches its
    model's ``stage``. ``mask`` says what a star offers at all.

    Attributes:
        exposure_d: exposure of each entry at each star, days, shape ``(S, M)``.
        mask: true where the entry is offered, shape ``(S, M)``.
        mode: the mode of each entry, static, one per entry.
        requires_stage: the stage each entry needs, static, one per entry; 0 (no
            need) by default.
        stage: the stage each star has reached now, shape ``(S,)``; 0 by default.

    Raises:
        ValueError: if the fields disagree on ``(S, M)``, or ``mode`` or
            ``requires_stage`` has not one entry per column.
    """

    exposure_d: Float[Array, "S M"] = eqx.field(converter=jnp.asarray)
    mask: Bool[Array, "S M"] = eqx.field(converter=jnp.asarray)
    mode: tuple[int, ...] = eqx.field(static=True)
    requires_stage: tuple[int, ...] = eqx.field(static=True)
    stage: Int[Array, " S"] = eqx.field(converter=as_index)

    def __init__(self, exposure_d, mask, mode, requires_stage=None, stage=None):
        """A menu; by default no entry needs a stage and every star is at stage 0."""
        self.exposure_d = exposure_d
        self.mask = mask
        self.mode = tuple(int(i) for i in mode)
        if requires_stage is None:
            requires_stage = (0,) * len(self.mode)
        self.requires_stage = tuple(int(x) for x in requires_stage)
        if stage is None:
            stage = jnp.zeros(jnp.shape(mask)[:-1], int)
        self.stage = stage

    def __check_init__(self):
        """Check that every field has the shape ``(S, M)`` with ``M = len(mode)``."""
        n_entry = (len(self.mode),)
        if (
            self.exposure_d.shape != self.mask.shape
            or self.mask.shape[-1:] != n_entry
            or (len(self.requires_stage),) != n_entry
            or self.stage.shape != self.mask.shape[:-1]
        ):
            raise ValueError(
                f"menu fields disagree on (S, M): {self.exposure_d.shape}, "
                f"{self.mask.shape}, {len(self.mode)} modes, "
                f"{len(self.requires_stage)} stage needs, "
                f"stage {self.stage.shape}"
            )


def _offered(menu: Menu, stage: Int[Array, "..."]) -> Bool[Array, "... M"]:
    """The entries a star offers at a stage: those it offers whose stage it has."""
    return menu.mask & (jnp.asarray(menu.requires_stage) <= stage[..., None])


def thin(pool: Pool, detect_prob: Float[Array, " Am"]) -> Pool:
    """The pool after an observation that would detect each mark with ``detect_prob``.

    The planets an observation misses are the pool thinned by the miss probability
    ``1 - d`` (a Poisson process thinned independently is a Poisson process). Role
    ``r``'s expected number is multiplied by ``1 - dbar_r``, with ``dbar_r`` the
    detection probability averaged under that role's own mark weights, and each mark's
    likelihood by ``1 - d``. The planets found leave the pool, whatever the observation
    shows; they become tracks only through a refit.

    A role whose every mark is detected with certainty is left with an empty pool
    (``log_mass = -inf``) and its mark weights as they were, so its rate is zero, not
    NaN. ``thin(pool, 1 - d)`` is the pool of the planets found.

    Args:
        pool: the undetected planets.
        detect_prob: probability that the observation detects each mark, ``(Am,)``.

    Returns:
        The planets the observation would miss.
    """
    miss = 1.0 - detect_prob
    kept = jnp.matmul(mark_weights(pool), miss, precision=EXACT)
    emptied = (kept <= 0)[:, None]
    return Pool(
        pool.marks,
        pool.log_prior_ratio,
        jnp.where(emptied, pool.log_like, pool.log_like + guarded_log(miss)),
        pool.log_mass + guarded_log(kept),
    )


def _detect_prob(
    model: AbstractObservationModel, marks: Atoms, action: Action
) -> Float[Array, " Am"]:
    """Probability that an action detects each mark.

    For a searching model, every cell but the miss, cell 0; zero for a model that does
    not search.
    """
    if not model.searches:
        return jnp.zeros(marks.exists.shape[0])
    return jnp.sum(model.cells(marks, action)[:, 1:], axis=-1)


def completeness(
    pool: Pool,
    model: AbstractObservationModel,
    action: Action,
    claim: Claim,
    star,
) -> Float[Array, " R"]:
    """Expected fraction of the pool's claim-true planets an action would detect.

    A mark is detected when a searching model's outcome on it falls outside cell 0, the
    miss; a model that does not search (``searches`` false) detects nothing and has
    completeness zero. Per role ``r`` this is
    ``sum_m w_r(m) p(m) d(m) / sum_m w_r(m) p(m)``, with ``w_r`` the role's mark
    weights, ``p`` the claim probability and ``d`` the detection probability of each
    mark: the claim-true rate of the planets found (the pool thinned by the miss
    probability) over the claim-true rate of the pool. Zero for a pool with no
    claim-true rate.

    Args:
        pool: the undetected planets.
        model: the observation model of the action.
        action: the action.
        claim: an unconditional claim.
        star: a pytree of scalar leaves describing the star.

    Returns:
        The completeness per role, shape ``(R,)``.
    """
    detect = _detect_prob(model, pool.marks, action)
    found = pool_rate(thin(pool, 1.0 - detect), claim, star)
    return safe_ratio(found, pool_rate(pool, claim, star))


def _entries(menu: Menu, mode: int, star_index) -> tuple[list[int], Action]:
    """The menu entries of one mode at one star, and their actions (leading axis)."""
    entries = [m for m, e in enumerate(menu.mode) if e == mode]
    exposure_d = menu.exposure_d[jnp.asarray(entries, int)]
    shape = exposure_d.shape
    actions = Action(jnp.full(shape, star_index), jnp.full(shape, mode), exposure_d)
    return entries, actions


class _Outcomes(eqx.Module):
    """Every menu entry's outcome cells at one star, read once at the root.

    On point draws an outcome changes only the weights and the pool, never the draws
    or the marks, so each entry's cells and detection probabilities are the same at
    every node of the lookahead. Cells are padded with zero-probability cells to the
    widest entry's ``Y``, so the entries of every mode share one shape.

    Attributes:
        forecast: cell probabilities under the forecast's draws, ``(M, A_F, Y)``.
        verdict: cell probabilities under the verdict's draws, ``(M, A_V, Y)``.
        detect_forecast: detection probability of the forecast pool's marks,
            ``(M, Am)``; zero for a model that does not search.
        detect_verdict: the same for the verdict's pool, ``(M, Am)``.
        reached: the stage each cell reaches, ``(M, Y)``: the entry's model's
            ``stage`` for every cell but cell 0, and 0 there.
    """

    forecast: Float[Array, "M AF Y"]
    verdict: Float[Array, "M AV Y"]
    detect_forecast: Float[Array, "M AmF"]
    detect_verdict: Float[Array, "M AmV"]
    reached: Int[Array, "M Y"]


def _outcomes(forecast, verdict, mode_models, menu, star_index) -> _Outcomes:
    """The cells and detection probabilities of every entry, in menu order."""
    order, parts = [], []
    for mode, model in enumerate(mode_models):
        entries, actions = _entries(menu, mode, star_index)
        if not entries:
            continue

        def one(action, model=model):
            return (
                model.cells(forecast.atoms, action),
                model.cells(verdict.atoms, action),
                _detect_prob(model, forecast.pool.marks, action),
                _detect_prob(model, verdict.pool.marks, action),
            )

        cells_f, cells_v, detect_f, detect_v = jax.vmap(one)(actions)
        n_cell = cells_f.shape[-1]
        reached = jnp.where(jnp.arange(n_cell) > 0, model.stage, 0)
        order += entries
        parts.append((cells_f, cells_v, detect_f, detect_v, reached, n_cell))
    width = max(part[-1] for part in parts)

    def pad(x):
        return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, width - x.shape[-1])])

    back = jnp.asarray(sorted(range(len(order)), key=order.__getitem__))
    cells_f, cells_v, detect_f, detect_v, reached = (
        jnp.concatenate(column)[back]
        for column in zip(
            *[
                (pad(f), pad(v), df, dv, pad(jnp.broadcast_to(d, (f.shape[0], n))))
                for f, v, df, dv, d, n in parts
            ],
            strict=True,
        )
    )
    return _Outcomes(cells_f, cells_v, detect_f, detect_v, reached)


def _after(b: Belief, pool: Pool, cell_prob: Float[Array, " A"]) -> Belief:
    """A belief after a hypothetical outcome: its pool thinned, its weights updated."""
    return update_by_cell(eqx.tree_at(lambda x: x.pool, b, pool), cell_prob)


def _charge(plan, stage) -> Float[Array, " M"]:
    """The price of each entry's time at a node, infinite where it is not offered."""
    offered = _offered(plan.menu, stage)
    return jnp.where(offered, plan.lam * plan.menu.exposure_d, jnp.inf)


def _worth(node, plan, depth: int) -> Float[Array, ""]:
    """The utility of a node with up to ``depth`` more actions, each chosen to raise it.

    A node is the pair of beliefs (forecast, verdict) and the stage the star has
    reached on the path to it. ``W_0 = U`` and
    ``W_k = max(U, max_m (E_F[W_{k-1}(after m)] - charge_m))``, with ``charge_m`` the
    price of entry ``m``'s time (infinite for an entry not offered there): stopping
    is always an option, so ``W_k`` never falls below ``W_{k-1}``.
    """
    forecast, verdict, stage = node
    now = plan.utility(forecast, verdict, plan.claim, plan.star)
    if depth == 0:
        return now
    later = _expected_after(node, plan, depth - 1)
    return jnp.maximum(now, jnp.max(later - _charge(plan, stage)))


def _expected_after(node, plan, depth: int) -> Float[Array, " M"]:
    """``E_F[W_depth(node after m)]`` for each entry ``m``, over the forecast's cells.

    The cells are the root's (``_Outcomes``): on point draws an outcome changes only
    the weights and the pool. After entry ``m`` with outcome in cell ``y`` each belief
    is reweighted by its draws' probability of the cell (``update_by_cell``) and its
    pool is thinned by the entry's detection probability in every cell; the star
    moves to the higher of its stage and the stage the cell reaches. Every entry
    and cell is one body under ``vmap``, so a level of the lookahead traces once. A
    cell the forecast gives no mass (a padded cell among them) adds nothing.
    """
    forecast, verdict, stage = node

    def entry(cells_f, cells_v, detect_f, detect_v, reached):
        p_cell = outcome_prob(forecast, cells_f)[0]  # (Y,), the forecast's
        seen = p_cell > 0
        pool_f, pool_v = thin(forecast.pool, detect_f), thin(verdict.pool, detect_v)

        def worth_after(cf, cv, reached_y):
            after = (
                _after(forecast, pool_f, cf),
                _after(verdict, pool_v, cv),
                jnp.maximum(stage, reached_y),
            )
            return _worth(after, plan, depth)

        safe_f, safe_v = (jnp.where(seen, c, 1.0) for c in (cells_f, cells_v))
        worth = jax.vmap(worth_after, in_axes=(1, 1, 0))(safe_f, safe_v, reached)
        return jnp.sum(jnp.where(seen, p_cell * worth, 0.0))

    o = plan.outcomes
    return jax.vmap(entry)(
        o.forecast, o.verdict, o.detect_forecast, o.detect_verdict, o.reached
    )


class _Plan(eqx.Module):
    """What the recursion at one star reads besides the node.

    The claim, the star, the utility, the menu, each entry's outcomes, and the price
    of time.
    """

    claim: Claim
    star: PyTree
    utility: AbstractUtility
    menu: Menu
    outcomes: _Outcomes
    lam: Float[Array, ""]


def frontier(
    forecast: Belief,
    verdict: Belief,
    claim: Claim,
    star,
    utility: AbstractUtility,
    mode_models: tuple[AbstractObservationModel, ...],
    menu: Menu,
    lam: Float[Array, ""],
    depth: int,
) -> Frontier:
    """The reward and expected time of each menu entry's plan at each star.

    The planner's beliefs are two: the forecast's and the verdict's, each from its own
    replay (``planning_beliefs``). Entry ``m``'s plan is the entry and up to
    ``depth - 1`` further entries of the same star's menu after it, each chosen on the
    outcomes so far to raise the utility net of its time at price ``lam`` (``lam``
    times its cost), or none. Its net value is ``E_F[W_{depth-1}(after m)] - U``, the
    utility ``U`` it is expected to add less the price of its follow-ups. The
    expectation is over the forecast's distribution of each outcome cell; an outcome
    reweights each belief by ``update_by_cell`` and thins its pool by the entry's
    detection probability of each mark (``thin``; zero for a model that does not
    search).

    The frontier holds each plan's gross reward and its expected time. The expected
    time of the follow-ups ``F_m`` is the negative slope of the net value in the price
    (envelope theorem), taken as one forward tangent (``jax.jvp``); the reward is the
    net value plus ``lam F_m`` and the cost the entry's exposure plus ``F_m``. So the
    dual counts the time a plan uses, not its first action alone, and at ``lam`` the
    frontier's net value of each option is the plan's. Under ``LogScore`` at depth 1
    the reward is the verdict gain ``E_F[ln p_gamma(N | Y) - ln p_gamma(N)]`` and the
    cost the exposure; ``lam`` enters only at ``depth > 1``, so the tangent is taken
    only there.

    An entry is offered only where the star has reached the entry's
    ``requires_stage``: at the root the menu's ``stage``, below it the highest stage
    an outcome on the path reached (``Menu``).

    The price a frontier is built at is not the price the dual then solves for. A
    planner (``PricePolicy``) passes the last solved price: one step of the fixed point
    ``lam = solve_price(frontier(..., lam, ...))``, exact at depth 1, where the reward
    does not depend on ``lam``. Between the two, the frontier's options are the
    tangents of the plans' net values at ``lam``.

    This is the generic bounded recursion, the reference: it evaluates
    ``(M Y) ** depth`` hypothetical belief pairs per star, so it serves small menus and
    shallow depths. The beliefs hold point draws (``planning_beliefs`` splits an
    uncertain gas block into quadrature sub-draws, ``quadrature_draws``); given a point
    draw the outcomes of rows are independent, so every entry's cells are read once at
    the root and a hypothetical outcome updates a belief exactly by reweighting. A
    belief with an uncertain block is refused (``require_point_draws``).

    A limit of the hypothetical update: a planet a hypothetical search finds is not
    spawned as a track (that takes a refit), so it leaves the count; value a search
    with ``Count``, which counts it as found.

    Every leaf of the two beliefs, ``star``, ``menu`` and the array leaves of
    ``utility`` (``Count``'s time-zero pool) carry a leading star axis ``S``; ``claim``
    and ``mode_models`` are shared by the stars. The menu's modes, the cell counts and
    ``depth`` are static, so another star or world of the same shapes reuses the
    compiled program.

    Args:
        forecast: the forecast's beliefs, one per star, one role each.
        verdict: the verdict's beliefs, one per star, one role each (planning reads
            one member).
        claim: an unconditional claim.
        star: the stars, a pytree whose leaves have a leading axis ``S``.
        utility: the utility.
        mode_models: the observation model of each mode (``mode_models`` of the
            design), static; ``menu.mode`` indexes it.
        menu: the menu of each star.
        lam: the price of the follow-up entries' time, reward per day, a scalar shared
            by the stars, ``lam >= 0``.
        depth: the number of entries a plan may take at the star, the first one
            included, static, at least 1.

    Returns:
        The frontier: each plan's reward and expected time (zero and the exposure at
        an entry not offered), and the entries offered at the root.

    Raises:
        ValueError: if ``depth < 1`` or a menu entry names a mode not in
            ``mode_models``.
    """
    if depth < 1:
        raise ValueError(f"depth is at least 1, not {depth}")
    if any(not 0 <= m < len(mode_models) for m in menu.mode):
        raise ValueError("a menu entry's mode is not in the mode table")

    lam = jnp.asarray(lam, float)
    forecast, verdict = require_point_draws(forecast), require_point_draws(verdict)

    def one_star(forecast, verdict, star, utility, menu, star_index):
        node = (forecast, verdict, menu.stage)
        offered = _offered(menu, menu.stage)
        now = utility(forecast, verdict, claim, star)
        outcomes = _outcomes(forecast, verdict, mode_models, menu, star_index)

        def net(price):
            plan = _Plan(claim, star, utility, menu, outcomes, price)
            return jnp.where(offered, _expected_after(node, plan, depth - 1) - now, 0.0)

        if depth == 1:  # no follow-up: the net value does not depend on the price
            value, follow_d = net(lam), jnp.zeros_like(menu.exposure_d)
        else:
            value, slope = jax.jvp(net, (lam,), (jnp.ones_like(lam),))
            follow_d = jnp.where(offered, -slope, 0.0)
        return value + lam * follow_d, menu.exposure_d + follow_d, offered

    star_index = jnp.arange(menu.mask.shape[0])
    reward, cost_d, offered = eqx.filter_vmap(one_star)(
        forecast, verdict, star, utility, menu, star_index
    )
    return Frontier(reward, cost_d, offered)
