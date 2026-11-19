"""Value and spend: what each star is worth to the planner at a given price of time.

A utility scores a belief about one star (``LogScore``, ``Count``). ``frontier`` turns a
menu of actions into the reward of each: the utility it is expected to add, averaged
over the outcome cells the forecast expects, with up to ``depth - 1`` further actions at
the same star after it, each charged at the price of time.

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
from jaxtyping import Array, Bool, Float

from spaceodyssey.belief import Belief, count_pmf, mark_weights, pool_rate
from spaceodyssey.claim import Claim
from spaceodyssey.information import expected_log_score, outcome_prob, update_by_cell
from spaceodyssey.types import AbstractObservationModel, Action, Atoms, Pool


class Frontier(eqx.Module):
    """One padded menu of options per star.

    Menus are padded to a shared ``M`` with ``mask``, so stars and worlds with shorter
    menus reuse the same compiled program. Padded entries are ignored; their reward and
    cost should still be finite. The null option is implied and is not stored.

    ``cost_d`` is the first action's cost alone. A reward that looks past the first
    action was built at one price of time (``frontier``); the frontier holds it fixed,
    so the dual and the spend see the price only through ``cost_d``.

    Attributes:
        reward: reward of each option, shape ``(S, M)``.
        cost_d: cost of each option in days, shape ``(S, M)``.
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
    option the star chooses. At a kink, where several options (the null among them)
    tie for the best net reward, ``max`` differentiates by averaging, so the spend is
    the mean cost of the tied options: one point of the subdifferential, not a proof
    that the budget clears (Boyd and Vandenberghe, Convex Optimization, sec. 3.2.3
    and 5.1, for the pointwise maximum and the dual function).

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
    """What a belief about one star is worth to the planner, a scalar.

    The reward of an action is the utility it is expected to add (``frontier``), so a
    utility matters only up to a constant.
    """

    @abc.abstractmethod
    def __call__(self, b: Belief, claim: Claim, star) -> Float[Array, ""]:
        """The utility of a belief.

        Args:
            b: the belief about one star.
            claim: an unconditional claim.
            star: a pytree of scalar leaves describing the star.

        Returns:
            The utility, a scalar.
        """


class LogScore(AbstractUtility):
    """The verdict's log score on the claim-true count, expected under the forecast.

    ``U(b) = sum_n P_F(N = n) ln P_gamma(N = n)``, with ``N`` the number of claim-true
    planets at the star (``count_pmf``), ``F`` the forecast (row 0) and ``gamma`` the
    verdict member (row 1). Planning reads one member (a declared deviation: a minimum
    over members would not be a proper score), so the belief has ``R = 2``.

    One action's expected gain in ``U`` is the verdict gain
    ``E_F[ln p_gamma(N | Y) - ln p_gamma(N)]`` over its outcome cells ``Y``: averaged
    over the forecast's cells, the forecast's posterior counts average back to its
    prior count. It equals the forecast's mutual information of ``N`` and ``Y`` only
    when forecast and verdict agree.

    Attributes:
        n_max: the count at which ``count_pmf`` lumps the tail, static.
    """

    n_max: int = eqx.field(static=True)

    def __call__(self, b, claim, star):
        """The forecast-expected verdict log score of the count.

        Raises:
            ValueError: if the belief does not have exactly one verdict member.
        """
        pmf = count_pmf(b, claim, star, self.n_max)
        if pmf.shape[0] != 2:
            raise ValueError(
                f"LogScore plans with one verdict member (R = 2), not R = "
                f"{pmf.shape[0]}"
            )
        return expected_log_score(pmf[1], pmf[0])


class Count(AbstractUtility):
    """Expected number of claim-true planets found since time zero, under the forecast.

    ``U(b) = rate0 - rate``, the forecast's claim-true pool rate (``pool_rate``, row 0)
    in the pool at time zero less the rate in ``b``'s pool. A planet found is one no
    longer in the pool, so a second look at a star counts only what the first look
    missed: ``thin`` has already taken the found planets out of the pool.

    Attributes:
        pool0: the pool at time zero, after the grant; held beside the beliefs, not in
            them.
    """

    pool0: Pool

    def __call__(self, b, claim, star):
        """The claim-true rate at time zero less the rate now, for the forecast."""
        return pool_rate(self.pool0, claim, star)[0] - pool_rate(b.pool, claim, star)[0]


class Menu(eqx.Module):
    """The actions each star offers, padded to a shared length ``M``.

    Entry ``m`` is the same mode at every star (``model[m]`` names its observation
    model in the model table), so the menus of all stars share one structure and a
    star with fewer options masks the rest. A masked entry is still evaluated: keep its
    action finite.

    Attributes:
        actions: an ``Action`` whose leaves have shape ``(S, M)``.
        cost_d: time each entry charges against the budget, days, shape ``(S, M)``.
        mask: true where the entry is offered, shape ``(S, M)``.
        model: the model slot of each entry, static, one per entry.

    Raises:
        ValueError: if the fields disagree on ``(S, M)`` or ``model`` has not one slot
            per entry.
    """

    actions: Action
    cost_d: Float[Array, "S M"] = eqx.field(converter=jnp.asarray)
    mask: Bool[Array, "S M"] = eqx.field(converter=jnp.asarray)
    model: tuple[int, ...] = eqx.field(
        static=True, converter=lambda s: tuple(int(i) for i in s)
    )

    def __check_init__(self):
        """Check that every field has the shape ``(S, M)`` with ``M = len(model)``."""
        shapes = {jnp.shape(x) for x in jax.tree.leaves(self.actions)}
        shapes |= {self.cost_d.shape, self.mask.shape}
        if len(shapes) != 1 or self.mask.shape[-1:] != (len(self.model),):
            raise ValueError(
                f"menu fields disagree on (S, M): {shapes}, {len(self.model)} slots"
            )


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
    kept = jnp.matmul(mark_weights(pool), miss, precision=jax.lax.Precision.HIGHEST)
    emptied = (kept <= 0)[:, None]
    log_miss = jnp.log(jnp.where(miss > 0, miss, 1.0))
    log_miss = jnp.where(miss > 0, log_miss, -jnp.inf)
    log_kept = jnp.log(jnp.where(kept > 0, kept, 1.0))
    return Pool(
        pool.marks,
        pool.log_prior,
        jnp.where(emptied, pool.log_like, pool.log_like + log_miss),
        pool.log_mass + jnp.where(kept > 0, log_kept, -jnp.inf),
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
    rate = pool_rate(pool, claim, star)
    return jnp.where(rate > 0, found / jnp.where(rate > 0, rate, 1.0), 0.0)


def _branches(
    b: Belief, models: tuple[AbstractObservationModel, ...], menu: Menu
) -> tuple[Float[Array, "M A Y"], Float[Array, "M Am"]]:
    """Cells of each entry on the draws and its detection probability of each mark.

    The entries of one model are evaluated together; the cell axis is padded with
    impossible cells to the widest model. Neither depends on the weights, so they are
    computed once per star and read at every node of the recursion.
    """
    order, cells, detect = [], [], []
    for slot, model in enumerate(models):
        entries = [m for m, s in enumerate(menu.model) if s == slot]
        if not entries:
            continue
        actions = jax.tree.map(lambda x, e=entries: x[jnp.asarray(e)], menu.actions)
        order += entries
        cells.append(jax.vmap(model.cells, in_axes=(None, 0))(b.atoms, actions))
        marks = b.pool.marks
        detect.append(
            jax.vmap(_detect_prob, in_axes=(None, None, 0))(model, marks, actions)
        )
    n_cell = max(c.shape[-1] for c in cells)
    cells = [jnp.pad(c, ((0, 0), (0, 0), (0, n_cell - c.shape[-1]))) for c in cells]
    back = jnp.asarray(sorted(range(len(order)), key=order.__getitem__))
    return jnp.concatenate(cells)[back], jnp.concatenate(detect)[back]


def _worth(b, claim, star, utility, branches, charge, depth: int) -> Float[Array, ""]:
    """The utility of ``b`` with up to ``depth`` more actions, each chosen to raise it.

    ``W_0 = U`` and ``W_k(b) = max(U(b), max_m (E_F[W_{k-1}(b after m)] - charge_m))``,
    with ``charge_m`` the price of entry ``m``'s time (infinite for a masked entry):
    stopping is always an option, so ``W_k`` never falls below ``W_{k-1}``.
    """
    now = utility(b, claim, star)
    if depth == 0:
        return now
    later = _expected_after(b, claim, star, utility, branches, charge, depth - 1)
    return jnp.maximum(now, jnp.max(later - charge))


def _expected_after(
    b, claim, star, utility, branches, charge, depth: int
) -> Float[Array, " M"]:
    """``E_F[W_depth(b after m)]`` for each entry ``m``, over the forecast's cells.

    After entry ``m`` with outcome in cell ``y`` the draws are weighted by their
    probability of ``y`` (``update_by_cell``) and the pool is thinned by the entry's
    detection probability in every cell. A cell the forecast gives no mass adds nothing.
    """

    def entry(cells, detect):
        p_cell = outcome_prob(b, cells)[0]  # (Y,), the forecast's
        seen = p_cell > 0
        thinned = eqx.tree_at(lambda x: x.pool, b, thin(b.pool, detect))

        def worth_after(c):
            after = update_by_cell(thinned, c)
            return _worth(after, claim, star, utility, branches, charge, depth)

        worth = jax.vmap(worth_after, in_axes=1)(jnp.where(seen, cells, 1.0))
        return jnp.sum(jnp.where(seen, p_cell * worth, 0.0))

    return jax.vmap(entry)(*branches)


def frontier(
    belief: Belief,
    claim: Claim,
    star,
    utility: AbstractUtility,
    models: tuple[AbstractObservationModel, ...],
    menu: Menu,
    lam: Float[Array, ""],
    depth: int,
) -> Frontier:
    """The reward and cost of each menu entry at each star.

    The reward of entry ``m`` is ``E_F[W_{depth-1}(b after m)] - U(b)``: the utility
    ``U`` it is expected to add, with up to ``depth - 1`` further entries of the same
    star's menu after it, each chosen on the outcomes so far to raise the utility net
    of its time at price ``lam`` (``lam`` times its cost), or none. The expectation is
    over the forecast's (row 0) distribution of each outcome cell; an outcome updates
    every role by ``update_by_cell`` and thins the pool by the entry's detection
    probability of each mark (``thin``; zero for a model that does not search). The
    first entry's cost is not netted: it stays in the frontier for the dual. Under
    ``LogScore`` at depth 1 the reward is the verdict gain
    ``E_F[ln p_gamma(N | Y) - ln p_gamma(N)]``; ``lam`` enters only at ``depth > 1``.

    The price a frontier is built at is not the price the dual then solves for. A
    planner (``PricePolicy``) passes the last solved price: one step of the fixed point
    ``lam = solve_price(frontier(..., lam, ...))``, exact at depth 1, where the reward
    does not depend on ``lam``.

    This is the generic bounded recursion, the reference: it evaluates
    ``(M Y) ** depth`` hypothetical beliefs per star, so it serves small menus and
    shallow depths. The cells of every entry are computed once per star, since a
    hypothetical outcome changes weights, not draws.

    Two limits of the hypothetical update. A planet a hypothetical search finds is not
    spawned as a track (that takes a refit), so it leaves the count; value a search
    with ``Count``, which counts it as found. And the draws are not updated by a cell
    (see ``update_by_cell``), so the reward is exact only where the claim and the cell
    are independent given each draw.

    Every leaf of ``belief``, ``star``, ``menu`` and the array leaves of ``utility``
    (``Count``'s time-zero pool) carry a leading star axis ``S``; ``claim`` and
    ``models`` are shared by the stars. Menu, cell counts and ``depth`` are static, so
    another star or world of the same shapes reuses the compiled program.

    Args:
        belief: the beliefs, one per star.
        claim: an unconditional claim.
        star: the stars, a pytree whose leaves have a leading axis ``S``.
        utility: the utility.
        models: the model table, static; ``menu.model`` indexes it.
        menu: the menu of each star.
        lam: the price of the follow-up entries' time, reward per day, a scalar shared
            by the stars, ``lam >= 0``.
        depth: the number of entries a plan may take at the star, the first one
            included, static, at least 1.

    Returns:
        The frontier: rewards (zero at masked entries), the menu's costs and mask.

    Raises:
        ValueError: if ``depth < 1`` or a menu entry names a slot not in ``models``.
    """
    if depth < 1:
        raise ValueError(f"depth is at least 1, not {depth}")
    if any(not 0 <= s < len(models) for s in menu.model):
        raise ValueError("a menu entry's model slot is not in the model table")

    def one_star(b, star, utility, menu):
        branches = _branches(b, models, menu)
        charge = jnp.where(menu.mask, lam * menu.cost_d, jnp.inf)
        later = _expected_after(b, claim, star, utility, branches, charge, depth - 1)
        return jnp.where(menu.mask, later - utility(b, claim, star), 0.0)

    reward = eqx.filter_vmap(one_star)(belief, star, utility, menu)
    return Frontier(reward, menu.cost_d, menu.mask)
