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
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PyTree

from spaceodyssey._numerics import EXACT, TIE_ULPS, guarded_log, safe_ratio
from spaceodyssey.belief import (
    Belief,
    count_pmf,
    detected_slots,
    found_masses,
    mark_weights,
    poisson_pmf,
    pool_rate,
)
from spaceodyssey.claim import Claim
from spaceodyssey.information import (
    OutcomeCells,
    cell_probs,
    expected_log_score,
    mask_cells,
    outcome_cells,
    outcome_prob,
    require_point_draws,
    update_by_cell,
)
from spaceodyssey.types import (
    ACCEPTED,
    BERNOULLI,
    AbstractObservationModel,
    Action,
    Atoms,
    ObservationLog,
    Pool,
    Refit,
    as_index,
    planet_tracks,
    real_rows,
)


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
        capped: true where the option's entry has more possible outcome cells than
            the lookahead keeps (``frontier``'s ``max_cells``) and a plan can take it,
            shape ``(S, M)``; all false for a frontier without a cell cap.
    """

    reward: Float[Array, "S M"] = eqx.field(converter=jnp.asarray)
    cost_d: Float[Array, "S M"] = eqx.field(converter=jnp.asarray)
    mask: Bool[Array, "S M"] = eqx.field(converter=jnp.asarray)
    capped: Bool[Array, "S M"] = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Check that the arrays share one ``(S, M)`` shape."""
        shapes = [
            self.reward.shape,
            self.cost_d.shape,
            self.mask.shape,
            self.capped.shape,
        ]
        if any(shape != shapes[0] for shape in shapes):
            raise ValueError(
                f"reward, cost_d, mask and capped must share one shape, got {shapes}"
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

    def found_cap(self) -> int:
        """The most planets found by one hypothetical search this utility tells apart.

        Static (``frontier``). 0, the default, for a utility that does not read the
        planets a search finds (``Belief.found``): the lookahead then keeps no found
        planet and does not branch on how many a Poisson pool gave.
        """
        return 0

    @abc.abstractmethod
    def __call__(
        self, forecast: Belief, verdict: Belief, claim: Claim, star
    ) -> Float[Array, ""]:
        """The utility of the forecast's and the verdict's beliefs about one star.

        Args:
            forecast: the forecast's belief, one role (``R = 1``).
            verdict: the verdict's belief, one role per member.
            claim: the claim.
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

    A conditional claim scores the claim-true count ``N`` given the count ``M`` of
    planets its condition admits (``count_pmf``):
    ``U = sum_{f,n} P_F(F = f, N = n) ln P_gamma(N = n | M = f + n)``, so learning the
    condition alone adds nothing in expectation; it pays only as a way to the claim.
    ``M`` is read off the bin indices, so mass lumped at ``n_max`` is scored as
    ``M = n_max``: the conditional score is exact when ``n_max`` exceeds every count
    with mass, up to the found tail: a Poisson lookahead's last branch holds exactly
    ``found_cap`` found planets for every ``k >= found_cap``, so its claim-true count
    is exact only to ``P(k >= n_max)``.

    The count includes the planets a hypothetical search found (``Belief.found``):
    a planet found in the pool is there, claim-true with the claim probability of a
    detected mark, until a refit makes it a track. A lookahead branches on how many
    planets a search finds in a Poisson pool up to ``n_max`` (``found_cap``), the last
    branch lumping the rest.

    Attributes:
        n_max: the count at which ``count_pmf`` lumps the tail, static; it belongs to
            this utility (``Count`` reads no distribution of the count).
    """

    n_max: int = eqx.field(static=True)

    def found_cap(self) -> int:
        """``n_max``: the count reads every planet found, up to its lumped tail."""
        return self.n_max

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
        if claim.given == 0:
            return expected_log_score(pmf[0], expected)
        return expected_log_score(_given_admitted(pmf[0]).ravel(), expected.ravel())


def _given_admitted(joint: Float[Array, "N N"]) -> Float[Array, "N N"]:
    """``P(N = n | M = f + n)`` from the joint of ``(F, N)`` (``count_pmf``).

    ``M = F + N`` is read off the bin indices, a lumped bin counting as its index.
    A total no outcome reaches gives 0, as ``safe_ratio`` does.
    """
    n_bins = joint.shape[-1]
    total = np.add.outer(np.arange(n_bins), np.arange(n_bins))  # M of each (f, n)
    by_total = jnp.zeros(2 * n_bins - 1, joint.dtype).at[total].add(joint)
    return safe_ratio(joint, by_total[total])


class Count(AbstractUtility):
    """Expected number of claim-true planets found, under the forecast, to a constant.

    ``U = -rate``, the negative of the forecast's claim-true pool rate (``pool_rate``)
    in the forecast belief's pool. A planet found is one no longer in the pool, so a
    second look at a star counts only what the first look missed: ``thin`` has already
    taken the found planets out of the pool. A utility matters only up to a constant,
    so ``U`` is the number found since any reference time less the rate then; the
    number a run found is ``expected_found``. The verdict is not read.

    Whether planning on ``Count`` can beat a static allocation depends on the pool's
    law (``Pool.law``). A Poisson pool thinned by a search is Poisson with the thinned
    rate whatever the search found, so a detection carries no information about the
    other planets at the star: the value of every later action is a function of the
    actions taken, not of their outcomes, and the best adaptive policy is the best
    open-loop one, the static allocation. A Bernoulli pool holds at most one planet: a
    detection empties it, so the planner stops searching a star whose planet it found
    and spends the time where its planet is still missing.
    """

    def __call__(self, forecast, verdict, claim, star):
        """The negative of the forecast's claim-true pool rate."""
        return -pool_rate(forecast.pool, claim, star)[0]


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

    By the pool's law (``Pool.law``). A Poisson pool's planets an observation misses
    are the pool thinned by the miss probability ``1 - d`` (a Poisson process thinned
    independently is a Poisson process), whatever the observation found: role ``r``'s
    expected number is multiplied by ``1 - dbar_r``, with ``dbar_r`` the detection
    probability averaged under that role's own mark weights. A Bernoulli pool is the
    pool given that the observation missed its planet: present with probability
    ``rho (1 - dbar) / (1 - rho dbar)`` (the planets found leave either pool; a
    Bernoulli pool whose planet was found is empty, which ``replay_pool`` reads off a
    row and the lookahead takes as an outcome of its own). Either way each mark's
    likelihood is multiplied by ``1 - d``. The planets found become tracks only
    through a refit.

    A role whose every mark is detected with certainty is left with an empty pool
    (``log_mass = -inf``) and its mark weights as they were, so its rate is zero, not
    NaN.

    Args:
        pool: the undetected planets.
        detect_prob: probability that the observation detects each mark, ``(Am,)``.

    Returns:
        The planets the observation missed.
    """
    miss = 1.0 - detect_prob
    kept = jnp.matmul(mark_weights(pool), miss, precision=EXACT)
    emptied = kept <= 0
    log_mass = pool.log_mass + guarded_log(kept)
    if pool.law == BERNOULLI:  # static: the planet's posterior given the miss
        rho = jnp.exp(pool.log_mass)
        still = jnp.log1p(-jnp.where(emptied, 0.0, rho * (1.0 - kept)))
        log_mass = jnp.where(emptied, -jnp.inf, log_mass - still)
    return eqx.tree_at(
        lambda p: (p.log_like, p.log_mass),
        pool,
        (
            jnp.where(
                emptied[:, None], pool.log_like, pool.log_like + guarded_log(miss)
            ),
            log_mass,
        ),
    )


def found_prob(pool: Pool, detect_prob: Float[Array, " Am"]) -> Float[Array, " R"]:
    """Probability, per role, that an observation finds a Bernoulli pool's planet.

    ``rho dbar``: the planet is present (``rho = exp(log_mass)``) and detected
    (``dbar``, the detection probability averaged under the role's mark weights).

    Args:
        pool: the undetected planets, a ``BERNOULLI`` pool.
        detect_prob: probability that the observation detects each mark, ``(Am,)``.

    Returns:
        The probability per role, shape ``(R,)``.
    """
    found = jnp.matmul(mark_weights(pool), detect_prob, precision=EXACT)
    return jnp.exp(pool.log_mass) * found


def _emptied(pool: Pool) -> Pool:
    """The pool with no planet left, every role: a Bernoulli pool's planet found."""
    return eqx.tree_at(
        lambda p: p.log_mass, pool, jnp.full_like(pool.log_mass, -jnp.inf)
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
        claim: the claim.
        star: a pytree of scalar leaves describing the star.

    Returns:
        The completeness per role, shape ``(R,)``.
    """
    detect = _detect_prob(model, pool.marks, action)
    # The expected number found is linear in the pool, the same under either law: the
    # Poisson thinning by the detection probability is the pool of the planets found.
    poisson = Pool(pool.marks, pool.log_prior_ratio, pool.log_like, pool.log_mass)
    found = pool_rate(thin(poisson, 1.0 - detect), claim, star)
    return safe_ratio(found, pool_rate(pool, claim, star))


def _row_cells(model: AbstractObservationModel, payload: PyTree) -> Int[Array, " W"]:
    """The outcome cell of every row of a log's payload buffer of one model."""
    return jax.vmap(model.cell)(payload)


def reached_stage(
    log: ObservationLog,
    models: tuple[AbstractObservationModel, ...],
    star_index: Int[Array, ""],
    time_d: Float[Array, ""],
) -> Int[Array, ""]:
    """The stage a star's released rows have reached: the planner's ``Menu.stage``.

    A row reaches its model's ``stage`` when its outcome is in any cell but cell 0
    (``AbstractObservationModel.cell``), as a hypothetical outcome does in the
    lookahead (``frontier``); the star is at the highest stage any of its accepted rows
    released by ``time_d`` reached, 0 before any. Rows in flight, refused rows and
    waits reach nothing.

    Args:
        log: the run's log, all stars.
        models: the run's model table, static; ``Row.model_slot`` indexes it.
        star_index: the star's index in the log's ``star`` column.
        time_d: the query time, days from the start of the survey.

    Returns:
        The stage.
    """
    rows = log.rows
    mine = (
        real_rows(log)
        & (rows.release_d <= time_d)
        & (rows.refusal == ACCEPTED)
        & (rows.star == star_index)
    )
    stage = jnp.zeros((), int)
    for slot, model in enumerate(models):
        if model.stage == 0:
            continue  # static: this model opens nothing
        found = _row_cells(model, rows.payload[slot]) > 0
        hit = mine & (rows.model_slot == slot) & found
        stage = jnp.maximum(stage, jnp.where(jnp.any(hit), model.stage, 0))
    return stage


def replay_pool(
    pool: Pool,
    refit: Refit,
    log: ObservationLog,
    models: tuple[AbstractObservationModel, ...],
    time_d: Float[Array, ""],
) -> Pool:
    """The pool at a query time: the refit's pool thinned by the searches since.

    Every accepted row of a searching model at the refit's star, released by
    ``time_d`` and not among the rows the refit represents (``Refit.row_mask``; the
    refit's pool already carries those), thins the pool by its probability of missing
    each mark (``thin``). For a Poisson pool that holds whatever the rows found: a
    Poisson process of undetected planets thinned by a search is the planets it
    missed, a Poisson process whose law does not depend on what was found (the finds
    become tracks only through a refit). So a Poisson pool depends on which searches
    were taken, never on their payloads. A Bernoulli pool reads them: a row that
    detected a planet in a slot the refit holds no track for (``detected_slots``) found
    the pool's one planet, and the pool is empty after it; otherwise the rows missed
    it. Thinning by several rows is thinning once by the product of their miss
    probabilities, in any order: this replays the whole log, as ``replay`` does, and a
    planner calls it afresh at each time rather than absorbing rows into it. Each row
    is read as its action at its realized start (``Row.time_d``), as it was taken. Rows
    in flight, refused rows, waits and rows of a model that does not search leave the
    pool as it is.

    Args:
        pool: the refit's pool, every role, as the refit returned it.
        refit: the refit of the same star: its index and the rows it represents.
        log: the run's log, all stars.
        models: the run's model table, static; ``Row.model_slot`` indexes it.
        time_d: the query time, days from the start of the survey.

    Returns:
        The pool at ``time_d``.
    """
    rows = log.rows
    due = (
        real_rows(log)
        & (rows.release_d <= time_d)
        & (rows.refusal == ACCEPTED)
        & (rows.star == refit.star_index)
        & ~refit.row_mask
    )
    actions = Action(rows.star, rows.mode, rows.exposure_d, rows.time_d)
    miss = jnp.ones(pool.marks.exists.shape[0], pool.log_mass.dtype)
    for slot, model in enumerate(models):
        if not model.searches:
            continue  # static: only a search thins the pool
        detect = jax.vmap(lambda a, model=model: _detect_prob(model, pool.marks, a))(
            actions
        )  # (W, Am)
        keep = (due & (rows.model_slot == slot))[:, None]
        miss = miss * jnp.prod(jnp.where(keep, 1.0 - detect, 1.0), axis=0)
    thinned = thin(pool, 1.0 - miss)
    if pool.law != BERNOULLI:  # static: a Poisson pool reads no payload
        return thinned
    n_slot = refit.atoms.exists.shape[-1]
    found = detected_slots(log, due, models, n_slot) & ~planet_tracks(refit.atoms)
    return jax.lax.cond(jnp.any(found), lambda: _emptied(thinned), lambda: thinned)


def expected_found(
    pool: Pool,
    refit: Refit,
    log: ObservationLog,
    models: tuple[AbstractObservationModel, ...],
    claim: Claim,
    star,
    time_d: Float[Array, ""],
) -> Float[Array, " R"]:
    """Expected number of claim-true planets a run's searches found at a star, per role.

    The claim-true rate of the refit's pool less that of the pool its searches
    released by ``time_d`` left (``replay_pool``): what ``Count`` values, read off the
    log. For a Poisson pool it reads no payload; a Bernoulli pool reads whether a
    search found its planet, and the drop is then its whole rate. Either way it is the
    forecast's expectation of the claim-true finds given what it reads (its mean over
    the outcomes is the expected finds of the searches taken), not the finds; the
    planets a run really found are read against the truth (``score.found_count``).

    Args:
        pool: the refit's pool, every role.
        refit: the refit of the same star.
        log: the run's log, all stars.
        models: the run's model table, static.
        claim: the claim.
        star: a pytree of scalar leaves describing the star.
        time_d: the query time, days from the start of the survey.

    Returns:
        The expected claim-true planets found, per role, shape ``(R,)``.
    """
    now = replay_pool(pool, refit, log, models, time_d)
    return pool_rate(pool, claim, star) - pool_rate(now, claim, star)


def _entries(menu: Menu, mode: int, star_index, time_d) -> tuple[list[int], Action]:
    """The menu entries of one mode at one star, and their actions (leading axis).

    Every action carries the planned start ``time_d``.
    """
    entries = [m for m, e in enumerate(menu.mode) if e == mode]
    exposure_d = menu.exposure_d[jnp.asarray(entries, int)]
    shape = exposure_d.shape
    actions = Action(
        jnp.full(shape, star_index), jnp.full(shape, mode), exposure_d, time_d
    )
    return entries, actions


class _Outcomes(eqx.Module):
    """Every menu entry's outcome cells at one star, read once at the root.

    On point states an outcome changes only the weights (of the draws and of their gas
    nodes) and the pool, never a draw, a node or a mark, so each entry's cells per
    draw and node and its detection probabilities are the same at every node of the
    lookahead; only their mixture over the node weights moves. Cells are padded with
    zero-probability cells to the widest entry's ``Y``, and bins with zero-probability
    bins to the widest ``B``, so the entries of every mode share one shape.

    Attributes:
        forecast: the cells on the forecast's draws, leading axis ``M``.
        verdict: the cells on the verdict's draws, leading axis ``M``.
        detect_forecast: detection probability of the forecast pool's marks,
            ``(M, Am)``; zero for a model that does not search.
        detect_verdict: the same for the verdict's pool, ``(M, Am)``.
        reached: the stage each cell reaches, ``(M, Y)``: the entry's model's
            ``stage`` for every cell but cell 0, and 0 there.
        found_stage: the stage a search reaches when it finds a pool planet,
            ``(M,)``: the entry's model's ``stage`` if it searches, else 0. The draws'
            cells do not hold the pool's planets, so a find in the pool raises the
            stage on its own branch, as the real row's cell (which holds the new
            slot's detection) raises it in a run.
        capped: whether the entry has more possible cells than ``max_cells``,
            ``(M,)`` (``_compact``); false without a cap.
    """

    forecast: OutcomeCells
    verdict: OutcomeCells
    detect_forecast: Float[Array, "M AmF"]
    detect_verdict: Float[Array, "M AmV"]
    reached: Int[Array, "M Y"]
    found_stage: Int[Array, " M"]
    capped: Bool[Array, " M"]


def _pad_cells(cells: OutcomeCells, n_cell: int, n_bin: int) -> OutcomeCells:
    """Cells padded to ``n_cell`` cells (impossible ones) and ``n_bin`` bins."""

    def pad(x, axis, width):
        widths = [(0, 0)] * x.ndim
        widths[axis] = (0, width - x.shape[axis])
        return jnp.pad(x, widths)

    return OutcomeCells(
        pad(cells.draw, -1, n_cell),
        pad(cells.node, -1, n_bin),
        pad(cells.codes, -3, n_cell),
    )


def _compact(
    forecast: Belief,
    cells_f: OutcomeCells,
    cells_v: OutcomeCells,
    reached: Int[Array, " Y"],
    max_cells: int,
) -> tuple[OutcomeCells, OutcomeCells, Int[Array, " C"], Bool[Array, ""]]:
    """One entry's cells cut to the ``max_cells`` the forecast's draws can give.

    A cell no draw of the forecast can give (``cell_probs`` zero on every draw, under
    any weights of its draws and nodes) adds nothing at any node of the lookahead,
    since an outcome only reweights the draws and nodes; so the cells some draw gives
    are kept, in their order, and padded with cells no draw gives. When more than
    ``max_cells`` cells are possible, the entry fails closed: its cells read NaN, so
    its reward is NaN and its star is refused (``refused_stars``), never valued on a
    part of its outcomes; the last return is whether it did.
    """
    support = jnp.any(cell_probs(forecast, cells_f) > 0, axis=0)  # (Y,)
    keep = jnp.argsort(~support, stable=True)[:max_cells]
    over = jnp.sum(support) > max_cells

    def take(cells, poison):
        draw = cells.draw[:, keep]
        if poison:
            draw = jnp.where(over, jnp.nan, draw)
        return OutcomeCells(draw, cells.node, cells.codes[keep])

    return take(cells_f, True), take(cells_v, False), reached[keep], over


def _outcomes(
    forecast, verdict, mode_models, menu, star_index, time_d, max_cells
) -> _Outcomes:
    """The cells and detection probabilities of every entry, in menu order."""
    order, parts = [], []
    for mode, model in enumerate(mode_models):
        entries, actions = _entries(menu, mode, star_index, time_d)
        if not entries:
            continue

        def one(action, model=model):
            return (
                outcome_cells(forecast, model, action),
                outcome_cells(verdict, model, action),
                _detect_prob(model, forecast.pool.marks, action),
                _detect_prob(model, verdict.pool.marks, action),
            )

        cells_f, cells_v, detect_f, detect_v = jax.vmap(one)(actions)
        n_cell = cells_f.draw.shape[-1]
        reached = jnp.where(jnp.arange(n_cell) > 0, model.stage, 0)
        reached = jnp.broadcast_to(reached, (len(entries), n_cell))
        capped = jnp.zeros(len(entries), bool)
        if max_cells is not None and n_cell > max_cells:  # static
            cells_f, cells_v, reached, capped = jax.vmap(
                lambda f, v, r: _compact(forecast, f, v, r, max_cells)
            )(cells_f, cells_v, reached)
        found_stage = jnp.full(len(entries), model.stage if model.searches else 0)
        order += entries
        parts.append(
            _Outcomes(
                forecast=cells_f,
                verdict=cells_v,
                detect_forecast=detect_f,
                detect_verdict=detect_v,
                reached=reached,
                found_stage=found_stage,
                capped=capped,
            )
        )
    n_cell = max(part.forecast.draw.shape[-1] for part in parts)
    n_bin = max(part.forecast.node.shape[-1] for part in parts)
    back = jnp.asarray(sorted(range(len(order)), key=order.__getitem__))

    def padded(part: _Outcomes) -> _Outcomes:
        return _Outcomes(
            forecast=_pad_cells(part.forecast, n_cell, n_bin),
            verdict=_pad_cells(part.verdict, n_cell, n_bin),
            detect_forecast=part.detect_forecast,
            detect_verdict=part.detect_verdict,
            reached=jnp.pad(
                part.reached, [(0, 0), (0, n_cell - part.reached.shape[-1])]
            ),
            found_stage=part.found_stage,
            capped=part.capped,
        )

    return jax.tree.map(
        lambda *x: jnp.concatenate(x)[back], *[padded(part) for part in parts]
    )


def _reachable(menu: Menu, outcomes: _Outcomes, depth: int) -> Bool[Array, " M"]:
    """The entries a plan of up to ``depth`` entries may take: now or as a follow-up.

    The entries offered at the root, and those offered at a stage some offered entry
    can raise the star to, ``depth - 1`` times over. An entry that no plan can take
    (a spectrum at a star that no search can bring to its stage) is not reachable.
    The reading is by stage alone: an outcome of zero probability still counts.
    """
    raises = jnp.maximum(outcomes.reached.max(axis=-1), outcomes.found_stage)
    stage = menu.stage
    for _ in range(depth - 1):  # static
        stage = jnp.maximum(stage, jnp.max(jnp.where(_offered(menu, stage), raises, 0)))
    return _offered(menu, stage)


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

    Ties are broken as ``dispatch`` breaks them, toward the cheapest: values within
    ``TIE_ULPS`` units of roundoff of their scale are equal, a follow-up is taken only
    when it beats stopping by more than that, and among tied follow-ups the shortest
    is taken (the first in menu order among equals). So a look worth exactly nothing
    more, as after a report that resolved the claim, is never bought at any price,
    zero included, and the slope in the price (the plan's follow-up time) is never a
    split decided by rounding. A NaN (a refused role) stays NaN.
    """
    forecast, verdict, stage = node
    now = plan.utility(forecast, verdict, plan.claim, plan.star)
    if depth == 0:
        return now
    later = _expected_after(node, plan, depth - 1)
    net = later - _charge(plan, stage)
    best = jnp.max(net)
    scale = jnp.maximum(
        jnp.abs(now), jnp.max(jnp.where(net > -jnp.inf, jnp.abs(net), 0))
    )
    tol = TIE_ULPS * jnp.finfo(net.dtype).eps * scale
    tied = net >= best - tol
    go = net[jnp.argmin(jnp.where(tied, plan.menu.exposure_d, jnp.inf))]
    worth = jnp.where(go > now + tol, go, now)
    return jnp.where(jnp.isnan(best) | jnp.isnan(now), jnp.nan, worth)


def _found_slots(utility: AbstractUtility, law: str) -> int:
    """Found-planet slots one level of the lookahead fills (``Belief.found``), static.

    None when the utility reads no found planet; one for a Bernoulli pool (at most one
    planet); ``found_cap`` for a Poisson pool.
    """
    cap = utility.found_cap()
    if cap == 0:
        return 0
    return 1 if law == BERNOULLI else cap


def _n_branches(utility: AbstractUtility, law: str) -> int:
    """The pool's outcomes of a search the lookahead tells apart, static.

    A Bernoulli pool: its planet found or missed. A Poisson pool: none found, one, ...
    up to the utility's ``found_cap`` (the last lumping the rest), or a single outcome
    when the utility reads no found planet: the thinned pool does not depend on what
    the search found.
    """
    if law == BERNOULLI:
        return 2
    return _found_slots(utility, law) + 1


def _expected_after(node, plan, depth: int) -> Float[Array, " M"]:
    """``E_F[W_depth(node after m)]`` for each entry ``m``, over the forecast's cells.

    The cells per draw and node are the root's (``_Outcomes``); their mixture over the
    node weights is read at each node. After entry ``m`` with outcome in cell ``y``
    each belief is reweighted by its draws' and nodes' probability of the cell
    (``update_by_cell``), and the star moves to the higher of its stage and the stage
    the cell reaches, and, in a branch that found a pool planet, the stage the search
    reaches (``_Outcomes.found_stage``).

    The pool's outcome is a branch of its own, independent of the draws' cells, its
    probability the forecast's (``_n_branches``). A Bernoulli pool's planet is found
    with the probability that the entry finds it (``found_prob``): both pools are then
    empty and each belief holds the planet found (``found_masses``, in
    ``Belief.found``); otherwise both are the pools given the miss (``thin``). A
    Poisson pool is the pool thinned whatever the search found, and ``k`` planets are
    found with the forecast's Poisson probability of mean ``exp(log_mass) dbar``,
    ``k = 0 .. found_cap`` (the last branch lumping the rest), each belief then
    holding ``k`` planets found. A utility that reads no found planet keeps none, and
    a Poisson pool is then one branch. The branches are one body under ``vmap`` over
    the branch index: only the work that reads the pool and the found planets is
    batched, and the cell update, the draws' counts and the next level's cells run
    once per cell. A level's found planets go to slots of their own, ``depth`` times
    the slots a level fills, so the planets found along a path of the lookahead add.

    Every entry and cell is one body, so a level of the lookahead traces once: under
    ``vmap`` at the last level, and as a loop (``lax.map``) above it, so the beliefs
    held at once are one node's children, not the whole level, and memory stays at
    ``M Y`` beliefs whatever the depth. A cell or branch the forecast gives no mass (a
    padded cell among them) adds nothing; a refused forecast (a NaN row) makes every
    entry NaN, never zero.
    """
    forecast, verdict, stage = node
    law = forecast.pool.law
    per_level = _found_slots(plan.utility, law)  # static
    n_branch = _n_branches(plan.utility, law)  # static
    first = depth * per_level  # this level's found slots

    def entry(cells_f, cells_v, detect_f, detect_v, reached, found_stage):
        p_cell = outcome_prob(forecast, cell_probs(forecast, cells_f))[0]  # (Y,)
        seen = ~(p_cell <= 0)  # a refused forecast (NaN) reads NaN, not zero
        missed_f = thin(forecast.pool, detect_f)
        missed_v = thin(verdict.pool, detect_v)
        safe_f, safe_v = mask_cells(cells_f, seen), mask_cells(cells_v, seen)
        if law == BERNOULLI:  # static
            found = found_prob(forecast.pool, detect_f)[0]
            p_branch = jnp.stack([1.0 - found, found])
        elif per_level:
            dbar = jnp.matmul(mark_weights(forecast.pool), detect_f, precision=EXACT)
            rate = jnp.exp(forecast.pool.log_mass[0]) * dbar[0]  # the planets found
            p_branch = poisson_pmf(rate, n_branch)
        planet_f = planet_v = None
        if per_level:
            planet_f = found_masses(forecast.pool, detect_f, plan.claim, plan.star)
            planet_v = found_masses(verdict.pool, detect_v, plan.claim, plan.star)

        def branch(b: Belief, missed: Pool, planet, i) -> tuple[Pool, Array | None]:
            """The pool and the found planets of one belief in branch ``i``."""
            if law == BERNOULLI:  # static: branch 1 found the planet, emptying it
                gone = i == 1
                pool = eqx.tree_at(
                    lambda p: (p.log_like, p.log_mass),
                    missed,
                    (
                        jnp.where(gone, b.pool.log_like, missed.log_like),
                        jnp.where(gone, -jnp.inf, missed.log_mass),
                    ),
                )
                if not per_level:
                    return pool, b.found
                block = jnp.where(i == 1, planet, 0.0)[:, None, :]
            else:  # Poisson: branch i found i planets
                pool = missed
                if not per_level:
                    return pool, b.found
                held = jnp.arange(per_level) < i  # (per_level,)
                block = jnp.where(held[None, :, None], planet[:, None, :], 0.0)
            found = jax.lax.dynamic_update_slice_in_dim(b.found, block, first, axis=-2)
            return pool, found

        def worth_after(cell, reached_y):
            after_f = update_by_cell(forecast, safe_f, cell)
            after_v = update_by_cell(verdict, safe_v, cell)
            stage_y = jnp.maximum(stage, reached_y)

            def worth(i):
                pool_f, found_f = branch(forecast, missed_f, planet_f, i)
                pool_v, found_v = branch(verdict, missed_v, planet_v, i)
                # A branch that found a pool planet (i > 0) reached the search's stage.
                stage_i = jnp.where(i > 0, jnp.maximum(stage_y, found_stage), stage_y)
                node = (
                    _with_pool(after_f, pool_f, found_f),
                    _with_pool(after_v, pool_v, found_v),
                    stage_i,
                )
                return _worth(node, plan, depth)

            if n_branch == 1:  # static
                return worth(0)
            w = jax.vmap(worth)(jnp.arange(n_branch))
            return jnp.sum(jnp.where(p_branch <= 0, 0.0, p_branch * w))

        worth = each(
            worth_after, jnp.arange(p_cell.shape[0]), reached, batch_size=batch
        )
        return jnp.sum(jnp.where(seen, p_cell * worth, 0.0))

    o = plan.outcomes
    batch = _batch(node, o, depth, n_branch)

    def each(fn, *xs, batch_size):
        if batch_size is None:
            return jax.vmap(fn)(*xs)
        return jax.lax.map(lambda x: fn(*x), xs, batch_size=batch_size)

    return each(
        entry,
        o.forecast,
        o.verdict,
        o.detect_forecast,
        o.detect_verdict,
        o.reached,
        o.found_stage,
        batch_size=None if batch is None else 1,
    )


def _with_pool(b: Belief, pool: Pool, found) -> Belief:
    """``b`` with another pool and other found planets."""
    return Belief(b.atoms, b.log_prior_ratio, b.log_like, pool, found)


_LEVEL_ELEMENTS = 2**25
"""Most belief elements one level of the lookahead holds at once (about 256 MiB of
float64 per array); above it the levels over the last are loops (``_batch``)."""


def _batch(node, outcomes: _Outcomes, depth: int, n_branch: int) -> int | None:
    """How many cells of a node a level above the last evaluates at once.

    None (all of them, under ``vmap``) at the last level, and wherever the whole
    level's children fit ``_LEVEL_ELEMENTS``; otherwise the entries run one at a time
    and the cells in batches that fit. A child holds its draws' arrays once and its
    pool and found planets once per branch of the pool's outcome (``_n_branches``),
    which the branch ``vmap`` batches; the counts are bounded by the branches of two
    levels, since a level under a batched one nests its own branch ``vmap`` in it.
    Static: read from shapes.
    """
    if depth == 0:
        return None
    n_entry, n_cell = outcomes.reached.shape
    fan = n_entry * n_cell  # children of one node
    child = sum(
        (b.atoms.x_mean.size if b.atoms.nodes is None else b.atoms.nodes.x.size)
        + b.log_like.size
        + n_branch**2
        * (
            b.pool.log_like.size
            + b.pool.marks.x_mean.size
            + (0 if b.found is None else b.found.size)
        )
        for b in node[:2]
    )
    if fan * fan * child <= _LEVEL_ELEMENTS:
        return None
    return max(1, _LEVEL_ELEMENTS // (fan * child))


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
    *,
    star_index: Int[Array, " S"] | None = None,
    time_d: Float[Array, ""] | float = 0.0,
    max_cells: int | None = None,
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
    shallow depths. The beliefs hold point states (``planning_beliefs`` puts an
    uncertain gas block on quadrature nodes per factor, ``quadrature_draws``); given a
    point the outcomes of rows are independent, so every entry's cells per draw and
    node are read once at the root and a hypothetical outcome updates a belief exactly
    by reweighting its draws and nodes (``update_by_cell``), at a cost linear in the
    nodes per factor. A belief with an uncertain block not on nodes is refused
    (``require_point_draws``).

    A planet a hypothetical search finds in the pool is not spawned as a track (that
    takes a refit): it stays in the count as a planet found (``Belief.found``), there
    and claim-true with the claim probability of a detected mark, for a utility that
    reads it (``AbstractUtility.found_cap``). A follow-up that reads such a planet (a
    spectrum of it) is not valued: no draw holds it, so its cells read nothing of it.

    With ``max_cells``, each entry keeps only the outcome cells the forecast's draws
    can give, at most ``max_cells`` of them (a slot no draw holds has the single
    outcome 0, ``AbstractObservationModel.cells``), which is exact; an entry with more
    possible cells fails closed: its reward is NaN, and so is that of every plan with
    it as a follow-up, its star is refused, and ``Frontier.capped`` marks it (only an
    entry a plan can take: offered at the root, or at a stage a plan can reach).

    Every action of a plan, the follow-ups included, is read at the planned start
    ``time_d``, the time the plan is built: the cells are read once at the root, so a
    follow-up's later start is not modeled. A model whose outcomes follow the time
    (orbital phase, zodi) reads every level of the lookahead at the plan's start.

    Every leaf of the two beliefs, ``star`` and ``menu`` carry a leading star axis
    ``S``; ``claim``, ``utility`` and ``mode_models`` are shared by the stars (a
    utility holds no per-star state). The menu's modes, the cell counts and ``depth``
    are static, so another star or world of the same shapes reuses the compiled
    program.

    Args:
        forecast: the forecast's beliefs, one per star, one role each.
        verdict: the verdict's beliefs, one per star, one role each (planning reads
            one member).
        claim: the claim.
        star: the stars, a pytree whose leaves have a leading axis ``S``.
        utility: the utility.
        mode_models: the observation model of each mode (``mode_models`` of the
            design), static; ``menu.mode`` indexes it.
        menu: the menu of each star.
        lam: the price of the follow-up entries' time, reward per day, a scalar shared
            by the stars, ``lam >= 0``.
        depth: the number of entries a plan may take at the star, the first one
            included, static, at least 1.
        star_index: each star's index in the survey, which its actions carry (a
            model with per-star configuration reads it); ``0 .. S - 1`` by default.
        time_d: the planned start of every action of the plans, days from the start
            of the survey; zero by default.
        max_cells: the most outcome cells an entry keeps, static; None keeps them
            all.

    Returns:
        The frontier: each plan's reward and expected time (zero and the exposure at
        an entry not offered), the entries offered at the root and the entries over
        the cell cap.

    Raises:
        ValueError: if ``depth < 1`` or a menu entry names a mode not in
            ``mode_models``.
    """
    if depth < 1:
        raise ValueError(f"depth is at least 1, not {depth}")
    if any(not 0 <= m < len(mode_models) for m in menu.mode):
        raise ValueError("a menu entry's mode is not in the mode table")

    lam = jnp.asarray(lam, float)
    time_d = jnp.asarray(time_d, float)
    forecast, verdict = require_point_draws(forecast), require_point_draws(verdict)

    n_found = depth * _found_slots(utility, forecast.pool.law)  # static

    def none_found(b: Belief) -> Belief:
        if not n_found:
            return b
        found = jnp.zeros((b.log_like.shape[0], n_found, len(claim.conjuncts) + 1))
        return _with_pool(b, b.pool, found)

    def one_star(forecast, verdict, star, menu, star_index):
        forecast, verdict = none_found(forecast), none_found(verdict)
        node = (forecast, verdict, menu.stage)
        offered = _offered(menu, menu.stage)
        now = utility(forecast, verdict, claim, star)
        outcomes = _outcomes(
            forecast, verdict, mode_models, menu, star_index, time_d, max_cells
        )

        def net(price):
            plan = _Plan(claim, star, utility, menu, outcomes, price)
            return jnp.where(offered, _expected_after(node, plan, depth - 1) - now, 0.0)

        if depth == 1:  # no follow-up: the net value does not depend on the price
            value, follow_d = net(lam), jnp.zeros_like(menu.exposure_d)
        else:
            value, slope = jax.jvp(net, (lam,), (jnp.ones_like(lam),))
            follow_d = jnp.where(offered, -slope, 0.0)
        capped = outcomes.capped & _reachable(menu, outcomes, depth)
        return value + lam * follow_d, menu.exposure_d + follow_d, offered, capped

    if star_index is None:
        star_index = jnp.arange(menu.mask.shape[0])
    reward, cost_d, offered, capped = eqx.filter_vmap(one_star)(
        forecast, verdict, star, menu, star_index
    )
    return Frontier(reward, cost_d, offered, capped)


def frontier_at_prices(
    forecast: Belief,
    verdict: Belief,
    claim: Claim,
    star,
    utility: AbstractUtility,
    mode_models: tuple[AbstractObservationModel, ...],
    menu: Menu,
    lam: Float[Array, " P"],
    depth: int,
    *,
    star_index: Int[Array, " S"] | None = None,
    time_d: Float[Array, ""] | float = 0.0,
    max_cells: int | None = None,
) -> Frontier:
    """Each star's plans built at several prices, offered side by side as options.

    A frontier's option is its plan's tangent at the price it was built at
    (``frontier``): exact there, a bound above it and not one below. Built at ``P``
    prices, each star offers ``P`` tangents of every entry's plan, so the star's value
    ``max(0, max_o (r_o - lam c_o))`` follows the convex curve of its plans over the
    whole range the prices span, and a price solved on it moves only as far as the
    plans really change (a plan that drops a follow-up when time is dear is offered
    beside the one that keeps it). Option ``p M + m`` is entry ``m``'s plan built at
    ``lam[p]``; ``option % M`` maps an option back to its entry. At ``depth = 1`` a plan
    does not depend on the price, so one price suffices.

    Args:
        forecast: the forecast's beliefs, one per star (``frontier``).
        verdict: the verdict's beliefs, one per star.
        claim: the claim.
        star: the stars, a pytree whose leaves have a leading axis ``S``.
        utility: the utility.
        mode_models: the observation model of each mode, static.
        menu: the menu of each star.
        lam: the prices the plans are built at, shape ``(P,)``.
        depth: the number of entries a plan may take at the star, static.
        star_index: each star's index in the survey (``frontier``).
        time_d: the planned start of every action of the plans (``frontier``).
        max_cells: the most outcome cells an entry keeps (``frontier``), static.

    Returns:
        The frontier, ``(S, P M)``.
    """

    def at(price):
        return frontier(
            forecast,
            verdict,
            claim,
            star,
            utility,
            mode_models,
            menu,
            price,
            depth,
            star_index=star_index,
            time_d=time_d,
            max_cells=max_cells,
        )

    f = jax.vmap(at)(jnp.asarray(lam, float))  # leaves (P, S, M)

    def options(x):
        return jnp.moveaxis(x, 0, 1).reshape(x.shape[1], -1)

    return Frontier(*(options(x) for x in (f.reward, f.cost_d, f.mask, f.capped)))
