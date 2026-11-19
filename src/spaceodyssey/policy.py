"""Policies: what to observe next, decided from the public record of a run.

A policy never sees the sealed world. It is handed a ``PublicView``, reductions of the
log at the current time: the budget left, the rows in flight and the rows released,
their payloads only once released. ``execute`` builds the view and asks the policy for
an action until the policy stops, threading the policy's state between decisions.
"""

import abc
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree

from spaceodyssey._numerics import safe_ratio
from spaceodyssey.belief import Replay, absorb, planning_beliefs, replay, select_roles
from spaceodyssey.claim import Claim
from spaceodyssey.design import Resources
from spaceodyssey.price import dispatch, solve_price
from spaceodyssey.roles import Roles
from spaceodyssey.types import (
    ACCEPTED,
    AbstractObservationModel,
    Action,
    ObservationLog,
    Pool,
    Refit,
    as_float,
    as_index,
    clock_d,
    real_rows,
    spent_d,
)
from spaceodyssey.value import (
    AbstractUtility,
    Frontier,
    Menu,
    frontier,
    frontier_at_prices,
    reached_stage,
    replay_pool,
)


class PublicView(eqx.Module):
    """What a policy may see of a run at the current time.

    Attributes:
        log: the log, with the payload of every row not yet released zeroed: a row in
            flight shows its star, mode, times and charge, not its data.
        time_d: the current calendar time, days: when the instrument is next free.
        budget_left_d: the budget not yet charged, days.
        in_flight: the rows taken and not yet released, shape ``(W,)``.
        n_taken: the actions taken so far, refused ones included: the rows after the
            grant.
    """

    log: ObservationLog
    time_d: Float[Array, ""] = eqx.field(converter=as_float)
    budget_left_d: Float[Array, ""] = eqx.field(converter=as_float)
    in_flight: Bool[Array, " W"] = eqx.field(converter=jnp.asarray)
    n_taken: Int[Array, ""] = eqx.field(converter=as_index)


def public_view(
    log: ObservationLog, n_granted: Int[Array, ""], resources: Resources
) -> PublicView:
    """The public view of a run at the calendar time its rows end (``clock_d``).

    Args:
        log: the log, grant rows first.
        n_granted: the number of grant rows at the head of the log.
        resources: the run's budget.

    Returns:
        The view a policy decides from.
    """
    rows = log.rows
    time_d = clock_d(log)
    real = real_rows(log)
    released = real & (rows.release_d <= time_d)

    def hide(x):
        return jnp.where(released.reshape(-1, *([1] * (x.ndim - 1))), x, 0)

    payload = jax.tree.map(hide, rows.payload)
    public = ObservationLog(eqx.tree_at(lambda r: r.payload, rows, payload), log.count)
    return PublicView(
        log=public,
        time_d=time_d,
        budget_left_d=resources.budget_d - spent_d(log),
        in_flight=real & ~released,
        n_taken=log.count - n_granted,
    )


class AbstractPolicy(eqx.Module):
    """A rule for the next action, from the public view alone.

    A policy is a record and a pair of functions on it. ``init`` gives the policy's
    state after the grant; ``decide`` takes a key, that state and the public view and
    returns the next action, whether to stop instead, and the new state. The state is
    a record (an ``eqx.Module``) that ``execute`` threads from one decision to the next
    without reading it, so a policy carries what it learns between decisions in its
    state and never in itself. ``stop`` is a traced boolean: when it is true the run
    ends and the action is ignored.

    ``execute`` never reads a belief: it shows the policy the public view and takes the
    action it returns. A policy that plans on a belief keeps that belief, and anything
    it caches from it (each star's frontier, the last solved price), in its state. It
    keeps them current through ``absorb``, applying the rows the view shows released
    since its last decision, and never replays rows it has not been shown.

    A planning policy reads the forecast and the verdict as two beliefs, each from its
    own replay of the log. Whether the forecast's replay follows the decision time (a
    learning forecast) or stays at time zero (a frozen forecast, the replay at
    ``time_d = 0``, which holds the grant's rows only) is the static field
    ``forecast_updates`` of the planner (``PricePolicy``), its only reader. With the
    forecast frozen the planner maximizes a frozen-forecast surrogate, which is not a
    martingale.

    ``execute`` compiles ``decide`` once (``eqx.filter_jit``), so a decision costs its
    arithmetic, not a trace: the policy record is an argument, its arrays traced and its
    static fields part of the compiled program. A policy that must see concrete values
    at each decision (one that records or checksums what it is shown on the host) sets
    the class constant ``host_only``, and ``execute`` then calls its ``decide``
    directly; it cannot run under a compiled loop.

    Attributes:
        host_only: whether ``decide`` runs on the host, uncompiled; false by default.
    """

    host_only: ClassVar[bool] = False

    @abc.abstractmethod
    def init(self, public: PublicView) -> eqx.Module:
        """The policy's state after the grant.

        Args:
            public: the public view of the run after the grant.

        Returns:
            The state, a record.
        """

    @abc.abstractmethod
    def decide(
        self, key: PRNGKeyArray, state: eqx.Module, public: PublicView
    ) -> tuple[Action, Bool[Array, ""], eqx.Module]:
        """The next action, whether to stop, and the new state.

        Args:
            key: the decision's key, folded from the run's policy key (``run_keys``)
                and the number of actions taken.
            state: the policy's state.
            public: the public view of the run.

        Returns:
            The action, the stop flag (when true the run ends and the action is not
            taken) and the new state.
        """


class TableState(eqx.Module):
    """Where a ``TablePolicy`` is in its table.

    Attributes:
        cursor: index of the table's next action.
    """

    cursor: Int[Array, ""] = eqx.field(converter=as_index)


class TablePolicy(AbstractPolicy):
    """Replays a fixed table of actions in order, then stops.

    The table is a schedule made outside the run (an EXOSIMS, AYO or M4OPT schedule,
    or an oracle's): action ``i`` is taken as the run's ``i``-th action, whatever the
    rows before it showed, and a refused action is not retried. The state is the index
    of the next action, a traced counter, so a decision reads nothing concrete.

    Attributes:
        actions: an ``Action`` whose leaves have a leading axis over the table.
    """

    actions: Action

    def init(self, public: PublicView) -> TableState:
        """The cursor at the table's first action."""
        return TableState(0)

    def decide(
        self, key: PRNGKeyArray, state: TableState, public: PublicView
    ) -> tuple[Action, Bool[Array, ""], TableState]:
        """The table's next action; stop past its end."""
        n_action = self.actions.star.shape[0]
        if n_action == 0:  # a static shape: an empty table stops at once
            return Action(0, 0, 0.0), jnp.asarray(True), state
        index = jnp.minimum(state.cursor, n_action - 1)
        action = jax.tree.map(lambda leaf: leaf[index], self.actions)
        return action, state.cursor >= n_action, TableState(state.cursor + 1)


class PlannerSettings(eqx.Module):
    """The planner's settings, one record.

    The reweighting thresholds and the price factors are arrays, so a run with other
    values traces nothing new; the shapes and the structure of the lookahead are
    static.

    Attributes:
        min_n_eff: the least Kish size of a role's reweighted draws or marks
            (``role_belief``, ``role_pool``).
        max_khat: the largest Pareto shape of a role's reweighted draws or marks.
        price_factors: the plans of each star are built at the last solved price times
            each factor and offered side by side (``frontier_at_prices``), shape
            ``(P,)``, beside the first action alone (the plan at an infinite price,
            which no factor of a zero price reaches); at ``depth = 1`` only the first
            factor is used, since a plan does not depend on the price there.
        depth: the number of actions a plan may take at a star, static.
        n_quad: quadrature nodes per gas factor (``quadrature_draws``), static: 24
            values a reading binned at the floor within 0.9 percent to S/N 50, 48
            within 0.06 percent; 16 is within 2.7 percent to S/N 20 only, at 0.7 times
            the cost.
        member: the verdict member the planner reads, static (planning reads one
            member, a declared deviation).
        price_rounds: how many times the time-zero price is re-solved on plans rebuilt
            at it, static; one at ``depth = 1``.
        star_batch: stars priced at once when every star is priced (at time zero),
            static: a memory bound.
        forecast_updates: whether the forecast's draws learn from the released rows,
            static. When false they stay at the replay at time zero, a frozen-forecast
            surrogate that is not a martingale; the pool follows the searches taken
            either way, since thinning reads no payload (``replay_pool``).
    """

    min_n_eff: Float[Array, ""] = eqx.field(converter=as_float)
    max_khat: Float[Array, ""] = eqx.field(converter=as_float)
    price_factors: Float[Array, " P"] = eqx.field(converter=as_float)
    depth: int = eqx.field(static=True)
    n_quad: int = eqx.field(static=True)
    member: int = eqx.field(static=True)
    price_rounds: int = eqx.field(static=True)
    star_batch: int = eqx.field(static=True)
    forecast_updates: bool = eqx.field(static=True)

    def __init__(
        self,
        min_n_eff,
        max_khat,
        depth: int,
        *,
        n_quad: int = 24,
        price_factors=(0.5, 1.0, 2.0),
        member: int = 0,
        price_rounds: int = 3,
        star_batch: int = 8,
        forecast_updates: bool = True,
    ):
        """Settings; the thresholds and the depth are required."""
        self.min_n_eff = min_n_eff
        self.max_khat = max_khat
        self.price_factors = price_factors
        self.depth = int(depth)
        self.n_quad = int(n_quad)
        self.member = int(member)
        self.price_rounds = int(price_rounds)
        self.star_batch = int(star_batch)
        self.forecast_updates = bool(forecast_updates)

    def __check_init__(self):
        """Check the static settings."""
        if self.depth < 1 or self.price_rounds < 1 or self.star_batch < 1:
            raise ValueError("depth, price_rounds and star_batch are at least 1")
        if self.price_factors.ndim != 1 or self.price_factors.shape[0] < 1:
            raise ValueError("price_factors is a nonempty vector")


class PriceState(eqx.Module):
    """What a ``PricePolicy`` carries from one decision to the next.

    Every per-star field has a leading star axis ``S``.

    Attributes:
        forecast: the forecast's replay of each star: at the decision time when the
            forecast learns, at time zero when it is frozen.
        verdict: the verdict's replay of each star, at the last time the star was
            priced.
        stage: the stage each star's released rows have reached (``reached_stage``).
        frontier: each star's options, its plans built at the price factors and, at
            ``depth > 1``, its entries alone (``_star_plans``), unmasked; a star is
            rebuilt only when rows of it are released.
        lam: the last solved price of time, reward per day.
        bound: the dual at ``lam``: no schedule of the offered plans, at most one per
            star, earns more within the budget left.
        refused: the stars with an option the menu offers that is not finite (a role
            the reweighting gate refused), priced as if absent and reported here,
            never read as worth zero; reported whether or not the star could be bought
            now (busy, or out of budget).
        seen: the log rows released by the last decision, ``(W,)``.
    """

    forecast: Replay
    verdict: Replay
    stage: Int[Array, " S"] = eqx.field(converter=as_index)
    frontier: Frontier
    lam: Float[Array, ""] = eqx.field(converter=as_float)
    bound: Float[Array, ""] = eqx.field(converter=as_float)
    refused: Bool[Array, " S"] = eqx.field(converter=jnp.asarray)
    seen: Bool[Array, " W"] = eqx.field(converter=jnp.asarray)


class PricePolicy(AbstractPolicy):
    """The planner: a price on survey time over each star's plans, re-solved as it goes.

    Each star offers plans (``frontier``: a first action and up to ``depth - 1``
    follow-ups at the star, each chosen on the outcomes so far), built at several
    prices around the last solved one (``frontier_at_prices``) and, at ``depth > 1``,
    each first action alone. The price of time
    ``lam`` minimizes the dual over the stars' options within the budget left
    (``solve_price``), and ``dispatch`` buys at most one option per star. Of the options
    bought, the one with the most reward per day of its expected time runs now; a star
    with a row in flight offers nothing until the row is released. When nothing is
    bought and rows are in flight, the policy waits for the next release (an action of
    zero exposure that starts then, ``WAITED``); when nothing is bought and nothing is
    in flight, or no entry fits the budget left, it stops.

    ``init`` replays every star at time zero, builds every star's plans in batches of
    stars, and re-solves the price ``price_rounds`` times on plans rebuilt at it.
    ``decide`` absorbs the rows released since the last decision into the replays of
    their stars, moves each such star to the stage its rows reached and its pool to
    the searches taken (``reached_stage``, ``replay_pool``), rebuilds only those stars'
    plans at the last price, and solves the price once. The beliefs are read as two,
    the forecast's and one verdict member's (``planning_beliefs``), gated per role:
    a star whose options come out NaN is refused, reported in the state and priced as
    if absent.

    The policy reads the public view only, never a world: everything it learns is in
    its state.

    Attributes:
        refits: each star's refit at time zero, leaves with a leading star axis.
        pool: each star's pool at time zero, every role, its role prior ratios formed
            under ``roles`` (``role_pool``).
        stars: the stars, a pytree whose leaves have a leading star axis.
        roles: the forecast and the stacked verdict members.
        claim: the claim the plans are valued on.
        utility: the utility; its array leaves (``Count``'s time-zero pool) carry a
            leading star axis.
        models: the run's model table, static in structure; ``Row.model_slot``
            indexes it.
        mode_models: the model of each mode (``mode_models`` of the design);
            ``menu.mode`` indexes it.
        menu: each star's menu; its ``stage`` is not read (the state's is).
        settings: the planner's settings.
    """

    refits: Refit
    pool: Pool
    stars: PyTree
    roles: Roles
    claim: Claim
    utility: AbstractUtility
    models: tuple[AbstractObservationModel, ...]
    mode_models: tuple[AbstractObservationModel, ...]
    menu: Menu
    settings: PlannerSettings

    def init(self, public: PublicView) -> PriceState:
        """Every star replayed and priced at the time of the view (after the grant)."""
        return _price_init(self, public)

    def decide(
        self, key: PRNGKeyArray, state: PriceState, public: PublicView
    ) -> tuple[Action, Bool[Array, ""], PriceState]:
        """Absorb the released rows, re-price their stars, and run or wait or stop."""
        return _price_decide(self, key, state, public)


def _pick(tree, s):
    """Star ``s`` of a pytree with a leading star axis."""
    return jax.tree.map(lambda x: x[s], tree)


def _factors(policy: PricePolicy) -> Float[Array, " P"]:
    """The price factors in use: the first alone at depth 1."""
    factors = policy.settings.price_factors
    return factors if policy.settings.depth > 1 else factors[:1]


def _star_plans(
    policy: PricePolicy,
    forecast: Replay,
    verdict: Replay,
    s: Int[Array, ""],
    log: ObservationLog,
    time_d: Float[Array, ""],
    lam: Float[Array, ""],
) -> tuple[Int[Array, ""], tuple[Array, Array, Array]]:
    """Star ``s``'s stage and its options: reward, cost and mask, a frontier row each.

    The frontier is formed once every star is in. Option ``p M + m`` is entry ``m``'s
    plan built at ``lam`` times factor ``p``; at ``depth > 1`` the last ``M`` options
    are the entries alone, no follow-up (the plans at an infinite price), so the time
    left can always be spent on single actions even when the last price, and so every
    factor of it, was zero. ``option % M`` is the entry either way.
    """
    settings = policy.settings
    star, refit = _pick(policy.stars, s), _pick(policy.refits, s)
    pool = replay_pool(_pick(policy.pool, s), refit, log, policy.models, time_d)
    stage = reached_stage(log, policy.models, s, time_d)
    forecast_b, verdict_b = planning_beliefs(
        forecast,
        verdict,
        policy.roles,
        policy.models,
        star,
        pool,
        pool,
        policy.claim,
        settings.n_quad,
        min_n_eff=settings.min_n_eff,
        max_khat=settings.max_khat,
    )
    member = slice(settings.member, settings.member + 1)
    verdict_b = select_roles(verdict_b, member)
    menu = policy.menu
    one_star = Menu(
        menu.exposure_d[s][None],
        menu.mask[s][None],
        menu.mode,
        menu.requires_stage,
        stage[None],
    )

    def one(tree):
        return jax.tree.map(lambda x: x[None], tree)

    utility = jax.tree.map(lambda x: x[s][None], policy.utility)
    args = (one(forecast_b), one(verdict_b), policy.claim, one(star), utility)
    args = (*args, policy.mode_models, one_star)
    f = frontier_at_prices(
        *args, lam * _factors(policy), settings.depth, star_index=s[None]
    )
    if settings.depth > 1:
        alone = frontier(*args, lam, 1, star_index=s[None])
        f = jax.tree.map(lambda a, b: jnp.concatenate([a, b], -1), f, alone)
    return stage, (f.reward[0], f.cost_d[0], f.mask[0])


def _usable(
    policy: PricePolicy, f: Frontier, busy: Bool[Array, " S"], budget_d
) -> Frontier:
    """The options a decision may buy: offered, at a free star, first action in budget.

    A star with a row in flight offers nothing; an option whose first action exceeds
    the budget left would be refused by ``step``, so it is not offered.
    """
    n_plan = f.reward.shape[-1] // policy.menu.exposure_d.shape[-1]
    exposure_d = jnp.tile(policy.menu.exposure_d, (1, n_plan))
    usable = f.mask & ~busy[:, None] & (exposure_d <= budget_d)
    return Frontier(f.reward, f.cost_d, usable)


def _refused(f: Frontier) -> Bool[Array, " S"]:
    """The stars with an offered option that is not finite."""
    finite = jnp.isfinite(f.reward) & jnp.isfinite(f.cost_d)
    return jnp.any(f.mask & ~finite, axis=-1)


@eqx.filter_jit
def _price_init(policy: PricePolicy, public: PublicView) -> PriceState:
    """``PricePolicy.init``, compiled once per policy structure."""
    log, time_d = public.log, public.time_d
    n_star = policy.menu.mask.shape[0]
    stars = jnp.arange(n_star)
    verdict = jax.vmap(
        lambda refit, star: replay(refit, log, policy.models, star, time_d)
    )(policy.refits, policy.stars)
    busy = jnp.zeros(n_star, bool).at[log.rows.star].max(public.in_flight)
    budget_d = public.budget_left_d

    def plans(lam):
        def one(s):
            return _star_plans(
                policy, _pick(verdict, s), _pick(verdict, s), s, log, time_d, lam
            )

        return jax.lax.map(one, stars, batch_size=policy.settings.star_batch)

    def round_(_, carry):
        lam = carry[0]
        stage, f = plans(lam)
        f = Frontier(*f)
        lam, bound, _ = solve_price(_usable(policy, f, busy, budget_d), budget_d)
        return lam, bound, _refused(f), stage, f

    start = jax.eval_shape(lambda: round_(0, (jnp.zeros(()),)))
    start = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), start)
    rounds = 1 if policy.settings.depth == 1 else policy.settings.price_rounds
    lam, bound, refused, stage, f = jax.lax.fori_loop(0, rounds, round_, start)
    released = real_rows(log) & (log.rows.release_d <= time_d)
    return PriceState(verdict, verdict, stage, f, lam, bound, refused, released)


@eqx.filter_jit
def _price_decide(
    policy: PricePolicy, key: PRNGKeyArray, state: PriceState, public: PublicView
) -> tuple[Action, Bool[Array, ""], PriceState]:
    """``PricePolicy.decide``, compiled once per policy structure."""
    log, time_d = public.log, public.time_d
    rows = log.rows
    n_star = policy.menu.mask.shape[0]
    released = real_rows(log) & (rows.release_d <= time_d)
    new = released & ~state.seen & (rows.refusal == ACCEPTED)
    touched = jnp.zeros(n_star, bool).at[rows.star].max(new)
    busy = jnp.zeros(n_star, bool).at[rows.star].max(public.in_flight)

    def one(s):
        forecast, verdict = _pick(state.forecast, s), _pick(state.verdict, s)
        row = _pick(state.frontier, s)
        kept = (forecast, verdict, state.stage[s], (row.reward, row.cost_d, row.mask))

        def rebuild(_):
            star = _pick(policy.stars, s)
            verdict_now = absorb(verdict, log, policy.models, star, time_d)
            forecast_now = (
                absorb(forecast, log, policy.models, star, time_d)
                if policy.settings.forecast_updates
                else forecast
            )
            stage, f = _star_plans(
                policy, forecast_now, verdict_now, s, log, time_d, state.lam
            )
            return forecast_now, verdict_now, stage, f

        return jax.lax.cond(touched[s], rebuild, lambda _: kept, None)

    forecast, verdict, stage, f = jax.lax.map(one, jnp.arange(n_star))
    f = Frontier(*f)
    budget_d = public.budget_left_d
    usable = _usable(policy, f, busy, budget_d)
    lam, bound, _ = solve_price(usable, budget_d)
    refused = _refused(f)
    bought, _ = dispatch(lam, usable, budget_d)

    per_day = jnp.where(bought, safe_ratio(usable.reward, usable.cost_d), -jnp.inf)
    best = jnp.max(per_day, axis=-1)
    s = jnp.argmax(best)
    entry = jnp.argmax(bought[s]) % policy.menu.mask.shape[-1]
    run = Action(
        s, jnp.asarray(policy.menu.mode)[entry], policy.menu.exposure_d[s, entry]
    )
    flight = jnp.where(public.in_flight, rows.release_d, jnp.inf)
    awaited = jnp.argmin(flight)
    wait = Action(rows.star[awaited], rows.mode[awaited], 0.0, flight[awaited])

    running = jnp.any(bought)
    affordable = jnp.any(policy.menu.mask & (policy.menu.exposure_d <= budget_d))
    waiting = ~running & jnp.any(public.in_flight) & affordable
    action = jax.tree.map(lambda a, b: jnp.where(running, a, b), run, wait)
    new_state = PriceState(forecast, verdict, stage, f, lam, bound, refused, released)
    return action, ~running & ~waiting, new_state
