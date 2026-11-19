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
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree

from spaceodyssey._numerics import safe_ratio
from spaceodyssey.belief import (
    REFIT_KEPT,
    REFIT_LANDED,
    REFIT_REFUSED,
    REFIT_RETRIED,
    Replay,
    absorb,
    due_rows,
    new_planets,
    planning_beliefs,
    refit_stars,
    refused_after_rows,
    replay,
)
from spaceodyssey.claim import Claim
from spaceodyssey.design import Keepout, Resources, fits, next_open, open_at
from spaceodyssey.price import dispatch, refused_stars, solve_price
from spaceodyssey.roles import Roles
from spaceodyssey.types import (
    ACCEPTED,
    AbstractObservationModel,
    AbstractRefitter,
    Action,
    ObservationLog,
    Pool,
    Refit,
    as_float,
    as_index,
    clock_d,
    planet_tracks,
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

REFIT_NEW_PLANET = 1
"""Why a planner queues a refit: the star's rows detected a planet its cloud holds no
track for (``new_planets``)."""

REFIT_GATE = 2
"""Why a planner queues a refit: the star's rows left a planner role's weights below
the reweighting gate (``refused_after_rows``)."""


class PublicView(eqx.Module):
    """What a policy may see of a run at the current time.

    Attributes:
        log: the log, with the payload of every row not yet released zeroed: a row in
            flight shows its star, mode, times and charge, not its data.
        time_d: the current calendar time, days: when the instrument is next free.
        budget_left_d: the budget not yet charged, days.
        end_d: the calendar's end, days: no row starts after it.
        keepout: when each star can be observed (the design's table).
        in_flight: the rows taken and not yet released, shape ``(W,)``.
        n_taken: the actions taken so far, refused ones included: the rows after the
            grant.
    """

    log: ObservationLog
    time_d: Float[Array, ""] = eqx.field(converter=as_float)
    budget_left_d: Float[Array, ""] = eqx.field(converter=as_float)
    end_d: Float[Array, ""] = eqx.field(converter=as_float)
    keepout: Keepout
    in_flight: Bool[Array, " W"] = eqx.field(converter=jnp.asarray)
    n_taken: Int[Array, ""] = eqx.field(converter=as_index)


def public_view(
    log: ObservationLog,
    n_granted: Int[Array, ""],
    resources: Resources,
    keepout: Keepout,
) -> PublicView:
    """The public view of a run at the calendar time its rows end (``clock_d``).

    Args:
        log: the log, grant rows first.
        n_granted: the number of grant rows at the head of the log.
        resources: the run's budget and the calendar's end.
        keepout: the design's keepout table.

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
        end_d=resources.end_d,
        keepout=keepout,
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
            (``role_belief``, ``role_pool``) and of a refit's new weights
            (``AbstractRefitter``); NaN is refused on the host.
        max_khat: the largest Pareto shape of a role's reweighted draws or marks
            and of a refit's new weights; NaN is refused on the host.
        price_factors: the plans of each star are built at the last solved price times
            each factor and offered side by side (``frontier_at_prices``), shape
            ``(P,)``, beside the first action alone (the plan at an infinite price,
            which no factor of a zero price reaches); at ``depth = 1`` only the first
            factor is used, since a plan does not depend on the price there. A star
            whose build price is so far from the solved price that the price has left
            the factors' range (below the least factor or above the greatest, times
            the build price) is re-priced (``PricePolicy``).
        depth: the number of actions a plan may take at a star, static.
        member: the verdict member the planner plans with, static, required: planning
            reads one member (declared deviation 6: a minimum over members inside
            ``LogScore`` is not a proper score), so which member steers the plan is a
            study's choice, recorded with its results. Only that member is gated and
            put on nodes; a star the gate refuses for it, or for the forecast, is
            refused (``PriceState.refused``) while the other members may still judge
            it in ``score``.
        n_quad: quadrature nodes per gas factor (``quadrature_draws``), static: 24
            values a reading binned at the floor within 0.9 percent to S/N 50, 48
            within 0.06 percent; 16 is within 2.7 percent to S/N 20 only, at 0.7 times
            the cost.
        price_rounds: how many times the time-zero price is re-solved on plans rebuilt
            at it, static; one at ``depth = 1``.
        star_batch: stars priced at once when every star is priced (at time zero),
            static: a memory bound.
        rebuild_capacity: the most stars a decision reads afresh (a refit, the rows
            released at them, or a re-price), static: a decision gathers at most this
            many and the rest wait their turn, so a decision's cost is bounded by it.
            Worlds run one at a time (``execute`` runs one); under ``vmap`` over
            worlds every gathered slot, and every refit slot, would be computed.
        refit_capacity: the most refits a decision runs, static, at most
            ``rebuild_capacity``: the gathered stars queued for a refit fill this many
            slots and only those slots are refit; a queued star beyond them is not
            read this decision and stays pending, first in line for the next.
        refit_batch: refit slots mapped at once (``refit_stars``), static: one maps
            them one at a time, so an empty slot costs nothing; more maps them under
            ``vmap``, where every slot of a chunk is refit.
        forecast_updates: whether the forecast's draws learn from the released rows,
            static. When false they stay at the replay at time zero, a frozen-forecast
            surrogate that is not a martingale; the pool follows the searches taken
            either way, since thinning reads no payload (``replay_pool``).
        retry_factor: the proposal factor of the one retry of a refit the gate
            refused (``refit_stars``), static: the retry draws this many times the
            refitter's proposals.
    """

    min_n_eff: Float[Array, ""] = eqx.field(converter=as_float)
    max_khat: Float[Array, ""] = eqx.field(converter=as_float)
    price_factors: Float[Array, " P"] = eqx.field(converter=as_float)
    depth: int = eqx.field(static=True)
    member: int = eqx.field(static=True)
    n_quad: int = eqx.field(static=True)
    price_rounds: int = eqx.field(static=True)
    star_batch: int = eqx.field(static=True)
    rebuild_capacity: int = eqx.field(static=True)
    forecast_updates: bool = eqx.field(static=True)
    retry_factor: int = eqx.field(static=True)
    refit_capacity: int = eqx.field(static=True)
    refit_batch: int = eqx.field(static=True)

    def __init__(
        self,
        min_n_eff,
        max_khat,
        depth: int,
        *,
        member: int,
        n_quad: int = 24,
        price_factors=(0.5, 1.0, 2.0),
        price_rounds: int = 3,
        star_batch: int = 8,
        rebuild_capacity: int = 2,
        forecast_updates: bool = True,
        retry_factor: int = 8,
        refit_capacity: int = 1,
        refit_batch: int = 1,
    ):
        """Settings; the thresholds, the depth and the member are required."""
        self.min_n_eff = min_n_eff
        self.max_khat = max_khat
        self.price_factors = price_factors
        self.depth = int(depth)
        self.member = int(member)
        self.n_quad = int(n_quad)
        self.price_rounds = int(price_rounds)
        self.star_batch = int(star_batch)
        self.rebuild_capacity = int(rebuild_capacity)
        self.forecast_updates = bool(forecast_updates)
        self.retry_factor = int(retry_factor)
        self.refit_capacity = min(int(refit_capacity), self.rebuild_capacity)
        self.refit_batch = int(refit_batch)

    def __check_init__(self):
        """Check the static settings, and that no threshold is NaN."""
        for name in ("min_n_eff", "max_khat"):
            try:
                nan = bool(np.any(np.isnan(np.asarray(getattr(self, name)))))
            except jax.errors.TracerArrayConversionError:
                nan = False  # traced: checked where it was set
            if nan:
                raise ValueError(f"{name} is NaN: a NaN threshold gates nothing")
        counts = (
            self.depth,
            self.price_rounds,
            self.star_batch,
            self.rebuild_capacity,
            self.retry_factor,
            self.refit_capacity,
            self.refit_batch,
        )
        if min(counts) < 1 or self.member < 0:
            raise ValueError(
                "depth, price_rounds, star_batch, rebuild_capacity, retry_factor, "
                "refit_capacity and refit_batch are at least 1, member at least 0"
            )
        if self.price_factors.ndim != 1 or self.price_factors.shape[0] < 1:
            raise ValueError("price_factors is a nonempty vector")


class PriceState(eqx.Module):
    """What a ``PricePolicy`` carries from one decision to the next.

    Every per-star field has a leading star axis ``S``. A star's refit, its pool and
    its replay live here, not in the policy: a refit that lands swaps them.

    Attributes:
        replay: each star's replay of the log into its refit now (``Replay.refit``),
            at the last time the star was read. The verdict reads it, and the forecast
            too when it learns.
        frozen: each star's replay at time zero, after the grant, which a frozen
            forecast reads; None when the forecast learns (one replay serves both).
        pool: each star's pool as its refit now left it: the searches its refit does
            not represent thin it at each decision (``replay_pool``).
        stage: the stage each star's released rows have reached (``reached_stage``).
        frontier: each star's options, its plans built at the price factors and, at
            ``depth > 1``, its entries alone (``_star_plans``), unmasked.
        built_lam: the price each star's plans were built at.
        lam: the last solved price of time, reward per day.
        bound: the dual at ``lam``: no schedule of the offered plans, at most one per
            star, earns more within the budget left.
        refused: the stars with an option the menu offers that is not finite (a role
            the reweighting gate refused): those options are priced as if absent and
            reported here, never read as worth zero, whether or not the star could be
            bought now (busy, in keepout, or out of budget).
        pending: the stars whose released rows the state has not read yet (a refit
            waiting among them): busy, offering nothing, until a decision reads them.
        seen: the log rows released by the last decision, ``(W,)``.
        refits: the refits that landed at each star in the run, ``(S,)``.
        refit_refused: why each star's last refit was refused after its retry
            (``REFIT_NEW_PLANET`` or ``REFIT_GATE``), 0 when its last refit landed or
            none was refused, ``(S,)``. A star refused for a new planet keeps its old
            cloud, the planet untracked and its detection unscored: it is planned on
            its pool and its tracks, and the refit is queued again when its next rows
            are read. A star refused at the gate keeps its old cloud and its rows, so
            the role the gate refused reads NaN and the star is reported in
            ``refused``, its options priced as if absent. Either way its pool keeps
            every search taken (``replay_pool`` reads the log), so the planets it found
            stay found and its expected planets stay in the forecast.
    """

    replay: Replay
    frozen: Replay | None
    pool: Pool
    stage: Int[Array, " S"] = eqx.field(converter=as_index)
    frontier: Frontier
    built_lam: Float[Array, " S"] = eqx.field(converter=as_float)
    lam: Float[Array, ""] = eqx.field(converter=as_float)
    bound: Float[Array, ""] = eqx.field(converter=as_float)
    refused: Bool[Array, " S"] = eqx.field(converter=jnp.asarray)
    pending: Bool[Array, " S"] = eqx.field(converter=jnp.asarray)
    seen: Bool[Array, " W"] = eqx.field(converter=jnp.asarray)
    refits: Int[Array, " S"] = eqx.field(converter=as_index)
    refit_refused: Int[Array, " S"] = eqx.field(converter=as_index)


class PricePolicy(AbstractPolicy):
    """The planner: a price on survey time over each star's plans, re-solved as it goes.

    Each star offers plans (``frontier``: a first action and up to ``depth - 1``
    follow-ups at the star, each chosen on the outcomes so far, all at the decision's
    time), built at several prices around the last solved one
    (``frontier_at_prices``) and, at ``depth > 1``, each first action alone. The price
    of time ``lam`` minimizes the dual over the stars' options within the budget left
    (``solve_price``), and ``dispatch`` buys at most one option per star.

    Keepout is a constraint on when an option runs, not on whether it is in the plan.
    The price is solved over every star that is not busy (a row of it in flight, or
    its released rows not yet read) and opens before the calendar's end: a star in
    keepout that opens before the end stays in the price as future demand, its
    options priced, so budget is held for it when it beats the open stars at the
    price. Of the options bought, the one at a star open now with the most reward per
    day of its expected time runs now. When every option bought is at a star in
    keepout, the policy waits (an action of zero exposure, ``WAITED``) for the
    earliest opening among those stars, or an earlier release, or, when stars wait
    to be read, now; when nothing is bought it waits for the next release or a read
    the same way. When there is nothing to wait for before the calendar's end, or no
    entry fits the budget left, it stops. An option whose first action does not fit
    the budget left (``fits``) is not offered. This formulation assumes that a
    star's window opens before the calendar's end with the budget it was bought with
    still left (the budget is a fraction of the calendar), and that calendar time is
    free: there is no capacity per window, so stars bought for one window may
    together ask for more of it than it holds. A price per window, with a capacity
    each, is the form that would handle a saturated window.

    ``init`` replays every star at the time of the view, builds every star's plans in
    batches of stars, and re-solves the price ``price_rounds`` times on plans rebuilt
    at it; a star whose granted rows found a planet its refit holds no track for, or
    left a planner role's weights below the gate, starts pending. ``decide`` marks the
    stars of the rows released since the last decision pending, then reads at most
    ``rebuild_capacity`` stars, pending ones first and then those whose price has left
    the range of their build factors. A pending star appends its released rows
    (``absorb``) and is gated after them (``refused_after_rows``); one whose rows
    found a new planet (``new_planets``) or left a planner role's weights below the
    gate is refit, the decision's queue in one call (``refit_stars``: the star's
    cloud, its new rows as the term, a key folded from the decision's key and the
    star, the gate's thresholds, one retry at ``retry_factor`` times the proposals),
    and its pool is the old one thinned by the searches the new refit represents
    (``replay_pool``). A refit refused after its retry leaves the star its appended
    replay and the reason in ``PriceState.refit_refused``. Each star read moves to the
    stage its rows reached and gets plans rebuilt at the last price, which becomes
    its build price. The gathered stars are scattered back, the price is solved once
    on the options offered, and the action chosen. The beliefs are read as two, the
    forecast's and the planner's verdict member's (``planning_beliefs``, only that
    member's roles and pool rows passed), gated per role after their rows: an option
    that comes out NaN is refused, its star reported in the state and the option
    priced as if absent.

    The policy reads the public view only, never a world: everything it learns is in
    its state.

    Attributes:
        refits: each star's refit at time zero, leaves with a leading star axis.
        pool: each star's pool at time zero, every role, its role prior ratios formed
            under ``roles`` (``role_pool``).
        stars: the stars, a pytree whose leaves have a leading star axis.
        roles: the forecast and the stacked verdict members.
        claim: the claim the plans are valued on.
        utility: the utility, shared by the stars.
        models: the run's model table, static in structure; ``Row.model_slot``
            indexes it.
        mode_models: the model of each mode (``mode_models`` of the design);
            ``menu.mode`` indexes it.
        menu: each star's menu; its ``stage`` is not read (the state's is).
        refitter: the producer of a star's new refit when its searches find a planet
            new to its cloud.
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
    refitter: AbstractRefitter
    settings: PlannerSettings

    def init(self, public: PublicView) -> PriceState:
        """Every star replayed and priced at the time of the view (after the grant)."""
        return _price_init(self, public)

    def decide(
        self, key: PRNGKeyArray, state: PriceState, public: PublicView
    ) -> tuple[Action, Bool[Array, ""], PriceState]:
        """Read the released rows and stale stars, re-price, and run, wait or stop."""
        return _price_decide(self, key, state, public)


def _pick(tree, s):
    """Star ``s`` of a pytree with a leading star axis."""
    return jax.tree.map(lambda x: x[s], tree)


def _factors(policy: PricePolicy) -> Float[Array, " P"]:
    """The price factors in use: the first alone at depth 1."""
    factors = policy.settings.price_factors
    return factors if policy.settings.depth > 1 else factors[:1]


def _planner_roles(policy: PricePolicy) -> tuple[Roles, Int[Array, " 2"]]:
    """The roles the planner reads (the forecast and its member) and their pool rows."""
    m = policy.settings.member
    member = jax.tree.map(lambda x: x[m : m + 1], policy.roles.verdict)
    return Roles(policy.roles.forecast, member), jnp.asarray([0, 1 + m])


def _pool_now(policy, pool, replay_s, log, time_d) -> Pool:
    """A star's pool at the decision time: its refit's pool thinned since the refit."""
    return replay_pool(pool, replay_s.refit, log, policy.models, time_d)


def _star_plans(
    policy: PricePolicy,
    forecast: Replay,
    verdict: Replay,
    pool: Pool,
    s: Int[Array, ""],
    log: ObservationLog,
    time_d: Float[Array, ""],
    lam: Float[Array, ""],
) -> tuple[Int[Array, ""], tuple[Array, Array, Array]]:
    """Star ``s``'s stage and its options: reward, cost and mask, a frontier row each.

    ``pool`` is the star's pool at ``time_d``. The frontier is formed once every star
    is in. Option ``p M + m`` is entry ``m``'s plan built at ``lam`` times factor
    ``p``; at ``depth > 1`` the last ``M`` options are the entries alone, no follow-up
    (the plans at an infinite price), so the time left can always be spent on single
    actions even when the last price, and so every factor of it, was zero.
    ``option % M`` is the entry either way. Every action is planned to start at
    ``time_d``.
    """
    settings = policy.settings
    star = _pick(policy.stars, s)
    stage = reached_stage(log, policy.models, s, time_d)
    roles, rows = _planner_roles(policy)
    pool = Pool(
        pool.marks,
        pool.log_prior_ratio[rows],
        pool.log_like[rows],
        pool.log_mass[rows],
        pool.law,
    )
    forecast_b, verdict_b = planning_beliefs(
        forecast,
        verdict,
        roles,
        policy.models,
        star,
        pool,
        policy.claim,
        settings.n_quad,
        min_n_eff=settings.min_n_eff,
        max_khat=settings.max_khat,
    )
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

    args = (one(forecast_b), one(verdict_b), policy.claim, one(star), policy.utility)
    args = (*args, policy.mode_models, one_star)
    where = {"star_index": s[None], "time_d": time_d}
    f = frontier_at_prices(*args, lam * _factors(policy), settings.depth, **where)
    if settings.depth > 1:
        alone = frontier(*args, lam, 1, **where)
        f = jax.tree.map(lambda a, b: jnp.concatenate([a, b], -1), f, alone)
    return stage, (f.reward[0], f.cost_d[0], f.mask[0])


def _usable(
    policy: PricePolicy, f: Frontier, closed: Bool[Array, " S"], budget_d
) -> Frontier:
    """The options a decision may buy: offered, at an open star, first action in budget.

    ``closed`` marks the stars that offer nothing now: busy (a row in flight, or rows
    not yet read), in keepout, or past the calendar's end. An option whose first
    action does not fit the budget left would be refused by ``step``, so it is not
    offered.
    """
    n_plan = f.reward.shape[-1] // policy.menu.exposure_d.shape[-1]
    exposure_d = jnp.tile(policy.menu.exposure_d, (1, n_plan))
    usable = f.mask & ~closed[:, None] & fits(exposure_d, budget_d)
    return Frontier(f.reward, f.cost_d, usable)


def _collapsed(policy: PricePolicy, r: Replay, s: Int[Array, ""]) -> Bool[Array, ""]:
    """Whether the gate refuses a planner role on a replay after the rows it applied.

    The roles read are the planner's (the forecast and its member); a frozen forecast
    reads the replay at time zero instead, so only the member is read off ``r`` then.
    """
    roles, _ = _planner_roles(policy)
    settings = policy.settings
    refused = refused_after_rows(
        r,
        roles,
        policy.models,
        _pick(policy.stars, s),
        min_n_eff=settings.min_n_eff,
        max_khat=settings.max_khat,
    )
    return jnp.any(refused if settings.forecast_updates else refused[1:])


def _append(policy, r, s, log, time_d):
    """Star ``s``'s replay with its released rows appended, and why to refit it.

    The rows are absorbed first (``absorb``), then the gate reads the weights they
    leave (``refused_after_rows``): a star whose rows detected a planet its refit holds
    no track for (``new_planets``, ``REFIT_NEW_PLANET``), or left a planner role's
    weights below the gate (``REFIT_GATE``), is queued for a refit; 0 for neither.
    """
    absorbed = absorb(r, log, policy.models, _pick(policy.stars, s), time_d)
    new = jnp.any(new_planets(r.refit, log, policy.models, time_d))
    collapsed = _collapsed(policy, absorbed, s)
    reason = jnp.where(new, REFIT_NEW_PLANET, jnp.where(collapsed, REFIT_GATE, 0))
    return absorbed, jnp.asarray(reason, int)


def _read_star(
    policy, fresh_s, landed_s, absorbed_s, pool_s, frozen_s, s, key, log, time_d, lam
):
    """Star ``s`` read afresh at ``time_d``: refit or absorbed, its stage and plans.

    A star whose refit landed takes its new refit ``fresh_s`` (``refit_stars``),
    replayed to ``time_d``, and its pool becomes the old one thinned by the searches the
    new refit represents; any other, a refused refit among them, keeps its replay with
    the released rows appended (``absorbed_s``) and its pool. A frozen forecast
    ``frozen_s`` (None when the forecast learns) gains each planet the refit added, at
    its time-zero belief (``AbstractRefitter.add_tracks``, with ``key``), replayed at
    time zero. Its plans are rebuilt at ``lam``.
    """
    star = _pick(policy.stars, s)
    models = policy.models

    def refit_now(_):
        pool = _pool_now(policy, pool_s, absorbed_s, log, time_d)
        return replay(fresh_s, log, models, star, time_d), pool

    def appended(_):
        return absorbed_s, pool_s

    replay_s, pool_s = jax.lax.cond(landed_s, refit_now, appended, None)
    if frozen_s is not None:
        added = planet_tracks(replay_s.refit.atoms)
        added = added & ~planet_tracks(frozen_s.refit.atoms)

        def extend(_):
            refit = policy.refitter.add_tracks(frozen_s.refit, added, key, star)
            return replay(refit, log, models, star, jnp.zeros_like(time_d))

        frozen_s = jax.lax.cond(jnp.any(added), extend, lambda _: frozen_s, None)
    forecast = replay_s if frozen_s is None else frozen_s
    pool_now = _pool_now(policy, pool_s, replay_s, log, time_d)
    stage, f = _star_plans(policy, forecast, replay_s, pool_now, s, log, time_d, lam)
    return replay_s, pool_s, frozen_s, stage, f


@eqx.filter_jit
def _price_init(policy: PricePolicy, public: PublicView) -> PriceState:
    """``PricePolicy.init``, compiled once per policy structure."""
    log, time_d = public.log, public.time_d
    n_star = policy.menu.mask.shape[0]
    stars = jnp.arange(n_star)
    models = policy.models
    replays = jax.vmap(lambda refit, star: replay(refit, log, models, star, time_d))(
        policy.refits, policy.stars
    )
    pending = jax.vmap(
        lambda r, s: (
            jnp.any(new_planets(r.refit, log, models, time_d))
            | _collapsed(policy, r, s)
        )
    )(replays, stars)
    budget_d = public.budget_left_d
    busy = jnp.zeros(n_star, bool).at[log.rows.star].max(public.in_flight)
    closed = busy | pending | ~_reachable(public, n_star)

    def plans(lam):
        def one(s):
            r = _pick(replays, s)
            pool = _pool_now(policy, _pick(policy.pool, s), r, log, time_d)
            return _star_plans(policy, r, r, pool, s, log, time_d, lam)

        return jax.lax.map(one, stars, batch_size=policy.settings.star_batch)

    def round_(_, carry):
        lam = carry[0]
        stage, f = plans(lam)
        f = Frontier(*f)
        new_lam, bound, _ = solve_price(_usable(policy, f, closed, budget_d), budget_d)
        return new_lam, bound, refused_stars(f), stage, f, lam

    start = jax.eval_shape(lambda: round_(0, (jnp.zeros(()),)))
    start = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), start)
    rounds = 1 if policy.settings.depth == 1 else policy.settings.price_rounds
    lam, bound, refused, stage, f, built = jax.lax.fori_loop(0, rounds, round_, start)
    released = real_rows(log) & (log.rows.release_d <= time_d)
    frozen = None if policy.settings.forecast_updates else replays
    return PriceState(
        replay=replays,
        frozen=frozen,
        pool=policy.pool,
        stage=stage,
        frontier=f,
        built_lam=jnp.full(n_star, built),
        lam=lam,
        bound=bound,
        refused=refused,
        pending=pending,
        seen=released,
        refits=jnp.zeros(n_star, int),
        refit_refused=jnp.zeros(n_star, int),
    )


def _open_now(public: PublicView, n_star: int) -> Bool[Array, " S"]:
    """The stars out of keepout at the view's time, none after the calendar's end."""
    stars = jnp.arange(n_star)
    is_open = jax.vmap(open_at, in_axes=(None, 0, None))(
        public.keepout, stars, public.time_d
    )
    return is_open & (public.time_d <= public.end_d)


def _opens_d(public: PublicView, n_star: int) -> Float[Array, " S"]:
    """When each star next leaves keepout, at or after the view's time; inf if never."""
    stars = jnp.arange(n_star)
    return jax.vmap(next_open, in_axes=(None, 0, None))(
        public.keepout, stars, public.time_d
    )


def _reachable(public: PublicView, n_star: int) -> Bool[Array, " S"]:
    """The stars open now or opening before the calendar's end: the stars priced.

    A star in keepout that opens before the end is future demand: its options stay in
    the price, so budget is held for it when it beats the open stars at the price.
    """
    return _opens_d(public, n_star) <= public.end_d


def _stale(policy: PricePolicy, state: PriceState) -> Bool[Array, " S"]:
    """The stars whose build price the last solved price has left the factors' range."""
    if policy.settings.depth == 1:  # static: a plan does not depend on the price
        return jnp.zeros_like(state.pending)
    factors = policy.settings.price_factors
    low, high = jnp.min(factors) * state.built_lam, jnp.max(factors) * state.built_lam
    return ~((state.lam >= low) & (state.lam <= high))


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
    pending = state.pending | jnp.zeros(n_star, bool).at[rows.star].max(new)

    # Gather the stars to read, pending first, then stale, lowest index first.
    rank = jnp.where(pending, 0, jnp.where(_stale(policy, state), 1, 2))
    picks = jnp.argsort(rank * n_star + jnp.arange(n_star))[
        : policy.settings.rebuild_capacity
    ]
    chosen = rank[picks] < 2

    # Append: each gathered star with rows released since it was read absorbs them;
    # one whose rows found a planet new to its cloud, or left a planner role's weights
    # below the gate, joins the refit queue.
    models, settings = policy.models, policy.settings
    reading = chosen & pending[picks]

    def append(i):
        s = picks[i]
        r = _pick(state.replay, s)
        return jax.lax.cond(
            reading[i],
            lambda _: _append(policy, r, s, log, time_d),
            lambda _: (r, jnp.asarray(0, int)),
            None,
        )

    absorbed, reason = jax.lax.map(append, jnp.arange(picks.shape[0]))
    queued = reason > 0

    # Refit: the queued stars fill the refit slots, queued first, and only those slots
    # are refit; a queued star left over is not read now and stays pending.
    n_pick = picks.shape[0]
    slots = jnp.argsort(jnp.where(queued, 0, 1) * n_pick + jnp.arange(n_pick))[
        : settings.refit_capacity
    ]
    deferred = queued & ~jnp.zeros(n_pick, bool).at[slots].set(True)
    old = absorbed.refit

    def refit_slots():
        slot_old = _pick(old, slots)
        slot_star = picks[slots]
        fresh_s, status_s = refit_stars(
            policy.refitter,
            slot_old,
            jax.vmap(lambda r: due_rows(log, r, time_d))(slot_old),
            _pick(policy.stars, slot_star),
            jax.vmap(jax.random.fold_in, in_axes=(None, 0))(key, slot_star),
            queued[slots],
            log,
            models,
            min_n_eff=settings.min_n_eff,
            max_khat=settings.max_khat,
            retry_factor=settings.retry_factor,
            batch_size=settings.refit_batch,
        )
        fresh = jax.tree.map(lambda f, part: f.at[slots].set(part), old, fresh_s)
        return fresh, jnp.zeros(n_pick, int).at[slots].set(status_s)

    fresh, status = jax.lax.cond(
        jnp.any(queued),
        refit_slots,
        lambda: (old, jnp.full(n_pick, REFIT_KEPT, int)),
    )
    landed = (status == REFIT_LANDED) | (status == REFIT_RETRIED)
    chosen = chosen & ~deferred

    def one(i):
        s = picks[i]
        replay_s, pool_s = _pick(state.replay, s), _pick(state.pool, s)
        frozen_s = None if state.frozen is None else _pick(state.frozen, s)
        row = _pick(state.frontier, s)
        kept = (
            replay_s,
            pool_s,
            frozen_s,
            state.stage[s],
            (row.reward, row.cost_d, row.mask),
        )
        args = (_pick(fresh, i), landed[i], _pick(absorbed, i), pool_s, frozen_s, s)
        added_key = jax.random.fold_in(jax.random.fold_in(key, s), 1)
        return jax.lax.cond(
            chosen[i],
            lambda _: _read_star(policy, *args, added_key, log, time_d, state.lam),
            lambda _: kept,
            None,
        )

    replay_c, pool_c, frozen_c, stage_c, f_c = jax.lax.map(
        one, jnp.arange(picks.shape[0])
    )

    def scatter(full, part):
        return full.at[picks].set(part)

    replays = jax.tree.map(scatter, state.replay, replay_c)
    frozen = (
        None if state.frozen is None else jax.tree.map(scatter, state.frozen, frozen_c)
    )
    pools = jax.tree.map(scatter, state.pool, pool_c)
    stage = state.stage.at[picks].set(stage_c)
    old = (state.frontier.reward, state.frontier.cost_d, state.frontier.mask)
    f = Frontier(*jax.tree.map(scatter, old, f_c))
    built_lam = state.built_lam.at[picks].set(
        jnp.where(chosen, state.lam, state.built_lam[picks])
    )
    pending = pending.at[picks].set(pending[picks] & ~chosen)
    refits = state.refits.at[picks].add(landed.astype(int))
    refit_refused = state.refit_refused.at[picks].set(
        jnp.where(
            status == REFIT_REFUSED,
            reason,
            jnp.where(status == REFIT_KEPT, state.refit_refused[picks], 0),
        )
    )

    budget_d = public.budget_left_d
    in_flight = jnp.zeros(n_star, bool).at[rows.star].max(public.in_flight)
    is_open = _open_now(public, n_star)
    opens_d = _opens_d(public, n_star)
    # Priced: every star not busy that opens before the calendar's end, closed ones as
    # future demand. Run: the best bought option at a star open now.
    usable = _usable(
        policy, f, in_flight | pending | ~(opens_d <= public.end_d), budget_d
    )
    lam, bound, _ = solve_price(usable, budget_d)
    refused = refused_stars(f)
    bought, _ = dispatch(lam, usable, budget_d)
    runnable = bought & is_open[:, None]

    per_day = jnp.where(runnable, safe_ratio(usable.reward, usable.cost_d), -jnp.inf)
    best = jnp.max(per_day, axis=-1)
    s = jnp.argmax(best)
    entry = jnp.argmax(runnable[s]) % policy.menu.mask.shape[-1]
    run = Action(
        s, jnp.asarray(policy.menu.mode)[entry], policy.menu.exposure_d[s, entry]
    )

    # What a wait could wake on: a release, the opening of a star in keepout holding a
    # bought option, or now, when stars wait to be read.
    flight = jnp.where(public.in_flight, rows.release_d, jnp.inf)
    awaited = jnp.argmin(flight)
    wanted = jnp.any(bought, axis=-1) & ~is_open
    opens_d = jnp.where(wanted, opens_d, jnp.inf)
    reading_d = jnp.where(jnp.any(pending), time_d, jnp.inf)
    wake_d = jnp.min(jnp.stack([flight[awaited], jnp.min(opens_d), reading_d]))
    wait_star = jnp.select(
        [wake_d == flight[awaited], wake_d == jnp.min(opens_d)],
        [rows.star[awaited], jnp.argmin(opens_d)],
        jnp.argmax(pending),
    )
    wait = Action(wait_star, policy.menu.mode[0], 0.0, wake_d)

    running = jnp.any(runnable)
    affordable = jnp.any(policy.menu.mask & fits(policy.menu.exposure_d, budget_d))
    waiting = ~running & affordable & (wake_d <= public.end_d)
    action = jax.tree.map(lambda a, b: jnp.where(running, a, b), run, wait)
    new_state = PriceState(
        replay=replays,
        frozen=frozen,
        pool=pools,
        stage=stage,
        frontier=f,
        built_lam=built_lam,
        lam=lam,
        bound=bound,
        refused=refused,
        pending=pending,
        seen=released,
        refits=refits,
        refit_refused=refit_refused,
    )
    return action, ~running & ~waiting, new_state
