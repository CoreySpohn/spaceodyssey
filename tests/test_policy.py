"""The planner, `PricePolicy`, through `execute`: milestone 2 (the AYO limit), dynamic
completeness, the refused set, rows in flight and waits, plans at several prices, the
realized spend, non-interference and the compile counter.

The constructed stars hold their planets in the pool (marks in the HZ or not; a search,
`DetectionModel`, finds a planet in one day with a probability set by its HZ bit), and
their four draws hold no planet, so `Count` is read off the pool alone.
"""

import math
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from spaceodyssey.belief import (
    REFIT_REFUSED,
    due_rows,
    new_planets,
    planning_beliefs,
    pool_rate,
    refit_stars,
    replay,
    replay_likelihood,
    role_pool,
)
from spaceodyssey.claim import Claim
from spaceodyssey.design import (
    Design,
    Keepout,
    Mode,
    Resources,
    mode_models,
    no_keepout,
)
from spaceodyssey.log import execute, grant_rows, start_log
from spaceodyssey.policy import (
    REFIT_GATE,
    AbstractPolicy,
    PlannerSettings,
    PricePolicy,
    TablePolicy,
    public_view,
)
from spaceodyssey.price import dispatch, solve_price
from spaceodyssey.roles import Roles
from spaceodyssey.score import found_count
from spaceodyssey.study import Study
from spaceodyssey.toy import DetectionModel, InHz, ToyPrior, ToyRefitter, point_atoms
from spaceodyssey.types import (
    ACCEPTED,
    OVER_BUDGET,
    WAITED,
    Action,
    Grant,
    Pool,
    Refit,
    spent_d,
)
from spaceodyssey.value import (
    Count,
    Menu,
    frontier_at_prices,
    pool_found,
    reached_stage,
    replay_pool,
)
from spaceodyssey.world import SealedWorld, run_keys

KEY = jax.random.key(0)
CAPACITY = 12
PRIOR = ToyPrior("toy", [0.5], 1.0, 0.5, 0.0, 1.0, n_gas=1)  # one slot, half the time
STAR = {"p_hz": jnp.asarray(0.5)}
CLAIM = Claim((InHz(),))
UNGATED = {"min_n_eff": 0.0, "max_khat": math.inf}
SEARCH = DetectionModel(0.8, 0.2)  # layer 5 test 5's search
N_STAR = 3  # the most stars a test here plans
END_D = 365.0  # the calendar's end, days
REFITTER = ToyRefitter(PRIOR)  # a new planet's state from the one-slot prior
DEPTH_1 = PlannerSettings(0.0, math.inf, 1, member=0)


def _stack(*trees):
    return jax.tree.map(lambda *x: jnp.stack(x), *trees)


def _pick(tree, s):
    return jax.tree.map(lambda x: x[s], tree)


def _no_grant():
    none = jnp.zeros(0, int)
    return Grant(Action(none, none, jnp.zeros(0)))


def _points(hz, exists, n):
    """`n` point draws (or marks) of one slot on the prior's chart."""
    hz = jnp.broadcast_to(jnp.asarray(hz, float), (n,))[:, None]
    exists = jnp.broadcast_to(jnp.asarray(exists, bool), (n,))[:, None]
    return point_atoms(hz, 0.0 * hz, jnp.zeros((n, 1, 1)), exists, PRIOR)


class Stars(eqx.Module):
    """Constructed stars: their refits (four draws, no planet), pools and roles."""

    refits: Refit
    pool: Pool
    stars: dict
    roles: Roles


def _stars(masses, hz=(1.0, 0.0, 1.0, 0.0), mark_log_weights=None, prior=PRIOR):
    """Stars whose planets are all in the pool: `masses` expected planets on marks of
    the HZ bits `hz`, weighed by `mark_log_weights` (equally by default) under the
    forecast `prior` and one verdict member.
    """
    roles = Roles(prior, _stack(prior))
    refits, pools = [], []
    for s, mass in enumerate(masses):
        draws = _points(0.0, False, 4)
        refits.append(Refit(draws, jnp.zeros(CAPACITY, bool), s))
        marks = _points(hz, True, len(hz))
        if mark_log_weights is not None:
            weights = mark_log_weights[s]
            marks = eqx.tree_at(lambda a: a.cloud.log_weights, marks, weights)
        log_mass = jnp.full(2, jnp.log(mass))  # an empty pool at mass 0
        pools.append(
            role_pool(marks, roles, jnp.zeros((2, len(hz))), log_mass, STAR, **UNGATED)
        )
    n = len(masses)
    return Stars(_stack(*refits), _stack(*pools), _stack(*[STAR] * n), roles)


def _world(n_star, hz=1.0, exists=True):
    """Every star holds one planet, in the HZ or not, or none. The planners' draws
    hold no planet, so a detection queues a refit (`ToyRefitter`); runs about the
    pool search worlds with none."""
    return SealedWorld(_stack(*[_points(hz, exists, 1)] * n_star))


def _design(budget_d, search=SEARCH, latency_d=0.0, keepout=None):
    """One search mode; no keepout unless given, a calendar of `END_D` days."""
    keepout = no_keepout(N_STAR) if keepout is None else keepout
    resources = Resources(budget_d, END_D)
    return Design((search,), (Mode(0, latency_d),), resources, keepout, CAPACITY)


def _menu(n_star, exposure_d=(1.0,)):
    exposure_d = jnp.broadcast_to(jnp.asarray(exposure_d), (n_star, len(exposure_d)))
    n_entry = exposure_d.shape[1]
    return Menu(exposure_d, jnp.ones(exposure_d.shape, bool), (0,) * n_entry)


def _policy(stars, design, menu, settings, refitter=REFITTER):
    return PricePolicy(
        stars.refits, stars.pool, stars.stars, stars.roles, CLAIM, Count(),
        design.models, mode_models(design), menu, refitter, settings,
    )  # fmt: skip


def _public(design):
    """The public view after an empty grant: time zero, the whole budget."""
    grant = _no_grant()
    payload = _world(1).reveal(run_keys(KEY)[1], grant, design.models, design.modes)
    log = start_log(grant_rows(grant, payload, design.modes), design.capacity)
    return public_view(log, jnp.asarray(0), design.resources, design.keepout)


@eqx.filter_jit
def _rates(pool, refit, log, models, time_d):
    """Each star's forecast claim-true pool rate at time zero and after the searches
    released by `time_d` (`replay_pool`)."""

    def one(pool, refit):
        now = replay_pool(pool, refit, log, models, time_d)
        return pool_rate(pool, CLAIM, STAR)[0], pool_rate(now, CLAIM, STAR)[0]

    return jax.vmap(one)(pool, refit)


@eqx.filter_jit
def _pool_found(stars, log, models, time_d):
    def one(pool, refit, star):
        return pool_found(pool, refit, log, models, CLAIM, star, time_d)[0]

    return jnp.sum(jax.vmap(one)(stars.pool, stars.refits, stars.stars))


def _found(stars, log, models, time_d):
    """Expected claim-true planets found by a run's searches over every star, under
    the forecast (`pool_found`)."""
    return float(_pool_found(stars, log, models, time_d))


def _column(log, name):
    return np.asarray(getattr(log.rows, name)[: int(log.count)])


class Trace(AbstractPolicy):
    """Defers to `inner` and records the state after `init` and each decision."""

    host_only: ClassVar[bool] = True  # records concrete values on the host

    inner: AbstractPolicy
    states: list = eqx.field(default_factory=list)

    def init(self, public):
        state = self.inner.init(public)
        self.states.append(state)
        return state

    def decide(self, key, state, public):
        action, stop, state = self.inner.decide(key, state, public)
        self.states.append(state)
        return action, stop, state


@pytest.mark.medium(
    reason="compiles the planner, a table run and the pool replay, about 7 s"
)
def test_milestone_2_the_ayo_limit():
    """Milestone 2: `Count` at depth 1 on the time-zero beliefs is the AYO allocation.

    Three stars with n_i expected claim-true planets, (n_i, tau_i) = (1.0, 1),
    (0.8, 2), (0.5, 1), each found by a search of t days with completeness
    C_i(t) = 1 - exp(-t / tau_i) (a one-day probability 1 - exp(-1 / tau_i) per star),
    budget 3 days, a menu of exposures h, 2h, ..., 3 days with h = 1/16 (exact in
    binary, so the bought exposures sum to the budget without rounding past it).
    `init` builds each option's reward through the planner's beliefs and `frontier`:
    n_i C_i(t), to 1e-12. The time-zero allocation (`dispatch` at the solved price)
    brackets the price by each funded star's marginal reward,
    V'(t_k + h) <= lam <= V'(t_k - h), as the continuous optimum (lam = 0.2512,
    t = 1.381, 0.930, 0.688 days; the closed form in `test_price`) equalizes them.
    Executed as a `TablePolicy`, its rows charge exactly the bought exposures, within
    the budget, and the pool thinned by the real rows (`replay_pool`) has found
    sum n_i C_i(t_k) planets, the reward bought.

    Fails if the search's completeness or `Count` misreads the pool (the rewards are
    not n_i C_i(t)), if the price misses the clearing breakpoint, if the allocation
    overspends, or if the real rows thin the pool other than the lookahead does.
    """
    n_bar, tau = np.array([1.0, 0.8, 0.5]), np.array([1.0, 2.0, 1.0])
    stars = _stars(n_bar, hz=(1.0,))
    design = _design(3.0, DetectionModel(-np.expm1(-1.0 / tau), 0.0))
    h = 1.0 / 16.0
    t = h * np.arange(1, 49)
    state = _policy(stars, design, _menu(3, t), DEPTH_1).init(_public(design))
    reward = n_bar[:, None] * -np.expm1(-t[None] / tau[:, None])
    assert_allclose(state.frontier.reward, reward, rtol=1e-12)

    lam = float(state.lam)
    bought = np.asarray(dispatch(state.lam, state.frontier, jnp.asarray(3.0))[0])
    assert np.all(bought.sum(axis=1) == 1)
    t_k = t[bought.argmax(axis=1)]
    slope = lambda x: n_bar / tau * np.exp(-x / tau)  # noqa: E731
    assert np.all(slope(t_k + h) <= lam) and np.all(lam <= slope(t_k - h))
    assert t_k.sum() <= 3.0 + 1e-12

    table = TablePolicy(Action(np.arange(3), np.zeros(3, int), t_k))
    log = execute(KEY, _no_grant(), _world(3), design, table)
    assert_allclose(_column(log, "charged_d"), t_k)
    assert_allclose(spent_d(log), t_k.sum())
    found = _found(stars, log, design.models, jnp.asarray(10.0))
    assert_allclose(found, reward[bought].sum(), rtol=1e-12)
    bound = float(state.bound)
    print(f"milestone 2: lam {lam:.4f}, t {t_k}, found {found:.4f}, bound {bound:.4f}")


@pytest.mark.medium(reason="compiles a planner per forecast mode, about 5 s each")
@pytest.mark.parametrize("updates", [True, False], ids=["learning", "frozen"])
def test_dynamic_completeness_follows_the_searches_taken(updates):
    """Spec layer 6 test 2: a search's worth is re-read after each real search.

    Layer 5 test 5's star (mass 2, marks in the HZ half the time, a one-day search
    finds an HZ planet with probability 0.8), `Count` at depth 1, budget 3 days, rows
    released at once. The first search finds 2 x 0.5 x 0.8 = 0.8 claim-true planets in
    expectation; each search leaves 0.2 of the claim-true rate, whatever it found, so
    the next is worth 0.16 and the third 0.032; then the budget is spent and the
    planner stops (its last state values a fourth at 0.0064), having found 0.992, the
    claim-true rate 1.0 less 0.2**3. A frozen forecast plans the same: the pool follows
    the searches taken, which read no payload.

    Fails if a star's options are not re-priced after its row releases (the second
    search would still be worth 0.8), if the pool is thinned only by a null or reads
    the payload, if a frozen forecast's pool stays at time zero (every search would be
    worth 0.8), or if the planner spends past its budget or stops early.
    """
    stars = _stars([2.0])
    design = _design(3.0)
    settings = PlannerSettings(0.0, math.inf, 1, member=0, forecast_updates=updates)
    trace = Trace(_policy(stars, design, _menu(1), settings))
    log = execute(KEY, _no_grant(), _world(1, exists=False), design, trace)
    rewards = [float(s.frontier.reward[0, 0]) for s in trace.states]
    assert_allclose(rewards, [0.8, 0.8, 0.16, 0.032, 0.0064], rtol=1e-10)
    assert int(log.count) == 3
    assert_allclose(spent_d(log), 3.0)
    assert_allclose(_found(stars, log, design.models, jnp.asarray(3.0)), 0.992)


def test_a_star_in_flight_offers_nothing_and_the_planner_waits():
    """Rows in flight and the wait: two stars (masses 2 and 1, so searches worth 0.8 and
    0.4), released 2 days after each one-day search, budget 3 days.

    Day 0: star 0 is searched (in flight until day 3). Day 1: star 0 offers nothing
    while its row is in flight, so star 1 is searched (until day 4). Day 2: both are in
    flight and nothing is bought, so the planner waits for the next release, a row of
    zero exposure not before day 3, charged nothing. Day 3: star 0's row is released,
    its next search is worth 0.16, and it is taken with the last day of budget. Day 4:
    nothing fits the budget left, so the planner stops without waiting on star 0's
    row, still in flight.

    Fails if a star with a row in flight offers an option (star 0 would be searched
    on day 1), if waiting is charged or uses a latency, if the planner waits when
    nothing can be bought, or stops while a release could still change the plan.
    """
    stars = _stars([2.0, 1.0])
    design = _design(3.0, latency_d=2.0)
    log = execute(
        KEY,
        _no_grant(),
        _world(2, exists=False),
        design,
        _policy(stars, design, _menu(2), DEPTH_1),
    )
    assert list(_column(log, "star")) == [0, 1, 0, 0]
    assert list(_column(log, "refusal")) == [ACCEPTED, ACCEPTED, WAITED, ACCEPTED]
    assert_allclose(_column(log, "time_d"), [0.0, 1.0, 3.0, 3.0])
    assert_allclose(_column(log, "charged_d"), [1.0, 1.0, 0.0, 1.0])
    assert_allclose(_column(log, "release_d"), [3.0, 4.0, 3.0, 6.0])
    assert_allclose(spent_d(log), 3.0)


def test_a_refused_star_is_reported_and_priced_as_absent():
    """A star whose roles the reweighting gate refuses is reported, not read as zero.

    Two stars (mass 2 each), gate `min_n_eff = 3`: star 0's four marks are equally
    weighted (Kish size 4) and star 1's hold one mark of weight (Kish size 1), so
    star 1's forecast and verdict pool rows are NaN, and so is every option it offers.
    Its draws pass the gate, so no refit is queued (a refit does not redraw a pool).
    Every state marks star 1 refused; the planner prices star 0 as if star 1 were
    absent and searches only it (0.8, then 0.16) within the budget of 2 days.

    Fails if a refused star's NaN reads as worth zero (it would not be reported) or
    stops the price (nothing would be bought at star 0), or if a refused star is
    searched.
    """
    weights = jnp.array([[0.0] * 4, [0.0, -jnp.inf, -jnp.inf, -jnp.inf]])
    stars = _stars([2.0, 2.0], mark_log_weights=weights)
    design = _design(2.0)
    settings = PlannerSettings(3.0, math.inf, 1, member=0)
    trace = Trace(_policy(stars, design, _menu(2), settings))
    log = execute(KEY, _no_grant(), _world(2, exists=False), design, trace)
    assert all(list(np.asarray(s.refused)) == [False, True] for s in trace.states)
    assert np.all(np.isnan(np.asarray(trace.states[0].frontier.reward[1])))
    assert list(_column(log, "star")) == [0, 0]


def test_the_pool_follows_the_searches_released_and_the_stage_their_finds():
    """`replay_pool` and `reached_stage` read the released rows of their star only.

    Two stars (masses 2 and 1, so claim-true rates 1.0 and 0.5), one-day searches
    released 2 days after they end, budget 3 days, a table: star 0 (released day 3),
    star 1 (day 4), a wait until day 4 at star 0, star 0 again (day 7) and 5 days at
    star 0, refused. A search leaves 0.2 of the claim-true rate, whatever it found:
    star 0's is 1.0 to day 3, 0.2 to day 7 and 0.04 after, the refused row and the
    wait adding nothing; star 1's 0.5, then 0.1 from day 4. A refit that represents
    star 0's first row (its pool carries that search already) is thinned only by the
    second. With a search certain to find an HZ planet at star 0 and blind at star 1
    (one-day probabilities 1 and 0), star 0 reaches stage 1 on day 3 and star 1 never.

    Fails if the pool is thinned by a row in flight, a refused row or a wait, by a
    row of another star or one the refit represents, or by a power other than one per
    search; or if a row with no detection, or one not yet released, moves the stage.
    """
    stars = _stars([2.0, 1.0])
    table = TablePolicy(
        Action([0, 1, 0, 0, 0], [0] * 5, [1.0, 1.0, 0.0, 1.0, 5.0], [0, 0, 4, 0, 0])
    )
    design = _design(3.0, latency_d=2.0)
    log = execute(KEY, _no_grant(), _world(2), design, table)
    assert list(_column(log, "refusal")) == [
        ACCEPTED, ACCEPTED, WAITED, ACCEPTED, OVER_BUDGET
    ]  # fmt: skip

    def rate(s, time_d, refits=stars.refits):
        return float(_rates(stars.pool, refits, log, design.models, time_d)[1][s])

    for time_d, expected in ((2.9, 1.0), (3.0, 0.2), (6.9, 0.2), (7.0, 0.04)):
        assert_allclose(rate(0, jnp.asarray(time_d)), expected, rtol=1e-12)
    assert_allclose(rate(0, jnp.asarray(100.0)), 0.04, rtol=1e-12)
    assert_allclose(rate(1, jnp.asarray(3.9)), 0.5, rtol=1e-12)
    assert_allclose(rate(1, jnp.asarray(4.0)), 0.1, rtol=1e-12)
    first = jnp.zeros((2, CAPACITY), bool).at[0, 0].set(True)  # star 0's first row
    represented = eqx.tree_at(lambda r: r.row_mask, stars.refits, first)
    assert_allclose(rate(0, jnp.asarray(100.0), represented), 0.2, rtol=1e-12)

    sure = _design(3.0, DetectionModel([1.0, 0.0], 0.0), latency_d=2.0)
    log = execute(KEY, _no_grant(), _world(2), sure, table)
    stages = [
        [
            int(reached_stage(log, sure.models, jnp.asarray(s), jnp.asarray(t)))
            for t in (2.9, 3.0)
        ]
        for s in (0, 1)
    ]
    assert stages == [[0, 1], [0, 0]]
    assert int(reached_stage(log, sure.models, jnp.asarray(1), jnp.asarray(100.0))) == 0


MEMBER = ToyPrior("toy", [0.5], 1.6, 0.5, 0.0, 1.0, n_gas=1)  # P(HZ) 0.8, not 0.5


def test_the_planner_weighs_the_pool_under_its_own_roles():
    """The planning beliefs weigh a pool's marks under the planner's roles and gate,
    not the ratios the pool carries in (`planning_beliefs` through `role_pool`).

    A star of mass 2 whose pool was formed under roles that are all the forecast prior
    (every ratio zero), read by a planner whose member puts P(HZ) = 0.8: the member's
    mark ratios are log(0.8 / 0.5) and log(0.2 / 0.5), its claim-true rate 2 x 0.8,
    the forecast's 2 x 0.5. At `min_n_eff = 3` the member's marks (Kish size 2.94) are
    refused: NaN, not the ratios carried in.

    Fails if the planner reads the pool's stored ratios, or gates the pool other than
    the draws.
    """
    stars = _stars([2.0])
    roles = Roles(PRIOR, _stack(MEMBER))
    log = _public(_design(1.0)).log
    refit, pool = _pick(stars.refits, 0), _pick(stars.pool, 0)
    r = replay(refit, log, (SEARCH,), STAR, jnp.asarray(0.0))

    def read(min_n_eff):
        return planning_beliefs(
            r, r, roles, (SEARCH,), STAR, pool, CLAIM, 4,
            min_n_eff=jnp.asarray(min_n_eff), max_khat=jnp.asarray(math.inf),
        )  # fmt: skip

    forecast, verdict = read(0.0)
    assert_allclose(
        verdict.pool.log_prior_ratio, np.log([[1.6, 0.4, 1.6, 0.4]]), rtol=1e-12
    )
    assert_allclose(pool_rate(verdict.pool, CLAIM, STAR), [1.6], rtol=1e-12)
    assert_allclose(pool_rate(forecast.pool, CLAIM, STAR), [1.0], rtol=1e-12)
    forecast, verdict = read(3.0)
    assert np.all(np.isnan(verdict.pool.log_prior_ratio))
    assert_allclose(pool_rate(forecast.pool, CLAIM, STAR), [1.0], rtol=1e-12)


# Plans at several prices. Three stars with 2.5 claim-true planets each, a one-day
# search finding each with probability 0.4: a first look finds 1.0, a revisit 0.6 more,
# so a star's value is V(lam) = max(0, 1 - lam, 1.6 - 2 lam), with its kink at 0.6.
# Budget 4 days: every first look fits (3 days), not every revisit (6).
REVISIT = DetectionModel(0.4, 0.0)


def _revisit_stars():
    return _stars([2.5] * 3, hz=(1.0,))


def _beliefs(stars, search):
    """Every star's planning beliefs at time zero, as the planner reads them."""
    log = _public(_design(1.0, search)).log

    def read(refit, pool):
        r = replay(refit, log, (search,), STAR, jnp.asarray(0.0))
        return planning_beliefs(
            r, r, stars.roles, (search,), STAR, pool, CLAIM, 4, **UNGATED
        )

    return eqx.filter_vmap(read)(stars.refits, stars.pool)


@pytest.mark.medium(reason="compiles a depth-2 frontier at three prices, about 5 s")
def test_plans_at_several_prices_hold_the_price():
    """A plan built at one price is a tangent there: at depth 2 a single price
    oscillates, several hold.

    On the three stars above, the dual of the exact V clears at the kink, lam = 0.6
    (slope B - 2N = -2 below it, B - N = 1 above). Re-solving on plans built at the
    last price alone: built at 0 every star revisits (reward 1.6 over 2 days), the
    demand is 6 days and the price 0.8; built at 0.8 no star revisits (1.0 over 1 day),
    the demand is 3 days and the price 0; then 0.8 again. Built at the last price times
    (0.5, 1, 2), the second round already offers both plans of each star, the price is
    0.6 and stays there, to 1e-12.

    Fails if a frontier's options are read as exact at every price (the single-price
    iteration would not oscillate) or if the plans at several prices are not offered
    side by side (the price would not settle at the kink).
    """
    stars = _revisit_stars()
    forecast, verdict = _beliefs(stars, REVISIT)
    menu, budget = _menu(3), jnp.asarray(4.0)

    @eqx.filter_jit
    def resolve(lam, factors):
        f = frontier_at_prices(
            forecast, verdict, CLAIM, stars.stars, Count(), (REVISIT,), menu,
            lam * factors, 2,
        )  # fmt: skip
        return solve_price(f, budget)[0]

    # One shape for both: three equal factors are the single price.
    for factors, expected in (
        ((1.0,) * 3, [0.8, 0.0, 0.8, 0.0]),
        ((0.5, 1.0, 2.0), None),
    ):
        lam, path = jnp.asarray(0.0), []
        for _ in range(4):
            lam = resolve(lam, jnp.asarray(factors))
            path.append(float(lam))
        if expected is not None:
            assert_allclose(path, expected, atol=1e-12)
        else:
            assert_allclose(path, [0.8, 0.6, 0.6, 0.6], atol=1e-12)


@pytest.mark.medium(reason="compiles the depth-2 planner, about 8 s")
def test_the_realized_spend_stays_within_the_budget():
    """Spec S10: a `PricePolicy` run at depth 2 never spends past its budget, and the
    spend of the log is the sum of its charged rows.

    The three stars above, the planner at depth 2 with plans at three prices, rows
    released one day after each search. Its time-zero price is the kink, 0.6; the run
    charges 4 days at most, no action it asks for is refused, and `spent_d` equals the
    charged column summed.

    Fails if follow-up time is left out of the plans' cost or the price misses the
    kink (the time-zero price would be 0.8 or 0), or if the planner asks for an
    action the budget left cannot pay for (a refused row).
    """
    stars = _revisit_stars()
    design = _design(4.0, REVISIT, latency_d=1.0)
    settings = PlannerSettings(0.0, math.inf, 2, member=0)
    trace = Trace(_policy(stars, design, _menu(3), settings))
    log = execute(KEY, _no_grant(), _world(3, exists=False), design, trace)
    assert_allclose(trace.states[0].lam, 0.6, atol=1e-12)
    charged = _column(log, "charged_d")
    assert OVER_BUDGET not in list(_column(log, "refusal"))
    assert spent_d(log) <= 4.0 + 1e-12
    assert_allclose(spent_d(log), charged.sum())


class BlindSearch(DetectionModel):
    """The search with a payload that never reads the truth: it never detects."""

    def sample(self, key, atoms, action):
        payload = super().sample(key, atoms, action)
        return {**payload, "detected": jnp.zeros_like(payload["detected"])}


def test_the_planner_cannot_read_the_truth(toy):
    """Non-interference for `PricePolicy` (layer 4 test 4's helper): two worlds, every
    planet in the HZ against none, whose rows a blind search forces identical, give
    the planner bit-identical keys, states and views and a bit-identical log; a search
    that reads the truth shows it different views.

    Fails if the planner reads anything of a sealed world beyond the released rows.
    """
    stars = _stars([2.0, 1.0])
    policy = _policy(stars, _design(3.0), _menu(2), DEPTH_1)
    worlds = (_world(2, hz=1.0), _world(2, hz=0.0))
    toy.assert_non_interference(
        KEY, policy, worlds, _design(3.0, BlindSearch(0.8, 0.2)), _design(3.0),
        _no_grant(),
    )  # fmt: skip


INIT_TRACES, DECIDE_TRACES = [], []


@eqx.filter_jit
def _counted_init(policy, public):
    INIT_TRACES.append(None)
    return policy.init(public)


class Counted(AbstractPolicy):
    """Defers to `inner`, counting the traces of its `init` and `decide`."""

    inner: AbstractPolicy

    def init(self, public):
        return _counted_init(self.inner, public)

    def decide(self, key, state, public):
        DECIDE_TRACES.append(None)
        return self.inner.decide(key, state, public)


def test_a_second_world_budget_and_prior_add_no_traces():
    """The planner compiles once per structure: a run on a second world (no planet in
    the HZ), with another budget, other pool masses, another forecast prior value and
    its searches at another star and another gate threshold, traces neither `init`
    nor `decide` again. Both worlds hold planets the draws do not, so each run's
    detections take the refit branch.

    Fails if a budget, a prior value, a mass or a threshold is a Python value or a
    static field, or a menu, star or row index is read on the host.
    """
    other = eqx.tree_at(lambda p: p.p_size, PRIOR, jnp.asarray(0.3))
    gated = PlannerSettings(0.5, math.inf, 1, member=0)  # another threshold value
    runs = (
        (_stars([2.0, 1.0]), 3.0, _world(2, hz=1.0), DEPTH_1),
        (_stars([0.5, 3.0], prior=other), 2.0, _world(2, hz=0.0), gated),
    )
    counts, starts = [], []
    for stars, budget_d, world, settings in runs:
        design = _design(budget_d)
        policy = Counted(_policy(stars, design, _menu(2), settings))
        log = execute(KEY, _no_grant(), world, design, policy)
        counts.append((len(INIT_TRACES), len(DECIDE_TRACES)))
        starts.append(int(log.rows.star[0]))
    assert counts[1] == counts[0]
    assert starts == [0, 1]


def test_a_new_detection_is_refit_and_its_star_planned_on():
    """A search finds a planet the time-zero cloud holds no track for: the detection is
    not scored against the draws, the star is refit, and the planner plans on it.

    Layer 5 test 5's star (mass 2, marks in the HZ half the time), its four draws
    holding no planet, in a world with one planet out of the HZ that the search finds
    for certain (one-day probabilities 0.8 in the HZ, 1 out of it). The first search
    detects it. Replayed over that row, the time-zero cloud keeps its weights: the
    detection is new (`new_planets`), so it is not scored. At the next decision the
    star is pending, then refit by `ToyRefitter`: every draw holds the planet in slot
    0, the refit represents the row, and the star's options stay finite and
    unrefused. The pool follows the searches, whatever they found, so the plan is test
    2's: rewards 0.8, 0.8, 0.16, 0.032, 0.0064 and three searches.

    The same refit through the queue (`refit_stars`) under a gate of five draws, more
    than the cloud holds, is refused at the first attempt and on the retry, and the
    star keeps its old refit, never a NaN cloud; a NaN threshold is refused when the
    settings are made.

    Fails if a new detection is scored against draws that hold no planet (their
    weights go to zero, the replay's to NaN), if the star is not refit (its draws hold
    no planet after the run) or is refused, if the refit thins the pool again by a
    row it represents, or if a refit's weights or a threshold escape the gate.
    """
    stars = _stars([2.0])
    design = _design(3.0, DetectionModel(0.8, 1.0))
    trace = Trace(_policy(stars, design, _menu(1), DEPTH_1))
    log = execute(KEY, _no_grant(), _world(1, hz=0.0), design, trace)
    assert list(_column(log, "star")) == [0, 0, 0]
    assert list(_column(log, "refusal")) == [ACCEPTED] * 3
    assert_allclose(log.rows.payload[0]["detected"][0], [1.0])

    refit0 = _pick(stars.refits, 0)
    after = jnp.asarray(1.0)
    assert list(np.asarray(new_planets(refit0, log, design.models, after))) == [True]
    r = replay(refit0, log, design.models, STAR, after)
    _, log_like = replay_likelihood(r, design.models, STAR)
    assert_allclose(log_like, np.zeros(4), atol=1e-12)

    rewards = [float(s.frontier.reward[0, 0]) for s in trace.states]
    assert_allclose(rewards, [0.8, 0.8, 0.16, 0.032, 0.0064], rtol=1e-10)
    assert list(np.asarray(trace.states[0].pending)) == [False]
    last = trace.states[-1]
    assert np.all(np.asarray(last.replay.refit.atoms.exists[0, :, 0]))
    assert bool(last.replay.refit.row_mask[0, 0])
    assert not np.any(np.asarray(last.refused))
    assert not np.any(np.asarray(last.pending))

    # The producer slot through its queue: a gate the four draws cannot meet refuses
    # the new cloud at both attempts, the old refit is kept, and a NaN threshold is
    # refused on the host.
    kept, status = refit_stars(
        REFITTER, _stack(refit0), due_rows(log, refit0, after)[None], _stack(STAR),
        jax.random.key(1)[None], jnp.array([True]), log, design.models,
        min_n_eff=5.0, max_khat=math.inf, retry_factor=8,
    )  # fmt: skip
    assert list(np.asarray(status)) == [REFIT_REFUSED]
    assert_allclose(kept.atoms.cloud.log_weights[0], 0.0)
    assert not np.any(np.asarray(kept.row_mask))
    with pytest.raises(ValueError, match="NaN"):
        PlannerSettings(math.nan, math.inf, 1, member=0)


LOW = ToyPrior("toy", [1.0], 0.4, 0.5, 0.0, 1.0, n_gas=1)  # in the HZ one time in five


def _collapsing_star(n_draw=64, n_mark=64):
    """One star whose `n_draw` draws hold its planet (in the HZ one time in five, the
    law `LOW`), and whose pool holds one planet in expectation, in the HZ."""
    roles = Roles(LOW, _stack(LOW))
    refit = Refit(LOW.draws(KEY, STAR, n_draw), jnp.zeros(CAPACITY, bool), 0)
    hz = jnp.ones((n_mark, 1))
    marks = point_atoms(hz, 0 * hz, jnp.zeros((n_mark, 1, 1)), hz > 0, LOW)
    pool = role_pool(
        marks, roles, jnp.zeros((2, n_mark)), jnp.zeros(2), STAR, **UNGATED
    )
    return Stars(_stack(refit), _stack(pool), _stack(STAR), roles)


def _collapsing_run(min_n_eff):
    """`_collapsing_star` searched by a search that finds an HZ planet in a day with
    probability 0.9 and never any other, in a world whose planet is in the HZ."""
    stars = _collapsing_star()
    design = _design(3.0, DetectionModel(0.9, 0.0))
    settings = PlannerSettings(min_n_eff, math.inf, 1, member=0)
    trace = Trace(_policy(stars, design, _menu(1), settings, ToyRefitter(LOW)))
    log = execute(KEY, _no_grant(), _world(1, hz=1.0), design, trace)
    return stars, design, trace, log


def test_rows_that_collapse_a_cloud_refit_the_star_and_it_stays_planned_on():
    """Rows appended to a star's cloud are gated after their likelihood is added; a
    star they collapse is refit, retried with more proposals, and planned on.

    The star's 64 draws hold its planet, in the HZ in about one in five; the search
    finds an HZ planet with probability 0.9 a day and no other. The first search finds
    the planet (a track), so every draw out of the HZ gets zero weight: the Kish size
    after the row is about 13, below the gate of 20, though the cloud before it held 64.
    The star is queued for a refit (`REFIT_GATE`). One proposal per draw gives a cloud
    of about 13 again, refused; the retry draws 8 per draw (`retry_factor`), Kish size
    about 43, and lands. The star is planned on: its pool (one HZ planet in
    expectation) is searched three times, rewards 0.9, 0.09, 0.009.

    Fails if the gate reads the weights before the rows (no refit: the planner plans on
    a cloud of 13 draws until the planning gate refuses the star, which is then never
    searched again), if a refused first attempt is not retried, or if a refit is
    queued for a star the rows did not collapse (the time-zero cloud passes).
    """
    _, _, trace, log = _collapsing_run(20.0)
    assert list(_column(log, "star")) == [0, 0, 0]
    assert_allclose(log.rows.payload[0]["detected"][0], [1.0])
    rewards = [float(s.frontier.reward[0, 0]) for s in trace.states]
    assert_allclose(rewards[:4], [0.9, 0.9, 0.09, 0.009], rtol=1e-10)
    assert list(np.asarray(trace.states[0].pending)) == [False]
    last = trace.states[-1]
    assert list(np.asarray(last.refits)) == [1]
    assert list(np.asarray(last.refit_refused)) == [0]
    assert not np.any(np.asarray(last.refused))
    assert bool(last.replay.refit.row_mask[0, 0])
    weights = np.exp(np.asarray(last.replay.refit.atoms.cloud.log_weights[0]))
    assert weights.sum() ** 2 / (weights**2).sum() >= 20.0
    hz = np.asarray(last.replay.refit.atoms.cloud.samples["hz"][0, :, 0])
    assert np.all(hz[weights > 0] == 1.0)  # every draw of weight holds an HZ planet


def test_a_refit_refused_after_its_retry_is_reported_and_the_star_kept():
    """A refit the gate refuses again on its retry leaves the star its old cloud, the
    reason reported, the star refused, and its pool with the searches taken.

    `_collapsing_run` with a gate of 50 draws: the retry's cloud (about 43) is refused
    too. The star keeps its time-zero refit and the row appended, so its roles read
    NaN: it is reported refused (`PriceState.refused`) with the reason `REFIT_GATE`
    (`PriceState.refit_refused`), and it is not searched again. Its pool is not
    frozen at time zero: it holds the search taken, so the forecast counts 0.9 planets
    found there and 0.1 left.

    Fails if a refused refit hands the planner a NaN cloud or a cloud the gate refused
    (it would be read as a refit landed), if the reason is lost, or if the star is
    planned on a cloud the gate refused (a second search).
    """
    stars, design, trace, log = _collapsing_run(50.0)
    assert list(_column(log, "star")) == [0]
    last = trace.states[-1]
    assert list(np.asarray(last.refits)) == [0]
    assert list(np.asarray(last.refit_refused)) == [REFIT_GATE]
    assert list(np.asarray(last.refused)) == [True]
    assert not bool(last.replay.refit.row_mask[0, 0])
    assert_allclose(last.replay.refit.atoms.cloud.log_weights[0], 0.0)
    end = jnp.asarray(10.0)
    assert_allclose(_found(stars, log, design.models, end), 0.9, rtol=1e-12)
    rate = _rates(last.pool, last.replay.refit, log, design.models, end)[1]
    assert_allclose(rate, [0.1], rtol=1e-10)


def test_a_star_in_keepout_offers_nothing_and_the_planner_waits_for_it():
    """A star in keepout offers no option, the planner waits for it to open, and no
    row starts after the calendar's end.

    Two stars, all marks in the HZ, masses 2 and 1, a search certain to find an HZ
    planet (worth 2 and 1, nothing after it), budget 3 days; star 0 is in keepout for
    the first 5 days. At day 0 star 0 offers nothing, so star 1 is searched though
    star 0 is worth more. At day 1 nothing open is worth its price, so the planner
    waits for star 0 to open: a wait at star 0, not before day 5, charged nothing; at
    day 5 it is searched. With the calendar ending at day 4, star 0 opens too late:
    the planner stops after star 1's search.

    Fails if a star in keepout offers an option (star 0 would be searched first, and
    `step` would delay it to day 5), if the planner does not wait for a star leaving
    keepout, or if it waits or asks for a row past the calendar's end.
    """
    stars = _stars([2.0, 1.0], hz=(1.0,))
    closed = np.zeros((N_STAR, 2), bool)
    closed[0, 0] = True  # star 0 closed over days [0, 5)
    keepout = Keepout(closed, 5.0)
    sure = DetectionModel(1.0, 0.0)
    design = _design(3.0, sure, keepout=keepout)
    log = execute(KEY, _no_grant(), _world(2, exists=False), design,
                  _policy(stars, design, _menu(2), DEPTH_1))  # fmt: skip
    assert list(_column(log, "star")) == [1, 0, 0]
    assert list(_column(log, "refusal")) == [ACCEPTED, WAITED, ACCEPTED]
    assert_allclose(_column(log, "time_d"), [0.0, 5.0, 5.0])
    assert_allclose(_column(log, "charged_d"), [1.0, 0.0, 1.0])

    short = eqx.tree_at(lambda d: d.resources.end_d, design, jnp.asarray(4.0))
    log = execute(KEY, _no_grant(), _world(2, exists=False), short,
                  _policy(stars, short, _menu(2), DEPTH_1))  # fmt: skip
    assert list(_column(log, "star")) == [1]
    assert list(_column(log, "refusal")) == [ACCEPTED]


def test_the_realized_count_is_read_against_the_truth():
    """`found_count` counts the claim-true planets of the truth a run's searches
    detected, new to the time-zero refits; `pool_found` is the forecast's expectation.

    Three stars, each searched once by a search certain to find any planet; the truth
    holds a planet at each: in the HZ at stars 0 and 2, out of it at star 1. Star 2's
    draws already hold its planet (a track). Realized: 1 at star 0 (claim-true, new),
    0 at star 1 (not claim-true), 0 at star 2 (tracked at time zero). The forecast
    expects 1.5 at star 0 (mass 3, half its marks in the HZ, all found) and 0.5 at
    stars 1 and 2.

    Fails if the realized count is read off the forecast (1.5, 0.5, 0.5), counts a
    planet that is not claim-true or one the time-zero refit held, or counts a row
    that was not accepted.
    """
    stars = _stars([3.0, 1.0, 1.0])
    tracked = _points(1.0, True, 4)
    refits = jax.tree.map(
        lambda all_, one: all_.at[2].set(one), stars.refits.atoms, tracked
    )
    stars = eqx.tree_at(lambda s: s.refits.atoms, stars, refits)
    design = _design(3.0, DetectionModel(1.0, 1.0))
    world = SealedWorld(
        _stack(_points(1.0, True, 1), _points(0.0, True, 1), _points(1.0, True, 1))
    )
    study = Study(
        CLAIM, PRIOR, PRIOR, _stack(PRIOR), stars.refits, stars.pool, stars.stars,
        design, jnp.zeros((3, 2)), alpha=0.05,
    )  # fmt: skip
    table = TablePolicy(Action([0, 1, 2], [0, 0, 0], [1.0, 1.0, 1.0]))
    log = execute(KEY, _no_grant(), world, design, table)
    assert_allclose(np.asarray(log.rows.payload[0]["detected"][:3, 0]), 1.0)
    assert list(np.asarray(found_count(log, world, study))) == [1, 0, 0]

    end = jnp.asarray(10.0)

    def expected(pool, refit, star):
        return pool_found(pool, refit, log, design.models, CLAIM, star, end)[0]

    forecast = jax.vmap(expected)(stars.pool, stars.refits, stars.stars)
    assert_allclose(forecast, [1.5, 0.5, 0.5], rtol=1e-12)


@pytest.mark.medium(reason="compiles the depth-2 planner's init and decide, about 6 s")
def test_a_star_whose_build_price_the_price_left_is_re_priced():
    """A star's plans are rebuilt when the solved price leaves its factors' range.

    The three revisit stars at depth 2, built at the time-zero price 0.6 (factors 0.5,
    1, 2: plans at 0.3, 0.6 and 1.2), then decided with a budget of 100 days, so every
    plan fits and the price falls to 0. The first decision reads no star (none is
    stale at 0.6) and solves 0. The next finds every star stale (0 is below 0.3) and
    re-prices the two it has room for, stars 0 and 1: their build price becomes 0 and
    their plans at every factor revisit (reward 1.6), while star 2 keeps its plan at
    1.2 (reward 1.0). The decision after re-prices star 2.

    Fails if a stale star is never re-priced (its build price stays 0.6), if a star
    is re-priced past the decision's capacity, or if the build price is not the price
    the plans were built at.
    """
    stars = _revisit_stars()
    settings = PlannerSettings(0.0, math.inf, 2, member=0)
    policy = _policy(stars, _design(4.0, REVISIT), _menu(3), settings)
    state = policy.init(_public(_design(4.0, REVISIT)))
    assert_allclose(state.lam, 0.6, atol=1e-12)
    assert_allclose(state.built_lam, 0.6, atol=1e-12)
    rich = _public(_design(100.0, REVISIT))
    built, lams, reward = [], [], []
    for _ in range(3):
        _, _, state = policy.decide(KEY, state, rich)
        built.append(np.asarray(state.built_lam))
        lams.append(float(state.lam))
        reward.append(np.asarray(state.frontier.reward[:, :3]))
    assert_allclose(lams, [0.0, 0.0, 0.0], atol=1e-12)
    assert_allclose(built, [[0.6] * 3, [0.0, 0.0, 0.6], [0.0] * 3], atol=1e-12)
    assert_allclose(reward[1][:2], 1.6, rtol=1e-12)
    assert_allclose(reward[1][2, 2], 1.0, rtol=1e-12)
    assert_allclose(reward[2], 1.6, rtol=1e-12)
