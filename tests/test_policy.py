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
    planning_beliefs,
    pool_rate,
    replay,
    role_pool,
)
from spaceodyssey.claim import Claim
from spaceodyssey.design import Design, Mode, Resources, mode_models
from spaceodyssey.log import execute, grant_rows, start_log
from spaceodyssey.policy import (
    AbstractPolicy,
    PlannerSettings,
    PricePolicy,
    TablePolicy,
    public_view,
)
from spaceodyssey.price import dispatch, solve_price
from spaceodyssey.roles import Roles
from spaceodyssey.toy import DetectionModel, InHz, ToyPrior, point_atoms
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
DEPTH_1 = PlannerSettings(0.0, math.inf, 1)


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


def _stars(masses, hz=(1.0, 0.0, 1.0, 0.0), log_weights=None, prior=PRIOR):
    """Stars whose planets are all in the pool: `masses` expected planets on marks of
    the HZ bits `hz`, weighed equally by the forecast `prior` and one verdict member.
    """
    roles = Roles(prior, _stack(prior))
    refits, pools = [], []
    for s, mass in enumerate(masses):
        draws = _points(0.0, False, 4)
        if log_weights is not None:
            draws = eqx.tree_at(lambda a: a.cloud.log_weights, draws, log_weights[s])
        refits.append(Refit(draws, jnp.zeros(CAPACITY, bool), s))
        marks = _points(hz, True, len(hz))
        log_mass = jnp.full(2, jnp.log(mass))  # an empty pool at mass 0
        pools.append(
            role_pool(marks, roles, jnp.zeros((2, len(hz))), log_mass, STAR, **UNGATED)
        )
    n = len(masses)
    return Stars(_stack(*refits), _stack(*pools), _stack(*[STAR] * n), roles)


def _world(n_star, hz=1.0, exists=True):
    """Every star holds one planet, in the HZ or not, or none: a planner whose draws
    hold no planet reads a detection as impossible (the star's beliefs go NaN until a
    refit explains it), so its runs here search worlds with none."""
    return SealedWorld(_stack(*[_points(hz, exists, 1)] * n_star))


def _design(budget_d, search=SEARCH, latency_d=0.0):
    return Design((search,), (Mode(0, latency_d),), Resources(budget_d), CAPACITY)


def _menu(n_star, exposure_d=(1.0,)):
    exposure_d = jnp.broadcast_to(jnp.asarray(exposure_d), (n_star, len(exposure_d)))
    n_entry = exposure_d.shape[1]
    return Menu(exposure_d, jnp.ones(exposure_d.shape, bool), (0,) * n_entry)


def _policy(stars, design, menu, settings):
    return PricePolicy(
        stars.refits, stars.pool, stars.stars, stars.roles, CLAIM, Count(stars.pool),
        design.models, mode_models(design), menu, settings,
    )  # fmt: skip


def _public(design):
    """The public view after an empty grant: time zero, the whole budget."""
    grant = _no_grant()
    payload = _world(1).reveal(run_keys(KEY)[1], grant, design.models, design.modes)
    log = start_log(grant_rows(grant, payload, design.modes), design.capacity)
    return public_view(log, jnp.asarray(0), design.resources)


@eqx.filter_jit
def _rates(pool, refit, log, models, time_d):
    """Each star's forecast claim-true pool rate at time zero and after the searches
    released by `time_d` (`replay_pool`)."""

    def one(pool, refit):
        now = replay_pool(pool, refit, log, models, time_d)
        return pool_rate(pool, CLAIM, STAR)[0], pool_rate(now, CLAIM, STAR)[0]

    return jax.vmap(one)(pool, refit)


def _found(stars, log, models, time_d):
    """Expected claim-true planets found by a run's searches over every star: the
    forecast's claim-true pool rate at time zero less its rate after them."""
    before, after = _rates(stars.pool, stars.refits, log, models, time_d)
    return float(jnp.sum(before - after))


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
    settings = PlannerSettings(0.0, math.inf, 1, forecast_updates=updates)
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

    Two stars (mass 2 each), gate `min_n_eff = 3`: star 0's four draws are equally
    weighted (Kish size 4) and star 1's hold one draw of weight (Kish size 1), so
    star 1's forecast and verdict rows are NaN, and so is every option it offers.
    Every state marks star 1 refused; the planner prices star 0 as if star 1 were
    absent and searches only it (0.8, then 0.16) within the budget of 2 days.

    Fails if a refused star's NaN reads as worth zero (it would not be reported) or
    stops the price (nothing would be bought at star 0), or if a refused star is
    searched.
    """
    weights = jnp.array([[0.0] * 4, [0.0, -jnp.inf, -jnp.inf, -jnp.inf]])
    stars = _stars([2.0, 2.0], log_weights=weights)
    design = _design(2.0)
    settings = PlannerSettings(3.0, math.inf, 1)
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
            r, r, roles, (SEARCH,), STAR, pool, pool, CLAIM, 4,
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
            r, r, stars.roles, (search,), STAR, pool, pool, CLAIM, 4, **UNGATED
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
            forecast, verdict, CLAIM, stars.stars, Count(stars.pool), (REVISIT,), menu,
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
    settings = PlannerSettings(0.0, math.inf, 2)
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
    its searches at another star, traces neither `init` nor `decide` again.

    Fails if a budget, a prior value, a mass or a threshold is a Python value or a
    static field, or a menu, star or row index is read on the host.
    """
    other = eqx.tree_at(lambda p: p.p_size, PRIOR, jnp.asarray(0.3))
    runs = (
        (_stars([2.0, 1.0]), 3.0, _world(2, hz=1.0)),
        (_stars([0.5, 3.0], prior=other), 2.0, _world(2, hz=0.0)),
    )
    counts, starts = [], []
    for stars, budget_d, world in runs:
        design = _design(budget_d)
        policy = Counted(_policy(stars, design, _menu(2), DEPTH_1))
        log = execute(KEY, _no_grant(), world, design, policy)
        counts.append((len(INIT_TRACES), len(DECIDE_TRACES)))
        starts.append(int(log.rows.star[0]))
    assert counts[1] == counts[0]
    assert starts == [0, 1]
