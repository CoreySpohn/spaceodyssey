"""Value and spend: rewards from beliefs, and the value of hand-built frontiers; closed
forms and the sign errors they catch.
"""

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from jax.scipy.stats import norm
from numpy.testing import assert_allclose

from spaceodyssey.belief import (
    Belief,
    count_pmf,
    planning_beliefs,
    q,
    replay,
    select_roles,
)
from spaceodyssey.claim import Claim, GasAbove
from spaceodyssey.information import cell_probs, outcome_cells, update_by_cell
from spaceodyssey.price import dispatch, solve_price
from spaceodyssey.roles import Roles
from spaceodyssey.toy import GaussianGasModel, StepReportModel
from spaceodyssey.types import (
    Action,
    Atoms,
    ObservationLog,
    Refit,
    Row,
    WeightedCloud,
)
from spaceodyssey.value import (
    Count,
    Frontier,
    LogScore,
    Menu,
    frontier,
    value_and_spend,
)

# Four stars, menus padded to three options (reward, cost in days).
# Star 0: three options with exact kinks at 0.5 (options 0 and 1 tie) and 1.0 (option 0
# ties with the null). Star 1: a dominated option and a padded entry that would win if
# the mask were ignored. Star 2: a losing option and an equal-cost duplicate. Star 3:
# no options at all.
REWARD = [[1.0, 1.5, 1.75], [0.5, 0.5, 9.0], [-0.2, 0.3, 0.3], [0.0, 0.0, 0.0]]
COST_D = [[1.0, 2.0, 3.0], [0.5, 1.0, 0.1], [1.0, 0.2, 0.2], [1.0, 1.0, 1.0]]
MASK = [[True] * 3, [True, True, False], [True] * 3, [False] * 3]

batched = jax.jit(jax.vmap(value_and_spend, in_axes=(0, None)))


def _toy():
    return Frontier(jnp.asarray(REWARD), jnp.asarray(COST_D), jnp.asarray(MASK))


def _with_null(f):
    """NumPy reward, cost and mask with the null option as column zero."""
    pad = lambda x, v: np.pad(np.asarray(x), ((0, 0), (1, 0)), constant_values=v)  # noqa: E731
    return pad(f.reward, 0.0), pad(f.cost_d, 0.0), pad(f.mask, True)


def _breakpoints(f):
    """Every price at which two options of one star have equal net reward."""
    reward, cost, mask = _with_null(f)
    out = []
    for s in range(reward.shape[0]):
        for m, n in itertools.combinations(np.flatnonzero(mask[s]), 2):
            if cost[s, m] != cost[s, n]:
                out.append((reward[s, m] - reward[s, n]) / (cost[s, m] - cost[s, n]))
    return np.asarray(out)


def _chosen_cost(lam, f):
    """Cost of each star's best option at price ``lam``, found by enumeration."""
    reward, cost, mask = _with_null(f)
    net = np.where(mask, reward - lam[:, None, None] * cost, -np.inf)
    cost = np.broadcast_to(cost, net.shape)
    return np.take_along_axis(cost, net.argmax(-1)[..., None], -1)[..., 0]


def test_spend_is_the_slope_and_the_cost_of_the_chosen_options():
    """Away from kinks, spend equals -dV/dlam by a centered difference and the cost of
    the option each star chooses, found by enumeration (a third path).

    Fails if the spend has the wrong sign, if a stop-gradient or a constant price cuts
    the tangent, if the price enters the reward itself, or if a padded option is read.
    """
    f = _toy()
    grid = np.linspace(0.01, 2.5, 60)
    bps = _breakpoints(f)
    lam = grid[np.min(np.abs(grid[:, None] - bps[None]), axis=1) > 1e-3]
    step = 1e-5
    _, spend = batched(jnp.asarray(lam), f)
    v_up, _ = batched(jnp.asarray(lam + step), f)
    v_down, _ = batched(jnp.asarray(lam - step), f)
    centered = -(np.asarray(v_up) - np.asarray(v_down)) / (2 * step)
    assert_allclose(spend, centered, atol=1e-8)
    assert_allclose(spend, _chosen_cost(lam, f), atol=1e-12)
    assert np.ptp(np.asarray(spend)[:, 0]) > 0  # star 0 changes option on the grid


def test_spend_at_a_kink_lies_in_the_subdifferential():
    """At an exact kink, each star's spend lies between its one-sided spends.

    At lam = 0.5 star 0's first two options tie (spend in [1, 2]); at lam = 1.0 its
    first option ties with the null (spend in [0, 1]). The one-sided spends are read
    just either side of the kink.

    Fails if the spend at a tie leaves the subdifferential, for example by summing the
    tied costs instead of choosing among them.
    """
    f = _toy()
    lam = np.array([0.5, 1.0])
    _, spend = batched(jnp.asarray(lam), f)
    _, right = batched(jnp.asarray(lam + 1e-6), f)
    _, left = batched(jnp.asarray(lam - 1e-6), f)
    assert_allclose(right[:, 0], [1.0, 0.0], atol=1e-12)
    assert_allclose(left[:, 0], [2.0, 1.0], atol=1e-12)
    assert np.all(np.asarray(right) <= np.asarray(spend) + 1e-12)
    assert np.all(np.asarray(spend) <= np.asarray(left) + 1e-12)


@given(
    reward=st.lists(st.floats(-1.0, 2.0), min_size=12, max_size=12),
    cost=st.lists(st.floats(0.0, 3.0), min_size=12, max_size=12),
    mask=st.lists(st.booleans(), min_size=12, max_size=12),
)
def test_value_is_convex_and_nonincreasing_in_the_price(reward, cost, mask):
    """V_i(lam) is convex and nonincreasing on lam in [0, 3] for any frontier.

    Fails if the price enters with the wrong sign (r + lam c makes V increase), or if a
    star's options are summed or averaged instead of maximized (convexity breaks at a
    switch).
    """
    f = Frontier(
        jnp.reshape(jnp.asarray(reward), (4, 3)),
        jnp.reshape(jnp.asarray(cost), (4, 3)),
        jnp.reshape(jnp.asarray(mask), (4, 3)),
    )
    values = np.asarray(batched(jnp.linspace(0.0, 3.0, 121), f)[0])
    assert np.all(np.diff(values, axis=0) <= 1e-12)
    assert np.all(np.diff(values, 2, axis=0) >= -1e-12)


# Rewards from beliefs. The planner reads two beliefs, the forecast's and one verdict
# member's; the fixtures build one two-role belief (forecast row 0, verdict row 1) and
# `toy.split` it. `frontier` runs on stacked stars.
traces = []
PLANET = jnp.asarray(0)


@eqx.filter_jit
def _traced_frontier(beliefs, claim, star, utility, models, menu, lam, depth):
    traces.append(None)
    return frontier(*beliefs, claim, star, utility, models, menu, lam, depth)


def _frontier_of(beliefs, claim, star, utility, models, menu, depth, lam=0.0):
    """`frontier` under one jit on the stacked (forecast, verdict) `beliefs`, with the
    price an array so it is traced.
    """
    lam = jnp.asarray(lam, float)
    return _traced_frontier(beliefs, claim, star, utility, models, menu, lam, depth)


def _menu(exposure_d, mode, mask=True):
    """A menu over stars: `exposure_d` has shape (S, M), `mode` one mode per entry."""
    exposure_d = jnp.asarray(exposure_d, float)
    mask = jnp.broadcast_to(jnp.asarray(mask, bool), exposure_d.shape)
    return Menu(exposure_d, mask, mode)


# The two spectra (V; T): water and oxygen reports at S/N 6 (above the step gate), each
# alone (modes 0 and 1) and both in one action (mode 2). The menu lists both first, so
# its entries are not in mode order.
SPECTRA = (
    StepReportModel(snr_one_day=[6.0, 0.0]),
    StepReportModel(snr_one_day=[0.0, 6.0]),
    StepReportModel(snr_one_day=[6.0, 6.0]),
)
SPECTRA_MENU = _menu([[1.0] * 3], (2, 0, 1))


def _spectra_belief(toy, psi):
    """Eight point draws over (candidate, water, oxygen), each gas 1 dex above or below
    its floor. Forecast: both gases certainly present, candidate with probability
    `psi`. Verdict: candidate with `psi`, each gas at even chance.
    """
    states = list(itertools.product((1.0, 0.0), (1.0, -1.0), (1.0, -1.0)))
    hz = [[h] for h, _, _ in states]
    gas = [[w, o] for _, w, o in states]
    atoms = toy.atoms(hz, np.ones_like(hz), np.ones(8), gas, 0.0, n_gas=2)
    cand = np.array([psi if h else 1.0 - psi for h, _, _ in states])
    present = np.array([w > 0 and o > 0 for _, w, o in states])
    log_prior = [np.where(present, np.log(cand), -np.inf), np.log(cand / 4.0)]
    return Belief(atoms, log_prior, np.zeros((2, 8)), toy.empty_pool(2, 2))


def _spectra_claim(toy):
    return Claim((toy.InHz(), GasAbove(0, 0.0), GasAbove(1, 0.0)))


@pytest.mark.parametrize(("psi", "memo"), [(0.70, 0.67), (0.30, 0.22)])
def test_two_spectra_are_worth_the_verdict_gain(toy, psi, memo):
    """Spec layer 5 test 1 (V; T). The forecast is certain both gases are present, so
    the outcome tells it nothing about the claim: its mutual information I_F is 0. The
    verdict gives each gas even odds, so the same outcome is worth
    psi ln 4 + (1 - psi) [ln(1 - psi) - ln(1 - psi / 4)] to it: 0.970 - 0.304 = 0.667
    at psi = 0.70 and 0.416 - 0.195 = 0.221 at psi = 0.30 (V: 0.67 and 0.22). The
    frontier's reward at depth 1 (count of one planet, empty pool) and the claim-bit
    information agree on it.

    Fails if the reward is the forecast's mutual information instead of the verdict
    gain (it would be 0), if the verdict's prior term uses the forecast, or if the two
    paths to the verdict gain disagree.
    """
    gain = psi * np.log(4.0) + (1.0 - psi) * (np.log(1.0 - psi) - np.log(1 - psi / 4))
    b = _spectra_belief(toy, psi)
    claim = _spectra_claim(toy)
    forecast, verdict = toy.split(b)
    action = Action(0, 0, 1.0)
    info = [
        toy.jit.claim_information(
            forecast, b, SPECTRA[2], action, claim, toy.star, PLANET
        )
        for b in (forecast, verdict)
    ]
    assert_allclose(np.concatenate(info), [0.0, gain], atol=1e-12)  # I_F, the gain
    f = _frontier_of(
        toy.split(toy.stack(b)), claim, toy.star, LogScore(1), SPECTRA, SPECTRA_MENU, 1
    )
    assert_allclose(f.reward[0, 0], gain, rtol=1e-12)
    assert round(float(f.reward[0, 0]), 2) == memo


def _coarsening_belief(toy):
    """Four point draws over (Z, gas): Z is the `hz` bit, the gas bit is the outcome.

    Forecast F: P(Z = 1) = 0.5, the gas bit Y even, F(Z = 1 | Y = 1) = 0.9 and
    F(Z = 1 | Y = 0) = 0.1. Verdict gamma reversed: gamma(Z = 1 | Y = 1) = 0.1 and
    gamma(Z = 1 | Y = 0) = 0.9, gamma(Z = 1) = 0.5.
    """
    hz = [[1.0], [0.0], [1.0], [0.0]]
    gas = [[1.0], [1.0], [-1.0], [-1.0]]
    atoms = toy.atoms(hz, np.ones_like(hz), np.ones(4), gas, 0.0, n_gas=1)
    log_prior = np.log([[0.45, 0.05, 0.05, 0.45], [0.05, 0.45, 0.45, 0.05]])
    return Belief(atoms, log_prior, np.zeros((2, 4)), toy.empty_pool(2, 1))


# One gas report at S/N 6 (fine: the bit is reported) or 3 (merged: nothing reported).
GAS_REPORT = (StepReportModel(snr_one_day=[6.0]),)
FINE_AND_MERGED = _menu([[1.0, 0.25]], (0, 0))


def test_coarsening_can_raise_the_verdict_gain(toy):
    """Spec layer 5 test 2 (C; T). With the verdict reversed against the forecast the
    fine outcome scores 0.9 ln 0.1 + 0.1 ln 0.9 = -2.0828 per cell; less
    E ln gamma(Z) = ln 0.5 the reward is -2.0828 + 0.6931 = -1.390 nats. Merging the
    two outcomes (below the report's gate) leaves gamma(Z | cell) = 0.5 and reward 0:
    coarsening raised the reward by 1.39 while the forecast's mutual information fell
    from ln 2 - h(0.9) = 0.6931 - 0.3251 = 0.368 to 0.

    Fails if the verdict's log score after an outcome is read off the forecast's
    posterior (the fine reward would be +0.368), if the verdict is not updated by the
    outcome (the fine reward would be 0), or if a bound is drawn from merged cells.
    """
    b = _coarsening_belief(toy)
    forecast, verdict = toy.split(b)
    claim = Claim((toy.InHz(),))
    f = _frontier_of(
        toy.split(toy.stack(b)),
        claim,
        toy.star,
        LogScore(1),
        GAS_REPORT,
        FINE_AND_MERGED,
        1,
    )
    assert_allclose(f.reward[0], [-1.390, 0.0], atol=5e-4)
    exact = 0.9 * np.log(0.1) + 0.1 * np.log(0.9) - np.log(0.5)
    assert_allclose(f.reward[0, 0], exact, rtol=1e-12)
    mutual = np.log(2.0) + 0.9 * np.log(0.9) + 0.1 * np.log(0.1)  # ln 2 - h(0.9)
    assert round(mutual, 3) == 0.368
    for exposure_d, row in ((1.0, [mutual, exact]), (0.25, [0.0, 0.0])):
        action = Action(0, 0, exposure_d)
        info = [
            toy.jit.claim_information(
                forecast, v, GAS_REPORT[0], action, claim, toy.star, PLANET
            )
            for v in (forecast, verdict)
        ]
        assert_allclose(np.concatenate(info), row, atol=1e-12)  # I_F, the reward


def test_value_does_not_fall_with_depth(toy):
    """Spec layer 5 test 4: the reward of every entry is nondecreasing in depth.

    On the two spectra (psi = 0.7) a water report alone is worth
    psi ln 2 + (1 - psi) [ln(1 - psi / 2) - ln(1 - psi / 4)] = 0.414 at depth 1; at
    depth 2 the oxygen report may follow, so it is worth the whole 0.667. On the
    coarsening fixture a noisy reading (Gaussian, S/N 2) loses under the reversed
    verdict, and so would every repeat of it: a deeper plan stops after the first, so
    depths 1 and 2 agree (a plan forced on falls to -0.61 at depth 2).

    Fails if a deeper plan cannot stop (it would be forced into a second losing
    reading), if the continuation is averaged instead of maximized, or if a follow-up
    does not see the earlier outcome (the oxygen report would be valued from the
    prior).
    """
    psi = 0.7
    alone = psi * np.log(2.0) + (1 - psi) * (np.log(1 - psi / 2) - np.log(1 - psi / 4))
    both = psi * np.log(4.0) + (1 - psi) * (np.log(1 - psi) - np.log(1 - psi / 4))
    b, claim = toy.split(toy.stack(_spectra_belief(toy, psi))), _spectra_claim(toy)
    rewards = np.stack(
        [
            _frontier_of(
                b, claim, toy.star, LogScore(1), SPECTRA, SPECTRA_MENU, d
            ).reward[0]
            for d in (1, 2)
        ]
    )
    assert_allclose(rewards[0], [both, alone, alone], rtol=1e-12)
    assert_allclose(rewards[1], both, rtol=1e-12)
    assert np.all(np.diff(rewards, axis=0) >= -1e-12)

    reading = (GaussianGasModel(snr_one_day=[2.0], edges=[0.0]),)
    losing = np.stack(
        [
            _frontier_of(
                toy.split(toy.stack(_coarsening_belief(toy))),
                Claim((toy.InHz(),)),
                toy.star,
                LogScore(1),
                reading,
                _menu([[1.0]], (0,)),
                d,
            ).reward[0, 0]
            for d in (1, 2)
        ]
    )
    assert losing[0] < 0.0
    assert_allclose(losing[1], losing[0], rtol=1e-12)


def test_a_frozen_forecast_plans_on_its_own_replay(toy):
    """The planner reads the forecast and the verdict each off its own replay, so a
    frozen forecast does not see the rows released after time zero.

    One draw (one planet), water N(0, 1) dex; one Gaussian reading y = 1 at S/N 5
    (sd 0.5), released at day 1. The verdict's replay at day 1 holds it: water
    N(0.8, 0.2), so P(water) = Phi(0.8 / sqrt(0.2)) = 0.9632. The frozen forecast is
    the replay at day 0: water N(0, 1), P(water) = 0.5. `LogScore` on the count of
    water planets is then 0.5 ln(1 - 0.9632) + 0.5 ln 0.9632 = -1.670, against the
    verdict's own -0.157 had the forecast been read off the verdict's replay.

    Fails if the forecast is read off the verdict's replay, or shares its draws: a
    frozen forecast would then see the reading released at day 1; or if it reads the
    pool now instead of the pool at time zero.
    """
    model = GaussianGasModel(snr_one_day=[5.0])
    models, capacity = (model,), 2
    atoms = toy.atoms([[1.0]], [[1.0]], [1.0], 0.0, 1.0, n_gas=1)
    zero_i, zero_d = jnp.zeros(capacity, int), jnp.zeros(capacity)
    payload = {"snr": jnp.full((capacity, 1, 1), 5.0), "reading": jnp.ones((2, 1, 1))}
    release_d = jnp.array([1.0, 0.0])
    rows = Row(
        zero_i, zero_i, zero_d, zero_d + 1.0, zero_d, release_d, zero_i, zero_i,
        (payload,),
    )  # fmt: skip
    log = ObservationLog(rows, 1)
    refit = Refit(atoms, jnp.zeros(capacity, bool), 0)
    prior = toy.GasPrior("toy", n_gas=1)
    roles = Roles(prior, toy.stack(prior))
    frozen = toy.jit.replay(refit, log, models, toy.star, jnp.asarray(0.0))
    now = toy.jit.replay(refit, log, models, toy.star, jnp.asarray(1.0))
    pool0, pool = toy.empty_pool(2, 1), toy.empty_pool(2, 1)
    pool = eqx.tree_at(lambda p: p.log_mass, pool, jnp.array([-1.0, -2.0]))
    claim = Claim((GasAbove(0, 0.0),))
    forecast, verdict = toy.jit.planning_beliefs(
        frozen, now, roles, models, toy.star, pool0, pool, claim, **toy.ungated
    )
    assert_allclose(forecast.pool.log_mass, [-np.inf])  # the pool at time zero
    assert_allclose(verdict.pool.log_mass, [-2.0])  # the verdict's row of the pool now

    p_water = float(norm.cdf(0.8 / np.sqrt(0.2)))
    for b, mean in ((forecast, 0.0), (verdict, 0.8)):  # on quadrature nodes
        nodes = b.atoms.nodes
        omega = np.asarray(nodes.weights)[0, 0, 0]
        # the mean to quadrature error; q below exact
        assert_allclose(omega @ np.asarray(nodes.x)[0, 0, 0], mean, atol=5e-3)
    assert_allclose(q(forecast, claim, toy.star, PLANET), [0.5], rtol=1e-12)
    assert_allclose(q(verdict, claim, toy.star, PLANET), [p_water], rtol=1e-12)
    expected = 0.5 * np.log(1.0 - p_water) + 0.5 * np.log(p_water)
    utility = LogScore(1)(forecast, verdict, claim, toy.star)
    assert_allclose(utility, expected, rtol=1e-12)
    assert round(expected, 3) == -1.670


def _utility_of_bit(q_bit):
    """`LogScore` of one planet's claim bit when forecast and verdict agree."""
    q_bit = np.clip(q_bit, 1e-300, 1.0 - 1e-16)
    return q_bit * np.log(q_bit) + (1.0 - q_bit) * np.log1p(-q_bit)


def _binned_reading_oracle(m, v, sigma, depth):
    """Lookahead value of a binned gas reading (one edge at 0) of one draw whose gas
    is N(m, v), under `LogScore` on x > 0, forecast = verdict, lam = 0: dense
    quadrature over x (40,001 points over 12 sd), the bins of the readings independent
    given x. Also the bound of `depth` continuous readings, the value of observing
    them instead of their bins (data processing).
    """
    sd = np.sqrt(v)
    x = np.linspace(m - 12.0 * sd, m + 12.0 * sd, 40001)
    p = np.asarray(norm.pdf(x, m, sd))
    p, above = p / p.sum(), x > 0.0
    below_edge = np.asarray(norm.cdf(-x / sigma))
    bins = np.stack([below_edge, 1.0 - below_edge], axis=1)  # P(bin | x)

    def worth(w, k):
        now = _utility_of_bit(np.sum(w * above) / np.sum(w))
        if k == 0:
            return now
        later = sum(np.sum(w * b) / np.sum(w) * worth(w * b, k - 1) for b in bins.T)
        return max(now, later)

    exact = sum(np.sum(p * b) * worth(p * b, depth - 1) for b in bins.T)
    exact -= _utility_of_bit(np.sum(p * above))
    s2 = sigma**2 / depth  # `depth` readings: their mean, noise variance s2
    sy = np.sqrt(v + s2)
    y = np.linspace(m - 12.0 * sy, m + 12.0 * sy, 40001)
    py = np.asarray(norm.pdf(y, m, sy))
    post = m + v / (v + s2) * (y - m), v * s2 / (v + s2)
    q_y = np.asarray(norm.cdf(post[0] / np.sqrt(post[1])))
    bound = np.sum(py * _utility_of_bit(q_y)) / np.sum(py)
    bound -= _utility_of_bit(float(norm.cdf(m / sd)))
    return exact, bound


# The gas-reading stars: (prior mean, prior variance, replay time, exposure). A reading
# y = 1 at S/N 5 released at day 1 turns N(0, 1) into N(0.8, 0.2); an exposure of 0.16
# days reads at S/N 2, one of 4 days at S/N 10. The last four are near certain: N(m, 1)
# with P(x > 0) = 0.9996 and 0.999996, each read at S/N 5 and 10.
NEAR_CERTAIN = tuple(float(norm.ppf(q0)) for q0 in (0.9996, 0.999996))
GAS_STARS = (
    (0.0, 1.0, 0.0, 1.0), (0.0, 1.0, 1.0, 1.0), (0.3, 0.1, 0.0, 1.0),
    (0.0, 1.0, 1.0, 0.16),
    *((m, 1.0, 0.0, e) for m in NEAR_CERTAIN for e in (1.0, 4.0)),
)  # fmt: skip
GAS_BLOCKS = (
    (0.0, 1.0), (0.8, 0.2), (0.3, 0.1), (0.8, 0.2),
    *((m, 1.0) for m in NEAR_CERTAIN for _ in range(2)),
)  # fmt: skip


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, None, 0, None, None, None, None))
def _gas_planning_beliefs(refit, log, time_d, roles, pool, claim, n):
    models = (GaussianGasModel(snr_one_day=[5.0]),)
    now = replay(refit, log, models, (), time_d)
    return planning_beliefs(
        now,
        now,
        roles,
        models,
        (),
        pool,
        pool,
        claim,
        n,
        min_n_eff=0.0,
        max_khat=jnp.inf,
    )


def test_a_binned_gas_reading_is_valued_exactly(toy):
    """The frontier's value of a binned reading of an uncertain gas matches a dense
    quadrature over the abundance, at depth 1 and 2, and never exceeds the value of
    the continuous reading.

    Eight one-draw stars, built through `replay` and `planning_beliefs` (24 nodes per
    factor): water N(0, 1); N(0.8, 0.2) after a reading y = 1 at S/N 5; a narrow
    N(0.3, 0.1); N(0.8, 0.2) read at S/N 2; and N(m, 1) with P(x > 0) = 0.9996 and
    0.999996, each read at S/N 5 and 10 (values 1.6e-3 to 2.4e-5 at depth 1, within
    0.6 percent of the oracle). One `GaussianGasModel` entry, one edge
    at the floor; `LogScore` on the count of water planets, forecast = verdict,
    lam = 0. The oracle (`_binned_reading_oracle`) at the prior, depth 1, is the
    closed form U(q) - ln 0.5 with q = 2 P(x > 0, y > 0) = 1/2 + arcsin(1 /
    sqrt(1.25)) / pi = 0.8524: 0.2747 (0.3778 at depth 2), against a continuous bound
    of 0.3733; the others are 0.0293, 0.0462 and 0.0068 at depth 1.

    Fails if a hypothetical bin conditions the block on one representative reading
    instead of splitting it (the value then runs +46 to +187 percent above the oracle at
    S/N 5, above the continuous bound, and -97 percent at S/N 2), if the split is
    skipped (an uncertain block is refused, and reweighting it alone values it near
    zero), if the split moves mass across the floor, or if the dominant side of a
    near-certain block has no nodes graded toward the floor (one uniform block per
    side misses the near-certain values by up to 33 percent).
    """
    log_of = jnp.zeros(2, int), jnp.zeros(2)
    payload = {"snr": jnp.full((2, 1, 1), 5.0), "reading": jnp.ones((2, 1, 1))}
    rows = Row(
        log_of[0], log_of[0], log_of[1], log_of[1] + 1.0, log_of[1],
        jnp.array([1.0, 0.0]), log_of[0], log_of[0], (payload,),
    )  # fmt: skip
    log = ObservationLog(rows, 1)
    refits = toy.stack(
        *[
            Refit(
                toy.atoms([[1.0]], [[1.0]], [1.0], m, v, n_gas=1), jnp.zeros(2, bool), 0
            )
            for m, v, _, _ in GAS_STARS
        ]
    )
    prior = toy.GasPrior("toy", n_gas=1)
    times = jnp.asarray([t for _, _, t, _ in GAS_STARS])
    claim = Claim((GasAbove(0, 0.0),))
    roles = Roles(prior, toy.stack(prior))
    beliefs = _gas_planning_beliefs(
        refits, log, times, roles, toy.empty_pool(2, 1), claim, 24
    )
    models = (GaussianGasModel(snr_one_day=[5.0]),)
    exposure_d = np.array([[e] for _, _, _, e in GAS_STARS])
    menu = _menu(exposure_d, (0,))
    sigma = 2.5 / (5.0 * np.sqrt(exposure_d[:, 0]))

    q_bit = 0.5 + np.arcsin(1.0 / np.sqrt(1.25)) / np.pi
    closed = _utility_of_bit(q_bit) - np.log(0.5)
    for depth, rounded in ((1, [0.2747, 0.0293, 0.0462, 0.0068]), (2, [0.3778])):
        f = _frontier_of(beliefs, claim, toy.star, LogScore(1), models, menu, depth)
        oracle = [
            _binned_reading_oracle(m, v, s, depth)
            for (m, v), s in zip(GAS_BLOCKS, sigma, strict=True)
        ]
        exact, bound = np.asarray(oracle).T
        assert np.round(exact[: len(rounded)], 4).tolist() == rounded
        assert_allclose(f.reward[:, 0], exact, rtol=0.01)
        assert np.all(np.asarray(f.reward[:, 0]) <= bound)
    assert_allclose(_binned_reading_oracle(0.0, 1.0, 0.5, 1)[0], closed, rtol=1e-5)
    assert round(float(_binned_reading_oracle(0.0, 1.0, 0.5, 1)[1]), 4) == 0.3733


@pytest.mark.medium(reason="compiles the planning beliefs at three node counts, 5 s")
def test_the_quadrature_converges_in_its_nodes_at_high_snr(toy):
    """At S/N 50 the binned reading's value converges in the number of nodes.

    One draw, water N(0, 1) (claim probability 0.5), one `GaussianGasModel` reading
    binned at the floor at S/N 50 (100 days at S/N 5 per day), `LogScore`, depth 1,
    against the dense oracle (`_binned_reading_oracle`, 0.6115 nats). The reading
    resolves structure within about 0.4 sd of the floor, so the error depends on the
    nodes there: 16 nodes (two near the floor on each side) overvalue it by 5.4
    percent, 24 (three) are within 0.6 percent and 48 (six) within 0.02 percent.

    Fails if the nodes near the floor stop growing with `n` (the error at 48 nodes
    would stay at 5 percent), or if the graded quadrature misplaces them.
    """
    z, zd = jnp.zeros(2, int), jnp.zeros(2)
    payload = {"snr": jnp.full((2, 1, 1), 5.0), "reading": jnp.zeros((2, 1, 1))}
    rows = Row(z, z, zd, zd + 1.0, zd, zd, z, z, (payload,))
    log = ObservationLog(rows, 0)
    refit = toy.stack(Refit(toy.atoms([[1.0]], [[1.0]], [1.0], 0.0, 1.0, n_gas=1),
                            jnp.zeros(2, bool), 0))  # fmt: skip
    prior = toy.GasPrior("toy", n_gas=1)
    roles, claim = Roles(prior, toy.stack(prior)), Claim((GasAbove(0, 0.0),))
    models = (GaussianGasModel(snr_one_day=[5.0]),)
    exact = _binned_reading_oracle(0.0, 1.0, 2.5 / 50.0, 1)[0]
    assert round(float(exact), 4) == 0.6115
    errors = []
    for n in (16, 24, 48):
        beliefs = _gas_planning_beliefs(
            refit, log, jnp.zeros(1), roles, toy.empty_pool(2, 1), claim, n
        )
        f = _frontier_of(
            beliefs, claim, toy.star, LogScore(1), models, _menu([[100.0]], (0,)), 1
        )
        errors.append(float(f.reward[0, 0]) / exact - 1.0)
    assert errors[0] > 0.04
    assert abs(errors[1]) < 6e-3 and abs(errors[2]) < 2e-4


class MeanReader(GaussianGasModel):
    """A gas reading without `gas_cells`: its `cells` read the block's mean."""

    def gas_cells(self, atoms, action):
        return None

    def cells(self, atoms, action):
        above = norm.cdf(atoms.x_mean[:, 0, 0] / 0.5)
        return jnp.stack([1.0 - above, above], axis=-1)


def test_a_model_without_gas_cells_cannot_read_a_belief_on_nodes(toy):
    """On a belief on quadrature nodes, a model that reads the gas block through
    `cells` alone is refused, never valued at the block the nodes were built from.

    One draw, water N(0.3, 1) put on 4 nodes split at the floor; a reading whose
    `cells` read `x_mean` (no `gas_cells`) raises there, under `jit`; on a point
    block (variance 0) the same model is read as it is, and a search, which reads no
    gas, is read on the nodes as on the points.

    Fails if a model without `gas_cells` is handed the stale mean of a node belief
    (it would be valued at that point, 46 to 187 percent off), or if the refusal also
    stops a model that reads no gas.
    """
    claim, action = Claim((GasAbove(0, 0.0),)), Action(0, 0, 1.0)
    reader = MeanReader(snr_one_day=[2.0], edges=[0.0])
    cells = eqx.filter_jit(outcome_cells)
    point = toy.belief(toy.atoms([[1.0]], [[1.0]], [1.0], 0.3, 0.0, n_gas=1))
    assert_allclose(cells(point, reader, action).draw[0, 1], norm.cdf(0.6), rtol=1e-12)
    gauss = toy.belief(toy.atoms([[1.0]], [[1.0]], [1.0], 0.3, 1.0, n_gas=1))
    on_nodes = toy.jit.quadrature_draws(gauss, claim, 4)
    with pytest.raises(eqx.EquinoxRuntimeError, match="without gas_cells"):
        cells(on_nodes, reader, action)
    search = toy.Search(hit=0.6, other=0.1)
    assert_allclose(
        cells(on_nodes, search, action).draw, cells(point, search, action).draw
    )


def _product_grid(b):
    """The belief on the product grid of its factors' nodes: one point draw per draw
    and choice of a node in every factor, weighted by the draw's weight times the
    chosen nodes' weights. The reference the per-factor nodes stand for.
    """
    nodes = jax.tree.map(np.asarray, b.atoms.nodes)
    n_draw, n_planet, n_gas, n_node = nodes.x.shape
    n_factor = n_planet * n_gas
    codes = np.array(list(itertools.product(range(n_node), repeat=n_factor)))
    factor = np.arange(n_factor)
    x = nodes.x.reshape(n_draw, n_factor, n_node)[:, factor, codes]  # (A, C, F)
    omega = nodes.weights.reshape(n_draw, n_factor, n_node)[:, factor, codes]
    n_sub = codes.shape[0]

    def repeat(leaf):
        return np.repeat(np.asarray(leaf), n_sub, axis=0)

    cloud = b.atoms.cloud
    log_w = np.asarray(cloud.log_weights)[:, None] + np.log(omega).sum(-1)
    cloud = WeightedCloud(
        jax.tree.map(repeat, cloud.samples),
        log_w.reshape(-1),
        repeat(cloud.log_target),
        cloud.log_evidence,
        cloud.prior,
    )
    x = x.reshape(n_draw * n_sub, n_planet, n_gas)
    atoms = Atoms(cloud, x, np.zeros_like(x), repeat(b.atoms.exists))
    return Belief(
        atoms,
        np.repeat(b.log_prior_ratio, n_sub, axis=-1),
        np.repeat(b.log_like, n_sub, axis=-1),
        b.pool,
    )


def _two_planet_two_gas_belief(toy, n_draw, seed=0):
    """`n_draw` draws of two planets with two uncertain gases each, two roles."""
    rng = np.random.default_rng(seed)
    hz = rng.uniform(0.2, 1.0, (n_draw, 2))
    mean = rng.normal(0.0, 0.7, (n_draw, 2, 2))
    var = rng.uniform(0.2, 1.5, (n_draw, 2, 2))
    atoms = toy.atoms(hz, np.ones_like(hz), np.ones(n_draw), n_gas=2)
    block = (jnp.asarray(mean), jnp.asarray(var))
    atoms = eqx.tree_at(lambda a: (a.x_mean, a.x_var), atoms, block)
    log_prior = rng.normal(0.0, 0.5, (2, n_draw))
    return Belief(atoms, log_prior, np.zeros((2, n_draw)), toy.empty_pool(2, 2))


TWO_GASES = (
    GaussianGasModel(snr_one_day=[3.0, 2.0]),
    StepReportModel(snr_one_day=[6.0, 6.0]),
)


@eqx.filter_jit
def _two_readings(b, models, claim, star):
    """After a reading by each of two models: every cell pair's log probability per
    role, and the claim probability of each planet and the count pmf after it."""
    action = Action(0, 0, 1.0)
    first, second = (outcome_cells(b, m, action) for m in models)

    def after(y1, y2):
        out = update_by_cell(update_by_cell(b, first, y1), second, y2)
        log_mass = jax.nn.logsumexp(
            out.atoms.cloud.log_weights + out.log_prior_ratio + out.log_like, axis=-1
        )
        planets = jnp.arange(out.atoms.exists.shape[-1])
        q_each = jax.vmap(lambda k: q(out, claim, star, k))(planets)
        return log_mass, q_each, count_pmf(out, claim, star, 3)

    y1, y2 = jnp.arange(first.draw.shape[-1]), jnp.arange(second.draw.shape[-1])
    return jax.vmap(lambda a: jax.vmap(lambda c: after(a, c))(y2))(y1)


@eqx.filter_jit
def _factor_and_joint_cells(b, models):
    """Each model's cell probabilities through its factors' cells and its own joint."""
    action = Action(0, 0, 1.0)
    return [
        (cell_probs(b, outcome_cells(b, m, action)), m.cells(b.atoms, action))
        for m in models
    ]


def test_per_factor_nodes_equal_the_product_grid(toy):
    """With two planets and two uncertain gases, the per-factor nodes give the
    probability of every pair of outcomes, the claim probabilities and count pmf after
    it, and the claim-bit information of the product grid of the same nodes, to
    rounding; on point draws the factors' cells are the model's own joint cells.

    Three draws, four factors of two nodes each (one on each side of the floor), so
    the product grid holds 3 x 2**4 = 48 point draws; the claim is (HZ, water,
    oxygen), two roles. A Gaussian reading of both gases (16 cells) and then a step
    report of both (81 cells, three bins per factor), the second updating the nodes the
    first moved; the claim-bit information of the Gaussian reading.

    Fails if an outcome updates a factor's nodes without the draw's weight (or the
    draw's weight without the nodes), mixes the bins of two factors, carries the
    nodes' weights from one update to the next wrongly, codes the cells in another
    order than the model's joint, or if `GasAbove` or `count_pmf` read a planet's gas
    off the block's mean instead of its nodes.
    """
    claim = Claim((toy.InHz(), GasAbove(0, 0.0), GasAbove(1, 0.0)))
    b = toy.jit.quadrature_draws(
        _two_planet_two_gas_belief(toy, 3), claim, 2, TWO_GASES
    )
    grid = _product_grid(b)
    assert grid.atoms.x_mean.shape[0] == 48
    per_factor, product = (
        _two_readings(x, TWO_GASES, claim, toy.star) for x in (b, grid)
    )
    for got, want in zip(per_factor, product, strict=True):
        assert_allclose(got, want, rtol=1e-10, atol=1e-14)
    for factors, joint in _factor_and_joint_cells(grid, TWO_GASES):
        assert_allclose(factors, joint, rtol=1e-12, atol=1e-15)
    info = [
        toy.jit.claim_information(
            *toy.split(x), TWO_GASES[0], Action(0, 0, 1.0), claim, toy.star, PLANET
        )
        for x in (b, grid)
    ]
    assert_allclose(info[0], info[1], rtol=1e-10)


@pytest.mark.medium(reason="the smoke size, 4,000 draws on 24 nodes, depth 2, 3 s")
def test_two_planets_and_two_gases_plan_at_the_smoke_size(toy):
    """Two planets with two uncertain gases each, 4,000 draws on 24 nodes per factor:
    a depth-2 frontier of a Gaussian reading of both gases runs within memory, and
    reading the gases is worth something.

    The product grid of these nodes would hold 4,000 x 24**4 = 1.3e9 draws per star.

    Fails if the quadrature forms the product grid of the factors' nodes, or if a
    lookahead level holds every grandchild belief at once (both run out of memory).
    """
    claim = Claim((toy.InHz(), GasAbove(0, 0.0), GasAbove(1, 0.0)))
    big = toy.jit.quadrature_draws(
        _two_planet_two_gas_belief(toy, 4000), claim, 24, TWO_GASES
    )
    f = _frontier_of(
        toy.split(toy.stack(big)), claim, toy.star, LogScore(2), TWO_GASES,
        _menu([[1.0]], (0,)), 2,
    )  # fmt: skip
    assert np.all(np.isfinite(f.reward)) and np.all(np.asarray(f.reward) > 0.0)


def test_a_spectrum_waits_for_a_detection(toy):
    """A spectrum is worth nothing in the lookahead before a planet is detected.

    Four equal draws in the HZ: the planet exists or not, its water 1 dex above or
    below the floor; claim (HZ, water), q = 0.25; forecast and verdict agree;
    `LogScore`, so U(q) = q ln q + (1 - q) ln(1 - q) and U(0.25) = -0.5623. A search
    finds the planet whenever it exists (cell 1, half the time), a water spectrum
    reports the bit at S/N 6 and needs a detection. At an undetected star the spectrum
    is not offered and is worth 0 at any depth; the search alone is worth
    0.5 U(0.5) + 0.5 U(0) - U(0.25) = 0.2158 at depth 1, and 0.5623 at depth 2,
    where the spectrum follows a detection and settles the claim. At a star whose
    planet is already detected the spectrum is offered and worth the same 0.2158.

    Fails if a spectrum is offered before a detection (at the root or after a miss),
    if a detection on the path does not raise the star to the stage the spectrum
    requires, or if the mask ignores the root's stage.
    """
    hz = [[1.0]] * 4
    exists = [[True], [True], [False], [False]]
    water = [[1.0], [-1.0], [1.0], [-1.0]]
    atoms = toy.atoms(hz, hz, np.ones(4), water, 0.0, n_gas=1, exists=exists)
    beliefs = toy.split(toy.stack(toy.belief(atoms, n_roles=2)))
    claim = Claim((toy.InHz(), GasAbove(0, 0.0)))
    models = (toy.Search(hit=1.0, other=1.0), StepReportModel(snr_one_day=[6.0]))

    def entropy(q):
        return q * np.log(q) + (1 - q) * np.log(1 - q)

    one_search = 0.5 * entropy(0.5) - entropy(0.25)
    assert round(one_search, 4) == 0.2158
    for detected, depth, expected in (
        (False, 1, [one_search, 0.0]),
        (False, 2, [-entropy(0.25), 0.0]),
        (True, 1, [one_search, one_search]),
    ):
        menu = Menu([[1.0, 1.0]], [[True, True]], (0, 1), (0, 1), [int(detected)])
        f = _frontier_of(beliefs, claim, toy.star, LogScore(1), models, menu, depth)
        assert_allclose(f.reward[0], expected, rtol=1e-12, atol=1e-12)
        assert list(np.asarray(f.mask[0])) == [True, detected]


LOOK = _menu([[1.0]], (0,))


def _search_pool(toy):
    """Test 5's pool: mass 2, marks with claim bits (1, 0), weighed (0.5, 0.5) by role
    0 and (0.2, 0.8) by role 1."""
    return toy.pool([1.0, 0.0], [[0.5, 0.5], [0.2, 0.8]], np.log(2.0), n_roles=2)


def _search(toy):
    """Test 5's search: it detects a planet in the HZ with probability 0.8 and any
    other with 0.2, in one day.
    """
    return toy.Search(hit=0.8, other=0.2)


def _search_belief(toy, pool):
    """A two-role belief with no draws, so every planet is in `pool`."""
    no_draws = toy.atoms([[0.0]], [[0.0]], [1.0], 0.0, 0.0, exists=False)
    return Belief(no_draws, np.zeros((2, 1)), np.zeros((2, 1)), pool)


def test_count_and_thin_by_hand(toy):
    """Spec layer 5 test 5. Mass 2, two marks with claim probabilities (1, 0) and
    detection (0.8, 0.2); role 0 weighs the marks (0.5, 0.5), role 1 (0.2, 0.8).

    Role 0: dbar = 0.5, so a null leaves mass 2 x 0.5 = 1.0, and the marks reweight by
    (0.2, 0.8) to (0.1 / 0.5, 0.4 / 0.5) = (0.2, 0.8); the claim-true rate falls from
    2 x 0.5 = 1.0 to 0.2. `Count` is then 1.0 - 0.2 = 0.8, the expected number found
    2 x (0.5 x 1 x 0.8 + 0.5 x 0 x 0.2). Role 1: dbar = 0.32, mass 1.36, rate from 0.4
    to 0.4 x 0.2 = 0.08. Completeness is 0.8 for both roles (only the first mark is
    claim-true). Through `frontier`, one search is worth 0.8 planets and a revisit at
    depth 2 adds the 0.2 x 0.8 = 0.16 the first look missed: 0.96.

    Fails if a null is not thinned (a revisit counts the same planets twice: 1.6 at
    depth 2, and `Count` stays 0), if the mass is thinned but the marks are not
    reweighted (1.2 at depth 2), if a role's mass is thinned with another role's mark
    weights, or if `Count` counts from the current pool rather than from time zero.
    """
    pool, claim = _search_pool(toy), Claim((toy.InHz(),))
    after = toy.jit.thin(pool, jnp.array([0.8, 0.2]))
    assert_allclose(jnp.exp(after.log_mass), [1.0, 1.36], rtol=1e-12)
    assert_allclose(toy.jit.mark_weights(after)[0], [0.2, 0.8], rtol=1e-12)
    assert_allclose(toy.jit.pool_rate(after, claim, toy.star), [0.2, 0.08], rtol=1e-12)
    assert_allclose(toy.jit.pool_rate(pool, claim, toy.star), [1.0, 0.4], rtol=1e-12)

    belief = _search_belief(toy, after)
    assert_allclose(Count(pool)(*toy.split(belief), claim, toy.star), 0.8, rtol=1e-12)

    action = Action(0, 0, 1.0)
    assert_allclose(
        toy.jit.completeness(pool, _search(toy), action, claim, toy.star), [0.8, 0.8]
    )

    start = toy.split(toy.stack(_search_belief(toy, pool)))
    for depth, found in ((1, 0.8), (2, 0.96)):
        f = _frontier_of(
            start, claim, toy.star, Count(toy.stack(pool)), (_search(toy),), LOOK, depth
        )
        assert_allclose(f.reward[0, 0], found, rtol=1e-12)


def test_a_follow_up_is_charged_at_the_price(toy):
    """A follow-up at depth 2 is taken when it adds more than lam times its cost, and
    the frontier counts its time: the reward is the plan's whole gain and the cost the
    first look plus the follow-up's expected time.

    On test 5's star the revisit adds 0.16 planets for one day. At lam = 0 and 0.1 the
    plan revisits: reward 0.8 + 0.16 = 0.96, cost 2 days, and net of the follow-up's
    price 0.96 and 0.86. At lam = 0.2 the revisit would net -0.04, so the plan stops
    after the first look: reward 0.8, cost 1 day.

    Fails if a follow-up is not charged (it is taken at every price), if the first
    entry is charged as well (net 0.76 at lam = 0.1), if a losing follow-up is forced
    (0.96 and 2 days at lam = 0.2), if the follow-up's time is left out of the cost,
    or if the price is read as a Python value (a second price retraces).
    """
    pool, claim = _search_pool(toy), Claim((toy.InHz(),))
    start = toy.split(toy.stack(_search_belief(toy, pool)))
    count = Count(toy.stack(pool))

    def plan(lam):
        f = _frontier_of(start, claim, toy.star, count, (_search(toy),), LOOK, 2, lam)
        return f.reward[0, 0], f.cost_d[0, 0]

    free = plan(0.0)
    before = len(traces)
    charged = [plan(lam) for lam in (0.1, 0.2)]
    assert len(traces) == before
    reward, cost_d = np.asarray([free, *charged]).T
    lam = np.array([0.0, 0.1, 0.2])
    assert_allclose(reward, [0.96, 0.96, 0.8], rtol=1e-12)
    assert_allclose(cost_d, [2.0, 2.0, 1.0], rtol=1e-12)
    net = reward - lam * (cost_d - 1.0)  # less the price of the follow-up
    assert_allclose(net, [0.96, 0.86, 0.8], rtol=1e-12)


def test_a_follow_up_worth_nothing_more_is_never_bought_at_price_zero(toy):
    """At a zero price a plan breaks a tie toward stopping, as `dispatch` does.

    Two stars at depth 2: test 5's star (a revisit adds 0.16 planets) and one whose
    pool is empty, so every look at it is worth exactly nothing. At lam = 0 the empty
    star's plan stops after its first look: cost 1 day, not the 1.5 of a derivative
    that splits a tie in half; test 5's star revisits: reward 0.96 over 2 days.

    Fails if a follow-up that adds exactly nothing is taken in part or whole at a zero
    price (the plan's follow-up time, and so the spend the price clears, is then
    decided by rounding), or if a follow-up worth something is dropped.
    """
    claim = Claim((toy.InHz(),))
    empty = toy.pool([1.0, 0.0], [[0.5, 0.5]] * 2, -np.inf, n_roles=2)
    pools = (empty, _search_pool(toy))
    start = toy.split(toy.stack(*[_search_belief(toy, pool) for pool in pools]))
    f = _frontier_of(
        start, claim, toy.star, Count(toy.stack(*pools)), (_search(toy),),
        _menu(np.ones((2, 1)), (0,)), 2, 0.0,
    )  # fmt: skip
    assert_allclose(f.cost_d[:, 0], [1.0, 2.0], rtol=1e-12)
    assert_allclose(f.reward[:, 0], [0.0, 0.96], rtol=1e-12, atol=1e-15)


def test_follow_up_time_raises_the_price_and_keeps_the_budget(toy):
    """With the follow-ups' time counted, the price clears the time the plans use.

    Eight stars like test 5's, scaled by their claim-true rate R (pool mass 2 R): two
    with R = 10 and six with R = 1.0, 0.9, ..., 0.5; one one-day search each; depth 2,
    built at lam = 0.72; budget 6 days. A first look finds 0.8 R and a revisit 0.16 R
    more, so the large stars revisit (1.6 > 0.72): reward 9.6 over 2 days; the small
    ones do not (0.16 R <= 0.16): reward 0.8 R over 1 day. The dual then clears at
    lam = 0.72 (4 days for the large stars, 1 for R = 1.0 and the tied R = 0.9 star
    by the tie rule), the price it was built at, and the plans bought use 6 days, the
    whole budget: their time is certain here, since a search's outcome on these
    draws, which hold no planet, is always the miss. Pricing the first looks alone
    (rewards net of the revisit, 8.88 for a large star, over one day) clears at
    lam = 0.56 and buys plans that use 8 days.

    Fails if the follow-up time is dropped from the spend (the price falls to 0.56
    and the plans bought overrun the budget), or counted where no follow-up is taken.
    """
    rates = np.array([10.0, 10.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    pools = [
        toy.pool([1.0, 0.0], [[0.5, 0.5]] * 2, np.log(2.0 * r), n_roles=2)
        for r in rates
    ]
    start = toy.split(toy.stack(*[_search_belief(toy, pool) for pool in pools]))
    count = Count(toy.stack(*pools))
    menu = _menu(np.ones((8, 1)), (0,))
    lam0, budget = 0.72, jnp.asarray(6.0)
    f = _frontier_of(
        start, Claim((toy.InHz(),)), toy.star, count, (_search(toy),), menu, 2, lam0
    )
    revisits = rates == 10.0
    assert_allclose(f.cost_d[:, 0], np.where(revisits, 2.0, 1.0), rtol=1e-12)
    assert_allclose(f.reward[:, 0], np.where(revisits, 0.96, 0.8) * rates, rtol=1e-12)

    lam, _, _ = solve_price(f, budget)
    bought = np.asarray(dispatch(lam, f, budget)[0])[:, 0]
    assert_allclose(lam, lam0, rtol=1e-12)
    assert list(np.flatnonzero(bought)) == [0, 1, 2, 3]
    assert_allclose(np.sum(np.asarray(f.cost_d)[bought, 0]), 6.0, rtol=1e-12)

    first_looks = Frontier(f.reward - lam0 * (f.cost_d - 1.0), np.ones((8, 1)), f.mask)
    lam_first, _, _ = solve_price(first_looks, budget)
    bought_first = np.asarray(dispatch(lam_first, first_looks, budget)[0])[:, 0]
    assert_allclose(lam_first, 0.56, rtol=1e-12)
    assert lam > lam_first
    assert_allclose(np.sum(np.asarray(f.cost_d)[bought_first, 0]), 8.0, rtol=1e-12)


def test_a_model_that_does_not_search_leaves_the_pool(toy):
    """Only a searching model thins the pool. A gas reading offered at test 5's star
    (claim-true pool rate 1.0) finds nothing: its completeness is 0 and its `Count`
    reward is 0, while the search beside it in the same menu is worth 0.8.

    Fails if a model that does not search thins the pool, for example by reading its
    cell 0 as the miss: both readings fall below the edge with probability 0.25, so it
    would detect 0.75 of each mark and be worth 0.75 planets.
    """
    pool, claim = _search_pool(toy), Claim((toy.InHz(),))
    reading = GaussianGasModel(snr_one_day=[2.0, 2.0], edges=[0.0])
    action = Action(0, 0, 1.0)
    assert_allclose(
        toy.jit.completeness(pool, reading, action, claim, toy.star), [0.0, 0.0]
    )
    f = _frontier_of(
        toy.split(toy.stack(_search_belief(toy, pool))),
        claim,
        toy.star,
        Count(toy.stack(pool)),
        (_search(toy), reading),
        _menu([[1.0, 1.0]], (0, 1)),
        1,
    )
    assert_allclose(f.reward[0], [0.8, 0.0], rtol=1e-12, atol=1e-12)


def test_a_refused_forecast_reads_nan_not_zero(toy):
    """A forecast the reweighting gate refuses (a NaN row) gives NaN rewards and NaN
    claim-bit information, never a confident zero, while the verdict is kept.

    Test 5's star, search and pool (one placeholder draw, so the programs are test 5's),
    both roles the gas prior N(0, 1): the draw's Kish size is 1, so a gate with
    `min_n_eff = 5` refuses the forecast's row. The `LogScore` utility, the `Count`
    rewards at depth 1 and 2 and `claim_information` are NaN.

    Fails if a guard reads a NaN forecast as giving no mass (`forecast > 0` in
    `expected_log_score`, `p_cell > 0` in `frontier` or `claim_information`), which
    values a refused star at zero and so never observes it, or if the log of a NaN
    mass is read as `-inf` (a refused pool row would thin to an empty pool).
    """
    pool, claim = _search_pool(toy), Claim((toy.InHz(),))
    atoms = _search_belief(toy, pool).atoms
    prior = toy.GasPrior("toy", n_gas=2)
    roles, log_like = Roles(prior, toy.stack(prior)), jnp.zeros((2, 1))
    refused = toy.jit.role_belief(
        atoms, roles, log_like, toy.star, pool, min_n_eff=jnp.asarray(5.0),
        max_khat=jnp.inf,
    )  # fmt: skip
    kept = toy.jit.role_belief(atoms, roles, log_like, toy.star, pool, **toy.ungated)
    forecast, verdict = select_roles(refused, slice(0, 1)), toy.split(kept)[1]
    assert np.all(np.isnan(np.asarray(forecast.log_prior_ratio)))
    assert np.all(np.isfinite(np.asarray(verdict.log_prior_ratio)))

    assert np.isnan(LogScore(1)(forecast, verdict, claim, toy.star))
    beliefs = (toy.stack(forecast), toy.stack(verdict))
    for depth in (1, 2):
        f = _frontier_of(
            beliefs, claim, toy.star, Count(toy.stack(pool)), (_search(toy),), LOOK,
            depth,
        )  # fmt: skip
        assert np.isnan(f.reward[0, 0]), depth
    info = toy.jit.claim_information(
        forecast, verdict, _search(toy), Action(0, 0, 1.0), claim, toy.star, PLANET
    )
    assert np.all(np.isnan(np.asarray(info)))
    # A refused pool row thins to an unknown mass, not to an empty pool.
    nan_row = eqx.tree_at(
        lambda p: p.log_prior_ratio, pool, pool.log_prior_ratio.at[0].set(np.nan)
    )
    thinned = toy.jit.thin(nan_row, jnp.array([0.8, 0.2]))
    assert np.isnan(thinned.log_mass[0]) and np.isfinite(thinned.log_mass[1])


def test_a_second_star_and_world_add_no_traces(toy):
    """`frontier` traces once per structure: two worlds of two stars, with other
    weights, menu exposures, costs, masks and price, run one program, and a star's
    rewards do not depend on the stars stacked beside it: the first world with its two
    stars swapped gives the same rewards, swapped, from the same program.

    Fails if the menu, its mask or a cell count is read as a Python value, if the
    depth or the model table is traced, or if a star's recursion reads another star's
    belief or menu.
    """
    claim = _spectra_claim(toy)

    def world(psis, exposure_d, mask):
        beliefs = toy.split(toy.stack(*[_spectra_belief(toy, p) for p in psis]))
        menu = _menu(exposure_d, (0, 1, 2), mask)
        return beliefs, menu

    first = world((0.7, 0.3), [[1.0] * 3, [0.5, 1.0, 1.0]], True)
    second = world((0.4, 0.9), [[1.0, 0.1, 1.0], [2.0] * 3], [[1, 0, 1], [1, 1, 0]])
    f1 = _frontier_of(first[0], claim, toy.star, LogScore(1), SPECTRA, first[1], 2)
    before = len(traces)
    f2 = _frontier_of(
        second[0], claim, toy.star, LogScore(1), SPECTRA, second[1], 2, lam=0.05
    )
    assert len(traces) == before
    exposure_d = np.array([[1.0, 0.1, 1.0], [2.0] * 3])
    masked = ~np.asarray(f2.mask)
    assert_allclose(np.asarray(f2.cost_d)[masked], exposure_d[masked])
    assert np.all(np.asarray(f2.cost_d) >= exposure_d)  # follow-ups add time
    assert_allclose(f2.reward[0, 1], 0.0)  # masked
    swapped = world((0.3, 0.7), [[0.5, 1.0, 1.0], [1.0] * 3], True)
    f3 = _frontier_of(swapped[0], claim, toy.star, LogScore(1), SPECTRA, swapped[1], 2)
    assert len(traces) == before
    assert_allclose(f3.reward[::-1], f1.reward, rtol=1e-12)
    assert_allclose(f3.cost_d[::-1], f1.cost_d, rtol=1e-12)
    assert not np.allclose(f1.reward, f2.reward)


@pytest.mark.slow(reason="nested Monte Carlo, 2**17 outcomes and 2**20 per-draw cells")
def test_cells_reward_matches_nested_monte_carlo(toy):
    """Spec layer 5 test 6: the depth-1 reward from a model's analytic cells equals a
    nested Monte Carlo built from its `sample` alone, within 3 standard errors.

    Outer loop: N = 2**17 outcomes, each a draw picked by the forecast weights, a claim
    bit drawn with that draw's claim probability, and a reading sampled from the model
    and binned by `cell`. Inner loop: each draw's cell probabilities binned from 2**20
    readings, which give the verdict's posterior on each sampled cell. The reward is the
    mean of ln p_gamma(Z | Y) - ln p_gamma(Z). The per-outcome sd is about 0.5, so the
    outer standard error is about 0.0014 (measured: the estimate lands 4e-5 from the
    analytic reward); the inner binning moves a cell probability by about 1e-3 of
    itself.

    Fails if the reward mixes cells with the wrong role's weights, reads the posterior
    of the wrong role, drops the cells' normalization, or if the analytic cells
    disagree with what the model samples.
    """
    hz = [[0.9], [0.2], [0.6], [1.0], [0.0], [0.5]]
    water = [[0.4], [-0.3], [1.2], [-0.8], [0.1], [-0.05]]
    atoms = toy.atoms(hz, np.ones_like(hz), np.ones(6), water, 0.0, n_gas=1)
    w_forecast = np.array([0.3, 0.1, 0.2, 0.1, 0.2, 0.1])
    w_verdict = np.array([0.1, 0.25, 0.1, 0.3, 0.05, 0.2])
    b = Belief(
        atoms, np.log([w_forecast, w_verdict]), np.zeros((2, 6)), toy.empty_pool(2, 1)
    )
    claim = Claim((toy.InHz(), GasAbove(0, 0.0)))
    model = GaussianGasModel(snr_one_day=[3.0], edges=[-0.5, 0.0, 0.5])
    menu = _menu([[1.0]], (0,))
    exact = _frontier_of(
        toy.split(toy.stack(b)), claim, toy.star, LogScore(1), (model,), menu, 1
    )
    exact = float(exact.reward[0, 0])

    action = Action(0, 0, 1.0)
    n_cell = model.n_cells(atoms)
    readings = jax.jit(jax.vmap(model.sample, in_axes=(0, None, None)))
    cell_of = jax.jit(jax.vmap(jax.vmap(model.cell)))

    inner = jax.random.split(jax.random.key(1), 2**20)
    binned = jax.nn.one_hot(cell_of(readings(inner, atoms, action)), n_cell).mean(0)
    h = np.asarray(hz)[:, 0] * (np.asarray(water)[:, 0] > 0)  # P(Z = 1 | draw)
    joint_g = w_verdict[:, None] * np.asarray(binned)  # (A, Y)
    post_g = (joint_g * h[:, None]).sum(0) / joint_g.sum(0)  # gamma(Z = 1 | Y)
    prior_g = (w_verdict * h).sum()

    rng = np.random.default_rng(0)
    n = 2**17
    draw = rng.choice(6, size=n, p=w_forecast)
    z = rng.uniform(size=n) < h[draw]
    outer = jax.random.split(jax.random.key(0), n)
    cells = np.asarray(cell_of(readings(outer, atoms, action)))[np.arange(n), draw]
    p_after = np.where(z, post_g[cells], 1.0 - post_g[cells])
    p_before = np.where(z, prior_g, 1.0 - prior_g)
    score = np.log(p_after) - np.log(p_before)
    se = score.std() / np.sqrt(n)
    assert abs(score.mean() - exact) < 3.0 * se
