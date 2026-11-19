"""Value and spend: rewards from beliefs, and the value of hand-built frontiers; closed
forms and the sign errors they catch.
"""

import itertools
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from spaceodyssey.belief import Belief, mark_weights, pool_rate
from spaceodyssey.claim import Claim, GasAbove
from spaceodyssey.information import claim_information
from spaceodyssey.toy import GaussianGasModel, StepReportModel
from spaceodyssey.types import AbstractObservationModel, Action
from spaceodyssey.value import (
    Count,
    Frontier,
    LogScore,
    Menu,
    completeness,
    frontier,
    thin,
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


# Rewards from beliefs. Planning beliefs have two roles: the forecast (row 0) and one
# verdict member (row 1). `frontier` runs on stacked stars; `_one` stacks a single star.
traces = []
PLANET = jnp.asarray(0)


@eqx.filter_jit
def _traced_frontier(belief, claim, star, utility, models, menu, lam, depth):
    traces.append(None)
    return frontier(belief, claim, star, utility, models, menu, lam, depth)


def _frontier_of(belief, claim, star, utility, models, menu, depth, lam=0.0):
    """`frontier` under one jit, with the price an array so it is traced."""
    lam = jnp.asarray(lam, float)
    return _traced_frontier(belief, claim, star, utility, models, menu, lam, depth)


def _one(x):
    """Stack one star's pytree along a leading star axis of length one."""
    return jax.tree.map(lambda leaf: jnp.asarray(leaf)[None], x)


def _menu(exposure_d, cost_d, model, mask=True):
    """A menu over stars: `exposure_d` and `cost_d` have shape (S, M)."""
    exposure_d = jnp.asarray(exposure_d, float)
    zero = jnp.zeros(exposure_d.shape, int)
    mask = jnp.broadcast_to(jnp.asarray(mask, bool), exposure_d.shape)
    return Menu(Action(zero, zero, zero, exposure_d), cost_d, mask, model)


def _empty_pool(toy, n_gas):
    """A two-role pool with no expected planets, its one mark a point gas state."""
    marks = toy.atoms([[0.0]], [[0.0]], [1.0], 0.0, 0.0, n_gas)
    return eqx.tree_at(lambda p: p.marks, toy.empty_pool(2), marks)


# The two spectra (V; T): water and oxygen reports at S/N 6 (above the step gate), each
# alone (slots 0 and 1) and both in one action (slot 2). The menu lists both first, so
# its entries are not in slot order.
SPECTRA = (
    StepReportModel(snr_one_day=[6.0, 0.0]),
    StepReportModel(snr_one_day=[0.0, 6.0]),
    StepReportModel(snr_one_day=[6.0, 6.0]),
)
SPECTRA_MENU = _menu([[1.0] * 3], [[1.0] * 3], (2, 0, 1))


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
    return Belief(atoms, log_prior, np.zeros((2, 8)), _empty_pool(toy, 2))


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
    info = claim_information(
        b, claim, toy.star, PLANET, SPECTRA[2], Action(0, 0, 0, 1.0)
    )
    assert_allclose(info, [0.0, gain], atol=1e-12)  # row 0 is I_F
    f = _frontier_of(_one(b), claim, toy.star, LogScore(1), SPECTRA, SPECTRA_MENU, 1)
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
    return Belief(atoms, log_prior, np.zeros((2, 4)), _empty_pool(toy, 1))


# One gas report at S/N 6 (fine: the bit is reported) or 3 (merged: nothing reported).
GAS_REPORT = (StepReportModel(snr_one_day=[6.0]),)
FINE_AND_MERGED = _menu([[1.0, 0.25]], [[1.0, 0.25]], (0, 0))


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
    claim = Claim((toy.InHz(),))
    f = _frontier_of(
        _one(b), claim, toy.star, LogScore(1), GAS_REPORT, FINE_AND_MERGED, 1
    )
    assert_allclose(f.reward[0], [-1.390, 0.0], atol=5e-4)
    exact = 0.9 * np.log(0.1) + 0.1 * np.log(0.9) - np.log(0.5)
    assert_allclose(f.reward[0, 0], exact, rtol=1e-12)
    mutual = np.log(2.0) + 0.9 * np.log(0.9) + 0.1 * np.log(0.1)  # ln 2 - h(0.9)
    assert round(mutual, 3) == 0.368
    for exposure_d, row in ((1.0, [mutual, exact]), (0.25, [0.0, 0.0])):
        action = Action(0, 0, 0, exposure_d)
        info = claim_information(b, claim, toy.star, PLANET, GAS_REPORT[0], action)
        assert_allclose(info, row, atol=1e-12)  # row 0 is I_F, row 1 the reward


def test_value_does_not_fall_with_depth(toy):
    """Spec layer 5 test 4: the reward of every entry is nondecreasing in depth.

    On the two spectra (psi = 0.7) a water report alone is worth
    psi ln 2 + (1 - psi) [ln(1 - psi / 2) - ln(1 - psi / 4)] = 0.414 at depth 1; at
    depth 2 the oxygen report may follow, so it is worth the whole 0.667, and depth 3
    adds nothing. On the coarsening fixture a noisy reading (Gaussian, S/N 2) loses
    under the reversed verdict, and so would every repeat of it: a deeper plan stops
    after the first, so depths 1, 2 and 3 agree (a plan forced on falls to -0.61 at
    depth 2).

    Fails if a deeper plan cannot stop (it would be forced into a second losing
    reading), if the continuation is averaged instead of maximized, or if a follow-up
    does not see the earlier outcome (the oxygen report would be valued from the
    prior).
    """
    psi = 0.7
    alone = psi * np.log(2.0) + (1 - psi) * (np.log(1 - psi / 2) - np.log(1 - psi / 4))
    both = psi * np.log(4.0) + (1 - psi) * (np.log(1 - psi) - np.log(1 - psi / 4))
    b, claim = _one(_spectra_belief(toy, psi)), _spectra_claim(toy)
    rewards = np.stack(
        [
            _frontier_of(
                b, claim, toy.star, LogScore(1), SPECTRA, SPECTRA_MENU, d
            ).reward[0]
            for d in (1, 2, 3)
        ]
    )
    assert_allclose(rewards[0], [both, alone, alone], rtol=1e-12)
    assert_allclose(rewards[1:], both, rtol=1e-12)
    assert np.all(np.diff(rewards, axis=0) >= -1e-12)

    reading = (GaussianGasModel(snr_one_day=[2.0], edges=[0.0]),)
    losing = np.stack(
        [
            _frontier_of(
                _one(_coarsening_belief(toy)),
                Claim((toy.InHz(),)),
                toy.star,
                LogScore(1),
                reading,
                _menu([[1.0]], [[1.0]], (0,)),
                d,
            ).reward[0, 0]
            for d in (1, 2, 3)
        ]
    )
    assert losing[0] < 0.0
    assert_allclose(losing[1:], losing[0], rtol=1e-12)


class Search(AbstractObservationModel):
    """A test search: it detects a planet with probability `hit` when its `hz` bit is
    set and `other` otherwise. Cell 0 is the miss, cell 1 the detection; a slot with no
    planet is always a miss. Read only through `cells`.
    """

    searches: ClassVar[bool] = True
    hit: jax.Array = eqx.field(converter=jnp.asarray)
    other: jax.Array = eqx.field(converter=jnp.asarray)

    def cells(self, atoms, action):
        d = jnp.where(atoms.cloud.samples["hz"][:, 0] > 0.5, self.hit, self.other)
        d = jnp.where(atoms.exists[:, 0], d, 0.0)
        return jnp.stack([1.0 - d, d], axis=-1)

    def sample(self, atoms, action, key):
        raise NotImplementedError("read only through cells")

    def stats(self, payload, atoms, star):
        raise NotImplementedError("read only through cells")

    def finalize(self, stats, atoms, star):
        raise NotImplementedError("read only through cells")


SEARCH = (Search(hit=0.8, other=0.2),)
LOOK = _menu([[1.0]], [[1.0]], (0,))


def _search_pool(toy):
    """Test 5's pool: mass 2, marks with claim bits (1, 0), weighed (0.5, 0.5) by role
    0 and (0.2, 0.8) by role 1."""
    return toy.pool([1.0, 0.0], [[0.5, 0.5], [0.2, 0.8]], np.log(2.0), n_roles=2)


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
    after = thin(pool, jnp.array([0.8, 0.2]))
    assert_allclose(jnp.exp(after.log_mass), [1.0, 1.36], rtol=1e-12)
    assert_allclose(mark_weights(after)[0], [0.2, 0.8], rtol=1e-12)
    assert_allclose(pool_rate(after, claim, toy.star), [0.2, 0.08], rtol=1e-12)
    assert_allclose(pool_rate(pool, claim, toy.star), [1.0, 0.4], rtol=1e-12)

    belief = _search_belief(toy, after)
    assert_allclose(Count(pool)(belief, claim, toy.star), 0.8, rtol=1e-12)

    action = Action(0, 0, 0, 1.0)
    assert_allclose(completeness(pool, SEARCH[0], action, claim, toy.star), [0.8, 0.8])

    start = _one(_search_belief(toy, pool))
    for depth, found in ((1, 0.8), (2, 0.96)):
        f = _frontier_of(start, claim, toy.star, Count(_one(pool)), SEARCH, LOOK, depth)
        assert_allclose(f.reward[0, 0], found, rtol=1e-12)


def test_a_follow_up_is_charged_at_the_price(toy):
    """A follow-up at depth 2 is worth what it adds less lam times its cost; the first
    entry's cost stays in the frontier.

    On test 5's star the revisit adds 0.16 planets for one day. At lam = 0 the depth-2
    reward is test 5's 0.96. At lam = 0.1 the revisit nets 0.16 - 0.1 = 0.06, so the
    reward is 0.86. At lam = 0.2 it would net -0.04, so the plan stops after the first
    look and the reward is the depth-1 0.8. The cost is the one day of the first look.

    Fails if a follow-up is not charged (0.96 at every price), if the first entry is
    charged as well (0.76 at lam = 0.1), if a losing follow-up is forced (0.76 at
    lam = 0.2), or if the price is read as a Python value (a second price retraces).
    """
    pool, claim = _search_pool(toy), Claim((toy.InHz(),))
    start, count = _one(_search_belief(toy, pool)), Count(_one(pool))

    def reward(lam):
        f = _frontier_of(start, claim, toy.star, count, SEARCH, LOOK, 2, lam)
        assert_allclose(f.cost_d, [[1.0]])
        return f.reward[0, 0]

    free = reward(0.0)
    before = len(traces)
    charged = [reward(lam) for lam in (0.1, 0.2)]
    assert len(traces) == before
    assert_allclose([free, *charged], [0.96, 0.86, 0.8], rtol=1e-12)


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
    action = Action(0, 0, 0, 1.0)
    assert_allclose(completeness(pool, reading, action, claim, toy.star), [0.0, 0.0])
    f = _frontier_of(
        _one(_search_belief(toy, pool)),
        claim,
        toy.star,
        Count(_one(pool)),
        (SEARCH[0], reading),
        _menu([[1.0, 1.0]], [[1.0, 1.0]], (0, 1)),
        1,
    )
    assert_allclose(f.reward[0], [0.8, 0.0], rtol=1e-12, atol=1e-12)


def test_a_second_star_and_world_add_no_traces(toy):
    """`frontier` traces once per structure: two worlds of two stars, with other
    weights, menu exposures, costs, masks and price, run one program, and a star's
    rewards do not depend on the stars stacked beside it.

    Fails if the menu, its mask or a cell count is read as a Python value, if the
    depth or the model table is traced, or if a star's recursion reads another star's
    belief or menu.
    """
    claim = _spectra_claim(toy)
    stack = lambda *x: jax.tree.map(lambda *leaf: jnp.stack(leaf), *x)  # noqa: E731

    def world(psis, exposure_d, mask):
        beliefs = stack(*[_spectra_belief(toy, p) for p in psis])
        menu = _menu(exposure_d, np.asarray(exposure_d) * 2.0, (0, 1, 2), mask)
        return beliefs, menu

    first = world((0.7, 0.3), [[1.0] * 3, [0.5, 1.0, 1.0]], True)
    second = world((0.4, 0.9), [[1.0, 0.1, 1.0], [2.0] * 3], [[1, 0, 1], [1, 1, 0]])
    f1 = _frontier_of(first[0], claim, toy.star, LogScore(1), SPECTRA, first[1], 2)
    before = len(traces)
    f2 = _frontier_of(
        second[0], claim, toy.star, LogScore(1), SPECTRA, second[1], 2, lam=0.05
    )
    assert len(traces) == before
    assert_allclose(f2.cost_d, [[2.0, 0.2, 2.0], [4.0] * 3])
    assert_allclose(f2.reward[0, 1], 0.0)  # masked
    alone = _frontier_of(
        _one(_spectra_belief(toy, 0.3)),
        claim,
        toy.star,
        LogScore(1),
        SPECTRA,
        jax.tree.map(lambda x: x[1:], first[1]),
        2,
    )
    assert_allclose(alone.reward[0], f1.reward[1], rtol=1e-12)
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
        atoms, np.log([w_forecast, w_verdict]), np.zeros((2, 6)), _empty_pool(toy, 1)
    )
    claim = Claim((toy.InHz(), GasAbove(0, 0.0)))
    model = GaussianGasModel(snr_one_day=[3.0], edges=[-0.5, 0.0, 0.5])
    menu = _menu([[1.0]], [[1.0]], (0,))
    exact = _frontier_of(_one(b), claim, toy.star, LogScore(1), (model,), menu, 1)
    exact = float(exact.reward[0, 0])

    action = Action(0, 0, 0, 1.0)
    n_cell = model.n_cells(atoms)
    readings = jax.jit(jax.vmap(model.sample, in_axes=(None, None, 0)))
    cell_of = jax.jit(jax.vmap(jax.vmap(model.cell)))

    inner = jax.random.split(jax.random.key(1), 2**20)
    binned = jax.nn.one_hot(cell_of(readings(atoms, action, inner)), n_cell).mean(0)
    h = np.asarray(hz)[:, 0] * (np.asarray(water)[:, 0] > 0)  # P(Z = 1 | draw)
    joint_g = w_verdict[:, None] * np.asarray(binned)  # (A, Y)
    post_g = (joint_g * h[:, None]).sum(0) / joint_g.sum(0)  # gamma(Z = 1 | Y)
    prior_g = (w_verdict * h).sum()

    rng = np.random.default_rng(0)
    n = 2**17
    draw = rng.choice(6, size=n, p=w_forecast)
    z = rng.uniform(size=n) < h[draw]
    outer = jax.random.split(jax.random.key(0), n)
    cells = np.asarray(cell_of(readings(atoms, action, outer)))[np.arange(n), draw]
    p_after = np.where(z, post_g[cells], 1.0 - post_g[cells])
    p_before = np.where(z, prior_g, 1.0 - prior_g)
    score = np.log(p_after) - np.log(p_before)
    se = score.std() / np.sqrt(n)
    assert abs(score.mean() - exact) < 3.0 * se
