"""The price of time on hand-built frontiers: the dual, its minimum and the tie rule."""

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from spaceodyssey.price import dispatch, dual, solve_price
from spaceodyssey.value import Frontier, value_and_spend


def _plan(f, capacity_d):
    lam, bound = solve_price(f, capacity_d)
    return lam, bound, dispatch(lam, f, capacity_d)


plan = eqx.filter_jit(_plan)
duals = jax.jit(jax.vmap(dual, in_axes=(0, None, None)))


def _frontier(reward, cost_d, mask=None):
    reward = jnp.asarray(reward, float)
    mask = jnp.ones(reward.shape, bool) if mask is None else mask
    return Frontier(reward, jnp.asarray(cost_d, float), jnp.asarray(mask))


def test_two_star_tie_is_priced_at_the_kink_and_funds_star_zero():
    """Two stars, each one one-day option worth one nat, budget one day.

    D(lam) = lam + 2 max(0, 1 - lam) is 2 - lam below 1 and lam above, so the minimum
    is D(1) = 1 at lam = 1: the bound is one nat, the best any schedule earns. The jvp
    spend there averages the tie to 1.0, inside the subdifferential [0, 2]; the
    one-sided spends bracket the budget; the tie rule funds star 0 only.

    Fails if lam B enters with the wrong sign, if the null option is dropped from the
    breakpoints (the price falls to 0 and the bound to 2), or if a tie funds both stars
    or the higher index.
    """
    f = _frontier([[1.0], [1.0]], [[1.0], [1.0]])
    lam, bound, bought = plan(f, jnp.asarray(1.0))
    assert_allclose([lam, bound], [1.0, 1.0], rtol=1e-12)
    spend = float(jnp.sum(value_and_spend(lam, f)[1]))
    assert 0.0 <= spend <= 2.0
    assert_allclose(spend, 1.0, rtol=1e-12)
    spend_right = float(jnp.sum(value_and_spend(lam + 1e-6, f)[1]))
    spend_left = float(jnp.sum(value_and_spend(lam - 1e-6, f)[1]))
    assert spend_right <= 1.0 <= spend_left
    assert np.array_equal(bought, [[True], [False]])


@pytest.mark.parametrize(
    ("reward", "capacity_d", "bought_expected", "bound_expected"),
    [
        # every reward nonpositive: nothing is worth buying
        ([[-1.0, 0.0], [-0.5, -2.0], [0.0, -0.1]], 2.0, [[0, 0], [0, 0], [0, 0]], 0.0),
        # the budget exceeds every cost: each star buys its best option
        ([[0.4, 0.9], [0.3, -2.0], [0.2, 0.1]], 100.0, [[0, 1], [1, 0], [1, 0]], 1.4),
    ],
)
def test_unused_budget_needs_a_zero_price(
    reward, capacity_d, bought_expected, bound_expected
):
    """When the budget cannot all be used profitably the price is zero.

    With every reward nonpositive (zero included) nothing is bought and the bound is 0;
    with a budget larger than every cost each star buys its best option and the bound
    is the sum of those rewards.

    Fails if the zero price is not a candidate, if a zero-reward option tied with the
    null is bought at price zero, or if a slack budget is priced above zero.
    """
    cost_d = [[1.0, 2.0], [0.5, 1.5], [1.0, 0.25]]
    lam, bound, bought = plan(_frontier(reward, cost_d), jnp.asarray(capacity_d))
    assert lam == 0.0
    assert_allclose(bound, bound_expected, atol=1e-12)
    assert np.array_equal(bought, np.asarray(bought_expected, bool))


@pytest.mark.parametrize(
    ("reward_a", "reward_b", "funded"),
    [(0.5, 0.1, range(10)), (0.1, 0.5, range(10, 20))],
    ids=["count", "log_score"],
)
def test_the_reward_decides_which_stars_are_funded(reward_a, reward_b, funded):
    """Ten A stars (index 0 to 9) and fifty B stars, one one-day option each, budget
    10 days.

    Count rewards (A 0.5 planets, B 0.1) fund the ten A for 5 planets. Under Count,
    D(lam) = 10 lam + 10 max(0, 0.5 - lam) + 50 max(0, 0.1 - lam) equals 5 on
    [0.1, 0.5]. LogScore rewards (A 0.1 nats, B 0.5) fund ten B, the lowest-index ones
    by the tie rule, for 5 nats. In both the bound equals the realized value.

    Fails if lam B enters with the wrong sign, if the null option is dropped from the
    breakpoints, or if the tie rule overspends or funds the wrong stars.
    """
    reward = [[reward_a]] * 10 + [[reward_b]] * 50
    f = _frontier(reward, np.ones((60, 1)))
    _, bound, bought = plan(f, jnp.asarray(10.0))
    expected = np.zeros((60, 1), bool)
    expected[list(funded)] = True
    assert np.array_equal(bought, expected)
    assert_allclose(bound, 5.0, rtol=1e-12)
    assert_allclose(jnp.sum(jnp.where(bought, f.reward, 0.0)), 5.0, rtol=1e-12)


def _brute_force(reward, cost_d, capacity_d):
    """Best value of any schedule within the budget, at most one option per star."""
    n_star, n_option = reward.shape
    best = 0.0
    for pick in itertools.product(range(n_option + 1), repeat=n_star):
        rows = [s for s in range(n_star) if pick[s]]
        cols = [pick[s] - 1 for s in rows]
        if cost_d[rows, cols].sum() <= capacity_d:
            best = max(best, reward[rows, cols].sum())
    return best


@given(
    reward=st.lists(st.floats(-1.0, 2.0), min_size=6, max_size=6),
    cost=st.lists(st.floats(0.05, 2.0), min_size=6, max_size=6),
    capacity_d=st.floats(0.0, 4.0),
)
# A flat dual whose two ends differ by less than roundoff: D(0) = 1 + 1e-20 rounds to
# D(1) = 1, yet at lam = 0 the 1e-20 option is bought outright and overspends.
@example(reward=[0.0, 0.0, 0.0, 1e-20, 0.0, 1.0], cost=[1.0] * 6, capacity_d=1.0)
def test_the_dual_bounds_every_schedule(reward, cost, capacity_d):
    """Three stars with two options each: the bound is at least the best of all 27
    schedules found by brute force, the dual at every price on a grid is too (weak
    duality), the bound is the least dual over every breakpoint (found here by NumPy
    enumeration, with the null option), and the dispatched schedule fits the budget.

    Fails if lam B enters with the wrong sign (the bound drops below the brute-force
    optimum), if a breakpoint is missing from the candidates or the search over them
    stops short of the minimum, or if the tie rule overspends, including when roundoff
    on a flat dual lands the price where the untied choices alone exceed the budget.
    """
    reward = np.reshape(reward, (3, 2))
    cost_d = np.reshape(cost, (3, 2))
    f = _frontier(reward, cost_d)
    lam, bound, bought = plan(f, jnp.asarray(capacity_d))
    best = _brute_force(reward, cost_d, capacity_d)
    grid = duals(jnp.linspace(0.0, 25.0, 501), f, jnp.asarray(capacity_d))
    r, c = (np.pad(x, ((0, 0), (1, 0))) for x in (reward, cost_d))
    with np.errstate(divide="ignore", invalid="ignore"):
        kinks = (r[:, :, None] - r[:, None, :]) / (c[:, :, None] - c[:, None, :])
    kinks = np.append(np.maximum(kinks[np.isfinite(kinks)], 0.0), 0.0)
    assert lam >= 0.0
    assert best <= bound + 1e-12
    assert np.all(best <= np.asarray(grid) + 1e-12)
    least = np.min(duals(jnp.asarray(kinks), f, jnp.asarray(capacity_d)))
    assert_allclose(bound, least, atol=1e-12)
    bought = np.asarray(bought)
    assert np.all(bought.sum(axis=1) <= 1)
    assert cost_d[bought].sum() <= capacity_d * (1 + 1e-12)
    assert reward[bought].sum() <= bound + 1e-12


def test_a_discrete_exposure_menu_reaches_the_ayo_limit():
    """Three stars with n_i C_i(t) = n_i (1 - exp(-t / tau_i)), (n_i, tau_i) = (1.0, 1),
    (0.8, 2), (0.5, 1), and a budget of 3 days.

    Continuous: the interior solution t_i = tau_i ln(n_i / (tau_i lam)) with
    sum t_i = 3, solved by bisection, is lam = 0.2512, t = (1.381, 0.930, 0.688), and
    every funded star's marginal reward dV/dt(t_i) equals lam to 1e-8. Discrete: on a
    menu with spacing h = 0.05 days, each chosen exposure t_k satisfies
    V'(t_{k+1}) <= lam <= V'(t_{k-1}) at the solved price, and the continuous price
    lies in the same bracket.

    Fails if the solver misses the clearing breakpoint, if a stop-gradient cuts the
    spend it bisects on, or if the tie rule overspends the budget.
    """
    n_bar = np.array([1.0, 0.8, 0.5])
    tau = np.array([1.0, 2.0, 1.0])

    def exposure(lam):
        return np.maximum(tau * np.log(n_bar / (tau * lam)), 0.0)

    low, high = 1e-6, 2.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        low, high = (mid, high) if exposure(mid).sum() > 3.0 else (low, mid)
    lam_c = 0.5 * (low + high)
    t_c = exposure(lam_c)
    assert_allclose(lam_c, 0.2512, atol=5e-5)
    assert_allclose(t_c, [1.381, 0.930, 0.688], atol=5e-4)

    def marginal(t, n, tau_i):
        return jax.grad(lambda x: n * (1.0 - jnp.exp(-x / tau_i)))(t)

    slopes = jax.vmap(marginal)(jnp.asarray(t_c), jnp.asarray(n_bar), jnp.asarray(tau))
    assert np.all(np.abs(np.asarray(slopes) - lam_c) < 1e-8)

    h = 0.05
    t = h * np.arange(1, 61)
    reward = n_bar[:, None] * (1.0 - np.exp(-t[None] / tau[:, None]))
    f = _frontier(reward, np.broadcast_to(t, reward.shape))
    lam, _, bought = plan(f, jnp.asarray(3.0))
    bought = np.asarray(bought)
    assert np.all(bought.sum(axis=1) == 1)
    t_k = h * (bought.argmax(axis=1) + 1)
    assert t_k.sum() <= 3.0 + 1e-12
    v_next = jax.vmap(marginal)(jnp.asarray(t_k + h), jnp.asarray(n_bar), tau)
    v_prev = jax.vmap(marginal)(jnp.asarray(t_k - h), jnp.asarray(n_bar), tau)
    for price in (lam, lam_c):
        assert np.all(np.asarray(v_next) <= price)
        assert np.all(price <= np.asarray(v_prev))


def test_a_second_frontier_of_the_same_shape_adds_no_traces():
    """Pricing and dispatch trace once per padded shape: a second frontier with other
    rewards, costs and mask and another budget reuses the program, and the two run as
    two worlds under one vmap with the same results.

    Fails if the budget, rewards or mask become static (a Python scalar or a static
    field), if the menu or star count is passed as a Python value, or if the solver
    branches in Python on a traced value (it then cannot be mapped over worlds).
    """
    traces = []

    @eqx.filter_jit
    def counted(f, capacity_d):
        traces.append(None)
        return _plan(f, capacity_d)

    first = _frontier([[1.0, 2.0], [0.5, 0.2]], [[1.0, 3.0], [0.5, 0.1]])
    one = counted(first, jnp.asarray(2.0))
    before = len(traces)
    mask = [[True, False], [True, True]]
    second = _frontier([[0.3, -1.0], [2.0, 0.7]], [[0.2, 1.0], [4.0, 0.3]], mask)
    two = counted(second, jnp.asarray(5.0))
    assert len(traces) - before == 0
    worlds = jax.tree.map(lambda *x: jnp.stack(x), first, second)
    mapped = jax.vmap(_plan)(worlds, jnp.asarray([2.0, 5.0]))
    for got, want in zip(mapped, zip(one, two, strict=True), strict=True):
        assert_allclose(got, np.stack(want), rtol=1e-12)
