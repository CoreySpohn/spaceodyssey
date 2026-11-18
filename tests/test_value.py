"""Value and spend on hand-built frontiers: closed forms and their sign errors."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from spaceodyssey.value import Frontier, value_and_spend

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
