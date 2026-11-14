"""Threshold selection, dispatch, and budget diagnostics on small packages."""

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core.enums import AdmissionCode
from spaceodyssey.studies.selection import (
    Selection,
    act,
    capacity_of,
    select,
    trace_scalars,
)

STREAMS = ("stationary", "front_loaded", "back_loaded", "lumpy", "busy_lane")


class Options(eqx.Module):
    gain: jax.Array
    time: jax.Array

    def __iter__(self):
        return iter((self.gain, self.time))

    def with_infinite_time(self, star, option):
        return eqx.tree_at(
            lambda x: x.time, self, self.time.at[star, 0, option, 0].set(jnp.inf)
        )


def toy_options(n):
    return Options(jnp.ones((n, 1, 2)), jnp.ones((n, 1, 2, 1)))


def synthetic_stream(stream, seed):
    rng = np.random.default_rng(seed)
    gain = jnp.asarray(rng.uniform(0.1, 1.0, (8, 1, 2)))
    time = jnp.asarray(rng.uniform(15.0, 25.0, (8, 1, 2, 1)))
    arrivals = {
        "stationary": np.arange(8),
        "front_loaded": np.zeros(8),
        "back_loaded": np.arange(8) + 8,
        "lumpy": np.repeat([0, 4, 8, 12], 2),
        "busy_lane": np.arange(8),
    }[stream]
    available = jnp.arange(32)[:, None] >= jnp.asarray(arrivals)[None, :]
    lane_free = jnp.arange(32) % 2 == 0 if stream == "busy_lane" else jnp.ones(32, bool)
    return gain, time, available, lane_free


@eqx.filter_jit
def simulate_refreshes(stream, budget_d):
    gain, time, available, lane_free = stream

    def refresh(state, opportunity):
        balance_d, done, last_refused_d = state
        visible, free = opportunity
        live_gain = jnp.where((visible & ~done)[:, None, None], gain, 0.0)
        sel = select(live_gain, time, balance_d[None])
        star, option, starts = act(
            sel, live_gain, time, jnp.zeros(8, jnp.int64), ranking_resource=0
        )
        starts &= free
        charge_d = jnp.where(starts, time[star, 0, option, 0], 0.0)
        refused = (live_gain > 0) & (time[..., 0] > balance_d)
        last_refused_d = jnp.where(
            jnp.any(refused),
            jnp.min(jnp.where(refused, time[..., 0], jnp.inf)),
            last_refused_d,
        )
        done = done.at[star].set(done[star] | starts)
        return (balance_d - charge_d, done, last_refused_d), None

    (leftover_d, _, last_refused_d), _ = jax.lax.scan(
        refresh,
        (jnp.asarray(budget_d), jnp.zeros(8, bool), jnp.asarray(0.0)),
        (available, lane_free),
    )
    return budget_d - leftover_d, leftover_d, last_refused_d


def branching_toy_options(n_stars, seed):
    rng = np.random.default_rng(seed)
    label_values = {"pass": 1.0, "fail": 0.1}
    pass_probability = rng.uniform(0.2, 0.9, (n_stars, 1, 2))
    gain = jnp.asarray(
        pass_probability * label_values["pass"]
        + (1 - pass_probability) * label_values["fail"]
    )
    time = jnp.asarray(rng.uniform(4.0, 12.0, (n_stars, 1, 2, 1)))
    return gain, time


def chosen_value(sel, gain):
    flat = gain.reshape(gain.shape[0], -1)
    return float(
        jnp.where(sel.choice >= 0, flat[jnp.arange(len(flat)), sel.choice], 0).sum()
    )


def brute_force_optimum(gain, time, capacity_d):
    gains = gain.reshape(gain.shape[0], -1).tolist()
    costs = time[..., 0].reshape(gain.shape[0], -1).tolist()
    best = 0.0
    for choices in itertools.product((-1, 0, 1), repeat=len(gains)):
        cost_d = sum(costs[s][c] for s, c in enumerate(choices) if c >= 0)
        if cost_d <= capacity_d:
            best = max(best, sum(gains[s][c] for s, c in enumerate(choices) if c >= 0))
    return best


def finishing_and_search_options():
    return jnp.asarray([[[0.45]], [[0.05]]]), jnp.asarray([[[[25.0]]], [[[10.0]]]])


def chosen_kinds(sel):
    return {
        kind
        for kind, c in zip(("finishing", "search"), sel.choice.tolist(), strict=True)
        if c >= 0
    }


def two_column_options(seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(0.1, 1, (8, 1, 2))), jnp.asarray(
        rng.uniform(1, 8, (8, 1, 2, 2))
    )


@pytest.mark.parametrize("stream", STREAMS)
def test_no_overspend_and_leftover_bounded_by_the_last_refusal(stream):
    """Fails if refreshes overspend or strand more than the last refused package."""
    spent, leftover, last_refused = simulate_refreshes(
        synthetic_stream(stream, seed=0), budget_d=100.0
    )
    assert spent <= 100.0 + 1e-9 and leftover <= last_refused + 1e-9
    assert spent > 0 and last_refused > 0


def test_selection_is_near_the_enumerated_optimum_on_a_branching_toy():
    """Fails if the price rule misses the optimum by more than one package's value."""
    gain, time = branching_toy_options(n_stars=8, seed=0)
    sel = select(gain, time, jnp.asarray([30.0]))
    assert chosen_value(sel, gain) >= brute_force_optimum(gain, time, 30.0) - float(
        gain.max()
    )


def test_an_infinite_time_variant_is_never_selected():
    """Fails if a mode that cannot reach its requirement is chosen."""
    gain, time = toy_options(n=4).with_infinite_time(star=0, option=0)
    sel = select(gain, time, jnp.asarray([1e6]))
    assert int(sel.choice[0]) == 1
    assert float(sel.selected_time[0]) == 4.0
    sel = select(gain, jnp.full_like(time, jnp.nan), jnp.asarray([1e6]))
    np.testing.assert_array_equal(sel.choice, [-1] * 4)
    np.testing.assert_array_equal(sel.selected_time, [0])


def test_finishing_packages_outrank_searches_when_the_budget_is_short():
    """Fails if a search starts with the days an open track needs."""
    gain, time = finishing_and_search_options()
    sel = select(gain, time, jnp.asarray([25.0]))
    assert chosen_kinds(sel) == {"finishing"}


def test_capacity_is_the_lesser_of_balance_and_days_left():
    """Fails if the price is set against days the mission no longer has."""
    cap = capacity_of(
        jnp.asarray([jnp.inf, jnp.inf, 80.0]), jnp.asarray(30.0), telescope_column=2
    )
    assert float(cap[2]) == 30.0 and bool(jnp.isinf(cap[0]))
    np.testing.assert_array_equal(
        capacity_of(
            jnp.asarray([4.0, 6.0, 20.0]), jnp.asarray(30.0), telescope_column=2
        ),
        [4, 6, 20],
    )


def test_two_binding_columns_are_each_respected():
    """Fails if a second resource column is ignored by the price."""
    gain, time = two_column_options(seed=0)
    sel = select(gain, time, jnp.asarray([20.0, 5.0]))
    assert bool(jnp.all(sel.selected_time <= jnp.asarray([20.0, 5.0]) + 1e-9))


def test_conflicting_variants_are_pruned_to_none():
    """Fails if coordinate repricing leaves an earlier column over capacity."""
    sel = select(
        jnp.asarray([[[10.0, 9.0]]]),
        jnp.asarray([[[[1.0, 10.0], [10.0, 1.0]]]]),
        jnp.asarray([5.0, 5.0]),
    )
    np.testing.assert_array_equal(sel.choice, [-1])
    np.testing.assert_array_equal(sel.selected_time, [0, 0])
    unpriced = select(
        jnp.asarray([[[10.0]], [[9.0]]]),
        jnp.asarray([[[[6.0, 1.0]]], [[[6.0, 1.0]]]]),
        jnp.asarray([6.0, 2.0]),
        sweeps=0,
    )
    np.testing.assert_array_equal(unpriced.choice, [0, -1])
    np.testing.assert_array_equal(unpriced.selected_time, [6, 1])


@pytest.mark.parametrize("capacity_d", [0.0, 5.0, 10.0])
def test_positive_net_and_none_ties(capacity_d):
    """Fails if an oversized package or a zero-gain option survives a price tie."""
    gain = jnp.asarray([[[1.0, 0.0]]])
    time = jnp.asarray([[[[10.0], [0.0]]]])
    sel = select(gain, time, jnp.asarray([capacity_d]))
    assert int(sel.choice[0]) == (0 if capacity_d == 10.0 else -1)
    assert float(sel.selected_time[0]) <= capacity_d
    if capacity_d == 5.0:
        assert float(sel.price[0]) == pytest.approx(0.1, abs=1e-14)


def test_zero_cost_columns_and_unbounded_resources():
    """Fails if zero costs break price bounds or unbounded columns get priced."""
    gain = jnp.asarray([[[1.0, 2.0]], [[0.0, -1.0]]])
    time = jnp.asarray(
        [[[[0.0, 2.0, 0.0], [0.0, 3.0, 10.0]]], [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
    )
    sel = select(gain, time, jnp.asarray([0.0, jnp.inf, 5.0]))
    np.testing.assert_array_equal(sel.choice, [0, -1])
    np.testing.assert_array_equal(sel.selected_time, [0, 2, 0])
    np.testing.assert_array_equal(sel.price[:2], [0, 0])


def test_dispatch_uses_named_resource_and_current_node():
    """Fails if dispatch ranks the wrong resource or ignores node eligibility."""
    gain = jnp.asarray([[[1.0], [4.0]], [[2.0], [3.0]]])
    time = jnp.asarray(
        [[[[0.0, 0.0, 1.0]], [[0.0, 0.0, 4.0]]], [[[0.0, 0.0, 5.0]], [[0.0, 0.0, 1.0]]]]
    )
    sel = Selection(
        price=jnp.zeros(3),
        choice=jnp.asarray([1, 1]),
        selected_time=jnp.asarray([0.0, 0.0, 5.0]),
    )
    dispatch = eqx.filter_jit(act)
    assert tuple(
        map(int, dispatch(sel, gain, time, jnp.asarray([1, 1]), ranking_resource=2))
    ) == (1, 1, 1)
    assert tuple(
        map(int, dispatch(sel, gain, time, jnp.asarray([1, 0]), ranking_resource=2))
    ) == (0, 1, 1)
    assert tuple(
        map(int, dispatch(sel, gain, time, jnp.asarray([0, 0]), ranking_resource=2))
    ) == (-1, -1, 0)
    sel = eqx.tree_at(lambda x: x.choice, sel, jnp.asarray([-1, -1]))
    assert tuple(
        map(int, dispatch(sel, gain, time, jnp.asarray([1, 1]), ranking_resource=2))
    ) == (-1, -1, 0)


def test_trace_diagnostics_use_balance_and_late_admission_evidence():
    """Fails if diagnostics infer balances or fund refusals from waits alone."""
    trace = jnp.asarray([[0.1, 20, 10, 30, 1, 0.5], [0.2, 20, 0, 10, 0, 0.5]])
    codes = jnp.asarray(
        [AdmissionCode.NO_FUNDS, AdmissionCode.NO_FUNDS, AdmissionCode.OK]
    )
    late = jnp.asarray([False, True, True])
    diagnostics = eqx.filter_jit(trace_scalars)
    for balance_d in (0.0, 80.0):
        early, stranded_d = diagnostics(
            trace,
            final_balance_d=jnp.asarray(balance_d),
            admission_code=codes,
            admission_late=late,
        )
        assert int(early) == 1 and float(stranded_d) == balance_d
    early, stranded_d = diagnostics(
        trace.at[:, 3].set(0),
        final_balance_d=jnp.asarray(80.0),
        admission_code=codes,
        admission_late=jnp.zeros(3, bool),
    )
    assert int(early) == 0 and float(stranded_d) == 0.0
