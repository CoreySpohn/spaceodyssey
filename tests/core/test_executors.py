"""Both executors on survey_1.

Both executors on survey_1: hand histories, parity between scan and host, bookkeeping.
"""

import dataclasses

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from spaceodyssey import core
from spaceodyssey.core.enums import EventKind, Stage
from spaceodyssey.core.executors.base import (
    canonical_history,
    histories_equal,
    index_world,
)
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.studies import builder
from tests.core.corpus import HAND_HISTORIES, HAND_INVARIANTS, HAND_ROWS, HAND_STUDIES
from tests.core.helpers import history_of, worlds

N_STEPS = 40


def test_baseline_hand_history_both_executors():
    st = builder.survey_1()
    truths, roots = worlds(8)
    cs, ts = CompiledExecutor().run(st, truths, roots, N_STEPS)
    layout = st.observer.layout(st.caps)
    for w in range(8):
        h = canonical_history(index_world(cs, w), index_world(ts, w), layout)
        assert h.events == HAND_HISTORIES["baseline"]
        assert tuple(r[3:] for r in h.rows) == HAND_ROWS["baseline"]
        assert all(inv(h) for inv in HAND_INVARIANTS["baseline"])
        channel_index, th = HostExecutor().run(
            st, index_world(truths, w), roots[w], N_STEPS
        )
        assert histories_equal(h, canonical_history(channel_index, th, layout)).ok
        assert int(index_world(cs, w).n_dec) == 6 and h.n_records == 5


@pytest.mark.parametrize("name", sorted(HAND_STUDIES))
@pytest.mark.parametrize("form", ["array", "python"])
def test_hand_histories_match_across_executors_and_policy_forms(name, form):
    st = HAND_STUDIES[name]()
    truths, roots = worlds(3)
    cs, ts = CompiledExecutor().run(st, truths, roots, N_STEPS)
    layout = st.observer.layout(st.caps)
    for w in range(3):
        h = canonical_history(index_world(cs, w), index_world(ts, w), layout)
        assert h.events == HAND_HISTORIES[name], (name, w)
        # full rows: the extra columns (lane, subject, acquisition_index, n_written,
        # wake) per step
        assert tuple(r[3:] for r in h.rows) == HAND_ROWS[name], (name, w)
        assert all(inv(h) for inv in HAND_INVARIANTS[name]), (name, w)
        channel_index, th = HostExecutor().run(
            st, index_world(truths, w), roots[w], N_STEPS, policy_form=form
        )
        r = histories_equal(h, canonical_history(channel_index, th, layout))
        assert r.ok, (name, form, w, r)


def test_n_dec_equals_woken_steps():
    cs, ts = CompiledExecutor().run(builder.survey_1(), *worlds(4), N_STEPS)
    assert jnp.array_equal(cs.n_dec, ts.wake.sum(axis=1))


def test_histories_equal_detects_one_tick():
    h = history_of(builder.survey_1())
    ev = (*h.events[:3], (131, Stage.RELEASE, EventKind.RELEASE), *h.events[4:])
    g = dataclasses.replace(h, events=ev)
    assert not histories_equal(h, g).ok


def test_histories_equal_detects_a_float_difference():
    h = history_of(builder.survey_1())
    floats = dict(h.floats, payload=h.floats["payload"] + 1e-8)
    g = dataclasses.replace(h, floats=floats)
    assert histories_equal(h, g).max_abs > 1e-9 and not histories_equal(h, g).ok


def test_empty_run_is_flagged_vacuous():
    h = history_of(builder.survey_1(), n_steps=1)
    assert h.n_records == 0 and histories_equal(h, h).vacuous


def test_scan_worlds_differ_but_are_reproducible():
    st = builder.survey_1()
    truths, roots = worlds(4)
    c1, _ = CompiledExecutor().run(st, truths, roots, N_STEPS)
    c2, _ = CompiledExecutor().run(st, truths, roots, N_STEPS)
    assert jnp.array_equal(c1.log.payload, c2.log.payload)
    assert not jnp.array_equal(c1.log.payload[0], c1.log.payload[1])


def test_no_flag_trips_on_baseline():
    cs, _ = CompiledExecutor().run(builder.survey_1(), *worlds(4), N_STEPS)
    for name in (
        "overflow",
        "stalled",
        "violation",
        "double_settle",
        "dup_fold",
        "idle",
    ):
        assert not bool(jnp.any(getattr(cs, name))), name
    assert bool(jnp.all(cs.done))


def test_executors_are_compiled_once():
    traces = []

    def counting_wake(carry, ev):
        traces.append(1)
        return core.wake_by_audience(carry, ev)

    truths, roots = worlds(2)
    CompiledExecutor().run(builder.survey_1(), truths, roots, 12, wake_fn=counting_wake)
    n = len(traces)
    assert n >= 1
    CompiledExecutor().run(builder.survey_1(), truths, roots, 12, wake_fn=counting_wake)
    assert len(traces) == n  # equal study structure: no retrace


class _DrawingWait(core.AbstractPolicy):
    def decide(self, pstate, view, study, key):
        effects = core.effects_none(study.caps)
        effects = dataclasses.replace(
            effects,
            kind=effects.kind.at[0].set(
                jnp.where(view.n_dec == 0, core.EffectKind.WAIT, core.EffectKind.NONE)
            ),
            tick=effects.tick.at[0].set(200),
        )
        return effects, pstate, jr.uniform(key, dtype=jnp.float64)


@pytest.mark.parametrize("form", ["array", "python", "compiled"])
@pytest.mark.parametrize("shared", [False, True])
def test_policy_draws_respect_declared_phase_coupling(form, shared):
    coupling = core.Coupling(
        shared=((core.Purpose.POLICY, ("phase",)),) if shared else ()
    )
    studies = (
        builder.survey_1(policies=(_DrawingWait(),), coupling=coupling),
        builder.survey_1(
            policies=(_DrawingWait(), _DrawingWait()),
            phase_until=(100, 1000),
            coupling=coupling,
        ),
    )
    truths, roots = worlds(1)
    draws = []
    for study in studies:
        if form == "compiled":
            result = CompiledExecutor().run(study, truths, roots, 2)
            trace = index_world(result.trace, 0)
        else:
            result = HostExecutor().run(
                study, index_world(truths, 0), roots[0], 2, policy_form=form
            )
            trace = result.trace
        assert int(trace.tick[1]) == 200 and bool(trace.wake[1])
        draws.append(float(trace.pol_u[1]))
    assert (draws[0] == draws[1]) == shared


@pytest.mark.parametrize("field", ["micro", "sid", "acquisition_id"])
def test_canonical_history_detects_corrupted_identities(field):
    study = builder.survey_1()
    truths, roots = worlds(1)
    carry, trace = HostExecutor().run(study, index_world(truths, 0), roots[0], 20)
    layout = study.observer.layout(study.caps)
    original = canonical_history(carry, trace, layout)
    assert original.n_records == 5
    if field == "acquisition_id":
        log = dataclasses.replace(
            carry.log, acquisition_id=jnp.zeros_like(carry.log.acquisition_id)
        )
        altered = canonical_history(dataclasses.replace(carry, log=log), trace, layout)
    else:
        altered_trace = dataclasses.replace(
            trace, **{field: getattr(trace, field) + trace.handled.astype(jnp.int64)}
        )
        altered = canonical_history(carry, altered_trace, layout)
    assert histories_equal(original, original).go
    assert not histories_equal(original, altered).ok
    if field == "acquisition_id":
        np.testing.assert_array_equal(
            original.ints[field],
            np.asarray(carry.log.acquisition_id)[np.asarray(carry.log.valid)],
        )
