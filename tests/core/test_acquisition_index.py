"""The acquisition index counts admissions per subject; a cancel never frees one."""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from spaceodyssey import core
from spaceodyssey.core.enums import (
    AdmissionCode,
    Disposition,
    EffectKind,
    EventKind,
    RecordStatus,
    Scope,
)
from spaceodyssey.core.executors.base import (
    canonical_history,
    index_world,
)
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.studies import toy as kinds
from tests.core.corpus import CORPUS
from tests.core.draw_helpers import draw_table, run_pair
from tests.core.helpers import (
    CAPS,
    admit_j,
    advance_j,
    after_decision_1_bundle,
    handle_event,
    observe_effect,
    one_world,
    worlds,
)
from tests.core.mutants import decrementing_advance_jit

STUDY = kinds.survey_1()
TRUTH, ROOT = one_world()
N_SUBJECTS = 2

# ---- a driver that feeds random effects straight into admission ----------------


def effect_table(op):
    """The effect table of one op: observe, group, cancel, wait or stop."""
    e = core.effects_none(CAPS)
    kind = op[0]
    if kind == "observe":
        _, subject, ch = op
        return dataclasses.replace(
            e,
            kind=e.kind.at[0].set(EffectKind.OBSERVE),
            subject=e.subject.at[0].set(subject),
            channel_index=e.channel_index.at[0].set(ch),
        )
    if kind == "group":
        _, subject_a, ch_a, subject_b, ch_b = op
        return dataclasses.replace(
            e,
            kind=jnp.full(CAPS.n_effects, EffectKind.OBSERVE, jnp.int64),
            subject=jnp.asarray([subject_a, subject_b], jnp.int64),
            channel_index=jnp.asarray([ch_a, ch_b], jnp.int64),
            group=jnp.asarray([3, 3], jnp.int64),
        )
    if kind == "cancel":
        return dataclasses.replace(
            e,
            kind=e.kind.at[0].set(EffectKind.CANCEL),
            lane=e.lane.at[0].set(op[1]),
        )
    if kind == "wait":
        return dataclasses.replace(
            e, kind=e.kind.at[0].set(EffectKind.WAIT), tick=e.tick.at[0].set(20)
        )
    _, scope, disposition = op
    return dataclasses.replace(
        e,
        kind=e.kind.at[0].set(EffectKind.STOP),
        lane=e.lane.at[0].set(1),
        scope=e.scope.at[0].set(scope),
        disposition=e.disposition.at[0].set(disposition),
    )


def new_settlements(before, after):
    """(subject, index) of the Done rows queued between two carries, in sid order."""
    ev = after.ev
    fresh = np.asarray(
        ev.valid & (ev.kind == EventKind.DONE) & (ev.sid >= before.next_sid)
    )
    order = np.argsort(np.asarray(ev.sid)[fresh])
    subjects = np.asarray(ev.subject)[fresh][order].tolist()
    indices = np.asarray(ev.acquisition_index)[fresh][order].tolist()
    return list(zip(subjects, indices, strict=True))


def drive(ops, advance=advance_j, steps=()):
    """Admit each op, then advance a few events; return the violations found."""
    c = eqx.filter_jit(core.init_carry)(STUDY, TRUTH)
    problems, admitted = [], {s: [] for s in range(N_SUBJECTS)}
    for n, op in enumerate(ops):
        before = c
        c, _codes = admit_j(c, effect_table(op), STUDY)
        counts_before = np.asarray(before.acquisition_counts)
        counts_after = np.asarray(c.acquisition_counts)
        fresh = new_settlements(before, c)
        expected = counts_before.copy()
        for subject, index in fresh:
            if index != expected[subject]:
                problems.append((n, "index", subject, index, int(expected[subject])))
            expected[subject] += 1
            admitted[subject].append(index)
        if not np.array_equal(counts_after, expected):
            problems.append((n, "counts", counts_after.tolist(), expected.tolist()))
        for _ in range(steps[n] if n < len(steps) else 1):
            if not bool(c.ev.valid.any()):
                break
            counts_pre = np.asarray(c.acquisition_counts)
            c, _, _ = advance(c, ROOT, STUDY, TRUTH, wake_fn=core.wake_by_audience)
            if np.any(np.asarray(c.acquisition_counts) < counts_pre):
                problems.append((n, "decrement", counts_pre.tolist()))
    for subject, got in admitted.items():
        if got != list(range(len(got))):
            problems.append(("dense", subject, got))
    log = c.log
    rows = [
        (int(log.subject[i]), int(log.acquisition_index[i]))
        for i in range(int(c.n_log))
        if int(log.status[i]) != RecordStatus.FAILED
    ]
    if len(rows) != len(set(rows)):
        problems.append(("duplicate record index", rows))
    return problems


op_strategy = st.one_of(
    st.tuples(st.just("observe"), st.integers(0, 1), st.integers(0, 2)),
    st.tuples(
        st.just("group"),
        st.integers(0, 1),
        st.integers(0, 2),
        st.integers(0, 1),
        st.integers(0, 2),
    ),
    st.tuples(st.just("cancel"), st.integers(0, 2)),
    st.tuples(st.just("wait")),
    st.tuples(
        st.just("stop"),
        st.sampled_from([Scope.LANE, Scope.ASSET]),
        st.sampled_from([Disposition.CLOSE, Disposition.ABORT]),
    ),
)


@settings(max_examples=40, deadline=None, derandomize=True)
@given(
    ops=st.lists(op_strategy, min_size=1, max_size=8),
    steps=st.lists(st.integers(0, 4), min_size=8, max_size=8),
)
def test_indices_are_dense_and_ordered(ops, steps):
    assert drive(ops, steps=steps) == []


# ---- direct cases ------------------------------------------------------------------


def test_cancel_keeps_its_index():
    study, c = after_decision_1_bundle()
    c = handle_event(
        c, study, TRUTH, ROOT, EventKind.CANCEL, lane=0, gen=int(c.ledger.lane_gen[0])
    )
    partial = [
        (int(c.log.subject[i]), int(c.log.acquisition_index[i]))
        for i in range(int(c.n_log))
        if int(c.log.status[i]) == RecordStatus.PARTIAL
    ]
    assert partial == [(1, 0)]
    before = int(c.acquisition_counts[1])
    c2, codes = admit_j(c, observe_effect(1, 10), study)
    assert int(codes[0]) == AdmissionCode.OK
    assert int(c2.acquisition_counts[1]) == before + 1
    assert new_settlements(c, c2) == [(1, before)]
    assert before >= 1  # the cancelled acquisition still counts


def test_refusal_and_group_failure_consume_nothing():
    c0 = eqx.filter_jit(core.init_carry)(STUDY, TRUTH)
    busy = c0
    for op in (("observe", 0, 0), ("observe", 0, 0)):  # the second finds lane 0 busy
        before = busy
        busy, codes = admit_j(busy, effect_table(op), STUDY)
    assert int(codes[0]) == AdmissionCode.LANE_BUSY
    assert np.array_equal(busy.acquisition_counts, before.acquisition_counts)
    # a group whose second row meets the busy lane takes no index either
    after, codes = admit_j(busy, effect_table(("group", 1, 1, 1, 0)), STUDY)
    assert [int(x) for x in codes] == [AdmissionCode.LANE_BUSY] * 2
    assert np.array_equal(after.acquisition_counts, busy.acquisition_counts)


def test_draw_follows_the_index():
    base, other = run_pair("cancel_reobserve", "scan", n_seeds=2)
    for b, o in zip(base, other, strict=True):
        tb, to = draw_table(b), draw_table(o)
        subject0 = [k for k in tb.keys() & to.keys() if k[0] == 0]
        assert subject0
        # an earlier cancelled acquisition of subject 1 moves no draw of subject 0
        assert all(tb[k] == to[k] for k in subject0)
        # consecutive indices of one subject draw different variates
        by_subject = {}
        for (subject, index, *_), value in to.items():
            by_subject.setdefault(subject, {})[index] = value
        for values in by_subject.values():
            ordered = [values[i] for i in sorted(values)]
            assert len(set(ordered)) == len(ordered)


@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.name)
def test_executors_agree_on_indices(case):
    study = case.make_study()
    truths, roots = worlds(1)
    cs, ts = CompiledExecutor().run(study, truths, roots, 40)
    scan = canonical_history(index_world(cs, 0), index_world(ts, 0))
    host_carry, th = HostExecutor().run(study, index_world(truths, 0), roots[0], 40)
    host = canonical_history(host_carry, th)
    for key in ("acquisition_index", "subject", "acquisition_counts"):
        assert np.array_equal(scan.ints[key], host.ints[key]), (case.name, key)
    per_subject = {}
    for subject, index in zip(
        scan.ints["subject"], scan.ints["acquisition_index"], strict=False
    ):
        per_subject.setdefault(int(subject), []).append(int(index))
    for subject, got in per_subject.items():
        assert len(got) == len(set(got)), (case.name, subject)
        assert max(got) < int(scan.ints["acquisition_counts"][subject])


def test_decrement_mutant_is_caught():
    ops = [("observe", 0, 0), ("cancel", 0), ("observe", 0, 0)]
    assert drive(ops, steps=[0, 3, 0]) == []
    problems = drive(ops, advance=decrementing_advance_jit, steps=[0, 3, 0])
    assert problems, "a cancel that frees its index must be detected"
