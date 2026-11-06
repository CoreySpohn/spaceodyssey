"""The acquisition index counts admissions per subject; a cancel never frees one."""

import equinox as eqx
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from spaceodyssey import core
from spaceodyssey.core.enums import (
    AdmissionCode,
    Disposition,
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
from tests.core.acquisition_helpers import (
    ROOT,
    STUDY,
    TRUTH,
    drive,
    effect_table,
    new_settlements,
)
from tests.core.corpus import CORPUS
from tests.core.draw_helpers import draw_table, run_pair
from tests.core.helpers import (
    admit_j,
    after_decision_1_bundle,
    handle_event,
    observe_effect,
    worlds,
)
from tests.core.mutants import decrementing_advance_jit

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


# records a process or a host service writes take no admission index, so the cases built
# on them are judged by the dense-index invariant instead
ADMITTED = [c for c in CORPUS if c.name not in ("host_service", "revision")]


@pytest.mark.parametrize("case", ADMITTED, ids=lambda c: c.name)
def test_executors_agree_on_indices(case):
    study = case.make_study()
    truths, roots = worlds(1)
    cs, ts = CompiledExecutor().run(study, truths, roots, 40)
    layout = study.observer.layout(study.caps)
    scan = canonical_history(index_world(cs, 0), index_world(ts, 0), layout)
    host_carry, th = HostExecutor().run(study, index_world(truths, 0), roots[0], 40)
    host = canonical_history(host_carry, th, layout)
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
