"""The six statuses, the five identities and the provenance of a record."""

import numpy as np
import pytest

from spaceodyssey.core import (
    CompiledExecutor,
    EventKind,
    HostExecutor,
    index_world,
    replay_belief,
    snapshot_view,
)
from spaceodyssey.core.enums import Audience, RecordStatus
from spaceodyssey.studies import toy
from tests.core.helpers import N_STEPS_CORPUS, baseline_host_run, worlds
from tests.core.record_helpers import scripted_study

STATUSES = (
    RecordStatus.COMPLETED,
    RecordStatus.FAILED,
    RecordStatus.PARTIAL,
    RecordStatus.CANCELLED,
    RecordStatus.NULL,
    RecordStatus.UNRELEASED,
)


def test_there_are_six_statuses():
    assert [int(s) for s in RecordStatus] == [0, 1, 2, 3, 4, 5]
    assert {s.name for s in RecordStatus} == {
        "COMPLETED", "FAILED", "PARTIAL", "CANCELLED", "NULL", "UNRELEASED"
    }  # fmt: skip


@pytest.fixture(scope="module")
def scripted_run():
    study = scripted_study(
        ticks=(10, 20, 30, 40, 50, 60),
        values=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        ids=(7001, 7002, 7003, 7004, 7005, 7006),
        statuses=tuple(int(s) for s in STATUSES),
    )
    truths, roots = worlds(1)
    carry, trace = HostExecutor().run(
        study, index_world(truths, 0), roots[0], N_STEPS_CORPUS
    )
    return study, carry, trace, truths, roots


@pytest.mark.medium  # over 3 s cold here, about 5 s on the runner
def test_every_status_is_logged_and_only_completed_values_are_folded(scripted_run):
    _study, carry, _, _, _ = scripted_run
    log = carry.log
    n = int(carry.n_log)
    assert n == 6
    assert [int(x) for x in np.asarray(log.status)[:n]] == [int(s) for s in STATUSES]
    # the running mean takes completed measurements only; null, failed and the rest
    # are records the observer accepts without folding a value
    assert int(carry.belief.n[0]) == 1 and float(carry.belief.mean[0]) == 1.0
    assert int(carry.pbelief.n[0]) == 1


def test_a_held_record_is_never_released_or_accepted(scripted_run):
    study, carry, _, _, _ = scripted_run
    held = int(RecordStatus.UNRELEASED)
    row = np.asarray(carry.log.status)[:6] == held
    assert row.sum() == 1
    assert np.all(np.asarray(carry.log.released_tick)[:6][row] == -1)
    assert np.all(np.asarray(carry.log.accepted_tick)[:6][row] == -1)
    assert np.all(np.asarray(carry.log.released_tick)[:6][~row] >= 0)
    for audience in (Audience.POLICY, Audience.OBSERVER):
        view = snapshot_view(carry, audience, study.caps)
        assert int(view.n_rows) == 5
        assert held not in np.asarray(view.log.status)[:5]


def test_null_is_released_like_any_completed_row(scripted_run):
    study, carry, _, _, _ = scripted_run
    view = snapshot_view(carry, Audience.POLICY, study.caps)
    assert int(RecordStatus.NULL) in np.asarray(view.log.status)[: int(view.n_rows)]


@pytest.mark.medium  # Cold host, compiled and replay programs exceed the unit budget.
def test_statuses_agree_between_executors_and_replay(scripted_run):
    study, carry, trace, truths, roots = scripted_run
    cs, _ = CompiledExecutor().run(study, truths, roots, N_STEPS_CORPUS)
    compiled = index_world(cs, 0)
    assert np.array_equal(np.asarray(compiled.log.status), np.asarray(carry.log.status))
    replayed = replay_belief(study, carry, trace)
    assert np.array_equal(np.asarray(replayed.n), np.asarray(carry.belief.n))
    assert np.array_equal(np.asarray(replayed.mean), np.asarray(carry.belief.mean))


def test_identities_and_provenance_of_a_settled_record():
    study, run, _ = baseline_host_run()
    carry, trace = run
    n = int(carry.n_log)
    log = carry.log
    ids = np.asarray(log.acquisition_id)[:n]
    assert len(set(ids.tolist())) == n  # one acquisition id per admission
    assert np.all(np.asarray(log.revision)[:n] == 0)
    assert np.all(np.asarray(log.superseded_by)[:n] == -1)
    released = np.asarray(log.released_tick)[:n]
    assert np.all(released >= np.asarray(log.available_tick)[:n])
    assert np.array_equal(np.asarray(log.accepted_tick)[:n], released)
    # provenance: the channel, the phase at admission and the servicing hook
    assert np.array_equal(np.asarray(log.source)[:n], np.asarray(log.channel)[:n])
    assert np.array_equal(np.asarray(log.vintage)[:n], np.asarray(log.phase)[:n])
    assert np.all(np.asarray(log.config_id)[:n] == 0)
    # the audience comes from the channel
    index = np.asarray(log.channel_index)[:n]
    assert np.array_equal(
        np.asarray(log.audience)[:n], np.asarray(study.channels.audience)[index]
    )
    handled = np.asarray(trace.handled)
    assert int((np.asarray(trace.kind)[handled] == EventKind.RELEASE).sum()) >= n
    assert toy.RECORD_MEASUREMENT in np.asarray(log.kind)[:n]
