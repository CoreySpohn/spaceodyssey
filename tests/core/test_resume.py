"""Checkpoint and resume on both executors: a resumed run is the uninterrupted run."""

import jax
import numpy as np
import pytest

from spaceodyssey.codec.checkpoint import DirectoryCheckpoint
from spaceodyssey.core import CompiledExecutor, HostExecutor, RunStatus
from spaceodyssey.core.checkpoint import CheckpointCode, CheckpointError
from spaceodyssey.core.executors.base import (
    canonical_history,
    histories_equal,
    index_world,
)
from tests.codec.run_cases import KINDS
from tests.core import refresh_helpers as rh
from tests.core.helpers import worlds

CHUNK = 7
HOLD_BUDGET = 4  # ends with the first refresh checkpoint held, on both executors


def sink(path):
    return DirectoryCheckpoint(path / "run", kinds=KINDS)


def same_tree(a, b):
    same = jax.tree_util.tree_map(
        lambda x, y: bool(np.array_equal(np.asarray(x), np.asarray(y))), a, b
    )
    return all(jax.tree_util.tree_leaves(same))


def host_args():
    truths, roots = worlds(1)
    return rh.refresh_study(100), index_world(truths, 0), roots[0]


def compiled_args(n=2):
    truths, roots = worlds(n)
    return rh.refresh_study(100), truths, roots


def histories(result):
    return [
        canonical_history(
            index_world(result.carry, w), index_world(result.trace, w), result.layout
        )
        for w in range(result.trace.handled.shape[0])
    ]


@pytest.mark.medium(reason="host refresh run, about 8 s cold")
def test_host_resume_is_the_uninterrupted_run(tmp_path):
    """Fails if a host run split at a checkpoint differs in any trace row or carry
    leaf, bit for bit, from the same run done in one call.
    """
    study, truth, root = host_args()
    full = HostExecutor().run(study, truth, root, 60, refit=rh.masked())
    HostExecutor().run(
        study, truth, root, 25, refit=rh.masked(), checkpoint=sink(tmp_path)
    )
    rest = HostExecutor().run(
        study,
        truth,
        root,
        35,
        refit=rh.masked(),
        checkpoint=sink(tmp_path),
        resume=True,
    )
    assert same_tree(full.trace, rest.trace)
    assert same_tree(full.carry, rest.carry)


@pytest.mark.medium(reason="two compiled refresh programs, about 15 s cold")
def test_compiled_resume_is_the_chunked_run(tmp_path):
    """Fails if a compiled run split at a chunk boundary differs from the chunked run
    at the same chunk size in the final carry or in the canonical history of any
    world. Raw trace rows are not compared: the empty rows of a waiting chunk move.
    """
    study, truths, roots = compiled_args()
    full = CompiledExecutor().run(
        study,
        truths,
        roots,
        60,
        refit=rh.masked(),
        chunk=CHUNK,
        checkpoint=sink(tmp_path / "a"),
    )
    CompiledExecutor().run(
        study,
        truths,
        roots,
        25,
        refit=rh.masked(),
        chunk=CHUNK,
        checkpoint=sink(tmp_path / "b"),
    )
    rest = CompiledExecutor().run(
        study,
        truths,
        roots,
        35,
        refit=rh.masked(),
        chunk=CHUNK,
        checkpoint=sink(tmp_path / "b"),
        resume=True,
    )
    assert same_tree(full.carry, rest.carry)
    for a, b in zip(histories(full), histories(rest), strict=True):
        assert (
            histories_equal(a, b).max_ulps == 0.0
            and histories_equal(a, b).structure_equal
        )


@pytest.mark.medium(reason="host refresh run, about 8 s cold")
def test_held_refresh_is_answered_once_on_resume(tmp_path):
    """Fails if a run that stopped with a refresh checkpoint held does not report
    TRUNCATED_AT_HOLD, answers the hold before the budget ended, or answers it more
    or fewer times than the uninterrupted run does in total.
    """
    study, truth, root = host_args()
    whole = rh.SpyBoundary(rh.masked())
    HostExecutor().run(
        study, truth, root, 60, refit=whole, checkpoint=sink(tmp_path / "a")
    )
    first = rh.SpyBoundary(rh.masked())
    cut = HostExecutor().run(
        study,
        truth,
        root,
        HOLD_BUDGET,
        refit=first,
        checkpoint=sink(tmp_path / "b"),
    )
    assert int(cut.status) == RunStatus.TRUNCATED_AT_HOLD
    assert bool(cut.carry.refresh.valid)
    second = rh.SpyBoundary(rh.masked())
    HostExecutor().run(
        study,
        truth,
        root,
        60 - HOLD_BUDGET,
        refit=second,
        checkpoint=sink(tmp_path / "b"),
        resume=True,
    )
    assert len(second.calls) >= 1
    assert len(first.calls) + len(second.calls) == len(whole.calls)


@pytest.mark.medium(reason="compiled refresh program, about 10 s cold")
def test_compiled_hold_is_reported_and_answered_on_resume(tmp_path):
    """Fails if a compiled budget that ends at a hold does not report
    TRUNCATED_AT_HOLD for that world or the resume leaves the hold unanswered.
    """
    study, truths, roots = compiled_args(1)
    cut = CompiledExecutor().run(
        study,
        truths,
        roots,
        HOLD_BUDGET,
        refit=rh.masked(),
        chunk=CHUNK,
        checkpoint=sink(tmp_path),
    )
    assert int(cut.status[0]) == RunStatus.TRUNCATED_AT_HOLD
    rest = CompiledExecutor().run(
        study,
        truths,
        roots,
        20,
        refit=rh.masked(),
        chunk=CHUNK,
        checkpoint=sink(tmp_path),
        resume=True,
    )
    assert int(rest.status[0]) != RunStatus.TRUNCATED_AT_HOLD
    assert not bool(rest.carry.refresh.valid[0])


@pytest.mark.medium(reason="host refresh run, about 8 s cold")
def test_budget_without_a_sink_still_drops_the_hold():
    """Fails if a run with no sink keeps its hold or reports TRUNCATED_AT_HOLD: the
    new status must never be emitted without a checkpoint.
    """
    study, truth, root = host_args()
    res = HostExecutor().run(study, truth, root, HOLD_BUDGET, refit=rh.masked())
    assert res.held_by_budget is None
    assert int(res.status) != RunStatus.TRUNCATED_AT_HOLD
    assert not bool(res.carry.refresh.valid)


@pytest.mark.medium(reason="host refresh run, about 8 s cold")
def test_existing_checkpoint_needs_resume_and_the_same_run(tmp_path):
    """Fails if a populated directory is reused without `resume`, or resumed with
    different roots.
    """
    study, truth, root = host_args()
    HostExecutor().run(
        study, truth, root, 10, refit=rh.masked(), checkpoint=sink(tmp_path)
    )
    with pytest.raises(CheckpointError) as err:
        HostExecutor().run(
            study, truth, root, 10, refit=rh.masked(), checkpoint=sink(tmp_path)
        )
    assert err.value.code == CheckpointCode.C008
    with pytest.raises(CheckpointError) as err:
        HostExecutor().run(
            study,
            truth,
            root + 1,
            10,
            refit=rh.masked(),
            checkpoint=sink(tmp_path),
            resume=True,
        )
    assert err.value.code == CheckpointCode.C002
