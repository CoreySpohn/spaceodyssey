"""Host checkpoint and resume: a resumed run is the uninterrupted run.

One study (the oracle study on a refresh grid) serves every test, and the uninterrupted
reference run and the saved checkpoints several tests resume from are built once per
module, so the host loop compiles once. The compiled executor's cases are in
`test_resume_compiled.py`.
"""

import dataclasses
import shutil

import jax
import pytest

from spaceodyssey.codec.checkpoint import GenerationStore
from spaceodyssey.core import HostExecutor, RunStatus
from spaceodyssey.core.checkpoint import CheckpointCode
from spaceodyssey.core.executors.base import index_world
from tests.core import refresh_helpers as rh
from tests.core.resume_helpers import (
    CUT,
    HOLD_BUDGET,
    TOTAL,
    exact_history,
    oracle_args,
    refused,
    same_tree,
    sink,
)


@pytest.fixture(scope="module")
def args():
    return oracle_args()


def host_run(args, steps, *, sink=None, resume=False, service=True, refit=None, **kw):
    study, svc, truths, roots = args
    return HostExecutor().run(
        study,
        index_world(truths, 0),
        roots[0],
        steps,
        service=svc if service else None,
        refit=refit or rh.masked(),
        checkpoint=sink,
        resume=resume,
        **kw,
    )


@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    return tmp_path_factory.mktemp("resume")


@pytest.fixture(scope="module")
def host_full(args):
    """The uninterrupted host run, with the number of refits it asked for."""
    spy = rh.SpyBoundary(rh.masked())
    return host_run(args, TOTAL, refit=spy), len(spy.calls)


@pytest.fixture(scope="module")
def host_split(args, dirs):
    """Saved after CUT steps, then resumed for the rest, in `dirs / "host"`."""
    path = dirs / "host"
    host_run(args, CUT, sink=sink(path))
    return host_run(args, TOTAL - CUT, sink=sink(path), resume=True), path


@pytest.mark.medium(reason="host refresh run with a service, compiled once")
def test_host_resume_is_the_uninterrupted_run(host_full, host_split):
    """Fails if a host run split at a checkpoint differs in any trace row or carry
    leaf, bit for bit, from the same run done in one call, or if its manifest does not
    count every step of the run after the resume (a later resume would then continue
    with the wrong budget).
    """
    full, _ = host_full
    rest, path = host_split
    assert same_tree(full.trace, rest.trace)
    assert same_tree(full.carry, rest.carry)
    manifest = GenerationStore(path / "run").load().manifest
    assert manifest["rows"] == TOTAL and manifest["steps"] == TOTAL


@pytest.mark.medium(reason="host runs of 4 and 36 steps on the shared program")
def test_held_refresh_is_answered_once_on_resume(args, dirs, host_full):
    """Fails if a run that stopped with a refresh checkpoint held does not report
    TRUNCATED_AT_HOLD, answers the hold before the budget ended, completes it
    differently from the uninterrupted run (the resumed hold must finish inside the
    new budget), or answers it more or fewer times than the uninterrupted run does in
    total. A run with no sink drops the same hold and never reports the new status.
    """
    full, calls = host_full
    first = rh.SpyBoundary(rh.masked())
    cut = host_run(args, HOLD_BUDGET, sink=sink(dirs / "hold"), refit=first)
    assert int(cut.status) == RunStatus.TRUNCATED_AT_HOLD
    assert bool(cut.carry.refresh.valid)
    second = rh.SpyBoundary(rh.masked())
    rest = host_run(
        args,
        TOTAL - HOLD_BUDGET,
        sink=sink(dirs / "hold"),
        resume=True,
        refit=second,
    )
    assert same_tree(full.trace, rest.trace)
    assert same_tree(full.carry, rest.carry)
    assert len(second.calls) >= 1
    assert len(first.calls) + len(second.calls) == calls
    plain = host_run(args, HOLD_BUDGET)
    assert plain.held_by_budget is None
    assert int(plain.status) != RunStatus.TRUNCATED_AT_HOLD
    assert not bool(plain.carry.refresh.valid)


@pytest.mark.medium(reason="host runs with and without a service on the shared program")
def test_host_resume_answers_a_pending_request_first(args, dirs, host_full):
    """Fails if a host resume does not answer the request the saved run was waiting
    on before it steps: the resumed run must equal the uninterrupted one.
    """
    full, _ = host_full
    cut = host_run(args, CUT, sink=sink(dirs / "hpend"), service=False)
    assert int(cut.status) == RunStatus.AWAITING and bool(cut.carry.request.valid)
    rest = host_run(args, TOTAL - CUT, sink=sink(dirs / "hpend"), resume=True)
    assert same_tree(full.carry, rest.carry)
    assert exact_history(full, rest)


@pytest.mark.medium(reason="refusals happen before any step, on the shared program")
def test_resume_refuses_a_changed_run_or_a_missing_checkpoint(
    args, host_split, tmp_path
):
    """Fails if a checkpoint is resumed against a different truth, study, key version,
    boundary, kept log or roots (each is part of the run identity), if a populated
    directory is reused without `resume`, or if `resume` on an empty directory starts
    fresh.
    """
    study, service, truths, roots = args
    shutil.copytree(host_split[1], tmp_path / "h")
    saved = sink(tmp_path / "h")

    def resume(args=args, **kw):
        return host_run(args, 5, sink=sink(tmp_path / "h"), resume=True, **kw)

    other_truth = jax.tree_util.tree_map(lambda x: x + 1, truths)
    refused(CheckpointCode.C002, lambda: resume((study, service, other_truth, roots)))
    changed, _ = rh.oracle_refresh_study(140)
    refused(CheckpointCode.C002, lambda: resume((changed, service, truths, roots)))
    newer = dataclasses.replace(study, key_version=study.key_version + 1)
    err = refused(CheckpointCode.C002, lambda: resume((newer, service, truths, roots)))
    assert "key_version" in err.message
    refused(CheckpointCode.C002, lambda: resume(refit=rh.host_loop()))
    # a log the saved run did not keep is a run the saved rows do not match
    refused(CheckpointCode.C002, lambda: resume(view_log=[]))
    refused(CheckpointCode.C002, lambda: resume((study, service, truths, roots + 1)))
    refused(CheckpointCode.C008, lambda: host_run(args, 5, sink=saved))
    refused(
        CheckpointCode.C010,
        lambda: host_run(args, 5, sink=sink(tmp_path / "none"), resume=True),
    )
