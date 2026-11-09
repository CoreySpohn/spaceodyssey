"""Compiled checkpoint and resume: a resumed run is the chunked run at the same chunk.

The same study as `test_resume.py`. These cases compile the chunked driver for a batch
of worlds, which costs more than the host cases, so they run in the slow tier.
"""

import shutil

import numpy as np
import pytest

from spaceodyssey.core import CompiledExecutor, RunStatus
from spaceodyssey.core.checkpoint import CheckpointCode
from tests.core import refresh_helpers as rh
from tests.core.resume_helpers import (
    CHUNK,
    CUT,
    HOLD_BUDGET,
    N_WORLDS,
    TOTAL,
    exact_history,
    oracle_args,
    refused,
    same_tree,
    sink,
    world,
)

pytestmark = pytest.mark.slow(
    reason="compiles the chunked driver for a batch, about 25 s cold"
)


@pytest.fixture(scope="module")
def args():
    return oracle_args()


def compiled_run(
    args, steps, *, sink=None, resume=False, service=True, chunk=CHUNK, **kw
):
    study, svc, truths, roots = args
    return CompiledExecutor().run(
        study,
        truths,
        roots,
        steps,
        service=svc if service else None,
        refit=rh.masked(),
        chunk=chunk,
        checkpoint=sink,
        resume=resume,
        **kw,
    )


@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    return tmp_path_factory.mktemp("resume_compiled")


@pytest.fixture(scope="module")
def compiled_full(args):
    return compiled_run(args, TOTAL)


@pytest.fixture(scope="module")
def compiled_split(args, dirs):
    path = dirs / "split"
    compiled_run(args, CUT, sink=sink(path))
    return compiled_run(args, TOTAL - CUT, sink=sink(path), resume=True), path


def test_compiled_resume_is_the_chunked_run(compiled_full, compiled_split):
    """Fails if a compiled run split at a chunk boundary differs from the chunked run
    at the same chunk size in the final carry or in the canonical history of any
    world. Raw trace rows are not compared: the empty rows of a waiting chunk move.
    """
    rest, _ = compiled_split
    assert same_tree(compiled_full.carry, rest.carry)
    for w in range(N_WORLDS):
        assert exact_history(world(compiled_full, w), world(rest, w))


def test_compiled_hold_is_reported_and_answered_on_resume(args, dirs, compiled_full):
    """Fails if a compiled budget that ends at a hold does not report
    TRUNCATED_AT_HOLD for each world, or the resume leaves a hold unanswered or ends
    differently from the chunked run.
    """
    cut = compiled_run(args, HOLD_BUDGET, sink=sink(dirs / "hold"))
    assert (np.asarray(cut.status) == RunStatus.TRUNCATED_AT_HOLD).all()
    rest = compiled_run(
        args, TOTAL - HOLD_BUDGET, sink=sink(dirs / "hold"), resume=True
    )
    assert not np.asarray(rest.carry.refresh.valid).any()
    assert same_tree(compiled_full.carry, rest.carry)


def test_compiled_resume_answers_a_pending_request_first(args, dirs, compiled_full):
    """Fails if a compiled resume does not answer the request the saved run was
    waiting on before its first chunk.
    """
    cut = compiled_run(args, CUT, sink=sink(dirs / "pend"), service=False)
    assert (np.asarray(cut.status) == RunStatus.AWAITING).all()
    rest = compiled_run(args, TOTAL - CUT, sink=sink(dirs / "pend"), resume=True)
    assert same_tree(compiled_full.carry, rest.carry)
    for w in range(N_WORLDS):
        assert exact_history(world(compiled_full, w), world(rest, w))
    # answered before the first chunk: that chunk steps, it is not an empty wait
    saved_rows = cut.trace.handled.shape[1]
    assert np.asarray(rest.trace.handled)[:, saved_rows : saved_rows + CHUNK].any()


def test_compiled_resume_refuses_another_chunk_size_and_a_bucket(
    args, compiled_split, tmp_path
):
    """Fails if a checkpoint resumes at a different chunk size (part of the run
    identity), or a bucket is silently ignored beside a checkpoint.
    """
    shutil.copytree(compiled_split[1], tmp_path / "c")
    refused(
        CheckpointCode.C002,
        lambda: compiled_run(
            args, 5, sink=sink(tmp_path / "c"), resume=True, chunk=CHUNK + 1
        ),
    )
    with pytest.raises(ValueError, match="bucket"):
        compiled_run(args, 5, sink=sink(tmp_path / "b"), bucket=8)
