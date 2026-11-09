"""A run killed with SIGKILL inside a checkpoint write resumes to the full run."""

import json
import signal
import subprocess
import sys

import jax
import numpy as np
import pytest

from spaceodyssey.codec.checkpoint import DirectoryCheckpoint
from spaceodyssey.core import HostExecutor
from spaceodyssey.core.executors.base import index_world
from tests.codec.run_cases import KINDS
from tests.core import refresh_helpers as rh
from tests.core.helpers import worlds

STEPS = 60
NTH = 3  # the publish the child dies in: some state already saved, more to do


def same_tree(a, b):
    same = jax.tree_util.tree_map(
        lambda x, y: bool(np.array_equal(np.asarray(x), np.asarray(y))), a, b
    )
    return all(jax.tree_util.tree_leaves(same))


def saved_steps(directory):
    """Step count in the manifest the pointer names, 0 when nothing was published."""
    try:
        pointer = json.loads((directory / "CURRENT").read_text())
    except FileNotFoundError:
        return 0
    gen = directory / f"gen-{pointer['generation']:06d}" / "manifest.json"
    return json.loads(gen.read_text())["steps"]


@pytest.fixture(scope="module")
def uninterrupted():
    study = rh.refresh_study(100)
    truths, roots = worlds(1)
    res = HostExecutor().run(
        study, index_world(truths, 0), roots[0], STEPS, refit=rh.masked()
    )
    return study, truths, roots, res


@pytest.mark.slow(reason="a child process compiles the host run, about 12 s per case")
@pytest.mark.parametrize("op", ["chunk", "manifest", "pointer-rename"])
def test_sigkill_in_a_write_resumes_to_the_uninterrupted_run(
    tmp_path, uninterrupted, op
):
    """Fails if a process killed at a durable write step leaves a checkpoint that
    cannot be resumed, or whose resume differs in any trace row or carry leaf from the
    run done in one call. Each case dies in the same publish, at a different step.
    """
    study, truths, roots, full = uninterrupted
    run_dir = tmp_path / "run"
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.core.resume_child_helpers",
            str(run_dir),
            op,
            str(NTH),
            str(STEPS),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert child.returncode == -signal.SIGKILL, child.stderr[-2000:]
    done = saved_steps(run_dir)
    assert 0 < done < STEPS
    rest = HostExecutor().run(
        study,
        index_world(truths, 0),
        roots[0],
        STEPS - done,
        refit=rh.masked(),
        checkpoint=DirectoryCheckpoint(run_dir, kinds=KINDS),
        resume=True,
    )
    assert same_tree(full.trace, rest.trace)
    assert same_tree(full.carry, rest.carry)
