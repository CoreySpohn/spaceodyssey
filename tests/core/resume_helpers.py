"""Shared pieces of the resume tests (not a test file)."""

import dataclasses

import jax
import numpy as np
import pytest

from spaceodyssey.codec.checkpoint import DirectoryCheckpoint
from spaceodyssey.core.checkpoint import CheckpointError
from spaceodyssey.core.executors.base import (
    canonical_history,
    histories_equal,
    index_world,
)
from tests.codec.run_cases import KINDS
from tests.core import refresh_helpers as rh
from tests.core.helpers import worlds

CHUNK = 7
TOTAL, CUT = 40, 12  # the run, and where the split runs stop
HOLD_BUDGET = 4  # ends with the first refresh checkpoint held, on both executors
N_WORLDS = 2


def sink(path):
    return DirectoryCheckpoint(path / "run", kinds=KINDS)


def same_tree(a, b):
    same = jax.tree_util.tree_map(
        lambda x, y: bool(np.array_equal(np.asarray(x), np.asarray(y))), a, b
    )
    return all(jax.tree_util.tree_leaves(same))


def exact_history(a, b):
    """Whether two one-world results have the same canonical history, to the bit."""
    ha, hb = (canonical_history(r.carry, r.trace, r.layout) for r in (a, b))
    rep = histories_equal(ha, hb)
    return rep.structure_equal and rep.max_ulps == 0.0


def world(result, w):
    return dataclasses.replace(
        result,
        carry=index_world(result.carry, w),
        trace=index_world(result.trace, w),
    )


def refused(code, run):
    with pytest.raises(CheckpointError) as err:
        run()
    assert err.value.code == code
    return err.value


def oracle_args():
    """The study every resume test runs, its service, the truths and the roots."""
    study, service = rh.oracle_refresh_study(130)
    truths, roots = worlds(N_WORLDS)
    return study, service, truths, roots
