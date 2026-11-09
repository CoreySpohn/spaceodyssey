"""Two-truth host runs and view comparisons for the trajectory surveys (not a test file)."""

from types import SimpleNamespace

import jax
import numpy as np

from spaceodyssey.core import HostExecutor
from spaceodyssey.core.enums import EventKind


def run_one(study, truth, root, n_steps):
    """A host run of one world, with every policy view handed out and its step."""
    views = []
    run = HostExecutor().run(study, truth, root, n_steps, view_log=views)
    trace = run.trace
    steps = np.flatnonzero(np.asarray(trace.handled) & np.asarray(trace.wake))
    assert len(steps) == len(views)
    return SimpleNamespace(run=run, truth=truth, steps=steps, views=views)


def first_step(result, kind, channel):
    """The first handled step of event `kind` on `channel`."""
    t = result.run.trace
    hit = (
        np.asarray(t.handled)
        & (np.asarray(t.kind) == kind)
        & (np.asarray(t.channel) == channel)
    )
    return int(np.flatnonzero(hit)[0])


def first_release(result, channel):
    """The step that first releases a record of `channel`."""
    return first_step(result, EventKind.RELEASE, channel)


def same_view(x, y):
    """Whether two policy views agree in every leaf, bit for bit."""
    leaves = zip(jax.tree.leaves(x), jax.tree.leaves(y), strict=True)
    return all(np.array_equal(p, q) for p, q in leaves)


def views_before(a, b, step):
    """Compare the policy views of two runs on every step before `step`.

    Returns `(fingerprints equal, full views equal, decisions compared)`. The
    fingerprint holds integer content only; the full comparison covers the floats.
    """
    fa, fb = (np.asarray(r.run.trace.snapshot_fp)[:step] for r in (a, b))
    early = [i for i, s in enumerate(a.steps) if s < step]
    full = all(same_view(a.views[i], b.views[i]) for i in early)
    return bool(np.array_equal(fa, fb)), full, len(early)
