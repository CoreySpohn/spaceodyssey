"""Shared helpers for the survey tests (not a test file)."""

import dataclasses
from types import SimpleNamespace

import jax
import numpy as np

from spaceodyssey.codec import Bindings, dump_run, load_run
from spaceodyssey.core import CompiledExecutor, HostExecutor
from spaceodyssey.core.battery.views import DecisionViews
from spaceodyssey.core.enums import EventKind
from spaceodyssey.core.executors.base import GO_TOLERANCE, histories_equal, index_world

EXECUTORS = {"host": HostExecutor, "compiled": CompiledExecutor}


@dataclasses.dataclass(frozen=True)
class ActionRun:
    """A batched executor result and its complete requested effects per world."""

    result: object
    decisions: tuple


def run_with_actions(name, study, truths, roots, n_steps, **kw):
    """Capture executed decisions on either executor, including rejected requests."""
    if name == "host":
        views, effects = [], []
        run = run_on(
            name,
            study,
            truths,
            roots,
            n_steps,
            view_log=views,
            effects_log=effects,
            **kw,
        )
        sizes = (np.asarray(run.trace.handled) & np.asarray(run.trace.wake)).sum(axis=1)
        assert sum(sizes) == len(views) == len(effects)
        entries = tuple(zip(views, effects, strict=True))
        ends = np.cumsum(sizes)
        starts = np.r_[0, ends[:-1]]
        return ActionRun(
            run,
            tuple(entries[start:end] for start, end in zip(starts, ends, strict=True)),
        )
    capture = DecisionViews()
    run = run_on(name, capture.instrument(study), truths, roots, n_steps, **kw)
    decisions = tuple(
        tuple(capture.read(study, root, index_world(run.trace, w)))
        for w, root in enumerate(roots)
    )
    assert all(v is not None for world in decisions for v in world)
    return ActionRun(run, decisions)


def run_on(name, study, truths, roots, n_steps, **kw):
    """Run a batch of worlds on the named executor."""
    return EXECUTORS[name]().run_worlds(study, truths, roots, n_steps, **kw)


def same_histories(a, b, n_worlds):
    """Whether two batched runs have equal canonical histories in every world."""
    return all(histories_equal(a.history(w), b.history(w)).go for w in range(n_worlds))


def round_trip(
    study, registry, executor, kinds, *, bindings=None, service=None, refit=None
):
    """Dump `study` as a run document and load it back: `(document, run)`."""
    bindings = Bindings(()) if bindings is None else bindings
    doc = dump_run(
        study,
        registry,
        executor=executor,
        bindings=bindings,
        service=service,
        refit=refit,
        kinds=kinds,
    )
    return doc, load_run(doc.to_json(), bindings, kinds)


def redump(run, executor, kinds, *, bindings=None, refit=None):
    """The document of a loaded run, written again on `executor`."""
    return dump_run(
        run.study,
        run.registry,
        executor=executor,
        bindings=Bindings(()) if bindings is None else bindings,
        service=run.make_service(),
        refit=refit,
        kinds=kinds,
    )


def action_projection(run, world, measurement_kind=1):
    """What a policy did and what it got, with refresh steps and product rows left out.

    Each decision's tick, complete requested effects and admission codes; each
    measurement's subject, channel, acquisition index, start tick and payload, in log
    order. Returns `(decisions, acquisition_coordinates, payloads)`.
    """
    trace = index_world(run.result.trace, world)
    keep = np.asarray(trace.handled) & np.asarray(trace.wake)
    decisions = tuple(
        (
            int(t),
            tuple(tuple(np.asarray(x).flat) for x in jax.tree_util.tree_leaves(eff)),
            tuple(map(int, disruptive)),
            tuple(map(int, c)),
        )
        for t, (_, eff), disruptive, c in zip(
            np.asarray(trace.tick)[keep],
            run.decisions[world],
            np.asarray(trace.eff_disruptive)[keep],
            np.asarray(trace.admit_code)[keep],
            strict=True,
        )
    )
    log = index_world(run.result.carry.log, world)
    rows = np.asarray(log.valid) & (np.asarray(log.kind) == measurement_kind)
    ints = tuple(
        tuple(int(x) for x in row)
        for row in zip(
            np.asarray(log.subject)[rows],
            np.asarray(log.channel)[rows],
            np.asarray(log.acquisition_index)[rows],
            np.asarray(log.start_tick)[rows],
            strict=True,
        )
    )
    return decisions, ints, np.asarray(log.payload)[rows]


def same_actions(a, b, n_worlds):
    """Whether two batched runs took the same actions and read the same payloads."""
    for w in range(n_worlds):
        da, ia, pa = action_projection(a, w)
        db, ib, pb = action_projection(b, w)
        if da != db or ia != ib or pa.shape != pb.shape:
            return False
        if not np.allclose(pa, pb, rtol=0, atol=GO_TOLERANCE):
            return False
    return True


# Two-truth host runs and view comparisons for the trajectory surveys (22 and 24).


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
