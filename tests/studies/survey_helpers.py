"""Shared helpers for the survey tests (not a test file)."""

import numpy as np

from spaceodyssey.codec import Bindings, dump_run, load_run
from spaceodyssey.core import CompiledExecutor, HostExecutor
from spaceodyssey.core.executors.base import GO_TOLERANCE, histories_equal, index_world

EXECUTORS = {"host": HostExecutor, "compiled": CompiledExecutor}


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

    Each decision's tick, effect kinds and admission codes, in order; each measurement
    acquisition's subject, channel, acquisition index, start tick and payload, in log
    order. Returns `(decisions, acquisitions)`: tuples of ints, and arrays of floats.
    """
    trace = index_world(run.trace, world)
    keep = np.asarray(trace.handled) & np.asarray(trace.wake)
    decisions = tuple(
        (int(t), tuple(map(int, k)), tuple(map(int, c)))
        for t, k, c in zip(
            np.asarray(trace.tick)[keep],
            np.asarray(trace.eff_kind)[keep],
            np.asarray(trace.admit_code)[keep],
            strict=True,
        )
    )
    log = index_world(run.carry.log, world)
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
