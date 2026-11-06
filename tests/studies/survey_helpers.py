"""Shared helpers for the survey tests (not a test file)."""

from spaceodyssey.codec import Bindings, dump_run, load_run
from spaceodyssey.core import CompiledExecutor, HostExecutor
from spaceodyssey.core.executors.base import histories_equal

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
