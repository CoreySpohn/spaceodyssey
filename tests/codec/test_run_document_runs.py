"""B3: a reloaded run reproduces the original run's history on both executors."""

import pytest

from spaceodyssey.codec import Bindings, dump_run, load_run
from spaceodyssey.core import CompiledExecutor, HostExecutor
from spaceodyssey.core.conformance.corpus import make_worlds
from spaceodyssey.core.executors.base import histories_equal
from spaceodyssey.core.records import KindRegistry, PayloadColumn, RecordKind
from spaceodyssey.studies.conformance import CORPUS
from tests.codec import run_cases
from tests.codec.run_cases import (
    KINDS,
    bindings,
    boundary,
    host_loop,
    refresh_run,
)
from tests.codec.test_document import TOY_KINDS
from tests.core.helpers import worlds
from tests.core.refresh_helpers import CorruptInstall

N_WORLDS = 2
N_STEPS = 60
N_CASE_STEPS = 40


def _original(executor, refit):
    """The run the document describes, executed directly, not through the codec."""
    study, _, service = refresh_run()
    truths, roots = worlds(N_WORLDS)
    if executor == "host":
        exe = HostExecutor()
        return exe.run_worlds(
            study, truths, roots, N_STEPS, service=service, refit=refit
        )
    return CompiledExecutor().run(
        study, truths, roots, N_STEPS, service=service, refit=refit, chunk=7
    )


def _document(refit, **kw):
    study, registry, service = refresh_run()
    return dump_run(
        study,
        registry,
        executor="host",
        bindings=bindings(),
        service=service,
        refit=refit,
        kinds=KINDS,
        **kw,
    )


def _reloaded(doc, executor, **kw):
    run = load_run(doc.to_json(), bindings(), KINDS, **kw)
    truths, roots = worlds(N_WORLDS)
    return run.execute(truths, roots, N_STEPS, executor=executor)


@pytest.mark.parametrize("executor", ["host", "compiled"])
def test_a_reloaded_run_reproduces_the_history_on_each_executor(executor):
    doc = _document(boundary())
    mine = _original(executor, boundary())
    again = _reloaded(doc, executor)
    for w in range(N_WORLDS):
        assert histories_equal(mine.history(w), again.history(w)).ok


def test_the_reloaded_run_agrees_across_executors_with_each_its_own_boundary():
    doc = _document({"host": host_loop(), "compiled": boundary()})
    host = _reloaded(doc, "host")
    compiled = _reloaded(doc, "compiled")
    mine = _original("host", host_loop())
    for w in range(N_WORLDS):
        assert histories_equal(host.history(w), compiled.history(w)).ok
        assert histories_equal(mine.history(w), host.history(w)).ok


def test_a_reloaded_run_through_array_files_reproduces_the_history(tmp_path):
    doc = _document(boundary(), files=tmp_path, max_inline=0)
    assert any(tmp_path.iterdir())
    again = _reloaded(doc, "host", files=tmp_path)
    mine = _original("host", boundary())
    for w in range(N_WORLDS):
        assert histories_equal(mine.history(w), again.history(w)).ok


def test_the_comparison_sees_a_different_boundary():
    # control: a corrupted boundary changes the history
    mine = _original("host", CorruptInstall(boundary()))
    again = _reloaded(_document(boundary()), "host")
    assert not all(
        histories_equal(mine.history(w), again.history(w)).ok for w in range(N_WORLDS)
    )


# ---- every corpus study ---------------------------------------------


def _registry_of(caps):
    """A registry that lowers to `caps` (column names are made up: caps keep none)."""
    kinds = []
    for i, name in enumerate(caps.kind_names):
        flags = [a for a in caps.payload_audience[i] if a]
        columns = tuple(PayloadColumn(f"c{j}", a) for j, a in enumerate(flags))
        kinds.append(RecordKind(name, columns, derived=i in caps.derived_kinds))
    return KindRegistry(kinds)


def _service_binding():
    return Bindings((run_cases._service_kind(),))


@pytest.mark.parametrize("executor", ["host", "compiled"])
@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.name)
def test_every_corpus_run_reloads_to_the_same_hash_and_history(case, executor):
    study = case.make_study()
    registry = _registry_of(study.caps)
    assert int(study.refresh.every_ticks) == 0  # no case needs a refit boundary
    service = None if case.make_service is None else case.make_service()
    table = _service_binding()
    doc = dump_run(
        study,
        registry,
        executor=executor,
        bindings=table,
        service=service,
        kinds=TOY_KINDS,
    )
    run = load_run(doc.to_json(), table, TOY_KINDS)
    again = dump_run(
        run.study,
        run.registry,
        executor=run.executor,
        bindings=table,
        service=run.make_service(),
        kinds=TOY_KINDS,
    )
    assert again.content_hash == doc.content_hash
    truths, roots = make_worlds(study, 2)
    exe = HostExecutor() if executor == "host" else CompiledExecutor()
    mine = exe.run_worlds(
        study,
        truths,
        roots,
        N_CASE_STEPS,
        service=None if case.make_service is None else case.make_service(),
    )
    reloaded = run.execute(truths, roots, N_CASE_STEPS)
    for w in range(2):
        assert histories_equal(mine.history(w), reloaded.history(w)).ok
