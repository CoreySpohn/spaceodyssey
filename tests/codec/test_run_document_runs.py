"""B3: a reloaded run reproduces the original run's history on both executors."""

import pytest

from spaceodyssey.codec import dump_run, load_run
from spaceodyssey.core import CompiledExecutor, HostExecutor
from spaceodyssey.core.executors.base import histories_equal
from tests.codec.run_cases import (
    KINDS,
    bindings,
    boundary,
    host_loop,
    refresh_run,
)
from tests.core.helpers import worlds
from tests.core.refresh_helpers import CorruptInstall

N_WORLDS = 2
N_STEPS = 60


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
    # control: a corrupted boundary changes the history, so equality is not vacuous
    mine = _original("host", CorruptInstall(boundary()))
    again = _reloaded(_document(boundary()), "host")
    assert not all(
        histories_equal(mine.history(w), again.history(w)).ok for w in range(N_WORLDS)
    )
