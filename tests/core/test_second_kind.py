"""The catalog kind: a delayed row released to the observer, never to the policy."""

import ast
from pathlib import Path

import numpy as np

from spaceodyssey import core
from spaceodyssey.core import (
    Audience,
    CompiledExecutor,
    HostExecutor,
    canonical_history,
    histories_equal,
    index_world,
    snapshot_view,
)
from spaceodyssey.studies import catalog, toy
from tests.core.corpus import frozen
from tests.core.helpers import N_STEPS_CORPUS, worlds

PLAN = ((0, 10), (0, catalog.CATALOG_CHANNEL_ID), (1, 10), (0, 10), (1, 10))


def _study():
    return catalog.catalog_study(policies=(frozen(plan=PLAN),))


def _run_host(study):
    truths, roots = worlds(1)
    return HostExecutor().run(study, index_world(truths, 0), roots[0], N_STEPS_CORPUS)


def _catalog_rows(carry, kind):
    n = int(carry.n_log)
    return np.flatnonzero(np.asarray(carry.log.kind)[:n] == kind)


def test_second_kind_needs_no_core_edit():
    core_dir = Path(core.__file__).parent
    # the core never names the catalog, in code, comments or strings
    for path in core_dir.rglob("*.py"):
        assert "catalog" not in path.read_text().lower(), path.name
    # the kind is defined under studies/ and nowhere else in the library
    src = core_dir.parent
    definers = [
        p.relative_to(src).as_posix()
        for p in src.rglob("*.py")
        if any(
            isinstance(n, ast.Call)
            and getattr(n.func, "id", getattr(n.func, "attr", "")) == "RecordKind"
            and n.args
            and getattr(n.args[0], "value", None) == "catalog"
            for n in ast.walk(ast.parse(p.read_text()))
        )
    ]
    assert definers == ["studies/catalog.py"]
    # it registers through the public API, after the toy kinds, which stay as they were
    registry = toy.toy_registry()
    before = [registry[i] for i in range(len(registry))]
    index = catalog.register_catalog(registry)
    assert index == len(before) and registry[index] is catalog.CATALOG_KIND
    assert [registry[i] for i in range(len(before))] == before


def test_catalog_row_reaches_the_observer_at_its_release_tick():
    study = _study()
    carry, _trace = _run_host(study)
    kind = study.channels.kind[3].item()
    (row,) = _catalog_rows(carry, kind)
    log = carry.log
    release = int(log.end_tick[row]) + catalog.CATALOG_DELAY_TICKS
    assert int(log.available_tick[row]) == release
    assert int(log.accepted_tick[row]) == release
    assert int(log.audience[row]) == int(Audience.OBSERVER)
    # the observer folded its value into the catalog slot, at that tick
    assert float(carry.belief.aux[catalog.CATALOG_AUX_SLOT]) == float(
        log.payload[row, 0]
    )
    obs = snapshot_view(carry, Audience.OBSERVER, study.caps)
    assert int(log.kind[row]) in np.asarray(obs.log.kind)[: int(obs.n_rows)]


def test_catalog_row_never_reaches_the_policy():
    study = _study()
    carry, _ = _run_host(study)
    kind = study.channels.kind[3].item()
    (row,) = _catalog_rows(carry, kind)
    assert int(carry.log.released_tick[row]) == -1
    view = snapshot_view(carry, Audience.POLICY, study.caps)
    assert kind not in np.asarray(view.log.kind)[: int(view.n_rows)]
    assert float(carry.pbelief.aux[catalog.CATALOG_AUX_SLOT]) == 0.0


def test_catalog_study_agrees_between_executors():
    study = _study()
    truths, roots = worlds(2)
    cs, ts = CompiledExecutor().run(study, truths, roots, N_STEPS_CORPUS)
    for w in range(2):
        host = HostExecutor().run(
            study, index_world(truths, w), roots[w], N_STEPS_CORPUS
        )
        r = histories_equal(
            canonical_history(*host),
            canonical_history(index_world(cs, w), index_world(ts, w)),
        )
        assert r.structure_equal and r.max_int_diff == 0 and r.max_abs <= 1e-9
