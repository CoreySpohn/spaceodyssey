"""B3: a reloaded run reproduces the original run's history on both executors.

`hash_pin.json` freezes the content hash of every corpus study's run document, and
the schema version, as they stood at a recorded commit. Nothing regenerates it: a
pinned hash that differs fails. It is written once, by running this file with
`SPACEODYSSEY_CAPTURE_PIN=<full commit sha>` while no pin file exists.
"""

import dataclasses
import json
import os
import re
from pathlib import Path

import equinox as eqx
import pytest

from spaceodyssey.codec import SCHEMA_VERSION, Bindings, dump_run, load_run
from spaceodyssey.core import CompiledExecutor, HostExecutor, preflight
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.conformance.corpus import make_worlds
from spaceodyssey.core.executors.base import histories_equal
from spaceodyssey.core.records import KindRegistry, PayloadColumn, RecordKind
from spaceodyssey.studies import toy
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
HASH_PIN = Path(__file__).with_name("hash_pin.json")
CAPTURE = os.environ.get("SPACEODYSSEY_CAPTURE_PIN", "")
EXECUTORS = ("host", "compiled")


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


def _corpus_document(case, executor, study=None):
    """The run document of one corpus case on one executor."""
    study = case.make_study() if study is None else study
    refit = None if case.make_refit is None else case.make_refit()
    service = None if case.make_service is None else case.make_service()
    return dump_run(
        study,
        _registry_of(study.caps),
        executor=executor,
        bindings=bindings(),
        service=service,
        refit=refit,
        kinds=TOY_KINDS,
    )


def _capture(commit):
    """Write the hash pin from the current documents; never replaces a pin."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("SPACEODYSSEY_CAPTURE_PIN names a full commit sha")
    if HASH_PIN.exists():
        raise FileExistsError(f"{HASH_PIN.name} exists; a pin is never rewritten")
    documents = {
        f"{case.name}/{executor}": _corpus_document(case, executor).content_hash
        for case in CORPUS
        for executor in EXECUTORS
    }
    pin = {
        "source_commit": commit,
        "schema_version": SCHEMA_VERSION,
        "documents": documents,
    }
    HASH_PIN.write_text(json.dumps(pin, indent=1, sort_keys=True) + "\n")


def _hash_pin():
    if CAPTURE and not HASH_PIN.exists():
        _capture(CAPTURE)
    return json.loads(HASH_PIN.read_text())


def test_the_hash_pin_names_its_commit_and_covers_every_corpus_document():
    """Fails if the pin lost its commit or a document, or the schema version moved."""
    pin = _hash_pin()
    assert re.fullmatch(r"[0-9a-f]{40}", pin["source_commit"])
    assert pin["schema_version"] == SCHEMA_VERSION
    assert set(pin["documents"]) == {
        f"{case.name}/{executor}" for case in CORPUS for executor in EXECUTORS
    }


def test_a_changed_study_misses_its_pinned_hash():
    """Control: one changed declaration value must move the document hash."""
    case = CORPUS[0]
    study = case.make_study()
    changed = eqx.tree_at(lambda s: s.horizon_tick, study, study.horizon_tick + 1)
    pinned = _hash_pin()["documents"][f"{case.name}/host"]
    assert _corpus_document(case, "host").content_hash == pinned
    assert _corpus_document(case, "host", changed).content_hash != pinned


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.name)
def test_every_corpus_run_reloads_to_the_same_hash_and_history(case, executor):
    study = case.make_study()
    refit = None if case.make_refit is None else case.make_refit()
    table = bindings()
    doc = _corpus_document(case, executor, study)
    assert doc.content_hash == _hash_pin()["documents"][f"{case.name}/{executor}"]
    run = load_run(doc.to_json(), table, TOY_KINDS)
    again = dump_run(
        run.study,
        run.registry,
        executor=run.executor,
        bindings=table,
        service=run.make_service(),
        refit=run.refit or None,
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
        refit=refit,
    )
    reloaded = run.execute(truths, roots, N_CASE_STEPS)
    for w in range(2):
        assert histories_equal(mine.history(w), reloaded.history(w)).ok


@pytest.mark.parametrize(
    "feature",
    ["programs", "refunds", "awards", "commitments", "balances", "key_version"],
)
def test_an_enabled_feature_run_recipe_redumps_exactly(feature):
    """Fails if a valid enabled feature changes its study, registry or bindings on a
    run document round trip.
    """
    from tests.core.preflight_cases import DOC_KINDS, DocPolicy

    kinds = {**TOY_KINDS, **DOC_KINDS}
    caps = {
        "programs": Capacities(n_programs=2),
        "refunds": Capacities(refunds=True),
        "awards": Capacities(awards=True),
        "commitments": Capacities(n_commitments=2),
        "balances": Capacities(snapshot_balances=True, balance_quantum=0.5),
        "key_version": Capacities(),
    }[feature]
    policy = DocPolicy(
        programs=(0, 1) if feature == "programs" else (),
        awards=((2, 10, 80),) if feature == "awards" else (),
        disruptive=feature == "commitments",
    )
    study = toy.survey_1(
        caps=caps, policies=(policy,), key_version=2 if feature == "key_version" else 1
    )
    if feature == "refunds":
        study = dataclasses.replace(
            study,
            channels=dataclasses.replace(
                study.channels, charge=study.channels.cost.at[1, 2].set(7.0)
            ),
        )
    if feature == "commitments":
        study = dataclasses.replace(
            study,
            assets=dataclasses.replace(
                study.assets,
                freeze_ticks=study.assets.freeze_ticks.at[0].set(5),
                disruptive_resource=study.assets.disruptive_resource.at[1].set(1),
            ),
        )
    assert preflight(study) == []
    table = bindings()
    doc = dump_run(
        study, toy.toy_registry(), executor="host", bindings=table, kinds=kinds
    )
    loaded = load_run(doc.to_json(), table, kinds)
    again = dump_run(
        loaded.study,
        loaded.registry,
        executor=loaded.executor,
        bindings=table,
        service=loaded.make_service(),
        refit=loaded.refit or None,
        kinds=kinds,
    )
    assert again.to_json() == doc.to_json()
