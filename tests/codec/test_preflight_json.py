"""JSON pre-flight: all thirteen finding codes, each with a pointer to its value."""

import copy
import dataclasses
import json
import sys

import jax.numpy as jnp
import pytest

from spaceodyssey.codec import (
    DocumentError,
    KindResolver,
    StudyDocument,
    dump_study,
    load_study,
    preflight_document,
)
from spaceodyssey.codec.preflight import DocumentCode, resolve_pointer
from spaceodyssey.core.declarations import DRAW_COORDS
from spaceodyssey.core.enums import (
    I64,
    Disposition,
    EffectKind,
    Purpose,
    Scope,
)
from spaceodyssey.core.keys import Coupling
from spaceodyssey.core.preflight import FindingCode, preflight
from spaceodyssey.studies import toy
from tests.codec.test_document import TOY_KINDS
from tests.core.preflight_cases import (
    DOC_KINDS,
    DocChannelModel,
    DocObserver,
    DocPolicy,
)

RESOLVER = KindResolver({**TOY_KINDS, **DOC_KINDS})
QUERY = int(EffectKind.QUERY)


def _coords_without(name):
    return tuple(c for c in DRAW_COORDS if c != name)


def _doc(study):
    """The document of a study as a parsed object, without its hash."""
    doc = json.loads(dump_study(study, RESOLVER).to_json())
    doc.pop("content_hash")
    return doc


def _channels(**columns):
    study = toy.survey_1()
    return dataclasses.replace(study.channels, **columns)


def _with(study, **fields):
    return dataclasses.replace(study, **fields)


def _policy_doc(**fields):
    return _doc(toy.survey_1(policies=(DocPolicy(**fields),)))


def _run(service=True, compiled_hash=7):
    refit = {"kind": "masked", "version": 1, "configuration": {}, "model_hash": 7}
    run = {
        "executor": "host",
        "refit": {"host": refit, "compiled": {**refit, "model_hash": compiled_hash}},
    }
    if service:
        run["service"] = {"kind": "oracle", "version": 1, "configuration": {}}
    return run


def _is(value):
    return lambda v: v == value


# Each builder takes `seed` and returns (document, run record). With `seed` false it
# returns the same document with the mistake repaired, a clean control.


def f001(seed):
    reads = ("truth",) if seed else ()
    return _doc(toy.survey_1(observer=DocObserver(reads=reads))), None


def f002(seed):
    reads = ("scorer",) if seed else ()
    return _doc(toy.survey_1(observer=DocObserver(reads=reads))), None


def c001(seed):
    return _policy_doc(needs=((10, 0, 19 if seed else 500),)), None


def k001(seed):
    coords = _coords_without("acquisition_index") if seed else None
    noise = toy.survey_1().channel_model.noise
    model = (
        DocChannelModel(noise=noise)
        if coords is None
        else DocChannelModel(noise=noise, key_coords=coords)
    )
    return _doc(toy.survey_1(channel_model=model)), None


def k002(seed):
    ids = [10, 10, 12] if seed else [10, 11, 12]
    return _doc(
        _with(toy.survey_1(), channels=_channels(id=jnp.asarray(ids, I64)))
    ), None


def x001(seed):
    effects = (int(EffectKind.ALLOCATE),) if seed else ()
    return _policy_doc(effects=effects), None


def x002(seed):
    study = toy.survey_1()
    ticks = jnp.asarray([5 if seed else 0, 0], I64)
    assets = dataclasses.replace(study.assets, freeze_ticks=ticks)
    return _doc(_with(study, assets=assets)), None


def t001(seed):
    scope = Scope.LANE if seed else Scope.CAMPAIGN
    return _policy_doc(stops=((int(scope), int(Disposition.DRAIN)),)), None


def t002(seed):
    order = jnp.asarray([0, 1, 7 if seed else 1], I64)
    return _doc(_with(toy.survey_1(), channels=_channels(coincident_order=order))), None


def r001(seed):
    kind = toy.survey_1().channels.kind.at[2].set(9 if seed else 1)
    return _doc(_with(toy.survey_1(), channels=_channels(kind=kind))), None


def d001(seed):
    doc = _doc(toy.survey_1())
    if seed:
        doc["study"]["observer"]["$kind"] = ["observer", "toy.no_such_kind", 1]
    return doc, None


def d002(seed):
    doc = _doc(toy.survey_1())
    doc["schema_version"] = 2 if seed else 1
    return doc, None


def d003_service(seed):
    return _policy_doc(effects=(QUERY,)), _run(service=not seed)


def d003_hash(seed):
    return _policy_doc(effects=(QUERY,)), _run(compiled_hash=8 if seed else 7)


# code, builder, pointer, check on the value at the pointer, and what it checks
CASES = {
    "F001": (
        FindingCode.F001,
        f001,
        "/study/observer",
        lambda v: v["reads"] == ["truth"],
    ),
    "F002": (
        FindingCode.F002,
        f002,
        "/study/observer",
        lambda v: v["reads"] == ["scorer"],
    ),
    "C001": (
        FindingCode.C001,
        c001,
        "/study/agenda/policies/0",
        lambda v: v["needs"] == [[10, 0, 19]],
    ),
    "K001": (
        FindingCode.K001,
        k001,
        "/study/channel_model",
        lambda v: "acquisition_index" not in v["key_coords"],
    ),
    "K002": (FindingCode.K002, k002, "/study/channels/id/data/1", _is(10)),
    "X001": (
        FindingCode.X001,
        x001,
        "/study/agenda/policies/0",
        lambda v: v["effects"] == [int(EffectKind.ALLOCATE)],
    ),
    "X002": (
        FindingCode.X002,
        x002,
        "/study/assets/freeze_ticks",
        lambda v: v["data"] == [5, 0],
    ),
    "T001": (
        FindingCode.T001,
        t001,
        "/study/agenda/policies/0",
        lambda v: v["stops"] == [[int(Scope.LANE), int(Disposition.DRAIN)]],
    ),
    "T002": (
        FindingCode.T002,
        t002,
        "/study/channels/coincident_order",
        lambda v: v["data"] == [0, 1, 7],
    ),
    "R001": (FindingCode.R001, r001, "/study/channels/kind/data/2", _is(9)),
    "D001": (
        DocumentCode.D001,
        d001,
        "/study/observer/$kind",
        _is(["observer", "toy.no_such_kind", 1]),
    ),
    "D002": (DocumentCode.D002, d002, "/schema_version", _is(2)),
    "D003-service": (
        DocumentCode.D003,
        d003_service,
        "/run",
        lambda v: "service" not in v,
    ),
    "D003-hash": (
        DocumentCode.D003,
        d003_hash,
        "/run/refit/compiled/model_hash",
        _is(8),
    ),
}


def _text(doc):
    return json.dumps(doc)


def _root(doc, run):
    return doc if run is None else {**doc, "run": run}


def _find(doc, run, **kwargs):
    return preflight_document(_text(doc), RESOLVER, run=run, **kwargs)


@pytest.mark.parametrize("name", list(CASES))
def test_seeded_document_reports_its_code_at_the_offending_value(name):
    """Fails if a seeded document does not report exactly its one code, if the
    finding's pointer does not resolve to the offending value, or, for the ten
    in-memory codes, if the finding is not the core pre-flight's finding on the
    loaded study with the same code.
    """
    code, build, where, offending = CASES[name]
    doc, run = build(True)
    found = _find(doc, run)
    assert {f.code for f in found} == {code}
    # One finding per in-memory finding: K001 is raised once per channel, all of
    # them about the one channel model.
    assert len(found) == (3 if name == "K001" else 1)
    assert {f.pointer for f in found} == {where}
    assert offending(resolve_pointer(_root(doc, run), where))
    finding = found[0]
    if isinstance(code, FindingCode):
        assert finding.in_memory is not None and finding.in_memory.code == code
        study = load_study(_text(doc), RESOLVER)
        assert finding.in_memory in preflight(study)
    else:
        assert finding.in_memory is None


@pytest.mark.parametrize("name", list(CASES))
def test_repaired_document_is_clean_and_the_check_discriminates(name):
    """Vacuity control. Fails if the repaired document of a case has any finding,
    or if the value check also holds at the pointer in the repaired document (the
    check would then pass without the mistake).
    """
    _, build, where, offending = CASES[name]
    doc, run = build(False)
    assert _find(doc, run) == []
    try:
        value = resolve_pointer(_root(doc, run), where)
    except KeyError:
        return
    assert not offending(value)


def test_all_findings_are_reported_not_only_the_first():
    """Fails if a document with several mistakes reports fewer than all of them."""
    study = toy.survey_1(
        policies=(
            DocPolicy(
                needs=((10, 0, 19),),
                stops=((int(Scope.LANE), int(Disposition.DRAIN)),),
            ),
        ),
        observer=DocObserver(reads=("truth", "scorer")),
    )
    kind = study.channels.kind.at[1].set(9)
    ids = jnp.asarray([10, 10, 12], I64)
    study = _with(study, channels=_channels(kind=kind, id=ids))
    found = _find(_doc(study), None)
    seen = sorted((str(f.code), f.pointer) for f in found)
    assert seen == [
        ("C001", "/study/agenda/policies/0"),
        ("F001", "/study/observer"),
        ("F002", "/study/observer"),
        ("K002", "/study/channels/id/data/1"),
        ("R001", "/study/channels/kind/data/1"),
        ("T001", "/study/agenda/policies/0"),
    ]
    assert [f.severity for f in found][-1] == "warning"


def test_every_unknown_reference_is_reported():
    """Fails if only the first unknown kind is reported, or if an unknown enum
    reference or an unknown version is missed.
    """
    doc = _doc(toy.survey_1())
    doc["study"]["observer"]["$kind"] = ["observer", "toy.no_such_kind", 1]
    doc["study"]["design"]["$kind"] = ["design", "toy.gain", 9]
    doc["study"]["fork"]["$kind"] = ["core", "no_fork", 1]
    found = _find(doc, None)
    assert [(f.code, f.pointer) for f in found] == [
        ("D001", "/study/design/$kind"),
        ("D001", "/study/fork/$kind"),
        ("D001", "/study/observer/$kind"),
    ]
    assert preflight_document(_text(doc), RESOLVER, run=_run())[0].code == "D001"


def test_unknown_enum_reference_is_reported():
    """Fails if an `$enum` reference is not walked."""
    doc = _doc(
        toy.survey_1(coupling=Coupling(shared=((Purpose.MEASUREMENT, ("asset",)),)))
    )
    enums = []

    def walk(node, path):
        if isinstance(node, dict):
            if "$enum" in node:
                enums.append(path)
            for k, v in node.items():
                walk(v, path + "/" + k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}/{i}")

    walk(doc["study"], "/study")
    assert enums
    node = resolve_pointer(doc, enums[0])
    node["$enum"] = ["core", "no_such_enum", 1]
    found = _find(doc, None)
    assert [(f.code, f.pointer) for f in found] == [("D001", enums[0] + "/$enum")]


def test_clean_document_has_no_findings():
    """Fails if a correct study, or the same study with a complete run record,
    reports anything.
    """
    study = toy.survey_1(policies=(DocPolicy(effects=(QUERY,)),))
    doc = _doc(study)
    assert _find(doc, None) == []
    assert _find(doc, _run()) == []
    assert preflight(load_study(_text(doc), RESOLVER)) == []


def test_document_forms_agree():
    """Fails if text, a parsed object, a StudyDocument and a run record inside the
    document do not give the same findings.
    """
    doc, run = d003_service(True)
    expected = _find(doc, run)
    assert [f.code for f in expected] == ["D003"]
    assert preflight_document(doc, RESOLVER, run=run) == expected
    assert preflight_document({**doc, "run": run}, RESOLVER) == expected
    clean, _ = d003_service(False)
    study_doc = StudyDocument.from_json(_text(clean))
    assert preflight_document(study_doc, RESOLVER, run=_run()) == []
    with pytest.raises(ValueError, match="once"):
        preflight_document({**doc, "run": run}, RESOLVER, run=run)


def test_run_record_selects_the_executor():
    """Fails if the record's executor is ignored by the capability check, or if an
    explicit executor does not win over it.
    """
    doc = _policy_doc(requires=("gradients",))
    assert _find(doc, None) == []
    on_host = _find(doc, {"executor": "host"})
    assert [f.code for f in on_host] == ["X001"]
    assert _find(doc, {"executor": "host"}, executor="compiled") == []


def test_service_is_required_only_when_a_component_asks():
    """Fails if a study that asks for no Query is told to bind a service, or if a
    record is checked when none was given.
    """
    doc = _policy_doc()
    assert _find(doc, _run(service=False)) == []
    asking = _policy_doc(effects=(QUERY,))
    assert _find(asking, None) == []


def test_incomplete_bindings_are_findings():
    """Fails if a binding missing a member, or a refit record missing a side, is
    accepted.
    """
    asking = _policy_doc(effects=(QUERY,))
    run = _run()
    del run["service"]["version"]
    del run["refit"]["host"]["model_hash"]
    del run["refit"]["compiled"]
    found = _find(asking, run)
    assert sorted(f.pointer for f in found) == [
        "/run/refit",
        "/run/refit/host",
        "/run/service",
    ]
    assert {f.code for f in found} == {"D003"}


def test_newer_schema_reports_only_d002_even_with_bad_kinds():
    """Fails if a document of an unreadable schema is walked under today's rules."""
    doc, _ = d002(True)
    doc["study"]["observer"]["$kind"] = ["observer", "toy.no_such_kind", 1]
    assert [(f.code, f.pointer) for f in _find(doc, None)] == [
        ("D002", "/schema_version")
    ]


def test_kinds_none_means_the_core_kinds():
    """Fails if `kinds=None` reads anything but the core kinds, as `load_study` does."""
    doc = _doc(toy.survey_1())
    found = preflight_document(_text(doc))
    assert {f.code for f in found} == {"D001"}
    assert "/study/observer/$kind" in {f.pointer for f in found}


def test_missing_id_is_a_finding_not_an_import_error():
    """Fails if a module path as an id is imported, or raises."""
    sys.modules.pop("antigravity", None)
    doc = _doc(toy.survey_1())
    doc["study"]["observer"]["$kind"] = ["observer", "antigravity", 1]
    found = _find(doc, None)
    assert [f.code for f in found] == ["D001"]
    assert "antigravity" not in sys.modules


def test_broken_installation_is_an_error_not_a_finding():
    """Fails if a bad registration becomes a finding on the document."""
    with pytest.raises(ValueError, match="already bound"):
        preflight_document(
            _text(_doc(toy.survey_1())), {("core", "study", 1): toy.ToyDesign}
        )


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d["study"]["budget"].update(shape=[3, 4]),
        lambda d: d.update(content_hash="0" * 64),
    ],
    ids=["wrong-shape", "edited-hash"],
)
def test_documents_that_cannot_load_raise_with_a_pointer(edit):
    """Fails if a document that does not load for a reason outside the thirteen
    codes returns findings, or raises without a pointer.
    """
    doc = _doc(toy.survey_1())
    edit(doc)
    with pytest.raises(DocumentError) as err:
        preflight_document(_text(doc), RESOLVER)
    assert err.value.pointer
    with pytest.raises(DocumentError):
        preflight_document("{not json", RESOLVER)


def test_resolve_pointer_follows_escapes_and_refuses_missing_values():
    """Fails if the pointer helper misreads escapes, or invents a value."""
    root = {"a/b": {"c~d": [10, 20]}}
    assert resolve_pointer(root, "/a~1b/c~0d/1") == 20
    assert resolve_pointer(root, "") is root
    with pytest.raises(KeyError):
        resolve_pointer(root, "/a~1b/c~0d/2")
    with pytest.raises(KeyError):
        resolve_pointer(copy.deepcopy(root), "/missing")
