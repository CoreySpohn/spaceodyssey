"""JSON pre-flight: all thirteen finding codes, each with a pointer to its value."""

import copy
import dataclasses
import json
import sys

import equinox as eqx
import jax
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
from spaceodyssey.codec.preflight import DocumentCode, _follow, resolve_pointer
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
from spaceodyssey.plugins import RegistrationError
from spaceodyssey.studies import toy
from tests.codec.test_document import TOY_KINDS
from tests.core.preflight_cases import (
    DOC_KINDS,
    DocChannelModel,
    DocObjective,
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
        "/study/observer/reads/0",
        _is("truth"),
    ),
    "F002": (
        FindingCode.F002,
        f002,
        "/study/observer/reads/0",
        _is("scorer"),
    ),
    "C001": (
        FindingCode.C001,
        c001,
        "/study/agenda/policies/0/needs/0",
        _is([10, 0, 19]),
    ),
    "K001": (
        FindingCode.K001,
        k001,
        "/study/channel_model/key_coords",
        lambda v: "acquisition_index" not in v,
    ),
    "K002": (FindingCode.K002, k002, "/study/channels/id/data/1", _is(10)),
    "X001": (
        FindingCode.X001,
        x001,
        "/study/agenda/policies/0/effects",
        _is([int(EffectKind.ALLOCATE)]),
    ),
    "X002": (
        FindingCode.X002,
        x002,
        "/study/assets/freeze_ticks/data/0",
        _is(5),
    ),
    "T001": (
        FindingCode.T001,
        t001,
        "/study/agenda/policies/0/stops/0",
        _is([int(Scope.LANE), int(Disposition.DRAIN)]),
    ),
    "T002": (
        FindingCode.T002,
        t002,
        "/study/channels/coincident_order/data/2",
        _is(7),
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
        ("C001", "/study/agenda/policies/0/needs/0"),
        ("F001", "/study/observer/reads/0"),
        ("F002", "/study/observer/reads/1"),
        ("K002", "/study/channels/id/data/1"),
        ("R001", "/study/channels/kind/data/1"),
        ("T001", "/study/agenda/policies/0/stops/0"),
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


def test_broken_installation_is_a_registration_error_not_a_finding():
    """Fails if a bad registration becomes a finding, or surfaces as a plain
    ValueError instead of RegistrationError, or loses its cause.
    """
    text = _text(_doc(toy.survey_1()))
    for bad in ({("core", "study", 1): toy.ToyDesign}, {("bad", "ref"): toy.ToyDesign}):
        with pytest.raises(RegistrationError) as err:
            preflight_document(text, bad)
        assert type(err.value) is RegistrationError
        assert type(err.value.__cause__) is ValueError


@pytest.mark.parametrize(
    ("edit", "where"),
    [
        (lambda d: d["study"]["budget"].update(shape=[3, 4]), "/study/budget/data"),
        (lambda d: d.update(content_hash="0" * 64), "/content_hash"),
        (lambda d: d.update(format="other"), "/format"),
        (lambda d: d.update(extra=1), "/extra"),
    ],
    ids=["wrong-shape", "edited-hash", "wrong-format", "extra-key"],
)
def test_documents_that_cannot_decode_are_reported_not_raised(edit, where):
    """Fails if a document that does not decode raises, or is reported without the
    pointer of the failure, or hides an independent finding beside it.
    """
    doc = _doc(toy.survey_1(policies=(DocPolicy(effects=(QUERY,)),)))
    edit(doc)
    found = _find(doc, _run(compiled_hash=8))
    assert sorted((str(f.code), f.pointer) for f in found) == sorted(
        [("L001", where), ("D003", "/run/refit/compiled/model_hash")]
    )
    assert all(f.in_memory is None for f in found)


def test_malformed_text_is_one_load_finding():
    """Fails if malformed JSON raises."""
    found = preflight_document("{not json", RESOLVER)
    assert [(f.code, f.pointer) for f in found] == [("L001", "")]


def test_repeated_key_and_non_finite_number_are_refused_like_the_loader():
    """Fails if text that load_study refuses certifies clean: a key written twice
    (the first value is dropped by a plain parse) or a number that overflows.
    """
    clean = json.dumps(_doc(toy.survey_1()))
    twice = clean.replace(
        '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
    )
    assert twice != clean
    with pytest.raises(DocumentError):
        load_study(twice, RESOLVER)
    assert [(f.code, f.pointer) for f in preflight_document(twice, RESOLVER)] == [
        ("L001", "/schema_version")
    ]
    doc = _doc(toy.survey_1())
    doc["study"]["clock"]["tick_length_s"]["data"] = "huge"
    huge = json.dumps(doc).replace('"huge"', "1e999")
    with pytest.raises(DocumentError):
        load_study(huge, RESOLVER)
    found = preflight_document(huge, RESOLVER)
    assert [(f.code, f.pointer) for f in found] == [
        ("L001", "/study/clock/tick_length_s/data")
    ]


def test_resolve_pointer_follows_escapes_and_refuses_missing_values():
    """Fails if the pointer helper misreads escapes, or invents a value."""
    root = {"a/b": {"c~d": [10, 20]}}
    assert resolve_pointer(root, "/a~1b/c~0d/1") == 20
    assert resolve_pointer(root, "") is root
    with pytest.raises(KeyError):
        resolve_pointer(root, "/a~1b/c~0d/2")
    with pytest.raises(KeyError):
        resolve_pointer(copy.deepcopy(root), "/missing")


# ---- provenance: the value, through the document's own encoding -------------------


def _variant_x001_requires(seed):
    return _policy_doc(requires=("gradients",) if seed else ()), {"executor": "host"}


def _variant_x001_executors(seed):
    executors = ("host",) if seed else ("compiled", "host")
    return _policy_doc(executors=executors), {"executor": "compiled"}


def _variant_x002_charge(seed):
    study = toy.survey_1()
    charge = study.channels.cost.at[1, 2].set(2.0 if seed else 1.0)
    return _doc(_with(study, channels=_channels(charge=charge))), None


def _variant_x002_resource(seed):
    study = toy.survey_1()
    rows = jnp.asarray([-1, 0 if seed else -1], I64)
    assets = dataclasses.replace(study.assets, disruptive_resource=rows)
    return _doc(_with(study, assets=assets)), None


def _variant_x002_policy(seed):
    return _policy_doc(disruptive=seed), None


def _variant_t002_shape(seed):
    order = jnp.asarray([0, 1, 1] if not seed else [0, 1], I64)
    return _doc(_with(toy.survey_1(), channels=_channels(coincident_order=order))), None


def _variant_c001_second_need(seed):
    return _policy_doc(needs=((10, 0, 500), (11, 0, 5) if seed else (11, 0, 500))), None


def _variant_t001_second_drain(seed):
    scope = Scope.ASSET if seed else Scope.CAMPAIGN
    stops = ((int(Scope.CAMPAIGN), int(Disposition.DRAIN)), (int(scope), 1))
    return _policy_doc(stops=stops), None


VARIANTS = {
    "X001-requires": (
        FindingCode.X001,
        _variant_x001_requires,
        "/study/agenda/policies/0/requires",
        _is(["gradients"]),
    ),
    "X001-executors": (
        FindingCode.X001,
        _variant_x001_executors,
        "/study/agenda/policies/0/executors",
        _is(["host"]),
    ),
    "X002-charge": (
        FindingCode.X002,
        _variant_x002_charge,
        "/study/channels/charge/data/1/2",
        _is(2.0),
    ),
    "X002-resource": (
        FindingCode.X002,
        _variant_x002_resource,
        "/study/assets/disruptive_resource/data/1",
        _is(0),
    ),
    "X002-policy": (
        FindingCode.X002,
        _variant_x002_policy,
        "/study/agenda/policies/0/disruptive",
        _is(True),
    ),
    "T002-shape": (
        FindingCode.T002,
        _variant_t002_shape,
        "/study/channels/coincident_order",
        lambda v: v["shape"] == [2],
    ),
    "C001-second-need": (
        FindingCode.C001,
        _variant_c001_second_need,
        "/study/agenda/policies/0/needs/1",
        _is([11, 0, 5]),
    ),
}


@pytest.mark.parametrize("name", list(VARIANTS))
def test_each_declaration_is_located_at_its_own_value(name):
    """Fails if a finding stops at the containing module or the whole column, or
    points at another value of the same component.
    """
    code, build, where, offending = VARIANTS[name]
    doc, run = build(True)
    found = [f for f in _find(doc, run) if f.code == code]
    assert found and {f.pointer for f in found} == {where}
    assert offending(resolve_pointer(_root(doc, run), where))
    clean_doc, clean_run = build(False)
    assert _find(clean_doc, clean_run) == []


def test_second_drain_is_located_at_the_second_pair():
    """Fails if a later offending pair is attributed to an earlier one."""
    doc, _ = _variant_t001_second_drain(True)
    found = _find(doc, None)
    assert [(f.code, f.pointer) for f in found] == [
        ("T001", "/study/agenda/policies/0/stops/1")
    ]


def test_values_are_located_through_container_tags():
    """Fails if a pointer assumes plain arrays: a field written as `$list` loads, and
    its findings must point through the tag.
    """
    doc, _ = f001(True)
    study_doc = _doc(toy.survey_1(policies=(DocPolicy(reads=("truth",)),)))
    policies = study_doc["study"]["agenda"]["policies"]
    study_doc["study"]["agenda"]["policies"] = {"$list": policies}
    reads = study_doc["study"]["agenda"]["policies"]["$list"][0]["reads"]
    study_doc["study"]["agenda"]["policies"]["$list"][0]["reads"] = {"$list": reads}
    found = _find(study_doc, None)
    assert [(f.code, f.pointer) for f in found] == [
        ("F001", "/study/agenda/policies/$list/0/reads/$list/0")
    ]
    assert resolve_pointer(study_doc, found[0].pointer) == "truth"
    assert doc  # the plain form of the same mistake is the F001 seed


def test_follow_reads_dict_entries_and_array_rows():
    """Fails if a `$dict` entry or a nested array element is not located."""
    node = {"$dict": [["a", 1], ["b", [5, 6]]]}
    assert _follow(node, ("b", 1), ()) == ("$dict", 1, 1, 1)
    array = {"$array": "int64", "shape": [2, 2], "data": [[1, 2], [3, 4]]}
    assert _follow(array, (1, 0), ()) == ("data", 1, 0)
    assert _follow(array, (1, 5, 9), ()) == ("data", 1)


def test_null_order_is_reported_beside_independent_findings():
    """Fails if a null coincident order stops the load: the core reports T002 and the
    other findings together.
    """
    doc = _doc(toy.survey_1(observer=DocObserver(reads=("truth",))))
    doc["study"]["channels"]["coincident_order"] = None
    found = _find(doc, None)
    assert sorted((str(f.code), f.pointer) for f in found) == [
        ("F001", "/study/observer/reads/0"),
        ("T002", "/study/channels/coincident_order"),
    ]


# ---- independent checks always run ------------------------------------------------


def test_document_and_binding_findings_are_reported_together():
    """Fails if an unknown kind or a newer schema hides the binding mismatch, or a
    newer schema hides the unknown kind.
    """
    doc, _ = d001(True)
    found = _find(doc, _run(compiled_hash=8))
    assert sorted((str(f.code), f.pointer) for f in found) == [
        ("D001", "/study/observer/$kind"),
        ("D003", "/run/refit/compiled/model_hash"),
    ]
    doc["schema_version"] = 2
    found = _find(doc, _run(compiled_hash=8))
    assert sorted((str(f.code), f.pointer) for f in found) == [
        ("D001", "/study/observer/$kind"),
        ("D002", "/schema_version"),
        ("D003", "/run/refit/compiled/model_hash"),
    ]


def test_every_component_that_asks_a_query_needs_a_service():
    """Fails if the service check misses a component the core pre-flight reads: a
    channel model, a policy, the observer.
    """
    model = DocChannelModel(noise=toy.survey_1().channel_model.noise, effects=(QUERY,))
    doc = _doc(toy.survey_1(channel_model=model))
    found = _find(doc, {})
    assert [(f.code, f.pointer) for f in found] == [("D003", "/run")]
    assert "channel_model" in found[0].message
    assert _find(doc, {"service": _run()["service"]}) == []


@pytest.mark.parametrize(
    ("edit", "where"),
    [
        (lambda r: r["service"].update(kind=None), "/run/service/kind"),
        (lambda r: r["service"].update(version=0), "/run/service/version"),
        (lambda r: r["service"].update(version=True), "/run/service/version"),
        (lambda r: r["service"].update(configuration=[]), "/run/service/configuration"),
        (lambda r: r.update(service=None), "/run/service"),
        (lambda r: r.update(refit=None), "/run/refit"),
        (lambda r: r.update(executor="gpu"), "/run/executor"),
        (
            lambda r: r["refit"]["host"].update(model_hash=None),
            "/run/refit/host/model_hash",
        ),
        (
            lambda r: r["refit"]["host"].update(model_hash=1.5),
            "/run/refit/host/model_hash",
        ),
        (lambda r: r["refit"]["host"].update(kind=""), "/run/refit/host/kind"),
    ],
    ids=[
        "null-kind",
        "zero-version",
        "bool-version",
        "list-configuration",
        "null-service",
        "null-refit",
        "unknown-executor",
        "null-hash",
        "float-hash",
        "empty-kind",
    ],
)
def test_binding_members_are_checked_against_the_schema(edit, where):
    """Fails if a binding with every key present but a null, mistyped or empty value
    passes as complete.
    """
    doc = _policy_doc(effects=(QUERY,))
    run = _run()
    edit(run)
    pointers = [(str(f.code), f.pointer) for f in _find(doc, run)]
    assert ("D003", where) in pointers
    assert {c for c, _ in pointers} == {"D003"}


def test_two_null_hashes_are_not_equal_hashes():
    """Fails if two null model hashes pass as a match."""
    run = _run()
    run["refit"]["host"]["model_hash"] = None
    run["refit"]["compiled"]["model_hash"] = None
    found = _find(_policy_doc(), run)
    assert sorted(f.pointer for f in found) == [
        "/run/refit/compiled/model_hash",
        "/run/refit/host/model_hash",
    ]


def test_an_incomplete_binding_does_not_hide_a_hash_mismatch():
    """Fails if a missing member of one binding stops the hash comparison."""
    run = _run(compiled_hash=8)
    del run["refit"]["host"]["configuration"]
    found = _find(_policy_doc(), run)
    assert sorted(f.pointer for f in found) == [
        "/run/refit/compiled/model_hash",
        "/run/refit/host",
    ]


# ---- second review: identity from the finding, not from values ---------------------


ALLOCATE = int(EffectKind.ALLOCATE)


def test_repeated_needs_on_one_channel_point_at_the_offending_entry():
    """Fails if a finding is attributed to an earlier entry that shares its channel
    id, or if three identical reads give anything but three entries.
    """
    doc = _policy_doc(needs=((10, 0, 500), (10, 0, 19)))
    found = _find(doc, None)
    assert [(f.code, f.pointer) for f in found] == [
        ("C001", "/study/agenda/policies/0/needs/1")
    ]
    doc = _doc(toy.survey_1(observer=DocObserver(reads=("truth",) * 3)))
    found = _find(doc, None)
    assert [(f.code, f.pointer) for f in found] == [
        ("F001", f"/study/observer/reads/{i}") for i in range(3)
    ]


def test_an_effect_number_shared_with_another_value_is_not_confused():
    """Fails if an unsupported effect is located at an equal number in another field
    of the same component.
    """
    doc = _policy_doc(needs=((ALLOCATE, 0, 500),), effects=(ALLOCATE,))
    pointers = {f.pointer for f in _find(doc, None) if f.code == FindingCode.X001}
    assert pointers == {"/study/agenda/policies/0/effects"}


def test_a_requirement_name_with_escapes_is_located():
    """Fails if a requirement is found from the printed message, which loses an
    escaped name.
    """
    doc = _policy_doc(requires=("a\\b'c",))
    found = _find(doc, {"executor": "host"})
    assert [(f.code, f.pointer) for f in found] == [
        ("X001", "/study/agenda/policies/0/requires")
    ]


@pytest.mark.parametrize("executor", [[], {}, 3, None, "gpu"])
def test_a_malformed_executor_is_a_binding_finding(executor):
    """Fails if a non-string executor raises, or is not reported at /run/executor."""
    found = _find(_policy_doc(), {"executor": executor})
    assert [(f.code, f.pointer) for f in found] == [("D003", "/run/executor")]


def test_hash_is_checked_whatever_else_is_wrong():
    """Fails if an unknown kind hides a wrong content hash."""
    doc, _ = d001(True)
    doc["content_hash"] = "0" * 64
    found = _find(doc, None)
    assert sorted((str(f.code), f.pointer) for f in found) == [
        ("D001", "/study/observer/$kind"),
        ("L001", "/content_hash"),
    ]


def test_a_study_document_object_is_checked_as_the_loader_reads_it():
    """Fails if an object that load_study accepts, with a raw NaN in it, is reported
    as broken because it was serialized first.
    """
    doc = _doc(toy.survey_1())
    doc["study"]["clock"]["tick_length_s"]["data"] = float("nan")
    obj = StudyDocument(study=doc["study"])
    load_study(obj, RESOLVER)
    assert preflight_document(obj, RESOLVER) == []


@pytest.mark.parametrize(
    ("poke", "where"),
    [
        (
            lambda d: d["study"]["clock"]["tick_length_s"].update(data=float("nan")),
            "/study/clock/tick_length_s/data",
        ),
        (lambda d: d["study"].update(extra={1, 2}), "/study/extra"),
        (lambda d: d["study"].update(extra={1: 2}), "/study/extra"),
    ],
    ids=["nan", "set", "int-key"],
)
def test_a_parsed_object_that_is_not_json_is_a_load_finding(poke, where):
    """Fails if a parsed object with a non-JSON value raises instead of giving L001."""
    doc = _doc(toy.survey_1())
    poke(doc)
    found = preflight_document(doc, RESOLVER)
    assert [(f.code, f.pointer) for f in found] == [("L001", where)]


class OrderLikeModel(toy.ToyPhotometry):
    """A plugin channel model with a required array field named like the table's."""

    coincident_order: jax.Array = eqx.field(kw_only=True)


def test_null_tolerance_is_only_for_the_channel_table():
    """Fails if a plugin field of the same name, encoded null, passes pre-flight
    while load_study refuses it.
    """
    kinds = KindResolver(
        {**TOY_KINDS, ("channel_model", "test.order_like", 1): OrderLikeModel}
    )
    noise = toy.survey_1().channel_model.noise
    study = toy.survey_1(
        channel_model=OrderLikeModel(noise=noise, coincident_order=jnp.zeros(3))
    )
    doc = json.loads(dump_study(study, kinds).to_json())
    doc.pop("content_hash")
    doc["study"]["channel_model"]["coincident_order"] = None
    with pytest.raises(DocumentError):
        load_study(json.dumps(doc), kinds)
    found = preflight_document(json.dumps(doc), kinds)
    assert [(f.code, f.pointer) for f in found] == [
        ("L001", "/study/channel_model/coincident_order")
    ]


def test_container_tag_indices_beyond_the_first_are_followed():
    """Fails if the second element of a `$list` is located as the first."""
    study = toy.survey_1(
        policies=(DocPolicy(), DocPolicy(reads=("truth",))), phase_until=(300, 1000)
    )
    doc = _doc(study)
    doc["study"]["agenda"]["policies"] = {"$list": doc["study"]["agenda"]["policies"]}
    found = _find(doc, None)
    assert [(f.code, f.pointer) for f in found] == [
        ("F001", "/study/agenda/policies/$list/1/reads/0")
    ]


def test_objective_and_observer_are_in_the_service_inventory():
    """Fails if a component other than the policies and channel model is skipped."""
    objective = DocObjective(threshold=3.0, effects=(QUERY,))
    found = _find(_doc(_with(toy.survey_1(), objective=objective)), {})
    assert [(f.code, f.pointer) for f in found] == [("D003", "/run")]
    assert "objective" in found[0].message


def test_binding_edge_cases_the_first_tests_missed():
    """Fails if a null service version passes, if a host binding without its kind
    hides a hash mismatch, or if a nested repeated key is not refused.
    """
    run = _run(compiled_hash=8)
    run["service"]["version"] = None
    del run["refit"]["host"]["kind"]
    found = _find(_policy_doc(effects=(QUERY,)), run)
    assert sorted(f.pointer for f in found) == [
        "/run/refit/compiled/model_hash",
        "/run/refit/host",
        "/run/service/version",
    ]
    text = json.dumps(_doc(toy.survey_1()))
    nested = text.replace('"clock": {', '"clock": {"$kind": ["core", "clock", 1], ', 1)
    assert nested != text
    found = preflight_document(nested, RESOLVER)
    assert [(f.code, f.pointer) for f in found] == [("L001", "/study/clock/$kind")]
