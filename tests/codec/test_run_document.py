"""Run documents: bindings, registry, file references, callables and no-switch."""

import copy
import dataclasses
import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.codec import (
    BindingKind,
    Bindings,
    CallableRefused,
    DocumentError,
    KindResolver,
    RunDocument,
    dump_run,
    dump_study,
    load_run,
    load_study,
    preflight_document,
)
from spaceodyssey.codec.records import encode_records
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import I64, EffectKind
from spaceodyssey.core.kinds import AbstractDesign
from spaceodyssey.core.refresh import HostLoopRefit, MaskedBatchRefit
from spaceodyssey.studies import oracle, toy
from spaceodyssey.studies.conformance import CORPUS
from tests.codec import run_cases
from tests.codec.run_cases import (
    KINDS,
    MODEL_HASH,
    bindings,
    boundary,
    host_loop,
    refresh_run,
)
from tests.codec.test_document import TOY_KINDS
from tests.core.preflight_cases import DOC_KINDS, DocPolicy


def _dump(**kw):
    study, registry, service = refresh_run()
    args = dict(
        executor="host",
        bindings=bindings(),
        service=service,
        refit=boundary(),
        kinds=KINDS,
    )
    return dump_run(study, registry, **{**args, **kw})


def _parsed(doc=None):
    """The document as a parsed object, without its hash."""
    data = json.loads((doc or _dump()).to_json())
    data.pop("content_hash")
    return data


def _load(data, **kw):
    return load_run(json.dumps(data), bindings(), KINDS, **kw)


def _refuse(data, pointer, code=None, **kw):
    with pytest.raises(DocumentError) as info:
        _load(data, **kw)
    assert info.value.pointer == pointer, str(info.value)
    assert info.value.code == code, str(info.value)
    return info.value


QUERY_KINDS = {**TOY_KINDS, **DOC_KINDS}


def _query_run():
    """A study whose policy declares a Query, and the service that answers it."""
    registry, kind = oracle.oracle_registry()
    caps = registry.lower(Capacities())
    study = toy.survey_1(caps, policies=(DocPolicy(effects=(int(EffectKind.QUERY),)),))
    return study, registry, oracle.ToyOracleService(kind)


def _query_data():
    study, registry, service = _query_run()
    doc = dump_run(
        study,
        registry,
        executor="host",
        bindings=bindings(),
        service=service,
        kinds=QUERY_KINDS,
    )
    data = json.loads(doc.to_json())
    data.pop("content_hash")
    return data


# ---- round trip ----------------------------


def test_a_run_document_round_trips_study_registry_and_bindings():
    doc = _dump()
    run = load_run(doc.to_json(), bindings(), KINDS)
    study, registry, _ = refresh_run()
    assert dump_study(run.study, KINDS).to_json() == dump_study(study, KINDS).to_json()
    assert encode_records(run.registry) == encode_records(registry)
    assert [run.registry[i].name for i in range(len(run.registry))][-1] == "answer"
    assert run.executor == "host"
    assert run.service["configuration"] == {"record_kind": "answer"}
    assert set(run.refit) == {"host", "compiled"}
    assert type(run.refit["host"]) is MaskedBatchRefit
    assert run.refit["compiled"].model_hash == MODEL_HASH
    assert run.make_service().kind == registry.index("answer")
    again = dump_run(
        run.study,
        run.registry,
        executor=run.executor,
        bindings=bindings(),
        service=run.make_service(),
        refit=run.refit,
        kinds=KINDS,
    )
    assert again.to_json() == doc.to_json()


def test_the_record_keeps_the_service_and_boundary_bindings_in_the_preflight_form():
    data = _parsed()
    assert data["format"] == "spaceodyssey.run_document"
    assert data["run"]["service"] == {
        "kind": "toy.oracle",
        "version": 1,
        "configuration": {"record_kind": "answer"},
    }
    host = data["run"]["refit"]["host"]
    assert host == {
        "kind": "toy.mean_kernel",
        "version": 1,
        "configuration": {"n_diag": 1},
        "model_hash": MODEL_HASH,
    }
    assert data["run"]["refit"]["compiled"] == host
    assert data["study"]["refresh"] if "refresh" in data["study"] else True


def test_each_executor_can_bind_its_own_boundary():
    doc = _dump(refit={"host": host_loop(), "compiled": boundary()})
    # the two boundaries answer one model, but are different kinds
    refit = doc.run["refit"]
    assert refit["host"]["kind"] == "toy.refit_fit"
    assert refit["compiled"]["kind"] == "toy.mean_kernel"
    run = load_run(doc.to_json(), bindings(), KINDS)
    assert type(run.refit["host"]) is HostLoopRefit
    assert type(run.refit["compiled"]) is MaskedBatchRefit


def test_boundaries_that_fit_different_models_are_refused_on_dump():
    other = HostLoopRefit(toy.toy_refit_fit, model_hash=MODEL_HASH + 1)
    with pytest.raises(DocumentError) as info:
        _dump(refit={"host": other, "compiled": boundary()})
    assert info.value.code == "D003"
    assert info.value.pointer == "/run/refit/compiled/model_hash"


def test_a_study_without_a_service_or_refit_dumps_a_bare_run():
    study = toy.survey_1()
    registry = toy.toy_registry()
    doc = dump_run(
        study, registry, executor="compiled", bindings=bindings(), kinds=TOY_KINDS
    )
    assert doc.run == {"executor": "compiled"}
    run = load_run(doc.to_json(), bindings(), TOY_KINDS)
    assert run.service is None and run.refit == {}
    assert run.make_service() is None


# ---- pre-flight consumes the run document ----------------------------


def test_preflight_accepts_a_run_document_object_and_its_text():
    doc = _dump()
    assert preflight_document(doc, KINDS) == []
    assert preflight_document(doc.to_json(), KINDS, bindings=bindings()) == []


def test_preflight_reads_the_run_record_of_a_run_document():
    data = _parsed()
    data["run"]["refit"]["compiled"]["model_hash"] = MODEL_HASH + 1
    found = preflight_document(data, KINDS)
    assert [(str(f.code), f.pointer) for f in found] == [
        ("D003", "/run/refit/compiled/model_hash")
    ]
    data = _query_data()
    assert preflight_document(data, QUERY_KINDS) == []
    del data["run"]["service"]
    found = preflight_document(data, QUERY_KINDS)
    assert [(str(f.code), f.pointer) for f in found] == [("D003", "/run")]


def test_preflight_names_an_unknown_binding_kind_and_a_bad_registry():
    data = _parsed()
    data["run"]["service"]["kind"] = "nobody"
    data["records"][1]["columns"][0][1] = 1
    found = preflight_document(data, KINDS, bindings=bindings())
    assert [(str(f.code), f.pointer) for f in found] == [
        ("D001", "/run/service/kind"),
        ("L001", "/records"),
    ]


def test_preflight_checks_the_hash_of_a_run_document():
    data = _parsed()
    text = json.dumps({**data, "content_hash": "0" * 64})
    found = preflight_document(text, KINDS)
    assert [(str(f.code), f.pointer) for f in found] == [("L001", "/content_hash")]
    clean = json.dumps({**data, "content_hash": _dump().content_hash})
    assert preflight_document(clean, KINDS) == []


def test_preflight_flags_a_callable_member_of_a_run_document():
    data = _parsed()
    data["run"]["wake_fn"] = "my_hook"
    data["run"]["extra"] = 1
    found = preflight_document(data, KINDS)
    assert [(str(f.code), f.pointer) for f in found] == [
        ("D003", "/run/extra"),
        ("D003", "/run/wake_fn"),
    ]
    assert "callable" in found[1].message


# ---- the record-kind registry ----------------------------


def test_the_registry_must_lower_to_the_study_capacities_on_dump():
    study, _, _ = refresh_run()
    other = toy.refresh_registry()
    with pytest.raises(DocumentError) as info:
        dump_run(study, other, executor="host", bindings=bindings(), kinds=KINDS)
    assert info.value.pointer == "/records"


@pytest.mark.parametrize(
    ("edit", "pointer"),
    [
        (lambda r: r.pop(), "/records"),
        (lambda r: r[1]["columns"].pop(), "/records"),
        (lambda r: r[1]["columns"][0].__setitem__(1, 4), "/records"),
        (lambda r: r.__setitem__(0, {"name": "none"}), "/records/0"),
        (lambda r: r[1]["columns"].__setitem__(0, ["value"]), "/records/1/columns"),
        (lambda r: r[1].__setitem__("columns", [["a", 0]]), "/records/1"),
        (lambda r: r.append(copy.deepcopy(r[1])), "/records/5"),
    ],
    ids=[
        "kind-dropped",
        "column-dropped",
        "audience-changed",
        "entry-shape",
        "column-shape",
        "audience-zero",
        "name-repeated",
    ],
)
def test_a_bad_registry_fails_with_its_pointer(edit, pointer):
    data = _parsed()
    edit(data["records"])
    _refuse(data, pointer)


def test_a_registry_that_is_not_a_list_fails():
    data = _parsed()
    data["records"] = {}
    _refuse(data, "/records")


# ---- binding failures ----------------------------


def _set(path, value):
    def edit(data):
        node = data
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return edit


BINDING_FAILURES = {
    "unknown-service-kind": (
        _set(("run", "service", "kind"), "x"),
        "/run/service/kind",
        "D001",
    ),
    "service-version-newer": (
        _set(("run", "service", "version"), 2),
        "/run/service/version",
        "D001",
    ),
    "boundary-version-zero": (
        _set(("run", "refit", "host", "version"), 0),
        "/run/refit/host/version",
        "D003",
    ),
    "unknown-boundary-kind": (
        _set(("run", "refit", "compiled", "kind"), "x"),
        "/run/refit/compiled/kind",
        "D001",
    ),
    "boundary-version-newer": (
        _set(("run", "refit", "host", "version"), 5),
        "/run/refit/host/version",
        "D001",
    ),
    "model-hash-edited": (
        lambda d: (
            _set(("run", "refit", "host", "model_hash"), 9)(d),
            _set(("run", "refit", "compiled", "model_hash"), 9)(d),
        ),
        "/run/refit/compiled/model_hash",
        "D003",
    ),
    "hashes-differ": (
        _set(("run", "refit", "compiled", "model_hash"), 8),
        "/run/refit/compiled/model_hash",
        "D003",
    ),
    "executor-missing": (lambda d: d["run"].pop("executor"), "/run", "D003"),
    "executor-unknown": (_set(("run", "executor"), "gpu"), "/run/executor", "D003"),
    "refit-side-missing": (
        lambda d: d["run"]["refit"].pop("host"),
        "/run/refit",
        "D003",
    ),
    "configuration-missing-key": (
        _set(("run", "service", "configuration"), {}),
        "/run/service/configuration",
        "D003",
    ),
    "configuration-wrong-key": (
        _set(("run", "refit", "host", "configuration"), {"n": 1}),
        "/run/refit/host/configuration",
        "D003",
    ),
    "run-member-unknown": (_set(("run", "other"), 1), "/run/other", "D003"),
    "schema-newer": (_set(("schema_version",), 2), "/schema_version", "D002"),
}


@pytest.mark.parametrize("name", BINDING_FAILURES)
def test_a_bad_binding_fails_with_its_pointer_and_code(name):
    edit, pointer, code = BINDING_FAILURES[name]
    data = _parsed()
    edit(data)
    _refuse(data, pointer, code)


def test_a_missing_service_is_d003_on_dump_and_on_load():
    study, registry, _ = _query_run()
    with pytest.raises(DocumentError) as info:
        dump_run(
            study, registry, executor="host", bindings=bindings(), kinds=QUERY_KINDS
        )
    assert (info.value.pointer, info.value.code) == ("/run", "D003")
    data = _query_data()
    del data["run"]["service"]
    with pytest.raises(DocumentError) as info:
        load_run(json.dumps(data), bindings(), QUERY_KINDS)
    assert (info.value.pointer, info.value.code) == ("/run", "D003")


def test_a_binding_at_an_older_version_without_migrate_is_d001():
    kind = dataclasses.replace(run_cases._masked_kind(), version=2)
    table = Bindings((run_cases._service_kind(), kind))
    with pytest.raises(DocumentError) as info:
        load_run(_dump().to_json(), table, KINDS)
    assert (info.value.pointer, info.value.code) == (
        "/run/refit/compiled/version",
        "D001",
    )


def test_a_binding_at_an_older_version_goes_through_migrate():
    seen = []

    def migrate(version, config):
        seen.append(version)
        return {"n_diag": config["diag"]}

    kind = BindingKind(
        "toy.mean_kernel",
        2,
        matches=lambda obj: False,
        describe=lambda obj, registry: {},
        build=lambda config, registry: MaskedBatchRefit(
            toy.toy_mean_kernel, n_diag=config["n_diag"], model_hash=MODEL_HASH
        ),
        migrate=migrate,
    )
    table = Bindings((run_cases._service_kind(), kind))
    data = _parsed()
    for side in ("host", "compiled"):
        data["run"]["refit"][side]["configuration"] = {"diag": 1}
    run = load_run(json.dumps(data), table, KINDS)
    assert set(seen) == {1} and run.refit["host"].n_diag == 1


def test_a_migrate_that_fails_is_refused_at_the_configuration():
    def migrate(version, config):
        raise KeyError("diag")

    kind = dataclasses.replace(run_cases._masked_kind(), version=2, migrate=migrate)
    table = Bindings((run_cases._service_kind(), kind))
    with pytest.raises(DocumentError) as info:
        load_run(_dump().to_json(), table, KINDS)
    assert info.value.pointer == "/run/refit/compiled/configuration"


def test_an_instance_no_binding_covers_is_refused_on_dump():
    class Stranger:
        model_hash = 0

    with pytest.raises(DocumentError) as info:
        _dump(refit=Stranger())
    assert info.value.pointer == "/run/refit/host"
    assert not isinstance(info.value, CallableRefused)


def test_a_configuration_must_be_plain_json():
    bad = dataclasses.replace(
        run_cases._masked_kind(), describe=lambda obj, registry: {"n": float("nan")}
    )
    table = Bindings((run_cases._service_kind(), bad))
    study, registry, _ = refresh_run()
    with pytest.raises(DocumentError) as info:
        dump_run(
            study,
            registry,
            executor="host",
            bindings=table,
            refit=boundary(),
            kinds=KINDS,
        )
    assert info.value.pointer == "/run/refit/host/configuration/n"


def test_two_binding_kinds_with_one_name_fail_to_register():
    with pytest.raises(ValueError, match="twice"):
        Bindings((run_cases._masked_kind(), run_cases._masked_kind()))


# ---- callables ----------------------------


class ActivationDesign(AbstractDesign):
    """A design that holds an activation function."""

    act: object = eqx.field(static=True, default=jax.nn.relu)
    gain: object = 1.0


def _with_design(design):
    return dataclasses.replace(toy.survey_1(), design=design)


def test_an_activation_function_is_refused_with_a_named_finding_and_pointer():
    kinds = {**TOY_KINDS, ("design", "test.activation", 1): ActivationDesign}
    with pytest.raises(CallableRefused) as info:
        dump_study(_with_design(ActivationDesign()), kinds)
    assert info.value.pointer == "/study/design/act"
    assert "callable refused" in str(info.value)
    assert isinstance(info.value, DocumentError)


def test_a_lambda_in_a_module_field_is_refused():
    kinds = {**TOY_KINDS, ("design", "test.activation", 1): ActivationDesign}
    with pytest.raises(CallableRefused) as info:
        dump_study(_with_design(ActivationDesign(act=lambda x: x)), kinds)
    assert info.value.pointer == "/study/design/act"
    assert "<lambda>" in info.value.name


def test_a_wake_fn_is_refused_on_dump():
    with pytest.raises(CallableRefused) as info:
        _dump(wake_fn=lambda carry, ev, study: None)
    assert info.value.pointer == "/run/wake_fn"
    assert info.value.name == "wake_fn"


def test_a_boundary_holding_an_unregistered_function_is_refused_by_name():
    with pytest.raises(CallableRefused) as info:
        _dump(refit=HostLoopRefit(lambda study, entry: None))
    assert info.value.pointer == "/run/refit/host"
    assert "HostLoopRefit.fit" in info.value.name
    with pytest.raises(CallableRefused) as info:
        _dump(refit=MaskedBatchRefit(lambda study, entry: None))
    assert "MaskedBatchRefit.kernel" in info.value.name


def test_a_configuration_holding_a_function_is_refused():
    bad = dataclasses.replace(
        run_cases._masked_kind(), describe=lambda obj, registry: {"act": jax.nn.relu}
    )
    study, registry, _ = refresh_run()
    with pytest.raises(CallableRefused) as info:
        dump_run(
            study,
            registry,
            executor="host",
            refit=boundary(),
            kinds=KINDS,
            bindings=Bindings((bad,)),
        )
    assert info.value.pointer == "/run/refit/host/configuration/act"


@pytest.mark.parametrize(
    "member", ["wake_fn", "decide_py", "activation", "fit", "kernel"]
)
def test_a_run_record_naming_code_is_refused_on_load(member):
    data = _parsed()
    data["run"][member] = "module.function"
    with pytest.raises(CallableRefused) as info:
        _load(data)
    assert info.value.pointer == f"/run/{member}"


@pytest.mark.parametrize("tag", ["$callable", "$function", "$lambda", "$fn"])
def test_a_document_value_tagged_as_code_is_refused_on_load(tag):
    data = _parsed()
    data["study"]["design"]["gain"] = {tag: "module.function"}
    with pytest.raises(CallableRefused) as info:
        _load(data)
    assert info.value.pointer == "/study/design/gain"


# ---- file references ----------------------------


class PlanTable(AbstractDesign):
    """A design holding a large table, as a plan does."""

    table: jax.Array
    weak: jax.Array = eqx.field(default_factory=lambda: jnp.asarray(1.5))


PLAN_KINDS = {**TOY_KINDS, ("design", "test.plan_table", 1): PlanTable}


def _plan_study(n=300):
    values = np.linspace(-1.0, 1.0, n)
    values[3], values[4], values[5] = np.nan, -0.0, -np.inf
    return _with_design(PlanTable(table=jnp.asarray(values)))


def test_a_large_array_is_a_file_reference_with_a_sha256(tmp_path):
    study = _plan_study()
    doc = dump_study(study, PLAN_KINDS, files=tmp_path, max_inline=64)
    node = doc.study["design"]["table"]
    assert "data" not in node and node["shape"] == [300]
    ref = node["file"]
    assert ref["name"] == f"{ref['sha256']}.bin" and ref["bytes"] == 300 * 8
    assert (tmp_path / ref["name"]).read_bytes() == np.asarray(
        study.design.table
    ).tobytes()
    assert "file" not in doc.study["design"]["weak"]
    again = load_study(doc.to_json(), PLAN_KINDS, files=tmp_path)
    assert (
        np.asarray(again.design.table).tobytes()
        == np.asarray(study.design.table).tobytes()
    )
    assert (
        dump_study(again, PLAN_KINDS, files=tmp_path, max_inline=64).to_json()
        == doc.to_json()
    )


def test_the_document_hash_covers_the_file_content(tmp_path):
    first = dump_study(_plan_study(), PLAN_KINDS, files=tmp_path, max_inline=64)
    other = _plan_study()
    other = eqx.tree_at(lambda s: s.design.table, other, other.design.table + 1.0)
    second = dump_study(other, PLAN_KINDS, files=tmp_path, max_inline=64)
    assert first.content_hash != second.content_hash
    inline = dump_study(_plan_study(), PLAN_KINDS)
    assert inline.content_hash != first.content_hash


def test_every_corpus_study_round_trips_through_files(tmp_path):
    # max_inline 0 sends every array with an element to a file, scalars included
    for case in CORPUS:
        study = case.make_study()
        doc = dump_study(study, TOY_KINDS, files=tmp_path, max_inline=0)
        again = load_study(doc.to_json(), TOY_KINDS, files=tmp_path)
        text = dump_study(again, TOY_KINDS, files=tmp_path, max_inline=0).to_json()
        assert text == doc.to_json(), case.name


def _file_doc(tmp_path):
    doc = dump_study(_plan_study(), PLAN_KINDS, files=tmp_path, max_inline=64)
    data = json.loads(doc.to_json())
    data.pop("content_hash")
    return data, tmp_path / doc.study["design"]["table"]["file"]["name"]


def _load_files(data, tmp_path):
    return load_study(json.dumps(data), PLAN_KINDS, files=tmp_path)


def _pointer_of(call):
    with pytest.raises(DocumentError) as info:
        call()
    return info.value.pointer


def test_a_missing_file_is_refused_with_a_pointer(tmp_path):
    data, path = _file_doc(tmp_path)
    path.unlink()
    where = _pointer_of(lambda: _load_files(data, tmp_path))
    assert where == "/study/design/table/file/name"


def test_a_file_with_the_wrong_hash_is_refused_with_a_pointer(tmp_path):
    data, path = _file_doc(tmp_path)
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1
    path.write_bytes(bytes(raw))
    where = _pointer_of(lambda: _load_files(data, tmp_path))
    assert where == "/study/design/table/file/sha256"


def test_a_file_of_the_wrong_size_is_refused_with_a_pointer(tmp_path):
    data, path = _file_doc(tmp_path)
    path.write_bytes(path.read_bytes()[:-8])
    where = _pointer_of(lambda: _load_files(data, tmp_path))
    assert where == "/study/design/table/file/bytes"


def test_a_document_with_file_references_needs_the_directory(tmp_path):
    data, _ = _file_doc(tmp_path)
    where = _pointer_of(lambda: load_study(json.dumps(data), PLAN_KINDS))
    assert where == "/study/design/table/file"


@pytest.mark.parametrize(
    ("edit", "pointer"),
    [
        (lambda r: r.__setitem__("name", "../x.bin"), "/study/design/table/file/name"),
        (
            lambda r: r.__setitem__("name", "a" * 64 + ".bin"),
            "/study/design/table/file/sha256",
        ),
        (lambda r: r.__setitem__("bytes", -1), "/study/design/table/file/bytes"),
        (lambda r: r.pop("bytes"), "/study/design/table/file"),
        (lambda r: r.__setitem__("extra", 1), "/study/design/table/file"),
    ],
    ids=["path", "name-not-hash", "size", "member-missing", "member-extra"],
)
def test_a_malformed_reference_is_refused_with_a_pointer(tmp_path, edit, pointer):
    data, _ = _file_doc(tmp_path)
    edit(data["study"]["design"]["table"]["file"])
    assert _pointer_of(lambda: _load_files(data, tmp_path)) == pointer


def test_a_reference_whose_size_does_not_fit_the_shape_is_refused(tmp_path):
    data, _ = _file_doc(tmp_path)
    data["study"]["design"]["table"]["shape"] = [299]
    where = _pointer_of(lambda: _load_files(data, tmp_path))
    assert where == "/study/design/table/file/bytes"


def test_an_array_holds_data_or_a_file_not_both(tmp_path):
    data, _ = _file_doc(tmp_path)
    data["study"]["design"]["table"]["data"] = [0.0] * 300
    assert _pointer_of(lambda: _load_files(data, tmp_path)) == "/study/design/table"


def test_a_run_document_round_trips_with_files(tmp_path):
    study, registry, service = refresh_run()
    doc = dump_run(
        study,
        registry,
        executor="host",
        bindings=bindings(),
        service=service,
        refit=boundary(),
        kinds=KINDS,
        files=tmp_path,
        max_inline=0,
    )
    assert any(tmp_path.iterdir())
    run = load_run(doc.to_json(), bindings(), KINDS, files=tmp_path)
    assert run.registry is not None and run.executor == "host"
    with pytest.raises(DocumentError):
        load_run(doc.to_json(), bindings(), KINDS)


# ---- no switch ----------------------------


def _phase_switch_study():
    case = next(c for c in CORPUS if c.name == "phase_switch")
    return case.make_study()


def _never_switching():
    study = _phase_switch_study()
    top = study.caps.max_tick
    agenda = dataclasses.replace(study.agenda, phase_until=jnp.asarray((top, top), I64))
    return dataclasses.replace(study, agenda=agenda)


def test_no_switch_is_written_back_and_lowered_to_the_capacity_tick():
    study = _never_switching()
    doc = dump_study(study, TOY_KINDS)
    data = doc.study["agenda"]["phase_until"]["data"]
    assert data == [{"no_switch": True}, {"no_switch": True}]
    again = load_study(doc.to_json(), TOY_KINDS)
    assert again.agenda.phase_until.tolist() == [study.caps.max_tick] * 2
    assert again.agenda.phase_until.dtype == jnp.int64
    assert dump_study(again, TOY_KINDS).to_json() == doc.to_json()


def test_a_mixed_boundary_list_lowers_only_the_tagged_entry():
    study = _phase_switch_study()
    doc = json.loads(dump_study(study, TOY_KINDS).to_json())
    doc.pop("content_hash")
    doc["study"]["agenda"]["phase_until"]["data"] = [300, {"no_switch": True}]
    again = load_study(json.dumps(doc), TOY_KINDS)
    assert again.agenda.phase_until.tolist() == [300, study.caps.max_tick]
    assert dump_study(again, TOY_KINDS).study["agenda"]["phase_until"]["data"] == [
        300,
        {"no_switch": True},
    ]


def test_a_literal_capacity_tick_is_written_back_as_the_tag():
    study = _phase_switch_study()
    doc = json.loads(dump_study(study, TOY_KINDS).to_json())
    doc.pop("content_hash")
    doc["study"]["agenda"]["phase_until"]["data"] = [300, study.caps.max_tick]
    again = load_study(json.dumps(doc), TOY_KINDS)
    out = dump_study(again, TOY_KINDS).study["agenda"]["phase_until"]["data"]
    assert out == [300, {"no_switch": True}]


@pytest.mark.parametrize(
    "value",
    [
        {"no_switch": False},
        {"no_switch": 1},
        {"never": True},
        {"no_switch": True, "x": 1},
    ],
)
def test_a_malformed_no_switch_tag_is_refused_with_a_pointer(value):
    doc = json.loads(dump_study(_phase_switch_study(), TOY_KINDS).to_json())
    doc.pop("content_hash")
    doc["study"]["agenda"]["phase_until"]["data"][0] = value
    with pytest.raises(DocumentError) as info:
        load_study(json.dumps(doc), TOY_KINDS)
    assert info.value.pointer == "/study/agenda/phase_until/data/0"


def test_a_horizon_at_the_capacity_tick_is_refused_on_load_and_dump():
    study = toy.survey_1()
    top = study.caps.max_tick
    doc = json.loads(dump_study(study, TOY_KINDS).to_json())
    doc.pop("content_hash")
    for horizon in (top, top + 1):
        doc["study"]["horizon_tick"] = {"$array": "int64", "shape": [], "data": horizon}
        with pytest.raises(DocumentError) as info:
            load_study(json.dumps(doc), TOY_KINDS)
        assert info.value.pointer == "/study/horizon_tick"
    doc["study"]["horizon_tick"]["data"] = top - 1
    assert load_study(json.dumps(doc), TOY_KINDS).horizon_tick == top - 1
    with pytest.raises(DocumentError) as info:
        dump_study(dataclasses.replace(study, horizon_tick=top), TOY_KINDS)
    assert info.value.pointer == "/study/horizon_tick"


def test_no_switch_inside_a_run_document_round_trips():
    study = _never_switching()
    registry = toy.toy_registry()
    doc = dump_run(
        study, registry, executor="host", bindings=bindings(), kinds=TOY_KINDS
    )
    assert "no_switch" in doc.to_json()
    run = load_run(doc.to_json(), bindings(), TOY_KINDS)
    assert run.study.agenda.phase_until.tolist() == [study.caps.max_tick] * 2


# ---- the envelope ----------------------------


def test_run_document_text_is_not_a_study_document_and_the_reverse():
    text = _dump().to_json()
    with pytest.raises(DocumentError) as info:
        load_study(text, KINDS)
    assert info.value.pointer == "/format" or info.value.pointer.startswith("/")
    study_text = dump_study(toy.survey_1(), TOY_KINDS).to_json()
    with pytest.raises(DocumentError) as info:
        RunDocument.from_json(study_text)
    assert info.value.pointer in ("/format", "")


def test_a_hand_edited_run_hash_is_refused():
    data = _parsed()
    data["run"]["executor"] = "compiled"
    with pytest.raises(DocumentError) as info:
        RunDocument.from_json(json.dumps({**data, "content_hash": "0" * 64}))
    assert info.value.pointer == "/content_hash"


def test_the_hash_covers_the_run_record_and_the_registry():
    base = _dump().content_hash
    assert _dump(executor="compiled").content_hash != base
    assert _dump(refit=None).content_hash != base


def test_the_kinds_resolver_is_accepted_wherever_a_mapping_is():
    resolver = KindResolver(KINDS)
    run = load_run(_dump().to_json(), bindings(), resolver)
    assert preflight_document(_dump(), resolver) == []
    assert run.executor == "host"


# ---- file safety ------------------------------------------------------------------


def _stored(tmp_path):
    """A file-backed document, the reference of its table and the file's path."""
    doc = dump_study(_plan_study(), PLAN_KINDS, files=tmp_path, max_inline=64)
    ref = doc.study["design"]["table"]["file"]
    return doc, ref, tmp_path / ref["name"]


def test_dump_never_writes_through_a_planted_symlink(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    raw = np.asarray(_plan_study().design.table).tobytes()
    import hashlib

    name = f"{hashlib.sha256(raw).hexdigest()}.bin"
    # the temporary names an earlier version used, and the final name, as symlinks
    for planted in (f".{name}.{os.getpid()}.tmp", f".{name}.tmp", name):
        (tmp_path / planted).symlink_to(victim)
    doc, _, path = _stored(tmp_path)
    assert victim.read_text() == "keep"
    assert not path.is_symlink() and path.read_bytes() == raw
    again = load_study(doc.to_json(), PLAN_KINDS, files=tmp_path)
    assert np.asarray(again.design.table).tobytes() == raw


def test_a_same_length_corruption_is_healed_by_the_next_dump(tmp_path):
    _, _, path = _stored(tmp_path)
    good = path.read_bytes()
    path.write_bytes(bytes([good[0] ^ 1]) + good[1:])
    doc = dump_study(_plan_study(), PLAN_KINDS, files=tmp_path, max_inline=64)
    assert path.read_bytes() == good
    load_study(doc.to_json(), PLAN_KINDS, files=tmp_path)


def test_a_wrong_length_file_is_replaced_by_the_next_dump(tmp_path):
    _, _, path = _stored(tmp_path)
    good = path.read_bytes()
    path.write_bytes(good[:-8])
    dump_study(_plan_study(), PLAN_KINDS, files=tmp_path, max_inline=64)
    assert path.read_bytes() == good


def test_load_refuses_a_symlink_even_when_its_target_matches(tmp_path):
    doc, _, path = _stored(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "copy").write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(elsewhere / "copy")
    where = _pointer_of(lambda: load_study(doc.to_json(), PLAN_KINDS, files=tmp_path))
    assert where == "/study/design/table/file/name"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX named pipes")
def test_load_refuses_a_fifo_without_blocking(tmp_path):
    doc, _, path = _stored(tmp_path)
    path.unlink()
    os.mkfifo(path)
    where = _pointer_of(lambda: load_study(doc.to_json(), PLAN_KINDS, files=tmp_path))
    assert where == "/study/design/table/file/name"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX named pipes")
def test_dump_replaces_a_fifo_at_the_digest_name(tmp_path):
    _, _, path = _stored(tmp_path)
    good = path.read_bytes()
    path.unlink()
    os.mkfifo(path)
    dump_study(_plan_study(), PLAN_KINDS, files=tmp_path, max_inline=64)
    assert path.is_file() and path.read_bytes() == good


def test_oversize_and_undersize_files_are_refused_by_size(tmp_path):
    doc, _, path = _stored(tmp_path)
    good = path.read_bytes()
    for raw in (good + b"\0" * 8, good[:-8], b""):
        path.write_bytes(raw)
        where = _pointer_of(
            lambda: load_study(doc.to_json(), PLAN_KINDS, files=tmp_path)
        )
        assert where == "/study/design/table/file/bytes"


def test_a_declared_size_is_checked_before_any_file_is_opened(tmp_path):
    doc, _, path = _stored(tmp_path)
    data = json.loads(doc.to_json())
    data.pop("content_hash")
    data["study"]["design"]["table"]["file"]["bytes"] = 1
    path.unlink()  # a missing file would be `name`; the size is judged first
    where = _pointer_of(
        lambda: load_study(json.dumps(data), PLAN_KINDS, files=tmp_path)
    )
    assert where == "/study/design/table/file/bytes"


def test_the_byte_limit_refuses_before_reading(tmp_path):
    doc, ref, _ = _stored(tmp_path)
    where = _pointer_of(
        lambda: load_study(
            doc.to_json(), PLAN_KINDS, files=tmp_path, max_file_bytes=ref["bytes"] - 1
        )
    )
    assert where == "/study/design/table/file/bytes"
    load_study(doc.to_json(), PLAN_KINDS, files=tmp_path, max_file_bytes=ref["bytes"])


# ---- pre-flight and load agree -----------------------------------------------------


def test_preflight_reads_file_backed_documents_and_checks_their_integrity(tmp_path):
    doc, _, path = _stored(tmp_path)
    assert preflight_document(doc, PLAN_KINDS, files=tmp_path) == []
    assert [f.pointer for f in preflight_document(doc, PLAN_KINDS)] == [
        "/study/design/table/file"
    ]
    good = path.read_bytes()
    path.write_bytes(bytes([good[0] ^ 1]) + good[1:])
    found = preflight_document(doc, PLAN_KINDS, files=tmp_path)
    assert [(str(f.code), f.pointer) for f in found] == [
        ("L001", "/study/design/table/file/sha256")
    ]


def _first_error(data):
    """What pre-flight lists first, and what `load_run` raises, for one document."""
    found = preflight_document(data, KINDS, bindings=bindings())
    errors = [f for f in found if f.severity == "error"]
    with pytest.raises(DocumentError) as info:
        _load(data)
    return errors[0], info.value


PARITY_EXTRA = {
    "run-null": (_set(("run",), None), "/run"),
    "service-unknown-and-executor-bad": (
        lambda d: (
            _set(("run", "service", "kind"), "nobody")(d),
            _set(("run", "executor"), "gpu")(d),
        ),
        "/run/service/kind",
    ),
    "wake-fn-member": (_set(("run", "wake_fn"), "hook"), "/run/wake_fn"),
    "callable-tag": (
        _set(("study", "design", "gain"), {"$lambda": "f"}),
        "/study/design/gain",
    ),
    "records-bad": (lambda d: d["records"].pop(), "/records"),
}


@pytest.mark.parametrize("name", [*BINDING_FAILURES, *PARITY_EXTRA])
def test_load_run_raises_the_first_finding_preflight_lists(name):
    edit, pointer, *_ = (BINDING_FAILURES | PARITY_EXTRA)[name]
    data = _parsed()
    edit(data)
    first, raised = _first_error(data)
    assert raised.pointer == first.pointer
    if not isinstance(raised, CallableRefused):
        assert raised.detail == first.message
    expected = None if str(first.code) == "L001" else str(first.code)
    assert raised.code == expected or isinstance(raised, CallableRefused)
    if name in PARITY_EXTRA:
        assert first.pointer == pointer


def test_preflight_rebuilds_boundaries_like_load_run():
    data = _parsed()
    for side in ("host", "compiled"):
        data["run"]["refit"][side]["model_hash"] = 9
    found = preflight_document(data, KINDS, bindings=bindings())
    assert [(str(f.code), f.pointer) for f in found] == [
        ("D003", "/run/refit/compiled/model_hash"),
        ("D003", "/run/refit/host/model_hash"),
    ]
    data = _parsed()
    data["run"]["refit"]["host"]["configuration"] = {}
    found = preflight_document(data, KINDS, bindings=bindings())
    assert [(str(f.code), f.pointer) for f in found] == [
        ("D003", "/run/refit/host/configuration")
    ]
    assert preflight_document(_parsed(), KINDS, bindings=bindings()) == []


# ---- the registry is hashed -------------------------------------------------------


def _mutated(edit):
    doc = _dump()
    records = copy.deepcopy(doc.records)
    edit(records)
    return dataclasses.replace(doc, records=records)


REGISTRY_MUTATIONS = {
    "kind-name": lambda r: r[1].__setitem__("name", "renamed"),
    "audience": lambda r: r[1]["columns"][0].__setitem__(1, 1),
    "derived-flag": lambda r: r[1].__setitem__("derived", True),
    "column-name": lambda r: r[1]["columns"][0].__setitem__(0, "renamed"),
}


@pytest.mark.parametrize("name", REGISTRY_MUTATIONS)
def test_every_registry_field_is_in_the_content_hash(name):
    base = _dump()
    other = _mutated(REGISTRY_MUTATIONS[name])
    assert other.content_hash != base.content_hash
    assert json.loads(other.to_json())["content_hash"] == other.content_hash


@pytest.mark.parametrize("name", ["kind-name", "audience", "derived-flag"])
def test_a_registry_edit_that_changes_the_lowering_is_refused(name):
    other = _mutated(REGISTRY_MUTATIONS[name])
    with pytest.raises(DocumentError) as info:
        load_run(other.to_json(), bindings(), KINDS)
    assert info.value.pointer.startswith("/records")


def test_a_column_name_the_lowering_drops_loads_but_changes_the_hash():
    other = _mutated(REGISTRY_MUTATIONS["column-name"])
    run = load_run(other.to_json(), bindings(), KINDS)
    assert run.registry[1].columns[0].name == "renamed"


# ---- review round two ---------------------------------------------------------------


@pytest.mark.parametrize("flag", ["O_NOFOLLOW", "O_NONBLOCK", "O_DIRECTORY"])
def test_a_platform_without_a_safe_open_flag_refuses_files(tmp_path, monkeypatch, flag):
    doc, _, path = _stored(tmp_path)
    elsewhere = tmp_path / "copy"
    elsewhere.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(elsewhere)  # would load if the symlink check were skipped
    monkeypatch.delattr(os, flag)
    with pytest.raises(DocumentError) as info:
        load_study(doc.to_json(), PLAN_KINDS, files=tmp_path)
    assert flag in str(info.value)
    assert info.value.pointer == "/study/design/table/file/name"


def test_a_platform_without_dir_fd_refuses_files(tmp_path, monkeypatch):
    doc, _, _ = _stored(tmp_path)
    monkeypatch.setattr(os, "supports_dir_fd", set())
    with pytest.raises(DocumentError) as info:
        load_study(doc.to_json(), PLAN_KINDS, files=tmp_path)
    assert "dir_fd" in str(info.value)
    assert info.value.pointer == "/study/design/table/file/name"


def test_a_file_that_grows_while_it_is_read_is_refused(tmp_path, monkeypatch):
    doc, _, path = _stored(tmp_path)
    real = os.read
    grown = []

    def read_then_append(fd, count):
        out = real(fd, count)
        if not grown:
            grown.append(True)
            with path.open("ab") as handle:
                handle.write(b"x")
        return out

    monkeypatch.setattr(os, "read", read_then_append)
    where = _pointer_of(lambda: load_study(doc.to_json(), PLAN_KINDS, files=tmp_path))
    assert where == "/study/design/table/file/bytes"


def test_a_stateful_builder_cannot_change_the_model_hash_after_validation():
    calls = []
    base = run_cases._masked_kind()

    def build(config, registry):
        calls.append(1)
        hashed = MODEL_HASH if len(calls) <= 2 else 999
        return MaskedBatchRefit(toy.toy_mean_kernel, model_hash=hashed)

    kind = dataclasses.replace(base, build=build)
    table = Bindings((run_cases._service_kind(), kind))
    with pytest.raises(DocumentError) as info:
        load_run(_dump().to_json(), table, KINDS)
    assert info.value.code == "D003"
    assert info.value.pointer.endswith("/model_hash")


def test_a_missing_schema_version_or_study_is_a_document_error_not_a_key_error():
    for missing, pointer in (("schema_version", ""), ("study", "")):
        data = _parsed()
        data.pop(missing)
        text = json.dumps({**data, "content_hash": "0" * 64})
        with pytest.raises(DocumentError) as info:
            load_run(text, bindings(), KINDS)
        assert info.value.pointer == pointer
        found = preflight_document(text, KINDS)
        assert found and found[0].pointer == pointer


@pytest.fixture(scope="module")
def allocator_recipe():
    """The allocator document and decoded recipe, with explicit registrations."""
    from tests.studies.scenarios import allocator

    kinds = {
        **TOY_KINDS,
        ("process", "test.proposals", 1): allocator.ProposalRecords,
        ("policy", "test.allocation", 1): allocator.AllocationPolicy,
    }
    table = Bindings(
        (
            BindingKind(
                "test.committee",
                1,
                cls=allocator.ToyCommittee,
                describe=lambda obj, registry: {"record_kind": registry[obj.kind].name},
                build=lambda config, registry: allocator.ToyCommittee(
                    registry.index(config["record_kind"])
                ),
            ),
        )
    )
    study = allocator.allocator_study()
    doc = dump_run(
        study,
        allocator.allocator_registry(),
        executor="host",
        bindings=table,
        service=allocator.ToyCommittee(),
        kinds=kinds,
    )
    loaded = load_run(doc.to_json(), table, kinds)
    return study, doc, loaded, kinds, table


def test_the_allocator_recipe_round_trips_process_policy_and_committee(
    allocator_recipe,
):
    """Fails if the allocator loses proposal configuration, policy type or committee
    binding on a run document round trip.
    """
    from tests.studies.scenarios import allocator

    study, doc, loaded, kinds, table = allocator_recipe
    assert type(loaded.study.process) is allocator.ProposalRecords
    for name in (
        "ticks",
        "values",
        "ids",
        "revisions",
        "statuses",
        "kinds",
        "programs",
        "audiences",
        "releases",
    ):
        assert getattr(loaded.study.process, name) == getattr(study.process, name)
    assert loaded.study.process.programs == tuple(
        p[1] for p in allocator.DEFAULT_PROPOSALS
    )
    assert loaded.study.process.audiences == tuple(
        int(p[3]) for p in allocator.DEFAULT_PROPOSALS
    )
    assert loaded.study.process.releases == tuple(
        p[4] for p in allocator.DEFAULT_PROPOSALS
    )
    assert type(loaded.study.agenda.policies[0]) is allocator.AllocationPolicy
    assert (
        loaded.study.agenda.policies[0].declare() == study.agenda.policies[0].declare()
    )
    service = loaded.make_service()
    assert type(service) is allocator.ToyCommittee
    assert service.kind == loaded.registry.index("award")
    assert loaded.service == {
        "kind": "test.committee",
        "version": 1,
        "configuration": {"record_kind": "award"},
    }
    again = dump_run(
        loaded.study,
        loaded.registry,
        executor=loaded.executor,
        bindings=table,
        service=service,
        refit=loaded.refit or None,
        kinds=kinds,
    )
    assert again.to_json() == doc.to_json()


def test_the_allocator_registry_preserves_order_columns_and_flags(allocator_recipe):
    """Fails if the allocator registry changes kind order, payload names, widths,
    audiences or derived flags on decoding.
    """
    _, doc, loaded, _, _ = allocator_recipe
    rows = encode_records(loaded.registry)
    assert rows == doc.records
    assert [row["name"] for row in rows] == [
        "none",
        "measurement",
        "auxiliary",
        "proposal",
        "award",
    ]
    assert rows[3:] == [
        {
            "name": "proposal",
            "derived": False,
            "columns": [["request", 2], ["program", 2]],
        },
        {
            "name": "award",
            "derived": False,
            "columns": [
                ["round", 2],
                ["window_from", 2],
                ["window_until", 2],
                ["amount_0", 2],
                ["amount_1", 2],
            ],
        },
    ]
    assert [len(row["columns"]) for row in rows[3:]] == [2, 5]
    assert loaded.study.caps.payload_width == 5
