"""The optional-field rule of the codec: a field added after the first document format.

An optional field (`OPTIONAL` metadata) is omitted by the encoder at its default and
filled by the loader when missing, so every document written before the field existed
keeps its hash. A field written at its default is refused, so one study has one
document. The corpus hashes themselves are pinned in `test_run_document_runs.py`.
"""

import dataclasses
import json

import jax.numpy as jnp
import pytest

from spaceodyssey.codec import DocumentError, dump_study, load_study
from spaceodyssey.codec import dump as dump_module
from spaceodyssey.codec import load as load_module
from spaceodyssey.codec.dump import encode
from spaceodyssey.codec.kinds import (
    RegistrationError,
    as_resolver,
    at_default,
    is_optional_field,
    optional_default,
)
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import I64, ChargePricing
from spaceodyssey.core.ledger import ChannelTable
from spaceodyssey.core.records import Record, record_none
from spaceodyssey.core.snapshot import EffectTable, effects_none
from spaceodyssey.core.study import Study
from spaceodyssey.studies import toy
from spaceodyssey.studies.conformance import CORPUS
from tests.codec.test_document import TOY_KINDS, assert_same_pytree

CAP_FIELDS = (
    "awards",
    "refunds",
    "n_commitments",
    "snapshot_balances",
    "balance_quantum",
)
STUDY_FIELDS = ("key_version",)
CHANNEL_FIELDS = ("pricing", "surcharge", "failure_kind", "refund_deadline_ticks")
RECORD_FIELDS = ("nonrefundable", "refund_index", "award_gen", "award_gens")
AWARD_COLUMNS = (
    "amount",
    "window_from",
    "window_until",
    "recipient",
    "remainder",
    "realloc_to",
)
EFFECT_OPTIONAL = ("program", *AWARD_COLUMNS)


def _text(study):
    return dump_study(study, TOY_KINDS).to_json()


def _doc(study):
    doc = json.loads(_text(study))
    doc.pop("content_hash")  # a document without a hash loads unchecked
    return doc


def test_the_optional_fields_are_the_ones_slice_3a_added():
    """Fails if a field gains or loses the marker: it would move every document."""
    marked = {
        (cls.__name__, f.name)
        for cls in (Capacities, Study, EffectTable, ChannelTable, Record)
        for f in dataclasses.fields(cls)
        if is_optional_field(f)
    }
    assert marked == (
        {("Capacities", n) for n in CAP_FIELDS}
        | {("Study", n) for n in STUDY_FIELDS}
        | {("EffectTable", n) for n in EFFECT_OPTIONAL}
        | {("ChannelTable", n) for n in CHANNEL_FIELDS}
        | {("Record", n) for n in RECORD_FIELDS}
    )


def test_the_encoder_omits_every_optional_field_at_its_default():
    """Fails if a default study writes a field its older documents did not hold."""
    doc = _doc(toy.survey_1())
    assert not set(STUDY_FIELDS) & set(doc["study"])
    assert not set(CAP_FIELDS) & set(doc["study"]["caps"])
    assert "n_programs" in doc["study"]["caps"]  # a field with a default, not optional
    assert not set(CHANNEL_FIELDS) & set(doc["study"]["channels"])


def test_no_corpus_document_holds_an_optional_field():
    """Fails if any corpus study writes an optional field (its hash would move)."""
    for case in CORPUS:
        doc = _doc(case.make_study())
        assert not set(STUDY_FIELDS) & set(doc["study"]), case.name
        assert not set(CAP_FIELDS) & set(doc["study"]["caps"]), case.name
        assert not set(CHANNEL_FIELDS) & set(doc["study"]["channels"]), case.name


def test_a_nondefault_optional_field_is_written_and_round_trips():
    """Fails if a value off its default is dropped, changed or not rebuilt exactly."""
    caps = dataclasses.replace(
        Capacities(),
        awards=True,
        refunds=True,
        n_commitments=3,
        snapshot_balances=True,
        balance_quantum=0.5,
    )
    study = toy.survey_1(caps=caps, key_version=2)
    doc = _doc(study)
    assert doc["study"]["key_version"] == 2
    assert [doc["study"]["caps"][n] for n in CAP_FIELDS] == [
        True,
        True,
        3,
        True,
        0.5,
    ]
    text = _text(study)
    loaded = load_study(text, TOY_KINDS)
    assert_same_pytree(study, loaded)
    assert _text(loaded) == text


def test_a_missing_optional_field_loads_at_its_default():
    """Fails if the loader refuses a document that omits an optional field, or fills
    it with anything but the default.
    """
    study = toy.survey_1()
    loaded = load_study(json.dumps(_doc(study)), TOY_KINDS)
    assert loaded.key_version == 1
    assert [getattr(loaded.caps, n) for n in CAP_FIELDS] == [
        False,
        False,
        0,
        False,
        1.0,
    ]
    assert_same_pytree(study, loaded)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("key_version",), 1),
        (("caps", "awards"), False),
        (("caps", "n_commitments"), 0),
        (("caps", "snapshot_balances"), False),
        (("caps", "balance_quantum"), 1.0),
    ],
    ids=["study", "caps-bool", "caps-int", "caps-balances", "caps-quantum"],
)
def test_a_field_written_at_its_default_is_refused(path, value):
    """Fails if a second document for the same study loads: it would hash apart."""
    doc = _doc(toy.survey_1())
    node = doc["study"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(DocumentError, match="at its default") as err:
        load_study(json.dumps(doc), TOY_KINDS)
    assert err.value.pointer == "/study/" + "/".join(path)


def test_a_wrong_typed_optional_value_is_refused_not_taken_for_the_default():
    """Fails if True loads as the integer default 1, or 1 as the bool default."""
    doc = _doc(toy.survey_1(key_version=2))
    doc["study"]["key_version"] = True
    with pytest.raises(DocumentError):
        load_study(json.dumps(doc), TOY_KINDS)
    doc = _doc(toy.survey_1())
    doc["study"]["caps"]["awards"] = 0
    with pytest.raises(DocumentError):
        load_study(json.dumps(doc), TOY_KINDS)


@pytest.mark.parametrize("name", ["snapshot_balances", "balance_quantum"])
def test_an_explicit_null_balance_field_is_refused(name):
    """Fails if `null` loads as the default of a balance field."""
    caps = Capacities(snapshot_balances=True, balance_quantum=0.5)
    doc = _doc(toy.survey_1(caps=caps))
    doc["study"]["caps"][name] = None
    with pytest.raises(DocumentError) as err:
        load_study(json.dumps(doc), TOY_KINDS)
    assert err.value.pointer.startswith("/study/caps")


def test_a_required_field_still_cannot_be_missing():
    """Fails if optionality leaks to a field without the marker."""
    doc = _doc(toy.survey_1())
    del doc["study"]["refresh"]  # has a default, but is not optional
    with pytest.raises(DocumentError, match="'refresh' is missing"):
        load_study(json.dumps(doc), TOY_KINDS)
    doc = _doc(toy.survey_1())
    del doc["study"]["caps"]["n_programs"]
    with pytest.raises(DocumentError, match="'n_programs' is missing"):
        load_study(json.dumps(doc), TOY_KINDS)


def test_an_unknown_field_is_still_refused():
    """Fails if the loader starts to accept fields no declaration holds."""
    doc = _doc(toy.survey_1())
    doc["study"]["caps"]["awards_pending"] = 1
    with pytest.raises(DocumentError):
        load_study(json.dumps(doc), TOY_KINDS)


def test_control_an_encoder_that_keeps_defaults_moves_the_document(monkeypatch):
    """Control: the omission is what holds the hash; without it the text differs."""
    study = toy.survey_1()
    kept = _text(study)
    monkeypatch.setattr(dump_module, "at_default", lambda *a: False)
    written = _text(study)
    assert written != kept
    assert json.loads(written)["study"]["key_version"] == 1


def test_control_a_loader_that_requires_every_field_refuses_the_document(monkeypatch):
    """Control: the fill is what lets an older document load."""
    text = _text(toy.survey_1())
    monkeypatch.setattr(load_module, "is_optional_field", lambda f: False)
    with pytest.raises(DocumentError, match="is missing"):
        load_study(text, TOY_KINDS)


def test_an_optional_default_must_be_plain():
    """Fails if a default with no single meaning (a factory, an array) is accepted."""

    @dataclasses.dataclass
    class Bad:
        items: list = dataclasses.field(default_factory=list, metadata={"optional": 1})
        rows: tuple = dataclasses.field(default=([],), metadata={"optional": 1})

    for f in dataclasses.fields(Bad):
        with pytest.raises(RegistrationError, match="plain default"):
            optional_default(f, Bad)


def test_at_default_compares_type_and_content():
    """Fails if True counts as the int 1, 1 as the float 1.0, or -0.0 as 0.0."""

    def field(default):
        return dataclasses.fields(
            dataclasses.make_dataclass(
                "D", [("x", type(default), dataclasses.field(default=default))]
            )
        )[0]

    assert at_default(field(1), 1)
    assert not at_default(field(1), True)
    assert not at_default(field(True), 1)
    assert not at_default(field(1.0), 1)
    assert at_default(field(0.0), 0.0)
    assert not at_default(field(0.0), -0.0)
    assert at_default(field(float("nan")), float("nan"))
    assert at_default(field((1, "a")), (1, "a"))
    assert not at_default(field((1, "a")), [1, "a"])


def test_at_default_compares_tuple_elements_by_type_and_content():
    """Fails if a tuple differing only in an element's type or sign counts as default.

    Such a value would be omitted by the encoder and reload as the default, so data
    would change silently.
    """

    def field(default):
        return dataclasses.fields(
            dataclasses.make_dataclass(
                "D", [("x", tuple, dataclasses.field(default=default))]
            )
        )[0]

    assert at_default(field((1, "a")), (1, "a"))
    assert not at_default(field((1,)), (True,))
    assert not at_default(field((True,)), (1,))
    assert not at_default(field((1,)), (1.0,))
    assert not at_default(field((0.0,)), (-0.0,))
    assert at_default(field((float("nan"),)), (float("nan"),))
    assert at_default(field(((1, 0.0), "b")), ((1, 0.0), "b"))
    assert not at_default(field(((1, 0.0), "b")), ((1, -0.0), "b"))
    assert not at_default(field((1, 2)), (1,))


# ---- the program column of an effect table (S2) -------------------------------------


def _effect_node(n_programs):
    caps = Capacities(n_programs=n_programs)
    return encode(effects_none(caps), ("effects",), as_resolver(None))


def test_a_one_program_effect_table_is_written_without_a_program_field():
    """Fails if the default table writes `"program": null` its old documents lack."""
    assert "program" not in _effect_node(1)


def test_an_effect_table_document_from_before_programs_still_loads():
    """Fails if a document without `program` is refused or fills anything but None."""
    node = {k: v for k, v in _effect_node(1).items() if k != "program"}  # as before
    table = as_resolver(None)
    built = load_module.decode(node, ("effects",), table)
    assert built.program is None


def test_an_effect_table_written_with_a_null_program_is_refused():
    """Control: `"program": null` is the default written out, so it is two documents."""
    node = {**_effect_node(1), "program": None}
    with pytest.raises(DocumentError, match="at its default"):
        load_module.decode(node, ("effects",), as_resolver(None))


def test_a_populated_program_column_round_trips():
    """Fails if a program column is dropped, omitted or rebuilt with another dtype."""
    caps = Capacities(n_programs=2)
    eff = effects_none(caps)
    eff = dataclasses.replace(eff, program=eff.program.at[1].set(1))
    node = encode(eff, ("effects",), as_resolver(None))
    assert "program" in node
    built = load_module.decode(node, ("effects",), as_resolver(None))
    assert_same_pytree(eff, built)


def _refund_channel_study():
    study = toy.survey_1(caps=dataclasses.replace(Capacities(), refunds=True))
    n, r = study.channels.cost.shape
    channels = dataclasses.replace(
        study.channels,
        pricing=jnp.full((n, r), int(ChargePricing.SURCHARGE), I64),
        surcharge=jnp.full((n, r), 2.0),
        failure_kind=jnp.asarray([-1, -1, 2], I64),
        refund_deadline_ticks=jnp.asarray([0, 0, 50], I64),
    )
    return dataclasses.replace(study, channels=channels)


def test_a_channel_refund_declaration_is_written_and_round_trips():
    """Fails if a pricing, surcharge, failure kind or deadline is dropped."""
    study = _refund_channel_study()
    doc = _doc(study)
    assert set(CHANNEL_FIELDS) <= set(doc["study"]["channels"])
    text = _text(study)
    loaded = load_study(text, TOY_KINDS)
    assert_same_pytree(study, loaded)
    assert _text(loaded) == text


def test_a_document_from_before_the_refund_columns_keeps_its_hash():
    """Fails if a default channel table starts writing a column: every hash moves."""
    study = toy.survey_1()
    before = json.loads(_text(study))["content_hash"]
    loaded = load_study(json.dumps(_doc(study)), TOY_KINDS)
    assert json.loads(_text(loaded))["content_hash"] == before
    assert loaded.channels.pricing is None
    assert loaded.channels.failure_kind is None


@pytest.mark.parametrize("name", CHANNEL_FIELDS)
def test_a_refund_column_written_as_null_is_refused(name):
    """Fails if an explicit null loads: the omitted form is the only default."""
    doc = _doc(_refund_channel_study())
    doc["study"]["channels"][name] = None
    with pytest.raises(DocumentError):
        load_study(json.dumps(doc), TOY_KINDS)


# ---- the refund columns of a record (S3a) -------------------------------------------


_RECORD_KINDS = as_resolver({("test", "record", 1): Record})


def _record_node(refunds):
    caps = dataclasses.replace(Capacities(), refunds=refunds)
    return encode(record_none(caps), ("record",), _RECORD_KINDS)


def test_a_record_without_refunds_is_written_without_the_refund_fields():
    """Fails if the default record writes nulls its older documents lack."""
    node = _record_node(False)
    assert node  # a nonempty node, so the absence below is about the fields
    assert not set(RECORD_FIELDS) & set(node)


def test_a_record_document_from_before_refunds_still_loads():
    """Fails if a document without the refund columns is refused or filled non-None."""
    node = {k: v for k, v in _record_node(False).items() if k not in RECORD_FIELDS}
    built = load_module.decode(node, ("record",), _RECORD_KINDS)
    assert all(getattr(built, n) is None for n in RECORD_FIELDS)


def test_a_record_written_with_a_null_refund_column_is_refused():
    """Control: a null column is the default written out, so it is two documents."""
    node = {**_record_node(False), "nonrefundable": None}
    with pytest.raises(DocumentError, match="at its default"):
        load_module.decode(node, ("record",), _RECORD_KINDS)


def test_a_record_written_with_a_null_award_generation_column_is_refused():
    """Control: the per-payer generations are optional too, and null is not omitted."""
    node = {**_record_node(False), "award_gens": None}
    with pytest.raises(DocumentError, match="at its default"):
        load_module.decode(node, ("record",), _RECORD_KINDS)


def test_a_populated_record_refund_columns_round_trip():
    """Fails if a refund column is dropped, omitted or rebuilt with another dtype."""
    caps = dataclasses.replace(Capacities(), refunds=True, awards=True)
    rec = dataclasses.replace(
        record_none(caps),
        nonrefundable=jnp.full(caps.n_resources, 2.0),
        refund_index=jnp.asarray(3, I64),
        award_gens=jnp.arange(caps.n_resources, dtype=I64),
    )
    node = encode(rec, ("record",), _RECORD_KINDS)
    assert set(RECORD_FIELDS) <= set(node)
    built = load_module.decode(node, ("record",), _RECORD_KINDS)
    assert_same_pytree(rec, built)


# ---- the award columns of an effect table (S5) --------------------------------------


def _award_node(awards):
    caps = dataclasses.replace(Capacities(), awards=awards)
    return encode(effects_none(caps), ("effects",), as_resolver(None))


def test_an_effect_table_without_awards_is_written_without_the_award_columns():
    """Fails if the default table writes columns its older documents lack."""
    node = _award_node(False)
    assert node  # a nonempty node, so the absence below is about the columns
    assert not set(AWARD_COLUMNS) & set(node)


def test_an_effect_table_document_from_before_awards_still_loads():
    """Fails if a document without the award columns is refused or filled non-None."""
    node = {k: v for k, v in _award_node(False).items() if k not in AWARD_COLUMNS}
    built = load_module.decode(node, ("effects",), as_resolver(None))
    assert all(getattr(built, n) is None for n in AWARD_COLUMNS)


@pytest.mark.parametrize("name", AWARD_COLUMNS)
def test_an_effect_table_written_with_a_null_award_column_is_refused(name):
    """Control: a null column is the default written out, so it is two documents."""
    node = {**_award_node(False), name: None}
    with pytest.raises(DocumentError, match="at its default"):
        load_module.decode(node, ("effects",), as_resolver(None))


def test_a_populated_award_column_set_round_trips():
    """Fails if an award column is dropped, omitted or rebuilt with another dtype."""
    caps = dataclasses.replace(Capacities(), awards=True)
    eff = effects_none(caps)
    eff = dataclasses.replace(
        eff,
        amount=eff.amount.at[0, 0].set(4.0),
        window_until=eff.window_until.at[0].set(9),
        remainder=eff.remainder.at[0].set(2),
    )
    node = encode(eff, ("effects",), as_resolver(None))
    assert set(AWARD_COLUMNS) <= set(node)
    built = load_module.decode(node, ("effects",), as_resolver(None))
    assert_same_pytree(eff, built)
