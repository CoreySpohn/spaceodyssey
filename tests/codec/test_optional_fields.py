"""The optional-field rule of the codec: a field added after the first document format.

An optional field (`OPTIONAL` metadata) is omitted by the encoder at its default and
filled by the loader when missing, so every document written before the field existed
keeps its hash. A field written at its default is refused, so one study has one
document. The corpus hashes themselves are pinned in `test_run_document_runs.py`.
"""

import dataclasses
import json

import pytest

from spaceodyssey.codec import DocumentError, dump_study, load_study
from spaceodyssey.codec import dump as dump_module
from spaceodyssey.codec import load as load_module
from spaceodyssey.codec.kinds import (
    RegistrationError,
    at_default,
    is_optional_field,
    optional_default,
)
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.study import Study
from spaceodyssey.studies import toy
from spaceodyssey.studies.conformance import CORPUS
from tests.codec.test_document import TOY_KINDS, assert_same_pytree

CAP_FIELDS = ("awards", "refunds", "n_commitments", "snapshot_balances")
STUDY_FIELDS = ("key_version",)


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
        for cls in (Capacities, Study)
        for f in dataclasses.fields(cls)
        if is_optional_field(f)
    }
    assert marked == {("Capacities", n) for n in CAP_FIELDS} | {
        ("Study", n) for n in STUDY_FIELDS
    }


def test_the_encoder_omits_every_optional_field_at_its_default():
    """Fails if a default study writes a field its older documents did not hold."""
    doc = _doc(toy.survey_1())
    assert not set(STUDY_FIELDS) & set(doc["study"])
    assert not set(CAP_FIELDS) & set(doc["study"]["caps"])
    assert "n_programs" in doc["study"]["caps"]  # a field with a default, not optional


def test_no_corpus_document_holds_an_optional_field():
    """Fails if any corpus study writes an optional field (its hash would move)."""
    for case in CORPUS:
        doc = _doc(case.make_study())
        assert not set(STUDY_FIELDS) & set(doc["study"]), case.name
        assert not set(CAP_FIELDS) & set(doc["study"]["caps"]), case.name


def test_a_nondefault_optional_field_is_written_and_round_trips():
    """Fails if a value off its default is dropped, changed or not rebuilt exactly."""
    caps = dataclasses.replace(
        Capacities(), awards=True, refunds=True, n_commitments=3, snapshot_balances=True
    )
    study = toy.survey_1(caps=caps, key_version=2)
    doc = _doc(study)
    assert doc["study"]["key_version"] == 2
    assert [doc["study"]["caps"][n] for n in CAP_FIELDS] == [True, True, 3, True]
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
    assert [getattr(loaded.caps, n) for n in CAP_FIELDS] == [False, False, 0, False]
    assert_same_pytree(study, loaded)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("key_version",), 1),
        (("caps", "awards"), False),
        (("caps", "n_commitments"), 0),
    ],
    ids=["study", "caps-bool", "caps-int"],
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
