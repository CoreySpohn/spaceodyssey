"""Survey 14, static checks: the no-switch declaration and its document form."""

import dataclasses
import json

import jax.numpy as jnp

from spaceodyssey.codec import dump_study, load_run
from spaceodyssey.core.enums import I64
from spaceodyssey.core.preflight import preflight
from spaceodyssey.studies import toy
from tests.codec.test_document import TOY_KINDS
from tests.studies.scenarios.survey_14 import no_switch, survey_14
from tests.studies.survey_helpers import round_trip

SWITCH = 300


def test_the_no_switch_boundary_is_the_canonical_integer():
    """Fails if the study holds a literal other than `caps.max_tick`, so that a
    document would not write the `no_switch` tag, or if a finite switch is rewritten.
    """
    study = survey_14()
    top = study.caps.max_tick
    assert no_switch(study.caps) == top
    assert study.agenda.phase_until.dtype == jnp.int64
    assert study.agenda.phase_until.tolist() == [top, top]
    assert len(study.agenda.policies) == 2
    assert survey_14(SWITCH).agenda.phase_until.tolist() == [SWITCH, top]
    assert not preflight(study, executor="compiled")


def test_a_dump_and_load_keep_the_declaration_unit():
    """Fails if the no-switch value is written as a literal tick, or a finite switch as
    the tag, or the tag is lowered to anything but `caps.max_tick`."""
    never_doc = dump_study(survey_14(), TOY_KINDS)
    data = never_doc.study["agenda"]["phase_until"]["data"]
    assert data == [{"no_switch": True}, {"no_switch": True}]
    finite = dump_study(survey_14(SWITCH), TOY_KINDS)
    assert finite.study["agenda"]["phase_until"]["data"] == [
        SWITCH,
        {"no_switch": True},
    ]
    assert finite.content_hash != never_doc.content_hash
    again = json.loads(never_doc.to_json())
    assert "no_switch" in json.dumps(again["study"]["agenda"])


def test_a_document_that_names_another_boundary_is_another_run():
    """Negative control for the round trip: a document with one boundary moved loads
    to a different declaration and a different hash."""
    study = survey_14()
    doc, _ = round_trip(study, toy.toy_registry(), "host", TOY_KINDS)
    moved = dataclasses.replace(
        study,
        agenda=dataclasses.replace(
            study.agenda, phase_until=jnp.asarray((SWITCH, study.caps.max_tick), I64)
        ),
    )
    other, run = round_trip(moved, toy.toy_registry(), "host", TOY_KINDS)
    assert other.content_hash != doc.content_hash
    assert run.study.agenda.phase_until.tolist() == [SWITCH, study.caps.max_tick]
    assert (
        load_run(
            doc.to_json(), run.bindings, TOY_KINDS
        ).study.agenda.phase_until.tolist()
        == [study.caps.max_tick] * 2
    )
