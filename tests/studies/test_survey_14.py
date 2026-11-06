"""Survey 14, a prescribed phased mission: no switch is survey 1."""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.codec import dump_study, load_run
from spaceodyssey.core.enums import I64
from spaceodyssey.core.preflight import preflight
from spaceodyssey.studies import toy
from spaceodyssey.studies.survey_14 import (
    SECOND_PHASE_PLAN,
    no_switch,
    survey_14,
)
from tests.codec.test_document import TOY_KINDS
from tests.core.helpers import worlds
from tests.studies.survey_helpers import redump, round_trip, run_on, same_histories

N_STEPS = 40
N_WORLDS = 2
SWITCH = 300


@pytest.fixture(scope="module")
def batch():
    return worlds(N_WORLDS)


@pytest.fixture(scope="module")
def baseline(batch):
    """Survey 1 on the host: the history the no-switch study must equal on both
    executors (the executors agree within the parity bands, so one reference serves)."""
    truths, roots = batch
    return run_on("host", toy.survey_1(), truths, roots, N_STEPS)


@pytest.fixture(scope="module")
def never(batch):
    truths, roots = batch
    study = survey_14()
    return study, {
        n: run_on(n, study, truths, roots, N_STEPS) for n in ("host", "compiled")
    }


@pytest.fixture(scope="module")
def switched(batch):
    truths, roots = batch
    study = survey_14(SWITCH)
    return study, {"host": run_on("host", study, truths, roots, N_STEPS)}


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


@pytest.mark.medium
@pytest.mark.parametrize("executor", ["host", "compiled"])
def test_a_threshold_that_never_switches_reduces_to_survey_1(executor, never, baseline):
    """Fails if a second phase that cannot start changes any step, record or belief
    value of the run: the full canonical history must equal survey 1's in each world.
    """
    _, runs = never
    assert bool(np.all(np.asarray(runs[executor].complete)))
    assert same_histories(runs[executor], baseline, N_WORLDS)
    # the second phase never decided: every record is the first phase's
    log = runs[executor].carry.log
    valid = np.asarray(log.valid)
    assert np.all(np.asarray(log.phase)[valid] == 0)


@pytest.mark.medium
def test_a_finite_switch_is_not_survey_1_and_hands_over_at_its_tick(switched, baseline):
    """Negative control for the reduction: a switch at tick 300 changes the history,
    so equality with survey 1 is not vacuous. Decisions before the tick follow the
    first plan, and decisions at or after it follow the second.
    """
    _, runs = switched
    for run in runs.values():
        assert not same_histories(run, baseline, N_WORLDS)
        log = jax.tree_util.tree_map(lambda a: np.asarray(a[0]), run.carry.log)
        keep = log.valid
        order = np.argsort(log.start_tick[keep])
        starts = log.start_tick[keep][order]
        channels = log.channel[keep][order]
        phases = log.phase[keep][order]
        # an acquisition belongs to the phase of the decision that started it
        decision_ticks = starts - 10  # the 10-tick transition precedes the start
        assert np.array_equal(phases, (decision_ticks >= SWITCH).astype(int))
        assert channels[phases == 0].tolist() == [10] * int((phases == 0).sum())
        assert channels[phases == 1].tolist() == [11] * int((phases == 1).sum())
        assert int((phases == 1).sum()) == len(SECOND_PHASE_PLAN)


@pytest.mark.medium
def test_an_early_switch_also_changes_the_history(batch, baseline):
    """Negative control at the other extreme: a switch at tick 1 hands every decision
    after the first to the second plan."""
    truths, roots = batch
    run = run_on("host", survey_14(1), truths, roots, N_STEPS)
    assert not same_histories(run, baseline, N_WORLDS)


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


@pytest.mark.medium
@pytest.mark.parametrize("executor", ["host", "compiled"])
def test_the_run_document_round_trip_keeps_history_and_hash(
    executor, never, switched, batch
):
    """Fails if a dump and load changes the declaration, the content hash or the
    history: the reloaded run must reproduce the original run, and a second dump must
    give the same document, for the no-switch study and a finite switch.
    """
    truths, roots = batch
    cases = (never, switched) if executor == "host" else (never,)
    for study, runs in cases:
        doc, run = round_trip(study, toy.toy_registry(), executor, TOY_KINDS)
        assert run.executor == executor
        assert run.study.agenda.phase_until.dtype == I64
        np.testing.assert_array_equal(
            run.study.agenda.phase_until, study.agenda.phase_until
        )
        assert redump(run, executor, TOY_KINDS).content_hash == doc.content_hash
        assert redump(run, executor, TOY_KINDS).to_json() == doc.to_json()
        again = run.execute(truths, roots, N_STEPS)
        assert same_histories(again, runs[executor], N_WORLDS)


def test_a_document_that_names_another_boundary_is_another_run(batch):
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
