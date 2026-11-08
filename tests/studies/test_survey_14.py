"""Survey 14, a prescribed phased mission: no switch is survey 1."""

import jax
import numpy as np
import pytest

from spaceodyssey.core.enums import I64
from spaceodyssey.studies import toy
from spaceodyssey.studies.survey_14 import (
    SECOND_PHASE_PLAN,
    survey_14,
)
from tests.codec.test_document import TOY_KINDS
from tests.core.helpers import worlds
from tests.studies.survey_helpers import redump, round_trip, run_on, same_histories

# Cold setup plus call exceeds 30 s for every case (42-54 s in isolated probes).
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


@pytest.mark.slow(reason="cold runs of the two-phase study on each executor")
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


@pytest.mark.slow(reason="negative control that shares the run store")
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


@pytest.mark.slow(reason="negative control that shares the run store")
def test_an_early_switch_also_changes_the_history(batch, baseline):
    """Negative control at the other extreme: a switch at tick 1 hands every decision
    after the first to the second plan."""
    truths, roots = batch
    run = run_on("host", survey_14(1), truths, roots, N_STEPS)
    assert not same_histories(run, baseline, N_WORLDS)


@pytest.mark.slow(reason="run document round trip of the shared runs")
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
