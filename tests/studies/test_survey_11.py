"""Survey 11, targets of opportunity: an alert makes the policy cancel the exposure.

Asset 0, lane 0 runs channel 10 (100 ticks of occupancy, 10 of transition, release
delay 20, a priced surcharge). A scripted alert fires at tick 40 and is released to the
policy `DELAY` ticks later. A level-1 array policy reads the released alert row in its
snapshot and, at that decision, emits a live `Cancel` of lane 0. The script, keyed by
tick, then starts the same subject again, so the next acquisition shows whether the
cancelled index was reused. The merge-time guard runs the study on the host executor
once and holds every assertion to that run; the checks are plain functions so a source
mutant can be held to the same ones.
"""

import dataclasses
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core.enums import (
    F64,
    I64,
    AdmissionCode,
    Audience,
    ChargePricing,
    EffectKind,
    EventKind,
    RecordStatus,
)
from spaceodyssey.core.executors.base import GO_TOLERANCE, index_world
from spaceodyssey.core.ledger import charge_parts
from spaceodyssey.core.lift import traced
from spaceodyssey.core.records import log_columns
from spaceodyssey.studies.scripted import SCRIPTED_CHANNEL, ScriptedRecords
from tests.core.helpers import worlds
from tests.core.program_helpers import ScriptedPrograms, observe, wait
from tests.core.refund_helpers import refund_study
from tests.studies.survey_helpers import run_on, same_histories

pytestmark = pytest.mark.medium(
    reason="one host run compiles the executor once; every check reads that run"
)

N_STEPS = 60
N_EFFECTS = 3  # the observe, the wait, and the slot the policy's Cancel takes
LEVEL = 1  # the policy level of the study: it reads arrays and may Cancel
LANE, SUBJECT, CHANNEL = 0, 0, 10
OCCUPANCY, TRANSITION = 100, 10  # channel 10 and the transition of asset 0
ALERT_CHANNEL = SCRIPTED_CHANNEL
CHARGE, SURCHARGE = 10.0, 2.0

# Ticks of the script. The first exposure starts at FIRST + TRANSITION and would finish
# at START + OCCUPANCY; the alert fires at FIRE and the policy sees it at FIRE + DELAY.
FIRST, FIRE, DELAY, AGAIN = 0, 40, 15, 120
START = FIRST + TRANSITION
SEEN = FIRE + DELAY
PLANNED_DONE = START + OCCUPANCY
FINISH = 250
SCRIPT = (
    observe(FIRST, SUBJECT, CHANNEL, 0),
    # the decision that sees the alert waits for the next request
    wait(SEEN, AGAIN - SEEN),
    observe(AGAIN, SUBJECT, CHANNEL, 0),
    wait(AGAIN, FINISH - AGAIN),
)


def _as_rows(rows):
    """The script as an int table; the shared helper's table holds only eight rows."""
    return jnp.asarray(np.asarray(rows, np.int64), I64)


class ScriptedAlerts(ScriptedRecords):
    """Scripted records released `delay` ticks after they are written.

    `ScriptedRecords` releases a record in the tick it is written. An alert is declared
    with a release delay: written at its fire tick, released to the policy `delay` ticks
    later. The delay is a leaf, so studies that differ only in it share a program.
    """

    delay: int = eqx.field(default=0, metadata=traced(I64))

    def handle(self, carry, ev, study):
        """Write the scripted record, available `delay` ticks after the fire."""
        carry, rec = super().handle(carry, ev, study)
        return carry, dataclasses.replace(rec, available_tick=ev.tick + self.delay)


class AlertWatch(ScriptedPrograms):
    """The script, read by tick, plus a policy that cancels the lane on an alert.

    A level-1 array policy: it scans the released rows of its snapshot for a row on the
    alert channel and, once, puts a live `Cancel(lane)` in the last effect row.
    """

    script: object = eqx.field(converter=_as_rows)

    def decide(self, pstate, view, study, key):
        """Emit the scripted rows of this tick, then the Cancel once an alert shows."""
        by_tick = eqx.tree_at(lambda v: v.n_dec, view, view.tick)
        eff, _, _ = super().decide(pstate, by_tick, study, key)
        alert = ((view.log.channel == ALERT_CHANNEL) & view.log.valid).any()
        due = alert & (pstate[0] == 0.0)
        last = N_EFFECTS - 1

        def put(column, value):
            return column.at[last].set(jnp.where(due, value, column[last]))

        eff = dataclasses.replace(
            eff,
            kind=put(eff.kind, EffectKind.CANCEL),
            lane=put(eff.lane, LANE),
            group=put(eff.group, view.tick * N_EFFECTS + last),
        )
        return eff, pstate.at[0].set(jnp.where(due, 1.0, pstate[0])), jnp.zeros((), F64)


def survey_study():
    """The refund toy study with channel 10 on a surcharge price and the alert process.

    Only the alert, the first exposure and the second exposure are written, so the log
    holds three rows. The agenda is at level 1.
    """
    study = refund_study(
        (),
        charge=CHARGE,
        pricing=ChargePricing.SURCHARGE,
        surcharge=SURCHARGE,
        policy=AlertWatch(script=SCRIPT),
    )
    alerts = ScriptedAlerts(
        ticks=(FIRE,),
        values=(1.0,),
        ids=(0,),
        revisions=(0,),
        statuses=(int(RecordStatus.COMPLETED),),
        audience=int(Audience.POLICY),
        delay=DELAY,
    )
    study = eqx.tree_at(lambda s: s.process, study, alerts)
    return eqx.tree_at(lambda s: s.agenda.levels, study, jnp.asarray([LEVEL]))


def _run(study):
    """The study on the host executor, one world."""
    truths, roots = worlds(1, study.caps)
    run = run_on("host", study, truths, roots, N_STEPS)
    return SimpleNamespace(
        study=study,
        batched=run,
        carry=jax.tree_util.tree_map(lambda x: x[0], run.carry),
        trace=index_world(run.trace, 0),
    )


def _rows(world):
    """The log rows as columns by name."""
    log, n = world.carry.log, int(world.carry.n_log)
    cols = {
        k: np.asarray(v)[:n]
        for k, v in log_columns(log).items()
        if np.ndim(v) and v.shape[0] == world.study.caps.n_log
    }
    cols["payload"] = np.asarray(log.payload)[:n]
    cols["charged"] = np.asarray(log.charged)[:n]
    cols["nonrefundable"] = np.asarray(log.nonrefundable)[:n]
    return cols


def _pick(rows, keep):
    return {k: v[keep] for k, v in rows.items()}


def _split(world):
    """The alert row, the partial record and the second exposure's record."""
    rows = _rows(world)
    alert = _pick(rows, rows["channel"] == ALERT_CHANNEL)
    work = _pick(rows, rows["channel"] == CHANNEL)
    assert len(alert["channel"]) == 1
    assert len(work["channel"]) == 2
    first, second = (_pick(work, work["acquisition_index"] == i) for i in (0, 1))
    return alert, work, first, second


def _steps(world, kind):
    """The ticks and written-row counts of every handled event of this kind."""
    t = world.trace
    keep = np.asarray(t.handled) & (np.asarray(t.kind) == kind)
    return np.asarray(t.tick)[keep].tolist(), np.asarray(t.n_written)[keep].tolist()


def check_a(world):
    """(a) The DONE of the cancelled generation settles nothing, writes no record."""
    _, work, first, _ = _split(world)
    # the cancelled exposure is one PARTIAL record; no COMPLETED one has its index
    assert first["status"].tolist() == [RecordStatus.PARTIAL]
    cancelled = work["acquisition_index"] == 0
    assert not (cancelled & (work["status"] == RecordStatus.COMPLETED)).any()
    # the only DONE handled in the run is the second exposure's, at its own tick, and
    # it wrote its row; the cancelled exposure's planned DONE tick handled none
    ticks, written = _steps(world, EventKind.DONE)
    assert PLANNED_DONE not in ticks
    assert ticks == [AGAIN + TRANSITION + OCCUPANCY]
    assert written == [1]
    assert not bool(world.carry.double_settle)
    # the planned DONE tick is later than the cancel, so there was something to cancel
    assert first["end_tick"][0] < PLANNED_DONE


def check_b(world):
    """(b) The partial record's fraction equals elapsed over planned."""
    _, _, first, _ = _split(world)
    planned = OCCUPANCY
    elapsed = SEEN - START
    assert elapsed > 0
    fraction = float(first["payload"][0, 2])
    assert abs(fraction - elapsed / planned) <= GO_TOLERANCE
    assert first["start_tick"][0] == START
    assert first["end_tick"][0] - first["start_tick"][0] == elapsed


def check_c(world):
    """(c) The retained charge and the credit follow the pricing table."""
    study = world.study
    _, _, first, second = _split(world)
    charge, fixed, pricing = (np.asarray(x)[0] for x in charge_parts(study, 0))
    assert pricing == ChargePricing.SURCHARGE
    assert fixed == SURCHARGE > 0
    assert charge == CHARGE + SURCHARGE
    part = (SEEN - START) / OCCUPANCY
    kept = (charge - fixed) * part + fixed
    assert first["charged"][0, 0] == pytest.approx(kept, abs=GO_TOLERANCE)
    assert first["nonrefundable"][0, 0] == fixed  # the surcharge is retained
    credit = charge - kept
    assert credit > 0
    ledger = world.carry.ledger
    refunded = np.asarray(ledger.refunded)[0, 0]
    assert refunded == pytest.approx(credit, abs=GO_TOLERANCE)
    # the second exposure is completed and keeps its whole price, surcharge included
    assert second["status"].tolist() == [RecordStatus.COMPLETED]
    assert second["charged"][0, 0] == charge
    balance = np.asarray(ledger.balance)[0, 0]
    start = float(np.asarray(study.budget)[0, 0])
    assert np.asarray(ledger.debited)[0, 0] == pytest.approx(2 * charge)
    assert balance == pytest.approx(start - 2 * charge + credit, abs=GO_TOLERANCE)


def check_d(world):
    """(d) The cancelled index is not reused."""
    _, _, first, second = _split(world)
    assert first["acquisition_index"].tolist() == [0]
    assert second["acquisition_index"].tolist() == [1]
    assert np.asarray(world.carry.acquisition_counts)[SUBJECT] == 2


def check_e(world):
    """(e) The alert-to-cancel latency equals the declared release delay."""
    study = world.study
    alert, _, first, _ = _split(world)
    declared = int(study.process.delay)
    assert declared == DELAY > 0
    # the alert is released to the policy the declared delay after it is written
    assert alert["end_tick"][0] == FIRE
    assert alert["available_tick"][0] - alert["end_tick"][0] == declared
    assert alert["released_tick"][0] == alert["available_tick"][0]
    # the policy answers in the tick it reads the alert, so the cancel lands then
    latency = first["end_tick"][0] - alert["end_tick"][0]
    assert latency == declared
    t = world.trace
    cancels = np.asarray(t.handled) & (np.asarray(t.kind) == EventKind.CANCEL)
    assert np.asarray(t.tick)[cancels].tolist() == [FIRE + declared]


CHECKS = (check_a, check_b, check_c, check_d, check_e)


def evidence(world):
    """What each assertion looked at, counted. Every count must be nonzero."""
    t = world.trace
    alert, _, first, second = _split(world)
    effects = np.asarray(t.handled)[:, None] & (
        np.asarray(t.eff_kind) == EffectKind.CANCEL
    )
    admitted = effects & (np.asarray(t.admit_code) == AdmissionCode.OK)
    return {
        "alert rows": len(alert["channel"]),
        "cancel effects emitted": int(effects.sum()),
        "cancel effects admitted": int(admitted.sum()),
        "cancel events handled": len(_steps(world, EventKind.CANCEL)[0]),
        "partial records": len(first["channel"]),
        "completed records": len(second["channel"]),
        "done events handled": len(_steps(world, EventKind.DONE)[0]),
        "process events handled": len(_steps(world, EventKind.PROCESS)[0]),
        "retained surcharge": int(first["nonrefundable"][0, 0] > 0),
        "credit rows": int(np.asarray(world.carry.ledger.refunded)[0, 0] > 0),
    }


@pytest.fixture(scope="module")
def world():
    return _run(survey_study())


def test_the_study_reads_and_cancels_on_the_alert(world):
    """Fails if the alert does not reach the policy, the Cancel is not admitted, or any
    counted event, record or credit the other checks rest on is absent."""
    assert world.study.agenda.levels.tolist() == [LEVEL]
    counts = evidence(world)
    assert all(v > 0 for v in counts.values()), counts
    assert counts["cancel effects emitted"] == counts["cancel effects admitted"] == 1
    assert int(world.carry.n_log) == 3


def test_the_cancelled_generation_settles_nothing(world):
    """Fails if the DONE of the cancelled generation reaches the log or the ledger."""
    check_a(world)


def test_the_partial_record_fraction_is_elapsed_over_planned(world):
    """Fails if the partial fraction is not elapsed over planned, in the go band."""
    check_b(world)


def test_the_retained_charge_and_credit_follow_the_pricing_table(world):
    """Fails if the surcharge is credited back or the credit is not charge less kept."""
    check_c(world)


def test_the_cancelled_index_is_not_reused(world):
    """Fails if the acquisition after the cancel takes the cancelled index again."""
    check_d(world)


def test_the_alert_to_cancel_latency_is_the_declared_release_delay(world):
    """Fails if the cancel lands at any tick but the alert's declared release."""
    check_e(world)


@pytest.mark.medium(reason="the compiled executor compiles every program again")
def test_both_executors_make_the_same_history_for_the_alert_study(world):
    """Parity for (a) to (e): ints exact, floats in the go band, on both executors."""
    truths, roots = worlds(1, world.study.caps)
    compiled = run_on("compiled", world.study, truths, roots, N_STEPS)
    assert int(np.asarray(compiled.carry.n_log)[0]) == 3
    assert same_histories(compiled, world.batched, 1)
