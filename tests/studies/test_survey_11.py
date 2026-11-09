"""Survey 11, targets of opportunity: an alert makes the policy cancel the exposure.

The study (`spaceodyssey.studies.survey_11`) runs channel 10 on lane 0, writes an
alert at tick 40 that the policy reads 15 ticks later, and has the policy cancel the
live work in that tick, then start the subject again. The merge-time guard runs it
once on the host executor and holds every check to that run; the checks are plain
functions so a source mutant can be held to the same ones. A late delivery of the
cancelled generation's DONE, a run on a frozen asset and a compiled-executor parity
run are one test each.
"""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from spaceodyssey.core import HostExecutor
from spaceodyssey.core.enums import (
    AdmissionCode,
    ChargePricing,
    EffectKind,
    EventKind,
    RecordStatus,
)
from spaceodyssey.core.executors.base import GO_TOLERANCE, index_world
from spaceodyssey.core.records import log_columns
from spaceodyssey.studies.survey_11 import (
    ALERT_CHANNEL,
    ALERT_DELAY,
    ALERT_FIRE,
    LANE,
    LEVEL,
    PRICING,
    STARTS,
    SUBJECT,
    survey_11,
)
from tests.core.helpers import handle_event, one_world, pending_done_row, worlds
from tests.studies.survey_helpers import run_on, same_histories

pytestmark = pytest.mark.medium(
    reason="one host run compiles the executor once; every check reads that run"
)

N_STEPS = 60
CHANNEL = 10
OCCUPANCY, TRANSITION, LATENCY = 100, 10, 20  # channel 10 and asset 0's transition
AGAIN = STARTS[1]
START = STARTS[0] + TRANSITION  # the first exposure starts after the transition
SEEN = ALERT_FIRE + ALERT_DELAY  # the tick the policy reads the alert and cancels
PLANNED_DONE = START + OCCUPANCY
SECOND_DONE = AGAIN + TRANSITION + OCCUPANCY

# The price of channel 10 per resource column (asset, program, shared) against a cost of
# 10 in each. The cancel comes at 45 of 100 ticks, so the elapsed fraction is 0.45.
#   column 0, SURCHARGE: charge 20 + surcharge 2 = 22, of which 2 is fixed.
#     kept = (22 - 2) * 0.45 + 2 = 11, credit = 22 - 11 = 11
#   column 1, OCCUPANCY: charge 12. kept = 12 * 0.45 = 5.4, credit = 12 - 5.4 = 6.6
#   column 2, FLAT_FRACTION: charge 8. A cancel keeps it all: kept = 8, credit = 0
# Two exposures are debited (the cancelled one and the completed one), so the debit is
# twice the charge in each column; only the cancelled one is credited.
#   column 0 on balance row 0 (asset 0): 100 - 44 + 11 = 67
#   column 1 on balance row 2 (the program row): 100 - 24 + 6.6 = 82.6
#   column 2 on balance row 3 (the shared row): 100 - 16 + 0 = 84
FRACTION = 0.45
KEPT = np.array([11.0, 5.4, 8.0])
FIXED = np.array([2.0, 0.0, 0.0])
FULL = np.array([22.0, 12.0, 8.0])
DEBITED = np.array([[44, 0, 0], [0, 0, 0], [0, 24, 0], [0, 0, 16]], float)
REFUNDED = np.array([[11, 0, 0], [0, 0, 0], [0, 6.6, 0], [0, 0, 0]], float)
BALANCE = np.array([[67, 100, 100], [100, 100, 100], [100, 82.6, 100], [100, 100, 84]])


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
    """The alert row, the work rows, the partial record and the second exposure's."""
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


def _cancel_codes(world):
    """The admission code of every Cancel the policy emitted."""
    t = world.trace
    sent = np.asarray(t.handled)[:, None] & (
        np.asarray(t.eff_kind) == EffectKind.CANCEL
    )
    return np.asarray(t.admit_code)[sent].tolist()


def check_a(world):
    """(a) The DONE of the cancelled generation settles nothing and writes no record."""
    _, work, first, _ = _split(world)
    # the cancelled exposure is one PARTIAL record; no COMPLETED one has its index
    assert first["status"].tolist() == [RecordStatus.PARTIAL]
    cancelled = work["acquisition_index"] == 0
    assert not (cancelled & (work["status"] == RecordStatus.COMPLETED)).any()
    # the only DONE handled in the run is the second exposure's, at its own tick, and
    # it wrote its row; the cancel invalidated the planned DONE row, so none came at 110
    ticks, written = _steps(world, EventKind.DONE)
    assert PLANNED_DONE not in ticks
    assert ticks == [SECOND_DONE]
    assert written == [1]
    assert not bool(world.carry.double_settle)
    # the planned DONE tick is later than the cancel, so there was something to cancel
    assert first["end_tick"][0] < PLANNED_DONE


def check_b(world):
    """(b) The partial record's fraction equals elapsed over planned."""
    _, _, first, _ = _split(world)
    elapsed = SEEN - START
    assert elapsed > 0
    fraction = float(first["payload"][0, 2])
    assert abs(fraction - elapsed / OCCUPANCY) <= GO_TOLERANCE
    assert abs(fraction - FRACTION) <= GO_TOLERANCE
    assert first["start_tick"][0] == START
    assert first["end_tick"][0] - first["start_tick"][0] == elapsed


def check_c(world):
    """(c) The retained charge and the credit follow the pricing table, per column."""
    _, _, first, second = _split(world)
    assert world.study.channels.pricing[0].tolist() == [int(p) for p in PRICING]
    assert PRICING[2] == ChargePricing.FLAT_FRACTION  # a cancel returns none of it
    # the records hold the retained charge and the fixed part in every column
    np.testing.assert_allclose(first["charged"][0], KEPT, atol=1e-9)
    np.testing.assert_allclose(first["nonrefundable"][0], FIXED, atol=1e-9)
    np.testing.assert_allclose(second["charged"][0], FULL, atol=1e-9)
    np.testing.assert_allclose(second["nonrefundable"][0], FIXED, atol=1e-9)
    assert second["status"].tolist() == [RecordStatus.COMPLETED]
    # the ledger debits and credits each payer row of each column, and no other cell
    ledger = world.carry.ledger
    np.testing.assert_allclose(np.asarray(ledger.debited), DEBITED, atol=1e-9)
    np.testing.assert_allclose(np.asarray(ledger.refunded), REFUNDED, atol=1e-9)
    np.testing.assert_allclose(np.asarray(ledger.balance), BALANCE, atol=1e-9)
    refunded = np.asarray(ledger.refunded)
    assert refunded[0, 0] > 0 and refunded[2, 1] > 0  # the asset and program credits
    assert refunded[3, 2] == 0 and (np.asarray(ledger.debited)[3, 2] > 0)  # flat price


def check_d(world):
    """(d) The cancelled index is not reused."""
    _, _, first, second = _split(world)
    assert first["acquisition_index"].tolist() == [0]
    assert second["acquisition_index"].tolist() == [1]
    assert np.asarray(world.carry.acquisition_counts)[SUBJECT] == 2


def check_e(world):
    """(e) The alert-to-cancel latency equals the declared release delay."""
    study = world.study
    alert, _, first, second = _split(world)
    declared = int(study.process.delay)
    assert declared == ALERT_DELAY > 0
    # the alert is released to the policy the declared delay after it is written
    assert alert["end_tick"][0] == ALERT_FIRE
    assert alert["available_tick"][0] - alert["end_tick"][0] == declared
    assert alert["released_tick"][0] == alert["available_tick"][0]
    # the policy answers in the tick it reads the alert, so the cancel lands then
    latency = first["end_tick"][0] - alert["end_tick"][0]
    assert latency == declared
    t = world.trace
    cancels = np.asarray(t.handled) & (np.asarray(t.kind) == EventKind.CANCEL)
    assert np.asarray(t.tick)[cancels].tolist() == [SEEN]
    # a work record is released after the channel's declared latency, cancelled or not
    assert int(study.channels.latency_ticks[0]) == LATENCY
    assert first["available_tick"][0] == SEEN + LATENCY
    assert first["released_tick"][0] == SEEN + LATENCY
    assert second["available_tick"][0] == SECOND_DONE + LATENCY


CHECKS = (check_a, check_b, check_c, check_d, check_e)


def evidence(world):
    """What each assertion looked at, counted. Every count must be nonzero."""
    t = world.trace
    alert, _, first, second = _split(world)
    effects = np.asarray(t.handled)[:, None] & (
        np.asarray(t.eff_kind) == EffectKind.CANCEL
    )
    refunded = np.asarray(world.carry.ledger.refunded)
    return {
        "alert rows": len(alert["channel"]),
        "cancel effects emitted": int(effects.sum()),
        "cancel effects admitted": _cancel_codes(world).count(AdmissionCode.OK),
        "cancel events handled": len(_steps(world, EventKind.CANCEL)[0]),
        "partial records": len(first["channel"]),
        "completed records": len(second["channel"]),
        "done events handled": len(_steps(world, EventKind.DONE)[0]),
        "process events handled": len(_steps(world, EventKind.PROCESS)[0]),
        "retained fixed parts": int((first["nonrefundable"][0] > 0).sum()),
        "credited cells": int((refunded > 0).sum()),
    }


@pytest.fixture(scope="module")
def world():
    return _run(survey_11())


def test_the_study_reads_and_cancels_on_the_alert(world):
    """Fails if the alert does not reach the policy, the Cancel is not admitted, or any
    counted event, record or credit the other checks rest on is absent."""
    assert world.study.agenda.levels.tolist() == [LEVEL]
    counts = evidence(world)
    assert all(v > 0 for v in counts.values()), counts
    assert counts["cancel effects emitted"] == counts["cancel effects admitted"] == 1
    assert counts["credited cells"] == 2
    assert int(world.carry.n_log) == 3


def test_the_cancelled_generation_settles_nothing(world):
    """Fails if the DONE of the cancelled generation reaches the log or the ledger."""
    check_a(world)


def late_done(world):
    """The post-cancel carry before and after the cancelled DONE is handled.

    The DONE row is read from the queue as it stood before the Cancel ran, then handed
    to the real handler path, with the generation it was queued with, in the tick of the
    cancel: the helper enqueues at the carry's tick, and the partial record's queued
    release is earlier than the planned DONE tick, so a later tick would handle that.
    """
    study = world.study
    truth, root = one_world(caps=study.caps)
    t = world.trace
    handled = np.asarray(t.handled) & (np.asarray(t.kind) == EventKind.CANCEL)
    at = int(np.flatnonzero(handled)[0])
    run = HostExecutor()
    before = run.run(study, truth, root, at).carry  # the Cancel is queued, not handled
    after = run.run(study, truth, root, at + 1).carry  # the Cancel has run
    row = pending_done_row(before, LANE)
    assert bool(row.valid)  # the queue held the exposure's DONE until the cancel
    assert not bool(pending_done_row(after, LANE).valid)
    old_gen = int(row.gen)
    assert int(after.ledger.lane_gen[LANE]) != old_gen  # the cancel retired it
    late = handle_event(
        after,
        study,
        truth,
        root,
        EventKind.DONE,
        lane=LANE,
        gen=old_gen,
        acquisition_id=int(row.acquisition_id),
        channel_index=int(row.channel_index),
        channel=int(row.channel),
        subject=int(row.subject),
        acquisition_index=int(row.acquisition_index),
        scope=int(row.scope),
    )
    return after, late


def test_a_late_done_of_the_cancelled_generation_settles_nothing(world):
    """Delivers the cancelled generation's DONE to the post-cancel carry by the real
    handler. Fails if the stale-generation guard is dropped, or the cancel does not
    retire the lane's generation: either lets the late DONE count as a double settle.
    """
    after, late = late_done(world)
    assert int(late.n_log) == int(after.n_log) == 2  # the alert and the partial record
    assert not bool(late.double_settle)
    for name in ("log", "ledger", "acquisition_counts"):
        pairs = zip(
            jax.tree_util.tree_leaves(getattr(after, name)),
            jax.tree_util.tree_leaves(getattr(late, name)),
            strict=True,
        )
        for a, b in pairs:
            assert np.array_equal(np.asarray(a), np.asarray(b)), name


def test_the_partial_record_fraction_is_elapsed_over_planned(world):
    """Fails if the partial fraction is not elapsed over planned, in the go band."""
    check_b(world)


def test_the_retained_charge_and_credit_follow_the_pricing_table(world):
    """Fails if the charge, the fixed part or the credit of any column or payer row
    differs from the hand-worked table, or the credit follows the cost, not the charge.
    """
    check_c(world)


def test_the_cancelled_index_is_not_reused(world):
    """Fails if the acquisition after the cancel takes the cancelled index again."""
    check_d(world)


def test_the_alert_to_cancel_latency_is_the_declared_release_delay(world):
    """Fails if the cancel lands at any tick but the alert's declared release, or a
    work record is released other than after the channel's declared latency."""
    check_e(world)


@pytest.mark.medium(reason="commitments change the program, so one more host compile")
def test_a_live_cancel_inside_the_frozen_horizon_is_admitted():
    """D8: cancelling an activity that has started is not frozen. Fails if the asset's
    horizon refuses the live Cancel, or the Cancel spends a disruptive activation."""
    frozen = _run(survey_11(frozen=True))
    assert _cancel_codes(frozen) == [AdmissionCode.OK]  # one Cancel, admitted
    _, _, first, second = _split(frozen)
    assert first["status"].tolist() == [RecordStatus.PARTIAL]
    assert first["end_tick"].tolist() == [SEEN]
    assert second["status"].tolist() == [RecordStatus.COMPLETED]
    # the two starts broke the freeze, one activation each; the Cancel took none
    assert np.asarray(frozen.carry.ledger.disruptive_used).tolist() == [2, 0]
    assert abs(float(first["payload"][0, 2]) - FRACTION) <= GO_TOLERANCE


@pytest.mark.medium(reason="the compiled executor compiles every program again")
def test_both_executors_make_the_same_history_for_the_alert_study(world):
    """Parity for (a) to (e): ints exact, floats in the go band, on both executors."""
    truths, roots = worlds(1, world.study.caps)
    compiled = run_on("compiled", world.study, truths, roots, N_STEPS)
    assert int(np.asarray(compiled.carry.n_log)[0]) == 3
    assert same_histories(compiled, world.batched, 1)
