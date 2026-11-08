"""Commitments and the frozen horizon: a start that is not yet due reserves its lane.

An Observe with a start tick after the decision's is admitted and debited now and
starts later through a COMMIT_START event. A start within an asset's `freeze_ticks`
is an insertion, refused unless the row is disruptive and the asset holds an activation;
a commitment that is already planned executes untouched, and a cancel of a pending one
inside the horizon is refused on the same terms. Each check has a control that differs
in the one thing it guards. The runs are host runs that share one compile.
"""

import dataclasses
import functools
from types import SimpleNamespace

import numpy as np
import pytest

from spaceodyssey.core import CompiledExecutor, HostExecutor, histories_equal
from spaceodyssey.core.conformance.invariants import balance_conserved
from spaceodyssey.core.enums import (
    AdmissionCode,
    ChargePricing,
    CommitState,
    Disposition,
    EffectKind,
    RecordStatus,
    Scope,
)
from tests.core.commitment_helpers import (
    DISRUPTIVE_COL,
    commitment_study,
    plan,
    priced,
)
from tests.core.helpers import one_world, worlds
from tests.core.program_helpers import cancel, end, observe, stop, wait

pytestmark = pytest.mark.medium  # A cold host run compiles the executor.

OK = int(AdmissionCode.OK)
FROZEN = int(AdmissionCode.FROZEN_HORIZON)
BUSY = int(AdmissionCode.LANE_BUSY)
NO_FUNDS = int(AdmissionCode.NO_FUNDS)
NO_SLOT = int(AdmissionCode.NO_SLOT)
UNSUPPORTED = int(AdmissionCode.UNSUPPORTED)
N_STEPS = 200

# lane 0 (channel 10, 100 ticks plus a transition of 10) is on the asset with the
# horizon; lane 2 (channel 12, 50 plus 5) is on the one without
COMMIT_100 = ((observe(0, 0, 10, 0), wait(0, 250), end(1)), (plan(0, 0, 100),))
SCENARIOS = {
    # the plan is made outside the horizon and executes inside it, unchanged
    "plan": COMMIT_100,
    "immediate": ((observe(0, 0, 10, 0), wait(0, 20), end(1)), ()),
    "immediate_disruptive": (
        (observe(0, 0, 10, 0), wait(0, 20), end(1)),
        (plan(0, 0, 0, 1),),
    ),
    # a cancel at tick 80 is inside the horizon of the start at 100, one at 10 is not
    "cancel_inside": (
        (observe(0, 0, 10, 0), wait(0, 80), cancel(1, 0), wait(1, 150), end(2)),
        (plan(0, 0, 100), plan(1, 0, 100)),
    ),
    "cancel_inside_disruptive": (
        (observe(0, 0, 10, 0), wait(0, 80), cancel(1, 0), wait(1, 150), end(2)),
        (plan(0, 0, 100), plan(1, 0, 100, 1)),
    ),
    "cancel_outside": (
        (observe(0, 0, 10, 0), wait(0, 10), cancel(1, 0), wait(1, 200), end(2)),
        (plan(0, 0, 100), plan(1, 0, 100)),
    ),
    "cancel_started": (
        (observe(0, 0, 10, 0), wait(0, 110), cancel(1, 0), wait(1, 200), end(2)),
        (plan(0, 0, 100),),
    ),
    # a request that ends before a reservation starts, and one that would overlap it
    "before_reservation": (
        (observe(0, 0, 12, 0), observe(0, 1, 12, 0), wait(0, 250)),
        (plan(0, 0, 100),),
    ),
    "overlaps_reservation": (
        (observe(0, 0, 12, 0), wait(0, 60), observe(1, 1, 12, 0), wait(1, 200), end(2)),
        (plan(0, 0, 100),),
    ),
}

SCENARIOS.update(
    {
        # a commitment that has started holds its lane: a request during the run is
        # refused for the live occupancy, the reservation being no longer pending
        "request_during_run": (
            (
                observe(0, 0, 10, 0),
                wait(0, 150),
                observe(1, 1, 10, 0),
                wait(1, 200),
                end(2),
            ),
            (plan(0, 0, 100),),
        ),
        # the horizon is 30: a start exactly 30 away is not inside it, 29 is
        "at_the_horizon": (
            (observe(0, 0, 10, 0), wait(0, 250), end(1)),
            (plan(0, 0, 30),),
        ),
        "inside_by_one": (
            (observe(0, 0, 10, 0), wait(0, 250), end(1)),
            (plan(0, 0, 29),),
        ),
        # two rows of one group, each on its own lane of the asset with the horizon,
        # each an insertion that spends an activation
        "group_activations": (
            (observe(0, 0, 10, 0), observe(0, 1, 11, 0), wait(0, 20), end(1)),
            (plan(0, 0, 0, 1, 1), plan(0, 1, 0, 1, 1)),
        ),
        # two Cancels in one decision name the one commitment, both disruptive
        "cancel_twice": (
            (
                observe(0, 0, 10, 0),
                wait(0, 80),
                cancel(1, 0),
                cancel(1, 0),
                wait(1, 150),
                end(2),
            ),
            (plan(0, 0, 100), plan(1, 0, 100, 1), plan(1, 1, 100, 1)),
        ),
        # two commitments on two lanes, then the same two in one group
        "two_commitments": (
            (observe(0, 0, 10, 0), observe(0, 1, 12, 0), wait(0, 250), end(1)),
            (plan(0, 0, 100), plan(0, 1, 100)),
        ),
        "group_of_two_commitments": (
            (observe(0, 0, 10, 0), observe(0, 1, 12, 0), wait(0, 250), end(1)),
            (plan(0, 0, 100, 0, 1), plan(0, 1, 100, 0, 1)),
        ),
    }
)

# a Stop at tick 10 of the lane (or program) holding a commitment that starts at 100
LANE = Scope.LANE
STOPS = {
    f"{what}_{how.name.lower()}": (
        (observe(0, 0, 10, 0), wait(0, 10), stop(1, LANE, how), wait(1, 200), end(2)),
        (plan(0, 0, 100),),
    )
    for what, how in (
        ("lane", Disposition.CLOSE),
        ("lane", Disposition.DRAIN),
        ("lane", Disposition.ABORT),
    )
}
STOPS["campaign_drain"] = (
    (observe(0, 0, 10, 0), wait(0, 10), end(1)),
    (plan(0, 0, 100),),
)
STOPS["campaign_abort"] = (
    (
        observe(0, 0, 10, 0),
        wait(0, 10),
        stop(1, Scope.CAMPAIGN, Disposition.ABORT),
    ),
    (plan(0, 0, 100),),
)
SCENARIOS.update(STOPS)
# program 1 holds the commitment on lane 0 and program 0 one on lane 2; the Stop is of
# program 1 alone
PROGRAM = (observe(0, 0, 10, 1), observe(0, 1, 12, 0), wait(0, 10))
# the abort writes a record, whose release wakes one more decision at tick 30
for _how, _tail in (
    (Disposition.CLOSE, (end(2),)),
    (Disposition.ABORT, (wait(2, 250), end(3))),
):
    STOPS[f"program_{_how.name.lower()}"] = (
        (*PROGRAM, stop(1, Scope.PROGRAM, _how, 1), wait(1, 250), *_tail),
        (plan(0, 0, 100), plan(0, 1, 100)),
    )
SCENARIOS.update(STOPS)
PROGRAM_SCENARIOS = {"program_close", "program_abort"}


def _study(name, **kw):
    script, plans = SCENARIOS[name]
    if name in PROGRAM_SCENARIOS:
        kw["n_programs"] = 2
    return commitment_study(script, plans, **kw)


@functools.cache
def _run(name, freeze=30, balance=None, k=4):
    study = _study(name, freeze=freeze, n_commitments=k)
    if balance is not None:
        # the cell the activations come from holds `balance`, which may be only what
        # the channels take
        budget = study.budget.at[0, DISRUPTIVE_COL].set(balance)
        study = dataclasses.replace(study, budget=budget)
    truth, root = one_world(caps=study.caps)
    result = HostExecutor().run(study, truth, root, N_STEPS)
    return SimpleNamespace(study=study, carry=result.carry, trace=result.trace)


def _codes(run):
    """The admission codes of each decision that asked for something."""
    t = run.trace
    asked = np.asarray(t.handled) & (np.asarray(t.eff_kind) != EffectKind.NONE).any(1)
    return np.asarray(t.admit_code)[asked]


def _log(run):
    log = run.carry.log
    valid = np.asarray(log.valid)
    return {
        "status": np.asarray(log.status)[valid],
        "start": np.asarray(log.start_tick)[valid],
        "end": np.asarray(log.end_tick)[valid],
        "fraction": np.asarray(log.payload)[valid][:, 2],
    }


def _ledger(run):
    return run.carry.ledger


def _sound(run):
    """Conservation holds and no balance went below zero."""
    return balance_conserved(run).passed and bool(
        (np.asarray(run.carry.ledger.balance) >= 0.0).all()
    )


def test_a_commitment_outside_the_horizon_is_admitted_and_executes_unchanged():
    run = _run("plan")
    assert _codes(run)[0][0] == OK
    # the plan is inside the horizon when it starts and nothing refuses it
    log = _log(run)
    assert log["status"].tolist() == [RecordStatus.COMPLETED]
    assert log["end"].tolist() == [210]
    led = _ledger(run)
    assert int(led.cm_state[0]) == CommitState.STARTED
    assert (int(led.cm_start[0]), int(led.cm_end[0])) == (100, 210)
    assert int(led.cm_index[0]) == 0 and int(led.cm_count[0, 0]) == 1
    assert int(led.disruptive_used[0]) == 0
    assert _sound(run)


def test_an_immediate_request_on_a_frozen_asset_is_refused_without_an_activation():
    run = _run("immediate")
    assert _codes(run)[0][0] == FROZEN
    assert not _log(run)["status"].size
    led = _ledger(run)
    assert float(led.balance[0, DISRUPTIVE_COL]) == 10.0
    assert int(led.disruptive_used[0]) == 0
    assert _sound(run)


def test_control_the_same_request_with_an_activation_is_admitted_and_debits_one():
    run = _run("immediate_disruptive")
    assert _codes(run)[0][0] == OK
    led = _ledger(run)
    assert int(led.disruptive_used[0]) == 1
    # the charge of the channel and the activation, from the same cell
    assert float(led.balance[0, DISRUPTIVE_COL]) == 8.0
    assert _log(run)["status"].tolist() == [RecordStatus.COMPLETED]
    assert _sound(run)


def test_an_activation_the_asset_cannot_cover_does_not_break_the_freeze():
    run = _run("immediate_disruptive", balance=1.0)
    assert _codes(run)[0][0] == FROZEN
    assert int(_ledger(run).disruptive_used[0]) == 0


def test_control_a_zero_horizon_leaves_admission_as_it_was():
    run = _run("immediate", freeze=0)
    assert _codes(run)[0][0] == OK
    assert _log(run)["status"].tolist() == [RecordStatus.COMPLETED]


def test_a_cancel_inside_the_horizon_is_refused_and_the_commitment_still_runs():
    run = _run("cancel_inside")
    assert _codes(run)[1][0] == FROZEN
    assert int(_ledger(run).cm_state[0]) == CommitState.STARTED
    assert _log(run)["status"].tolist() == [RecordStatus.COMPLETED]
    assert _sound(run)


def test_control_a_disruptive_cancel_inside_the_horizon_cancels_and_costs_one():
    run = _run("cancel_inside_disruptive")
    assert _codes(run)[1][0] == OK
    led = _ledger(run)
    assert int(led.cm_state[0]) == CommitState.CANCELLED
    assert int(led.disruptive_used[0]) == 1
    # the charge came back, the activation did not: 10 - 1 + 1 - 1
    assert float(led.balance[0, DISRUPTIVE_COL]) == 9.0
    assert _log(run)["status"].tolist() == [RecordStatus.PARTIAL]
    assert _sound(run)


def test_a_cancel_outside_the_horizon_credits_the_whole_charge():
    run = _run("cancel_outside")
    assert _codes(run)[1][0] == OK
    led = _ledger(run)
    assert int(led.cm_state[0]) == CommitState.CANCELLED
    log = _log(run)
    assert log["status"].tolist() == [RecordStatus.PARTIAL]
    assert log["fraction"].tolist() == [0.0]
    np.testing.assert_array_equal(np.asarray(led.balance), np.asarray(run.study.budget))
    # the lane was never occupied: nothing started and no completion ran
    assert int(led.lane_gen[0]) == 0 and int(led.lane_busy_until[0]) == 0
    assert _sound(run)


def test_cancelling_an_activity_that_has_started_is_not_frozen():
    run = _run("cancel_started")
    assert _codes(run)[1][0] == OK
    assert _log(run)["status"].tolist() == [RecordStatus.PARTIAL]
    assert int(_ledger(run).disruptive_used[0]) == 0
    assert _sound(run)


def test_work_that_ends_before_a_reservation_is_admitted():
    run = _run("before_reservation")
    assert _codes(run)[0][:2].tolist() == [OK, OK]
    log = _log(run)
    assert sorted(log["end"].tolist()) == [55, 155]
    assert set(log["status"].tolist()) == {RecordStatus.COMPLETED}
    assert _sound(run)


def test_work_that_overlaps_a_reservation_is_refused():
    run = _run("overlaps_reservation")
    assert _codes(run)[1][0] == BUSY
    assert _log(run)["end"].tolist() == [155]
    assert _sound(run)


# ---- a Stop of the lane or program holding a pending commitment ----------------------


def _refused_at_its_start(run):
    """The commitment never ran: cancelled at fraction 0 and the lane never occupied."""
    led = _ledger(run)
    log = _log(run)
    assert int(led.cm_state[0]) == CommitState.CANCELLED
    assert log["status"].tolist() == [RecordStatus.PARTIAL]
    assert log["fraction"].tolist() == [0.0]
    assert int(led.lane_gen[0]) == 0 and int(led.lane_busy_until[0]) == 0
    np.testing.assert_array_equal(np.asarray(led.balance), np.asarray(run.study.budget))
    assert _sound(run)
    return log


@pytest.mark.parametrize("how", ["close", "drain"])
def test_a_close_or_drain_refuses_the_start_of_a_pending_commitment(how):
    log = _refused_at_its_start(_run(f"lane_{how}"))
    # refused when its start came due, not when the lane closed
    assert log["end"].tolist() == [100]


def test_an_abort_cancels_a_pending_commitment_when_it_is_ordered():
    log = _refused_at_its_start(_run("lane_abort"))
    assert log["end"].tolist() == [10]


def test_a_program_close_refuses_only_that_programs_commitment():
    run = _run("program_close")
    led = _ledger(run)
    assert [int(led.cm_state[i]) for i in range(2)] == [
        CommitState.CANCELLED,
        CommitState.STARTED,
    ]
    log = _log(run)
    assert sorted(log["status"].tolist()) == [
        RecordStatus.COMPLETED,
        RecordStatus.PARTIAL,
    ]
    assert _sound(run)


def test_a_program_abort_cancels_only_that_programs_pending_commitment():
    run = _run("program_abort")
    led = _ledger(run)
    assert [int(led.cm_state[i]) for i in range(2)] == [
        CommitState.CANCELLED,
        CommitState.STARTED,
    ]
    log = _log(run)
    assert sorted(log["end"].tolist()) == [10, 155]
    assert _sound(run)


@pytest.mark.slow
@pytest.mark.parametrize(
    "name",
    [
        "lane_close",
        "lane_drain",
        "lane_abort",
        "program_close",
        "program_abort",
        "campaign_drain",
        "campaign_abort",
    ],
)
def test_the_executors_agree_on_a_stop_of_a_pending_commitment(name):
    study = _study(name)
    truths, roots = worlds(1, study.caps)
    compiled = CompiledExecutor().run_worlds(study, truths, roots, N_STEPS)
    host = HostExecutor().run_worlds(study, truths, roots, N_STEPS)
    rep = histories_equal(compiled.history(0), host.history(0))
    assert rep.structure_equal and rep.max_int_diff == 0 and rep.max_abs <= 1e-9


def test_a_campaign_drain_waits_for_a_pending_commitment_and_refuses_its_start():
    run = _run("campaign_drain")
    log = _refused_at_its_start(run)
    # the drain did not end the run at tick 10: it waited for the refused start
    assert log["end"].tolist() == [100]
    assert bool(run.carry.done) and int(run.carry.tick) >= 100


def test_a_campaign_abort_cancels_a_pending_commitment():
    log = _refused_at_its_start(_run("campaign_abort"))
    assert log["end"].tolist() == [10]


# ---- what a group, a cancel and the capacity may not overdraw ----------------------


def test_a_group_of_activations_that_the_cell_cannot_cover_is_refused_whole():
    run = _run("group_activations", balance=3.0)
    # each row costs 1 and an activation: 4 against 3, so neither is admitted
    assert _codes(run)[0][:2].tolist() == [NO_FUNDS, NO_FUNDS]
    led = _ledger(run)
    assert float(led.balance[0, DISRUPTIVE_COL]) == 3.0
    assert int(led.disruptive_used[0]) == 0
    assert not _log(run)["status"].size
    assert _sound(run)


def test_control_the_same_group_with_funds_for_both_is_admitted():
    run = _run("group_activations", balance=4.0)
    assert _codes(run)[0][:2].tolist() == [OK, OK]
    led = _ledger(run)
    assert float(led.balance[0, DISRUPTIVE_COL]) == 0.0
    assert int(led.disruptive_used[0]) == 2
    assert _sound(run)


def test_a_second_cancel_of_one_commitment_spends_no_activation():
    run = _run("cancel_twice")
    assert _codes(run)[1][:2].tolist() == [OK, UNSUPPORTED]
    led = _ledger(run)
    assert int(led.disruptive_used[0]) == 1
    assert float(led.balance[0, DISRUPTIVE_COL]) == 9.0
    assert int(led.cm_state[0]) == CommitState.CANCELLED
    assert _sound(run)


def test_a_commitment_beyond_the_capacity_is_refused_and_the_table_keeps_its_rows():
    run = _run("two_commitments", k=1)
    assert _codes(run)[0][:2].tolist() == [OK, NO_SLOT]
    assert int((np.asarray(_ledger(run).cm_state) != CommitState.NONE).sum()) == 1
    assert _sound(run)


def test_control_the_same_two_commitments_fit_in_a_table_of_four():
    run = _run("two_commitments")
    assert _codes(run)[0][:2].tolist() == [OK, OK]
    assert int((np.asarray(_ledger(run).cm_state) != CommitState.NONE).sum()) == 2


def test_a_group_that_asks_for_more_slots_than_remain_is_refused_whole():
    run = _run("group_of_two_commitments", k=1)
    assert _codes(run)[0][:2].tolist() == [NO_SLOT, NO_SLOT]
    assert int((np.asarray(_ledger(run).cm_state) != CommitState.NONE).sum()) == 0
    assert not _log(run)["status"].size
    assert _sound(run)


# ---- the boundary of the horizon and a lane that is running --------------------------


def test_a_start_exactly_a_horizon_away_is_not_an_insertion():
    run = _run("at_the_horizon")
    assert _codes(run)[0][0] == OK
    assert int(_ledger(run).disruptive_used[0]) == 0


def test_control_a_start_one_tick_inside_the_horizon_is_refused():
    assert _codes(_run("inside_by_one"))[0][0] == FROZEN


def test_a_started_commitment_holds_its_lane_for_the_run():
    run = _run("request_during_run")
    assert _codes(run)[1][0] == BUSY
    log = _log(run)
    assert log["status"].tolist() == [RecordStatus.COMPLETED]
    assert log["end"].tolist() == [210]
    assert _sound(run)


# ---- a pending cancel credits what its price allows ----------------------------------

CHARGE = (6.0, 4.0, 2.0)
SURCHARGE = (2.0, 1.0, 0.5)
BALANCE = 100.0
# the part of the charge each pricing keeps: none, none, the surcharge, all of it
KEPT = {
    ChargePricing.OCCUPANCY: (0.0, 0.0, 0.0),
    ChargePricing.STATISTICAL: (0.0, 0.0, 0.0),
    ChargePricing.SURCHARGE: SURCHARGE,
    ChargePricing.FLAT_FRACTION: CHARGE,
}
FIXED = {m: (SURCHARGE if m == ChargePricing.SURCHARGE else (0.0,) * 3) for m in KEPT}


@functools.cache
def _priced_run(mode):
    script, plans = SCENARIOS["cancel_outside"]
    study = commitment_study(script, plans, refunds=True)
    study = priced(study, mode, CHARGE, SURCHARGE, BALANCE)
    truth, root = one_world(caps=study.caps)
    result = HostExecutor().run(study, truth, root, N_STEPS)
    return SimpleNamespace(study=study, carry=result.carry, trace=result.trace)


@pytest.mark.parametrize("mode", list(KEPT), ids=lambda m: m.name.lower())
def test_a_pending_cancel_credits_the_refundable_part_of_its_price(mode):
    run = _priced_run(mode)
    assert _codes(run)[1][0] == OK
    led = _ledger(run)
    assert int(led.cm_state[0]) == CommitState.CANCELLED
    # the asset, program and shared rows each hold one resource of the charge
    rows = (0, run.study.caps.n_assets, run.study.caps.n_balance_rows - 1)
    held = np.asarray(led.balance)[rows, (0, 1, 2)]
    np.testing.assert_allclose(held, BALANCE - np.asarray(KEPT[mode]))
    log = run.carry.log
    valid = np.asarray(log.valid)
    assert np.asarray(log.status)[valid].tolist() == [RecordStatus.PARTIAL]
    assert np.asarray(log.payload)[valid][:, 2].tolist() == [0.0]
    np.testing.assert_allclose(np.asarray(log.charged)[valid][0], KEPT[mode])
    np.testing.assert_allclose(np.asarray(log.nonrefundable)[valid][0], FIXED[mode])
    assert _sound(run)
