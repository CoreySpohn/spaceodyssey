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

from spaceodyssey.core import HostExecutor
from spaceodyssey.core.conformance.invariants import balance_conserved
from spaceodyssey.core.enums import AdmissionCode, CommitState, EffectKind, RecordStatus
from tests.core.commitment_helpers import (
    DISRUPTIVE_COL,
    commitment_study,
    plan,
)
from tests.core.helpers import one_world
from tests.core.program_helpers import cancel, end, observe, wait

pytestmark = pytest.mark.medium  # A cold host run compiles the executor.

OK = int(AdmissionCode.OK)
FROZEN = int(AdmissionCode.FROZEN_HORIZON)
BUSY = int(AdmissionCode.LANE_BUSY)
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


def _study(name, **kw):
    script, plans = SCENARIOS[name]
    return commitment_study(script, plans, **kw)


@functools.cache
def _run(name, freeze=30, tight=False):
    study = _study(name, freeze=freeze)
    if tight:
        # the asset holds only what its channel takes, so it has no activation to spare
        budget = study.budget.at[0, DISRUPTIVE_COL].set(1.0)
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
    assert balance_conserved(run).passed


def test_an_immediate_request_on_a_frozen_asset_is_refused_without_an_activation():
    run = _run("immediate")
    assert _codes(run)[0][0] == FROZEN
    assert not _log(run)["status"].size
    led = _ledger(run)
    assert float(led.balance[0, DISRUPTIVE_COL]) == 10.0
    assert int(led.disruptive_used[0]) == 0
    assert balance_conserved(run).passed


def test_control_the_same_request_with_an_activation_is_admitted_and_debits_one():
    run = _run("immediate_disruptive")
    assert _codes(run)[0][0] == OK
    led = _ledger(run)
    assert int(led.disruptive_used[0]) == 1
    # the charge of the channel and the activation, from the same cell
    assert float(led.balance[0, DISRUPTIVE_COL]) == 8.0
    assert _log(run)["status"].tolist() == [RecordStatus.COMPLETED]
    assert balance_conserved(run).passed


def test_an_activation_the_asset_cannot_cover_does_not_break_the_freeze():
    run = _run("immediate_disruptive", tight=True)
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
    assert balance_conserved(run).passed


def test_control_a_disruptive_cancel_inside_the_horizon_cancels_and_costs_one():
    run = _run("cancel_inside_disruptive")
    assert _codes(run)[1][0] == OK
    led = _ledger(run)
    assert int(led.cm_state[0]) == CommitState.CANCELLED
    assert int(led.disruptive_used[0]) == 1
    # the charge came back, the activation did not: 10 - 1 + 1 - 1
    assert float(led.balance[0, DISRUPTIVE_COL]) == 9.0
    assert _log(run)["status"].tolist() == [RecordStatus.PARTIAL]
    assert balance_conserved(run).passed


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
    assert balance_conserved(run).passed


def test_cancelling_an_activity_that_has_started_is_not_frozen():
    run = _run("cancel_started")
    assert _codes(run)[1][0] == OK
    assert _log(run)["status"].tolist() == [RecordStatus.PARTIAL]
    assert int(_ledger(run).disruptive_used[0]) == 0
    assert balance_conserved(run).passed


def test_work_that_ends_before_a_reservation_is_admitted():
    run = _run("before_reservation")
    assert _codes(run)[0][:2].tolist() == [OK, OK]
    log = _log(run)
    assert sorted(log["end"].tolist()) == [55, 155]
    assert set(log["status"].tolist()) == {RecordStatus.COMPLETED}
    assert balance_conserved(run).passed


def test_work_that_overlaps_a_reservation_is_refused():
    run = _run("overlaps_reservation")
    assert _codes(run)[1][0] == BUSY
    assert _log(run)["end"].tolist() == [155]
    assert balance_conserved(run).passed
