"""Survey 12: a failure sets the ops state; cost and floor follow; a record reports it.

The study (`spaceodyssey.studies.survey_12`) fails asset 0 at tick 100 and writes a
notice released 25 ticks later. Five host runs share one compiled structure (every
number that differs is a leaf): the fixed plan with a cancel, the record-reading
policy at equal costs with the declared delay and with it doubled, the same policy at
unequal costs, and a balance-reading policy at unequal costs (the negative control
of the causal check).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from spaceodyssey.core import HostExecutor
from spaceodyssey.core.enums import EffectKind, RecordStatus
from spaceodyssey.studies.survey_12 import (
    COSTS,
    DELAY,
    FAIL,
    FAILURE_CHANNEL,
    FLOORS,
    survey_12,
)
from tests.core.helpers import one_world

pytestmark = pytest.mark.slow(
    reason="one compiled structure shared by five host runs, about 15 s cold"
)

N_STEPS = 80
CANCEL = 150
LOW, HIGH = COSTS
EQUAL = (LOW, LOW)  # cost_by_ops[1] == cost_by_ops[0]: the notice is the only route


def _run(study):
    """A host run of one world with its decisions: `(run, truth, [(tick, effects)])`."""
    truth, root = one_world(caps=study.caps)
    views, effects = [], []
    run = HostExecutor().run(
        study, truth, root, N_STEPS, view_log=views, effects_log=effects
    )
    trace = run.trace
    keep = np.asarray(trace.handled) & np.asarray(trace.wake)
    ticks = np.asarray(trace.tick)[keep]
    assert len(ticks) == len(effects)
    decisions = [(int(t), e) for t, e in zip(ticks, effects, strict=True)]
    return SimpleNamespace(run=run, truth=truth, decisions=decisions)


def records(result):
    """The valid log rows as dicts."""
    log = result.run.carry.log
    out = []
    for i in np.flatnonzero(np.asarray(log.valid)):
        out.append(
            dict(
                channel=int(log.channel[i]),
                subject=int(log.subject[i]),
                status=int(log.status[i]),
                start=int(log.start_tick[i]),
                ops=int(log.ops[i]),
                charged=float(log.charged[i][0]),
                released=int(log.released_tick[i]),
                payload=np.asarray(log.payload[i]),
            )
        )
    return out


def first_changed_tick(result):
    """The tick of the first decision that re-points the observation to subject 1."""
    ticks = [
        t
        for t, e in result.decisions
        if int(e.kind[0]) == EffectKind.OBSERVE and int(e.subject[0]) == 1
    ]
    return min(ticks) if ticks else None


def release_tick(result):
    """The release tick of the failure notice, from the log."""
    notice = [r for r in records(result) if r["channel"] == FAILURE_CHANNEL]
    assert len(notice) == 1
    return notice[0]["released"]


def acts_only_from_the_notice(result):
    """The causal check: the first changed action is not before the notice's release."""
    changed = first_changed_tick(result)
    return changed is not None and changed >= release_tick(result)


@pytest.fixture(scope="module")
def runs():
    """Fixed plan with a cancel (unequal costs), then the four adaptive runs."""
    return SimpleNamespace(
        fixed=_run(survey_12(react=0, cancel_tick=CANCEL)),
        equal=_run(survey_12(costs=EQUAL)),
        slow=_run(survey_12(costs=EQUAL, delay=2 * DELAY)),
        unequal=_run(survey_12()),
        balance=_run(survey_12(react=0, read_balance=1)),
    )


def test_admissions_after_the_failure_are_debited_and_realized_in_state_one(runs):
    """Fails if admission prices from the table of state 0 after the failure, the floor
    ignores the stored state, or a refund of such an admission is not the debit less
    what is kept: the two admissions after tick 100 cost 5 and realize with floor 0.5,
    the one before costs 2 with floor 0.1, and the cancelled one is credited exactly.
    """
    r = runs.fixed
    log = [x for x in records(r) if x["channel"] != FAILURE_CHANNEL]
    after = [x for x in log if x["ops"] == 1]
    before = [x for x in log if x["ops"] == 0]
    assert len(after) == 2 and len(before) == 1
    assert all(x["start"] > FAIL for x in after)
    ledger = r.run.carry.ledger
    assert float(ledger.debited[0, 0]) == pytest.approx(LOW + 2 * HIGH)
    for x in log:
        gain, signal = 1.0, float(r.truth.signal[x["subject"]])
        resid = x["payload"][0] - gain * signal
        assert resid == pytest.approx(FLOORS[x["ops"]] * x["payload"][1])
        assert abs(x["payload"][1]) > 0
    cancelled = [x for x in after if x["status"] == int(RecordStatus.PARTIAL)]
    assert len(cancelled) == 1
    assert 0.0 < cancelled[0]["charged"] < HIGH
    refund = float(ledger.refunded[0, 0])
    assert refund > 0
    assert refund + cancelled[0]["charged"] == pytest.approx(HIGH)


def test_the_failure_record_is_released_after_the_declared_delay(runs):
    """Fails if the notice is released at the failure tick or the delay is ignored, or
    the fixed plan acts on it: release equals failure plus delay and no action changes.
    """
    assert release_tick(runs.fixed) == FAIL + DELAY
    assert release_tick(runs.slow) == FAIL + 2 * DELAY
    assert first_changed_tick(runs.fixed) is None


def test_the_policy_acts_only_after_the_notice_is_released(runs):
    """Fails if anything but the released notice reaches the policy in the equal-cost
    variant, or the delay does not move the action: the first changed action is at the
    release tick, and doubling the delay moves it by the added delay. Control: a policy
    reading balances in the unequal-cost variant changes its action before the release,
    so the check fails for it (the notice is not the only route there).
    """
    equal, slow = runs.equal, runs.slow
    assert acts_only_from_the_notice(equal) and acts_only_from_the_notice(slow)
    first, later = first_changed_tick(equal), first_changed_tick(slow)
    assert first == release_tick(equal) == FAIL + DELAY
    assert later - first == DELAY
    assert not acts_only_from_the_notice(runs.balance)
    assert first_changed_tick(runs.balance) < release_tick(runs.balance)
    # the record reader is not moved by the unequal costs: the notice is still its cue
    assert first_changed_tick(runs.unequal) == FAIL + DELAY
