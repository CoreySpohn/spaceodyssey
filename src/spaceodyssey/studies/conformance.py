"""The conformance corpus: study builders, hand histories and the cases.

The cases are studies, so they live here and not in the core; the core holds the case
type, the runner and the invariants (`spaceodyssey.core.conformance`).

Hand histories are tuples of (tick, stage, kind) in handled order. Every history starts
with the kick-off wake (0, Stage.ADMIT, EventKind.WAIT), where the policy decides first.
"""

import equinox as eqx
import numpy as np

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.conformance.corpus import CorpusCase
from spaceodyssey.core.enums import (
    AdmissionCode,
    Audience,
    Disposition,
    EffectKind,
    EventKind,
    RecordStatus,
    RunStatus,
    Scope,
    Stage,
    effect_mask,
)
from spaceodyssey.studies import catalog, oracle
from spaceodyssey.studies import toy as kinds
from spaceodyssey.studies.scripted import scripted_study
from spaceodyssey.studies.toy import RECORD_AUXILIARY

S, R, A = Stage.SETTLE, Stage.RELEASE, Stage.ADMIT
DONE, REL, WAIT, STOP, CANCEL = (
    EventKind.DONE,
    EventKind.RELEASE,
    EventKind.WAIT,
    EventKind.STOP,
    EventKind.CANCEL,
)
PROCESS, REQUEST, RESULT = EventKind.PROCESS, EventKind.REQUEST, EventKind.RESULT

KICKOFF = (0, A, WAIT)

HAND_HISTORIES = {
    "baseline": (
        KICKOFF,
        (110, S, DONE),
        (130, R, REL),
        (240, S, DONE),
        (260, R, REL),
        (370, S, DONE),
        (390, R, REL),
        (500, S, DONE),
        (520, R, REL),
        (630, S, DONE),
        (650, R, REL),
        (650, A, STOP),
    ),
    "wait": (
        KICKOFF,
        (110, S, DONE),
        (130, R, REL),
        (180, R, WAIT),
        (290, S, DONE),
        (310, R, REL),
        (420, S, DONE),
        (440, R, REL),
        (550, S, DONE),
        (570, R, REL),
        (680, S, DONE),
        (700, R, REL),
        (700, A, STOP),
    ),
    "stop_lane_close": (
        KICKOFF,
        (110, S, DONE),
        (130, R, REL),
        (185, S, DONE),
        (185, R, REL),
        (185, A, STOP),
        (240, S, DONE),
        (260, R, REL),
    ),
    "stop_campaign_abort": (
        KICKOFF,
        (110, S, DONE),
        (130, R, REL),
        (185, S, DONE),
        (185, R, REL),
        (185, A, STOP),
    ),
    "cancel_live": (
        KICKOFF,
        (110, S, DONE),
        (130, R, REL),
        (185, S, DONE),
        (185, R, REL),
        (185, A, CANCEL),
        (205, R, REL),
        (315, S, DONE),
        (335, R, REL),
        (445, S, DONE),
        (465, R, REL),
        (465, A, STOP),
    ),
    "coincident": (
        KICKOFF,
        (55, S, DONE),
        (55, R, REL),
        (105, R, WAIT),
        (160, S, DONE),
        (160, S, DONE),
        (160, R, REL),
        (160, A, STOP),
        (180, R, REL),
    ),
    # the two bundled occupancies release at 260; one decision sees both records
    "bundle": (
        KICKOFF,
        (110, S, DONE),
        (130, R, REL),
        (240, S, DONE),
        (260, S, DONE),
        (260, R, REL),
        (260, R, REL),
        (370, S, DONE),
        (390, R, REL),
        (500, S, DONE),
        (520, R, REL),
        (630, S, DONE),
        (650, R, REL),
        (650, A, STOP),
    ),
    "phase_switch": (
        KICKOFF,
        (110, S, DONE),
        (130, R, REL),
        (240, S, DONE),
        (260, R, REL),
        (370, S, DONE),
        (390, R, REL),
        (445, S, DONE),
        (445, R, REL),
        (500, S, DONE),
        (500, R, REL),
        (500, A, STOP),
    ),
    # two releases at 130, from two lanes: one decision sees both
    "coalesced_release": (
        KICKOFF,
        (110, S, DONE),
        (130, S, DONE),
        (130, R, REL),
        (130, R, REL),
        (240, S, DONE),
        (260, R, REL),
        (370, S, DONE),
        (390, R, REL),
        (500, S, DONE),
        (520, R, REL),
        (630, S, DONE),
        (650, R, REL),
        (650, A, STOP),
    ),
    # the bundle at 130 needs 2 of the 1.5 left in the program: neither row is admitted
    # and no wake follows, so the run ends idle
    "bundle_unaffordable": (
        KICKOFF,
        (110, S, DONE),
        (130, R, REL),
    ),
    # the Stop at the second decision (55) and the closing Stop (650) are forbidden by
    # the mask and refused; the plan carries on between them
    "forbidden_effect": (
        KICKOFF,
        (55, S, DONE),
        (55, R, REL),
        (110, S, DONE),
        (130, R, REL),
        (240, S, DONE),
        (260, R, REL),
        (370, S, DONE),
        (390, R, REL),
        (500, S, DONE),
        (520, R, REL),
        (630, S, DONE),
        (650, R, REL),
    ),
    # the catalog row (channel 13) is released to the observer at 325 and wakes no one
    "catalog_record": (
        KICKOFF,
        (110, S, DONE),
        (130, R, REL),
        (175, S, DONE),
        (290, S, DONE),
        (310, R, REL),
        (325, R, REL),
        (420, S, DONE),
        (440, R, REL),
        (550, S, DONE),
        (570, R, REL),
        (680, S, DONE),
        (700, R, REL),
        (700, A, STOP),
    ),
    # the question at 130 is answered at its own tick and stage, and the answer wakes
    # the decision that resumes the plan
    "host_service": (
        KICKOFF,
        (110, S, DONE),
        (130, R, REL),
        (130, A, REQUEST),
        (130, A, RESULT),
        (240, S, DONE),
        (260, R, REL),
        (370, S, DONE),
        (390, R, REL),
        (500, S, DONE),
        (520, R, REL),
        (630, S, DONE),
        (650, R, REL),
        (650, A, STOP),
    ),
    # two revisions of one acquisition arrive at 100 and 200
    "revision": (
        KICKOFF,
        (100, R, PROCESS),
        (100, R, REL),
        (200, R, PROCESS),
        (200, R, REL),
    ),
}


# Per handled row: (lane, subject, acquisition_index, n_written, wake), beside
# HAND_HISTORIES.
HAND_ROWS = {
    "baseline": (
        (0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
        (0, 1, 0, 1, 0),
        (0, 1, 0, 0, 1),
        (0, 0, 1, 1, 0),
        (0, 0, 1, 0, 1),
        (0, 1, 1, 1, 0),
        (0, 1, 1, 0, 1),
        (0, 0, 2, 1, 0),
        (0, 0, 2, 0, 1),
        (0, 0, 0, 0, 0),
    ),
    "wait": (
        (0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
        (0, 0, 0, 0, 1),
        (0, 1, 0, 1, 0),
        (0, 1, 0, 0, 1),
        (0, 0, 1, 1, 0),
        (0, 0, 1, 0, 1),
        (0, 1, 1, 1, 0),
        (0, 1, 1, 0, 1),
        (0, 0, 2, 1, 0),
        (0, 0, 2, 0, 1),
        (0, 0, 0, 0, 0),
    ),
    "stop_lane_close": (
        (0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
        (2, 1, 1, 1, 0),
        (2, 1, 1, 0, 1),
        (0, 0, 0, 0, 0),
        (0, 1, 0, 1, 0),
        (0, 1, 0, 0, 1),
    ),
    "stop_campaign_abort": (
        (0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
        (2, 1, 1, 1, 0),
        (2, 1, 1, 0, 1),
        (0, 0, 0, 1, 0),
    ),
    "cancel_live": (
        (0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
        (2, 1, 1, 1, 0),
        (2, 1, 1, 0, 1),
        (0, 0, 0, 1, 0),
        (0, 1, 0, 0, 1),
        (0, 1, 2, 1, 0),
        (0, 1, 2, 0, 1),
        (0, 0, 1, 1, 0),
        (0, 0, 1, 0, 1),
        (0, 0, 0, 0, 0),
    ),
    "coincident": (
        (0, 0, 0, 0, 1),
        (2, 0, 1, 1, 0),
        (2, 0, 1, 0, 1),
        (0, 0, 0, 0, 1),
        (1, 0, 0, 1, 0),
        (2, 1, 0, 1, 0),
        (2, 1, 0, 0, 1),
        (0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
    ),
    "bundle": (
        (0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
        (0, 1, 0, 1, 0),
        (2, 1, 1, 1, 0),
        (0, 1, 0, 0, 0),
        (2, 1, 1, 0, 1),
        (0, 0, 1, 1, 0),
        (0, 0, 1, 0, 1),
        (0, 1, 2, 1, 0),
        (0, 1, 2, 0, 1),
        (0, 0, 2, 1, 0),
        (0, 0, 2, 0, 1),
        (0, 0, 0, 0, 0),
    ),
    "phase_switch": (
        (0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
        (0, 1, 0, 1, 0),
        (0, 1, 0, 0, 1),
        (0, 0, 1, 1, 0),
        (0, 0, 1, 0, 1),
        (2, 0, 2, 1, 0),
        (2, 0, 2, 0, 1),
        (2, 1, 1, 1, 0),
        (2, 1, 1, 0, 1),
        (0, 0, 0, 0, 0),
    ),
    "coalesced_release": (
        (0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0),
        (2, 0, 1, 1, 0),
        (0, 0, 0, 0, 0),
        (2, 0, 1, 0, 1),
        (0, 1, 0, 1, 0),
        (0, 1, 0, 0, 1),
        (0, 0, 2, 1, 0),
        (0, 0, 2, 0, 1),
        (0, 1, 1, 1, 0),
        (0, 1, 1, 0, 1),
        (0, 0, 3, 1, 0),
        (0, 0, 3, 0, 1),
        (0, 0, 0, 0, 0),
    ),
    "bundle_unaffordable": (
        (0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
    ),
    "forbidden_effect": (
        (0, 0, 0, 0, 1),
        (2, 0, 1, 1, 0),
        (2, 0, 1, 0, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
        (0, 1, 0, 1, 0),
        (0, 1, 0, 0, 1),
        (0, 0, 2, 1, 0),
        (0, 0, 2, 0, 1),
        (0, 1, 1, 1, 0),
        (0, 1, 1, 0, 1),
        (0, 0, 3, 1, 0),
        (0, 0, 3, 0, 1),
    ),
    "catalog_record": (
        (0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
        (2, 0, 1, 1, 0),
        (1, 0, 2, 1, 0),
        (1, 0, 2, 0, 1),
        (2, 0, 1, 0, 0),
        (0, 1, 0, 1, 0),
        (0, 1, 0, 0, 1),
        (0, 0, 3, 1, 0),
        (0, 0, 3, 0, 1),
        (0, 1, 1, 1, 0),
        (0, 1, 1, 0, 1),
        (0, 0, 0, 0, 0),
    ),
    "host_service": (
        (0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1),
        (0, 1, 0, 1, 0),
        (0, 1, 0, 0, 1),
        (0, 0, 1, 1, 0),
        (0, 0, 1, 0, 1),
        (0, 1, 1, 1, 0),
        (0, 1, 1, 0, 1),
        (0, 0, 2, 1, 0),
        (0, 0, 2, 0, 1),
        (0, 0, 0, 0, 0),
    ),
    "revision": (
        (0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
    ),
}


# ---- invariants of the hand histories: the side claims of the plan text -----------


def _step(h, tick, stage, kind):
    return h.events.index((tick, stage, kind))


def _partials(h):
    i = np.flatnonzero(h.ints["status"] == RecordStatus.PARTIAL)
    return i


def _all_completed(h):
    return bool(np.all(h.ints["status"] == RecordStatus.COMPLETED))


def _one_partial_at(h, subject, acquisition_index, fraction):
    i = _partials(h)
    return (
        len(i) == 1
        and h.ints["subject"][i[0]] == subject
        and h.ints["acquisition_index"][i[0]] == acquisition_index
        and abs(h.floats["payload"][i[0], 2] - fraction) < 1e-12
    )


def _no_refusal(h):
    refused = (h.ints["admit_code"] != AdmissionCode.OK) & (
        h.ints["admit_code"] != AdmissionCode.NONE
    )
    return not bool(refused.any())


HAND_INVARIANTS = {
    "baseline": (
        lambda h: h.ints["n_dec"][-1] == 6 and h.n_records == 5,
        _all_completed,
        lambda h: h.flags == (True, False, False, False, False),
    ),
    "wait": (
        lambda h: h.ints["n_dec"][-1] == 7 and h.n_records == 5,
        _all_completed,
    ),
    "stop_lane_close": (
        # the decision at the 260 Release is refused on the closed lane 0
        lambda h: (
            h.ints["admit_code"][_step(h, 260, Stage.RELEASE, EventKind.RELEASE)][0]
            == AdmissionCode.LANE_CLOSED
        ),
        lambda h: h.flags[:2] == (True, True),
        lambda h: h.n_records == 3,
    ),
    "stop_campaign_abort": (
        lambda h: _one_partial_at(h, 1, 0, 0.45),
        lambda h: bool(
            h.ints["released_tick"][_partials(h)[0]] == -1
            and h.ints["accepted_tick"][_partials(h)[0]] == -1
        ),
        lambda h: h.flags[0] and not h.flags[1] and h.n_records == 3,
    ),
    "cancel_live": (
        lambda h: _one_partial_at(h, 1, 0, 0.45),
        # the re-observation of subject 1 takes the next acquisition_index index, 2
        lambda h: bool(
            np.any((h.ints["subject"] == 1) & (h.ints["acquisition_index"] == 2))
        ),
    ),
    "coincident": (
        lambda h: h.rows[4][3] == 1 and h.rows[5][3] == 2,
        lambda h: not h.double_settle and h.n_records == 3,
    ),
    "bundle": (
        # one decision at 260 sees both records, so no plan entry meets a busy lane
        lambda h: h.n_records == 6 and _all_completed(h),
        _no_refusal,
        # the bundle admits two acquisitions of subject 1 at one decision
        lambda h: (
            sorted(h.ints["acquisition_index"][h.ints["subject"] == 1].tolist())
            == [0, 1, 2]
        ),
        lambda h: h.ints["n_dec"][-1] == 6,
    ),
    "phase_switch": (
        lambda h: h.ints["phase"].tolist() == [0, 0, 0, 1, 1],
        lambda h: h.ints["vintage"].tolist() == [0, 0, 0, 1, 1],
        # phase B restarts its plan: subject 0, channel 12 first
        lambda h: (h.ints["subject"][3], h.ints["channel"][3]) == (0, 12),
    ),
    "coalesced_release": (
        # the two records released at 130 are seen by one decision: six decisions in all
        lambda h: h.ints["n_dec"][-1] == 6 and h.n_records == 6,
        _all_completed,
        _no_refusal,
        lambda h: (
            sum(e == (130, Stage.RELEASE, EventKind.RELEASE) for e in h.events) == 2
        ),
    ),
    "bundle_unaffordable": (
        # both rows of the group are refused for funds, and nothing of it is kept
        lambda h: (
            h.ints["admit_code"][
                _step(h, 130, Stage.RELEASE, EventKind.RELEASE)
            ].tolist()
            == [AdmissionCode.NO_FUNDS] * 2
        ),
        lambda h: h.n_records == 1 and h.ints["acquisition_counts"][0] == 1,
        lambda h: h.ints["next_sid"] == 3,
        lambda h: h.flags == (True, True, False, False, False),
    ),
    "forbidden_effect": (
        # a Stop the mask forbids is refused whole and costs nothing
        lambda h: (
            h.ints["admit_code"][_step(h, 55, Stage.RELEASE, EventKind.RELEASE)][0]
            == AdmissionCode.FORBIDDEN
        ),
        lambda h: (
            h.ints["admit_code"][_step(h, 650, Stage.RELEASE, EventKind.RELEASE)][0]
            == AdmissionCode.FORBIDDEN
        ),
        lambda h: h.n_records == 6 and _all_completed(h),
        lambda h: h.flags == (True, True, False, False, False),
    ),
    "catalog_record": (
        # the catalog row (channel 13) reaches the observer at 325 and never the policy
        lambda h: _catalog_row_is_observer_only(h, 325),
        lambda h: h.ints["belief_n"].sum() == 5 and h.ints["pbelief_n"].sum() == 5,
        lambda h: _all_completed(h) and h.n_records == 6,
    ),
    "host_service": (
        # the request and the result name the snapshot of the decision that asked
        lambda h: _service_names_one_snapshot(h),
        # the answer is one more record, released at the tick it was asked
        lambda h: _answer_is_released_at(h, 130),
        lambda h: h.status == RunStatus.COMPLETE and h.n_records == 6,
    ),
    "revision": (
        lambda h: h.ints["revision"].tolist() == [0, 1],
        lambda h: h.ints["superseded_by"].tolist() == [1, -1],
        # the observer folds the first revision it accepts, once
        lambda h: h.ints["belief_n"][0] == 1 and h.floats["belief_final"][0] == 2.0,
    ),
}


def _catalog_row_is_observer_only(h, tick):
    row = np.flatnonzero(h.ints["channel"] == catalog.CATALOG_CHANNEL_ID)
    return bool(
        len(row) == 1
        and h.ints["kind"][row[0]] == catalog.register_catalog(kinds.toy_registry())
        and h.floats["belief_aux"][catalog.CATALOG_AUX_SLOT]
        == h.floats["payload"][row[0], 0]
        and h.ints["accepted_tick"][row[0]] == tick
        and h.ints["released_tick"][row[0]] == -1
        and h.ints["audience"][row[0]] == Audience.OBSERVER
    )


def _service_names_one_snapshot(h):
    i = h.events.index((130, Stage.ADMIT, EventKind.REQUEST))
    asked = h.ints["snapshot_fp"][i - 1]
    return bool(
        asked != 0
        and h.ints["event_fp"][i] == asked
        and h.ints["event_fp"][i + 1] == asked
    )


def _answer_is_released_at(h, tick):
    row = np.flatnonzero((h.ints["start_tick"] == tick) & (h.ints["end_tick"] == tick))
    return bool(len(row) == 1 and h.ints["released_tick"][row[0]] == tick)


# ---- the conformance corpus ---------------------------------------------------------


def stop_grid():
    """Build the twelve combinations of stop scope and disposition."""

    def make_case(scope_name, scope, disp_name, disp):
        def make_study():
            return kinds.survey_1(
                policies=(
                    kinds.FixedPlanPolicy(
                        plan=((0, 10), (1, 10), (0, 10), (1, 10), (0, 10)),
                        bundle_at=1,
                        stop_at=2,
                        stop_scope=scope,
                        stop_disp=disp,
                    ),
                )
            )

        campaign_stop = scope == Scope.CAMPAIGN and disp != Disposition.CLOSE
        invariants = (
            lambda h: h.n_records == 3,
            lambda h: h.flags == (True, not campaign_stop, False, False, False),
            lambda h: (
                bool(np.any(h.ints["admit_code"] == AdmissionCode.LANE_CLOSED))
                == (not campaign_stop)
            ),
        )
        if disp == Disposition.ABORT:
            invariants += (
                lambda h: _one_partial_at(h, 1, 0, 0.45),
                lambda h: (
                    np.count_nonzero(h.ints["status"] == RecordStatus.COMPLETED) == 2
                ),
            )
            if campaign_stop:
                invariants += (
                    lambda h: bool(
                        np.all(h.ints["released_tick"][_partials(h)] == -1)
                        and np.all(h.ints["accepted_tick"][_partials(h)] == -1)
                    ),
                )
            else:
                invariants += (
                    lambda h: bool(
                        np.all(h.ints["released_tick"][_partials(h)] >= 0)
                        and np.all(h.ints["accepted_tick"][_partials(h)] >= 0)
                    ),
                )
        else:
            invariants += (_all_completed,)

        name = f"stop_{scope_name}_{disp_name}"
        return CorpusCase(
            name=name,
            make_study=make_study,
            n_seeds=64,
            hand_history=HAND_HISTORIES.get(name),
            invariants=invariants,
            hand_rows=HAND_ROWS.get(name),
        )

    return [
        make_case(scope_name, scope, disp_name, disp)
        for scope_name, scope in (
            ("lane", Scope.LANE),
            ("asset", Scope.ASSET),
            ("program", Scope.PROGRAM),
            ("campaign", Scope.CAMPAIGN),
        )
        for disp_name, disp in (
            ("close", Disposition.CLOSE),
            ("drain", Disposition.DRAIN),
            ("abort", Disposition.ABORT),
        )
    ]


# ---- study builders ---------------------------------------------------------------

PLAN5 = ((0, 10), (1, 10), (0, 10), (1, 10), (0, 10))


def frozen(plan=PLAN5, **kw):
    """A fixed-plan policy; keywords set its special decisions."""
    return kinds.FixedPlanPolicy(plan=plan, **kw)


HAND_STUDIES = {
    "baseline": lambda: kinds.survey_1(),
    "wait": lambda: kinds.survey_1(policies=(frozen(wait_at=1),)),
    "stop_lane_close": lambda: kinds.survey_1(
        policies=(
            frozen(
                bundle_at=1,
                stop_at=2,
                stop_scope=Scope.LANE,
                stop_disp=Disposition.CLOSE,
            ),
        )
    ),
    "stop_campaign_abort": lambda: kinds.survey_1(
        policies=(
            frozen(
                bundle_at=1,
                stop_at=2,
                stop_scope=Scope.CAMPAIGN,
                stop_disp=Disposition.ABORT,
            ),
        )
    ),
    "cancel_live": lambda: kinds.survey_1(
        policies=(
            frozen(plan=((0, 10), (1, 10), (1, 10), (0, 10)), bundle_at=1, cancel_at=2),
        )
    ),
    "coincident": lambda: kinds.survey_1(
        policies=(frozen(plan=((0, 11), (1, 12)), bundle_at=0, wait_at=1),)
    ),
    "bundle": lambda: _retimed_bundle_study(),
    "phase_switch": lambda: kinds.survey_1(
        policies=(
            frozen(plan=((0, 10), (1, 10), (0, 10))),
            frozen(plan=((0, 12), (1, 12))),
        ),
        phase_until=(300, 1000),
    ),
}


def _retimed_bundle_study():
    """A bundle whose two occupancies release at one tick.

    Channel 12 occupies its lane for 125 ticks instead of 50, so the bundle's records
    release together at 260 and one decision sees both.
    """
    base = kinds.survey_1(policies=(frozen(bundle_at=1),))
    occupancy = base.channels.occupancy_ticks.at[2].set(125)
    return eqx.tree_at(lambda st: st.channels.occupancy_ticks, base, occupancy)


def _coalesced_release_study():
    """A bundle at the first decision whose two records release at one tick.

    Channel 12 occupies its lane for 125 ticks, so its record (no latency) releases at
    130, the tick the channel 10 record (done at 110, latency 20) releases.
    """
    base = kinds.survey_1(policies=(frozen(bundle_at=0),))
    occupancy = base.channels.occupancy_ticks.at[2].set(125)
    return eqx.tree_at(lambda st: st.channels.occupancy_ticks, base, occupancy)


def _unaffordable_bundle_study():
    """The retimed bundle with 2.5 left in the program.

    The first observation leaves 1.5 of the 2 the bundle needs.
    """
    base = _retimed_bundle_study()
    row = base.caps.n_assets
    return eqx.tree_at(lambda st: st.budget, base, base.budget.at[row, 1].set(2.5))


def _forbidden_effect_study():
    """A phase that permits Observe only, under a plan that also asks to Stop."""
    return kinds.survey_1(
        policies=(frozen(bundle_at=0, stop_at=1),),
        permitted_effects=(effect_mask(EffectKind.OBSERVE),),
    )


def _catalog_record_study():
    """A plan that observes the catalog channel at its second decision.

    The catalog row is released to the observer 150 ticks after its occupancy ends.
    The bundle there adds channel 11, so the policy keeps waking while the row is held.
    """
    plan = ((0, 10), (0, catalog.CATALOG_CHANNEL_ID), (1, 10), (0, 10), (1, 10))
    return catalog.catalog_study(
        policies=(frozen(plan=plan, bundle_at=1, bundle_ch=11),)
    )


def _host_service_study():
    return oracle.oracle_study()[0]


def _host_service():
    return oracle.ToyOracleService(oracle.oracle_registry()[1])


def _revision_study():
    """Two revisions of one acquisition, at 100 and 200, with values 2 and 9."""
    return scripted_study(
        ticks=(100, 200), values=(2.0, 9.0), ids=(7001, 7001), revisions=(0, 1)
    )


def _delayed_release_study():
    """Handle a transaction state update.

    Channel 11 (latency 20) logs records of the extra kind, which only the observer in
    `kinds.py` interprets.
    """
    base = kinds.survey_1(policies=(frozen(plan=((0, 10), (1, 11), (0, 10))),))
    kind = base.channels.kind.at[1].set(RECORD_AUXILIARY)
    return eqx.tree_at(lambda st: st.channels.kind, base, kind)


def _aux_changes_only_at_release(h):
    aux = h.floats["belief_aux_path"]
    changed = [i for i in range(1, len(aux)) if np.any(aux[i] != aux[i - 1])]
    return len(changed) >= 1 and all(
        h.events[i][2] == EventKind.RELEASE for i in changed
    )


def _case(
    name, make, hand=None, inv=(), forms=("array", "python"), n_seeds=64, service=None
):
    return CorpusCase(
        name, make, n_seeds, hand, tuple(inv), forms, HAND_ROWS.get(name), service
    )


def build_corpus():
    """Build every case of the corpus, in a fixed order."""
    hs = HAND_STUDIES
    cases = [
        _case(
            "baseline",
            hs["baseline"],
            HAND_HISTORIES["baseline"],
            HAND_INVARIANTS["baseline"],
        ),
        _case("wait", hs["wait"], HAND_HISTORIES["wait"], HAND_INVARIANTS["wait"]),
    ]
    cases += list(stop_grid())
    cases += [
        _case(
            "delayed_release",
            _delayed_release_study,
            None,
            (
                _aux_changes_only_at_release,
                lambda h: bool(np.any(h.floats["payload"][:, 0] != 0)),
            ),
        ),
        _case(
            "cancel_stale",
            lambda: kinds.survey_1(
                policies=(frozen(bundle_at=1, cancel_at=2, cancel_gen_offset=-1),)
            ),
            None,
            (
                # a Cancel carrying an old generation changes nothing
                _all_completed,
                lambda h: h.n_records == 6 and not h.double_settle,
            ),
        ),
        _case(
            "cancel_live",
            hs["cancel_live"],
            HAND_HISTORIES["cancel_live"],
            HAND_INVARIANTS["cancel_live"],
        ),
        _case(
            "bundle",
            hs["bundle"],
            HAND_HISTORIES["bundle"],
            HAND_INVARIANTS["bundle"],
        ),
        _case(
            "phase_switch",
            hs["phase_switch"],
            HAND_HISTORIES["phase_switch"],
            HAND_INVARIANTS["phase_switch"],
        ),
        _case(
            "coincident",
            hs["coincident"],
            HAND_HISTORIES["coincident"],
            HAND_INVARIANTS["coincident"],
        ),
        _case(
            "overflow_stall",
            lambda: kinds.survey_1(policies=(kinds.SpinWaitPolicy(),)),
            None,
            (lambda h: h.flags[2] and not h.flags[0],),
        ),
        _case(
            "overflow_log",
            lambda: kinds.survey_1(caps=Capacities(n_log=4)),
            None,
            (
                lambda h: h.flags[3] and h.n_records == 4,
                # the fifth completion is refused its log row, so no fifth Release
                lambda h: sum(e[2] == EventKind.RELEASE for e in h.events) == 4,
            ),
        ),
        _case(
            "coalesced_release",
            _coalesced_release_study,
            HAND_HISTORIES["coalesced_release"],
            HAND_INVARIANTS["coalesced_release"],
        ),
        _case(
            "bundle_unaffordable",
            _unaffordable_bundle_study,
            HAND_HISTORIES["bundle_unaffordable"],
            HAND_INVARIANTS["bundle_unaffordable"],
        ),
        _case(
            "forbidden_effect",
            _forbidden_effect_study,
            HAND_HISTORIES["forbidden_effect"],
            HAND_INVARIANTS["forbidden_effect"],
        ),
        _case(
            "catalog_record",
            _catalog_record_study,
            HAND_HISTORIES["catalog_record"],
            HAND_INVARIANTS["catalog_record"],
        ),
        _case(
            "host_service",
            _host_service_study,
            HAND_HISTORIES["host_service"],
            HAND_INVARIANTS["host_service"],
            forms=("array",),
            service=_host_service,
        ),
        _case(
            "revision",
            _revision_study,
            HAND_HISTORIES["revision"],
            HAND_INVARIANTS["revision"],
        ),
    ]
    return cases


CORPUS = build_corpus()
CORPUS_BY_NAME = {c.name: c for c in CORPUS}
