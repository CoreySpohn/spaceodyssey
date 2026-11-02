"""Studies of the conformance corpus.

Study builders and hand histories for transaction conformance.

Hand histories are tuples of (tick, stage, kind) in handled order. Every history starts
with the kick-off wake (0, Stage.ADMIT, EventKind.WAIT), where the policy decides first.
"""

import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import (
    AdmissionCode,
    Audience,
    Disposition,
    EventKind,
    RecordStatus,
    Scope,
    Stage,
)
from spaceodyssey.core.study import Study
from spaceodyssey.studies import toy as kinds
from spaceodyssey.studies.toy import RECORD_AUXILIARY

S, R, A = Stage.SETTLE, Stage.RELEASE, Stage.ADMIT
DONE, REL, WAIT, STOP, CANCEL = (
    EventKind.DONE,
    EventKind.RELEASE,
    EventKind.WAIT,
    EventKind.STOP,
    EventKind.CANCEL,
)

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
}


# ---- the conformance corpus ---------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CorpusCase:
    """One study of the corpus.

    `make_study` builds it, `hand_history` pins its handled rows when given,
    `invariants` are checks on the canonical history.
    """

    name: str
    make_study: Callable[[], Study]
    n_seeds: int
    hand_history: tuple | None
    invariants: tuple
    policy_forms: tuple = ("array", "python")


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


def _case(name, make, hand=None, inv=(), forms=("array", "python"), n_seeds=64):
    return CorpusCase(name, make, n_seeds, hand, tuple(inv), forms)


def build_corpus():
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
    ]
    return cases


CORPUS = build_corpus()
CORPUS_BY_NAME = {c.name: c for c in CORPUS}


# ---- leak scenarios ---------------------------------------------------------------

LEAK_PLAN = PLAN5


def leak_study(process):
    """survey_1 under a policy that ignores wakes with nothing new in its view."""
    return kinds.survey_1(
        policies=(kinds.ReactivePolicy(plan=LEAK_PLAN),), process=process
    )


LEAK_SCENARIOS = {
    "heartbeat": kinds.HeartbeatProcess(
        period_ticks=37, n_fires=10, audience=Audience.OBSERVER, writes=False
    ),
    "private_writer": kinds.HeartbeatProcess(
        period_ticks=37, n_fires=10, audience=Audience.OBSERVER, writes=True
    ),
    "releasing": kinds.HeartbeatProcess(
        period_ticks=130, n_fires=1, audience=Audience.POLICY, writes=False
    ),
}


# ---- draw-invariance perturbations --------------------------------------------------
# name -> (baseline factory, perturbed factory). A perturbation changes how the study is
# built or when things happen, never which occupancy a draw belongs to.


def _prepend_kind_study():
    """Renumber the measurement and auxiliary record kinds to 5 and 9."""
    base = kinds.survey_1(
        observer=kinds.ToyRunningMean(measurement_kind=5, auxiliary_kind=9)
    )
    kind = jnp.full_like(base.channels.kind, 5)
    return eqx.tree_at(lambda st: st.channels.kind, base, kind)


PERTURBATIONS = {
    "insert_wait": (HAND_STUDIES["baseline"], HAND_STUDIES["wait"]),
    "prepend_kind": (HAND_STUDIES["baseline"], _prepend_kind_study),
    # the cancelled occupancy has to be live at a decision, which takes a bundle, so the
    # baseline is the bundle study (the same keys, none cancelled)
    "cancel_reobserve": (
        lambda: kinds.survey_1(policies=(frozen(bundle_at=1),)),
        HAND_STUDIES["cancel_live"],
    ),
    "extra_kinds": (HAND_STUDIES["baseline"], lambda: kinds.survey_1()),
    "payload_width": (
        HAND_STUDIES["baseline"],
        lambda: kinds.survey_1(caps=Capacities(payload_width=6)),
    ),
}
