"""Unit tests of the transaction core.

Unit tests of the transaction core: handlers, admission, cancel, stop, abort, phases.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from spaceodyssey import core
from spaceodyssey import core as sch
from spaceodyssey.core.enums import (
    AdmissionCode,
    Audience,
    Disposition,
    EffectKind,
    EventKind,
    RecordStatus,
    Scope,
    Stage,
)
from spaceodyssey.core.executors.base import index_world
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.studies import toy as kinds
from tests.core.corpus import HAND_HISTORIES, HAND_STUDIES, frozen
from tests.core.helpers import (
    CAPS,
    admit_j,
    advance_j,
    after_decision_1_bundle,
    continue_until,
    effects_of,
    hand_run,
    handle_event,
    log_row,
    observe_effect,
    one_world,
    pending_done_row,
    private_channel_study,
    run_until,
    run_until_quiet,
    transition_j,
    worlds,
)

STUDY = kinds.survey_1()
TRUTH, ROOT = one_world()


def carry0(study=STUDY, truth=TRUTH):
    return eqx.filter_jit(core.init_carry)(study, truth)


@pytest.fixture(scope="module")
def bundle_state():
    """The shared immutable starting point for the cancel and stop checks."""
    return after_decision_1_bundle()


def ev_of_kind(kind):
    c = carry0()
    ev = jax.tree_util.tree_map(lambda a: a[0], c.ev)
    return dataclasses.replace(ev, kind=jnp.asarray(kind, jnp.int64))


@pytest.mark.parametrize("kind", range(8))
def test_every_handler_has_uniform_output_structure(kind):
    args = (carry0(), ev_of_kind(kind), STUDY, TRUTH, ROOT)
    out = eqx.filter_eval_shape(core.HANDLERS[kind], *args)
    ref = eqx.filter_eval_shape(core.HANDLERS[0], *args)
    assert jax.tree_util.tree_structure(out) == jax.tree_util.tree_structure(ref)
    for a, b in zip(
        jax.tree_util.tree_leaves(out), jax.tree_util.tree_leaves(ref), strict=False
    ):
        assert a.shape == b.shape and a.dtype == b.dtype


def test_init_carry_has_the_public_kickoff_wake():
    c = carry0()
    assert int(c.ev.valid.sum()) == 1 and int(c.next_sid) == 1
    row = jax.tree_util.tree_map(lambda a: a[int(jnp.argmax(c.ev.valid))], c.ev)
    assert (int(row.tick), int(row.stage), int(row.kind)) == (
        0,
        Stage.ADMIT,
        EventKind.WAIT,
    )
    assert (int(row.audience) & Audience.POLICY) != 0


def test_settle_exactly_once():
    c = carry0()
    c = eqx.tree_at(lambda x: x.ev.valid, c, jnp.zeros_like(c.ev.valid))
    for _ in range(2):
        c = sch.enqueue(
            c,
            True,
            kind=EventKind.DONE,
            tick=10,
            stage=Stage.SETTLE,
            lane=0,
            channel_index=0,
            channel=10,
            subject=0,
            acquisition_index=0,
            acquisition_id=7,
            caps=CAPS,
        )
    c, _, _ = advance_j(c, ROOT, STUDY, TRUTH)
    c, _, _ = advance_j(c, ROOT, STUDY, TRUTH)
    assert int(c.n_log) == 1 and bool(c.double_settle)


def test_unsupported_effect_refused_without_change():
    c0 = carry0()
    c2, code = admit_j(c0, effects_of(EffectKind.ALLOCATE), STUDY)
    r = sch.tree_equal_report(c0, c2)
    assert int(code[0]) == AdmissionCode.UNSUPPORTED
    assert r["max_abs"] == 0 and r["max_int_diff"] == 0


def test_no_funds_and_lane_busy_codes():
    zero = kinds.survey_1(budget=jnp.zeros_like(STUDY.budget))
    c = eqx.filter_jit(core.init_carry)(zero, TRUTH)
    assert int(admit_j(c, observe_effect(0, 10), zero)[1][0]) == AdmissionCode.NO_FUNDS
    busy = eqx.tree_at(
        lambda x: x.ledger.lane_busy_until, carry0(), jnp.asarray([50, 0, 0], jnp.int64)
    )
    assert (
        int(admit_j(busy, observe_effect(0, 10), STUDY)[1][0])
        == AdmissionCode.LANE_BUSY
    )


def test_admission_is_cumulative_across_groups_of_a_decision():
    # two occupancies of one decision on the same lane, in different groups: the
    # second finds the lane busy after the first
    e = observe_effect(0, 10)
    e = dataclasses.replace(
        e,
        kind=e.kind.at[1].set(1),
        subject=e.subject.at[1].set(1),
        channel_index=e.channel_index.at[1].set(0),
        group=e.group.at[1].set(1),
    )
    c, codes = admit_j(carry0(), e, STUDY)
    assert [int(x) for x in codes] == [0, AdmissionCode.LANE_BUSY]
    assert int(c.acquisition_counts[0]) == 1 and int(c.acquisition_counts[1]) == 0


def test_gain_enters_measurement():
    def realize_once(gain):
        study = kinds.survey_1(design=kinds.ToyDesign(gain=gain))
        c = run_until(study, lambda c: c.n_log >= 1)
        return float(c.log.payload[0, 0]), float(c.log.payload[0, 1])

    a, b = realize_once(1.0), realize_once(2.0)
    assert a[1] == b[1] and b[0] != a[0]


# ---- cancel, stop, abort, phases --------------------------------------


def test_stale_cancel_no_effect(bundle_state):
    study, c = bundle_state
    c2 = handle_event(
        c,
        study,
        TRUTH,
        ROOT,
        EventKind.CANCEL,
        lane=0,
        gen=int(c.ledger.lane_gen[0]) - 1,
    )
    assert int(c2.n_log) == int(c.n_log) and jnp.array_equal(c2.ev.valid, c.ev.valid)
    assert sch.tree_equal_report(c.ledger, c2.ledger)["max_abs"] == 0


def test_live_cancel_partial_fraction(bundle_state):
    study, c = bundle_state
    c2 = handle_event(
        c, study, TRUTH, ROOT, EventKind.CANCEL, lane=0, gen=int(c.ledger.lane_gen[0])
    )
    rec = log_row(c2, int(c2.n_log) - 1)
    assert int(rec.status) == RecordStatus.PARTIAL and float(
        rec.payload[2]
    ) == pytest.approx(0.45)
    assert int(rec.subject) == 1 and int(rec.acquisition_index) == 0
    assert not bool(pending_done_row(c2, lane=0).valid)
    assert int(c2.ledger.lane_busy_until[0]) <= 185
    # the unused 55 percent of the reservation comes back
    assert float(c2.ledger.balance[0, 0]) == pytest.approx(
        float(c.ledger.balance[0, 0]) + 0.55
    )
    # the partial record gets its own Release after the channel latency
    released_tick = [
        i
        for i in range(CAPS.n_events)
        if bool(c2.ev.valid[i]) and int(c2.ev.kind[i]) == EventKind.RELEASE
    ]
    assert any(int(c2.ev.tick[i]) == 205 for i in released_tick)


def test_abort_writes_partial_once(bundle_state):
    study, c = bundle_state
    c2 = handle_event(
        c,
        study,
        TRUTH,
        ROOT,
        EventKind.STOP,
        lane=0,
        scope=Scope.CAMPAIGN,
        disposition=Disposition.ABORT,
    )
    assert int(c2.n_log) == int(c.n_log) + 1 and bool(c2.done)
    c3, _, _ = advance_j(c2, ROOT, study, TRUTH)
    assert int(c3.n_log) == int(c2.n_log)


@pytest.mark.parametrize(
    "scope", [Scope.LANE, Scope.ASSET, Scope.PROGRAM, Scope.CAMPAIGN]
)
def test_close_refuses_new_admissions(scope, bundle_state):
    study, c = bundle_state
    c = handle_event(
        c,
        study,
        TRUTH,
        ROOT,
        EventKind.STOP,
        lane=0,
        scope=scope,
        disposition=Disposition.CLOSE,
    )
    assert (
        int(admit_j(c, observe_effect(0, 10), study)[1][0]) == AdmissionCode.LANE_CLOSED
    )


def test_close_scope_membership(bundle_state):
    study, c = bundle_state
    lane_only = handle_event(
        c,
        study,
        TRUTH,
        ROOT,
        EventKind.STOP,
        lane=0,
        scope=Scope.LANE,
        disposition=Disposition.CLOSE,
    )
    assert [bool(x) for x in lane_only.ledger.lane_closed] == [True, False, False]
    asset = handle_event(
        c,
        study,
        TRUTH,
        ROOT,
        EventKind.STOP,
        lane=0,
        scope=Scope.ASSET,
        disposition=Disposition.CLOSE,
    )
    assert [bool(x) for x in asset.ledger.lane_closed] == [True, True, False]
    program = handle_event(
        c,
        study,
        TRUTH,
        ROOT,
        EventKind.STOP,
        lane=0,
        scope=Scope.PROGRAM,
        disposition=Disposition.CLOSE,
    )
    assert all(bool(x) for x in program.ledger.lane_closed)


def test_abort_by_scope_cancels_only_lanes_in_scope(bundle_state):
    study, c = bundle_state
    lane2 = handle_event(
        c,
        study,
        TRUTH,
        ROOT,
        EventKind.STOP,
        lane=2,
        scope=Scope.LANE,
        disposition=Disposition.ABORT,
    )
    assert int(lane2.n_log) == int(c.n_log) and not bool(lane2.done)
    lane0 = handle_event(
        c,
        study,
        TRUTH,
        ROOT,
        EventKind.STOP,
        lane=0,
        scope=Scope.LANE,
        disposition=Disposition.ABORT,
    )
    assert int(lane0.n_log) == int(c.n_log) + 1 and not bool(lane0.done)


def test_drain_sets_done_when_scope_empty(bundle_state):
    study, c = bundle_state
    c = handle_event(
        c,
        study,
        TRUTH,
        ROOT,
        EventKind.STOP,
        lane=0,
        scope=Scope.CAMPAIGN,
        disposition=Disposition.DRAIN,
    )
    c = run_until_quiet(c, study, TRUTH, ROOT)
    assert bool(c.done) and not bool(c.idle)
    assert int(c.n_log) == 3  # the running occupancy completed and settled


def test_drain_does_not_wake_the_policy(bundle_state):
    study, c = bundle_state
    c = handle_event(
        c,
        study,
        TRUTH,
        ROOT,
        EventKind.STOP,
        lane=0,
        scope=Scope.CAMPAIGN,
        disposition=Disposition.DRAIN,
    )
    n = int(c.n_dec)
    c = run_until_quiet(c, study, TRUTH, ROOT)
    assert int(c.n_dec) == n


def test_every_hand_study_reproduces_its_history():
    for name in HAND_STUDIES:
        if name not in HAND_HISTORIES:
            continue
        assert hand_run(name).history(0).events == HAND_HISTORIES[name], name


# ---- firewall, zero latency, stale completions -------------------


def test_done_stops_all_writes(bundle_state):
    study, c = bundle_state
    c = eqx.tree_at(lambda x: x.done, c, jnp.asarray(True))
    c2, _ = transition_j(c, jnp.asarray(0), ROOT, study, TRUTH)
    r = sch.tree_equal_report(c, c2)
    assert r["max_abs"] == 0 and r["max_int_diff"] == 0 and r["structure_equal"]


def test_observer_only_release_leaves_policy_view_unchanged():
    study = private_channel_study(plan=((0, 12), (1, 10)))
    c = run_until(study, lambda c: c.belief.n.sum() >= 1)
    assert int(c.belief.n[0]) == 1  # the observer folded the private record
    view = sch.snapshot_view(c, Audience.POLICY, CAPS)
    ref = sch.snapshot_view(
        eqx.filter_jit(core.init_carry)(study, TRUTH), Audience.POLICY, CAPS
    )
    assert int(view.n_rows) == 0
    assert sch.tree_equal_report(view.belief, ref.belief)["max_int_diff"] == 0
    assert sch.tree_equal_report(view.belief, ref.belief)["max_abs"] == 0
    assert int(c.pbelief.n.sum()) == 0 and int(c.belief.n.sum()) == 1


def test_public_release_updates_policy_belief_once():
    c = run_until(STUDY, lambda c: c.pbelief.n.sum() >= 1)
    assert int(c.pbelief.n[0]) == 1
    assert sch.tree_equal_report(c.belief, c.pbelief)["max_abs"] == 0


def test_zero_latency_cancel_does_not_violate():
    study = kinds.survey_1(policies=(frozen(plan=((0, 12), (1, 10))),))
    c = run_until(study, lambda c: c.n_dec >= 1)
    c = eqx.tree_at(
        lambda x: (x.tick, x.stage),
        c,
        (jnp.asarray(30, c.tick.dtype), jnp.asarray(Stage.ADMIT, c.stage.dtype)),
    )
    c2 = handle_event(
        c, study, TRUTH, ROOT, EventKind.CANCEL, lane=2, gen=int(c.ledger.lane_gen[2])
    )
    assert not bool(c2.violation) and int(c2.n_log) == 1
    row = log_row(c2, 0)
    assert int(row.status) == RecordStatus.PARTIAL and float(
        row.payload[2]
    ) == pytest.approx(0.5)
    assert int(row.available_tick) == 31  # never released in the tick it is cancelled
    assert int(row.available_tick) > int(c.tick)


def test_zero_latency_abort_does_not_violate():
    study = kinds.survey_1(policies=(frozen(plan=((0, 12), (1, 10))),))
    c = run_until(study, lambda c: c.n_dec >= 1)
    c = eqx.tree_at(
        lambda x: (x.tick, x.stage),
        c,
        (jnp.asarray(30, c.tick.dtype), jnp.asarray(Stage.ADMIT, c.stage.dtype)),
    )
    c2 = handle_event(
        c,
        study,
        TRUTH,
        ROOT,
        EventKind.STOP,
        lane=0,
        scope=Scope.CAMPAIGN,
        disposition=Disposition.ABORT,
    )
    assert not bool(c2.violation) and bool(c2.done)


def test_cancel_bumps_lane_generation(bundle_state):
    study, c = bundle_state
    g = int(c.ledger.lane_gen[0])
    c2 = handle_event(c, study, TRUTH, ROOT, EventKind.CANCEL, lane=0, gen=g)
    assert int(c2.ledger.lane_gen[0]) == g + 1
    stale = handle_event(c2, study, TRUTH, ROOT, EventKind.CANCEL, lane=0, gen=g)
    assert int(stale.n_log) == int(c2.n_log)  # the same cancel again is now stale


def _done_row(c, **kw):
    fields = dict(
        lane=0,
        channel_index=0,
        channel=10,
        subject=2,
        acquisition_index=0,
        acquisition_id=999,
        scope=0,
    )
    fields.update(kw)
    return fields


def test_stale_completion_settles_nothing(bundle_state):
    study, c = bundle_state
    g = int(c.ledger.lane_gen[0])
    n_rel = int(jnp.sum(c.ev.valid & (c.ev.kind == EventKind.RELEASE)))
    c2 = handle_event(c, study, TRUTH, ROOT, EventKind.DONE, gen=g - 1, **_done_row(c))
    assert int(c2.n_log) == int(c.n_log) and not bool(c2.double_settle)
    assert int(jnp.sum(c2.ev.valid & (c2.ev.kind == EventKind.RELEASE))) == n_rel
    # control: the same completion at the current generation settles
    c3 = handle_event(c, study, TRUTH, ROOT, EventKind.DONE, gen=g, **_done_row(c))
    assert int(c3.n_log) == int(c.n_log) + 1
    assert int(jnp.sum(c3.ev.valid & (c3.ev.kind == EventKind.RELEASE))) == n_rel + 1


def test_cancelled_occupancy_never_completes(bundle_state):
    study, c = bundle_state
    c2 = handle_event(
        c, study, TRUTH, ROOT, EventKind.CANCEL, lane=0, gen=int(c.ledger.lane_gen[0])
    )
    c3 = run_until_quiet(c2, study, TRUTH, ROOT)
    keys = [
        (
            int(c3.log.subject[i]),
            int(c3.log.acquisition_index[i]),
            int(c3.log.status[i]),
        )
        for i in range(int(c3.n_log))
    ]
    assert keys.count((1, 0, RecordStatus.PARTIAL)) == 1
    assert (1, 0, RecordStatus.COMPLETED) not in keys and not bool(c3.double_settle)


# ---- partial records, belief and draws ------------------------------------------


def _release_at(c, tick):
    return (c.tick == tick) & (c.stage == Stage.RELEASE)


def test_partial_draw_key_ignores_status(bundle_state):
    study, c = bundle_state
    full = run_until(
        kinds.survey_1(policies=(frozen(bundle_at=1),)), lambda x: x.n_log >= 3
    )
    c2 = handle_event(
        c, study, TRUTH, ROOT, EventKind.CANCEL, lane=0, gen=int(c.ledger.lane_gen[0])
    )
    partial = log_row(c2, int(c2.n_log) - 1)
    assert int(partial.status) == RecordStatus.PARTIAL and float(
        partial.payload[2]
    ) == pytest.approx(0.45)
    twin = [
        i
        for i in range(int(full.n_log))
        if int(full.log.subject[i]) == 1 and int(full.log.acquisition_index[i]) == 0
    ]
    other = [i for i in range(int(full.n_log)) if int(full.log.subject[i]) == 0]
    assert float(full.log.payload[twin[0], 1]) == float(partial.payload[1])
    assert float(full.log.payload[other[0], 1]) != float(partial.payload[1])  # control


def test_phase_switch_changes_policy_at_boundary():
    run = hand_run("phase_switch")
    for w in range(2):
        h = run.history(w)
        assert h.ints["phase"].tolist() == [0, 0, 0, 1, 1]
        assert h.ints["vintage"].tolist() == [0, 0, 0, 1, 1]
        # phase B restarts its own plan: first subject 0 on channel 12 at 445
        assert (h.ints["subject"][3], h.ints["channel"][3], h.ints["end_tick"][3]) == (
            0,
            12,
            445,
        )
        assert h.ints["acquisition_index"][3] == 2
    assert [int(p) for p in run.carry.phase] == [1, 1]


def test_cancel_partial_joins_observer_not_belief_mean(bundle_state):
    study, c = bundle_state
    c = handle_event(
        c, study, TRUTH, ROOT, EventKind.CANCEL, lane=0, gen=int(c.ledger.lane_gen[0])
    )
    slot = int(c.n_log) - 1
    n_before, mean_before = c.belief.n, c.belief.mean
    assert int(c.log.accepted_tick[slot]) == -1
    c2 = continue_until(c, study, TRUTH, ROOT, lambda x: x.log.accepted_tick[slot] >= 0)
    assert (
        int(c2.log.status[slot]) == RecordStatus.PARTIAL
        and int(c2.log.released_tick[slot]) == 205
    )
    assert jnp.array_equal(c2.belief.n, n_before) and jnp.array_equal(
        c2.belief.mean, mean_before
    )


def test_abort_partial_is_never_released_or_folded(bundle_state):
    study, c = bundle_state
    n_before, p_before = c.belief.n, c.pbelief.n
    c2 = handle_event(
        c,
        study,
        TRUTH,
        ROOT,
        EventKind.STOP,
        lane=0,
        scope=Scope.CAMPAIGN,
        disposition=Disposition.ABORT,
    )
    slot = int(c2.n_log) - 1
    assert int(c2.log.status[slot]) == RecordStatus.PARTIAL and bool(c2.done)
    c3 = run_until_quiet(c2, study, TRUTH, ROOT, max_steps=8)
    assert (
        int(c3.log.released_tick[slot]) == -1 and int(c3.log.accepted_tick[slot]) == -1
    )
    assert jnp.array_equal(c3.belief.n, n_before) and jnp.array_equal(
        c3.pbelief.n, p_before
    )
    assert int(c3.n_log) == int(c2.n_log)


# The distinct adaptive-policy program and six-world drain check run nightly.
@pytest.mark.slow
def test_threshold_stop_policy_drains_after_threshold():
    study = kinds.survey_1(policies=(kinds.ThresholdStopPolicy(stop_snr=3.0),))
    cs, ts = CompiledExecutor().run(study, *worlds(6), 80)
    for w in range(6):
        c, t = index_world(cs, w), index_world(ts, w)
        assert bool(c.draining) and bool(c.done) and not bool(c.idle)
        handled = [i for i in range(len(t.kind)) if bool(t.handled[i])]
        assert int(t.kind[handled[-1]]) == EventKind.STOP
        assert int(t.kind[handled[-2]]) == EventKind.RELEASE
        snr = c.pbelief.mean * jnp.sqrt(c.pbelief.n.astype(jnp.float64)) / 0.5
        assert float(jnp.max(snr)) > 3.0
