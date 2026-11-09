"""Handlers."""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
from jax import lax

from spaceodyssey.core.carry import StepEmit, _tree_set
from spaceodyssey.core.enums import (
    F64,
    I64,
    Audience,
    ChargePricing,
    CommitState,
    Disposition,
    EventKind,
    Purpose,
    RecordStatus,
    RemainderRule,
    Scope,
    Stage,
    WindowState,
)
from spaceodyssey.core.events import empty_event_row, enqueue, requeue, where_tree
from spaceodyssey.core.keys import asset_coordinate, draw_key, fold_key
from spaceodyssey.core.kinds import realize_row
from spaceodyssey.core.ledger import _resource_rows, charge_parts
from spaceodyssey.core.lift import lifting_jit
from spaceodyssey.core.records import (
    AWARD_COLUMNS,
    REFUND_COLUMNS,
    Record,
    derived_rows,
    masked_record,
    record_none,
    write_log,
)
from spaceodyssey.core.refresh import refresh_enabled, schedule_refresh
from spaceodyssey.core.snapshot import snapshot_fingerprint, snapshot_view


def _emit(n, slot):
    return StepEmit(n_written=jnp.asarray(n, I64), last_slot=jnp.asarray(slot, I64))


# the columns a log row always has; `program` is present only with several programs
# and the refund and award columns only while refunds or awards are on
_OPTIONAL_ROW = ("program", *REFUND_COLUMNS, *AWARD_COLUMNS)
_ROW_FIELDS = tuple(
    f for f in dataclasses.fields(Record) if f.name not in _OPTIONAL_ROW
)


def _log_rec(log, slot):
    held = {
        n: getattr(log, n)[slot] for n in _OPTIONAL_ROW if getattr(log, n) is not None
    }
    return Record(**{f.name: getattr(log, f.name)[slot] for f in _ROW_FIELDS}, **held)


def h_none(c, ev, study, truth, root):
    """Handle an inert event without writing a record."""
    return c, _emit(0, 0), record_none(study.caps)


def _release_row(c, written, rec, slot, audience, study):
    return enqueue(
        c,
        written,
        kind=EventKind.RELEASE,
        tick=rec.available_tick,
        stage=Stage.RELEASE,
        lane=rec.lane,
        channel_index=rec.channel_index,
        channel=rec.channel,
        subject=rec.subject,
        acquisition_index=rec.acquisition_index,
        audience=audience,
        slot=slot,
        acquisition_id=rec.acquisition_id,
        caps=study.caps,
    )


def _with_award_capture(base, ledger, lane):
    """`base` carrying the award identity a lane's live occupancy was debited under."""
    return dataclasses.replace(
        base,
        award_gens=ledger.lane_award_gens[lane],
        award_rules=ledger.lane_award_rules[lane],
        award_tos=ledger.lane_award_tos[lane],
    )


def _realize_rec(c, study, truth, root, base, fraction):
    """Fill a record's measurement.

    The draw key always uses the completed-status coordinate, so a partial occupancy
    shares its variate with its full twin.
    """
    coords = dataclasses.replace(base, status=jnp.asarray(RecordStatus.COMPLETED, I64))
    key = draw_key(
        root,
        coords,
        Purpose.MEASUREMENT,
        asset=asset_coordinate(study, base.channel_index),
        branch=study.fork.branch_of(base.acquisition_index),
        coupling=study.coupling,
    )
    payload = realize_row(study, truth, base, fraction, key)
    return dataclasses.replace(base, payload=payload)


def _program_field(ev):
    """The record's program column, copied from a row that holds one (else nothing)."""
    return {} if ev.program is None else {"program": ev.program}


def h_done(c, ev, study, truth, root):
    """Settle one completed occupancy exactly once: realize, log, queue its Release."""
    caps, channels = study.caps, study.channels
    channel_index = ev.channel_index
    stale = ev.gen != c.ledger.lane_gen[ev.lane]
    dup = ~stale & jnp.any(c.log.valid & (c.log.acquisition_id == ev.acquisition_id))
    dur, lat = (
        channels.occupancy_ticks[channel_index],
        channels.latency_ticks[channel_index],
    )
    base = dataclasses.replace(
        record_none(caps),
        kind=channels.kind[channel_index],
        subject=ev.subject,
        channel=ev.channel,
        channel_index=channel_index,
        lane=ev.lane,
        acquisition_index=ev.acquisition_index,
        start_tick=ev.tick - dur,
        end_tick=ev.tick,
        available_tick=ev.tick + lat,
        status=jnp.asarray(RecordStatus.COMPLETED, I64),
        phase=ev.scope,
        acquisition_id=ev.acquisition_id,
        revision=jnp.asarray(0, I64),
        **_program_field(ev),
        released_tick=jnp.asarray(-1, I64),
        accepted_tick=jnp.asarray(-1, I64),
        superseded_by=jnp.asarray(-1, I64),
        audience=channels.audience[channel_index],
        source=ev.channel,
        vintage=ev.scope,
        config_id=jnp.asarray(0, I64),
        cost=channels.cost[channel_index],
        charged=channels.cost[channel_index],
        valid=~dup & ~stale,
    )
    if caps.refunds:
        charge, fixed, _ = charge_parts(study, channel_index)
        base = dataclasses.replace(base, charged=charge, nonrefundable=fixed)
    if caps.awards:
        base = _with_award_capture(base, c.ledger, ev.lane)
    rec = eqx.filter_jit(_realize_rec)(
        c, study, truth, root, base, jnp.asarray(1.0, F64)
    )
    c1 = _tree_set(c, double_settle=c.double_settle | dup)
    c2, slot = write_log(c1, rec, caps)
    written = c2.n_log > c1.n_log
    c3 = _release_row(c2, written, rec, slot, channels.audience[channel_index], study)
    return c3, _emit(written.astype(I64), slot), rec


def h_release(c, ev, study, truth, root):
    """Make a record visible and fold it into the observer, once per acquisition.

    Each audience accepts only rows addressed to it by both the event and the record,
    and folds only the first accepted revision of an acquisition. When a
    later revision is accepted, the earlier rows are marked `superseded_by` it. A row
    held back (`RecordStatus.UNRELEASED`) is neither released nor accepted.
    """
    caps = study.caps
    slot = ev.slot
    log = c.log
    rec = _log_rec(log, slot)
    held = rec.status == RecordStatus.UNRELEASED
    audience = ev.audience & rec.audience
    readable = rec.valid & ~held
    public = ((audience & Audience.POLICY) != 0) & readable
    to_observer = ((audience & Audience.OBSERVER) != 0) & readable
    already = log.accepted_tick[slot] >= 0
    others = (
        log.valid
        & (log.acquisition_id == rec.acquisition_id)
        & (jnp.arange(caps.n_log) != slot)
    )
    # a derived row neither replaces nor is replaced by another row
    derived = jnp.isin(rec.kind, jnp.asarray(caps.derived_kinds, I64))
    if caps.derived_kinds:
        others = others & ~derived & ~derived_rows(log, caps)
    accepted = others & (log.accepted_tick >= 0)
    obs = masked_record(rec, caps, Audience.OBSERVER)
    # one key per record identity serves both audiences' folds
    key = fold_key(root, rec, study)
    # An observer that is not incremental changes only by refresh: nothing folds.
    incremental = study.observer.capabilities().incremental
    # a derived record (a product or a declared summary) is accepted and released like
    # any row but never folded: a product cannot count its own source data
    belief = c.belief
    if incremental:
        folded = study.observer.analyze(
            study.observer.fold(c.belief, obs, caps, key=key)
        )
        if caps.derived_kinds:
            folded = where_tree(~derived, folded, c.belief)
        belief = where_tree(to_observer & ~already & ~accepted.any(), folded, c.belief)
    # the policy summary folds only records released to the policy audience, once
    seen = log.released_tick[slot] >= 0
    released = others & (log.released_tick >= 0)
    # The policy fold sees supersession only from revisions it can read, just
    # like its snapshot; observer-only revisions cannot change this metadata.
    pol_rec = dataclasses.replace(
        rec,
        superseded_by=jnp.max(
            jnp.where(released & (log.revision > rec.revision), log.revision, -1)
        ),
    )
    pol = masked_record(pol_rec, caps, Audience.POLICY)
    pbelief = c.pbelief
    if incremental:
        pfolded = study.observer.fold_summary(c.pbelief, pol, caps, key=key)
        if caps.derived_kinds:
            pfolded = where_tree(~derived, pfolded, c.pbelief)
        pbelief = where_tree(public & ~seen & ~released.any(), pfolded, c.pbelief)
    older = jnp.where(accepted & (log.revision < rec.revision), rec.revision, -1)
    newer = jnp.max(jnp.where(accepted, log.revision, -1))
    newer = jnp.where(newer > rec.revision, newer, -1)
    superseded = jnp.maximum(log.superseded_by, jnp.where(to_observer, older, -1))
    superseded = superseded.at[slot].set(
        jnp.where(
            to_observer, jnp.maximum(superseded[slot], newer), log.superseded_by[slot]
        )
    )
    log = eqx.tree_at(
        lambda lg: (lg.released_tick, lg.accepted_tick, lg.superseded_by),
        log,
        (
            log.released_tick.at[slot].set(
                jnp.where(public, ev.tick, log.released_tick[slot])
            ),
            log.accepted_tick.at[slot].set(
                jnp.where(to_observer & ~already, ev.tick, log.accepted_tick[slot])
            ),
            superseded,
        ),
    )
    # a version counts a release the audience can read, as its snapshot shows it
    versions = c.release_version + jnp.stack(
        [~already & to_observer, public & ~seen]
    ).astype(I64)
    c1 = _tree_set(
        c,
        belief=belief,
        pbelief=pbelief,
        log=log,
        release_version=versions,
        woke=public & ~seen,
        dup_fold=c.dup_fold | (to_observer & already) | (public & seen),
    )
    if caps.refunds:
        c1 = _apply_refund(c1, rec, slot, public & ~seen, study)
    if refresh_enabled(caps):
        c1 = _mark_refresh(c1, rec, to_observer & ~already, public & ~seen, study)
    return c1, _emit(0, slot), record_none(caps)


def _apply_refund(c, rec, slot, first, study):
    """Credit the refundable remainder of the acquisition a failure receipt names.

    Runs where `first` holds: the release of the receipt to the policy audience, which
    is the settlement boundary (a receipt written but not yet released, held back, or
    released to the observer alone credits nothing). A record of a declared failure
    kind is a receipt; it names its origin by subject, channel (`source`) and the
    visible `refund_index`. The origin may share the receipt's kind; a record written
    on the channel it names is not a receipt, and none names its own row. The credit
    applies when the origin exists, the kind is the failure cause its channel
    declares, the release is within the deadline, and no receipt has credited the
    acquisition before. A receipt that does not apply counts in `refund_ignored`. The
    credit is written on the receipt row as a negative `charged`, so the retained
    records still net to the ledger.
    """
    caps, channels, log = study.caps, study.channels, c.log
    n = jnp.arange(caps.n_log)
    declared = channels.failure_kind is not None
    if not declared:
        return c
    # a receipt is written by something other than the channel it names, so an
    # acquisition of the failure kind (its channel is its source) is never one
    attempt = (
        first
        & rec.valid
        & jnp.isin(rec.kind, channels.failure_kind)
        & (rec.channel != rec.source)
    )
    named = channels.id == rec.source
    ti = jnp.argmax(named)
    cause = named.any() & (channels.failure_kind[ti] == rec.kind)
    cand = (
        log.valid
        & (log.subject == rec.subject)
        & (log.channel == rec.source)
        & (log.acquisition_index == rec.refund_index)
        & (n != slot)
    )
    origin = jnp.argmax(cand)
    deadline = (
        jnp.zeros((), I64)
        if channels.refund_deadline_ticks is None
        else channels.refund_deadline_ticks[ti]
    )
    in_time = c.tick <= log.end_tick[origin] + deadline
    paid = (log.valid & (log.refund_of == log.acquisition_id[origin])).any()
    applies = attempt & cand.any() & cause & in_time & ~paid
    owner = 0 if log.program is None else log.program[origin]
    rows = _resource_rows(study, log.lane[origin], owner)
    cols = jnp.arange(caps.n_resources)
    credit = jnp.where(
        applies,
        jnp.maximum(log.charged[origin] - log.nonrefundable[origin], 0.0),
        0.0,
    )
    led = c.ledger
    balance = led.balance.at[rows, cols].add(credit)
    refunded = led.refunded.at[rows, cols].add(credit)
    extra = {}
    if caps.awards:
        # A refund reaches the row directly only while nothing has opened or closed
        # on it since the debit. Otherwise it is disposed under the rule captured at
        # the debit: FORFEIT lapses it, NONE and CARRY_OVER credit the row unless a
        # newer award holds it (then it lapses), REALLOCATE credits the destination row.
        gens = log.award_gens[origin]
        rules, tos = log.award_rules[origin], log.award_tos[origin]
        same = led.win_gen[rows] == gens
        newer = led.win_state[rows] == WindowState.WINDOWED
        keeps = (rules == RemainderRule.NONE) | (rules == RemainderRule.CARRY_OVER)
        moves = ~same & (rules == RemainderRule.REALLOCATE)
        direct = same | (keeps & ~newer)
        land = jnp.where(moves, tos, rows)
        balance = led.balance.at[land, cols].add(jnp.where(direct | moves, credit, 0.0))
        refunded = led.refunded.at[land, cols].add(credit)
        lapse = ~direct & ~moves
        extra = {
            "ledger.lapsed": led.lapsed.at[rows, cols].add(
                jnp.where(lapse, credit, 0.0)
            )
        }
    return _tree_set(
        c,
        **{
            "ledger.balance": balance,
            "ledger.refunded": refunded,
            **extra,
            "ledger.refund_ignored": c.ledger.refund_ignored
            + (attempt & ~applies).astype(I64),
            "log.refund_of": log.refund_of.at[slot].set(
                jnp.where(applies, log.acquisition_id[origin], log.refund_of[slot])
            ),
            "log.charged": log.charged.at[slot].add(-credit),
        },
    )


def _mark_refresh(c, rec, accepted, released, study):
    """Mark the subject dirty for each audience that took an eligible row, and queue.

    A row is eligible when its kind is one the observer refits; its status is not
    consulted, so a correction that retracts a value also marks the subject. Later
    marks coalesce into the checkpoint already queued.
    """
    eligible = jnp.isin(rec.kind, jnp.asarray(study.caps.refit_kinds, I64))
    mark_obs, mark_pol = accepted & eligible, released & eligible
    subject = rec.subject
    dirty = c.refresh_dirty
    dirty = dirty.at[0, subject].set(dirty[0, subject] | mark_obs)
    dirty = dirty.at[1, subject].set(dirty[1, subject] | mark_pol)
    return schedule_refresh(
        _tree_set(c, refresh_dirty=dirty), mark_obs | mark_pol, study
    )


def h_result(c, ev, study, truth, root):
    """Commit a host's answer: release the record it names at its own tick.

    The row sits at the tick and stage of its request, so the record is released and
    the audience wakes within the same stage; it passes through the release rules
    unchanged.
    """
    return h_release(c, ev, study, truth, root)


def _stale_request(c, ev, study):
    """Whether the policy snapshot differs from the one the request names.

    Rebuilt at the decision that asked, as `answer_pending` rebuilds it.
    """
    view = snapshot_view(c, Audience.POLICY, study.caps)
    view = dataclasses.replace(view, n_dec=ev.decision)
    layout = study.observer.layout(study.caps)
    return snapshot_fingerprint(view, layout) != ev.snapshot_fp


def h_request(c, ev, study, truth, root):
    """Hold the run for the host: the row moves into the carry until it is answered.

    While `caps.snapshot_balances` is on, a request whose snapshot is no longer the one
    the decision read (a refund or settlement queued ahead of it moved the balance) is
    refused: it is not held, nothing is computed, the trace row records
    `AdmissionCode.QUERY_STALE`, the policy view shows the asking decision's index
    (`query_refused`), and a same-tick wait wakes the policy, which may ask again
    against the state it then reads.
    """
    held = dataclasses.replace(ev, valid=jnp.asarray(True))
    if not study.caps.snapshot_balances:
        return _tree_set(c, request=held), _emit(0, 0), record_none(study.caps)
    stale = _stale_request(c, ev, study)
    c = _tree_set(
        c,
        request=where_tree(stale, empty_event_row(study.caps), held),
        query_refused=jnp.where(stale, ev.decision, c.query_refused),
    )
    c = enqueue(
        c,
        stale,
        kind=EventKind.WAIT,
        tick=c.tick,
        stage=Stage.ADMIT,
        group=ev.group,
        audience=PUBLIC,
        caps=study.caps,
    )
    return c, _emit(0, 0), record_none(study.caps)


def h_refresh(c, ev, study, truth, root):
    """Hold the world at a checkpoint, then consume the row once it is installed.

    The first handling copies the row into `Carry.refresh` and puts it back in the queue
    unchanged, so a decision pending at this tick stays pending behind it and the world
    takes no step until the host installs (`refresh.commit_refresh`). That install marks
    the queued row (`disposition`) and its second handling is the completion: a no-op
    that reports the product rows written, after which the owed decision runs.
    """
    installed = ev.disposition == 1
    held = dataclasses.replace(ev, valid=jnp.asarray(True))
    c1 = requeue(_tree_set(c, refresh=held), ev, True)
    c2 = where_tree(installed, c, c1)
    return (
        c2,
        _emit(jnp.where(installed, ev.group, 0), jnp.where(installed, ev.slot, 0)),
        record_none(study.caps),
    )


def h_wait(c, ev, study, truth, root):
    """Mark a wait event as eligible to wake its audience."""
    return _tree_set(c, woke=jnp.asarray(True)), _emit(0, 0), record_none(study.caps)


def h_process(c, ev, study, truth, root):
    """Handle a process event and release any record it writes."""
    caps = study.caps
    c1, rec = study.process.handle(c, ev, study)
    rec = dataclasses.replace(rec, audience=jnp.asarray(ev.audience, I64))
    c2, slot = write_log(c1, rec, caps)
    written = c2.n_log > c1.n_log
    c3 = _release_row(c2, written, rec, slot, ev.audience, study)
    c4 = _tree_set(c3, woke=jnp.asarray(True))
    return c4, _emit(written.astype(I64), slot), rec


def _cancel_lane(
    c,
    study,
    truth,
    root,
    lane,
    enable,
    status,
    refund,
    program=None,
    by_program=False,
    slot=None,
):
    """Cancel a live occupancy and refund its unused cost.

    Live-cancel path: invalidate the lane's pending Done row, write the partial record
    at the elapsed fraction, refund the unused cost, free the lane. With `refund` false
    the record is charged in full and nothing comes back (a fault). With programs
    the credit goes to the row of the program that admitted the occupancy, and where
    `by_program` is set only an occupancy of `program` is cancelled. With commitments, a
    `slot` of 0 or more names a pending commitment instead: its start event is the row
    cancelled, the record is the partial one at fraction 0 and the lane is left alone.
    """
    caps, channels = study.caps, study.channels
    pend = c.ev.valid & (c.ev.kind == EventKind.DONE) & (c.ev.lane == lane)
    # a cancel of a pending commitment, where commitments are on; else always False
    named = caps.commitments and slot is not None
    reserved = slot >= 0 if named else None
    if named:
        pend = jnp.where(
            reserved,
            c.ev.valid & (c.ev.kind == EventKind.COMMIT_START) & (c.ev.slot == slot),
            pend,
        )
    if c.ev.program is not None and program is not None:
        pend = pend & (~by_program | (c.ev.program == program))
    has = enable & pend.any()
    i = jnp.argmax(pend)
    channel_index = c.ev.channel_index[i]
    dur, lat = (
        channels.occupancy_ticks[channel_index],
        channels.latency_ticks[channel_index],
    )
    start_tick = c.ev.tick[i] - dur
    if named:
        start_tick = jnp.where(reserved, c.tick, start_tick)
    frac = jnp.clip((c.tick - start_tick).astype(F64) / dur.astype(F64), 0.0, 1.0)
    spent = jnp.where(refund, frac, jnp.asarray(1.0, F64))
    base = dataclasses.replace(
        record_none(caps),
        kind=channels.kind[channel_index],
        subject=c.ev.subject[i],
        channel=c.ev.channel[i],
        channel_index=channel_index,
        lane=lane,
        acquisition_index=c.ev.acquisition_index[i],
        start_tick=start_tick,
        end_tick=c.tick,
        # a cancellation record is never released in the tick it is cancelled
        available_tick=c.tick + jnp.maximum(lat, 1),
        status=jnp.asarray(status, I64),
        phase=c.ev.scope[i],
        acquisition_id=c.ev.acquisition_id[i],
        revision=jnp.asarray(0, I64),
        released_tick=jnp.asarray(-1, I64),
        accepted_tick=jnp.asarray(-1, I64),
        superseded_by=jnp.asarray(-1, I64),
        audience=channels.audience[channel_index],
        source=c.ev.channel[i],
        vintage=c.ev.scope[i],
        config_id=jnp.asarray(0, I64),
        cost=channels.cost[channel_index] * spent,
        charged=channels.cost[channel_index] * spent,
        valid=has,
        **({} if c.ev.program is None else {"program": c.ev.program[i]}),
    )
    if caps.refunds:
        charge, fixed, pricing = charge_parts(study, channel_index)
        # per resource column: a flat price returns nothing on a cancel, and the fixed
        # part of any price stays with the record
        part = jnp.where(pricing == ChargePricing.FLAT_FRACTION, 1.0, spent)
        kept = (charge - fixed) * part + fixed
        base = dataclasses.replace(base, charged=kept, nonrefundable=fixed)
    if caps.awards:
        base = _with_award_capture(base, c.ledger, lane)
    rec = eqx.filter_jit(_realize_rec)(c, study, truth, root, base, frac)
    owner = 0 if c.ev.program is None else c.ev.program[i]
    rows = _resource_rows(study, lane, owner)
    if caps.refunds:
        credit = jnp.where(has, charge - kept, 0.0)
    else:
        credit = jnp.where(has, channels.cost[channel_index] * (1.0 - spent), 0.0)
    # the lane is freed only by cancelling what occupies it
    occupied = has & ~reserved if named else has
    c1 = _tree_set(
        c,
        **{
            "ev": eqx.tree_at(
                lambda e: e.valid,
                c.ev,
                c.ev.valid.at[i].set(jnp.where(has, False, c.ev.valid[i])),
            ),
            "ledger.balance": c.ledger.balance.at[
                rows, jnp.arange(caps.n_resources)
            ].add(credit),
            "ledger.refunded": c.ledger.refunded.at[
                rows, jnp.arange(caps.n_resources)
            ].add(credit),
            "ledger.lane_busy_until": c.ledger.lane_busy_until.at[lane].set(
                jnp.where(occupied, c.tick, c.ledger.lane_busy_until[lane])
            ),
            # a cancel retires the lane's generation: any completion still carrying
            # the old one is stale and settles nothing
            "ledger.lane_gen": c.ledger.lane_gen.at[lane].add(occupied.astype(I64)),
        },
    )
    if named:
        mark = jnp.clip(slot, 0, caps.n_commitments - 1)
        c1 = _tree_set(
            c1,
            **{
                "ledger.cm_state": c1.ledger.cm_state.at[mark].set(
                    jnp.where(
                        has & reserved, CommitState.CANCELLED, c1.ledger.cm_state[mark]
                    )
                )
            },
        )
    c2, slot = write_log(c1, rec, caps)
    written = c2.n_log > c1.n_log
    c3 = _release_row(c2, written, rec, slot, channels.audience[channel_index], study)
    return c3, written.astype(I64), slot, rec


def h_cancel(c, ev, study, truth, root):
    """A Cancel is live only if its generation is the lane's current one.

    A fault event runs this same body under its own kind (see `step`): its status and
    refund follow from the kind, so one program serves both.
    """
    live = ev.gen == c.ledger.lane_gen[ev.lane]
    fault = ev.kind == EventKind.FAULT
    status = jnp.where(fault, RecordStatus.FAILED, RecordStatus.PARTIAL).astype(I64)
    extra = {}
    if study.caps.commitments:
        # a Cancel that names a commitment acts while that commitment is pending
        named = ev.slot >= 0
        pending = c.ledger.cm_state[jnp.clip(ev.slot, 0, None)] == CommitState.PENDING
        live = jnp.where(named, pending, live)
        extra = {"slot": ev.slot}
    c1, n, slot, rec = eqx.filter_jit(_cancel_lane)(
        c, study, truth, root, ev.lane, live, status, ~fault, **extra
    )
    return c1, _emit(n, slot), rec


def h_fault(c, ev, study, truth, root):
    """A fault on a lane fails its live occupancy: charged in full, nothing refunded.

    Like a Cancel it names a generation and acts only if that is the lane's current one.
    Which of a fault and a settlement at one tick runs first is the channel's declared
    coincident order, applied by the event queue.
    """
    live = ev.gen == c.ledger.lane_gen[ev.lane]
    c1, n, slot, rec = eqx.filter_jit(_cancel_lane)(
        c,
        study,
        truth,
        root,
        ev.lane,
        live,
        jnp.asarray(RecordStatus.FAILED, I64),
        jnp.asarray(False),
    )
    return c1, _emit(n, slot), rec


def stop_reaches(study, scope, stop_lane, stop_program, lane, program):
    """Whether a Stop of `scope` at `stop_lane` reaches work on `lane` of `program`.

    The one reach rule: h_stop cancels with it, and admission asks it to tell a Cancel
    that a queued abort will already have taken the commitment.
    """
    in_lane = jnp.select(
        [scope == Scope.LANE, scope == Scope.ASSET],
        [
            lane == stop_lane,
            study.assets.lane_asset[lane] == study.assets.lane_asset[stop_lane],
        ],
        True,
    )
    if study.caps.n_programs > 1:
        return jnp.where(scope == Scope.PROGRAM, program == stop_program, in_lane)
    return in_lane


def h_stop(c, ev, study, truth, root):
    """Close, drain or abort a scope (lane, asset, program, campaign).

    Below campaign scope a drain closes exactly as a close does: the running occupancies
    finish either way. At campaign scope a drain also ends the run once they settle.
    """
    caps = study.caps
    lanes = jnp.arange(caps.n_lanes)
    in_scope = jnp.select(
        [ev.scope == Scope.LANE, ev.scope == Scope.ASSET],
        [lanes == ev.lane, study.assets.lane_asset == study.assets.lane_asset[ev.lane]],
        True,
    )
    campaign = ev.scope == Scope.CAMPAIGN
    drain = ev.disposition == Disposition.DRAIN
    abort = ev.disposition == Disposition.ABORT
    reach = in_scope  # the lanes an abort looks at
    only = {}  # the program filter of the cancel, passed only where programs exist
    by_program = None
    if caps.n_programs > 1:
        # a program Stop closes the program, not its lanes, and acts only on that
        # program's occupancies; any lane may hold one
        by_program = ev.scope == Scope.PROGRAM
        reach = jnp.where(by_program, True, in_scope)
        in_scope = jnp.where(by_program, False, in_scope)
        only = {"program": ev.program, "by_program": by_program}
    updates = {
        "ledger.lane_closed": c.ledger.lane_closed | in_scope,
        "draining": c.draining | (campaign & drain),
    }
    if by_program is not None:
        closed = c.ledger.program_closed
        updates["ledger.program_closed"] = closed.at[ev.program].set(
            closed[ev.program] | by_program
        )
    c1 = _tree_set(c, **updates)

    def cancel(lane, state):
        carry, n_total, slot, rec = state
        carry, n, s, r = eqx.filter_jit(_cancel_lane)(
            carry,
            study,
            truth,
            root,
            jnp.asarray(lane, I64),
            reach[lane] & abort,
            jnp.asarray(RecordStatus.PARTIAL, I64),
            jnp.asarray(True),
            **only,
        )
        return carry, n_total + n, jnp.where(n > 0, s, slot), where_tree(n > 0, r, rec)

    # Trace one lane body while preserving the sequential cancellation order.
    c1, n_total, slot, rec = lax.fori_loop(
        0,
        caps.n_lanes,
        cancel,
        (c1, jnp.asarray(0, I64), jnp.asarray(0, I64), record_none(caps)),
    )
    if caps.commitments:

        def cancel_pending(s, state):
            carry, n_total, slot, rec = state
            led = carry.ledger
            owner = led.cm_program[s] if by_program is not None else 0
            live = (
                abort
                & (led.cm_state[s] == CommitState.PENDING)
                & stop_reaches(
                    study, ev.scope, ev.lane, ev.program, led.cm_lane[s], owner
                )
            )
            carry, n, sl, r = eqx.filter_jit(_cancel_lane)(
                carry,
                study,
                truth,
                root,
                led.cm_lane[s],
                live,
                jnp.asarray(RecordStatus.PARTIAL, I64),
                jnp.asarray(True),
                slot=jnp.asarray(s, I64),
            )
            return (
                carry,
                n_total + n,
                jnp.where(n > 0, sl, slot),
                where_tree(n > 0, r, rec),
            )

        # a pending commitment has started nothing, so an abort cancels it as well
        c1, n_total, slot, rec = lax.fori_loop(
            0, caps.n_commitments, cancel_pending, (c1, n_total, slot, rec)
        )
    c2 = _tree_set(c1, done=c1.done | (campaign & abort))
    return c2, _emit(n_total, slot), rec


def h_window_end(c, ev, study, truth, root):
    """Close an award window: apply the row's remainder rule once per generation.

    The event names its row in `lane` and the window generation it closes in `gen`; an
    expiry whose generation is no longer the row's (a later award replaced the window)
    changes nothing. A live one bumps the generation and then applies the rule the
    award declared: NONE keeps the balance (row IDLE), FORFEIT lapses it (IDLE),
    CARRY_OVER keeps it and reopens the row (ALWAYS), REALLOCATE moves it to the
    declared row (granted there, lapsed here; IDLE).
    """
    ledger = c.ledger
    row = ev.lane
    live = (ev.gen == ledger.win_gen[row]) & (
        ledger.win_state[row] == WindowState.WINDOWED
    )
    rule = ledger.remainder_rule[row]
    left = ledger.balance[row]
    drop = live & ((rule == RemainderRule.FORFEIT) | (rule == RemainderRule.REALLOCATE))
    move = live & (rule == RemainderRule.REALLOCATE)
    to = ledger.realloc_row[row]
    carry_over = live & (rule == RemainderRule.CARRY_OVER)
    gone = jnp.where(drop, left, 0.0)
    got = jnp.where(move, left, 0.0)
    c1 = _tree_set(
        c,
        **{
            "ledger.balance": ledger.balance.at[row].add(-gone).at[to].add(got),
            "ledger.lapsed": ledger.lapsed.at[row].add(gone),
            "ledger.granted": ledger.granted.at[to].add(got),
            "ledger.win_gen": ledger.win_gen.at[row].add(live.astype(I64)),
            "ledger.win_state": ledger.win_state.at[row].set(
                jnp.where(
                    live,
                    jnp.where(carry_over, WindowState.ALWAYS, WindowState.IDLE),
                    ledger.win_state[row],
                )
            ),
        },
    )
    return c1, _emit(0, 0), record_none(study.caps)


@eqx.filter_jit
def h_commit_start(c, ev, study, truth, root):
    """Start a commitment still pending: occupy its lane and queue the settlement.

    The lane's generation moves here, so the settlement carries the generation of the
    occupancy the commitment becomes. An executing commitment is not an insertion and
    meets no freeze check; one cancelled meanwhile has no queued start. A start on a
    lane that has been closed, for a program that has, or in a draining campaign is
    refused: the commitment is cancelled as a Cancel would, the record at fraction 0 and
    the credit its pricing allows.
    """
    caps, channels = study.caps, study.channels
    mark = jnp.clip(ev.slot, 0, caps.n_commitments - 1)
    pending = (ev.slot >= 0) & (c.ledger.cm_state[mark] == CommitState.PENDING)
    shut = c.ledger.lane_closed[ev.lane] | c.draining
    if caps.n_programs > 1:
        shut = shut | c.ledger.program_closed[ev.program]
    refused = pending & shut
    ok = pending & ~shut
    # the pop freed the start's row; put it back so the cancel finds it
    c = requeue(c, ev, refused)
    c, n, slot, rec = eqx.filter_jit(_cancel_lane)(
        c,
        study,
        truth,
        root,
        ev.lane,
        refused,
        jnp.asarray(RecordStatus.PARTIAL, I64),
        jnp.asarray(True),
        slot=ev.slot,
    )
    led = c.ledger
    done_t = (
        ev.tick
        + study.assets.transition_ticks[ev.lane]
        + channels.occupancy_ticks[ev.channel_index]
    )
    gen = led.lane_gen[ev.lane] + ok.astype(I64)
    c1 = _tree_set(
        c,
        **{
            "ledger.cm_state": led.cm_state.at[mark].set(
                jnp.where(ok, CommitState.STARTED, led.cm_state[mark])
            ),
            "ledger.lane_busy_until": led.lane_busy_until.at[ev.lane].set(
                jnp.where(ok, done_t, led.lane_busy_until[ev.lane])
            ),
            "ledger.lane_gen": led.lane_gen.at[ev.lane].set(gen),
        },
    )
    c2 = enqueue(
        c1,
        ok,
        kind=EventKind.DONE,
        tick=done_t,
        stage=Stage.SETTLE,
        lane=ev.lane,
        channel_index=ev.channel_index,
        channel=ev.channel,
        subject=ev.subject,
        acquisition_index=ev.acquisition_index,
        gen=gen,
        group=ev.group,
        audience=ev.audience,
        scope=ev.scope,
        acquisition_id=ev.acquisition_id,
        program=0 if ev.program is None else ev.program,
        caps=caps,
    )
    return c2, _emit(n, slot), rec


@lifting_jit
def step(carry, ev, study, truth, root):
    """The transition for one event: lax.switch over `HANDLERS` by event kind.

    The window handler is part of the table only while awards are on and the commitment
    handler only while commitments are on, so a study without them lowers the table it
    always had. With commitments the window kind is an inert row.
    """
    caps = study.caps
    table = HANDLERS[:11]
    if caps.awards or caps.commitments:
        table = (*table, h_window_end if caps.awards else h_none)
    if caps.commitments:
        table = (*table, h_commit_start)
    # A handler that is another's body under a different kind is traced once: the
    # switch holds the distinct bodies and a table maps each kind to its body.
    bodies = [_SAME_BODY.get(h, h) for h in table]
    distinct = list(dict.fromkeys(bodies))
    branches = tuple((lambda c, h=h: h(c, ev, study, truth, root)) for h in distinct)
    kind_to_branch = jnp.asarray([distinct.index(h) for h in bodies], jnp.int32)
    index = kind_to_branch[jnp.clip(ev.kind, 0, len(table) - 1).astype(jnp.int32)]
    return lax.switch(index, branches, carry)


HANDLERS = (
    h_none,
    h_done,
    h_release,
    h_wait,
    h_stop,
    h_process,
    h_cancel,
    h_result,
    h_fault,
    h_request,
    h_refresh,
    h_window_end,
    h_commit_start,
)
PUBLIC = Audience.OBSERVER | Audience.POLICY

# kinds whose handler runs another handler's body unchanged: a RESULT is a RELEASE, and
# a FAULT is a CANCEL whose status and refund follow from the event kind
_SAME_BODY = {h_result: h_release, h_fault: h_cancel}
