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
    Disposition,
    EventKind,
    Purpose,
    RecordStatus,
    Scope,
    Stage,
)
from spaceodyssey.core.events import enqueue, requeue, where_tree
from spaceodyssey.core.keys import asset_coordinate, draw_key, fold_key
from spaceodyssey.core.ledger import _resource_rows
from spaceodyssey.core.records import (
    Record,
    derived_rows,
    masked_record,
    record_none,
    write_log,
)
from spaceodyssey.core.refresh import refresh_enabled, schedule_refresh


def _emit(n, slot):
    return StepEmit(n_written=jnp.asarray(n, I64), last_slot=jnp.asarray(slot, I64))


# the columns a log row always has; `program` is present only with several programs
_ROW_FIELDS = tuple(f for f in dataclasses.fields(Record) if f.name != "program")


def _log_rec(log, slot):
    program = None if log.program is None else log.program[slot]
    return Record(
        **{f.name: getattr(log, f.name)[slot] for f in _ROW_FIELDS}, program=program
    )


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
    meas, z = study.channel_model.realize(
        truth,
        base.subject,
        base.channel_index,
        fraction,
        key,
        study.design,
        study.channels,
    )
    payload = jnp.zeros(study.caps.payload_width, F64).at[0].set(meas).at[1].set(z)
    return dataclasses.replace(base, payload=payload.at[2].set(fraction))


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
    if refresh_enabled(caps):
        c1 = _mark_refresh(c1, rec, to_observer & ~already, public & ~seen, study)
    return c1, _emit(0, slot), record_none(caps)


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


def h_request(c, ev, study, truth, root):
    """Hold the run for the host: the row moves into the carry until it is answered."""
    held = dataclasses.replace(ev, valid=jnp.asarray(True))
    return _tree_set(c, request=held), _emit(0, 0), record_none(study.caps)


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
    status=RecordStatus.PARTIAL,
    refund=True,
    program=None,
    by_program=False,
):
    """Cancel a live occupancy and refund its unused cost.

    Live-cancel path: invalidate the lane's pending Done row, write the partial record
    at the elapsed fraction, refund the unused cost, free the lane. With `refund` false
    the record is charged in full and nothing comes back (a fault). With programs
    the credit goes to the row of the program that admitted the occupancy, and where
    `by_program` is set only an occupancy of `program` is cancelled.
    """
    caps, channels = study.caps, study.channels
    pend = c.ev.valid & (c.ev.kind == EventKind.DONE) & (c.ev.lane == lane)
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
    frac = jnp.clip((c.tick - start_tick).astype(F64) / dur.astype(F64), 0.0, 1.0)
    spent = frac if refund else jnp.asarray(1.0, F64)
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
    rec = eqx.filter_jit(_realize_rec)(c, study, truth, root, base, frac)
    owner = 0 if c.ev.program is None else c.ev.program[i]
    rows = _resource_rows(study, lane, owner)
    credit = jnp.where(has, channels.cost[channel_index] * (1.0 - spent), 0.0)
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
                jnp.where(has, c.tick, c.ledger.lane_busy_until[lane])
            ),
            # a cancel retires the lane's generation: any completion still carrying
            # the old one is stale and settles nothing
            "ledger.lane_gen": c.ledger.lane_gen.at[lane].add(has.astype(I64)),
        },
    )
    c2, slot = write_log(c1, rec, caps)
    written = c2.n_log > c1.n_log
    c3 = _release_row(c2, written, rec, slot, channels.audience[channel_index], study)
    return c3, written.astype(I64), slot, rec


def h_cancel(c, ev, study, truth, root):
    """A Cancel is live only if its generation is the lane's current one."""
    live = ev.gen == c.ledger.lane_gen[ev.lane]
    c1, n, slot, rec = eqx.filter_jit(_cancel_lane)(
        c, study, truth, root, ev.lane, live
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
        c, study, truth, root, ev.lane, live, RecordStatus.FAILED, False
    )
    return c1, _emit(n, slot), rec


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
    c2 = _tree_set(c1, done=c1.done | (campaign & abort))
    return c2, _emit(n_total, slot), rec


@eqx.filter_jit
def step(carry, ev, study, truth, root):
    """The transition for one event: lax.switch over `HANDLERS` by event kind."""
    table = HANDLERS
    branches = tuple((lambda c, h=h: h(c, ev, study, truth, root)) for h in table)
    return lax.switch(
        jnp.clip(ev.kind, 0, len(table) - 1).astype(jnp.int32), branches, carry
    )


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
)
PUBLIC = Audience.OBSERVER | Audience.POLICY
