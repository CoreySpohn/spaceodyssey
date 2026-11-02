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
from spaceodyssey.core.events import enqueue, where_tree
from spaceodyssey.core.keys import draw_key
from spaceodyssey.core.ledger import _resource_rows
from spaceodyssey.core.records import Record, record_none, write_log


def _emit(n, slot):
    return StepEmit(n_written=jnp.asarray(n, I64), last_slot=jnp.asarray(slot, I64))


def _log_rec(log, slot):
    return Record(
        **{f.name: getattr(log, f.name)[slot] for f in dataclasses.fields(Record)}
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
    key = draw_key(root, coords, Purpose.MEASUREMENT)
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
        released_tick=jnp.asarray(-1, I64),
        accepted_tick=jnp.asarray(-1, I64),
        superseded_by=jnp.asarray(-1, I64),
        source=ev.channel,
        vintage=ev.scope,
        config_id=jnp.asarray(0, I64),
        cost=channels.cost[channel_index],
        charged=channels.cost[channel_index],
        valid=~dup & ~stale,
    )
    rec = _realize_rec(c, study, truth, root, base, jnp.asarray(1.0, F64))
    c1 = _tree_set(c, double_settle=c.double_settle | dup)
    c2, slot = write_log(c1, rec, caps)
    written = c2.n_log > c1.n_log
    c3 = _release_row(c2, written, rec, slot, channels.audience[channel_index], study)
    return c3, _emit(written.astype(I64), slot), rec


def h_release(c, ev, study, truth, root):
    """Make a record visible and fold it into the observer, once."""
    caps = study.caps
    slot = ev.slot
    rec = _log_rec(c.log, slot)
    public = (ev.audience & Audience.POLICY) != 0
    already = c.log.accepted_tick[slot] >= 0
    folded = study.observer.analyze(study.observer.fold(c.belief, rec, caps))
    belief = where_tree(~already, folded, c.belief)
    # the policy's belief folds only records released to the policy audience, once
    seen = c.log.released_tick[slot] >= 0
    pfolded = study.observer.analyze(study.observer.fold(c.pbelief, rec, caps))
    pbelief = where_tree(public & ~seen, pfolded, c.pbelief)
    log = eqx.tree_at(
        lambda lg: (lg.released_tick, lg.accepted_tick),
        c.log,
        (
            c.log.released_tick.at[slot].set(
                jnp.where(public, ev.tick, c.log.released_tick[slot])
            ),
            c.log.accepted_tick.at[slot].set(
                jnp.where(already, c.log.accepted_tick[slot], ev.tick)
            ),
        ),
    )
    c1 = _tree_set(
        c,
        belief=belief,
        pbelief=pbelief,
        log=log,
        woke=jnp.asarray(True),
        dup_fold=c.dup_fold | already,
    )
    return c1, _emit(0, slot), record_none(caps)


def h_wait(c, ev, study, truth, root):
    """Mark a wait event as eligible to wake its audience."""
    return _tree_set(c, woke=jnp.asarray(True)), _emit(0, 0), record_none(study.caps)


def h_process(c, ev, study, truth, root):
    """Handle a process event and release any record it writes."""
    caps = study.caps
    c1, rec = study.process.handle(c, ev, study)
    c2, slot = write_log(c1, rec, caps)
    written = c2.n_log > c1.n_log
    c3 = _release_row(c2, written, rec, slot, ev.audience, study)
    c4 = _tree_set(c3, woke=jnp.asarray(True))
    return c4, _emit(written.astype(I64), slot), rec


def _cancel_lane(
    c, study, truth, root, lane, enable, status=RecordStatus.PARTIAL, refund=True
):
    """Cancel a live occupancy and refund its unused cost.

    Live-cancel path: invalidate the lane's pending Done row, write the partial record
    at the elapsed fraction, refund the unused cost, free the lane. With `refund` false
    the record is charged in full and nothing comes back (a fault).
    """
    caps, channels = study.caps, study.channels
    pend = c.ev.valid & (c.ev.kind == EventKind.DONE) & (c.ev.lane == lane)
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
        source=c.ev.channel[i],
        vintage=c.ev.scope[i],
        config_id=jnp.asarray(0, I64),
        cost=channels.cost[channel_index] * spent,
        charged=channels.cost[channel_index] * spent,
        valid=has,
    )
    rec = _realize_rec(c, study, truth, root, base, frac)
    rows = _resource_rows(study, lane)
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
    c1, n, slot, rec = _cancel_lane(c, study, truth, root, ev.lane, live)
    return c1, _emit(n, slot), rec


def h_fault(c, ev, study, truth, root):
    """A fault on a lane fails its live occupancy: charged in full, nothing refunded.

    Like a Cancel it names a generation and acts only if that is the lane's current one.
    Which of a fault and a settlement at one tick runs first is the channel's declared
    coincident order, applied by the event queue.
    """
    live = ev.gen == c.ledger.lane_gen[ev.lane]
    c1, n, slot, rec = _cancel_lane(
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
    c1 = _tree_set(
        c,
        **{
            "ledger.lane_closed": c.ledger.lane_closed | in_scope,
            "draining": c.draining | (campaign & drain),
        },
    )
    n_total = jnp.asarray(0, I64)
    slot = jnp.asarray(0, I64)
    rec = record_none(caps)
    for lane in range(caps.n_lanes):
        c1, n, s, r = _cancel_lane(
            c1, study, truth, root, jnp.asarray(lane, I64), in_scope[lane] & abort
        )
        n_total = n_total + n
        slot = jnp.where(n > 0, s, slot)
        rec = where_tree(n > 0, r, rec)
    c2 = _tree_set(c1, done=c1.done | (campaign & abort))
    return c2, _emit(n_total, slot), rec


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
    h_none,
    h_fault,
)
PUBLIC = Audience.OBSERVER | Audience.POLICY
