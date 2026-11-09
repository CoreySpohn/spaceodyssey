"""Events."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.enums import (
    BIG,
    I64,
    N_STAGES,
    N_TIES,
    TIE_NEUTRAL,
    B,
    Capacity,
)


def _order_key_py(tick, stage, micro, sid, caps, tie=TIE_NEUTRAL):
    pos = ((tick * N_STAGES + stage) * caps.max_micro + micro) * N_TIES + tie
    return pos * caps.max_sid + sid


def order_key(tick, stage, micro, sid, caps, tie=TIE_NEUTRAL):
    """Packed key, lexicographic in (tick, stage, micro, tie, sid) within the caps.

    `tie` ranks rows that share a tick, stage and microstep before the stable id breaks
    the remaining ties; every row except a fault row carries the neutral rank.
    """
    key = _order_key_py(
        jnp.asarray(tick, I64),
        jnp.asarray(stage, I64),
        jnp.asarray(micro, I64),
        jnp.asarray(sid, I64),
        caps,
        jnp.asarray(tie, I64),
    )
    return key.astype(I64)


class EventTable(eqx.Module):
    """One row per queued event; `valid` false means an empty slot."""

    kind: jax.Array
    tick: jax.Array
    stage: jax.Array
    micro: jax.Array
    tie: jax.Array  # rank among rows at one (tick, stage, micro), before sid
    sid: jax.Array
    lane: jax.Array
    channel_index: jax.Array  # index into the channel table
    channel: jax.Array  # stable channel id
    subject: jax.Array
    acquisition_index: jax.Array
    gen: jax.Array  # lane generation named by a Cancel
    group: jax.Array  # admission group of the decision that created the row
    audience: jax.Array
    slot: jax.Array  # log slot a Release row refers to
    scope: jax.Array
    disposition: jax.Array
    acquisition_id: jax.Array  # sid of the Done row a record settles under
    snapshot_fp: jax.Array  # fingerprint of the snapshot a request or result names
    version: jax.Array  # release version of that snapshot
    decision: jax.Array  # index of the decision that asked
    valid: jax.Array
    # The program a Done row charges and settles under; absent (None, no leaf) unless
    # the capacities hold more than one program, so a study with one is unchanged.
    program: jax.Array | None = None
    # The bundle a Done row belongs to (`decision * max_bundles + ordinal`, -1 for
    # none); absent unless the capacities hold bundles.
    bundle: jax.Array | None = None


def empty_events(caps):
    """Return an empty fixed-capacity event table."""
    z = jnp.zeros(caps.n_events, I64)
    return EventTable(
        **{n: z for n in EVENT_INT_FIELDS},
        valid=jnp.zeros(caps.n_events, B),
        program=z if caps.n_programs > 1 else None,
        bundle=z if caps.bundles else None,
    )


def empty_event_row(caps=None):
    """One unoccupied event row with scalar leaves.

    A row that sits beside the table (a request, a refresh) takes `caps` so it has the
    table's program leaf exactly where the table has one.
    """
    z = jnp.asarray(0, I64)
    on = caps is not None and caps.n_programs > 1
    return EventTable(
        **{n: z for n in EVENT_INT_FIELDS},
        valid=jnp.asarray(False),
        program=z if on else None,
        bundle=z if caps is not None and caps.bundles else None,
    )


def blank_row(row):
    """An unoccupied row with the structure of `row` (a request or refresh row)."""
    return jax.tree_util.tree_map(jnp.zeros_like, row)


def where_tree(pred, a, b):
    """`jnp.where(pred, x, y)` on every leaf pair."""
    return jax.tree_util.tree_map(lambda x, y: jnp.where(pred, x, y), a, b)


def _set_row(table, i, write, **fields):
    """Write named fields of row i of a struct-of-arrays module where `write`."""
    names = tuple(fields)
    new = tuple(
        getattr(table, n).at[i].set(jnp.where(write, v, getattr(table, n)[i]))
        for n, v in fields.items()
    )
    return eqx.tree_at(lambda t: tuple(getattr(t, n) for n in names), table, new)


def enqueue(
    carry,
    ok,
    *,
    kind,
    tick,
    stage,
    lane=0,
    channel_index=0,
    channel=0,
    subject=0,
    acquisition_index=0,
    gen=0,
    group=0,
    audience=0,
    slot=-1,
    scope=0,
    disposition=0,
    acquisition_id=-1,
    snapshot_fp=0,
    version=0,
    decision=0,
    tie=TIE_NEUTRAL,
    program=0,
    bundle=-1,
    caps,
):
    """Add one row to the queue where `ok`, applying the micro placement rule.

    A row at the current (tick, stage) gets micro + 1; a later one gets micro 0; an
    earlier one is placed at the current position with micro + 1 and sets
    `violation`. A full queue or an exhausted `next_sid` sets `overflow`, records which
    in `overflow_mask` and leaves every existing row alone.
    """
    tick = jnp.asarray(tick, I64)
    stage = jnp.asarray(stage, I64)
    ok = jnp.asarray(ok, B)
    free = ~carry.ev.valid
    has_slot = free.any()
    sid_ok = carry.next_sid < caps.max_sid
    write = ok & has_slot & sid_ok
    i = jnp.argmax(free)

    same = (tick == carry.tick) & (stage == carry.stage)
    later = (tick > carry.tick) | ((tick == carry.tick) & (stage > carry.stage))
    earlier = ~same & ~later
    micro = jnp.where(later, 0, carry.micro + 1).astype(I64)
    tick_w = jnp.where(earlier, carry.tick, tick)
    stage_w = jnp.where(earlier, carry.stage, stage)

    # only a table that holds a program column takes one (static: the leaf is absent)
    extra = {}
    if carry.ev.program is not None:
        extra["program"] = jnp.asarray(program, I64)
    if carry.ev.bundle is not None:
        extra["bundle"] = jnp.asarray(bundle, I64)
    ev = _set_row(
        carry.ev,
        i,
        write,
        **extra,
        kind=jnp.asarray(kind, I64),
        tick=tick_w,
        stage=stage_w,
        micro=micro,
        tie=jnp.asarray(tie, I64),
        sid=carry.next_sid,
        lane=jnp.asarray(lane, I64),
        channel_index=jnp.asarray(channel_index, I64),
        channel=jnp.asarray(channel, I64),
        subject=jnp.asarray(subject, I64),
        acquisition_index=jnp.asarray(acquisition_index, I64),
        gen=jnp.asarray(gen, I64),
        group=jnp.asarray(group, I64),
        audience=jnp.asarray(audience, I64),
        slot=jnp.asarray(slot, I64),
        scope=jnp.asarray(scope, I64),
        disposition=jnp.asarray(disposition, I64),
        acquisition_id=jnp.asarray(acquisition_id, I64),
        snapshot_fp=jnp.asarray(snapshot_fp, I64),
        version=jnp.asarray(version, I64),
        decision=jnp.asarray(decision, I64),
        valid=jnp.asarray(True),
    )
    return eqx.tree_at(
        lambda c: (c.ev, c.next_sid, c.overflow, c.overflow_mask, c.violation),
        carry,
        (
            ev,
            carry.next_sid + write.astype(I64),
            carry.overflow | (ok & ~(has_slot & sid_ok)),
            carry.overflow_mask
            | jnp.where(ok & ~has_slot, Capacity.QUEUE_SLOTS, 0)
            | jnp.where(ok & has_slot & ~sid_ok, Capacity.STABLE_IDS, 0),
            carry.violation | (write & earlier),
        ),
    )


def requeue(carry, row, ok):
    """Put a row `select_next` popped back unchanged, in a free slot, where `ok`.

    Every field, position and id included: the row keeps its place in the order, so a
    handler can hold a row while it waits for the host. The pop freed a slot, so one is
    always available.
    """
    free = ~carry.ev.valid
    i = jnp.argmax(free)
    write = jnp.asarray(ok, B) & free.any()
    ev = jax.tree_util.tree_map(
        lambda col, x: col.at[i].set(jnp.where(write, x, col[i])), carry.ev, row
    )
    ev = eqx.tree_at(
        lambda e: e.valid, ev, carry.ev.valid.at[i].set(carry.ev.valid[i] | write)
    )
    return eqx.tree_at(lambda c: c.ev, carry, ev)


def select_next(carry, caps):
    """Pop the queued row with the smallest order key.

    Return (carry, one-row EventTable, has_row). Free the slot before running any
    handler, allowing a replacement even when the queue was full. Update the current
    position to the selected row and set stalled when its microstep reaches the guard.
    """
    ev = carry.ev
    keys = jnp.where(
        ev.valid, order_key(ev.tick, ev.stage, ev.micro, ev.sid, caps, ev.tie), BIG
    )
    i = jnp.argmin(keys)
    has = ev.valid.any()
    row = jax.tree_util.tree_map(lambda a: a[i], ev)
    cleared = ev.valid.at[i].set(jnp.where(has, False, ev.valid[i]))
    new = eqx.tree_at(
        lambda c: (c.ev, c.tick, c.stage, c.micro, c.stalled),
        carry,
        (
            eqx.tree_at(lambda e: e.valid, ev, cleared),
            jnp.where(has, row.tick, carry.tick),
            jnp.where(has, row.stage, carry.stage),
            jnp.where(has, row.micro, carry.micro),
            carry.stalled | (has & (row.micro >= caps.micro_guard)),
        ),
    )
    return new, row, has


EVENT_INT_FIELDS = tuple(
    f.name
    for f in dataclasses.fields(EventTable)
    if f.name not in ("valid", "program", "bundle")
)
