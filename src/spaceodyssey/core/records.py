"""Records."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.enums import F64, I64, B


class Record(eqx.Module):
    """One record with provenance, timing, a cost vector, and a typed payload."""

    kind: jax.Array
    subject: jax.Array
    channel: jax.Array
    channel_index: jax.Array
    lane: jax.Array
    acquisition_index: jax.Array
    start_tick: jax.Array
    end_tick: jax.Array
    available_tick: jax.Array
    status: jax.Array
    phase: jax.Array
    acquisition_id: jax.Array
    revision: jax.Array  # 0
    released_tick: jax.Array  # release tick, -1 until released
    accepted_tick: jax.Array  # acceptance tick, -1 until folded
    superseded_by: jax.Array  # -1: supersession is not modeled
    source: jax.Array  # channel id
    vintage: jax.Array  # phase index at admission
    config_id: jax.Array  # 0
    refund_of: jax.Array  # reserved: acquisition_id of the receipt this row credits
    cost: jax.Array  # F64[R]
    charged: jax.Array  # F64[R], reserved: what the receipt debited; equals cost
    payload: jax.Array  # F64[P]
    valid: jax.Array


def record_none(caps):
    """An invalid record (what a handler returns when it writes nothing)."""
    z = jnp.asarray(0, I64)
    return Record(
        **{n: z for n in RECORD_INT_FIELDS if n != "refund_of"},
        refund_of=jnp.asarray(-1, I64),
        cost=jnp.zeros(caps.n_resources, F64),
        charged=jnp.zeros(caps.n_resources, F64),
        payload=jnp.zeros(caps.payload_width, F64),
        valid=jnp.asarray(False),
    )


class RecordLog(eqx.Module):
    """The record log: the columns of `Record` with a leading row axis."""

    kind: jax.Array
    subject: jax.Array
    channel: jax.Array
    channel_index: jax.Array
    lane: jax.Array
    acquisition_index: jax.Array
    start_tick: jax.Array
    end_tick: jax.Array
    available_tick: jax.Array
    status: jax.Array
    phase: jax.Array
    acquisition_id: jax.Array
    revision: jax.Array
    released_tick: jax.Array
    accepted_tick: jax.Array
    superseded_by: jax.Array
    source: jax.Array
    vintage: jax.Array
    config_id: jax.Array
    refund_of: jax.Array
    cost: jax.Array  # F64[L, R]
    charged: jax.Array  # F64[L, R]
    payload: jax.Array  # F64[L, P]
    valid: jax.Array


def empty_log(caps):
    """Return an empty fixed-capacity record log."""
    z = jnp.zeros(caps.n_log, I64)
    return RecordLog(
        **{n: z for n in LOG_INT_FIELDS if n != "refund_of"},
        refund_of=jnp.full(caps.n_log, -1, I64),
        cost=jnp.zeros((caps.n_log, caps.n_resources), F64),
        charged=jnp.zeros((caps.n_log, caps.n_resources), F64),
        payload=jnp.zeros((caps.n_log, caps.payload_width), F64),
        valid=jnp.zeros(caps.n_log, B),
    )


def write_log(carry, rec, caps):
    """Append `rec` to the log where it is valid and a row is free.

    A full log sets `overflow`, skips the write and returns the clamped slot; the
    caller checks `n_log` before enqueueing a Release for the row.
    """
    ok = rec.valid & (carry.n_log < caps.n_log)
    s = jnp.minimum(carry.n_log, caps.n_log - 1)
    cols = {}
    for name in (*LOG_INT_FIELDS, "cost", "charged", "payload"):
        col = getattr(carry.log, name)
        cols[name] = col.at[s].set(jnp.where(ok, getattr(rec, name), col[s]))
    cols["valid"] = carry.log.valid.at[s].set(jnp.where(ok, True, carry.log.valid[s]))
    log = RecordLog(**cols)
    out = eqx.tree_at(
        lambda c: (c.log, c.n_log, c.overflow),
        carry,
        (log, carry.n_log + ok.astype(I64), carry.overflow | (rec.valid & ~ok)),
    )
    return out, s


RECORD_INT_FIELDS = tuple(
    f.name
    for f in dataclasses.fields(Record)
    if f.name not in ("cost", "charged", "payload", "valid")
)
LOG_INT_FIELDS = tuple(
    f.name
    for f in dataclasses.fields(RecordLog)
    if f.name not in ("cost", "charged", "payload", "valid")
)
