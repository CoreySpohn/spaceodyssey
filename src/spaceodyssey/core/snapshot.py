"""Snapshot."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.enums import I64, Audience, B
from spaceodyssey.core.records import LOG_INT_FIELDS, RecordLog


class MeanBelief(eqx.Module):
    """Per-subject means, sample counts, and auxiliary analysis state."""

    mean: jax.Array
    n: jax.Array
    aux: jax.Array


class Snapshot(eqx.Module):
    """What a policy sees: released rows compacted to the front, zeros elsewhere."""

    log: RecordLog
    n_rows: jax.Array
    belief: MeanBelief
    tick: jax.Array
    n_dec: jax.Array


class EffectTable(eqx.Module):
    """A decision's fixed-capacity effect rows.

    For Cancel, tick is a generation offset: 0 names the live generation and -1 a stale
    one. For Wait it is the wake tick. For Stop, scope, disposition, and lane identify
    what stops.
    """

    kind: jax.Array
    subject: jax.Array
    channel_index: jax.Array
    group: jax.Array
    tick: jax.Array
    lane: jax.Array
    scope: jax.Array
    disposition: jax.Array
    disruptive: jax.Array  # reserved: asks to break a frozen horizon


def effects_none(caps):
    """Return an effect table with no active rows."""
    z = jnp.zeros(caps.n_effects, I64)
    return EffectTable(
        **{n: z for n in EFFECT_FIELDS if n != "disruptive"},
        disruptive=jnp.zeros(caps.n_effects, B),
    )


def visible_rows(log, audience):
    """Select released policy rows or folded observer rows by audience."""
    return (
        jnp.where(
            audience == Audience.POLICY, log.released_tick >= 0, log.accepted_tick >= 0
        )
        & log.valid
    )


def snapshot_view(carry, audience, caps):
    """Compact visible rows and zero every invisible column.

    The stable sort preserves public row positions when private rows are inserted.
    Settlement identities are hidden because event ids reveal private queue activity.
    """
    vis = visible_rows(carry.log, audience)
    order = jnp.argsort(~vis, stable=True)
    cols = {}
    for name in (*LOG_INT_FIELDS, "cost", "charged", "payload", "valid"):
        col = getattr(carry.log, name)
        mask = vis if col.ndim == 1 else vis[:, None]
        cols[name] = jnp.where(mask, col, jnp.zeros_like(col))[order]
    # Settlement ids shift when private events are queued; hide them from policies.
    cols["acquisition_id"] = jnp.zeros_like(cols["acquisition_id"])
    return Snapshot(
        log=RecordLog(**cols),
        n_rows=vis.sum().astype(I64),
        belief=carry.pbelief,
        tick=carry.tick,
        n_dec=carry.n_dec,
    )


EFFECT_FIELDS = tuple(f.name for f in dataclasses.fields(EffectTable))
