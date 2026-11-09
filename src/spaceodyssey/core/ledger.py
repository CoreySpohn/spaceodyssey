"""Ledger."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.capacities import OPTIONAL
from spaceodyssey.core.enums import I64, TIE_NEUTRAL, ChargePricing, CoincidentOrder


class ChannelTable(eqx.Module):
    """Per-channel declarations of lane, timing, kind, audience, and cost."""

    id: jax.Array
    lane: jax.Array
    occupancy_ticks: jax.Array
    latency_ticks: jax.Array
    kind: jax.Array
    audience: jax.Array
    cost: jax.Array  # F64[C, R]
    # F64[C, R], what an admission debits (the variable part). It equals cost, and is
    # read only while the capacities hold refunds; otherwise cost is debited.
    charge: jax.Array
    coincident_order: jax.Array  # CoincidentOrder per channel; no default
    # Refund declarations, read only while the capacities hold refunds. Each is absent
    # at its default: every column OCCUPANCY, no surcharge, no failure kind.
    pricing: jax.Array | None = dataclasses.field(default=None, metadata=OPTIONAL)
    surcharge: jax.Array | None = dataclasses.field(default=None, metadata=OPTIONAL)
    failure_kind: jax.Array | None = dataclasses.field(default=None, metadata=OPTIONAL)
    refund_deadline_ticks: jax.Array | None = dataclasses.field(
        default=None, metadata=OPTIONAL
    )
    # F64[n_ops, C, R]: the cost of each channel per asset ops state, present only while
    # the capacities hold ops states. Entry 0 is the cost at the nominal state.
    cost_by_ops: jax.Array | None = dataclasses.field(default=None, metadata=OPTIONAL)


class Ledger(eqx.Module):
    """Balances (asset rows, program rows, then shared) and per-lane state."""

    balance: jax.Array  # F64[BR, R]
    lane_busy_until: jax.Array
    lane_gen: jax.Array
    lane_closed: jax.Array
    # reserved, read by no rule yet
    program_charge_rule: jax.Array  # I64[P]
    valid_from_tick: jax.Array  # I64[BR]
    valid_until_tick: jax.Array  # I64[BR]
    remainder_rule: jax.Array  # I64[BR]
    disruptive_used: jax.Array  # I64[A]
    # reserved accounting counters, written beside `balance` and read by no rule:
    # what admissions have debited and cancellations have credited back, by row
    debited: jax.Array  # F64[BR, R]
    refunded: jax.Array  # F64[BR, R]
    # Programs a PROGRAM Stop has closed, B[P]; present only with more than one program
    program_closed: jax.Array | None = None
    # Failure receipts that credited nothing; present only while refunds are on
    refund_ignored: jax.Array | None = None
    # Award state, present only while awards are on: the window state and the monotone
    # window generation of each row, the row a REALLOCATE moves its remainder to, and
    # what awards have granted and expiries have lapsed (F64[BR, R])
    win_state: jax.Array | None = None
    win_gen: jax.Array | None = None
    # per lane and payer slot, the window generation of the paying row at the debit of
    # the lane's live occupancy (I64[L, R]); read back into the record's `award_gens`
    lane_award_gens: jax.Array | None = None
    # and the remainder rule and reallocation destination of that row's window then
    lane_award_rules: jax.Array | None = None
    lane_award_tos: jax.Array | None = None
    realloc_row: jax.Array | None = None
    granted: jax.Array | None = None
    lapsed: jax.Array | None = None
    # The commitment table, present only while commitments are on (a slot each, I64[K]):
    # the `CommitState`, lane, program, start and end tick and the commitment's semantic
    # index among those of its program and lane; `cm_count` (I64[P, L]) counts them.
    cm_state: jax.Array | None = None
    cm_lane: jax.Array | None = None
    cm_program: jax.Array | None = None
    cm_start: jax.Array | None = None
    cm_end: jax.Array | None = None
    cm_index: jax.Array | None = None
    cm_count: jax.Array | None = None
    # the configuration each commitment was admitted under (I64[K]), present only while
    # a configuration timeline is on (of size zero without commitments)
    cm_config: jax.Array | None = None
    # the ops state of each asset (I64[A]) and the ops state each commitment was
    # admitted under (I64[K]); present only while ops states are on
    ops_state: jax.Array | None = None
    cm_ops: jax.Array | None = None


def charge_parts(study, channel_index, ops=0):
    """`(charge, fixed, pricing)` of a channel, each of one entry per resource.

    With `cost_by_ops` declared, the variable part is `cost_by_ops[ops]`, the cost at
    the asset's ops state when it was admitted, in place of `cost` and `charge`.

    `charge` is what an admission debits, `fixed` the part of it that is never
    refundable (a surcharge), `pricing` the column's `ChargePricing`. While the
    capacities hold no refunds, the charge is the cost, nothing is fixed and every
    column is OCCUPANCY, so a study without refunds computes as before.
    """
    ch = study.channels
    priced = ch.cost_by_ops is not None
    if not study.caps.refunds:
        cost = ch.cost_by_ops[ops, channel_index] if priced else ch.cost[channel_index]
        zero = jnp.zeros_like(cost)
        return cost, zero, jnp.zeros(cost.shape, I64)
    base = ch.cost_by_ops[ops, channel_index] if priced else ch.charge[channel_index]
    pricing = (
        jnp.zeros(base.shape, I64)
        if ch.pricing is None
        else ch.pricing[channel_index].astype(I64)
    )
    extra = (
        jnp.zeros_like(base) if ch.surcharge is None else ch.surcharge[channel_index]
    )
    fixed = jnp.where(pricing == ChargePricing.SURCHARGE, extra, 0.0)
    return base + fixed, fixed, pricing


def set_ops_state(carry, asset, state, caps, where=True):
    """`carry` with the ops state of `asset` set to `state` where `where` holds.

    This is how a process moves an asset between ops states from its `handle`;
    admissions after it are priced at the new state, and occupancies admitted before it
    keep the state they were admitted under. A state outside 0 to `n_ops - 1` is
    clamped to it. A study with one ops state has no such leaf and the carry is
    returned as it is.
    """
    if not caps.ops:
        return carry
    asset = jnp.asarray(asset, I64)
    state = jnp.clip(jnp.asarray(state, I64), 0, caps.n_ops - 1)
    held = carry.ledger.ops_state
    new = held.at[asset].set(jnp.where(where, state, held[asset]))
    return eqx.tree_at(lambda c: c.ledger.ops_state, carry, new)


def _resource_rows(study, lane, program=0):
    """Balance row charged by each resource: r0 asset, r1 program, r2 shared.

    The program row is `n_assets + program`; with one program it is the one row.
    """
    caps = study.caps
    if caps.n_resources != 3:
        raise ValueError("the ledger charges exactly three resources")
    if caps.n_programs == 1:
        program_row = jnp.asarray(caps.n_assets, I64)
    else:
        program_row = jnp.asarray(caps.n_assets, I64) + jnp.asarray(program, I64)
    return jnp.stack(
        [
            study.assets.lane_asset[lane].astype(I64),
            program_row,
            jnp.asarray(caps.n_balance_rows - 1, I64),
        ]
    )


class AssetTable(eqx.Module):
    """Per-lane asset membership and transition duration.

    `freeze_ticks` and `disruptive_resource` have one entry per asset and are reserved:
    no rule reads them yet.

    `config_switch_ticks` (I64[A, S], `S = caps.max_config_switches`) holds each asset's
    configuration switch ticks, sorted and padded on the right with `CONFIG_PAD`; it is
    present only while the capacities hold a timeline. The configuration of an asset at
    a tick is the number of its switch ticks at or before that tick. The ticks are
    traced, so a batch of timelines shares one program.
    """

    lane_asset: jax.Array
    transition_ticks: jax.Array
    freeze_ticks: jax.Array  # I64[A], 0 means no horizon
    disruptive_resource: jax.Array  # I64[A], -1 means none
    config_switch_ticks: jax.Array | None = dataclasses.field(
        default=None, metadata=OPTIONAL
    )


# Pads the unused tail of an asset's switch ticks; no tick reaches it.
CONFIG_PAD = 2**62


def fault_tie(channels, channel_index):
    """Rank of a fault row on a channel among rows sharing its tick, stage and micro.

    A channel that orders faults first ranks them below the neutral rank, one that
    settles first ranks them above it.
    """
    first = channels.coincident_order[channel_index] == CoincidentOrder.FAULT_FIRST
    return jnp.where(first, TIE_NEUTRAL - 1, TIE_NEUTRAL + 1).astype(I64)
