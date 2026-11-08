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


def charge_parts(study, channel_index):
    """`(charge, fixed, pricing)` of a channel, each of one entry per resource.

    `charge` is what an admission debits, `fixed` the part of it that is never
    refundable (a surcharge), `pricing` the column's `ChargePricing`. While the
    capacities hold no refunds, the charge is the cost, nothing is fixed and every
    column is OCCUPANCY, so a study without refunds computes as before.
    """
    ch = study.channels
    if not study.caps.refunds:
        cost = ch.cost[channel_index]
        zero = jnp.zeros_like(cost)
        return cost, zero, jnp.zeros(cost.shape, I64)
    base = ch.charge[channel_index]
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
    """

    lane_asset: jax.Array
    transition_ticks: jax.Array
    freeze_ticks: jax.Array  # I64[A], 0 means no horizon
    disruptive_resource: jax.Array  # I64[A], -1 means none


def fault_tie(channels, channel_index):
    """Rank of a fault row on a channel among rows sharing its tick, stage and micro.

    A channel that orders faults first ranks them below the neutral rank, one that
    settles first ranks them above it.
    """
    first = channels.coincident_order[channel_index] == CoincidentOrder.FAULT_FIRST
    return jnp.where(first, TIE_NEUTRAL - 1, TIE_NEUTRAL + 1).astype(I64)
