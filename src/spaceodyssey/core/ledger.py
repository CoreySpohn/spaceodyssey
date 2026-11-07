"""Ledger."""

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.enums import I64, TIE_NEUTRAL, CoincidentOrder


class ChannelTable(eqx.Module):
    """Per-channel declarations of lane, timing, kind, audience, and cost."""

    id: jax.Array
    lane: jax.Array
    occupancy_ticks: jax.Array
    latency_ticks: jax.Array
    kind: jax.Array
    audience: jax.Array
    cost: jax.Array  # F64[C, R]
    charge: jax.Array  # F64[C, R], reserved: what a receipt debits; equals cost
    coincident_order: jax.Array  # CoincidentOrder per channel; no default


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
