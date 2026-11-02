"""Ledger."""

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.enums import I64


class ChannelTable(eqx.Module):
    """Per-channel declarations of lane, timing, kind, audience, and cost."""

    id: jax.Array
    lane: jax.Array
    occupancy_ticks: jax.Array
    latency_ticks: jax.Array
    kind: jax.Array
    audience: jax.Array
    cost: jax.Array  # F64[C, R]


class Ledger(eqx.Module):
    """Balances (asset rows, program rows, then shared) and per-lane state."""

    balance: jax.Array  # F64[BR, R]
    lane_busy_until: jax.Array
    lane_gen: jax.Array
    lane_closed: jax.Array
    lane_draining: jax.Array


def _resource_rows(study, lane):
    """Balance row charged by each resource: r0 asset, r1 program, r2 shared."""
    caps = study.caps
    if caps.n_resources != 3:
        raise ValueError("the ledger charges exactly three resources")
    return jnp.stack(
        [
            study.assets.lane_asset[lane].astype(I64),
            jnp.asarray(caps.n_assets, I64),
            jnp.asarray(caps.n_balance_rows - 1, I64),
        ]
    )


class AssetTable(eqx.Module):
    """Per-lane asset membership and transition duration."""

    lane_asset: jax.Array
    transition_ticks: jax.Array
