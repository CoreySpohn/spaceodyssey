"""Keys."""

import jax
import jax.numpy as jnp
import jax.random as jr

from spaceodyssey.core.enums import Purpose

MASK32 = (1 << 32) - 1
COORDS = (
    "world",
    "subject",
    "channel",
    "acquisition_index",
    "phase",
    "lineage",
    "purpose",
    "revision",
    "status",
)


def fold64(key, x) -> jax.Array:
    """Fold both words of a 64-bit integer, low word first."""
    lo = jnp.asarray(x & MASK32, dtype=jnp.uint32)
    hi = jnp.asarray((x >> 32) & MASK32, dtype=jnp.uint32)
    return jr.fold_in(jr.fold_in(key, lo), hi)


def key_for(root, **coords) -> jax.Array:
    """Fold the ordered coordinates, defaulting missing coordinates to zero."""
    unknown = coords.keys() - COORDS
    if unknown:
        raise TypeError(f"Unknown coordinates: {', '.join(sorted(unknown))}")
    key = root
    for name in COORDS:
        key = fold64(key, coords.get(name, 0))
    return key


def draw_key(root, rec, purpose) -> jax.Array:
    """Key a draw by record identity and purpose."""
    return key_for(
        root,
        world=0,
        subject=rec.subject,
        channel=rec.channel,
        acquisition_index=rec.acquisition_index,
        phase=rec.phase,
        lineage=0,
        purpose=purpose,
        revision=rec.revision,
        status=rec.status,
    )


def policy_key(root, n_dec) -> jax.Array:
    """Key a policy draw by its decision index."""
    return fold64(fold64(root, Purpose.POLICY), n_dec)
