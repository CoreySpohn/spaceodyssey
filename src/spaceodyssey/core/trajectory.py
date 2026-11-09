"""World-drawn trajectories: a table of values at declared ticks, drawn with the world.

A world source may return a `TrajectoryTable` inside its truth, drawn under its own
purpose key so that no other draw moves it and it moves none. Only a version-2 channel
model reads it, at the start tick of the record it realizes; no policy or observer sees
the table.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from spaceodyssey.core.enums import F64, Purpose


def trajectory_key(world_key) -> jax.Array:
    """The key a world source draws its trajectory tables from.

    `world_key` is the key the world source is handed. The result is a different
    stream from the world key itself and from the keys of every other purpose.
    """
    return jr.fold_in(world_key, Purpose.TRAJECTORY)


class TrajectoryTable(eqx.Module):
    """Values at ascending ticks; `at` interpolates linearly, holding the ends."""

    ticks: jax.Array
    values: jax.Array

    def at(self, tick) -> jax.Array:
        """The value at `tick`: linear between declared ticks, constant outside them."""
        return jnp.interp(jnp.asarray(tick, F64), self.ticks, self.values)


def draw_walk(key, ticks, scale) -> TrajectoryTable:
    """A Gaussian random walk from zero with step standard deviation `scale` per row."""
    ticks = jnp.asarray(ticks, F64)
    steps = scale * jr.normal(key, ticks.shape, dtype=F64)
    return TrajectoryTable(ticks=ticks, values=jnp.cumsum(steps))
