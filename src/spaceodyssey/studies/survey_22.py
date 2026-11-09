"""Survey 22: the target's trajectory is in the world draw and reaches only records.

The world source draws, besides a stream of normals for each purpose, a trajectory table
of target positions under its own purpose key. A version-2 channel model samples the
table at the start tick of the record it realizes and writes the position as the first
payload column; nothing else reads it, and a process sees no truth. Two truths that
differ only in the table therefore move the realized positions and, until a record
carrying a position is released, nothing the policy reads.

The leaky variant adds a finder: channel 12, on its own lane with no transition and no
occupancy, realized and released at tick 0 with the same model, so its payload is the
table's position at tick 0. The plan observes it first, so a position reaches the policy
before the target's first record does.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from spaceodyssey.core.enums import F64, I64, Purpose
from spaceodyssey.core.kinds import AbstractChannelModel, AbstractWorldSource
from spaceodyssey.core.trajectory import TrajectoryTable, draw_walk, trajectory_key
from spaceodyssey.studies import toy

__all__ = [
    "FINDER",
    "N_PURPOSES",
    "TARGET",
    "TICKS",
    "TrajectoryChannel",
    "TrajectoryTruth",
    "TrajectoryWorldSource",
    "survey_22",
]

TARGET, FINDER = 10, 12  # channel ids of the target observation and the finder
TICKS = jnp.arange(0, 1001, 50)  # the ticks of the trajectory table
N_PURPOSES = len(Purpose)
PLAN = ((0, TARGET),) * 4


class TrajectoryTruth(eqx.Module):
    """Study-defined truth: one stream of normals per purpose and the trajectory table.

    `streams[p]` is drawn under purpose `p` (the row of the table's own purpose is
    unused); `table` holds the target's position at `TICKS`.
    """

    streams: jax.Array
    table: TrajectoryTable


class TrajectoryWorldSource(AbstractWorldSource):
    """Draw each purpose's stream and the table from keys folded by purpose.

    `salts[p]` moves the draw of purpose `p` and no other: the stream of purpose `p`
    comes from `fold_in(fold_in(key, p), salts[p])`, the table from the trajectory key
    folded with `salts[TRAJECTORY]`.
    """

    salts: tuple[int, ...] = eqx.field(default=(0,) * N_PURPOSES, static=True)

    def draw(self, key, capacities):
        """The streams of the five stream purposes and the table of the sixth."""
        own = Purpose.TRAJECTORY
        streams = jnp.stack(
            [
                jr.normal(
                    jr.fold_in(jr.fold_in(key, p), self.salts[p]),
                    (capacities.n_subjects,),
                    dtype=F64,
                )
                for p in range(N_PURPOSES)
                if p != own
            ]
            + [jnp.zeros(capacities.n_subjects, F64)]
        )
        walk = draw_walk(jr.fold_in(trajectory_key(key), self.salts[own]), TICKS, 1.0)
        return TrajectoryTruth(streams=streams, table=walk)


class TrajectoryChannel(AbstractChannelModel):
    """Version 2: the row is `[table position at the start tick, variate, fraction]`."""

    version: int = eqx.field(default=2, kw_only=True)

    @classmethod
    def migrate(cls, from_version, params):
        """A version-1 document carries no field this model needs to change."""
        return params

    def realize(
        self, truth, subject, channel_index, fraction, key, design, channels, *, context
    ):
        """Sample the world's table at the record's start tick."""
        z = jr.normal(key, dtype=F64)
        row = (truth.table.at(context.start_tick), z, context.fraction)
        return jnp.stack([jnp.asarray(x, F64) for x in row])


def survey_22(*, leaky=False):
    """The toy study observing the target four times, with the finder first if `leaky`.

    The finder is channel 12 on lane 2 (asset 1), with the lane's transition and the
    channel's occupancy and latency all zero, so its record is released at tick 0.
    """
    base = toy.survey_1()
    channels = dataclasses.replace(
        base.channels,
        occupancy_ticks=base.channels.occupancy_ticks.at[2].set(0),
        latency_ticks=base.channels.latency_ticks.at[2].set(0),
    )
    plan = ((0, FINDER), *PLAN) if leaky else PLAN
    study = toy.survey_1(
        channels=channels,
        transition_ticks=jnp.asarray([10, 10, 0], I64),
        policies=(toy.ReactivePolicy(plan=plan),),
        channel_model=TrajectoryChannel(),
    )
    return dataclasses.replace(study, world_source=TrajectoryWorldSource())
