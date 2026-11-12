"""Joint planning across facilities: an idle facility changes nothing.

`JointPolicy` plans two facilities together: each round it requests one observation on
asset 0 (channel 10) and one on asset 1 (channel 12) as one admission group, so the pair
is admitted whole, and it plans the next round when both records are in. The study uses
key version 2, so asset 1's draws carry a nonzero asset coordinate.

`with_idle_facility` adds a third facility whose time resource holds zero, with no
channel on its lane. Every key, payload, belief, action and balance row of the first two
facilities must be the same in the larger study. `measurement_keys` is the draw key of
each logged measurement under a given asset lookup, so the comparison covers the keys
themselves and a lookup that counts facilities (`counting_asset`) can be run against it.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.battery.metamorphic import with_zero_time_asset
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import (
    F64,
    I64,
    Disposition,
    EffectKind,
    Purpose,
    RecordStatus,
    Scope,
)
from spaceodyssey.core.keys import asset_coordinate, draw_key
from spaceodyssey.core.kinds import AbstractPolicy
from spaceodyssey.studies import builder, effect_rows

FACILITY_CHANNELS = (10, 12)  # on asset 0 (lane 0) and asset 1 (lane 2)
N_ROUNDS = 2


class JointPolicy(AbstractPolicy):
    """Each round, observe subject `round % 2` on both facilities as one group.

    A round is planned when the records of all earlier rounds are in
    (`pstate[0]` counts rounds planned, `view.belief.n` the records seen); after
    `n_rounds` rounds the next such decision drains.
    """

    n_rounds: int = eqx.field(default=N_ROUNDS, static=True)

    def declare(self):
        """Observe and the closing campaign drain."""
        return Declaration(
            effects=frozenset((EffectKind.OBSERVE, EffectKind.STOP)),
            stops=((Scope.CAMPAIGN, Disposition.DRAIN),),
        )

    def decide(self, pstate, view, study, key):
        """Plan the next round of the joint pair, or drain after the last."""
        rounds = pstate[0].astype(I64)
        ready = jnp.sum(view.belief.n) == len(FACILITY_CHANNELS) * rounds
        plan = ready & (rounds < self.n_rounds)
        stop = ready & (rounds >= self.n_rounds)
        kind = jnp.where(plan, EffectKind.OBSERVE, EffectKind.NONE)
        eff = effect_rows.effects(
            study.caps,
            kind=jnp.stack([jnp.where(stop, EffectKind.STOP, kind), kind]).astype(I64),
            subject=rounds % 2,
            channel_index=jnp.stack(
                [effect_rows.chan_index(study, c) for c in FACILITY_CHANNELS]
            ),
            lane=jnp.asarray([0, 2], I64),
            group=view.n_dec,
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
        )
        return eff, pstate.at[0].add(plan.astype(F64)), jnp.asarray(0.0, F64)


def survey_19(**overrides):
    """The toy study at key version 2 with the joint policy on two facilities."""
    overrides.setdefault("policies", (JointPolicy(),))
    overrides.setdefault("key_version", 2)
    return builder.survey_1(**overrides)


def with_idle_facility(study):
    """`study` plus a third facility whose time resource holds zero (and no channel)."""
    return with_zero_time_asset(study)


def counting_asset(study, channel_index):
    """The asset-key mutant `asset * n_assets`, which moves when a facility is added."""
    lane = study.channels.lane[channel_index]
    return study.assets.lane_asset[lane].astype(I64) * study.caps.n_assets


def measurement_keys(study, root, log, rows, asset_of=asset_coordinate):
    """The measurement draw key of each logged row in `rows`, one key row per record."""
    ids = list(map(int, study.channels.id))
    keys = []
    for i in rows:
        rec = jax.tree_util.tree_map(lambda a, i=i: a[i], log)
        index = ids.index(int(rec.channel))
        coords = dataclasses.replace(
            rec, status=jnp.asarray(RecordStatus.COMPLETED, I64)
        )
        keys.append(
            draw_key(
                root,
                coords,
                Purpose.MEASUREMENT,
                asset=asset_of(study, jnp.asarray(index, I64)),
                branch=study.fork.branch_of(rec.acquisition_index),
                coupling=study.coupling,
            )
        )
    return jnp.stack(keys)


__all__ = [
    "FACILITY_CHANNELS",
    "N_ROUNDS",
    "JointPolicy",
    "counting_asset",
    "measurement_keys",
    "survey_19",
    "with_idle_facility",
]
