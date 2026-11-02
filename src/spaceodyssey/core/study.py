"""Study."""

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.capacities import Capacities, require_x64
from spaceodyssey.core.enums import F64
from spaceodyssey.core.kinds import (
    AbstractChannelModel,
    AbstractDesign,
    AbstractObjective,
    AbstractObserver,
    AbstractPolicy,
    AbstractProcess,
    AbstractWorldSource,
)
from spaceodyssey.core.ledger import AssetTable, ChannelTable


class Clock(eqx.Module):
    """Duration in seconds of one integer tick."""

    tick_length_s: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, F64))


class Agenda(eqx.Module):
    """Policies, the phase boundaries that select them, and what each phase may ask.

    `permitted_effects` holds one bitmask over `EffectKind` per phase (see
    `effect_mask`); admission refuses an effect whose bit is clear.
    """

    policies: tuple[AbstractPolicy, ...]
    phase_until: tuple[int, ...]
    permitted_effects: tuple[int, ...]

    def __check_init__(self):
        """Require one mask per phase."""
        if len(self.permitted_effects) != len(self.policies):
            raise ValueError("permitted_effects needs one mask per policy phase")


class Study(eqx.Module):
    """A composed transaction study with typed kind contracts."""

    caps: Capacities = eqx.field(static=True)
    clock: Clock
    channels: ChannelTable
    channel_model: AbstractChannelModel
    assets: AssetTable
    budget: jax.Array
    horizon_tick: int
    agenda: Agenda
    process: AbstractProcess
    observer: AbstractObserver
    design: AbstractDesign
    world_source: AbstractWorldSource
    objective: AbstractObjective

    def __check_init__(self):
        """Validate the arithmetic precision and that record kinds are registered.

        Payload column audiences come from the registered record kinds. Capacities with
        no registered kinds would leave every column readable by every audience, so a
        study refuses them.
        """
        require_x64()
        if not self.caps.payload_audience:
            raise ValueError(
                "Study capacities have no registered record kinds; lower a registry "
                "into the capacities so payload columns carry their audiences"
            )
