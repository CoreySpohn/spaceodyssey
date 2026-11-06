"""Study."""

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.capacities import Capacities, require_x64
from spaceodyssey.core.enums import F64, I64
from spaceodyssey.core.keys import DEFAULT_COUPLING, Coupling, Fork
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


def _as_horizon_tick(value):
    """Check precision before converting the study horizon to integer ticks."""
    require_x64()
    return jnp.asarray(value, I64)


class Clock(eqx.Module):
    """Duration in seconds of one integer tick."""

    tick_length_s: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, F64))


class Agenda(eqx.Module):
    """Policies, the phase boundaries that select them, and what each phase may ask.

    `permitted_effects` holds one bitmask over `EffectKind` per phase (see
    `effect_mask`); admission refuses an effect whose bit is clear.
    """

    policies: tuple[AbstractPolicy, ...]
    phase_until: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, I64))
    permitted_effects: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, I64))

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
    horizon_tick: jax.Array = eqx.field(converter=_as_horizon_tick)
    agenda: Agenda
    process: AbstractProcess
    observer: AbstractObserver
    design: AbstractDesign
    world_source: AbstractWorldSource
    objective: AbstractObjective
    coupling: Coupling = eqx.field(default_factory=lambda: DEFAULT_COUPLING)
    fork: Fork = eqx.field(default_factory=Fork)

    def __check_init__(self):
        """Validate precision, registered record kinds and the observer's layout.

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
        _check_layout(self.observer, self.caps)


def _shapes(tree):
    """Leaf paths, shapes and dtypes of a pytree, comparable with `==`."""
    paths, _ = jax.tree_util.tree_flatten_with_path(tree)
    return [(path, leaf.shape, leaf.dtype) for path, leaf in paths]


def _check_layout(observer, caps):
    """Require the observer's layout to describe the beliefs it initializes."""
    layout = observer.layout(caps)
    belief = jax.eval_shape(lambda: observer.init(caps))
    summary = jax.eval_shape(lambda: observer.init_summary(caps))
    for which, declared, made in (
        ("belief", layout.belief, belief),
        ("summary", layout.summary, summary),
    ):
        if _shapes(declared) != _shapes(made):
            raise ValueError(
                f"the observer's layout does not describe the {which} it initializes"
            )
