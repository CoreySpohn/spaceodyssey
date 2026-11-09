"""Reference doubles shared by the unit tests."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from spaceodyssey.core.enums import I64
from spaceodyssey.core.kinds import AbstractChannelModel
from spaceodyssey.core.ledger import set_ops_state
from spaceodyssey.core.lift import traced
from spaceodyssey.studies.survey_11 import ScriptedAlerts


class ToyEpochPhotometry(AbstractChannelModel):
    """A version-2 toy channel model: its row records the whole realize context.

    The row is `[start_tick, end_tick, acquisition_index, fraction, z]` with `z` a
    standard normal draw from the key, so every context field is checked and the draw
    stays sensitive to the key. The study must register a five-column kind.
    """

    version: int = eqx.field(default=2, kw_only=True)

    @classmethod
    def migrate(cls, from_version, params):
        """A version-1 document carries no field this model needs to change."""
        return params

    def realize(
        self, truth, subject, channel_index, fraction, key, design, channels, *, context
    ):
        """Return the full payload row."""
        fields = (
            context.start_tick,
            context.end_tick,
            context.acquisition_index,
            context.fraction,
        )
        z = jr.normal(key, dtype=jnp.float64)
        return jnp.stack([*(jnp.asarray(f, jnp.float64) for f in fields), z])


class OpsFailure(ScriptedAlerts):
    """A scripted record, released `delay` ticks late, that moves an asset's ops state.

    At each fire the process writes its record (the failure notice) and sets the ops
    state of `asset` to `state` in the same handler.
    """

    asset: int = eqx.field(default=0, metadata=traced(I64))
    state: int = eqx.field(default=1, metadata=traced(I64))

    def handle(self, carry, ev, study):
        """Write the notice and set the asset's ops state."""
        carry, rec = super().handle(carry, ev, study)
        carry = set_ops_state(carry, self.asset, self.state, study.caps)
        return carry, rec


class ToyOpsPhotometry(ToyEpochPhotometry):
    """A version-2 toy model whose row is `[ops_state, start, end, fraction, z]`."""

    def realize(
        self, truth, subject, channel_index, fraction, key, design, channels, *, context
    ):
        """Return the row; its first column is the stored admission ops state."""
        fields = (
            context.ops_state,
            context.start_tick,
            context.end_tick,
            context.fraction,
        )
        z = jr.normal(key, dtype=jnp.float64)
        return jnp.stack([*(jnp.asarray(f, jnp.float64) for f in fields), z])
