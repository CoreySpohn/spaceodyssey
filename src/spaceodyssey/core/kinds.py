"""Kinds."""

import equinox as eqx
import jax.numpy as jnp

from spaceodyssey.core.enums import F64, I64
from spaceodyssey.core.records import record_none
from spaceodyssey.core.snapshot import MeanBelief


class AbstractObserver(eqx.Module):
    """Observer contract for record folding and analysis."""

    version: int = eqx.field(default=1, kw_only=True)

    def init(self, caps):
        """Initialize a fixed-shape belief state."""
        return MeanBelief(
            mean=jnp.zeros(caps.n_subjects, F64),
            n=jnp.zeros(caps.n_subjects, I64),
            aux=jnp.zeros(caps.n_an, F64),
        )

    def fold(self, belief, rec, caps):
        """Fold a single released record into the belief."""
        raise NotImplementedError

    def analyze(self, belief):
        """Analyze the folded belief."""
        return belief


class AbstractProcess(eqx.Module):
    """Contract for event seeding and process transitions."""

    version: int = eqx.field(default=1, kw_only=True)

    def seed(self, carry, caps):
        """Seed process events into the queue."""
        return carry

    def handle(self, carry, ev, study):
        """Handle a process event and return an optional record."""
        return carry, record_none(study.caps)


class AbstractPolicy(eqx.Module):
    """Policy contract for snapshot-based decisions."""

    version: int = eqx.field(default=1, kw_only=True)

    def decide(self, pstate, view, study, key):
        """Returns (EffectTable, new pstate, uniform draw u)."""
        raise NotImplementedError

    def decide_py(self, pstate, view_np, study, key):
        """Python-form policy for the host executor; same contract as `decide`."""
        return self.decide(pstate, view_np, study, key)


class AbstractWorldSource(eqx.Module):
    """Source of a world's latent state."""

    version: int = eqx.field(default=1, kw_only=True)

    def draw(self, key, capacities):
        """Draw one world's state."""
        raise NotImplementedError


class AbstractDesign(eqx.Module):
    """Continuous study design leaves."""

    version: int = eqx.field(default=1, kw_only=True)


class AbstractChannelModel(eqx.Module):
    """Realization contract for a channel's payload."""

    version: int = eqx.field(default=1, kw_only=True)

    def realize(self, truth, subject, channel_index, fraction, key, design, channels):
        """Return a realized payload and its random variate."""
        raise NotImplementedError


class AbstractObjective(eqx.Module):
    """Objective evaluated from the completed state."""

    version: int = eqx.field(default=1, kw_only=True)
    unit: str = eqx.field(default="subject", kw_only=True)

    def evaluate(self, carry, study):
        """Evaluate the objective."""
        raise NotImplementedError
