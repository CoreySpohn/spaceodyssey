"""Criterion contracts and their test declarations."""

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.criteria.base import AbstractCriterion


class ImagingCriterion(AbstractCriterion):
    """Explicit claims and no accumulated state for the existing imaging fixtures."""

    claims: tuple = eqx.field(kw_only=True)
    mode_names: tuple = eqx.field(static=True, default=())
    labels: tuple = eqx.field(static=True, default=())
    label_values: jax.Array = eqx.field(default_factory=lambda: jnp.zeros(0))

    def state_template(self, n_tracks, n_bands):
        return eqx.Module()

    def init_state(self, n_tracks, n_bands):
        return eqx.Module()

    def update(self, state, row, modes):
        return state

    def value(self, state, p_claim, tick):
        raise NotImplementedError("this fixture only tests belief folding")

    def options(self, state, summary_star, tick, modes, settings):
        raise NotImplementedError("this fixture only tests belief folding")
