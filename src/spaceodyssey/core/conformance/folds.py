"""Count committed observer folds by acquisition without changing their values."""

import dataclasses

import jax
import jax.numpy as jnp

from spaceodyssey.core.enums import I64
from spaceodyssey.core.kinds import AbstractObserver
from spaceodyssey.core.snapshot import MeanBelief


class CountedBelief(MeanBelief):
    """A belief with fixed-capacity acquisition identities and fold counts."""

    ids: jax.Array
    counts: jax.Array


def plain(belief):
    """The observer's value without the conformance counters."""
    return MeanBelief(belief.mean, belief.n, belief.aux)


class CountingObserver(AbstractObserver):
    """Delegate values to an observer and retain counts in committed belief state."""

    inner: AbstractObserver

    def init(self, caps):
        """Initialize the original belief and empty counters."""
        value = self.inner.init(caps)
        return CountedBelief(
            value.mean,
            value.n,
            value.aux,
            jnp.full(caps.n_log, -1, I64),
            jnp.zeros(caps.n_log, I64),
        )

    def fold(self, belief, rec, caps):
        """Count this acquisition whenever the caller commits the folded state."""
        value = self.inner.fold(plain(belief), rec, caps)
        same = belief.ids == rec.acquisition_id
        slot = jnp.where(same.any(), jnp.argmax(same), jnp.argmin(belief.counts > 0))
        return CountedBelief(
            value.mean,
            value.n,
            value.aux,
            belief.ids.at[slot].set(rec.acquisition_id),
            belief.counts.at[slot].add(1),
        )

    def analyze(self, belief):
        """Analyze the original values while preserving the counters."""
        value = self.inner.analyze(plain(belief))
        return dataclasses.replace(belief, mean=value.mean, n=value.n, aux=value.aux)
