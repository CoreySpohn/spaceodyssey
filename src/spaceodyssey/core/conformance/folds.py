"""Count committed observer folds by acquisition without changing their values."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.enums import I64
from spaceodyssey.core.kinds import AbstractObserver
from spaceodyssey.core.snapshot import BeliefLayout


class CountedBelief(eqx.Module):
    """An observer's belief or summary with acquisition identities and fold counts.

    `value` is the wrapped observer's own value, of whichever type its layout
    declares, and its fields read through the wrapper; `ids` and `counts` have one
    slot per log row.
    """

    value: eqx.Module
    ids: jax.Array
    counts: jax.Array

    def __getattr__(self, name):
        """Read any other field from the wrapped value.

        A policy or host service reads the summary it is shown by field name; under
        the counter it must see the wrapped observer's fields unchanged.
        """
        if name.startswith("__") or name == "value":
            raise AttributeError(name)
        return getattr(self.value, name)


def plain(belief):
    """The observer's value without the conformance counters."""
    return belief.value


def _counted(value, caps):
    return CountedBelief(
        value, jnp.full(caps.n_log, -1, I64), jnp.zeros(caps.n_log, I64)
    )


def _count(belief, value, rec):
    """Count the acquisition of `rec` in `belief` and store the folded `value`."""
    # Fold inputs carry no acquisition id, so an acquisition is named by its
    # channel, subject and per-subject admission index. The channel separates
    # rows that are not admissions (a service answer) from the acquisitions.
    key = (
        jnp.asarray(rec.channel, I64) * (1 << 40)
        + jnp.asarray(rec.subject, I64) * (1 << 20)
        + jnp.asarray(rec.acquisition_index, I64)
    )
    same = belief.ids == key
    slot = jnp.where(same.any(), jnp.argmax(same), jnp.argmin(belief.counts > 0))
    return CountedBelief(
        value, belief.ids.at[slot].set(key), belief.counts.at[slot].add(1)
    )


class CountingObserver(AbstractObserver):
    """Delegate values to an observer and retain counts in committed belief state.

    Both belief types are wrapped, whatever the inner observer's layout declares;
    the values, the count leaf and the projections are the inner observer's.
    """

    inner: AbstractObserver

    def init(self, caps):
        """Initialize the original belief and empty counters."""
        return _counted(self.inner.init(caps), caps)

    def init_summary(self, caps):
        """Initialize the original summary and empty counters."""
        return _counted(self.inner.init_summary(caps), caps)

    def layout(self, caps):
        """The inner layout, with each type wrapped in `CountedBelief`."""
        inner = self.inner.layout(caps)
        return BeliefLayout(
            belief=jax.eval_shape(lambda: self.init(caps)),
            summary=jax.eval_shape(lambda: self.init_summary(caps)),
            count_leaf=f"value.{inner.count_leaf}",
            project=lambda belief: inner.project(belief.value),
            project_summary=lambda summary: inner.summary_projection(summary.value),
        )

    def fold(self, belief, rec, caps):
        """Count this acquisition whenever the caller commits the folded state."""
        return _count(belief, self.inner.fold(belief.value, rec, caps), rec)

    def fold_summary(self, summary, rec, caps):
        """Count this acquisition in the summary the same way."""
        return _count(summary, self.inner.fold_summary(summary.value, rec, caps), rec)

    def analyze(self, belief):
        """Analyze the original values while preserving the counters."""
        return dataclasses.replace(belief, value=self.inner.analyze(belief.value))
