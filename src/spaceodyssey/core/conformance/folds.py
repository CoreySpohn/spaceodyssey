"""Count committed observer folds by acquisition without changing their values."""

import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.enums import I64
from spaceodyssey.core.kinds import AbstractObserver
from spaceodyssey.core.snapshot import BeliefLayout

# The counter's own fields. Every other attribute read goes to the wrapped value, so
# an observer value may not use these names; `_counted` refuses one that does.
RESERVED = ("_fold_value", "_fold_ids", "_fold_counts")


class CountedBelief(eqx.Module):
    """An observer's belief or summary with acquisition identities and fold counts.

    `_fold_value` is the wrapped observer's own value, of whichever type its layout
    declares; every other attribute reads through to it, so a policy or host
    service sees the observer's fields unchanged. `_fold_ids` and `_fold_counts`
    have one slot per log row.
    """

    _fold_value: eqx.Module
    _fold_ids: jax.Array
    _fold_counts: jax.Array

    def __getattr__(self, name):
        """Read any attribute other than the counter's own from the wrapped value."""
        if name.startswith("__") or name in RESERVED:
            raise AttributeError(name)
        return getattr(self._fold_value, name)


def plain(belief):
    """The observer's value without the conformance counters."""
    return belief._fold_value


def fold_counts(belief):
    """`(ids, counts)` of the acquisitions folded into a counted belief."""
    return belief._fold_ids, belief._fold_counts


def _counted(value, caps):
    taken = [name for name in RESERVED if hasattr(value, name)]
    if taken:
        raise ValueError(f"an observer value may not define the counter's {taken}")
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
    ids, counts = fold_counts(belief)
    same = ids == key
    slot = jnp.where(same.any(), jnp.argmax(same), jnp.argmin(counts > 0))
    return CountedBelief(value, ids.at[slot].set(key), counts.at[slot].add(1))


class OnValue(eqx.Module):
    """A projection of the wrapped value, comparable by the function it wraps."""

    project: Callable[[eqx.Module], tuple[jax.Array, jax.Array]] = eqx.field(
        static=True
    )

    def __call__(self, belief):
        """Project the value inside a counted belief."""
        return self.project(plain(belief))


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
        summary = inner.project_summary
        return BeliefLayout(
            belief=jax.eval_shape(lambda: self.init(caps)),
            summary=jax.eval_shape(lambda: self.init_summary(caps)),
            count_leaf=f"{RESERVED[0]}.{inner.count_leaf}",
            project=OnValue(inner.project),
            project_summary=None if summary is None else OnValue(summary),
        )

    def capabilities(self):
        """The inner observer's capabilities."""
        return self.inner.capabilities()

    def fold(self, belief, rec, caps, key=None):
        """Count this acquisition whenever the caller commits the folded state."""
        value = self.inner.fold(plain(belief), rec, caps, key=key)
        return _count(belief, value, rec)

    def fold_summary(self, summary, rec, caps, key=None):
        """Count this acquisition in the summary the same way."""
        value = self.inner.fold_summary(plain(summary), rec, caps, key=key)
        return _count(summary, value, rec)

    def summary(self, belief):
        """The inner summary of the wrapped belief, beside the belief's counters."""
        return dataclasses.replace(
            belief, _fold_value=self.inner.summary(plain(belief))
        )

    def refit(self, belief0, rows, caps, key):
        """The inner refit; a refit is not a fold, so the counters are kept."""
        return dataclasses.replace(
            belief0, _fold_value=self.inner.refit(plain(belief0), rows, caps, key)
        )

    def install(self, belief, fitted, subject):
        """The inner install of the refit value; the counters of `belief` are kept."""
        return dataclasses.replace(
            belief,
            _fold_value=self.inner.install(plain(belief), plain(fitted), subject),
        )

    def install_summary(self, summary, fitted, subject):
        """The inner summary install; the counters of `summary` are kept."""
        return dataclasses.replace(
            summary,
            _fold_value=self.inner.install_summary(
                plain(summary), plain(fitted), subject
            ),
        )

    def analyze(self, belief):
        """Analyze the original values while preserving the counters."""
        return dataclasses.replace(
            belief, _fold_value=self.inner.analyze(plain(belief))
        )
