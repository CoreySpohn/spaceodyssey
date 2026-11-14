"""Decision criteria with a fixed-shape, codec-packed state per star.

This interface has no physical dependencies. Concrete criteria declare static
mode names and labels, traced label values [L], and their tuple of claims.
"""

from abc import abstractmethod

import equinox as eqx
import jax


class StarOptions(eqx.Module):
    """Package gains [G, V], resource times [G, V, R], and first channels [G, V].

    track_gain [G, V, S], when supplied, retains each track's contribution under
    the package's shared allocation. None uses independent track feasibility.
    """

    gain: jax.Array
    time: jax.Array
    first_channel: jax.Array
    track_gain: jax.Array | None = None


class AbstractCriterion(eqx.Module):
    """A pure measurement fold and its value and package decision rules.

    State leaves have leading track axis [S]. The observer packs the whole
    per-star module into one F64[W] leaf, where W is its template's layout size.
    Claims are AbstractClaim instances supplied by the concrete criterion.
    """

    version: int = eqx.field(default=1, kw_only=True)
    mode_names: eqx.AbstractVar[tuple[str, ...]]
    labels: eqx.AbstractVar[tuple[str, ...]]
    label_values: eqx.AbstractVar[jax.Array]
    claims: eqx.AbstractVar[tuple]

    @abstractmethod
    def state_template(self, n_tracks, n_bands) -> eqx.Module:
        """One star's state shapes and dtypes, with leaves [S, ...]."""

    @abstractmethod
    def init_state(self, n_tracks, n_bands) -> eqx.Module:
        """The initial state of one star, matching state_template."""

    @abstractmethod
    def update(self, state, row, modes):
        """Fold a MeasurementRow in time order, using its mode and assignments."""

    @abstractmethod
    def value(self, state, p_claim, tick):
        """Return values F64[S] in [0, 1] and label indices I64[S], or -1.

        p_claim is F64[S, C], the probabilities of this criterion's claims.
        """

    @abstractmethod
    def options(self, state, summary_star, tick, modes, settings) -> StarOptions:
        """The package table for one star at tick under the supplied settings."""
