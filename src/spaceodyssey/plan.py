"""Plan-level contracts for certificates that integrate over a batch of planets.

A profile certificate scores what an observing plan would let it certify,
planet by planet, without drawing a mission. Its input is a
:class:`PlanSummary` (what the plan delivers for each planet) paired with the
truth planets in a :class:`PlanPredictive`; its output is a
:class:`CertificationOutcome` (per-planet certification probabilities). Every
array carries one canonical leading batch axis P of planets.

Nothing here raises inside ``jit``: shape checks run at construction on
static shapes, and a tier asked outside its validated range reports it
through :attr:`CertificationOutcome.in_range`, which the host-side
:meth:`CertificationOutcome.raise_if_out_of_range` turns into an exception.
"""

from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.errors import OutOfRangeError


def _as_bool(value) -> jax.Array:
    return jnp.asarray(value, dtype=bool)


@final
class PlanSummary(eqx.Module):
    """What an observing plan delivers for each planet in a batch.

    Every field describes the plan's data on one planet, so each carries the
    leading planet axis P. Failed visits (for example a phase inside the
    inner working angle) keep their epoch and are excluded through
    ``visit_success``, a mask rather than a branch, so a batch keeps one
    static shape.

    Attributes:
        snr: Signal-to-noise ratio per spectral bin accumulated over the
            plan, shape (P, B).
        spectral_resolution: Spectral resolving power lambda / delta-lambda
            of the plan's spectra, shape (P,).
        visit_epochs_d: Scheduled visit epochs in days of mission time,
            shape (P, V).
        visit_success: True where a visit returned usable data, shape (P, V).
        astrometric_precision_mas: One-sigma astrometric precision of each
            visit in milliarcseconds, shape (P, V).
        mass_measured: True where the plan includes a mass measurement of
            the planet, shape (P,).
    """

    snr: jax.Array = eqx.field(converter=jnp.asarray)
    spectral_resolution: jax.Array = eqx.field(converter=jnp.asarray)
    visit_epochs_d: jax.Array = eqx.field(converter=jnp.asarray)
    visit_success: jax.Array = eqx.field(converter=_as_bool)
    astrometric_precision_mas: jax.Array = eqx.field(converter=jnp.asarray)
    mass_measured: jax.Array = eqx.field(converter=_as_bool)

    def __check_init__(self):
        """Validate that every field shares the planet axis."""
        if self.snr.ndim != 2:
            raise ValueError(f"snr must have shape (P, B), got {self.snr.shape}")
        n_planets = self.snr.shape[0]
        if self.visit_epochs_d.ndim != 2 or self.visit_epochs_d.shape[0] != n_planets:
            raise ValueError(
                f"visit_epochs_d must have shape ({n_planets}, V), got "
                f"{self.visit_epochs_d.shape}"
            )
        visits = self.visit_epochs_d.shape
        for name in ("visit_success", "astrometric_precision_mas"):
            shape = getattr(self, name).shape
            if shape != visits:
                raise ValueError(f"{name} must have shape {visits}, got {shape}")
        for name in ("spectral_resolution", "mass_measured"):
            shape = getattr(self, name).shape
            if shape != (n_planets,):
                raise ValueError(f"{name} must have shape ({n_planets},), got {shape}")

    @property
    def n_planets(self) -> int:
        """The batch size P."""
        return self.snr.shape[0]


@final
class PlanPredictive(eqx.Module):
    """The data distribution a plan induces on a batch of truth planets.

    The integrate operator evaluates a certificate at fixed truth over the
    noise realizations of the plan's data, so the truth enters here and only
    here. A certificate's decision rule never reads ``truth``; it is used to
    define the data distribution and to score whether a verdict is wrong.

    Attributes:
        plan: The plan summary for the batch.
        truth: The truth planets as a pytree whose array leaves carry the
            leading planet axis P; its structure is set by the certificate
            that consumes it.
    """

    plan: PlanSummary
    truth: Any

    def __check_init__(self):
        """Validate that every truth leaf shares the plan's planet axis."""
        n_planets = self.plan.n_planets
        for leaf in jax.tree_util.tree_leaves(self.truth):
            shape = jnp.shape(leaf)
            if not shape or shape[0] != n_planets:
                raise ValueError(
                    f"truth leaves must lead with the planet axis ({n_planets}), "
                    f"got shape {shape}"
                )


@final
class CertificationOutcome(eqx.Module):
    """Per-planet certification probabilities at fixed truth under a plan.

    The counted direction is the one a yield counts (positive for a
    candidate profile). ``p_wrong`` is the probability of a certification in
    the counted direction whose certified claims are false for the truth
    planet; ``p_neg_wrong`` is the probability of a certified negative
    verdict that is false for it. Closed-form tiers report zero standard
    errors.

    Attributes:
        p_pos: Probability of certification in the counted (positive)
            direction, shape (P,).
        p_neg: Probability of a certified negative verdict, shape (P,).
        p_wrong: Probability of a counted-direction certification that is
            wrong about the truth, shape (P,).
        p_neg_wrong: Probability of a certified negative verdict that is
            wrong about the truth, shape (P,).
        p_pos_se: Monte Carlo standard error of ``p_pos``, shape (P,).
        p_neg_se: Monte Carlo standard error of ``p_neg``, shape (P,).
        p_wrong_se: Monte Carlo standard error of ``p_wrong``, shape (P,).
        p_neg_wrong_se: Monte Carlo standard error of ``p_neg_wrong``,
            shape (P,).
        truth_positive: True where the truth planet satisfies the counted
            claims (a truth candidate), shape (P,).
        in_range: True where the evaluating tier is inside its validated
            range; values outside it are returned but not trusted, shape (P,).
    """

    p_pos: jax.Array = eqx.field(converter=jnp.asarray)
    p_neg: jax.Array = eqx.field(converter=jnp.asarray)
    p_wrong: jax.Array = eqx.field(converter=jnp.asarray)
    p_neg_wrong: jax.Array = eqx.field(converter=jnp.asarray)
    p_pos_se: jax.Array = eqx.field(converter=jnp.asarray)
    p_neg_se: jax.Array = eqx.field(converter=jnp.asarray)
    p_wrong_se: jax.Array = eqx.field(converter=jnp.asarray)
    p_neg_wrong_se: jax.Array = eqx.field(converter=jnp.asarray)
    truth_positive: jax.Array = eqx.field(converter=_as_bool)
    in_range: jax.Array = eqx.field(converter=_as_bool)

    def __check_init__(self):
        """Validate that every field is one value per planet."""
        shape = self.p_pos.shape
        if len(shape) != 1:
            raise ValueError(f"p_pos must have shape (P,), got {shape}")
        for name in (
            "p_neg",
            "p_wrong",
            "p_neg_wrong",
            "p_pos_se",
            "p_neg_se",
            "p_wrong_se",
            "p_neg_wrong_se",
            "truth_positive",
            "in_range",
        ):
            other = getattr(self, name).shape
            if other != shape:
                raise ValueError(f"{name} must have shape {shape}, got {other}")

    def raise_if_out_of_range(self) -> None:
        """Raise when any planet lies outside the tier's validated range.

        Host-side only; never call it under ``jit``.

        Raises:
            OutOfRangeError: Naming the out-of-range planet indices.
        """
        bad = np.flatnonzero(~np.asarray(self.in_range))
        if bad.size:
            raise OutOfRangeError(
                f"{bad.size} of {self.in_range.shape[0]} planets lie outside "
                f"the certificate's validated range: indices {bad.tolist()}"
            )
