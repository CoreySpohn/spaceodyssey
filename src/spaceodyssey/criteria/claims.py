"""Physical claims evaluated on belief draws or, by the scorer, on truth."""

from abc import abstractmethod
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import AbstractVar
from skyscapes.population import earth_like

from spaceodyssey.adapters.parameters import (
    PlanetParameters,
    semimajor_axis_au,
    truth_parameters,
)


class AbstractClaim(eqx.Module):
    """A named predicate on physical planet parameters."""

    name: AbstractVar[str]

    @abstractmethod
    def evaluate(self, params: PlanetParameters, luminosity_lsun):
        """Return bool[...] on the common leading parameter axes."""


class EarthCandidateClaim(AbstractClaim):
    """The population catalog's exoEarth-candidate box."""

    name: ClassVar[str] = "earth_candidate"

    def evaluate(self, params: PlanetParameters, luminosity_lsun):
        """Evaluate the catalog rule with the host luminosity in solar units."""
        return earth_like(
            semimajor_axis_au(params), params.radius_rearth, luminosity_lsun
        )


class ClaimSummary(eqx.Module):
    """Weighted probabilities [C] of the declared claims."""

    p_claim: jax.Array


def summarize_claims(params, log_w, claims, luminosity_lsun):
    """Summarize claim predicates over weighted draws, including an empty tuple."""
    values = _evaluate_claims(params, claims, luminosity_lsun)
    return ClaimSummary(jax.nn.softmax(log_w) @ values.astype(log_w.dtype))


def _evaluate_claims(params, claims, luminosity_lsun):
    if not claims:
        return jnp.zeros((*params.ln_period_d.shape, 0), bool)
    return jnp.stack([c.evaluate(params, luminosity_lsun) for c in claims], axis=-1)


def claims_on_truth(truth, claims, star_mass_msun, luminosity_lsun, epoch_jd):
    """Scorer predicates bool[W, n, S, C], false for every absent slot."""
    params = jax.vmap(lambda t: truth_parameters(t, star_mass_msun, epoch_jd))(truth)
    return (
        _evaluate_claims(params, claims, luminosity_lsun[:, None])
        & truth.present[..., None]
    )
