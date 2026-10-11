"""The predictive-distribution waist.

Every measurement model returns a predictive distribution over its dataset;
what a run does with that distribution is the outcome operator. The sampling
engine draws from it against a drawn truth, the belief-space engine pushes
its moments, and the closed-form engine integrates functionals against it.
One distribution type serves all three, and an implementation that cannot
support a method raises :class:`spaceodyssey.errors.UnsupportedOperator`; the
declared capability lives in registry metadata and is checked at
configuration-validation time, never discovered mid-run.

Method names deliberately mirror the prior layer of the inference library
(``sample``, ``log_prob``) so likelihood factories and belief code read the
same on both sides of the waist.
"""

import abc
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import multivariate_normal

from spaceodyssey.errors import UnsupportedOperator


class AbstractPredictive(eqx.Module):
    """A predictive distribution over one observation's dataset."""

    @abc.abstractmethod
    def sample(self, key):
        """Draw one realized dataset payload."""

    @abc.abstractmethod
    def moments(self):
        """Return the (mean, covariance) pair of the predictive."""

    @abc.abstractmethod
    def integrate(self, functional):
        """Return the closed-form expectation of a functional."""

    @abc.abstractmethod
    def log_prob(self, value):
        """Return the log-density of a dataset payload."""


@final
class GaussianSummary(AbstractPredictive):
    """A multivariate Gaussian predictive over summary measurements.

    Supports sampling and moment propagation; closed-form integration of
    arbitrary functionals is not available.

    Attributes:
        mean: Predicted summary values, shape (d,).
        cov: Covariance of the summary values, shape (d, d).
    """

    mean: jax.Array = eqx.field(converter=jnp.asarray)
    cov: jax.Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Validate that the covariance matches the mean dimension."""
        if self.mean.ndim != 1:
            raise ValueError("mean must be one-dimensional")
        d = self.mean.shape[0]
        if self.cov.shape != (d, d):
            raise ValueError(f"cov must have shape ({d}, {d}), got {self.cov.shape}")

    @classmethod
    def from_sigmas(cls, mean, sigma) -> "GaussianSummary":
        """Build a diagonal-covariance predictive from per-element sigmas."""
        sigma = jnp.asarray(sigma)
        return cls(mean=mean, cov=jnp.diag(sigma**2))

    def sample(self, key):
        """Draw one summary vector."""
        return jr.multivariate_normal(key, self.mean, self.cov)

    def moments(self):
        """Return (mean, cov)."""
        return self.mean, self.cov

    def integrate(self, functional):
        """Raise: arbitrary functionals have no closed form here."""
        raise UnsupportedOperator(
            "GaussianSummary supports the sample and propagate operators; "
            "closed-form integration of arbitrary functionals is not available"
        )

    def log_prob(self, value):
        """Return the Gaussian log-density of a summary vector."""
        return multivariate_normal.logpdf(value, self.mean, self.cov)
