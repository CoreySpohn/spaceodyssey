"""Tests for the predictive-distribution waist."""

import jax.numpy as jnp
import jax.random as jr
import pytest

from spaceodyssey.dist import AbstractPredictive, GaussianSummary
from spaceodyssey.errors import UnsupportedOperator
from spaceodyssey.testing import check_predictive


def make_predictive() -> GaussianSummary:
    """A small two-dimensional Gaussian predictive."""
    return GaussianSummary.from_sigmas(mean=[1.0, -2.0], sigma=[0.5, 2.0])


class TestGaussianSummary:
    """The Gaussian summary predictive."""

    def test_from_sigmas_builds_diagonal_cov(self):
        """Per-element sigmas become a diagonal covariance."""
        pred = make_predictive()
        assert pred.cov.shape == (2, 2)
        assert pred.cov[0, 0] == pytest.approx(0.25)
        assert pred.cov[0, 1] == 0.0

    def test_log_prob_peaks_at_mean(self):
        """The density at the mean beats the density three sigma out."""
        pred = make_predictive()
        at_mean = pred.log_prob(pred.mean)
        away = pred.log_prob(pred.mean + 3.0 * jnp.asarray([0.5, 2.0]))
        assert float(at_mean) > float(away)

    def test_moments_round_trip(self):
        """moments() returns the constructor's mean and covariance."""
        pred = make_predictive()
        mean, cov = pred.moments()
        assert jnp.allclose(mean, pred.mean)
        assert jnp.allclose(cov, pred.cov)

    def test_integrate_unsupported(self):
        """Closed-form integration is declared out of scope."""
        with pytest.raises(UnsupportedOperator):
            make_predictive().integrate(lambda x: x)

    def test_shape_validation(self):
        """Mismatched covariance shapes are rejected at construction."""
        with pytest.raises(ValueError, match="cov"):
            GaussianSummary(mean=[1.0, 2.0], cov=jnp.eye(3))


class TestConformance:
    """The predictive conformance suite."""

    def test_gaussian_passes(self):
        """The reference implementation satisfies the suite."""
        check_predictive(make_predictive(), key=jr.PRNGKey(0))

    def test_lying_moments_fail(self):
        """A predictive whose declared mean is wrong is caught."""

        class LyingMoments(AbstractPredictive):
            """Samples around one mean, declares another."""

            inner: GaussianSummary

            def sample(self, key):
                """Draw from the honest inner predictive."""
                return self.inner.sample(key)

            def moments(self):
                """Declare a mean far from the sampling mean."""
                mean, cov = self.inner.moments()
                return mean + 10.0, cov

            def integrate(self, functional):
                """Match the inner predictive's unsupported integration."""
                raise UnsupportedOperator("no closed form")

            def log_prob(self, value):
                """Delegate to the honest inner predictive."""
                return self.inner.log_prob(value)

        liar = LyingMoments(inner=make_predictive())
        with pytest.raises(AssertionError, match="moments"):
            check_predictive(liar, key=jr.PRNGKey(0))
