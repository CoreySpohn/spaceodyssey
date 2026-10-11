"""Tests for observation records and likelihood factories."""

import math

import jax
import jax.numpy as jnp
import pytest
from planit_py.vocabulary import Action, FixedDuration, Mode

from spaceodyssey.data import Observation, SummaryDataset


def make_dataset(**overrides) -> SummaryDataset:
    """A two-quantity summary dataset."""
    fields = {
        "values": [1.0, -0.5],
        "sigma": [0.1, 0.2],
        "snr": 8.0,
        "epoch_d": 12.0,
        "labels": ("x_arcsec", "y_arcsec"),
        "band": "500nm",
    }
    fields.update(overrides)
    return SummaryDataset(**fields)


def make_action() -> Action:
    """A minimal imaging action."""
    return Action(
        target_id=3,
        mode=Mode.IMAGING,
        bands=("500nm",),
        integration=FixedDuration(duration_d=1.0),
        start_time_d=12.0,
    )


class TestSummaryDataset:
    """Shape checks and the likelihood factory."""

    def test_shape_mismatch_rejected(self):
        """Values and sigma must agree in shape."""
        with pytest.raises(ValueError, match="share a shape"):
            make_dataset(sigma=[0.1])

    def test_label_count_checked(self):
        """The label tuple must match the value count when provided."""
        with pytest.raises(ValueError, match="labels"):
            make_dataset(labels=("only_one",))

    def test_likelihood_matches_hand_gaussian(self):
        """to_likelihood reproduces the Gaussian log-density exactly."""
        dataset = make_dataset()
        loglike = dataset.to_likelihood(lambda params: params)
        params = jnp.asarray([1.1, -0.4])
        residual = (dataset.values - params) / dataset.sigma
        expected = (
            -0.5 * float(jnp.sum(residual**2))
            - float(jnp.sum(jnp.log(dataset.sigma)))
            - 0.5 * 2 * math.log(2.0 * math.pi)
        )
        assert float(loglike(params)) == pytest.approx(expected, rel=1e-6)

    def test_likelihood_is_differentiable(self):
        """The factory output is grad-able and stationary at the data."""
        dataset = make_dataset()
        loglike = dataset.to_likelihood(lambda params: params)
        gradient = jax.grad(loglike)(dataset.values)
        assert jnp.allclose(gradient, 0.0, atol=1e-6)


class TestObservation:
    """The observation record."""

    def test_carries_realized_duration(self):
        """The record stores the realized duration, not the requested one."""
        obs = Observation(
            action=make_action(),
            epoch_d=12.0,
            duration_d=1.4,
            dataset=make_dataset(),
        )
        assert obs.duration_d == 1.4
        assert obs.meta == {}
