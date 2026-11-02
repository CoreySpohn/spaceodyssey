"""A world with nothing in it credits nothing; an informative prior does not."""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import pytest

from spaceodyssey.core import battery
from spaceodyssey.core.enums import F64
from spaceodyssey.studies import toy as kinds
from tests.core.battery_helpers import PriorBiasedMean, batched_truth, survey
from tests.core.helpers import worlds

N_WORLDS = 16
N_STEPS = 40
# P(a standard normal exceeds the threshold of 3) for a world with no signal
FALSE_ALARM_RATE = 0.00135


@pytest.fixture(scope="module")
def roots():
    return worlds(N_WORLDS)[1]


def noiseless(study):
    return eqx.tree_at(lambda s: s.channel_model.noise, study, jnp.zeros(3, F64))


def test_null_reference_exact(roots):
    """Without noise the closed form is zero credit, with no tolerance."""
    result = battery.null_reference(
        noiseless(survey()), batched_truth(0.0, N_WORLDS), roots, N_STEPS, expected=0.0
    )
    assert result, result.diagnostic
    assert min(result.detail["n_records"]) >= 5


def test_null_reference_false_alarm_interval(roots):
    """With noise the credit stays inside the binomial false-alarm interval."""
    interval = battery.false_alarm_interval(FALSE_ALARM_RATE, N_WORLDS, N_WORLDS)
    result = battery.null_reference(
        survey(), batched_truth(0.0, N_WORLDS), roots, N_STEPS, interval=interval
    )
    assert result, result.diagnostic


@pytest.mark.parametrize("mode", ["exact", "interval"])
def test_informative_prior_is_caught(mode, roots):
    """Negative control: an observer that adds a prior to every record earns credit."""
    biased = dataclasses.replace(survey(), observer=PriorBiasedMean())
    truths = batched_truth(0.0, N_WORLDS)
    if mode == "exact":
        result = battery.null_reference(
            noiseless(biased), truths, roots, N_STEPS, expected=0.0
        )
    else:
        interval = battery.false_alarm_interval(FALSE_ALARM_RATE, N_WORLDS, N_WORLDS)
        result = battery.null_reference(
            biased, truths, roots, N_STEPS, interval=interval
        )
    assert not result
    assert "credit" in result.diagnostic


def test_null_reference_needs_one_mode(roots):
    with pytest.raises(ValueError):
        battery.null_reference(survey(), batched_truth(0.0, N_WORLDS), roots, N_STEPS)


def test_null_truth_is_not_a_default_world():
    """The null world is empty: its signals are zero, unlike the drawn worlds."""
    drawn, _ = worlds(2)
    assert float(jnp.min(drawn.signal)) >= 0.5
    assert float(jnp.max(batched_truth(0.0, 2).signal)) == 0.0
    assert isinstance(batched_truth(0.0, 2), kinds.ToyTruth)
