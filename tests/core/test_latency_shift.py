"""A release delay shifts belief updates only, on a fixed plan."""

import dataclasses

import pytest

from spaceodyssey.core import battery
from spaceodyssey.core.battery.replay_latency import delay_in_ticks
from spaceodyssey.core.study import Clock
from tests.core.battery_helpers import (
    AbsolutePlanPolicy,
    draw_by_release_time_executor,
    survey,
)
from tests.core.helpers import worlds

N_STEPS = 40
TWO_DAYS_S = 2 * 86400.0


@pytest.fixture(scope="module")
def study():
    """One tick is one day, so a two-day delay is two ticks."""
    return dataclasses.replace(survey(AbsolutePlanPolicy()), clock=Clock(86400.0))


@pytest.fixture(scope="module")
def inputs():
    return worlds(2)


def test_latency_shift(study, inputs):
    truths, roots = inputs
    result = battery.latency_shift(study, truths, roots, N_STEPS, delay_s=TWO_DAYS_S)
    assert result, result.diagnostic
    assert result.detail["delay_ticks"] == 2
    assert result.detail["moved"] >= 2


def test_delay_is_counted_on_the_study_clock(study):
    assert delay_in_ticks(study, TWO_DAYS_S) == 2
    with pytest.raises(ValueError):
        battery.latency_shift(study, None, None, 1, delay_s=1.0)


def test_latency_shift_catches_draws_keyed_by_release_time(study, inputs):
    """Negative control: a payload that depends on when its record is released."""
    truths, roots = inputs
    result = battery.latency_shift(
        study,
        truths,
        roots,
        N_STEPS,
        delay_s=TWO_DAYS_S,
        executor=draw_by_release_time_executor(),
    )
    assert not result
    assert "payload" in result.diagnostic


def test_latency_shift_rejects_a_plan_that_reacts_to_releases(inputs):
    """Negative control: a release-driven plan moves its acquisitions with the delay."""
    truths, roots = inputs
    reactive = dataclasses.replace(survey(), clock=Clock(86400.0))
    result = battery.latency_shift(reactive, truths, roots, N_STEPS, delay_s=TWO_DAYS_S)
    assert not result
    assert "steps other than releases changed" in result.diagnostic
