"""Swapping the truth changes nothing the policy does until a truth-bearing release."""

import pytest

from spaceodyssey.core import battery
from tests.core.battery_helpers import (
    ValueReadingPolicy,
    batched_truth,
    survey,
    unreleased_view_executor,
)
from tests.core.helpers import worlds

N_WORLDS = 2
N_STEPS = 30
# release of the first record: occupancy 100 plus latency 20
FIRST_RELEASE_TICK = 120


@pytest.fixture(scope="module")
def study():
    return survey(ValueReadingPolicy())


@pytest.fixture(scope="module")
def roots():
    return worlds(N_WORLDS)[1]


def test_truth_swap_bitwise_until_first_release(study, roots):
    """Truths of 0 and 5 give identical traces up to the first released record."""
    result = battery.truth_swap(
        study,
        batched_truth(0.0, N_WORLDS),
        batched_truth(5.0, N_WORLDS),
        roots,
        N_STEPS,
    )
    assert result, result.diagnostic
    # the prefix is not empty: the kickoff, the wait and the settlement come first
    assert all(step >= 3 for step in result.detail["first_release_steps"])


def test_truth_swap_catches_a_policy_that_sees_unreleased_records(study, roots):
    """Negative control: a snapshot that shows written but unreleased records."""
    result = battery.truth_swap(
        study,
        batched_truth(0.0, N_WORLDS),
        batched_truth(5.0, N_WORLDS),
        roots,
        N_STEPS,
        executor=unreleased_view_executor(),
    )
    assert not result
    assert "eff_kind" in result.diagnostic


def test_identical_truths_are_a_vacuous_swap(study, roots):
    result = battery.truth_swap(
        study,
        batched_truth(1.0, N_WORLDS),
        batched_truth(1.0, N_WORLDS),
        roots,
        N_STEPS,
    )
    assert not result
    assert "no truth-dependent record" in result.diagnostic
