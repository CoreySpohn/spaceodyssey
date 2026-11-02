"""Private events must not wake the policy or change what it sees."""

import pytest

from spaceodyssey.core.transaction import wake_by_audience
from spaceodyssey.studies.toy import NullProcess
from tests.core.corpus import LEAK_SCENARIOS
from tests.core.leak_helpers import leaky_wake, run_leak


@pytest.fixture(scope="module")
def baseline():
    return run_leak(NullProcess(), wake_by_audience, n_worlds=2)


@pytest.mark.parametrize("name", ["heartbeat", "private_writer"])
def test_private_events_do_not_wake_and_views_match(name, baseline):
    sc = run_leak(LEAK_SCENARIOS[name], wake_by_audience, n_worlds=2)
    assert sc.wakes == baseline.wakes and sc.views_equal and sc.n_private_events >= 1


def test_leaky_wake_is_caught(baseline):
    sc = run_leak(LEAK_SCENARIOS["heartbeat"], leaky_wake, n_worlds=2)
    assert sc.wakes > baseline.wakes


def test_leaky_wake_with_private_writer_also_caught(baseline):
    sc = run_leak(LEAK_SCENARIOS["private_writer"], leaky_wake, n_worlds=2)
    assert sc.wakes > baseline.wakes


def test_releasing_process_wakes_once_and_is_coincident(baseline):
    sc = run_leak(LEAK_SCENARIOS["releasing"], wake_by_audience, n_worlds=2)
    assert sc.coincident_present
    assert sc.wakes == baseline.wakes + 2  # one extra wake per world


def test_baseline_policy_is_not_trivially_idle(baseline):
    assert baseline.wakes >= 12 and baseline.decisions == baseline.wakes
