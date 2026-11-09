"""Private events must not wake the policy or change what it sees."""

import pytest

from spaceodyssey.core.transaction import wake_by_audience
from spaceodyssey.studies.toy import NullProcess
from tests.core.corpus import LEAK_SCENARIOS
from tests.core.leak_helpers import historical_leaky_wake, leaky_wake, run_leak
from tests.core.mutants import PerWakeHostExecutor


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


def test_releasing_process_coalesces_with_the_release_it_coincides_with(baseline):
    sc = run_leak(LEAK_SCENARIOS["releasing"], wake_by_audience, n_worlds=2)
    assert sc.coincident_present
    # the process event and the record release share a tick: one decision, not two
    assert sc.wakes == baseline.wakes


def test_baseline_policy_is_not_trivially_idle(baseline):
    assert baseline.wakes >= 12 and baseline.decisions == baseline.wakes


@pytest.mark.parametrize(
    ("scenario", "extra_wakes"), [("heartbeat", 80), ("private_writer", 160)]
)
def test_the_historical_leaky_wake_is_caught_on_the_per_wake_executor(
    scenario, extra_wakes
):
    """Fails if the per-event executor with a wake rule that lets every release wake
    stops adding the extra wakes (80 and 160 here), so the wake isolation check could
    no longer see the historical leak that coalescing and the release-side audience
    filter each blocked on their own.
    """
    process = LEAK_SCENARIOS[scenario]
    kwargs = {"n_worlds": 8, "executor": PerWakeHostExecutor}
    clean = run_leak(process, wake_by_audience, **kwargs)
    assert clean.wakes == clean.wakes_baseline and clean.views_equal
    broken = run_leak(process, historical_leaky_wake, **kwargs)
    assert broken.n_private_events > 0
    assert broken.wakes - broken.wakes_baseline == extra_wakes
