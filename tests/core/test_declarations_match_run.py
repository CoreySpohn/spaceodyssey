"""The stops and needs a policy declares match what a run emits."""

import pytest

from spaceodyssey.core import battery
from spaceodyssey.core.declarations import Declaration, Need
from spaceodyssey.core.enums import Disposition, Scope
from tests.core.battery_helpers import PLAN, StatedPlan, survey
from tests.core.corpus import CORPUS_BY_NAME
from tests.core.helpers import worlds

N_STEPS = 40
DRAIN = (Scope.CAMPAIGN, Disposition.DRAIN)


@pytest.fixture(scope="module")
def inputs():
    return worlds(2)


def run(study, inputs):
    truths, roots = inputs
    return battery.declared_matches_run(study, truths, roots, N_STEPS)


def stated(**declared):
    return survey(StatedPlan(plan=PLAN, stated=Declaration(**declared)))


def test_the_plan_declares_the_stop_it_emits(inputs):
    result = run(survey(), inputs)
    assert result, result.diagnostic
    assert result.detail["declared_not_emitted"] == {0: []}


def test_a_special_stop_is_declared_and_emitted(inputs):
    result = run(CORPUS_BY_NAME["stop_lane_close"].make_study(), inputs)
    assert result, result.diagnostic


def test_needs_that_the_run_meets_pass(inputs):
    result = run(stated(stops=(DRAIN,), needs=(Need(10, 10, 130),)), inputs)
    assert result, result.diagnostic


def test_an_undeclared_stop_is_caught(inputs):
    """Negative control: the policy drains the campaign but declares no stop."""
    result = run(stated(stops=()), inputs)
    assert not result
    assert "undeclared stop" in result.diagnostic


def test_a_stop_declared_for_another_scope_is_caught(inputs):
    result = run(stated(stops=((Scope.LANE, Disposition.CLOSE),)), inputs)
    assert not result


def test_a_need_on_a_channel_never_acquired_is_caught(inputs):
    """Negative control: static pre-flight passes this declaration; the run does not."""
    result = run(stated(stops=(DRAIN,), needs=(Need(11, 0, 500),)), inputs)
    assert not result
    assert "channel 11" in result.diagnostic


def test_a_need_decided_at_a_tick_with_no_decision_is_caught(inputs):
    result = run(stated(stops=(DRAIN,), needs=(Need(10, 10, 131),)), inputs)
    assert not result
    assert "channel 10" in result.diagnostic
