"""Raw measurement variates are invariant under perturbations that move events."""

import pytest

from spaceodyssey.core.enums import RecordStatus
from spaceodyssey.studies import toy as kinds
from tests.core.corpus import PERTURBATIONS
from tests.core.draw_helpers import (
    DrawingSpinPolicy,
    compare_draws,
    compare_policy_draws,
    draw_table,
    policy_draw_table,
    policy_draws_changed,
    run_pair,
    run_study,
)


@pytest.mark.parametrize("name", list(PERTURBATIONS))
@pytest.mark.parametrize("executor", ["scan", "host"])
def test_draws_invariant(name, executor):
    base, other = run_pair(name, executor, n_seeds=4)
    n_compared, n_baseline, n_changed = compare_draws(base, other)
    assert n_compared == n_baseline >= 5 * 4 and n_changed == 0, (name, executor)


def test_cancel_partial_shares_variate_with_full():
    base, other = run_pair("cancel_reobserve", "scan", n_seeds=2)
    key = (1, 0, 10, 0, RecordStatus.COMPLETED)
    assert draw_table(other[0])[key] == draw_table(base[0])[key]
    # the re-observation took the next acquisition_index index
    assert (1, 2, 10, 0, RecordStatus.COMPLETED) in draw_table(other[0])
    # control: the draws are not all equal to each other
    assert len(set(draw_table(other[0]).values())) == len(draw_table(other[0]))


def test_mutant_key_with_n_dec_is_caught():
    base, other = run_pair("insert_wait", "scan", n_seeds=2, mutant=True)
    assert compare_draws(base, other)[2] > 0


def test_mutant_is_caught_on_the_host_executor_too():
    base, other = run_pair("insert_wait", "host", n_seeds=2, mutant=True)
    assert compare_draws(base, other)[2] > 0


@pytest.mark.parametrize("name", list(PERTURBATIONS))
@pytest.mark.parametrize("executor", ["scan", "host"])
def test_policy_draws_invariant(name, executor):
    """Policy draws follow phase, decision tick and ordinal, so none moves."""
    base, other = run_pair(name, executor, n_seeds=2)
    n_compared, _, n_changed = compare_policy_draws(base, other)
    assert n_compared >= 2 * 2 and n_changed == 0, (name, executor)


@pytest.mark.parametrize("executor", ["scan", "host"])
def test_inserted_wait_changes_no_policy_draw(executor):
    """A Wait at another tick moves none of the other policy draws."""
    base, other = run_pair("insert_wait_alongside", executor, n_seeds=2)
    n_compared, n_baseline, n_changed = compare_policy_draws(base, other)
    assert n_changed == 0
    assert n_compared == n_baseline == 6 * 2
    # the inserted Wait adds one decision, at tick 7
    assert len(policy_draw_table(other[0])) == len(policy_draw_table(base[0])) + 1
    assert (7, 0) in policy_draw_table(other[0])
    # and it moves no measurement draw either
    assert compare_draws(base, other)[2] == 0


@pytest.mark.parametrize("executor", ["scan", "host"])
def test_decision_count_policy_key_mutant_is_caught(executor):
    base, other = run_pair(
        "insert_wait_alongside", executor, n_seeds=2, policy_mutant=True
    )
    assert policy_draws_changed(base, other) > 0


@pytest.mark.parametrize("executor", ["scan", "host"])
def test_two_decisions_at_one_tick_draw_distinct_variates(executor):
    (world,) = run_study(
        lambda: kinds.survey_1(policies=(DrawingSpinPolicy(),)),
        executor,
        n_seeds=1,
        n_steps=20,
    )
    table = policy_draw_table(world)
    at_130 = [table[k] for k in sorted(table) if k[0] == 130]
    assert len(at_130) >= 3, "the spin policy must decide repeatedly at one tick"
    assert len(set(at_130)) == len(at_130)
    assert len(set(table.values())) == len(table)
