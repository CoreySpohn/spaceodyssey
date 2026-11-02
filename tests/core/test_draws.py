"""Raw measurement variates are invariant under perturbations that move events."""

import pytest

from spaceodyssey.core.enums import RecordStatus
from tests.core.corpus import PERTURBATIONS
from tests.core.draw_helpers import (
    compare_draws,
    draw_table,
    policy_draws_changed,
    run_pair,
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


def test_policy_draws_shift_with_inserted_wait():
    base, other = run_pair("insert_wait", "scan", n_seeds=2)
    assert policy_draws_changed(base, other) > 0  # documented behavior, not a kill


def test_policy_draws_do_not_shift_without_a_perturbation_that_moves_decisions():
    base, other = run_pair("payload_width", "scan", n_seeds=2)
    assert policy_draws_changed(base, other) == 0
