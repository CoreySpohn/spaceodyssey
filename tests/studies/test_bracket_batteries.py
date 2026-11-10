"""The firewall batteries on the oracle arm, through the bound executor."""

import jax
import pytest

from spaceodyssey.core.battery.fork import fork_check
from spaceodyssey.core.battery.truth_swap import truth_swap

from .bracket_helpers import N_STEPS, build

pytestmark = pytest.mark.slow(reason="each battery repeats the chunked driver runs")


@pytest.fixture(scope="module")
def built():
    return build(run=False, horizon_d=30.0)


def test_truth_swap_passes_for_the_oracle_arm(built):
    """Swapping the worlds changes nothing until a truth record is read."""
    _, study, executor, truths, roots, _ = built
    swapped = jax.tree_util.tree_map(lambda x: x[::-1], truths)
    truth_swap(study, truths, swapped, roots, N_STEPS, executor=executor)


def test_fork_check_passes_for_the_oracle_arm(built):
    """A forked run replays the main run up to the fork."""
    _, study, executor, truths, roots, _ = built
    fork_check(study, truths, roots, N_STEPS, executor=executor)
