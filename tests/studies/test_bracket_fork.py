"""The fork battery on the oracle arm, through the bound executor."""

import pytest

from spaceodyssey.core.battery.fork import fork_check, forked
from spaceodyssey.core.enums import EffectKind

from .bracket_helpers import (
    N_STEPS,
    assert_trees_equal,
    build,
    decide_once,
    effect_kind,
    state_with,
    summary_selecting_node,
)

pytestmark = pytest.mark.slow(reason="each battery repeats the chunked driver runs")


@pytest.fixture(scope="module")
def built():
    return build(run=False, horizon_d=30.0)


def test_fork_check_passes_for_the_oracle_arm(built):
    """Fails if package state or the pre-fork records change across a fork."""
    _, study, executor, truths, roots, _ = built
    summary, state = summary_selecting_node(study, node=0), state_with(study)
    trunk = decide_once(study, summary=summary, state=state, balance_d=10.0)
    side = decide_once(
        forked(study, at=2), summary=summary, state=state, balance_d=10.0
    )
    assert effect_kind(trunk) == EffectKind.OBSERVE
    assert int(trunk.state.n_trace) == 1 and int(trunk.state.n_spectra.sum()) == 1
    assert_trees_equal(trunk.effects, side.effects)
    assert_trees_equal(trunk.state, side.state)
    fork_check(study, truths, roots, N_STEPS, executor=executor)
