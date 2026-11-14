"""The truth-swap firewall on the oracle arm, through the bound executor."""

import jax
import pytest

from spaceodyssey.core.battery.truth_swap import truth_swap
from spaceodyssey.core.enums import EffectKind

from .bracket_helpers import (
    N_STEPS,
    build,
    decide_once,
    effect_kind,
)

pytestmark = pytest.mark.slow(reason="each battery repeats the chunked driver runs")


@pytest.fixture(scope="module")
def built():
    return build(run=False, horizon_d=30.0)


def test_truth_swap_passes_for_the_oracle_arm(built):
    """Fails if package kickoff or later decisions read unreleased truth."""
    _, study, executor, truths, roots, _ = built
    kickoff = decide_once(study)
    assert effect_kind(kickoff) == EffectKind.OBSERVE
    assert (
        int(kickoff.effects.channel_index[0])
        == study.channel_model.mode_table.catalog_channel
    )
    assert int(kickoff.state.kick) == 1 and int(kickoff.state.n_trace) == 0
    swapped = jax.tree_util.tree_map(lambda x: x[::-1], truths)
    truth_swap(study, truths, swapped, roots, N_STEPS, executor=executor)
