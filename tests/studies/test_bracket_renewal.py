"""Forecast renewal and the imaging-only decision summary."""

import dataclasses
import types

import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.adapters.jaxedith import ObservingModeChannel
from spaceodyssey.adapters.photomancy import N_GRID, DecisionSummary
from spaceodyssey.core import index_world, replay_run
from spaceodyssey.studies import bracket as br

from .bracket_helpers import build

BUDGET_D = 30.0


def test_a_summary_carries_no_block_or_reveal_leaf():
    """Fails if a removed block leaf survives in the summary or the channel."""
    names = {f.name for f in dataclasses.fields(DecisionSummary)}
    assert not names & {"revealed", "n_spectra", "n_blocks", "p_block"}
    channel_names = {f.name for f in dataclasses.fields(ObservingModeChannel)}
    assert "block_snr_required" not in channel_names
    assert not any(
        hasattr(ObservingModeChannel, n) for n in ("block_snr", "block_time")
    )


@pytest.mark.slow(reason="a refresh study past its forecast window, replayed")
def test_a_star_past_its_forecast_window_is_refit_without_new_rows():
    """The summary is renewed before its clock passes the last forecast node.

    Fails if a star's forecast freezes at its last node once the clock is past the
    window, or if the wait-time renewal is not reproduced by `replay_run`.
    """
    horizon_d = 2.0 * br.N_NODES * br.NODE_SPACING_D
    _, study, executor, _, roots, run = build(horizon_d=horizon_d, budget_d=BUDGET_D)
    last = np.asarray(run.carry.pbelief.ref_tick)
    now = np.asarray(run.carry.tick)
    window = (br.N_NODES - 1) * br.NODE_SPACING_D * br.TICKS_PER_DAY
    assert (now[:, None] - last <= window).all(), "a forecast window ran out"
    for w in range(roots.shape[0]):
        replay_run(
            study,
            roots[w],
            index_world(run.carry, w),
            index_world(run.trace, w),
            refit=executor.boundary,
        )


def test_the_observer_expires_a_summary_at_its_last_node():
    """`expired` is true from the last node on and false before it.

    Fails if the renewal threshold drifts from the policy's current-node rule
    (`g_cur` reaching the last node).
    """
    zeros = DecisionSummary.zeros(2, 3, N_GRID, 1, n_an=1)
    zeros = dataclasses.replace(zeros, ref_tick=jnp.asarray([0, 0]))
    observer = br.SummaryObserver(prior=zeros)
    caps = types.SimpleNamespace(n_subjects=2)
    day = br.TICKS_PER_DAY
    spacing = br.NODE_SPACING_D
    before = (br.N_NODES - 1.6) * spacing * day
    at_last = (br.N_NODES - 1.4) * spacing * day
    assert not observer.expired(zeros, jnp.asarray(round(before)), caps).any()
    assert observer.expired(zeros, jnp.asarray(round(at_last)), caps).all()
