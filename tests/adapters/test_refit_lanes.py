"""Named track results preserve the posterior boundary's summary slices."""

import dataclasses

import jax
import numpy as np
import pytest
from photomancy.orbit.updater import UpdaterStatus

from spaceodyssey.adapters import photomancy as ph
from spaceodyssey.adapters.parameters import PlanetParameters
from tests.adapters.posterior_helpers import (
    golden_slices,
    realize_visits,
    visit_request,
)
from tests.adapters.test_report_posterior_refit import _thin
from tests.studies.bracket_helpers import N_SLOTS, build

CONSTANTS = dict(
    p_det=0.2,
    scatter=0.3,
    pos_mean=0.4,
    p_claim=0.5,
    p_exist=0.6,
    ln_period_q=0.7,
    ess=8.0,
)


def constant_result(n_slots, updater_status=None):
    """Each summary leaf has a distinct value, so swaps cannot pass unnoticed."""
    shapes = dict(
        p_det=(2, 2),
        scatter=(),
        pos_mean=(2,),
        p_claim=(1,),
        p_exist=(),
        ln_period_q=(3,),
        ess=(),
    )
    params = PlanetParameters(
        **{
            f.name: np.ones((n_slots, 4, 2) if f.name == "albedo" else (n_slots, 4))
            for f in dataclasses.fields(PlanetParameters)
        }
    )
    return ph.LaneResult(
        **{
            name: np.full((n_slots, *shapes[name]), value)
            for name, value in CONSTANTS.items()
        },
        updater_status=np.zeros(n_slots, np.int64)
        if updater_status is None
        else np.asarray(updater_status),
        finite=np.ones(n_slots, bool),
        draws=params,
    )


def empty_summary():
    """A writable host summary with world and audience axes like refit slices."""
    summary = ph.DecisionSummary.zeros(2, N_SLOTS, 2, 2, n_claims=1)
    return jax.tree.map(lambda x: np.broadcast_to(x, (1, 2, *x.shape)).copy(), summary)


def test_a_lane_result_scatters_into_the_summary_by_name():
    """Fails if a positional unpack sends one result field into another summary leaf."""
    result = constant_result(n_slots=N_SLOTS)
    summary = ph.scatter_results(empty_summary(), {(0, 0, 1): result})
    for name, value in CONSTANTS.items():
        np.testing.assert_array_equal(getattr(summary, name)[0, 0, 1], value)
        np.testing.assert_array_equal(getattr(summary, name)[0, 1, 1], 0.0)
    np.testing.assert_array_equal(summary.status[0, 0, 1], ph.SLOT_OK)
    for broken, code in (
        (dataclasses.replace(result, finite=np.zeros(N_SLOTS, bool)), ph.SLOT_FAILED),
        (dataclasses.replace(result, scatter=np.full(N_SLOTS, np.nan)), ph.SLOT_FAILED),
        (
            dataclasses.replace(
                result, updater_status=np.full(N_SLOTS, int(UpdaterStatus.NO_SUPPORT))
            ),
            ph.SLOT_NO_SUPPORT,
        ),
    ):
        held = ph.scatter_results(summary, {(0, 0, 1): broken})
        np.testing.assert_array_equal(held.status[0, 0, 1], code)
        for name in CONSTANTS:
            np.testing.assert_array_equal(getattr(held, name), getattr(summary, name))


def test_the_retry_mask_reads_the_updater_status():
    """Fails if retry reads a slot code or another field instead of the raw status."""
    result = constant_result(
        n_slots=3, updater_status=[0, int(UpdaterStatus.LOW_ESS), 0]
    )
    assert ph.retry_mask(result).tolist() == [False, True, False]


@pytest.fixture(scope="module")
def posterior_inputs():
    """Small proposals retain the real single, collapsed and dark-slot paths."""
    source, study, *_ = build(run=False, arm="replan")
    channel = study.channel_model
    truth = jax.tree.map(lambda x: x[0], source.population)
    visits = realize_visits(
        channel,
        truth,
        0,
        channel.mode_table.channel("detection", 1),
        [30.0 * i for i in range(1, 7)],
    )
    boundary = ph.ReportRefit(
        kernel="posterior",
        caps=study.caps,
        lanes=2,
        resummary=False,
        n_particles=16,
        n_round=64,
        n_rounds=1,
        n_centers=4,
        n_constants=1,
        max_rounds=2,
        n_first=16,
        n_estimate_systems=8,
        grid_points=16,
        ess_floor=1.0,
    )
    rich = visit_request(study, {0: visits}, tick=250 * 86_400_000)
    single = visit_request(study, {0: _thin(visits)}, tick=250 * 86_400_000)
    request = jax.tree.map(lambda *xs: np.concatenate(xs), rich, single)
    return study, boundary, request


@pytest.mark.slow(
    reason="real single and collapsed posterior fits with reduced proposals"
)
def test_lane_results_equal_the_previous_slices(posterior_inputs, monkeypatch):
    """Fails if the module gather changes any slice (bitwise at a fixed lane count)."""
    study, boundary, request = posterior_inputs
    engines = set()
    fit = ph._fit_lanes

    def tracked(*args):
        engines.add(args[-2])
        return fit(*args)

    monkeypatch.setattr(ph, "_fit_lanes", tracked)
    expected = golden_slices(boundary, study, request)
    actual = boundary.refit_batch(study, request)
    assert engines == {"single", "multi"}
    slices = dict(
        zip((name for name, _, _ in ph.slice_spec(study)), actual.slices, strict=True)
    )
    for world in (0, 1):
        live = np.asarray(slices["n_det"][world, 0, 0]) > 0
        assert np.any(np.asarray(slices["ess"][world, 0, 0])[live] > 0)
    for name, want, got in zip(
        ("slices", "status", "diagnostics", "computed"),
        (expected.slices, expected.status, expected.diagnostics, expected.computed),
        (actual.slices, actual.status, actual.diagnostics, actual.computed),
        strict=True,
    ):
        for a, b in zip(jax.tree.leaves(want), jax.tree.leaves(got), strict=True):
            np.testing.assert_array_equal(a, b, err_msg=name)
