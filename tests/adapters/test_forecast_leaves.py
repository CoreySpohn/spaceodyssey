"""Forecast quantiles, paired model variance, and summary memory admission."""

import dataclasses
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import s2d

from spaceodyssey.adapters import photomancy as ph
from spaceodyssey.adapters.jaxedith import ScienceInterval
from spaceodyssey.adapters.parameters import scale_au2, truth_parameters
from tests.adapters.test_first_tier import hand_cloud
from tests.adapters.test_report_refit import StubChannel
from tests.studies.bracket_helpers import fixture_channel, fixture_source

LOW_WEIGHTS = jnp.log(jnp.asarray([0.05, 0.9, 0.05]))
HIGH_WEIGHTS = jnp.log(jnp.asarray([0.9, 0.05, 0.05]))


class PreparedChannel(StubChannel):
    """Prepared counts with deliberately different count and variance ordering."""

    counts: jax.Array = eqx.field(
        default_factory=lambda: jnp.asarray([30.0, 10.0, 20.0])
    )
    variance: jax.Array = eqx.field(
        default_factory=lambda: jnp.asarray([2.0, 80.0, 5.0])
    )

    def predicted_counts(self, subject, channel_index, start_d, params):
        offset = subject + 100.0 * channel_index + start_d
        return self.counts + offset, self.variance + offset


@pytest.fixture(scope="module")
def cloud():
    return jax.tree.map(lambda x: x[:3], hand_cloud()[0])


def summarize_forecast(channel, cloud, log_w, **kwargs):
    """Three explicit nodes, anchored at a nonzero tick."""
    return ph.summarize(cloud, log_w, 0, channel, ref_tick=7, n_grid=3, **kwargs)


def test_quantiles_are_weighted(cloud):
    """Fails if the forecast ignores the cloud's weights or sorts the wrong axis."""
    channel = PreparedChannel()
    heavy_low, heavy_high = (
        summarize_forecast(channel, cloud, log_w)
        for log_w in (LOW_WEIGHTS, HIGH_WEIGHTS)
    )
    assert bool(jnp.all(heavy_low.fc_counts[..., 1] <= heavy_high.fc_counts[..., 1]))
    assert bool(jnp.any(heavy_low.fc_counts[..., 1] < heavy_high.fc_counts[..., 1]))
    offsets = 7 + ph.GRID_SPACING_D * np.arange(3)[:, None]
    offsets = offsets + 100 * np.arange(channel.mode_table.catalog_channel)[None, :]
    for name, value in (("fc_counts", 10), ("fc_variance", 80)):
        np.testing.assert_allclose(
            getattr(heavy_low, name), (value + offsets)[..., None] + np.zeros(3)
        )
    np.testing.assert_allclose(
        heavy_high.fc_counts, (30 + offsets)[..., None] + np.zeros(3)
    )
    np.testing.assert_array_equal(heavy_low.available, True)


def test_forecast_variance_follows_the_selected_count_draw(cloud):
    """Fails if variance is quantiled separately or Q is fixed at three."""
    quantiles = (0.0, 0.25, 0.5, 1.0)
    leaves = summarize_forecast(
        PreparedChannel(), cloud, jnp.zeros(3), forecast_quantiles=quantiles
    )
    np.testing.assert_allclose(leaves.fc_counts[0, 0], 7 + np.asarray([10, 10, 20, 30]))
    np.testing.assert_allclose(leaves.fc_variance[0, 0], 7 + np.asarray([80, 80, 5, 2]))
    boundary = ph.ReportRefit(n_grid=3, forecast_quantiles=quantiles)
    assert boundary.configuration()["forecast_quantiles"] == list(quantiles)
    assert boundary.model_hash != ph.ReportRefit(n_grid=3).model_hash


def test_dark_slots_forecast_their_cloud(cloud):
    """Fails if first-tier slots drop forecast leaves or occupancy scales counts."""
    channel = PreparedChannel()
    settings = SimpleNamespace(
        claims=(),
        n_grid=3,
        spacing_d=ph.GRID_SPACING_D,
        position_lag_d=ph.POSITION_LAG_D,
        forecast_quantiles=(1 / 6, 1 / 2, 5 / 6),
    )
    leaves = ph._first_leaves(channel, 0, 7, 2, cloud, 0.0, 0.5, settings)
    result = ph._dark_result(leaves, jnp.asarray(3.0), jnp.asarray(0), cloud)
    expected = summarize_forecast(channel, cloud, jnp.zeros(3))
    assert float(result.p_exist) == 0.25
    np.testing.assert_array_equal(result.fc_counts, expected.fc_counts)
    np.testing.assert_array_equal(result.fc_variance, expected.fc_variance)
    np.testing.assert_array_equal(result.available, True)


def summary_shapes(*, S, G, V, Q, C, B):
    """Three imaging durations and two packed count accumulators per track/band."""
    return ph.DecisionSummary.zeros(
        2,
        S,
        G,
        3,
        n_claims=C,
        n_state=2 * S * B,
        n_variants=V,
        n_quantiles=Q,
    )


def test_bytes_per_star_follows_the_declared_shapes():
    """Fails if a leaf is omitted from memory accounting or the ceiling is exceeded."""
    S, G, Q, C, B = 6, 16, 3, 1, 3
    a, b = (summary_shapes(S=S, G=G, V=V, Q=Q, C=C, B=B) for V in (4, 8))
    assert b.bytes_per_star() - a.bytes_per_star() == 4 * (2 * S * G * Q * 8 + G)
    V = 36
    summary = summary_shapes(S=S, G=G, V=V, Q=Q, C=C, B=B)
    expected_bytes = (
        8 * (2 + S * (10 + C + G * 3) + 2 * S * G * V * Q + 2 * S * B) + G * V
    )
    actual_bytes = summary.bytes_per_star()
    message = (
        f"{actual_bytes} bytes/star/audience; "
        f"{2 * actual_bytes} bytes/star for both audiences"
    )
    print(message)
    assert actual_bytes == expected_bytes, message
    assert actual_bytes <= 256 * 1024, message


@pytest.mark.slow(
    reason="real multi-band radiometry and forecast integration, 11.32 s focused cold"
)
def test_a_one_point_cloud_forecasts_its_own_counts():
    """Fails if grid times shift, a band/duration is wrong, or truth mass is read."""
    source = fixture_source(1)
    channel = fixture_channel(source)
    channel = dataclasses.replace(
        channel, catalog_mass_kg=channel.catalog_mass_kg * 1.2
    )
    truth = jax.tree.map(lambda x: x[0], source.population)
    params = truth_parameters(
        truth, jnp.asarray([s.mass_msun for s in source.stars]), source.epoch_jd
    )
    cloud = jax.tree.map(lambda x: x[0, 1][None], params)
    leaves = ph.summarize(cloud, jnp.zeros(1), 0, channel, ref_tick=0, n_grid=3)
    table = channel.mode_table
    for name in ("detection", "characterization"):
        v = table.channel(name, 1)
        start_d = 2 * ph.GRID_SPACING_D
        n, variance = channel.predicted_counts(0, v, start_d, cloud)
        np.testing.assert_allclose(
            leaves.fc_counts[2, v], jnp.full(3, n[0]), rtol=1e-10
        )
        np.testing.assert_allclose(
            leaves.fc_variance[2, v], jnp.full(3, variance[0]), rtol=1e-10
        )
        band = table.band_index(name)
        report = channel.assets().reports[band]
        prepared = channel._prepared(0, band, nominal=True)
        planet, _ = channel._belief_planet(jax.tree.map(lambda x: x[0], cloud))
        planet["Lambda"] = scale_au2(cloud, band)[0]
        interval = ScienceInterval(
            channel.epoch_jd + start_d + 0.5 * table.duration_s(v) * s2d,
            jnp.asarray(table.duration_s(v)),
        )
        expected = report.budget(prepared, planet, interval)
        np.testing.assert_allclose(n, jnp.atleast_1d(expected.planet), rtol=1e-10)
        np.testing.assert_allclose(
            variance, jnp.atleast_1d(expected.variance), rtol=1e-10
        )
    np.testing.assert_array_equal(leaves.available, True)
