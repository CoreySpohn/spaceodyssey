"""Named measurement rows, band isolation, noise, and audience boundaries."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.adapters.jaxedith import planet_free
from spaceodyssey.adapters.measurement import measurement_kind
from spaceodyssey.core.enums import Audience
from spaceodyssey.core.records import layout_size, pack, unpack
from tests.adapters.test_observing_mode import _context, _truth
from tests.studies.bracket_helpers import N_SLOTS, fixture_channel, fixture_source

KEY = jax.random.key(0)


@pytest.fixture(scope="module")
def source():
    return fixture_source()


@pytest.fixture(scope="module")
def channel(source):
    return fixture_channel(source)


def realize(channel, source, channel_index, key=KEY):
    """Packed row of the first fixture star, with science starting near day 100."""
    context = _context(channel, 100.0, channel.mode_table.duration_s(channel_index))
    return channel.realize(
        _truth(source, 0), 0, channel_index, 1.0, key, None, None, context=context
    )


def realize_many(channel, source, channel_index, n):
    """Unpacked batch of independent realizations of one exposure."""
    rows = jax.vmap(lambda k: realize(channel, source, channel_index, k))(
        jax.random.split(KEY, n)
    )
    return unpack(channel.measurement_template, rows)


@pytest.mark.medium(reason="builds the fixture and realizes one imaging row")
def test_an_imaging_row_fills_only_its_band(source, channel):
    """Fails if a writer reorders leaves or writes an unexposed band."""
    t = channel.mode_table
    row = unpack(
        channel.measurement_template,
        realize(channel, source, t.channel("detection", 0)),
    )
    other = [b for b in range(t.n_bands) if b != t.band_index("detection")]
    assert row.counts.shape == (N_SLOTS, t.n_bands)
    for values in (row.counts, row.variance, row.true_counts, row.true_variance):
        assert not bool(jnp.any(values[:, other]))
    np.testing.assert_array_equal(row.assignment, np.arange(N_SLOTS))
    np.testing.assert_array_equal(row.identity, row.assignment)
    b = t.band_index("detection")
    detected = np.asarray(row.detected) > 0.5
    assert detected.any(), "the control must expose a signal"
    for values in (row.counts, row.variance, row.true_counts, row.true_variance):
        np.testing.assert_array_equal(values[~detected], 0)
    np.testing.assert_allclose(
        row.variance[detected, b],
        row.limit_counts[detected] ** 2 + jnp.maximum(row.counts[detected, b], 0),
    )
    context = _context(channel, 100.0, t.duration_s(t.channel("detection", 0)))
    interval = channel._interval(context, settle=True)[0]
    prepared = channel._prepared(0, b, nominal=False)
    truth = _truth(source, 0)
    planets = channel._slot_planets(
        truth, 0, truth.band_index(t.imaging_band.center_nm)
    )
    budgets = jax.vmap(
        lambda p: channel.assets().reports[b].budget(prepared, p, interval)
    )(planets)
    np.testing.assert_allclose(row.true_counts[detected, b], budgets.planet[detected])
    np.testing.assert_allclose(
        row.true_variance[detected, b], budgets.variance[detected]
    )
    summary = channel.report_data(
        0, realize(channel, source, t.channel("detection", 0))[None]
    ).row
    np.testing.assert_array_equal(summary.snr[0], row.snr)
    np.testing.assert_array_equal(summary.cadence[0], row.cadence)


def test_truth_columns_are_scorer_only():
    """Fails if identity or a true count is open to the observer or the policy."""
    kind = measurement_kind(3, 2)
    for c in kind.columns:
        hidden = c.name.startswith(("identity", "true_"))
        assert (c.audience == Audience.SCORER) == hidden, c.name


@pytest.mark.medium(reason="realizes one spectrograph row through the channel switch")
def test_a_measurement_row_round_trips_through_the_codec(source, channel):
    """Fails if a measurement row packs to another width or reorders a leaf."""
    channel = dataclasses.replace(
        channel, catalog_mass_kg=channel.catalog_mass_kg * 1.2
    )
    mode = channel.mode_table.channel("characterization", 1)
    flat = realize(channel, source, mode)
    t = channel.measurement_template
    np.testing.assert_array_equal(pack(unpack(t, flat)), flat[: layout_size(t)])
    row = unpack(t, flat)
    assert int(row.mode) == channel.mode_table.mode_of(mode)
    assert int(row.duration_index) == 1
    assert int(row.calibration_id) == channel.calibration
    assert not bool(jnp.any(channel.report_data(0, flat[None]).valid))
    band_index = channel.mode_table.band_index("characterization")
    truth = _truth(source, 0)
    planets = channel._slot_planets(
        truth, 0, truth.band_index(channel.bands[band_index].center_nm)
    )
    context = _context(channel, 100.0, channel.mode_table.duration_s(mode))
    interval = channel._interval(context, settle=True)[0]
    report = channel.assets().reports[band_index]
    true_scene = channel._prepared(0, band_index, nominal=False)
    nominal_scene = channel._prepared(0, band_index, nominal=True)
    budgets = jax.vmap(lambda p: report.budget(true_scene, p, interval))(planets)
    references = jax.vmap(
        lambda p: report.budget(nominal_scene, planet_free(p), interval)
    )(planets)
    np.testing.assert_allclose(row.true_counts[:, band_index], budgets.planet)
    np.testing.assert_allclose(row.true_variance[:, band_index], budgets.variance)
    np.testing.assert_allclose(
        row.limit_counts, channel.assets().law.limit_counts(references)
    )


@pytest.mark.medium(reason="vmaps one spectrograph row over 400 keys")
def test_spectrograph_counts_are_unbiased_with_the_model_variance(source, channel):
    """Fails if measured counts carry a bias or the wrong variance.

    N = (4 / 0.2)**2 = 400 draws resolves a bias of 0.2 sigma at 4 sigma.
    """
    source = eqx.tree_at(
        lambda s: s.population.present,
        source,
        source.population.present.at[0, 0, 0].set(False),
    )
    rows = realize_many(
        channel, source, channel.mode_table.channel("characterization", 1), n=400
    )
    live = rows.true_variance > 0
    z = (
        (rows.counts - rows.true_counts)
        / jnp.sqrt(jnp.where(live, rows.true_variance, 1))
    )[live]
    assert abs(float(z.mean())) < 4 / 20 and abs(float(z.std()) - 1.0) < 0.1
    assert bool(jnp.any(live))
    np.testing.assert_array_equal(rows.detected, 0)
    for leaf in (rows.snr, rows.ra, rows.dec, rows.ra_err, rows.dec_err, rows.corr):
        np.testing.assert_array_equal(leaf, 0)
    b = channel.mode_table.band_index("characterization")
    np.testing.assert_array_equal(rows.true_counts[:, 0, b], 0)
    assert bool(jnp.any(rows.counts[:, 0, b] < 0)), "spectral counts must be uncensored"
    np.testing.assert_allclose(
        rows.variance[..., b],
        rows.limit_counts**2 + jnp.maximum(rows.counts[..., b], 0),
    )
    other = [i for i in range(channel.n_bands) if i != b]
    for leaf in (rows.counts, rows.variance, rows.true_counts, rows.true_variance):
        np.testing.assert_array_equal(leaf[..., other], 0)


@pytest.mark.medium(
    reason="compares absent and present-undetected controls on one channel"
)
def test_absent_slot_row_equals_a_present_undetected_slot_row(source, channel):
    """Fails if a null measurement reveals whether the same slot exists."""
    truth = _truth(source, 0)
    quiet = eqx.tree_at(lambda t: t.Rp_AU, truth, jnp.zeros_like(truth.Rp_AU))
    present = eqx.tree_at(lambda t: t.present, quiet, jnp.ones_like(quiet.present))
    absent = eqx.tree_at(lambda t: t.present, quiet, jnp.zeros_like(quiet.present))
    mode = channel.mode_table.channel("detection", 0)
    context = _context(channel, 100.0, channel.mode_table.duration_s(mode))
    kind = measurement_kind(N_SLOTS, channel.n_bands)
    rows = [
        channel.realize(t, 0, mode, 1.0, KEY, None, None, context=context)
        for t in (absent, present)
    ]
    payload = (
        "detected",
        "counts",
        "variance",
        "snr",
        "ra",
        "dec",
        "ra_err",
        "dec_err",
        "corr",
        "limit_counts",
    )
    for audience in (Audience.OBSERVER, Audience.POLICY):
        mask = jnp.asarray([bool(c.audience & audience) for c in kind.columns])
        visible = [
            unpack(channel.measurement_template, jnp.where(mask, r[: kind.width], 0))
            for r in rows
        ]
        assert not bool(jnp.any(visible[1].detected))
        for name in payload:
            np.testing.assert_array_equal(
                getattr(visible[0], name), getattr(visible[1], name)
            )


def test_association_inference_is_explicitly_unbuilt(channel):
    """Fails if an unsupported association silently enables truth assignment."""
    with pytest.raises(NotImplementedError, match=r"association inference.*not built"):
        dataclasses.replace(channel, association="inferred")
