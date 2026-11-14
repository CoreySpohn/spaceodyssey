"""The posterior kernel of the report refit, on the fixture stars and the real channel.

Each test names the property or number it protects. The photomancy report refit is
optional here: the file skips where the installed photomancy lacks it.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("photomancy")
pytest.importorskip("photomancy.orbit.refit", reason="needs the report refit")

from hwoutils.constants import G, Msun2kg, Rearth2AU
from hwoutils.testbudget.compiles import cache_hit_count, compile_count
from photomancy.orbit import refit
from photomancy.orbit.forward import predict_photometry
from photomancy.orbit.updater import UpdaterStatus

from spaceodyssey.adapters import jaxedith as jx
from spaceodyssey.adapters.parameters import from_photomancy
from spaceodyssey.adapters.photomancy import (
    SLOT_DEGRADED,
    SLOT_FAILED,
    SLOT_NEVER_FIT,
    SLOT_NO_SUPPORT,
    SLOT_OK,
    photomancy_report,
    posterior_occupancy,
    slot_status,
)
from spaceodyssey.core.enums import RefitStatus
from spaceodyssey.core.records import pack

STAR, MODE, PROBE_NODE = 0, 1, 80


@pytest.fixture(scope="module")
def channel(bracket_channel):
    return bracket_channel


def _probe_rows(channel, sep_arcsec, days=(100.0, 140.0)):
    """Two visit rows of slot 0, a null and a detection, for a probe planet.

    Returns `(rows, intervals, planet, limits)`: the payloads in the layout of the
    channel, the `ScienceInterval` of each, the planet dict and the planet-free noise
    standard deviation of each row at the planet's geometry.
    """
    built = channel.assets()
    report, law = built.reports[channel.imaging_band_index], built.law
    prepared = channel._prepared(STAR, channel.imaging_band_index, nominal=True)
    a_au = sep_arcsec * prepared.dist_pc
    planet = jx.nominal_planet(
        a_au=a_au,
        e=0.0,
        inclination_rad=0.0,
        W_rad=0.0,
        w_rad=0.0,
        tp_d=0.0,
        Rp_au=Rearth2AU,
        Ag=0.3,
    )
    planet["Mp"] = jnp.zeros(())
    rows, intervals, limits = [], [], []
    for i, day in enumerate(days):
        duration = channel.mode_table.duration_s(MODE)
        interval = jx.ScienceInterval(
            jnp.asarray(channel.epoch_jd + day), jnp.asarray(duration)
        )
        reference = report.budget(prepared, jx.planet_free(planet), interval)
        limit = float(law.limit_counts(reference))
        row = jax.tree.map(
            lambda t: jnp.zeros(t.shape, t.dtype), channel.measurement_template
        )
        row = dataclasses.replace(
            row,
            detected=row.detected.at[0].set(float(i == 1)),
            time_d=interval.midpoint_jd,
            science_s=jnp.asarray(duration),
            occupied_s=jnp.asarray(duration),
            valid_science=jnp.asarray(1.0),
            limit_counts=row.limit_counts.at[0].set(limit),
        )
        if i == 1:
            row = dataclasses.replace(
                row,
                counts=row.counts.at[0, channel.imaging_band_index].set(
                    (law.threshold + 1.0) * limit
                ),
                ra_err=row.ra_err.at[0].set(1e-3),
                dec_err=row.dec_err.at[0].set(1e-3),
            )
        rows.append(pack(row))
        intervals.append(interval)
        limits.append(limit)
    return np.stack(rows), intervals, planet, limits


@pytest.mark.medium(reason="builds the channel grid and four budgets, 8 s cold")
def test_report_likelihood_equals_the_channel_law_at_truth(channel):
    """One sensitivity model: photomancy's `loglike_report` on the converted rows equals
    the channel law's likelihood at the truth to 1e-3 nats (2.4e-4 measured).

    The probe planet is face-on and circular, so its separation and phase do not move
    over an exposure. Its separation is a node of the grid of the zero-row table, not
    exactly of the grid of the real rows (the pilot depends on the exposure), so the
    remaining difference is the linear interpolation between nodes.

    Fails if the grid inverts the counts model (a wrong dmag0 sign or zero point), uses
    the wrong time axis, reads the limit from the wrong row, or rebuilds the grid
    program for new report values after the same shape has been warmed.
    """
    before = compile_count() + cache_hit_count()
    data0 = photomancy_report(channel, STAR, np.zeros((2, channel.widths()[0])), 0)
    assert compile_count() + cache_hit_count() - before <= 10
    sep = float(data0.sep_grid[0, PROBE_NODE])
    rows, intervals, planet, _limits = _probe_rows(channel, sep)
    before = compile_count() + cache_hit_count()
    data = photomancy_report(channel, STAR, rows, 0)
    assert compile_count() + cache_hit_count() == before
    prepared = channel._prepared(STAR, channel.imaging_band_index, nominal=True)
    mass = float(prepared.star_mass_kg)
    a_au = float(planet["a"])
    period = (
        2
        * np.pi
        * np.sqrt((a_au * 1.495978707e11) ** 3 / (6.6743e-11 * mass))
        / 86400.0
    )
    el_channel = jnp.asarray(
        [np.log(period), 0.0, 1.0, 0.0, 0.0, 0.0, np.log(mass / Msun2kg), np.log(0.3)]
    )
    el_phot = el_channel.at[7].add(2 * jnp.log(Rearth2AU))
    params = jax.tree.map(
        lambda x: x[0],
        from_photomancy(el_phot[None], jnp.full((1, channel.n_bands), 0.3), 0),
    )
    alpha, d_mag = predict_photometry(
        data.epochs,
        a_au,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        mass,
        jnp.exp(el_phot[7]),
        float(prepared.dist_pc),
    )
    ours = float(refit.loglike_report(alpha, d_mag, data))
    law = jx.ReportingLaw(*channel.law)
    payload = channel.report_data(STAR, jnp.asarray(rows))
    total = 0.0
    for i, interval in enumerate(intervals):
        row = jax.tree_util.tree_map(lambda x, i=i: x[i, 0], payload.row)
        predicted = channel.predict_budget(STAR, params, interval)
        seen = bool(row.detected > 0.5)
        total += float(jx.loglike_report(law, row, predicted))
        if seen:
            total -= float(law.log_position_density(row, predicted, True))
    assert np.isfinite(ours) and np.isfinite(total)
    np.testing.assert_allclose(ours, total, atol=1e-3, rtol=0)
    before = compile_count() + cache_hit_count()
    other = photomancy_report(channel, STAR + 1, rows, 0)
    assert compile_count() + cache_hit_count() == before
    assert not np.allclose(other.limit_grid, data.limit_grid, equal_nan=True)
    for invalid in (0.5, True):
        with pytest.raises(TypeError, match="subject must be an integer"):
            photomancy_report(channel, invalid, rows, 0)


def _circular(channel, sep_lod, radius_re, albedo):
    """Photomancy elements of a face-on circular planet at `sep_lod` lambda over D."""
    prepared = channel._prepared(STAR, channel.imaging_band_index, nominal=True)
    mass = float(prepared.star_mass_kg)
    lod = float(channel.assets().reports[channel.imaging_band_index].lod_arcsec)
    a_au = sep_lod * lod * float(prepared.dist_pc)
    period = 2.0 * np.pi * np.sqrt(a_au**3 / (G * mass))  # days, G in AU^3/(kg d^2)
    scale = albedo * (radius_re * Rearth2AU) ** 2
    return jnp.asarray(
        [np.log(period), 0.0, 1.0, 0.0, 0.0, 0.0, np.log(mass / Msun2kg), np.log(scale)]
    )


def _agreement(channel, el_phot, mode, days, detected):
    """`(photomancy, channel)` log likelihood of visit rows of one planet.

    `el_phot` are photomancy elements. The rows are those the channel would report
    for the planet: a detection carries the expected counts and position, a null the
    planet-free noise level. The channel side converts the elements with
    `from_photomancy` and evaluates its own budget over each exposure; the
    photomancy side places the planet from the photomancy convention directly (the
    periastron time is the phase times the period) and reads the converted report.
    """
    el_channel = jax.tree.map(
        lambda x: x[0],
        from_photomancy(el_phot[None], jnp.full((1, channel.n_bands), 0.3), 0),
    )
    planet, mass = channel._belief_planet(el_channel)
    prepared = channel._prepared(
        STAR, channel.imaging_band_index, nominal=True, mass_kg=mass
    )
    built = channel.assets()
    report, law_ = built.reports[channel.imaging_band_index], built.law
    rows, intervals = [], []
    for day, seen in zip(days, detected, strict=True):
        duration = channel.mode_table.duration_s(mode)
        interval = jx.ScienceInterval(
            jnp.asarray(channel.epoch_jd + day), jnp.asarray(duration)
        )
        expected = report.budget(prepared, planet, interval)
        free = report.budget(prepared, jx.planet_free(planet), interval)
        row = jax.tree.map(
            lambda t: jnp.zeros(t.shape, t.dtype), channel.measurement_template
        )
        row = dataclasses.replace(
            row,
            detected=row.detected.at[0].set(float(seen)),
            time_d=interval.midpoint_jd,
            science_s=jnp.asarray(duration),
            occupied_s=jnp.asarray(duration),
            valid_science=jnp.asarray(1.0),
            limit_counts=row.limit_counts.at[0].set(law_.limit_counts(free)),
        )
        if seen:
            row = dataclasses.replace(
                row,
                counts=row.counts.at[0, channel.imaging_band_index].set(
                    expected.planet
                ),
                ra=row.ra.at[0].set(expected.ra_arcsec),
                dec=row.dec.at[0].set(expected.dec_arcsec),
                ra_err=row.ra_err.at[0].set(1e-3),
                dec_err=row.dec_err.at[0].set(1e-3),
            )
        rows.append(pack(row))
        intervals.append(interval)
    rows = np.stack(rows)
    data = photomancy_report(channel, STAR, rows, 0)
    period = float(jnp.exp(el_phot[0]))
    star_kg = float(jnp.exp(el_phot[6])) * Msun2kg
    a_au = (G * star_kg / (2.0 * np.pi / period) ** 2) ** (1.0 / 3.0)
    alpha, d_mag = predict_photometry(
        data.epochs,
        a_au,
        float(el_phot[1]),
        float(el_phot[2]),
        float(el_phot[3]),
        float(jnp.cos(el_phot[4])),
        float(jnp.sin(el_phot[4])),
        float(el_phot[5]) * period,
        star_kg,
        float(jnp.exp(el_phot[7])),
        float(prepared.dist_pc),
    )
    ours = float(refit.loglike_report(alpha, d_mag, data))
    law = jx.ReportingLaw(*channel.law)
    payload = channel.report_data(STAR, jnp.asarray(rows))
    theirs = 0.0
    for i, interval in enumerate(intervals):
        row = jax.tree_util.tree_map(lambda x, i=i: x[i, 0], payload.row)
        predicted = channel.predict_budget(STAR, el_channel, interval)
        theirs += float(jx.loglike_report(law, row, predicted))
        if bool(row.detected > 0.5):
            theirs -= float(law.log_position_density(row, predicted, True))
    return ours, theirs, data


@pytest.mark.medium(reason="builds the channel grid and eight budgets, 10 s cold")
@pytest.mark.parametrize("radius_re", [9.0, 28.0])
def test_a_bright_planet_past_the_outer_angle_agrees_with_the_channel(
    channel, radius_re
):
    """Fails if the separation grid cuts off detectable planets beyond the outer angle.

    Two detections of a 9 or 28 Earth-radius probe at 29 lambda/D agree with the
    channel to 0.01 nats, and the grid reaches past 40 lambda/D. Both probes exceed
    the survey reporting threshold at the declared intermediate duration.

    The grid cut gave this detection a zero likelihood (minus infinity). The noise
    level rises by a factor of two every 0.1 lambda/D here, and the report model
    interpolates it linearly between grid nodes: on 1024 geometric nodes that cost
    0.15 nats at 3.5 Earth radii and 6.6 nats at 11. The grid places half its nodes
    where a coarse pilot sees the curve change, which brings both to about 1e-3.
    """
    lod = float(channel.assets().reports[channel.imaging_band_index].lod_arcsec)
    el = _circular(channel, 29.0, radius_re, 0.5)
    ours, theirs, data = _agreement(channel, el, MODE, (100.0, 140.0), (True, True))
    top = float(jnp.max(jnp.where(jnp.isfinite(data.dmag0_grid), data.sep_grid, 0.0)))
    assert top / lod > 40.0
    assert np.isfinite(ours) and np.isfinite(theirs)
    np.testing.assert_allclose(ours, theirs, atol=0.01, rtol=0)


@pytest.mark.medium(reason="builds the channel grid and nine budgets, 10 s cold")
def test_a_general_orbit_agrees_with_the_channel_within_the_declared_terms(channel):
    """Fails if orbit-convention conversion changes the report likelihood.

    Three rows of an eccentric, inclined orbit agree within the original 0.174-nat
    allowance for motion during an exposure and sensitivity-grid interpolation.
    The declared intermediate exposure is shorter than the 320,000 s exposure
    used to set that allowance. The detection rows exceed the survey's threshold.
    """
    base = _circular(channel, 8.0, 0.5, 0.4)
    el = base.at[1].set(0.3).at[2].set(0.5).at[3].set(1.0).at[4].set(2.0).at[5].set(0.3)
    days = (85.0, 140.0, 230.0)
    ours, theirs, _ = _agreement(
        channel,
        el,
        channel.mode_table.channel("detection", 1),
        days,
        (True, False, True),
    )
    assert np.isfinite(ours) and np.isfinite(theirs)
    np.testing.assert_allclose(ours, theirs, atol=3 * (5.3e-2 + 5e-3), rtol=0)
    # the rows are sensitive to the phase: a different periastron phase is not close
    moved, _, _ = _agreement(
        channel,
        el.at[5].set(0.8),
        channel.mode_table.channel("detection", 1),
        days,
        (True, False, True),
    )
    assert not np.isfinite(moved) or abs(moved - ours) > 1.0


def test_the_posterior_occupancy_matches_its_closed_form():
    """`q L / (1 - q + q L)` at known points, and the limits `L = 0` and `q = 0, 1`."""
    like = np.array([0.25, 1.0, 4.0, 0.0])
    for q in (0.1, 0.5, 0.9):
        got = np.asarray(posterior_occupancy(q, np.log(np.maximum(like, 1e-300))))
        want = q * like / (1.0 - q + q * like)
        np.testing.assert_allclose(got[:3], want[:3], rtol=1e-12)
        assert got[3] < 1e-290
    assert float(posterior_occupancy(0.0, 0.0)) == 0.0
    assert float(posterior_occupancy(1.0, -np.inf)) == 0.0
    np.testing.assert_allclose(float(posterior_occupancy(1.0, 0.0)), 1.0)
    np.testing.assert_allclose(float(posterior_occupancy(0.3, -np.inf)), 0.0)


def test_slot_codes_are_one_set_whatever_the_kernel():
    """Photomancy's statuses map onto `RefitStatus`; a non-finite cloud is FAILED."""
    assert slot_status(UpdaterStatus.OK, True) == SLOT_OK
    assert slot_status(UpdaterStatus.LOW_ESS, True) == SLOT_DEGRADED
    assert slot_status(UpdaterStatus.NO_SUPPORT, True) == SLOT_NO_SUPPORT
    assert slot_status(UpdaterStatus.OK, False) == SLOT_FAILED
    codes = (SLOT_OK, SLOT_NO_SUPPORT, SLOT_DEGRADED, SLOT_FAILED)
    assert codes == tuple(int(x) for x in RefitStatus)
    assert SLOT_NEVER_FIT not in codes
