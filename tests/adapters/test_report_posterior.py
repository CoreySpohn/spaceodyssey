"""The posterior kernel of the report refit, on the fixture stars and the real channel.

Each test names the property or number it protects. The photomancy report refit is
optional here: the file skips where the installed photomancy lacks it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("photomancy")
pytest.importorskip("photomancy.orbit.refit", reason="needs the report refit")

from hwoutils.constants import G, Msun2kg, Rearth2AU
from photomancy.orbit import refit
from photomancy.orbit.forward import predict_photometry

from spaceodyssey.adapters import jaxedith as jx
from spaceodyssey.adapters.photomancy import (
    channel_elements,
    photomancy_elements,
    photomancy_report,
    posterior_occupancy,
)
from tests.studies.bracket_helpers import fixture_channel, fixture_source

STAR, MODE, PROBE_NODE = 0, 4, 80


@pytest.fixture(scope="module")
def channel():
    return fixture_channel(fixture_source())


def _probe_rows(channel, sep_arcsec, days=(100.0, 140.0)):
    """Two visit rows of slot 0, a null and a detection, for a probe planet.

    Returns `(rows, intervals, planet, limits)`: the payloads in the layout of the
    channel, the `ScienceInterval` of each, the planet dict and the planet-free noise
    standard deviation of each row at the planet's geometry.
    """
    built = channel.assets()
    report, law = built.report, built.law
    n = channel.n_slots
    prepared = channel._prepared(STAR, nominal=True)
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
    names = jx.visit_columns(n)
    rows, intervals, limits = [], [], []
    for i, day in enumerate(days):
        duration = channel.exposures_s[MODE]
        interval = jx.ScienceInterval(
            jnp.asarray(channel.epoch_jd + day), jnp.asarray(duration)
        )
        reference = report.budget(prepared, jx.planet_free(planet), interval)
        limit = float(law.limit_counts(reference))
        row = np.zeros(len(names))
        row[names.index("detected_0")] = float(i == 1)
        row[names.index("time_d")] = float(interval.midpoint_jd)
        row[names.index("science_s")] = duration
        row[names.index("occupied_s")] = duration
        row[names.index("valid_science")] = 1.0
        row[names.index("limit_counts_0")] = limit
        if i == 1:
            row[names.index("counts_0")] = 6.0 * limit
            for c in ("ra_err_0", "dec_err_0"):
                row[names.index(c)] = 1e-3
        rows.append(row)
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
    the wrong time axis, or reads the limit from the wrong row.
    """
    data0 = photomancy_report(channel, STAR, np.zeros((2, channel.widths()[0])), 0)
    sep = float(data0.sep_grid[0, PROBE_NODE])
    rows, intervals, planet, _limits = _probe_rows(channel, sep)
    data = photomancy_report(channel, STAR, rows, 0)
    prepared = channel._prepared(STAR, nominal=True)
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
    el_phot = photomancy_elements(el_channel)
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
        predicted = channel.predict_budget(STAR, el_channel, interval)
        seen = bool(row.detected > 0.5)
        total += float(jx.loglike_report(law, row, predicted))
        if seen:
            total -= float(law.log_position_density(row, predicted, True))
    np.testing.assert_allclose(ours, total, atol=1e-3, rtol=0)


def _circular(channel, sep_lod, radius_re, albedo):
    """Photomancy elements of a face-on circular planet at `sep_lod` lambda over D."""
    prepared = channel._prepared(STAR, nominal=True)
    mass = float(prepared.star_mass_kg)
    lod = float(channel.assets().report.lod_arcsec)
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
    `channel_elements` and evaluates its own budget over each exposure; the
    photomancy side places the planet from the photomancy convention directly (the
    periastron time is the phase times the period) and reads the converted report.
    """
    el_channel = channel_elements(el_phot)
    planet, mass = channel._belief_planet(el_channel)
    prepared = channel._prepared(STAR, nominal=True, mass_kg=mass)
    built = channel.assets()
    report, law_ = built.report, built.law
    names = jx.visit_columns(channel.n_slots)
    rows, intervals = [], []
    for day, seen in zip(days, detected, strict=True):
        duration = channel.exposures_s[mode]
        interval = jx.ScienceInterval(
            jnp.asarray(channel.epoch_jd + day), jnp.asarray(duration)
        )
        expected = report.budget(prepared, planet, interval)
        free = report.budget(prepared, jx.planet_free(planet), interval)
        row = np.zeros(len(names))
        row[names.index("detected_0")] = float(seen)
        row[names.index("time_d")] = float(interval.midpoint_jd)
        row[names.index("science_s")] = duration
        row[names.index("occupied_s")] = duration
        row[names.index("valid_science")] = 1.0
        row[names.index("limit_counts_0")] = float(law_.limit_counts(free))
        if seen:
            row[names.index("counts_0")] = float(expected.planet)
            row[names.index("ra_0")] = float(expected.ra_arcsec)
            row[names.index("dec_0")] = float(expected.dec_arcsec)
            for c in ("ra_err_0", "dec_err_0"):
                row[names.index(c)] = 1e-3
        rows.append(row)
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
@pytest.mark.parametrize("radius_re", [3.5, 11.0])
def test_a_bright_planet_past_the_outer_angle_agrees_with_the_channel(
    channel, radius_re
):
    """A 3.5 or an 11 Earth-radius planet at 29 lambda/D, beyond the 28 lambda/D where
    an outer-angle grid stopped (the signal-to-noise is about 10 and 100): two
    detections agree with the channel to 0.01 nats, and the grid reaches past 40.

    The grid cut gave this detection a zero likelihood (minus infinity). The noise
    level rises by a factor of two every 0.1 lambda/D here, and the report model
    interpolates it linearly between grid nodes: on 1024 geometric nodes that cost
    0.15 nats at 3.5 Earth radii and 6.6 nats at 11. The grid places half its nodes
    where a coarse pilot sees the curve change, which brings both to about 1e-3.
    """
    lod = float(channel.assets().report.lod_arcsec)
    el = _circular(channel, 29.0, radius_re, 0.5)
    ours, theirs, data = _agreement(channel, el, MODE, (100.0, 140.0), (True, True))
    top = float(jnp.max(jnp.where(jnp.isfinite(data.dmag0_grid), data.sep_grid, 0.0)))
    assert top / lod > 40.0
    assert np.isfinite(ours) and np.isfinite(theirs)
    np.testing.assert_allclose(ours, theirs, atol=0.01, rtol=0)


@pytest.mark.medium(reason="builds the channel grid and nine budgets, 10 s cold")
def test_a_general_orbit_agrees_with_the_channel_within_the_declared_terms(channel):
    """An eccentric, inclined, phased orbit given in photomancy's convention: the
    report likelihood equals the channel's at the truth to 0.163 nats over three rows.

    The elements pass through `channel_elements` for the channel (the periastron phase
    becomes days, the scale loses the Earth-radius unit) and are placed directly in
    photomancy's convention for the belief, so a wrong time axis or phase conversion
    moves the planet and the likelihood by many nats. The bound is three rows at the
    declared terms: planet motion over a 320,000 s exposure (5.3e-2 nats per row at
    most) and grid interpolation (5e-3 at most). The two detections
    and one null sit at signal-to-noise of 4 to 11.
    """
    base = _circular(channel, 8.0, 0.45, 0.4)
    el = base.at[1].set(0.3).at[2].set(0.5).at[3].set(1.0).at[4].set(2.0).at[5].set(0.3)
    days = (85.0, 140.0, 230.0)
    ours, theirs, _ = _agreement(channel, el, 5, days, (True, False, True))
    assert np.isfinite(ours) and np.isfinite(theirs)
    np.testing.assert_allclose(ours, theirs, atol=3 * (5.3e-2 + 5e-3), rtol=0)
    # the rows are sensitive to the phase: a different periastron phase is not close
    moved, _, _ = _agreement(channel, el.at[5].set(0.8), 5, days, (True, False, True))
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
    from photomancy.orbit.updater import UpdaterStatus

    from spaceodyssey.adapters.photomancy import (
        SLOT_DEGRADED,
        SLOT_FAILED,
        SLOT_NEVER_FIT,
        SLOT_NO_SUPPORT,
        SLOT_OK,
        slot_status,
    )
    from spaceodyssey.core.enums import RefitStatus

    assert slot_status(UpdaterStatus.OK, True) == SLOT_OK
    assert slot_status(UpdaterStatus.LOW_ESS, True) == SLOT_DEGRADED
    assert slot_status(UpdaterStatus.NO_SUPPORT, True) == SLOT_NO_SUPPORT
    assert slot_status(UpdaterStatus.OK, False) == SLOT_FAILED
    codes = (SLOT_OK, SLOT_NO_SUPPORT, SLOT_DEGRADED, SLOT_FAILED)
    assert codes == tuple(int(x) for x in RefitStatus)
    assert SLOT_NEVER_FIT not in codes
