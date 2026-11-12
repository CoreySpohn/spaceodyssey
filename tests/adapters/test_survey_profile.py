"""The survey observing model: profile, per-mode reports, star flux, characterization.

The survey profile `jatis-eac1-v1` holds the JATIS baseline (Spohn et al. 2026, JATIS
12(4) 041010, Table 3) as two observing modes. The channel it builds has an imaging
detection report at 500 nm and an IFS characterization report at 910 nm.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Rearth2AU
from hwoutils.conversions import jy_to_photons_per_nm_per_m2, mag_to_flux_jy
from jaxedith import zodi_fn_ayo
from optixstuff import ExposureConfig

from spaceodyssey.adapters import jaxedith as jx
from spaceodyssey.adapters.physical_profile import (
    FIRST_SUMMARY_V1,
    JATIS_EAC1_V1,
    check_profile,
    load_profile,
)
from spaceodyssey.adapters.skyscapes import StarSpec, star_spec_from_model
from spaceodyssey.studies import bracket as br

BANDS = ((500.0, 100.0), (910.0, 6.5), (551.0, 88.0))  # detection, IFS, V
EPOCH_JD = 2462502.5


def test_profile_loads_and_is_complete():
    """The survey profile is a complete profile and holds the Table 3 values."""
    profile = load_profile(JATIS_EAC1_V1)
    assert profile["label"] == JATIS_EAC1_V1
    assert "extends" not in profile
    assert check_profile(profile) == []
    survey = profile["survey"]
    det, char = survey["modes"]["detection"], survey["modes"]["characterization"]
    assert (det["wavelength_nm"], det["bin_width_nm"], det["snr_required"]) == (
        500.0,
        100.0,
        7.0,
    )
    assert math.isclose(det["optics_throughput"], 0.362 * 0.95)
    assert (char["wavelength_nm"], char["snr_required"]) == (910.0, 5.0)
    assert math.isclose(char["bin_width_nm"], 910.0 / char["resolving_power"])
    assert math.isclose(char["optics_throughput"], 0.307 * 0.95)
    assert survey["post_processing_factor"] == 30.0
    overhead = survey["overhead_d"]
    assert math.isclose(profile["keepout"]["settling_time_d"], sum(overhead.values()))
    assert math.isclose(overhead["settling"], 1.0 / 24.0)
    assert math.isclose(overhead["coronagraph"], 2.4 / 24.0)


def test_the_first_summary_profile_is_unchanged():
    """The default profile has no survey table and keeps its settling and columns."""
    base, survey = load_profile(FIRST_SUMMARY_V1), load_profile(JATIS_EAC1_V1)
    assert load_profile() == base
    assert "survey" not in base
    assert base["keepout"]["settling_time_d"] == 1.0
    assert [c["name"] for c in base["payload_columns"]] == [
        c["name"] for c in survey["payload_columns"]
    ]
    names = [c["name"] for c in survey["conventions"]]
    assert set(c["name"] for c in base["conventions"]) <= set(names)
    assert len(names) == len(set(names))


def test_a_star_spec_reads_its_flux_by_band():
    """`flux_at` returns the band's flux; an unknown band is refused."""
    star = StarSpec("s", 1.0, 10.0, 0.0, 0.0, 1.0, 0.0, 5.0, ((500.0, 7.0),))
    assert star.flux_at() == 5.0
    assert star.flux_at(500.0) == 7.0
    with pytest.raises(ValueError, match="no flux at 910"):
        star.flux_at(910.0)


@pytest.fixture(scope="module")
def stars():
    """Two PHOENIX hosts carrying their 500 nm and 910 nm band fluxes."""
    return tuple(
        star_spec_from_model(
            name,
            mass_msun=mass,
            dist_pc=dist,
            luminosity_lsun=lum,
            teff_k=teff,
            radius_rsun=radius,
            ra_deg=270.0,
            dec_deg=66.56,
            wavelength_nm=500.0,
            bands_nm=BANDS,
        )
        for name, mass, dist, lum, teff, radius in (
            ("G dwarf", 1.0, 10.0, 1.0, 5772.0, 1.0),
            ("K dwarf", 0.8, 6.0, 0.4, 5000.0, 0.78),
        )
    )


@pytest.fixture(scope="module")
def channel(stars):
    return jx.ObservingModeChannel.from_stars(
        stars, n_slots=2, epoch_jd=EPOCH_JD, profile=JATIS_EAC1_V1, n_grid=4
    )


medium = pytest.mark.medium(reason="builds two reports on the response package")


@medium
def test_band_flux_is_the_mean_over_the_band(stars):
    """The 500 nm flux is a band mean: positive and unlike the 910 nm flux."""
    for star in stars:
        assert [c for c, _ in star.band_fluxes_jy] == [500.0, 910.0, 551.0]
        assert star.flux_at(500.0) > 0 and star.flux_at(910.0) > 0
        assert star.flux_at(500.0) != star.flux_at(910.0)
        assert star.flux_density_jy == pytest.approx(star.flux_at(500.0), rel=0.2)


@medium
def test_from_yip_defaults_to_the_header_wavelength(channel):
    """With no wavelength the report reads the header's, as before the argument."""
    default = jx.ObservingModeChannel.from_stars(
        [_spec()], n_slots=1, epoch_jd=EPOCH_JD
    )
    assert default.assets().report.wavelength_nm == pytest.approx(1000.0)
    assert default.assets().char_report is None
    assert not default.survey
    assert default.exposures_s == jx.DETECTION_EXPOSURES_S
    assert default.block_days == jx.BLOCK_DAYS
    assert float(default.block_snr_required) == jx.BLOCK_SNR_REQUIRED
    assert default.block_rate_divisor == jx.BLOCK_RATE_DIVISOR
    assert float(default.assets().law.threshold) == 5.0


def _spec():
    return StarSpec("one", 1.0, 10.0, 0.0, 0.0, 1.0, 0.0, 80.0)


@medium
def test_lambda_over_d_zodi_and_exozodi_follow_the_report_wavelength(channel, stars):
    """Each report's lambda/D, zero point and (exo)zodi are those of its wavelength."""
    built = channel.assets()
    d = channel.diameter_m
    channel_star = stars[0]
    for report, wl in ((built.report, 500.0), (built.char_report, 910.0)):
        assert report.wavelength_nm == wl
        expected = wl * 1e-9 / d * 206264.80624709636
        assert float(report.lod_arcsec) == pytest.approx(expected, rel=1e-12)
        scene = jax.tree_util.tree_map(lambda x: x[0], channel.scenes)
        prepared = report.prepare(scene, channel.leakage[0])
        zero = jy_to_photons_per_nm_per_m2(mag_to_flux_jy(0.0), wl)
        assert float(prepared.zero_point) == pytest.approx(float(zero), rel=1e-12)
        exposure = ExposureConfig(
            start_time_jd=jnp.asarray(0.0),
            exposure_time_s=jnp.asarray(0.0),
            central_wavelength_nm=jnp.asarray(wl),
            bin_width_nm=jnp.asarray(report.bin_width_nm),
            position_angle_deg=jnp.asarray(0.0),
        )
        # The host's own colour: n 10^(-0.4 22) L_V/L_V,sun (f_lambda/F0)/(f_V/F0_V),
        # written out with the V band explicit; it equals the V-free form.
        flux_ratio = float(
            jy_to_photons_per_nm_per_m2(scene.star_flux_density_jy, wl) / zero
        )
        v_mag = -2.5 * math.log10(channel_star.flux_at(551.0) / 3631.0)
        v_flux_ratio = 10.0 ** (-0.4 * v_mag)
        abs_v = v_mag - 5.0 * math.log10(float(scene.dist_pc) / 10.0)
        lum_v = 10.0 ** (-0.4 * (abs_v - 4.83))
        ez = (
            float(scene.n_exozodi)
            * 10.0 ** (-0.4 * 22.0)
            * lum_v
            * flux_ratio
            / v_flux_ratio
        )
        assert float(prepared.exozodi_1au) == pytest.approx(ez, rel=1e-9)
        # and it is not the bolometric-luminosity form of the baseline profile
        zodi = zodi_fn_ayo(report.observatory, exposure, scene)
        old_form = float(scene.n_exozodi * zodi * scene.luminosity_lsun)
        assert float(prepared.exozodi_1au) != pytest.approx(old_form, rel=0.05)
    ratio = built.report.lod_arcsec / built.char_report.lod_arcsec
    assert float(ratio) == pytest.approx(500.0 / 910.0)


@medium
def test_the_channel_holds_the_survey_declarations(channel):
    """Grids, thresholds, aperture and overhead of the survey channel."""
    grid = np.geomspace(1.0 / 24.0, 60.0, 4)
    assert channel.survey
    assert channel.n_detect == channel.n_block_modes == 4
    np.testing.assert_allclose(channel.block_days, grid)
    np.testing.assert_allclose(channel.exposures_s, grid * 86400.0)
    assert float(channel.assets().law.threshold) == 7.0
    assert float(channel.block_snr_required) == 5.0
    assert channel.block_rate_divisor == 1.0
    # 8 m inscribed, obscured area fraction 0.17481
    assert channel.diameter_m == pytest.approx(8.0 / (1.0 - 0.17481001240599225))
    assert channel.overhead_d == pytest.approx(3.4 / 24.0, abs=1e-6)
    built = channel.assets()
    assert built.report.wavelength_nm == 500.0 and built.report.ppfact == 30.0
    assert built.char_report.bin_width_nm == pytest.approx(6.5)
    assert built.char_report.detector_spec.frame_time_s == 300.0


@medium
def test_channel_table_charges_the_overhead_to_every_observation(channel):
    """Detection and characterization both occupy settling plus coronagraph overhead."""
    n = int(channel.catalog_mass_kg.shape[0])
    caps = br.default_capacities(n, channel.n_slots, n_channels=channel.n_channels)
    table = br.channel_table(channel, caps)
    per_day = int(channel.assets().context.clock.ticks_per_day)
    settle = int(channel.assets().context.settle_ticks)
    assert settle == round(3.4 / 24.0 * per_day)
    occupancy = np.asarray(table.occupancy_ticks)
    det = [settle + round(s * 1000) for s in channel.exposures_s]
    char = [settle + round(d * per_day) for d in channel.block_days]
    assert occupancy[:4].tolist() == det
    assert occupancy[4:8].tolist() == char
    cost = np.asarray(table.cost)[:8, br.SHARED_RESOURCE]
    np.testing.assert_allclose(cost, occupancy[:8] / per_day)


def _face_on_planet(scene, a_au, radius_earth=1.0, albedo=0.3):
    return jx.nominal_planet(
        a_au=a_au,
        e=0.0,
        inclination_rad=0.0,
        W_rad=0.0,
        w_rad=0.0,
        tp_d=0.0,
        Rp_au=radius_earth * Rearth2AU,
        Ag=albedo,
    )


@medium
def test_ifs_snr_matches_a_hand_calculation(channel):
    """One-channel SNR of a fixed planet: first-principles counts, hand variances.

    The planet rate is the photon flux in the 6.5 nm bin times the collecting area,
    optics, QE and counting efficiency, the contrast and the response throughput; the
    noise adds the planet, the leakage, zodi and exozodi (no detector terms, no rate
    divisor) and a systematic floor of leakage / 30.
    """
    built = channel.assets()
    report = built.char_report
    subject = 1
    prepared = channel._prepared(subject, nominal=True, char=True)
    scene = prepared.scene
    planet = _face_on_planet(scene, a_au=1.5 * math.sqrt(0.4))
    t_s = 3.0 * 3600.0
    interval = jx.ScienceInterval(jnp.asarray(EPOCH_JD), jnp.asarray(t_s))
    r = report.rates(prepared, planet, jnp.asarray([EPOCH_JD]))
    contrast = float(r.contrast[0])
    sep = float(r.separation_lod[0])
    # planet rate from first principles
    star_ph = float(jy_to_photons_per_nm_per_m2(scene.star_flux_density_jy, 910.0))
    area = math.pi / 4 * channel.diameter_m**2 * (1 - 0.17481001240599225)
    thr = float(report.optical_path.coronagraph.throughput(sep, 910.0))
    rate = star_ph * area * 0.29165 * 0.9 * 0.99 * 6.5 * contrast * thr
    assert float(r.planet[0]) == pytest.approx(rate, rel=1e-6)
    # signal-to-noise by hand from the rate components
    signal = float(r.planet[0]) * t_s
    background = float(r.leakage[0] + r.zodi[0] + r.exozodi[0]) * t_s
    floor = float(r.leakage[0]) / 30.0 * t_s
    expected = signal / math.sqrt(signal + background + floor**2)
    snr, observable = channel._characterize(prepared, planet, interval)
    assert float(snr) == pytest.approx(expected, rel=1e-6)
    assert float(observable) == 1.0
    # the belief's view: a face-on circular orbit in the log-parameter set of ELEMENTS
    a_au = 1.5 * math.sqrt(0.4)
    mass = 0.8
    period_d = math.sqrt(a_au**3 / mass) * 365.25
    row = jnp.asarray(
        [math.log(period_d), 0.0, 1.0, 0.0, 0.0, 0.0, math.log(mass), math.log(0.3)]
    )
    mode = 2
    via_channel = float(channel.block_snr(subject, mode, 0.0, row[None])[0])
    belief_planet, central = channel._belief_planet(row)
    belief = channel._prepared(subject, nominal=True, mass_kg=central, char=True)
    seconds = channel.block_days[mode] * 86400.0
    after = EPOCH_JD + channel.overhead_d + 0.5 * channel.block_days[mode]
    direct = float(
        channel._characterize(
            belief,
            belief_planet,
            jx.ScienceInterval(jnp.asarray(after), jnp.asarray(seconds)),
        )[0]
    )
    assert via_channel == pytest.approx(direct, rel=1e-9)
    assert via_channel > 0


@medium
def test_characterization_grid_and_threshold(channel):
    """SNR rises along the grid; the delivery threshold is the IFS requirement of 5."""
    subject = 1
    prepared = channel._prepared(subject, nominal=True, char=True)
    planet = _face_on_planet(prepared.scene, a_au=1.5 * math.sqrt(0.4))
    snr = [
        float(
            channel._characterize(
                prepared,
                planet,
                jx.ScienceInterval(jnp.asarray(EPOCH_JD), jnp.asarray(d * 86400.0)),
            )[0]
        )
        for d in channel.block_days
    ]
    assert snr == sorted(snr) and snr[0] < snr[-1]
    assert float(channel.block_snr_required) == 5.0
    delivered = [s >= float(channel.block_snr_required) for s in snr]
    assert delivered == sorted(delivered)


@medium
def test_solved_times_meet_their_requirements(channel):
    """The twin's detection and IFS times solve counts = 7 limit and SNR = 5."""
    subject = 0
    built = channel.assets()
    t_det = float(channel.detect_time(subject))
    t_char = float(channel.characterize_time(subject))
    assert 0 < t_det < t_char < 60.0
    assert float(channel.block_time(subject)) == pytest.approx(t_char)
    # detection: expected counts equal the threshold times the planet-free noise
    prepared = channel._prepared(subject, nominal=True)
    planet = _face_on_planet(prepared.scene, a_au=1.0)
    planet["Mp"] = jnp.zeros(())
    interval = jx.ScienceInterval(jnp.asarray(EPOCH_JD), jnp.asarray(t_det * 86400.0))
    counts = built.report.budget(prepared, planet, interval).planet
    free = built.report.budget(prepared, jx.planet_free(planet), interval)
    assert float(counts) == pytest.approx(
        7.0 * float(built.law.limit_counts(free)), rel=1e-6
    )
    # characterization
    prepared = channel._prepared(subject, nominal=True, char=True)
    interval = jx.ScienceInterval(jnp.asarray(EPOCH_JD), jnp.asarray(t_char * 86400.0))
    snr, _ = channel._characterize(prepared, planet, interval)
    assert float(snr) == pytest.approx(5.0, rel=1e-6)


@medium
def test_a_survey_channel_needs_the_band_fluxes():
    """A star without the survey bands is refused, not read at the wrong wavelength."""
    with pytest.raises(ValueError, match="no flux at 500"):
        jx.ObservingModeChannel.from_stars(
            [_spec()], n_slots=1, epoch_jd=EPOCH_JD, profile=JATIS_EAC1_V1, n_grid=2
        )


@medium
def test_the_photometric_aperture_is_that_of_the_inscribed_diameter(channel):
    """Both survey reports use a core of 0.7 lambda/D of the inscribed diameter."""
    ratio = 1.0 / (1.0 - 0.17481001240599225)
    want = math.pi * (0.7 * ratio) ** 2
    for report in (channel.assets().report, channel.assets().char_report):
        area = float(report.optical_path.coronagraph.core_area(20.0, 500.0))
        assert area == pytest.approx(want, rel=0.03)
    assert jx.survey_aperture_radius_lod(jx.survey_table(JATIS_EAC1_V1)) == (
        pytest.approx(0.7 * ratio)
    )
    assert jx.survey_aperture_radius_lod({}) is None


@medium
def test_solved_times_follow_the_closed_form_of_the_rates(channel):
    """With a noiseless detector the times are the closed form of the term rates.

    Detection: t = SNR^2 C_b / (C_p^2 - (SNR C_sp)^2) with the planet out of the noise;
    characterization: the same with the planet counted in C_b. The rates are the
    report's own planet, leakage, zodi and exozodi terms for the Earth twin.
    """
    subject = 0
    built = channel.assets()
    for report, char, snr, t_solved in (
        (built.report, False, 7.0, float(channel.detect_time(subject))),
        (built.char_report, True, 5.0, float(channel.characterize_time(subject))),
    ):
        prepared = channel._prepared(subject, nominal=True, char=char)
        planet = _face_on_planet(prepared.scene, a_au=math.sqrt(1.0))
        planet["Mp"] = jnp.asarray(5.972e24)
        r = report.rates(prepared, planet, jnp.asarray([EPOCH_JD]))
        c_p, c_sp = float(r.planet[0]), float(r.floor[0])
        c_b = float(r.leakage[0] + r.zodi[0] + r.exozodi[0])
        c_b += c_p if char else 0.0
        closed = snr**2 * c_b / (c_p**2 - (snr * c_sp) ** 2) / 86400.0
        assert t_solved == pytest.approx(closed, rel=1e-3)


@medium
def test_a_requirement_above_the_floor_limit_has_infinite_time(channel):
    """Where the closed-form denominator is not positive the solved time is infinite.

    The signal-to-noise ratio of a noiseless detector saturates at C_p / C_sp, so a
    requirement above it has `C_p^2 - (SNR C_sp)^2 <= 0` and no finite time; one below
    it has the closed-form time.
    """
    built = channel.assets()
    prepared = channel._prepared(0, nominal=True, char=True)
    planet = _face_on_planet(prepared.scene, a_au=math.sqrt(1.0))
    planet["Mp"] = jnp.asarray(5.972e24)
    r = built.char_report.rates(prepared, planet, jnp.asarray([EPOCH_JD]))
    c_p, c_sp = float(r.planet[0]), float(r.floor[0])
    limit = c_p / c_sp

    def solve(snr):
        def short(budget, _):
            return snr - jx._scaled_snr(budget, 1.0)

        return float(channel._twin_time(0, built.char_report, prepared, short))

    assert math.isinf(solve(1.5 * limit))
    assert math.isfinite(solve(0.5 * limit))
