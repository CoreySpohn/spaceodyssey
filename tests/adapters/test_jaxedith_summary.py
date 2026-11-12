"""Summary radiometry through jaxedith: hand budgets, parity, refinement, anchors.

Unit tests run on an analytic toy coronagraph, so every expected value is written by
hand. Medium and slow tests use the pinned public response package. The library
anchors (QE and dQE applied once, the fixed-frame detector variance and SNR, the
primary built from the package header) run against the pinned library commits.
"""

import itertools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from coronalyze import PPConfig
from hwoutils.constants import Jy, Rearth2AU, h
from hwoutils.conversions import AB_ZERO_POINT_JY, Msun_to_kg
from jaxedith import (
    ETCScene,
    count_rates_ayo,
    count_rates_from_system_ayo,
    exptime_ayo,
    snr_ayo,
    zodi_fn_leinert,
)
from optixstuff import (
    AbstractScalarCoronagraph,
    ConstantThroughput,
    ExposureConfig,
    OpticalPath,
    SimplePrimary,
)
from orbix.kepler.shortcuts.grid import get_grid_solver
from orbix.orbit import KeplerianOrbit
from photomancy.orbit.forward import predict_photometry, predict_relative_astrometry
from scipy.ndimage import map_coordinates
from skyscapes.physical_model import GridPhysicalModel
from skyscapes.scene import Planet, Star, System
from yippy.datasets import CATALOG

from spaceodyssey.adapters.jaxedith import (
    CADENCE_CODES,
    YIP_DIGEST,
    DetectorSpec,
    ScienceInterval,
    SummaryReport,
    SummaryScene,
    SummaryTruth,
    fetch_pinned_yip,
    nominal_planet,
)
from tests.adapters.conftest import EPOCH_JD, uma_scene

WAVELENGTH_NM = 1000.0
BAND_NM = 100.0
DIAMETER_M = 7.2
OBSCURED = 0.17  # toy obscured area fraction
OPTICS = 0.5
LEGACY_PHOTON_COUNTING = 6.73  # jaxedith's legacy photon-counting constant


class ToyCoronagraph(AbstractScalarCoronagraph):
    """Analytic response: smooth throughput rise, constant aperture, linear leakage."""

    pixel_scale_lod: float = 0.25
    IWA: float = 3.0
    OWA: float = 32.0
    peak: float = 0.3
    area_lod2: float = 1.5
    leak: float = 1e-10
    sky: float = 0.5

    def throughput(self, separation_lod, wavelength_nm, *, time_s=0.0):
        return self.peak * (1.0 - jnp.exp(-((separation_lod / self.IWA) ** 2)))

    def core_area(self, separation_lod, wavelength_nm, *, time_s=0.0):
        return self.area_lod2 * jnp.ones_like(separation_lod)

    def core_mean_intensity(
        self, separation_lod, wavelength_nm, *, time_s=0.0, stellar_diam_lod=0.0
    ):
        return self.leak * (1.0 + 0.1 * separation_lod)

    def occulter_transmission(self, separation_lod, wavelength_nm, *, time_s=0.0):
        return self.sky * jnp.ones_like(separation_lod)


def toy_scene(**overrides):
    fields = dict(
        star_flux_density_jy=42.0,
        dist_pc=10.0,
        star_mass_kg=Msun_to_kg(1.0),
        luminosity_lsun=1.5,
        ra_deg=164.87,
        dec_deg=40.43,
        star_diameter_mas=0.0,
        n_exozodi=3.0,
    )
    fields.update(overrides)
    return SummaryScene(**fields)


def toy_report(observatory, *, spec=None, scene=None, coronagraph=None, **kwargs):
    spec = DetectorSpec.fixture() if spec is None else spec
    path = OpticalPath(
        primary=SimplePrimary(DIAMETER_M, math.sqrt(OBSCURED)),
        attenuating_elements=(ConstantThroughput(throughput=OPTICS, name="optics"),),
        coronagraph=ToyCoronagraph() if coronagraph is None else coronagraph,
        detector=spec.detector(),
    )
    options = dict(
        wavelength_nm=WAVELENGTH_NM,
        bin_width_nm=BAND_NM,
        orbit_epoch_jd=EPOCH_JD,
        bg_multiplier=1.0,
        ppfact=math.inf,
        eps_warm_T_cold=0.0,
        n_time_nodes=8,
        field_radius_lod=32.0,
        leakage_step_lod=0.0625,
    )
    options.update(kwargs)
    return SummaryReport(
        path, observatory, spec, toy_scene() if scene is None else scene, **options
    )


def face_on(a_au, *, Rp_au=1e-4, Ag=0.3):
    """A face-on circular orbit: constant separation and a 90 degree phase angle."""
    return nominal_planet(
        a_au=a_au,
        e=0.0,
        inclination_rad=0.0,
        W_rad=0.0,
        w_rad=0.0,
        tp_d=0.0,
        Rp_au=Rp_au,
        Ag=Ag,
    )


def lod_arcsec():
    return WAVELENGTH_NM * 1e-9 / DIAMETER_M * 206264.80624709636


def hand_rates(report, scene, a_au, Lambda, times_jd, *, ppfact=math.inf):
    """Every rate of a face-on circular planet against the toy, written out."""
    coro = ToyCoronagraph()
    zero_point = AB_ZERO_POINT_JY * Jy / (WAVELENGTH_NM * h)
    star = scene.star_flux_density_jy / AB_ZERO_POINT_JY
    area = math.pi * (DIAMETER_M / 2) ** 2 * (1.0 - OBSCURED)
    common = zero_point * area * OPTICS * BAND_NM
    sep_arcsec = a_au / scene.dist_pc
    sep = sep_arcsec / lod_arcsec()
    contrast = Lambda / math.pi / a_au**2  # Lambert phase at 90 degrees is 1/pi
    planet = common * star * contrast * float(coro.throughput(sep, 0.0))
    intensity = coro.leak * (1.0 + 0.1 * sep)
    leakage = common * star * intensity * coro.area_lod2
    aperture = lod_arcsec() ** 2 * coro.area_lod2 * coro.sky
    fzodi = np.array(
        [
            zodi_fn_leinert(report.observatory, report._exposure(t), scene)
            for t in times_jd
        ]
    )
    zodi = common * fzodi * aperture
    # One zodi is 22 mag/arcsec^2 in V at 1 AU around a star of the Sun's V-band
    # luminosity; the host's own colour makes it that host's flux density at the
    # report wavelength scaled to 10 pc.
    one_zodi_host = 10.0 ** (-0.4 * 22.0) * 10.0 ** (0.4 * 4.83)
    host = one_zodi_host * (scene.dist_pc / 10.0) ** 2 * star
    exozodi = common * scene.n_exozodi * host * aperture
    exozodi /= a_au**2  # surface brightness at 1 AU scaled to the physical radius
    spec = report.detector_spec
    n_pix = coro.area_lod2 / (spec.pixel_scale_arcsec / lod_arcsec()) ** 2
    photons = planet + leakage + zodi + exozodi
    detector = n_pix * spec.dark_current_rate_e_per_s + (
        spec.clock_induced_charge_rate_e_per_frame * LEGACY_PHOTON_COUNTING * photons
    )
    floor = common * star * intensity * coro.area_lod2 / ppfact
    return dict(
        planet=planet,
        leakage=leakage,
        zodi=zodi,
        exozodi=exozodi,
        detector=detector,
        floor=floor,
    )


@pytest.mark.medium(reason="over 3 s cold here, about 5 s on the runner")
def test_detector_spec_and_report_refuse_undeclared_choices(observatory):
    """Fails if a detector without a name, family, dQE meaning or cadence is accepted.

    Also fails if the fixed-frame cadence runs on a jaxedith that has no finite-frame
    variance (or is refused on one that has it), or if a non-finite or out-of-range
    hardware number passes.
    """
    bad = [
        dict(name=""),
        dict(name=None),
        dict(family="  "),
        dict(dqe_meaning="additive"),
        dict(cadence="photon_counting_ayo"),
        dict(quantum_efficiency=float("nan")),
        dict(quantum_efficiency=0.0),
        dict(dqe=1.2),
        dict(dark_current_rate_e_per_s=-1e-5),
        dict(frame_time_s=0.0),
        dict(pixel_scale_arcsec=float("inf")),
        dict(read_noise_e=True),
    ]
    for override in bad:
        with pytest.raises(ValueError):
            DetectorSpec.fixture(**override)
    spec = DetectorSpec.fixture()
    assert spec.cadence == "fixed_frame"
    assert spec.cadence_code == CADENCE_CODES["fixed_frame"] == 0
    legacy = DetectorSpec.fixture(cadence="photon_counting_legacy")
    assert legacy.cadence_code == CADENCE_CODES["photon_counting_legacy"] == 1
    interval = ScienceInterval.from_start(EPOCH_JD + 10.0, 3600.0)
    for declared in (spec, legacy):
        budget = toy_report(observatory, spec=declared).predict(face_on(1.5), interval)
        assert int(budget.cadence) == declared.cadence_code
    for kwargs in (
        dict(n_time_nodes=0),
        dict(bg_multiplier=0.0),
        dict(ppfact=-1.0),
        dict(ppfact=float("nan")),
    ):
        with pytest.raises(ValueError):
            toy_report(observatory, **kwargs)


@pytest.mark.medium(reason="over 3 s cold here, about 5 s on the runner")
def test_hand_count_and_variance_budget(observatory):
    """Fails if any count or variance component departs from its hand formula.

    The planet is on a face-on circular orbit (constant separation, Lambert phase at
    90 degrees), the detector has QE 1 and dQE 1 so the budget does not depend on
    where QE is applied and runs in the legacy cadence (the fixed-frame detector
    term has its own anchor), and the collecting area is pi (D/2)^2 (1 - OBSCURED). Also
    fails if a component is dropped from the variance, if the background multiplier
    is not applied, or if the floor is not (t * leakage / ppfact)^2. A zero-length
    interval has zero counts, zero variance, SNR 0 and a finite centroid (not NaN).
    """
    spec = DetectorSpec.fixture(
        quantum_efficiency=1.0, dqe=1.0, cadence="photon_counting_legacy"
    )
    scene = toy_scene()
    a_au, Rp_au, Ag = 1.5, 1e-4, 0.3
    interval = ScienceInterval.from_start(EPOCH_JD + 120.0, 7200.0)
    for k, ppfact in ((1.0, math.inf), (2.0, 30.0)):
        report = toy_report(observatory, spec=spec, bg_multiplier=k, ppfact=ppfact)
        theta = face_on(a_au, Rp_au=Rp_au, Ag=Ag)
        budget = report.predict(theta, interval)
        times = np.asarray(report.node_times_jd(interval))
        rates = hand_rates(report, scene, a_au, Ag * Rp_au**2, times, ppfact=ppfact)
        w = np.asarray(report.weights) * 0.5 * 7200.0
        hand = {key: float(np.sum(w * value)) for key, value in rates.items()}
        for key in ("planet", "leakage", "zodi", "exozodi"):
            np.testing.assert_allclose(getattr(budget, key), hand[key], rtol=1e-10)
        background = hand["leakage"] + hand["zodi"] + hand["exozodi"]
        np.testing.assert_allclose(budget.var_background, k * background, rtol=1e-10)
        np.testing.assert_allclose(
            budget.var_detector, k * hand["detector"], rtol=1e-10
        )
        np.testing.assert_allclose(
            budget.var_systematic, hand["floor"] ** 2, rtol=1e-10
        )
        total = (
            hand["planet"] + k * (background + hand["detector"]) + hand["floor"] ** 2
        )
        np.testing.assert_allclose(budget.variance, total, rtol=1e-10)
        np.testing.assert_allclose(budget.snr, hand["planet"] / math.sqrt(total))
        assert float(budget.thermal) == 0.0
        assert int(budget.cadence) == CADENCE_CODES["photon_counting_legacy"]
        assert bool(budget.in_field)
        empty = report.predict(theta, ScienceInterval(interval.midpoint_jd, 0.0))
        assert float(empty.planet) == float(empty.variance) == float(empty.snr) == 0.0
        assert np.isfinite(empty.ra_arcsec) and np.isfinite(empty.dec_arcsec)
    # Control: a raw area fraction as the linear obscuration overstates the area.
    raw = SimplePrimary(DIAMETER_M, OBSCURED).area_m2
    assert raw / report.optical_path.primary.area_m2 - 1.0 > 0.1


def test_exozodi_scales_with_physical_radius_host_flux_and_aperture(observatory):
    """Fails if the exozodi rate is not n_ez F_host / r_AU^2 per aperture.

    At a fixed host the rate falls as 1/r_AU^2; it is linear in the host's flux
    density and in the aperture solid angle, and the host's own colour sets it: the
    catalog luminosity does not enter, and at a fixed flux density a host moved
    to twice the distance is four times as luminous. A mutant that scales by the angular
    separation instead of the physical radius changes with distance.
    """
    interval = ScienceInterval.from_start(EPOCH_JD + 30.0, 3600.0)

    def exozodi(a_au, *, area_lod2=1.5, **scene):
        report = toy_report(
            observatory,
            scene=toy_scene(**scene),
            coronagraph=ToyCoronagraph(area_lod2=area_lod2),
        )
        return float(report.predict(face_on(a_au), interval).exozodi)

    base = exozodi(1.0)
    np.testing.assert_allclose(exozodi(2.0), base / 4.0, rtol=1e-12)
    np.testing.assert_allclose(exozodi(1.0, luminosity_lsun=3.0), base, rtol=1e-12)
    np.testing.assert_allclose(
        exozodi(1.0, star_flux_density_jy=84.0), 2.0 * base, rtol=1e-12
    )
    np.testing.assert_allclose(exozodi(1.0, n_exozodi=6.0), 2.0 * base, rtol=1e-12)
    np.testing.assert_allclose(exozodi(1.0, area_lod2=3.0), 2.0 * base, rtol=1e-12)
    np.testing.assert_allclose(exozodi(1.0, dist_pc=20.0), 4.0 * base, rtol=1e-12)
    # Mutant: separation in arcsec in place of the physical radius in AU.
    assert abs((base * 10.0**2) / (base * 20.0**2) - 1.0) > 0.5


def test_expected_and_predict_share_one_function_and_split_components(observatory):
    """Fails if predict reads the truth scene, or the two paths integrate differently.

    The truth scene here has more exozodi and a brighter star than the nominal one:
    `predict` must not see either, `expected` must see both, and with the nominal
    scene as truth the two are equal bit for bit. Doubling the exozodi changes the
    exozodi component and leaves the planet and leakage components unchanged.
    """
    report = toy_report(observatory)
    theta = face_on(1.2)
    interval = ScienceInterval.from_start(EPOCH_JD + 60.0, 86400.0)
    predicted = report.predict(theta, interval)
    same = report.expected(SummaryTruth(toy_scene(), theta), interval)
    for a, b in zip(jax.tree.leaves(predicted), jax.tree.leaves(same), strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    truth = SummaryTruth(toy_scene(n_exozodi=6.0, star_flux_density_jy=84.0), theta)
    true = report.expected(truth, interval)
    # twice the dust around a star twice as bright: four times the exozodi
    np.testing.assert_allclose(true.exozodi, 4.0 * predicted.exozodi, rtol=1e-12)
    np.testing.assert_allclose(true.planet, 2.0 * predicted.planet, rtol=1e-12)
    np.testing.assert_allclose(true.leakage, 2.0 * predicted.leakage, rtol=1e-12)
    np.testing.assert_allclose(true.zodi, predicted.zodi, rtol=1e-12)
    dusty = report.expected(SummaryTruth(toy_scene(n_exozodi=6.0), theta), interval)
    np.testing.assert_allclose(dusty.exozodi, 2.0 * predicted.exozodi, rtol=1e-12)
    assert float(dusty.planet) == float(predicted.planet)
    assert float(dusty.leakage) == float(predicted.leakage)


def test_moving_planet_counts_and_centroid_match_a_fine_grid(observatory):
    """Fails if the counts or the centroid are not the time integral over the interval.

    An edge-on planet crosses the toy's throughput rise during a two-day exposure.
    The default eight nodes match a 4001-point Simpson integral of the planet rate
    written by hand from photomancy's geometry and the toy throughput, to 1e-6; the
    centroid
    is weighted by detected planet rate, so a mutant that averages positions in
    time, or one that evaluates the midpoint only, misses by more than the match.
    """
    scene = toy_scene(dist_pc=3.0)
    spec = DetectorSpec.fixture(quantum_efficiency=1.0, dqe=1.0)
    report = toy_report(observatory, spec=spec, scene=scene)
    theta = nominal_planet(
        a_au=0.5,
        e=0.0,
        inclination_rad=0.5 * math.pi,
        W_rad=0.3,
        w_rad=0.0,
        tp_d=0.0,
        Rp_au=1e-4,
        Ag=0.3,
    )
    # Approaching conjunction the separation sweeps down through the throughput rise.
    period_d = 365.25 * 0.5**1.5
    interval = ScienceInterval.from_start(EPOCH_JD + 0.158 * period_d, 2 * 86400.0)
    budget = report.predict(theta, interval)
    # Reference: photomancy's geometry and the toy throughput on a fine grid.
    start = interval.midpoint_jd - interval.duration_s / 2 / 86400.0
    t_d = start + np.linspace(0.0, 2.0, 4001) - EPOCH_JD
    orbit = [theta[k] for k in ("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp")]
    mass, dist = scene.star_mass_kg, scene.dist_pc
    alpha, dmag = predict_photometry(t_d, *orbit, mass, theta["Lambda"], dist)
    ra, _ = predict_relative_astrometry(t_d, *orbit, mass, dist)
    sep = np.asarray(alpha) / lod_arcsec()
    assert sep.max() - sep.min() > 0.4  # the fixture moves across the response
    zero_point = AB_ZERO_POINT_JY * Jy / (WAVELENGTH_NM * h)
    star = scene.star_flux_density_jy / AB_ZERO_POINT_JY
    area = math.pi * (DIAMETER_M / 2) ** 2 * (1.0 - OBSCURED)
    planet = (
        zero_point * star * area * OPTICS * BAND_NM
        * 10.0 ** (-0.4 * np.asarray(dmag))
        * np.asarray(ToyCoronagraph().throughput(sep, 0.0))
    )  # fmt: skip
    simpson = (
        np.array([1.0] + [4.0, 2.0] * 1999 + [4.0, 1.0]) * (2 * 86400.0 / 4000) / 3
    )
    np.testing.assert_allclose(budget.planet, np.sum(simpson * planet), rtol=1e-6)
    ra = np.asarray(ra)
    centroid = np.sum(simpson * planet * ra) / np.sum(simpson * planet)
    np.testing.assert_allclose(budget.ra_arcsec, centroid, rtol=1e-6, atol=1e-12)
    time_mean = np.sum(simpson * ra) / np.sum(simpson)
    assert abs(time_mean - centroid) > 1e3 * abs(float(budget.ra_arcsec) - centroid)
    midpoint = planet[2000]  # the fine grid's middle sample is the midpoint
    assert abs(midpoint * 2 * 86400.0 / float(budget.planet) - 1) > 1e-3


# -- Library anchors ----------------------------------------------------------


def test_quantum_efficiency_applies_exactly_once(observatory):
    """Fails if the planet and photon background rates do not scale with QE once."""
    interval = ScienceInterval.from_start(EPOCH_JD + 120.0, 3600.0)
    unit = toy_report(observatory, spec=DetectorSpec.fixture(quantum_efficiency=1.0))
    half = toy_report(observatory, spec=DetectorSpec.fixture(quantum_efficiency=0.5))
    a = unit.predict(face_on(1.5), interval)
    b = half.predict(face_on(1.5), interval)
    for key in ("planet", "leakage", "zodi", "exozodi"):
        np.testing.assert_allclose(getattr(b, key), 0.5 * getattr(a, key), rtol=1e-12)


def test_dqe_is_a_survival_fraction(observatory):
    """Fails if a 0.8 dQE does not scale planet and photon backgrounds by 0.8 once."""
    interval = ScienceInterval.from_start(EPOCH_JD + 120.0, 3600.0)
    neutral = toy_report(observatory, spec=DetectorSpec.fixture(dqe=1.0))
    worn = toy_report(observatory, spec=DetectorSpec.fixture(dqe=0.8))
    a = neutral.predict(face_on(1.5), interval)
    b = worn.predict(face_on(1.5), interval)
    for key in ("planet", "leakage", "zodi", "exozodi"):
        np.testing.assert_allclose(getattr(b, key), 0.8 * getattr(a, key), rtol=1e-12)


def test_fixed_frame_detector_variance(observatory):
    """Fails if the detector variance is not n_pix (dark t + (RN^2 + CIC) ceil(t/T)).

    Sub-frame, exactly one frame, exactly n frames and n frames plus a partial last
    frame; read noise and clock-induced charge are paid per frame, dark current per
    second, and the clock-induced charge does not depend on the source rate.
    """
    from jaxedith import detector_variance_counts  # noqa: F401

    spec = DetectorSpec.fixture(cadence="fixed_frame", read_noise_e=0.5)
    report = toy_report(observatory, spec=spec)
    n_pix = 1.5 / (spec.pixel_scale_arcsec / lod_arcsec()) ** 2
    for duration_s, frames in ((400.0, 1), (1000.0, 1), (3000.0, 3), (3500.0, 4)):
        interval = ScienceInterval.from_start(EPOCH_JD + 10.0, duration_s)
        budget = report.predict(face_on(1.5), interval)
        hand = n_pix * (
            spec.dark_current_rate_e_per_s * duration_s
            + (spec.read_noise_e**2 + spec.clock_induced_charge_rate_e_per_frame)
            * frames
        )
        np.testing.assert_allclose(budget.var_detector, hand, rtol=1e-12)


@pytest.mark.medium(reason="over 3 s cold here, about 5 s on the runner")
def test_fixed_frame_floor_agrees_with_jaxedith(observatory):
    """Fails if the fixed-frame SNR differs from jaxedith's with the floor off or on."""
    from jaxedith import count_variance_components_ayo

    spec = DetectorSpec.fixture(cadence="fixed_frame")
    for ppfact in (math.inf, 30.0):
        report = toy_report(observatory, spec=spec, ppfact=ppfact, n_time_nodes=1)
        interval = ScienceInterval.from_start(EPOCH_JD + 10.0, 4000.0)
        budget = report.predict(face_on(1.5), interval)
        rates = report.rates(report.nominal, face_on(1.5), interval.midpoint_jd)
        scene = etc_scene(report, rates, 0, interval.midpoint_jd)
        components = count_variance_components_ayo(
            report.optical_path,
            scene,
            WAVELENGTH_NM,
            rates.separation_lod[0],
            BAND_NM,
            4000.0,
            cadence="fixed_frame",
            bg_multiplier=1.0,
            ppfact=ppfact,
        )
        np.testing.assert_allclose(budget.variance, components["total"], rtol=1e-6)


def test_primary_from_yip_matches_the_adapter_area():
    """Fails if the library's YIP primary differs from pi (D/2)^2 (1 - OBSCURED)."""
    primary = SimplePrimary.from_yip(fetch_pinned_yip())
    obscured, diameter = header_obscuration_and_diameter()
    np.testing.assert_allclose(
        primary.area_m2, math.pi * (diameter / 2) ** 2 * (1.0 - obscured), rtol=1e-12
    )


# -- The pinned public response -----------------------------------------------


def header_obscuration_and_diameter():
    """OBSCURED and D read from the package's stellar intensity header."""
    with fits.open(f"{fetch_pinned_yip()}/stellar_intens.fits") as hdul:
        header = hdul[0].header
        return float(header["OBSCURED"]), float(header["D"])


def etc_scene(report, rates, i, epoch_jd):
    """The jaxedith scene of node `i` at `epoch_jd`, from the cached factors."""
    prepared = report.nominal
    return ETCScene(
        F0=prepared.zero_point,
        Fs_over_F0=prepared.star_ratio,
        Fp_over_Fs=rates.contrast[i],
        Fzodi=zodi_fn_leinert(
            report.observatory, report._exposure(epoch_jd), prepared.scene
        ),
        Fexozodi=prepared.exozodi_1au,
        dist_pc=prepared.dist_pc,
        sep_arcsec=rates.separation_lod[i] * report.lod_arcsec,
        Fbinary=0.0,
        stellar_diam_mas=prepared.scene.star_diameter_mas,
    )


def full_call_rates(report, theta, times_jd):
    """jaxedith's full scalar public at every node, with nothing cached."""
    rates = report.rates(report.nominal, theta, times_jd)
    prepared = report.nominal

    def one(contrast, sep_lod, t):
        scene = ETCScene(
            F0=prepared.zero_point,
            Fs_over_F0=prepared.star_ratio,
            Fp_over_Fs=contrast,
            Fzodi=zodi_fn_leinert(
                report.observatory, report._exposure(t), prepared.scene
            ),
            Fexozodi=prepared.exozodi_1au,
            dist_pc=prepared.dist_pc,
            sep_arcsec=sep_lod * report.lod_arcsec,
            Fbinary=0.0,
            stellar_diam_mas=prepared.scene.star_diameter_mas,
        )
        return count_rates_ayo(
            report.optical_path,
            scene,
            report.wavelength_nm,
            sep_lod,
            report.bin_width_nm,
            temp_K=report.observatory.temperature_K,
            ez_ppf=jnp.inf,
            ppfact=report.ppfact,
            eps_warm_T_cold=0.0,
        )

    full = jax.vmap(one)(rates.contrast, rates.separation_lod, times_jd)
    return rates, full


def random_proposals(n, rng, *, dist_pc):
    """Orbits whose separations span the inner working angle to about 7 lambda/D.

    Returned as one batch: each parameter is an array over proposals.
    """
    return nominal_planet(
        a_au=rng.uniform(3.0, 7.0, n) * lod_arcsec() * dist_pc,
        e=rng.uniform(0.0, 0.4, n),
        inclination_rad=np.arccos(rng.uniform(-1, 1, n)),
        W_rad=rng.uniform(0, 2 * np.pi, n),
        w_rad=rng.uniform(0, 2 * np.pi, n),
        tp_d=rng.uniform(0, 4000.0, n),
        Rp_au=11.2 * Rearth2AU,
        Ag=rng.uniform(0.1, 0.6, n),
    )


@eqx.filter_jit
def cached_and_full(report, thetas, interval):
    """Cached rates, the full call and the integrated counts for a batch of orbits."""
    times = report.node_times_jd(interval)

    def one(theta):
        rates, full = full_call_rates(report, theta, times)
        return rates, full, report.predict(theta, interval).planet

    return jax.vmap(one)(thetas)


def relative_error(candidate, reference):
    """Largest relative difference per proposal (leading axis)."""
    ratio = np.asarray(candidate) / np.asarray(reference)
    return np.abs(ratio - 1.0).reshape(ratio.shape[0], -1).max(axis=1)


@pytest.mark.slow(
    reason="the response package loads in 4 s cold, then jaxedith compiles"
)
def test_cached_rates_match_the_full_jaxedith_call_near_the_inner_working_angle(
    yip_report,
):
    """Fails if the cached evaluation departs from jaxedith's full call at any node.

    Twenty-four random orbit proposals (seed 0), integrated over a two-day exposure
    with eight nodes, at least four of them within 1 lambda/D of the inner working
    angle, are compared with jaxedith's scalar public `count_rates_ayo` called per
    node with nothing cached: the planet rate, the total background including the
    detector, the floor rate (ppfact 30) and the integrated counts agree to the
    declared cache tolerance of 1e-9. A single-coefficient mutant, which scales one
    per-visit counts-per-contrast coefficient taken at the planned geometry, misses
    by more than 1 percent at every other near-IWA proposal.
    """
    report = eqx.tree_at(lambda r: r.ppfact, yip_report, 30.0)
    thetas = random_proposals(24, np.random.default_rng(0), dist_pc=13.8)
    interval = ScienceInterval.from_start(EPOCH_JD + 400.0, 2 * 86400.0)
    rates, (cp, cb, cnf), planet = cached_and_full(report, thetas, interval)
    keep = np.all(np.asarray(rates.in_field), axis=1)
    assert keep.sum() >= 16
    background = rates.leakage + rates.zodi + rates.exozodi + rates.thermal
    w = np.asarray(report.weights) * 0.5 * interval.duration_s
    for cached, full in (
        (rates.planet, cp),
        (background + rates.detector, cb),
        (rates.floor, cnf),
        (planet, np.sum(w * np.asarray(cp), axis=1)),
    ):
        assert np.max(relative_error(cached, full)[keep]) < 1e-9
    iwa = float(report.optical_path.coronagraph.IWA)
    sep = np.asarray(rates.separation_lod)
    near = np.flatnonzero(keep & (np.min(np.abs(sep - iwa), axis=1) < 1.0))
    assert near.size >= 4
    # Mutant: counts per unit contrast fixed at the planned (first near-IWA) orbit.
    planned = near[0]
    coefficient = np.asarray(rates.planet[planned] / rates.contrast[planned])
    mutant = np.sum(w * coefficient * np.asarray(rates.contrast), axis=1)
    mutant_error = np.abs(mutant / np.asarray(planet) - 1.0)
    assert np.min(mutant_error[near[1:]]) > 1e-2


@pytest.mark.slow(reason="shares the response package fixture; cold jaxedith programs")
def test_variance_agrees_with_jaxedith_and_the_system_wrapper(yip_report, observatory):
    """Fails if the summary SNR or rates disagree with jaxedith's own entry points.

    With one node and a stationary planet the budget's SNR equals jaxedith's
    `snr_ayo` to 1e-6 with the floor off (ppfact infinite) and on (ppfact 30), in
    the fixed-frame cadence (the default) and the legacy one; the SNR at jaxedith's
    `exptime_ayo` for SNR 10 is 10 (fixed frames planned to the minimum duration;
    whole-frame planning reaches at least 10, and one frame less falls short). The
    rates of a face-on circular planet equal `count_rates_from_system_ayo` on an
    equivalent skyscapes system, which builds the zero point, the stellar ratio, the
    local zodi and the stellar diameter itself, to 1e-5 (the system's own Kepler
    solve); a 1 percent zero-point or stellar-flux error fails. The collecting area
    is pi (D/2)^2 (1 - OBSCURED) from the package header; the raw fraction as the
    obscuration overstates it by over 10 percent.
    """
    obscured, diameter = header_obscuration_and_diameter()
    area = yip_report.optical_path.primary.area_m2
    np.testing.assert_allclose(
        area, math.pi * (diameter / 2) ** 2 * (1 - obscured), rtol=1e-12
    )
    assert SimplePrimary(diameter, obscured).area_m2 / area - 1 > 0.1

    a_au, Lambda = 2.1, 0.3 * (11.2 * Rearth2AU) ** 2
    theta = face_on(a_au, Rp_au=11.2 * Rearth2AU, Ag=0.3)
    start = EPOCH_JD + 200.0
    legacy_report = SummaryReport.from_yip(
        DetectorSpec.fixture(cadence="photon_counting_legacy"),
        uma_scene(),
        orbit_epoch_jd=EPOCH_JD,
        observatory=observatory,
    )
    cadences = ((yip_report, "fixed_frame"), (legacy_report, "photon_counting_ayo"))
    for (base, cadence), ppfact in itertools.product(cadences, (math.inf, 30.0)):
        report = eqx.tree_at(
            lambda r: (r.ppfact, r.nodes, r.weights),
            base,
            (ppfact, jnp.zeros(1), jnp.full(1, 2.0)),
        )
        rates = report.rates(report.nominal, theta, start)
        budget = report.predict(theta, ScienceInterval(start, 36500.0))
        scene = etc_scene(report, rates, 0, start)
        args = (report.optical_path, scene, WAVELENGTH_NM, rates.separation_lod[0])
        options = dict(
            temp_K=observatory.temperature_K,
            ez_ppf=jnp.inf,
            ppfact=ppfact,
            bg_multiplier=1.0,
            eps_warm_T_cold=0.0,
            cadence=cadence,
        )
        snr = snr_ayo(*args, BAND_NM, 36500.0, **options)
        np.testing.assert_allclose(budget.snr, snr, rtol=1e-6)
        if cadence == "fixed_frame":
            options["planning"] = "min_duration"
        t10 = exptime_ayo(*args, BAND_NM, 10.0, **options)
        planned = report.predict(theta, ScienceInterval(start, t10))
        np.testing.assert_allclose(planned.snr, 10.0, rtol=1e-6)
        if cadence == "fixed_frame":
            options["planning"] = "whole_frames"
            whole = float(exptime_ayo(*args, BAND_NM, 10.0, **options))
            frame = report.detector_spec.frame_time_s
            assert whole % frame == pytest.approx(0.0, abs=1e-6)
            above = report.predict(theta, ScienceInterval(start, whole))
            below = report.predict(theta, ScienceInterval(start, whole - frame))
            assert float(above.snr) >= 10.0 * (1 - 1e-9) > float(below.snr)

    star = Star(
        Ms_kg=Msun_to_kg(1.03),
        dist_pc=13.8,
        ra_deg=164.867,
        dec_deg=40.430,
        diameter_arcsec=0.79e-3,
        luminosity_lsun=1.54,
        wavelengths_nm=jnp.array([900.0, WAVELENGTH_NM, 1100.0]),
        times_jd=jnp.array([start - 10.0, start + 10.0]),
        flux_density_jy=jnp.full((3, 2), 42.0),
    )
    orbit = KeplerianOrbit(
        a_AU=jnp.array([a_au]),
        e=jnp.zeros(1),
        W_rad=jnp.zeros(1),
        i_rad=jnp.zeros(1),
        w_rad=jnp.zeros(1),
        M0_rad=jnp.zeros(1),
        t0_d=jnp.zeros(1),
    )
    planet = Planet(
        Rp_Rearth=jnp.ones(1),
        Mp_Mearth=jnp.ones(1),
        orbit=orbit,
        physical_model=GridPhysicalModel(
            wavelengths_nm=jnp.array([900.0, 1100.0]),
            phase_angle_deg=jnp.array([0.0, 180.0]),
            contrast_grid=jnp.full((1, 2, 2), Lambda / math.pi / a_au**2),
        ),
    )
    system = System(
        star=star,
        planets=(planet,),
        trig_solver=get_grid_solver(level="scalar", E=False, trig=True, jit=True),
    )
    exposure = ExposureConfig(
        start_time_jd=jnp.array([start]),
        exposure_time_s=jnp.asarray(3600.0),
        central_wavelength_nm=jnp.asarray(WAVELENGTH_NM),
        bin_width_nm=jnp.asarray(BAND_NM),
        position_angle_deg=jnp.asarray(0.0),
    )
    cp, cb, _ = count_rates_from_system_ayo(
        system,
        yip_report.optical_path,
        observatory,
        exposure,
        PPConfig(ppfact=math.inf, n_rolls=1, ez_ppf=jnp.inf),
        zodi_fn=zodi_fn_leinert,
        Fexozodi=yip_report.nominal.exozodi_1au,
        eps_warm_T_cold=0.0,
    )
    rates = yip_report.rates(yip_report.nominal, theta, start)
    background = rates.leakage + rates.zodi + rates.exozodi + rates.detector
    # skyscapes' grid Kepler solver places the planet within 1e-6 of a / d.
    np.testing.assert_allclose(rates.planet[0], cp[0, 0], rtol=1e-5)
    np.testing.assert_allclose(background[0], cb[0, 0], rtol=1e-5)


# -- Leakage operator (P3) ---------------------------------------------------


def circle_mean(image, radius_pix, center, n_theta=2048):
    """Mean of cubic-spline samples at uniformly spaced angles on one circle."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    rows = center + radius_pix * np.sin(theta)
    cols = center + radius_pix * np.cos(theta)
    return map_coordinates(image, [rows, cols], order=3, mode="nearest").mean()


def annulus_mean(image, radius_pix, center):
    """Mean of the pixels whose centers fall within half a pixel of the radius."""
    rows, cols = np.indices(image.shape)
    r = np.hypot(rows - center, cols - center)
    return image[(r >= radius_pix - 0.5) & (r < radius_pix + 0.5)].mean()


@pytest.mark.slow(
    reason="shares the response package; an independent pass over the cube"
)
def test_leakage_operator_matches_a_circle_average_of_the_cube(yip_report):
    """Fails if the leakage per (lambda/D)^2 is not the cube's circle mean / 0.25^2.

    The reference reads the package's stellar intensity cube directly, averages
    cubic-spline samples on circles about pixel-center index 127.5 (header center
    128 minus half a pixel) and divides by the pixel solid angle 0.25^2. It agrees
    with the adapter's operator to 1 percent (the allocation is 5 percent) from 3 to
    20 lambda/D. Each mutant misses: origin 128, annular pixel bins (at 3 lambda/D,
    where the leakage varies fastest), and the omitted division (a factor of 16).
    The cube rebinned to 0.5 lambda/D pixels and divided by 0.5^2 gives the same
    density where the profile is smooth (8 to 20 lambda/D). A host 0.03 lambda/D
    across leaks within 1 percent of a point star at 5 lambda/D.
    """
    with fits.open(f"{fetch_pinned_yip()}/stellar_intens.fits") as hdul:
        header = hdul[0].header
        point = np.asarray(hdul[0].data[0], dtype=np.float64)
    assert header["XCENTER"] == header["YCENTER"] == 128.0
    assert header["PIXSCALE"] == 0.25
    center = header["XCENTER"] - 0.5
    point_scene = eqx.tree_at(lambda s: s.star_diameter_mas, uma_scene(), 0.0)
    curve = yip_report.prepare(point_scene).leakage_curve
    seps = np.array([3.0, 4.0, 5.0, 8.0, 12.0, 20.0])
    adapter = np.asarray(curve(jnp.asarray(seps)))
    pixel = header["PIXSCALE"] ** 2
    reference = np.array([circle_mean(point, s / 0.25, center) for s in seps]) / pixel
    np.testing.assert_allclose(adapter, reference, rtol=1e-2)
    origin_128 = np.array([circle_mean(point, s / 0.25, 128.0) for s in seps]) / pixel
    assert np.max(np.abs(origin_128 / reference - 1.0)) > 1e-2
    annular = annulus_mean(point, 3.0 / 0.25, center) / pixel
    assert abs(annular / reference[0] - 1.0) > 1e-2
    assert np.max(reference * pixel / adapter) < 1.0 / 15.0  # omitted division
    binned = point.reshape(128, 2, 128, 2).sum(axis=(1, 3))
    smooth = seps >= 8.0
    rebinned = np.array([circle_mean(binned, s / 0.5, 63.5) for s in seps[smooth]])
    np.testing.assert_allclose(rebinned / 0.5**2, reference[smooth], rtol=1e-2)
    across = 0.03 * yip_report.lod_arcsec * 1000.0
    finite_scene = eqx.tree_at(lambda s: s.star_diameter_mas, uma_scene(), across)
    finite = yip_report.prepare(finite_scene).leakage_curve(jnp.asarray(5.0))
    assert abs(float(finite) / float(curve(jnp.asarray(5.0))) - 1.0) < 1e-2


# -- Survey: detectability calendar ------------------------------------------


@pytest.mark.slow(reason="one exposure solve and budget per phase on the real response")
def test_survey_snr_and_exposure_time_over_one_orbit(yip_report, observatory):
    """Fails if the summary SNR at jaxedith's planned exposure strays from 10.

    47 UMa b on general-knowledge elements (a = 2.1 AU, P near 1078 d, i = 60 deg)
    at twelve phases: jaxedith's fixed-frame `exptime_ayo` for SNR 10 (minimum
    duration) at the exposure start, then the integrated summary SNR over that
    exposure. Where the planet is in the field
    and outside the inner working angle they agree to 1 percent (the planet barely
    moves during the exposure); elsewhere the exposure time is infinite or the
    phase is flagged.
    """
    theta = nominal_planet(
        a_au=2.1,
        e=0.03,
        inclination_rad=math.radians(60.0),
        W_rad=0.5,
        w_rad=0.3,
        tp_d=0.0,
        Rp_au=11.2 * Rearth2AU,
        Ag=0.3,
    )
    iwa = float(yip_report.optical_path.coronagraph.IWA)
    solved = 0
    for phase in np.linspace(0.0, 1.0, 12, endpoint=False):
        start = EPOCH_JD + phase * 1078.0
        rates = yip_report.rates(yip_report.nominal, theta, start)
        sep = float(rates.separation_lod[0])
        scene = ETCScene(
            F0=yip_report.nominal.zero_point,
            Fs_over_F0=yip_report.nominal.star_ratio,
            Fp_over_Fs=rates.contrast[0],
            Fzodi=zodi_fn_leinert(
                observatory, yip_report._exposure(start), uma_scene()
            ),
            Fexozodi=yip_report.nominal.exozodi_1au,
            dist_pc=13.8,
            sep_arcsec=sep * yip_report.lod_arcsec,
            Fbinary=0.0,
            stellar_diam_mas=0.79,
        )
        t10 = float(
            exptime_ayo(
                yip_report.optical_path,
                scene,
                WAVELENGTH_NM,
                sep,
                BAND_NM,
                10.0,
                temp_K=observatory.temperature_K,
                ez_ppf=jnp.inf,
                ppfact=math.inf,
                bg_multiplier=1.0,
                eps_warm_T_cold=0.0,
                cadence="fixed_frame",
                planning="min_duration",
            )
        )
        if sep < iwa or not math.isfinite(t10):
            continue
        budget = yip_report.predict(theta, ScienceInterval.from_start(start, t10))
        np.testing.assert_allclose(budget.snr, 10.0, rtol=1e-2)
        solved += 1
    assert solved >= 6


def test_pinned_package_digest_matches_yippy_catalog():
    """Fails if yippy's catalog digest for the pinned package moves."""
    assert CATALOG["eac1_aavc_2d"]["md5"] == YIP_DIGEST
