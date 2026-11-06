"""Summary radiometry of one companion through jaxedith, integrated over an exposure.

`SummaryReport` turns a planet's orbit and scale (photomancy's physical parameter
names) and a science interval into expected counts and count variances, with every
component separate. It answers two questions with one function:

- `expected(truth, interval)`: the counts of the true planet against the true scene
  (truth side: the true star, the true exozodi level).
- `predict(theta, interval)`: the counts a proposed orbit predicts against the
  model's nominal scene.

Both integrate over the science interval with Gauss-Legendre nodes in time. At each
node the planet position and contrast come from photomancy's forward model and the
count rates from jaxedith's primitives and intermediates, with the coronagraph
response from optixstuff's `YippyCoronagraph`. Planet counts are linear in contrast
only at fixed geometry: the core throughput, the stellar leakage and the exozodi
rate all depend on the separation, so the rates are evaluated at every node of every
proposal. What is cached is what does not depend on the geometry: the flux zero
point, the stellar flux ratio, the collecting area, the optical throughput, the band,
the detector, the exozodi surface brightness at 1 AU, and the stellar leakage curve
at the host's diameter (tabulated once on the response's own radial nodes, since
evaluating it at a finite diameter re-averages a whole stellar map).

Declared conventions:

- Time is JD (TDB) as float64. Orbit times are days from `orbit_epoch_jd`, so `tp`
  in a parameter dict is a time of periapsis in days from that epoch.
- Stellar leakage is the response's azimuthal-mean intensity per pixel divided by the
  stellar pixel's solid angle (optixstuff `core_mean_intensity`) times the
  photometric aperture area (`core_area`).
- The collecting area uses the linear obscuration `sqrt(OBSCURED)` of the response
  package, which records the obscured area fraction.
- Local zodiacal light is `zodi_fn_leinert` at each node's epoch and the target's
  coordinates. The exozodi surface brightness at 1 AU is
  `n_exozodi * F_1zodi(lambda) * (L / Lsun)`, with `F_1zodi` the wavelength-only
  value of `zodi_fn_ayo` (one zodi seen face-on at equivalent insolation).
- Thermal emission is evaluated with the declared `eps_warm_T_cold` (0 switches it
  off whatever the temperature).
- The background variance multiplier `bg_multiplier` is declared (1 for one roll with
  a known mean subtracted). The systematic floor is `leakage / ppfact` and is off
  for an infinite `ppfact`. The exozodi floor is off.
- Cadence: only `photon_counting_legacy` runs, jaxedith's rate-based detector term
  whose clock-induced charge follows its photon-counting time. A detector declared
  with the `fixed_frame` cadence is refused until jaxedith provides a finite-frame
  variance. In the legacy cadence the background multiplier also scales the detector
  term, as jaxedith's exposure-time equation does.

The variance of the measured counts is
``planet + bg_multiplier * background + detector + systematic`` with
``systematic = (integral of leakage / ppfact dt)**2``. The flux-weighted centroid
uses the same detected planet rates as weights, so a throughput change along the arc
moves it.
"""

import dataclasses
import math

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import nm2m, rad2arcsec, s2d
from hwoutils.conversions import jy_to_photons_per_nm_per_m2, mag_to_flux_jy
from jaxedith import detector_noise, thermal_background, zodi_fn_ayo, zodi_fn_leinert
from jaxedith.primitives import (
    exozodi_rate,
    noise_floor_stellar,
    planet_rate,
    stellar_leakage_rate,
    zodi_rate,
)
from optixstuff import (
    ConstantThroughput,
    Detector,
    ExposureConfig,
    OpticalPath,
    SimplePrimary,
    YippyCoronagraph,
)
from photomancy.orbit.forward import predict_photometry, predict_relative_astrometry
from yippy import EqxCoronagraph, fetch_yip
from yippy.datasets import CATALOG, DATA_RELEASE_TAG
from yippy.performance import PROFILE_STEP_PIX

# The public response package, release and digest the summary path is pinned to.
YIP_NAME = "eac1_aavc_2d"
YIP_RELEASE = "data-v2"
YIP_DIGEST = "md5:1f4892faff18e55cbec9781a055bea4d"

# Detector cadence modes and the integer code a record column carries for each.
CADENCE_CODES = {"photon_counting_legacy": 0, "fixed_frame": 1}

# The only meaning of dQE the summary path defines: the surviving fraction of the
# baseline quantum efficiency (1.0 is no degradation).
DQE_MEANINGS = ("survival_fraction",)

# Planet parameter names a summary evaluation reads (photomancy's physical names).
PLANET_KEYS = ("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp", "Lambda")

# The summary path never forms an image, so the detector array size is nominal.
_DETECTOR_SHAPE = (64, 64)


def fetch_pinned_yip():
    """The pinned response package's directory, fetched by yippy's hash-checked loader.

    Refuses when yippy's catalog or release tag no longer match the pinned digest
    and release, so a changed package cannot be read under the old name.
    """
    entry = CATALOG.get(YIP_NAME)
    if entry is None or entry.get("md5") != YIP_DIGEST:
        found = None if entry is None else entry.get("md5")
        msg = f"{YIP_NAME}: yippy declares digest {found!r}, pinned {YIP_DIGEST!r}"
        raise ValueError(msg)
    if DATA_RELEASE_TAG != YIP_RELEASE:
        msg = f"yippy data release is {DATA_RELEASE_TAG!r}, pinned {YIP_RELEASE!r}"
        raise ValueError(msg)
    return fetch_yip(YIP_NAME)


def _finite(name, value, *, low=None, high=None, low_open=False):
    """Check a declared scalar is a finite real inside its range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if low is not None and (value <= low if low_open else value < low):
        raise ValueError(f"{name} must be {'>' if low_open else '>='} {low}")
    if high is not None and value > high:
        raise ValueError(f"{name} must be <= {high}")


@dataclasses.dataclass(frozen=True)
class DetectorSpec:
    """A declared, swappable detector for the summary path.

    Every field is required in meaning: a detector without a name, a family (which
    may read "unspecified"), a dQE meaning and a cadence mode is refused at
    construction. `dqe` is the surviving fraction of the baseline quantum efficiency
    (1.0 is none). The read time is the frame time: one read per frame.

    `fixture()` is jaxedith's test detector: quantum efficiency 0.9, dark current
    3e-5 e/pix/s, no read noise, clock-induced charge 1.3e-3 e/pix/frame, 1000 s
    frames and 0.01 arcsec pixels.
    """

    name: str
    family: str
    quantum_efficiency: float
    dqe: float
    dqe_meaning: str
    dark_current_rate_e_per_s: float
    read_noise_e: float
    clock_induced_charge_rate_e_per_frame: float
    frame_time_s: float
    pixel_scale_arcsec: float
    cadence: str

    def __post_init__(self):
        """Refuse an undeclared or out-of-range detector."""
        for field in ("name", "family"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"DetectorSpec.{field} must be a non-empty string")
        if self.dqe_meaning not in DQE_MEANINGS:
            msg = f"DetectorSpec.dqe_meaning must be one of {DQE_MEANINGS}"
            raise ValueError(msg)
        if self.cadence not in CADENCE_CODES:
            msg = f"DetectorSpec.cadence must be one of {tuple(CADENCE_CODES)}"
            raise ValueError(msg)
        _finite("quantum_efficiency", self.quantum_efficiency, low=0.0, high=1.0)
        _finite("dqe", self.dqe, low=0.0, high=1.0)
        if self.quantum_efficiency == 0.0:
            raise ValueError("quantum_efficiency must be > 0")
        _finite("dark_current_rate_e_per_s", self.dark_current_rate_e_per_s, low=0.0)
        _finite("read_noise_e", self.read_noise_e, low=0.0)
        _finite(
            "clock_induced_charge_rate_e_per_frame",
            self.clock_induced_charge_rate_e_per_frame,
            low=0.0,
        )
        _finite("frame_time_s", self.frame_time_s, low=0.0, low_open=True)
        _finite("pixel_scale_arcsec", self.pixel_scale_arcsec, low=0.0, low_open=True)

    @classmethod
    def fixture(cls, **overrides):
        """The jaxedith test detector, in the legacy photon-counting cadence."""
        fields = dict(
            name="jaxedith-fixture",
            family="unspecified",
            quantum_efficiency=0.9,
            dqe=1.0,
            dqe_meaning="survival_fraction",
            dark_current_rate_e_per_s=3e-5,
            read_noise_e=0.0,
            clock_induced_charge_rate_e_per_frame=1.3e-3,
            frame_time_s=1000.0,
            pixel_scale_arcsec=0.01,
            cadence="photon_counting_legacy",
        )
        fields.update(overrides)
        return cls(**fields)

    @property
    def cadence_code(self):
        """The integer code of the cadence mode."""
        return CADENCE_CODES[self.cadence]

    def detector(self):
        """The optixstuff detector this declaration describes."""
        return Detector(
            pixel_scale_arcsec=self.pixel_scale_arcsec,
            shape=_DETECTOR_SHAPE,
            quantum_efficiency=self.quantum_efficiency,
            dark_current_rate_e_per_s=self.dark_current_rate_e_per_s,
            read_noise_e=self.read_noise_e,
            clock_induced_charge_rate_e_per_frame=(
                self.clock_induced_charge_rate_e_per_frame
            ),
            frame_time_s=self.frame_time_s,
            read_time_s=self.frame_time_s,
            dqe=self.dqe,
        )


class SummaryScene(eqx.Module):
    """The host and its dust, as the summary path sees them.

    Attributes:
        star_flux_density_jy: Stellar flux density at the band center [Jy].
        dist_pc: Distance [pc].
        star_mass_kg: Stellar mass [kg] (fixed in the orbit model).
        luminosity_lsun: Bolometric luminosity [Lsun]; scales the exozodi.
        ra_deg: Right ascension [deg]; sets the local zodiacal light.
        dec_deg: Declination [deg].
        star_diameter_mas: Stellar angular diameter [mas]; 0 is a point star.
        n_exozodi: Exozodi level in zodis (always passed explicitly).
    """

    star_flux_density_jy: float
    dist_pc: float
    star_mass_kg: float
    luminosity_lsun: float
    ra_deg: float
    dec_deg: float
    star_diameter_mas: float
    n_exozodi: float


class SummaryTruth(eqx.Module):
    """The true scene and the true planet parameters of one subject.

    `planet` maps photomancy's physical names (`PLANET_KEYS`, optionally `Mp` in kg)
    to scalars.
    """

    scene: SummaryScene
    planet: dict


class ScienceInterval(eqx.Module):
    """One realized science interval: its midpoint (JD) and duration (s)."""

    midpoint_jd: float
    duration_s: float

    @classmethod
    def from_start(cls, start_jd, duration_s):
        """The interval that starts at `start_jd` and lasts `duration_s`."""
        start = jnp.asarray(start_jd, dtype=jnp.float64)
        duration = jnp.asarray(duration_s, dtype=jnp.float64)
        return cls(start + 0.5 * duration * s2d, duration)


class RateComponents(eqx.Module):
    """Count rates [e/s] and geometry at a set of epochs, one value per epoch.

    `detector` is jaxedith's legacy rate-based detector term; `floor` is the
    systematic-floor rate `leakage / ppfact`. Positions are (east, north) offsets in
    arcsec; `in_field` is true where the separation lies inside the response field.
    """

    planet: jax.Array
    leakage: jax.Array
    zodi: jax.Array
    exozodi: jax.Array
    thermal: jax.Array
    detector: jax.Array
    floor: jax.Array
    contrast: jax.Array
    ra_arcsec: jax.Array
    dec_arcsec: jax.Array
    separation_lod: jax.Array
    in_field: jax.Array


class CountBudget(eqx.Module):
    """Expected counts and count variances over one science interval, by component.

    Counts are electrons over the interval; variances are electrons squared.
    `variance` is the sum of the four variance components. `ra_arcsec` and
    `dec_arcsec` are the flux-weighted centroid over the interval; `in_field` is
    true when every node lies inside the response field.
    """

    planet: jax.Array
    leakage: jax.Array
    zodi: jax.Array
    exozodi: jax.Array
    thermal: jax.Array
    var_planet: jax.Array
    var_background: jax.Array
    var_detector: jax.Array
    var_systematic: jax.Array
    ra_arcsec: jax.Array
    dec_arcsec: jax.Array
    science_s: jax.Array
    in_field: jax.Array
    cadence: jax.Array

    @property
    def background(self):
        """Photon background counts: leakage, zodi, exozodi and thermal."""
        return self.leakage + self.zodi + self.exozodi + self.thermal

    @property
    def variance(self):
        """Total variance of the measured planet counts."""
        return (
            self.var_planet
            + self.var_background
            + self.var_detector
            + self.var_systematic
        )

    @property
    def snr(self):
        """Expected planet counts over the square root of the total variance."""
        return self.planet / jnp.sqrt(self.variance)


class PreparedScene(eqx.Module):
    """Geometry-invariant factors of one scene, cached for repeated evaluation.

    `leakage_curve` is the response's stellar leakage per (lambda/D)^2 at the host's
    diameter, tabulated on the response's radial profile nodes.
    """

    zero_point: jax.Array
    star_ratio: jax.Array
    exozodi_1au: jax.Array
    dist_pc: jax.Array
    star_mass_kg: jax.Array
    stellar_diam_lod: jax.Array
    leakage_curve: interpax.PchipInterpolator
    scene: SummaryScene


class SummaryReport(eqx.Module):
    """Expected and predicted summary counts of one companion over an exposure.

    Build it with `from_yip`. `predict` evaluates against `nominal_scene`;
    `expected` against the truth's own scene. Both share `budget`, which integrates
    `rates` over Gauss-Legendre nodes in time.
    """

    optical_path: OpticalPath
    observatory: eqx.Module
    detector_spec: DetectorSpec = eqx.field(static=True)
    nominal: PreparedScene
    wavelength_nm: float
    bin_width_nm: float
    orbit_epoch_jd: float
    bg_multiplier: float
    ppfact: float
    eps_warm_T_cold: float
    nodes: jax.Array
    weights: jax.Array
    field_radius_lod: float
    leakage_step_lod: float
    leakage_max_lod: float

    def __init__(
        self,
        optical_path,
        observatory,
        detector_spec,
        nominal_scene,
        *,
        wavelength_nm,
        bin_width_nm,
        orbit_epoch_jd,
        bg_multiplier,
        ppfact,
        eps_warm_T_cold,
        n_time_nodes,
        field_radius_lod,
        leakage_step_lod,
    ):
        """Validate the declarations and cache the nominal scene."""
        if not isinstance(detector_spec, DetectorSpec):
            raise TypeError("detector_spec must be a DetectorSpec")
        if detector_spec.cadence != "photon_counting_legacy":
            msg = (
                f"cadence {detector_spec.cadence!r} needs jaxedith's finite-frame "
                "variance API, which this jaxedith does not provide"
            )
            raise NotImplementedError(msg)
        if isinstance(n_time_nodes, bool) or not isinstance(n_time_nodes, int):
            raise ValueError("n_time_nodes must be an integer")
        if n_time_nodes < 1:
            raise ValueError("n_time_nodes must be at least 1")
        _finite("wavelength_nm", wavelength_nm, low=0.0, low_open=True)
        _finite("bin_width_nm", bin_width_nm, low=0.0, low_open=True)
        _finite("orbit_epoch_jd", orbit_epoch_jd)
        _finite("bg_multiplier", bg_multiplier, low=0.0, low_open=True)
        if isinstance(ppfact, bool) or not isinstance(ppfact, (int, float)):
            raise ValueError("ppfact must be a real number")
        if not ppfact > 0:
            raise ValueError("ppfact must be positive (infinity turns the floor off)")
        _finite("eps_warm_T_cold", eps_warm_T_cold, low=0.0)
        _finite("field_radius_lod", field_radius_lod, low=0.0, low_open=True)
        _finite("leakage_step_lod", leakage_step_lod, low=0.0, low_open=True)
        self.optical_path = optical_path
        self.observatory = observatory
        self.detector_spec = detector_spec
        self.wavelength_nm = float(wavelength_nm)
        self.bin_width_nm = float(bin_width_nm)
        self.orbit_epoch_jd = float(orbit_epoch_jd)
        self.bg_multiplier = float(bg_multiplier)
        self.ppfact = float(ppfact)
        self.eps_warm_T_cold = float(eps_warm_T_cold)
        x, w = np.polynomial.legendre.leggauss(n_time_nodes)
        self.nodes = jnp.asarray(x, dtype=jnp.float64)
        self.weights = jnp.asarray(w, dtype=jnp.float64)
        self.field_radius_lod = float(field_radius_lod)
        self.leakage_step_lod = float(leakage_step_lod)
        # The leakage table reaches the corner of a square field.
        n_steps = math.ceil(math.sqrt(2.0) * field_radius_lod / leakage_step_lod)
        self.leakage_max_lod = n_steps * self.leakage_step_lod
        self.nominal = self.prepare(nominal_scene)

    @classmethod
    def from_yip(
        cls,
        detector_spec,
        nominal_scene,
        *,
        orbit_epoch_jd,
        observatory,
        yip_path=None,
        bin_width_nm=100.0,
        optical_throughput=0.5,
        bg_multiplier=1.0,
        ppfact=math.inf,
        eps_warm_T_cold=0.0,
        n_time_nodes=8,
    ):
        """Build the summary path on a response package.

        The aperture diameter and the central wavelength come from the package
        header; the linear obscuration is the square root of its obscured area
        fraction. Without `yip_path` the pinned package is fetched through yippy's
        hash-checked loader.
        """
        path = fetch_pinned_yip() if yip_path is None else yip_path
        backend = EqxCoronagraph(path)
        wavelength_nm, diameter_m = backend.stellar_frame
        optical_path = OpticalPath(
            primary=SimplePrimary(
                diameter_m=diameter_m, obscuration=math.sqrt(backend.frac_obscured)
            ),
            attenuating_elements=(
                ConstantThroughput(throughput=optical_throughput, name="optics"),
            ),
            coronagraph=YippyCoronagraph(backend=backend),
            detector=detector_spec.detector(),
        )
        return cls(
            optical_path,
            observatory,
            detector_spec,
            nominal_scene,
            wavelength_nm=wavelength_nm,
            bin_width_nm=bin_width_nm,
            orbit_epoch_jd=orbit_epoch_jd,
            bg_multiplier=bg_multiplier,
            ppfact=ppfact,
            eps_warm_T_cold=eps_warm_T_cold,
            n_time_nodes=n_time_nodes,
            field_radius_lod=backend.OWA,
            leakage_step_lod=PROFILE_STEP_PIX * backend.stellar_pixel_scale_lod,
        )

    # -- Cached factors ----------------------------------------------------

    @property
    def lod_arcsec(self):
        """lambda/D at the band center [arcsec]."""
        return (
            self.wavelength_nm * nm2m / self.optical_path.primary.diameter_m
        ) * rad2arcsec

    def prepare(self, scene):
        """Cache the geometry-invariant factors of `scene`."""
        wl = self.wavelength_nm
        zero_point = jy_to_photons_per_nm_per_m2(mag_to_flux_jy(0.0), wl)
        star = jy_to_photons_per_nm_per_m2(scene.star_flux_density_jy, wl)
        exposure = self._exposure(jnp.asarray(0.0))
        one_zodi = zodi_fn_ayo(self.observatory, exposure, scene)
        diam_lod = jnp.asarray(scene.star_diameter_mas) / (1000.0 * self.lod_arcsec)
        n = round(self.leakage_max_lod / self.leakage_step_lod)
        grid = jnp.arange(n + 1, dtype=jnp.float64) * self.leakage_step_lod
        leakage = self.optical_path.coronagraph.core_mean_intensity(
            grid, wl, stellar_diam_lod=diam_lod
        )
        return PreparedScene(
            zero_point=jnp.asarray(zero_point, dtype=jnp.float64),
            star_ratio=star / zero_point,
            exozodi_1au=jnp.asarray(scene.n_exozodi)
            * one_zodi
            * jnp.asarray(scene.luminosity_lsun),
            dist_pc=jnp.asarray(scene.dist_pc, dtype=jnp.float64),
            star_mass_kg=jnp.asarray(scene.star_mass_kg, dtype=jnp.float64),
            stellar_diam_lod=diam_lod,
            leakage_curve=interpax.PchipInterpolator(
                grid, leakage, extrapolate=False, check=False
            ),
            scene=scene,
        )

    def _exposure(self, start_jd):
        """An exposure record for the zodi callables at one epoch."""
        return ExposureConfig(
            start_time_jd=start_jd,
            exposure_time_s=jnp.asarray(0.0),
            central_wavelength_nm=jnp.asarray(self.wavelength_nm),
            bin_width_nm=jnp.asarray(self.bin_width_nm),
            position_angle_deg=jnp.asarray(0.0),
        )

    def node_times_jd(self, interval):
        """Gauss-Legendre epochs [JD] over `interval`."""
        half_d = 0.5 * jnp.asarray(interval.duration_s) * s2d
        return jnp.asarray(interval.midpoint_jd) + self.nodes * half_d

    # -- Evaluation --------------------------------------------------------

    def rates(self, prepared, planet, times_jd):
        """Count rates and geometry of `planet` against `prepared` at `times_jd`."""
        wl = self.wavelength_nm
        dl = self.bin_width_nm
        path = self.optical_path
        coro = path.coronagraph
        times_jd = jnp.atleast_1d(jnp.asarray(times_jd, dtype=jnp.float64))
        mp = planet.get("Mp", 0.0)
        orbit = (
            planet["a"],
            planet["e"],
            planet["cos_i"],
            planet["W"],
            planet["cos_w"],
            planet["sin_w"],
            planet["tp"],
        )
        t_d = times_jd - self.orbit_epoch_jd
        ra, dec = predict_relative_astrometry(
            t_d, *orbit, prepared.star_mass_kg, prepared.dist_pc, Mp=mp
        )
        alpha, dmag = predict_photometry(
            t_d,
            *orbit,
            prepared.star_mass_kg,
            planet["Lambda"],
            prepared.dist_pc,
            Mp=mp,
        )
        contrast = 10.0 ** (-0.4 * dmag)
        lod = self.lod_arcsec
        sep = alpha / lod
        area = path.primary.area_m2
        optics = path.system_throughput(wl)
        n_ch = path.n_channels
        core_tp = coro.throughput(sep, wl)
        core_area = coro.core_area(sep, wl)
        sky = coro.occulter_transmission(sep, wl)
        intensity = prepared.leakage_curve(jnp.clip(sep, 0.0, self.leakage_max_lod))
        f0 = prepared.zero_point
        fs = prepared.star_ratio
        fzodi = jax.vmap(
            lambda t: zodi_fn_leinert(
                self.observatory, self._exposure(t), prepared.scene
            )
        )(times_jd)
        planet_e = planet_rate(f0, fs, contrast, area, optics, core_tp, dl, n_ch)
        leakage_e = stellar_leakage_rate(
            f0, fs, area, optics, dl, n_ch, core_area, intensity
        )
        zodi_e = zodi_rate(f0, fzodi, lod, sky, area, optics, dl, n_ch, core_area)
        exozodi_e = exozodi_rate(
            f0,
            prepared.exozodi_1au,
            lod,
            sky,
            area,
            optics,
            dl,
            n_ch,
            core_area,
            prepared.dist_pc,
            alpha,
        )
        thermal_e = thermal_background(
            path,
            wl,
            sep,
            dl,
            self.observatory.temperature_K,
            eps_warm_T_cold=self.eps_warm_T_cold,
        ) * jnp.ones_like(sep)
        detector_e = detector_noise(
            path,
            wl,
            sep,
            planet_e + leakage_e + zodi_e + exozodi_e,
            npix_multiplier=path.npix_multiplier,
        )
        floor_e = noise_floor_stellar(
            f0, fs, area, optics, dl, n_ch, intensity / self.ppfact, core_area
        )
        return RateComponents(
            planet=planet_e,
            leakage=leakage_e,
            zodi=zodi_e,
            exozodi=exozodi_e,
            thermal=thermal_e,
            detector=detector_e,
            floor=floor_e,
            contrast=contrast,
            ra_arcsec=ra,
            dec_arcsec=dec,
            separation_lod=sep,
            in_field=sep <= self.field_radius_lod,
        )

    def budget(self, prepared, planet, interval):
        """Counts and variances of `planet` against `prepared` over `interval`."""
        r = self.rates(prepared, planet, self.node_times_jd(interval))
        duration = jnp.asarray(interval.duration_s, dtype=jnp.float64)
        w = self.weights * 0.5 * duration
        k = self.bg_multiplier
        planet_e = jnp.sum(w * r.planet)
        leakage_e = jnp.sum(w * r.leakage)
        zodi_e = jnp.sum(w * r.zodi)
        exozodi_e = jnp.sum(w * r.exozodi)
        thermal_e = jnp.sum(w * r.thermal)
        flux_w = w * r.planet
        total_w = jnp.sum(flux_w)
        safe = jnp.where(total_w > 0, total_w, 1.0)
        mean_w = w / jnp.sum(w)
        # Flux-weighted centroid; a planet with no counts falls back to the time mean.
        ra, dec = (
            jnp.where(total_w > 0, jnp.sum(flux_w * x) / safe, jnp.sum(mean_w * x))
            for x in (r.ra_arcsec, r.dec_arcsec)
        )
        return CountBudget(
            planet=planet_e,
            leakage=leakage_e,
            zodi=zodi_e,
            exozodi=exozodi_e,
            thermal=thermal_e,
            var_planet=planet_e,
            var_background=k * (leakage_e + zodi_e + exozodi_e + thermal_e),
            var_detector=k * jnp.sum(w * r.detector),
            var_systematic=jnp.sum(w * r.floor) ** 2,
            ra_arcsec=ra,
            dec_arcsec=dec,
            science_s=duration,
            in_field=jnp.all(r.in_field),
            cadence=jnp.asarray(self.detector_spec.cadence_code),
        )

    def expected(self, truth, interval):
        """The true planet's budget against the true scene (truth side only)."""
        return self.budget(self.prepare(truth.scene), truth.planet, interval)

    def predict(self, theta, interval):
        """A proposed planet's budget against the nominal scene."""
        return self.budget(self.nominal, theta, interval)


def nominal_planet(*, a_au, e, inclination_rad, W_rad, w_rad, tp_d, Rp_au, Ag):
    """A planet parameter dict in photomancy's physical names from plain elements."""
    return {
        "a": jnp.asarray(a_au, dtype=jnp.float64),
        "e": jnp.asarray(e, dtype=jnp.float64),
        "cos_i": jnp.cos(jnp.asarray(inclination_rad, dtype=jnp.float64)),
        "W": jnp.asarray(W_rad, dtype=jnp.float64),
        "cos_w": jnp.cos(jnp.asarray(w_rad, dtype=jnp.float64)),
        "sin_w": jnp.sin(jnp.asarray(w_rad, dtype=jnp.float64)),
        "tp": jnp.asarray(tp_d, dtype=jnp.float64),
        "Lambda": jnp.asarray(Ag, dtype=jnp.float64)
        * jnp.asarray(Rp_au, dtype=jnp.float64) ** 2,
    }
