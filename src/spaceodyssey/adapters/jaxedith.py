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
count rates from jaxedith's intermediates (which apply the detector the way jaxedith
does), with the coronagraph response from optixstuff's `YippyCoronagraph`. Planet
counts are linear in contrast only at fixed geometry: the core throughput, the
stellar leakage and the exozodi rate all depend on the separation, so the rates are
evaluated at every node of every proposal. What is cached is what does not depend
on the geometry: the flux zero point, the stellar flux ratio, the band, the exozodi
surface brightness at 1 AU, and the stellar leakage curve at the host's diameter
(tabulated once on the response's own radial nodes, since evaluating it at a finite
diameter re-averages a whole stellar map). The table is served to jaxedith through a
coronagraph that delegates every other curve to the response.

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
- Detector quantum efficiency and dQE (a survival fraction) enter only where
  jaxedith's intermediates apply them, once on photon rates; the adapter never
  applies them itself.
- The background variance multiplier `bg_multiplier` is declared (1 for one roll with
  a known mean subtracted). The systematic floor is `leakage / ppfact` and is off
  for an infinite `ppfact`. The exozodi floor is off.
- Cadence: `photon_counting_legacy` is jaxedith's rate-based detector term whose
  clock-induced charge follows its photon-counting time; there the background
  multiplier also scales the detector term, as jaxedith's exposure-time equation does.
  `fixed_frame` takes the detector variance of the whole interval from jaxedith's
  finite-frame variance (dark current per second, read noise and clock-induced charge
  per frame including a partial last frame, no background multiplier), with the
  aperture pixel count averaged over the nodes. `fixed_frame` is the default
  (`DetectorSpec.fixture`); the legacy cadence is used only when declared.

The variance of the measured counts is
``planet + bg_multiplier * background + detector + systematic`` with
``systematic = (integral of leakage / ppfact dt)**2``. The flux-weighted centroid
uses the same detected planet rates as weights, so a throughput change along the arc
moves it.

The reporting law turns budgets into records. `ReportingLaw` is one normalized
joint law of measured counts and position: detection when the counts reach
`threshold` times the planet-free noise of the model's nominal scene
(`limit_counts`), null otherwise. `sample_report` draws a `SummaryRow` from the
truth side's budget and `loglike_report` scores a row against a proposal's
predicted budget. Every reported uncertainty comes from the row's own observables.
`summary_kind` is the record kind, its payload columns and audiences read from the
profile table; `calibration_id` digests the declarations a row was made under.

`JaxedithSummaryChannel` is the version-2 channel model that writes those rows from
an executed occupancy: the science interval and its epoch come from the record's own
start and end ticks, so a delayed admission or a cancellation moves them, and a visit
outside its certified window is a no-science row.
"""

import concurrent.futures
import dataclasses
import functools
import hashlib
import math

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import AU2m, Mearth2kg, Msun2kg, nm2m, rad2arcsec, s2d
from hwoutils.conversions import (
    jy_to_photons_per_nm_per_m2,
    mag_to_flux_jy,
    photons_per_nm_per_m2_to_jy,
)
from jax.scipy.special import log_ndtr
from jax.scipy.stats import norm
from jaxedith import (
    ETCScene,
    count_variance_components_ayo,
    detector_noise,
    exozodi_background,
    planet_signal,
    stellar_leakage,
    stellar_noise_floor,
    thermal_background,
    zodi_background,
    zodi_fn_ayo,
    zodi_fn_leinert,
)
from optixstuff import (
    AbstractScalarCoronagraph,
    ConstantThroughput,
    Detector,
    ExposureConfig,
    OpticalPath,
    SimplePrimary,
    YippyCoronagraph,
)
from orbix.equations.orbit import period_to_sma
from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.forward import predict_photometry, predict_relative_astrometry
from photomancy.orbit.likelihoods import loglike_relative_astrom
from yippy import EqxCoronagraph, fetch_yip
from yippy.datasets import CATALOG, DATA_RELEASE_TAG
from yippy.performance import PROFILE_STEP_PIX

from spaceodyssey.adapters.physical_context import (
    Certificate,
    PlannedVisit,
    context_from_profile,
    require_certified,
)
from spaceodyssey.adapters.physical_profile import (
    FIRST_SUMMARY_V1,
    audience_flags,
    check_profile,
    load_profile,
    payload_column_names,
)
from spaceodyssey.core.enums import I64
from spaceodyssey.core.kinds import AbstractChannelModel
from spaceodyssey.core.records import PayloadColumn, RecordKind

# The public response package, release and digest the summary path is pinned to.
YIP_NAME = "eac1_aavc_2d"
YIP_RELEASE = "data-v2"
YIP_DIGEST = "md5:1f4892faff18e55cbec9781a055bea4d"

# Detector cadence modes and the integer code a record column carries for each (the
# profile table's `cadence` column).
CADENCE_CODES = {"fixed_frame": 0, "photon_counting_legacy": 1}

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
    frames and 0.01 arcsec pixels, read in fixed frames.
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
        """The jaxedith test detector in the fixed-frame cadence (the profile's).

        Pass ``cadence="photon_counting_legacy"`` for jaxedith's rate-based term.
        """
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
            cadence="fixed_frame",
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

    `detector` is jaxedith's legacy rate-based detector term (the fixed-frame
    cadence takes its detector variance per interval instead); `floor` is the
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
        """Expected planet counts over the square root of the total variance.

        Zero for an interval with no variance (a zero-duration interval).
        """
        variance = self.variance
        safe = jnp.where(variance > 0, variance, 1.0)
        return jnp.where(variance > 0, self.planet / jnp.sqrt(safe), 0.0)


class TabulatedLeakageCoronagraph(AbstractScalarCoronagraph):
    """A response whose stellar leakage is read from a table at one host diameter.

    Throughput, core area and occulter transmission come from `response`; the
    leakage per (lambda/D)^2 comes from `leakage_curve`, the response's own curve at
    the host's diameter tabulated on its radial profile nodes, clipped to the table
    beyond its last node. It serves the summary rates only and forms no images.
    """

    response: eqx.Module
    leakage_curve: interpax.PchipInterpolator
    leakage_max_lod: float
    pixel_scale_lod: float
    IWA: float
    OWA: float

    def throughput(self, separation_lod, wavelength_nm, *, time_s=0.0):
        """Core throughput of the response."""
        return self.response.throughput(separation_lod, wavelength_nm, time_s=time_s)

    def core_area(self, separation_lod, wavelength_nm, *, time_s=0.0):
        """Photometric aperture area of the response [(lambda/D)^2]."""
        return self.response.core_area(separation_lod, wavelength_nm, time_s=time_s)

    def core_mean_intensity(
        self, separation_lod, wavelength_nm, *, time_s=0.0, stellar_diam_lod=0.0
    ):
        """Tabulated leakage per (lambda/D)^2 at the host's diameter."""
        clipped = jnp.clip(jnp.asarray(separation_lod), 0.0, self.leakage_max_lod)
        return self.leakage_curve(clipped)

    def occulter_transmission(self, separation_lod, wavelength_nm, *, time_s=0.0):
        """Sky transmission of the response."""
        return self.response.occulter_transmission(
            separation_lod, wavelength_nm, time_s=time_s
        )


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
    optical_path: OpticalPath
    scene: SummaryScene

    @property
    def leakage_curve(self):
        """The tabulated leakage per (lambda/D)^2 at this host's diameter."""
        return self.optical_path.coronagraph.leakage_curve


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
        response = self.optical_path.coronagraph
        leakage = response.core_mean_intensity(grid, wl, stellar_diam_lod=diam_lod)
        tabulated = TabulatedLeakageCoronagraph(
            response=response,
            leakage_curve=interpax.PchipInterpolator(
                grid, leakage, extrapolate=False, check=False
            ),
            leakage_max_lod=self.leakage_max_lod,
            pixel_scale_lod=float(response.pixel_scale_lod),
            IWA=float(response.IWA),
            OWA=float(response.OWA),
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
            optical_path=eqx.tree_at(
                lambda p: p.coronagraph, self.optical_path, tabulated
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
        sep = alpha / self.lod_arcsec
        path = prepared.optical_path
        f0 = prepared.zero_point
        fs = prepared.star_ratio
        n_ch = path.n_channels
        diam = prepared.stellar_diam_lod
        fzodi = jax.vmap(
            lambda t: zodi_fn_leinert(
                self.observatory, self._exposure(t), prepared.scene
            )
        )(times_jd)
        planet_e = planet_signal(path, wl, sep, dl, f0, fs, contrast, n_channels=n_ch)
        leakage_e = stellar_leakage(
            path, wl, sep, dl, f0, fs, n_channels=n_ch, stellar_diam_lod=diam
        )
        zodi_e = zodi_background(path, wl, sep, dl, f0, fzodi, n_channels=n_ch)
        exozodi_e = exozodi_background(
            path,
            wl,
            sep,
            dl,
            f0,
            prepared.exozodi_1au,
            prepared.dist_pc,
            alpha,
            n_channels=n_ch,
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
        floor_e = stellar_noise_floor(
            path,
            wl,
            sep,
            dl,
            f0,
            fs,
            n_channels=n_ch,
            ppfact=self.ppfact,
            stellar_diam_lod=diam,
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
        mean_w = self.weights / jnp.sum(self.weights)
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
            var_detector=self._detector_variance(prepared, r, duration, w),
            var_systematic=jnp.sum(w * r.floor) ** 2,
            ra_arcsec=ra,
            dec_arcsec=dec,
            science_s=duration,
            in_field=jnp.all(r.in_field),
            cadence=jnp.asarray(self.detector_spec.cadence_code),
        )

    def _detector_variance(self, prepared, rates, duration, weights):
        """Detector variance over the interval for the declared cadence."""
        if self.detector_spec.cadence != "fixed_frame":
            return self.bg_multiplier * jnp.sum(weights * rates.detector)

        # Only the aperture pixel count at each node's separation enters the
        # fixed-frame detector term; it is averaged over the nodes.
        def at_node(contrast, sep):
            scene = ETCScene(
                F0=prepared.zero_point,
                Fs_over_F0=prepared.star_ratio,
                Fp_over_Fs=contrast,
                dist_pc=prepared.dist_pc,
                sep_arcsec=sep * self.lod_arcsec,
            )
            return count_variance_components_ayo(
                prepared.optical_path,
                scene,
                self.wavelength_nm,
                sep,
                self.bin_width_nm,
                duration,
                cadence="fixed_frame",
                bg_multiplier=self.bg_multiplier,
                ppfact=self.ppfact,
                temp_K=self.observatory.temperature_K,
                eps_warm_T_cold=self.eps_warm_T_cold,
            )["detector"]

        per_node = jax.vmap(at_node)(rates.contrast, rates.separation_lod)
        return jnp.sum(self.weights * per_node) / jnp.sum(self.weights)

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


def planet_free(planet):
    """`planet` with no reflected light: the geometry stays, the counts are zero.

    The budget of a planet-free planet is the reference the detection limit is
    built from (`ReportingLaw.limit_counts`).
    """
    out = dict(planet)
    out["Lambda"] = jnp.zeros_like(jnp.asarray(planet["Lambda"], dtype=jnp.float64))
    return out


# -- Reporting law and record kind ---------------------------------------------

SUMMARY_KIND_NAME = "summary_measurement"

# The summary record's payload columns, in record order (the profile table's).
SUMMARY_COLUMNS = (
    "time_d",
    "science_s",
    "occupied_s",
    "detected",
    "counts",
    "limit_counts",
    "snr",
    "ra",
    "dec",
    "ra_err",
    "dec_err",
    "corr",
    "cadence",
    "calibration_id",
    "valid_science",
)

# A calibration id is six bytes: every value is an integer a float64 holds exactly.
CALIBRATION_ID_BYTES = 6


def summary_kind(profile=None):
    """The summary record kind, its columns and audiences read from the profile.

    Refuses a profile that is incomplete (KP5: every column needs a unit, a sign, a
    measure and an audience) or whose columns are not `SUMMARY_COLUMNS` in order.
    """
    profile = load_profile() if profile is None else profile
    problems = check_profile(profile)
    if problems:
        raise ValueError(f"the profile is incomplete: {problems}")
    names = payload_column_names(profile)
    if names != SUMMARY_COLUMNS:
        msg = f"profile columns {names} are not the summary columns {SUMMARY_COLUMNS}"
        raise ValueError(msg)
    columns = tuple(
        PayloadColumn(c["name"], audience_flags(c["audience"]))
        for c in profile["payload_columns"]
    )
    return RecordKind(SUMMARY_KIND_NAME, columns)


class SummaryRow(eqx.Module):
    """One summary record's payload, one field per column (see `SUMMARY_COLUMNS`).

    `detected` and `valid_science` are 0 or 1. A null row carries its epoch,
    durations, `limit_counts`, `corr`, `cadence` and `calibration_id`, and zero
    counts, signal-to-noise, position and errors. A row with no science interval
    (`valid_science = 0`) carries zero `limit_counts` as well.
    """

    time_d: jax.Array
    science_s: jax.Array
    occupied_s: jax.Array
    detected: jax.Array
    counts: jax.Array
    limit_counts: jax.Array
    snr: jax.Array
    ra: jax.Array
    dec: jax.Array
    ra_err: jax.Array
    dec_err: jax.Array
    corr: jax.Array
    cadence: jax.Array
    calibration_id: jax.Array
    valid_science: jax.Array

    def payload(self):
        """The payload vector, float64 in column order (batched rows stack last)."""
        return jnp.stack(
            [jnp.asarray(getattr(self, n), jnp.float64) for n in SUMMARY_COLUMNS],
            axis=-1,
        )

    @classmethod
    def from_payload(cls, payload):
        """The row a payload vector holds (the inverse of `payload`)."""
        payload = jnp.asarray(payload, jnp.float64)
        if payload.shape[-1] != len(SUMMARY_COLUMNS):
            msg = f"a summary payload has {len(SUMMARY_COLUMNS)} columns"
            raise ValueError(f"{msg}, got {payload.shape[-1]}")
        return cls(**{n: payload[..., i] for i, n in enumerate(SUMMARY_COLUMNS)})


def observed_snr(counts, limit_counts):
    """Signal-to-noise from observables: counts over sqrt(limit^2 + max(counts, 0)).

    Zero where the noise estimate is zero (a row with no science interval).
    """
    noise2 = limit_counts**2 + jnp.maximum(counts, 0.0)
    safe = jnp.where(noise2 > 0, noise2, 1.0)
    return jnp.where(noise2 > 0, counts / jnp.sqrt(safe), 0.0)


class ReportingLaw(eqx.Module):
    """One normalized joint law of measured counts and position, detection and null.

    Per exposure with expected planet counts `N` and total count variance `s^2`
    (planet shot noise, background, detector and systematic floor; `CountBudget`):

    - the measured counts are ``f = N + s * eps1``;
    - a detection is the event ``f >= threshold * limit_counts``, where
      `limit_counts` is the planet-free noise standard deviation of the model's
      nominal scene (`limit_counts`); a null is the complementary event;
    - on a detection the row reports `f`, the observable signal-to-noise
      `observed_snr(f, limit_counts)`, and a position drawn about the flux-weighted
      centroid with per-axis standard deviation ``kappa * (lambda/D) / snr`` and
      correlation `corr`.

    The detection density is ``Normal(f; N, s^2) * Normal2(pos; mu, Sigma(f))`` on
    ``f >= threshold * limit_counts`` and the null probability is
    ``Phi((threshold * limit_counts - N) / s)``; together they integrate to one.
    `Sigma` depends on the observable `f` only, so the photometric, astrometric and
    selection terms separate given the row. Every reported uncertainty is computed
    from the row's own observables; no expected signal-to-noise is reported.

    Attributes:
        threshold: Detection threshold in planet-free standard deviations.
        kappa: Centroid error per axis in units of (lambda/D) / snr.
        corr: Correlation of the east and north position errors.
    """

    threshold: float = eqx.field(static=True, default=5.0)
    kappa: float = eqx.field(static=True, default=1.0)
    corr: float = eqx.field(static=True, default=0.0)

    def __check_init__(self):
        """Refuse a nonpositive threshold or kappa, or a correlation outside (-1, 1)."""
        _finite("threshold", self.threshold, low=0.0, low_open=True)
        _finite("kappa", self.kappa, low=0.0, low_open=True)
        _finite("corr", self.corr)
        if not -1.0 < self.corr < 1.0:
            raise ValueError(f"corr must lie in (-1, 1), got {self.corr!r}")

    @staticmethod
    def limit_counts(reference):
        """Planet-free noise standard deviation of a reference budget [e-].

        `reference` is the model's budget of the planned planet with no reflected
        light (`planet_free`) against the nominal scene over the realized science
        interval: the background, detector and systematic variances only.
        """
        variance = (
            reference.var_background + reference.var_detector + reference.var_systematic
        )
        return jnp.sqrt(jnp.maximum(variance, 0.0))

    def position_error(self, snr, lod_arcsec):
        """Per-axis position standard deviation [arcsec]; zero where snr is zero."""
        safe = jnp.where(snr > 0, snr, 1.0)
        return jnp.where(snr > 0, self.kappa * lod_arcsec / safe, 0.0)

    def log_null_probability(self, limit_counts, budget):
        """The log probability of a null, log P(null | theta)."""
        s = _noise(budget)
        return log_ndtr((self.threshold * limit_counts - budget.planet) / s)

    def log_counts_density(self, counts, limit_counts, budget):
        """Log density of the measured counts on the detection region; -inf below."""
        s = _noise(budget)
        inside = counts >= self.threshold * limit_counts
        return jnp.where(inside, norm.logpdf(counts, budget.planet, s), -jnp.inf)

    @staticmethod
    def log_position_density(row, budget, valid):
        """Log density of the reported position given its reported covariance.

        Masked to zero where `valid` is false (photomancy's relative-astrometry
        likelihood with the row's errors and correlation).
        """
        has_errors = (row.ra_err > 0) & (row.dec_err > 0)
        data = RelativeAstromData(
            times=jnp.zeros(1),
            ra=jnp.atleast_1d(row.ra),
            dec=jnp.atleast_1d(row.dec),
            ra_err=jnp.atleast_1d(jnp.where(has_errors, row.ra_err, 1.0)),
            dec_err=jnp.atleast_1d(jnp.where(has_errors, row.dec_err, 1.0)),
            corr=jnp.atleast_1d(row.corr),
            planet_id=jnp.zeros(1, dtype=int),
            is_valid=jnp.atleast_1d(valid & has_errors),
        )
        ll = loglike_relative_astrom(
            jnp.atleast_1d(budget.ra_arcsec), jnp.atleast_1d(budget.dec_arcsec), data
        )
        return jnp.where(valid & ~has_errors, -jnp.inf, ll)


def _noise(budget):
    """Standard deviation of the measured counts; 1 where the variance is not > 0."""
    variance = budget.variance
    return jnp.sqrt(jnp.where(variance > 0, variance, 1.0))


def sample_report(
    key,
    law,
    expected,
    limit_counts,
    *,
    lod_arcsec,
    time_d,
    occupied_s,
    calibration_id,
    valid_science=True,
):
    """Draw one summary row under `law` from the truth side's budget.

    Args:
        key: PRNG key; three standard normals are drawn (counts, then position).
        law: The `ReportingLaw`.
        expected: The true planet's `CountBudget` against the true scene over the
            realized science interval (`SummaryReport.expected`).
        limit_counts: The planet-free noise standard deviation of the model's
            nominal scene at the same interval (`ReportingLaw.limit_counts`).
        lod_arcsec: lambda/D at the band center [arcsec].
        time_d: Julian Date (TDB) of the science-interval midpoint.
        occupied_s: Executed occupancy, settling plus science [s].
        calibration_id: The declarations' digest (`calibration_id`).
        valid_science: False when the visit produced no certified science interval.

    Returns:
        A `SummaryRow`. A row whose science duration or limit is zero, or whose
        `valid_science` is false, has no science: zero counts and limit. Non-finite
        expected counts, variance or limit give NaN `detected`, `counts` and
        `limit_counts`, never a silent null.
    """
    eps = jax.random.normal(key, (3,), dtype=jnp.float64)
    limit = jnp.asarray(limit_counts, jnp.float64)
    science = jnp.asarray(expected.science_s, jnp.float64)
    valid = jnp.asarray(valid_science) & (science > 0) & (limit > 0)
    finite = (
        jnp.isfinite(expected.planet)
        & jnp.isfinite(expected.variance)
        & jnp.isfinite(limit)
    )
    counts = expected.planet + _noise(expected) * eps[0]
    detected = valid & (counts >= law.threshold * limit)
    counts = jnp.where(detected, counts, 0.0)
    snr = observed_snr(counts, limit)
    sigma = law.position_error(snr, lod_arcsec)
    rho = law.corr
    ra = expected.ra_arcsec + sigma * eps[1]
    dec = expected.dec_arcsec + sigma * (rho * eps[1] + math.sqrt(1 - rho**2) * eps[2])
    nan = jnp.asarray(jnp.nan)
    zero = jnp.asarray(0.0)
    return SummaryRow(
        time_d=jnp.asarray(time_d, jnp.float64),
        science_s=jnp.where(valid, science, zero),
        occupied_s=jnp.asarray(occupied_s, jnp.float64),
        detected=jnp.where(finite, detected.astype(jnp.float64), nan),
        counts=jnp.where(finite, counts, nan),
        limit_counts=jnp.where(finite, jnp.where(valid, limit, zero), nan),
        snr=snr,
        ra=jnp.where(detected, ra, zero),
        dec=jnp.where(detected, dec, zero),
        ra_err=sigma,
        dec_err=sigma,
        corr=jnp.asarray(rho, jnp.float64),
        cadence=jnp.asarray(expected.cadence, jnp.float64),
        calibration_id=jnp.asarray(calibration_id, jnp.float64),
        valid_science=valid.astype(jnp.float64),
    )


def loglike_report(law, row, predicted):
    """The log likelihood log p(row | theta) under `law`, from a predicted budget.

    `predicted` is `SummaryReport.predict(theta, interval)` over the row's science
    interval. A detection contributes the counts density on the detection region
    plus the position density with the row's own errors and correlation; a null
    contributes log P(null | theta) at the row's `limit_counts`; a row with no
    science contributes zero.
    """
    valid = row.valid_science > 0.5
    detected = valid & (row.detected > 0.5)
    log_det = law.log_counts_density(
        row.counts, row.limit_counts, predicted
    ) + law.log_position_density(row, predicted, detected)
    log_null = law.log_null_probability(row.limit_counts, predicted)
    out = jnp.where(detected, log_det, jnp.where(valid, log_null, 0.0))
    # A NaN detection flag (a non-finite draw) never reads as a null.
    return jnp.where(jnp.isnan(row.detected), jnp.nan, out)


def _canonical(value):
    """A platform-stable text form of a declaration (floats to 12 digits)."""
    if isinstance(value, dict):
        items = (f"{k}:{_canonical(value[k])}" for k in sorted(value))
        return "{" + ",".join(items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical(v) for v in value) + "]"
    if value is None or isinstance(value, (str, bool)):
        return repr(value)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return f"{float(array):.12g}"
    # Curves are digested to 9 digits so the last bits of a platform do not move it.
    return "[" + ",".join(f"{v:.9g}" for v in array.ravel()) + "]"


def _declaration(obj, skip=()):
    """Every field of a module, recursively, as plain values for `_canonical`.

    Modules and dataclasses become their type name plus their fields (static and
    dynamic) except those in `skip`; arrays become arrays; anything else (a
    callable) is named by its type.
    """
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, (jax.Array, np.ndarray, np.generic)):
        return np.asarray(obj)
    if dataclasses.is_dataclass(obj):
        out = {"type": type(obj).__name__}
        for f in dataclasses.fields(obj):
            if f.name not in skip:
                out[f.name] = _declaration(getattr(obj, f.name))
        return out
    if isinstance(obj, dict):
        return {str(k): _declaration(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_declaration(v) for v in obj]
    return type(obj).__name__


def calibration_id(report, law):
    """Six-byte digest of every declaration a predicted budget depends on.

    Covers the detector spec and its cadence; every field of the optical path the
    nominal scene is evaluated on (primary, attenuating elements, detector,
    disperser, `n_channels`, `npix_multiplier`, speckle) except the coronagraph,
    which enters through its working angles, pixel scale and its throughput, core
    area, occulter transmission and stellar leakage curves at the nominal host on
    the leakage grid; the observatory (its orbit sets the local zodiacal light, its
    temperature the thermal term); the band; the variance declarations and time
    nodes; the field radius; the nominal scene; and the law. The integer is below
    2^48, so a float64 payload column holds it exactly; it is never zero.
    """
    path = report.nominal.optical_path
    coronagraph = path.coronagraph
    grid = jnp.arange(0.0, report.leakage_max_lod, report.leakage_step_lod)
    wl = report.wavelength_nm
    declarations = {
        "detector": dataclasses.asdict(report.detector_spec),
        "optical_path": _declaration(path, skip=("coronagraph",)),
        "response": {
            "IWA": coronagraph.IWA,
            "OWA": coronagraph.OWA,
            "pixel_scale_lod": coronagraph.pixel_scale_lod,
            "throughput": coronagraph.throughput(grid, wl),
            "core_area": coronagraph.core_area(grid, wl),
            "occulter_transmission": coronagraph.occulter_transmission(grid, wl),
            "leakage": coronagraph.core_mean_intensity(grid, wl),
            "field_radius_lod": report.field_radius_lod,
            "leakage_step_lod": report.leakage_step_lod,
        },
        "observatory": _declaration(report.observatory),
        "band": {"wavelength_nm": wl, "bin_width_nm": report.bin_width_nm},
        "variance": {
            "bg_multiplier": report.bg_multiplier,
            "ppfact": "inf" if math.isinf(report.ppfact) else report.ppfact,
            "eps_warm_T_cold": report.eps_warm_T_cold,
            "nodes": report.nodes,
            "weights": report.weights,
            "orbit_epoch_jd": report.orbit_epoch_jd,
        },
        "nominal_scene": _declaration(report.nominal.scene),
        "law": {"threshold": law.threshold, "kappa": law.kappa, "corr": law.corr},
    }
    text = _canonical(declarations).encode("utf-8")
    digest = hashlib.blake2b(text, digest_size=CALIBRATION_ID_BYTES).digest()
    return int.from_bytes(digest, "big") or 1


# -- Summary channel -----------------------------------------------------------------

# Exozodi level the first profile passes explicitly (its `exozodi` convention).
PROFILE_N_EXOZODI = 3.0


class _ChannelAssets(eqx.Module):
    """What a summary channel builds from its declarations, once per declaration set."""

    report: SummaryReport
    context: eqx.Module
    law: ReportingLaw
    nominal: dict
    star_mass_kg: float
    epoch_jd: float
    calibration_id: int


@functools.lru_cache(maxsize=4)
def _channel_assets(profile, target, detector, n_time_nodes, n_exozodi, law):
    """Build the report, observing context and nominal planet a channel realizes with.

    The scene and the nominal planet come from the profile's target entry, the
    summary path from the pinned response, and the true scene is the nominal one
    (a declared simplification: the truth side differs from the model in the planet
    only). Cached on the channel's plain declarations, so a study that is rebuilt or
    reloaded shares one build.
    """
    from spaceodyssey.adapters.skyscapes import SceneSpec, build_star

    entry = load_profile(profile)
    spec = SceneSpec.from_profile(entry, target)
    context = context_from_profile(entry)
    epoch = entry["time"]["study_epoch_jd"]
    photons = build_star(spec).spec_flux_density(spec.wavelength_nm, epoch)
    scene = SummaryScene(
        star_flux_density_jy=float(
            photons_per_nm_per_m2_to_jy(photons, spec.wavelength_nm)
        ),
        dist_pc=spec.dist_pc,
        star_mass_kg=spec.mass_msun * Msun2kg,
        luminosity_lsun=spec.luminosity_lsun,
        ra_deg=spec.ra_deg,
        dec_deg=spec.dec_deg,
        star_diameter_mas=spec.angular_diameter_mas,
        n_exozodi=n_exozodi,
    )
    report = SummaryReport.from_yip(
        DetectorSpec(*detector),
        scene,
        orbit_epoch_jd=epoch,
        observatory=context.observatory,
        n_time_nodes=n_time_nodes,
    )
    mp = spec.planet_mass_mearth * Mearth2kg
    nominal = nominal_planet(
        a_au=float(period_to_sma(spec.period_d, scene.star_mass_kg + mp)),
        e=spec.ecc,
        inclination_rad=math.radians(spec.inclination_deg),
        W_rad=math.radians(spec.node_deg),
        w_rad=math.radians(spec.omega_planet_deg),
        tp_d=spec.tp_jd - epoch,
        Rp_au=spec.planet_radius_m / AU2m,
        Ag=spec.geometric_albedo,
    )
    nominal["Mp"] = jnp.asarray(mp)
    law = ReportingLaw(*law)
    return _ChannelAssets(
        report=report,
        context=context,
        law=law,
        nominal=nominal,
        star_mass_kg=scene.star_mass_kg,
        epoch_jd=epoch,
        calibration_id=calibration_id(report, law),
    )


def _build_assets(*declarations):
    """`_channel_assets`, built outside any trace the caller is in.

    The first call may come while a program is being traced (a reloaded study builds
    its channel at its first realization). Tracing state belongs to the thread, so
    the build runs on a fresh one, where it is the ordinary eager build; the response
    tables then equal an offline report's to the last bit, whichever call came first.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_channel_assets, *declarations).result()


class JaxedithSummaryChannel(AbstractChannelModel):
    """A version-2 channel model: one summary record per planned visit.

    Channel `i` of the study is planned visit `i`. At realization the record's own
    occupancy (`start_tick`, `end_tick` from the realize context, so a delayed
    admission or a cancellation moves them) gives the executed science interval:
    ``[start + settle, end]``, rechecked against the visit's certificate. The row is
    `sample_report` of the true planet's budget over that interval, with the
    planet-free limit of the nominal planet over the same interval, so the epoch is
    the midpoint of the science interval and an interval outside the certified
    window is a no-science row (`valid_science = 0`, occupancy still reported).
    Nothing scales counts by the occupancy fraction: the realized science duration
    is ``max(0, end - start - settle)`` ticks.

    The model holds plain declarations only, so a study document stores it; the
    report, the observing context and the nominal planet are built from them on
    first use (the pinned response through yippy's hash-checked loader) and shared
    by every model with the same declarations. `calibration` is the digest of what
    that build declares; a build that differs raises instead of writing rows under
    another calibration.

    The truth must be a `PhysicalTruth` (photomancy's physical names, one entry per
    subject); `subject` picks the entry. The payload is `SUMMARY_COLUMNS` wide.

    Attributes:
        profile: Profile label (`physical_profile`).
        target: Profile target index of the host.
        detector: `DetectorSpec` field values in field order.
        n_time_nodes: Gauss-Legendre nodes over the science interval.
        n_exozodi: Exozodi level in zodis.
        law: ``(threshold, kappa, corr)`` of the `ReportingLaw`.
        windows: Per channel, ``(window_start_tick, window_end_tick, outcome)`` of
            the visit's `Certificate`.
        calibration: The expected `calibration_id`.
    """

    version: int = eqx.field(default=2, kw_only=True)
    profile: str = eqx.field(static=True)
    target: int = eqx.field(static=True)
    detector: tuple = eqx.field(static=True)
    n_time_nodes: int = eqx.field(static=True)
    n_exozodi: float = eqx.field(static=True)
    law: tuple = eqx.field(static=True)
    windows: tuple = eqx.field(static=True)
    calibration: int = eqx.field(static=True)

    def __check_init__(self):
        """Refuse malformed declarations before anything is built."""
        load_profile(self.profile)
        if type(self.target) is not int or self.target < 0:
            raise ValueError("target must be a nonnegative integer")
        if type(self.n_time_nodes) is not int or self.n_time_nodes < 1:
            raise ValueError("n_time_nodes must be a positive integer")
        _finite("n_exozodi", self.n_exozodi, low=0.0)
        DetectorSpec(*self.detector)
        ReportingLaw(*self.law)
        if not self.windows:
            raise ValueError("a summary channel needs at least one planned visit")
        for window in self.windows:
            ok = len(window) == 3 and all(type(x) is int for x in window)
            if not ok or window[1] < window[0]:
                raise ValueError(f"a window is (start, end, outcome) ticks: {window!r}")
        if type(self.calibration) is not int or self.calibration < 1:
            raise ValueError("calibration must be a positive integer")

    @classmethod
    def from_plan(
        cls,
        visits,
        *,
        profile=FIRST_SUMMARY_V1,
        detector=None,
        law=None,
        n_time_nodes=8,
        n_exozodi=PROFILE_N_EXOZODI,
        support=None,
    ):
        """The channel model of a plan: one channel per `PlannedVisit`.

        Every visit is certified (`require_certified`, with `support` for the
        response support when given) and an uncertified plan raises
        `UncertifiedPlan`. The calibration digest is computed from the build.
        """
        visits = tuple(visits)
        if not visits:
            raise ValueError("a summary channel needs at least one planned visit")
        targets = {v.target for v in visits}
        if len(targets) != 1 or not all(isinstance(v, PlannedVisit) for v in visits):
            raise ValueError("every planned visit must name the same target")
        (target,) = targets
        detector = dataclasses.astuple(detector or DetectorSpec.fixture())
        law = law or ReportingLaw()
        law = (law.threshold, law.kappa, law.corr)
        args = (profile, target, detector, n_time_nodes, float(n_exozodi), law)
        assets = _build_assets(*args)
        certificates = require_certified(assets.context, visits, support)
        windows = tuple(
            (
                int(c.window_start_tick),
                int(c.window_end_tick),
                int(c.outcome),
            )
            for c in certificates
        )
        return cls(
            profile=profile,
            target=target,
            detector=detector,
            n_time_nodes=n_time_nodes,
            n_exozodi=float(n_exozodi),
            law=law,
            windows=windows,
            calibration=assets.calibration_id,
        )

    def assets(self):
        """The report, context, law and nominal planet this model realizes with."""
        built = _build_assets(
            self.profile,
            self.target,
            self.detector,
            self.n_time_nodes,
            self.n_exozodi,
            self.law,
        )
        if built.calibration_id != self.calibration:
            raise ValueError(
                f"the channel was declared under calibration {self.calibration} but "
                f"its declarations build calibration {built.calibration_id}"
            )
        return built

    def realize(
        self, truth, subject, channel_index, fraction, key, design, channels, *, context
    ):
        """The summary row of the executed science interval; `fraction` is unused."""
        built = self.assets()
        report, clock = built.report, built.context.clock
        window = jnp.asarray(self.windows, I64)[channel_index]
        certificate = Certificate(
            target=jnp.asarray(self.target, I64),
            window_start_tick=window[0],
            window_end_tick=window[1],
            outcome=window[2],
        )
        sci = built.context.realized_science(
            certificate, context.start_tick, context.end_tick
        )
        valid = sci.valid_science > 0
        # A no-science row still needs a finite budget to draw from; the row's flag
        # and zero science time clear every science column.
        frame_s = jnp.asarray(report.detector_spec.frame_time_s, jnp.float64)
        interval = ScienceInterval(
            sci.epoch_jd,
            jnp.where(valid, clock.seconds(sci.science_ticks), frame_s),
        )
        planet = _truth_planet(truth, subject, built)
        expected = report.budget(report.nominal, planet, interval)
        reference = report.predict(planet_free(built.nominal), interval)
        row = sample_report(
            key,
            built.law,
            expected,
            built.law.limit_counts(reference),
            lod_arcsec=report.lod_arcsec,
            time_d=sci.epoch_jd,
            occupied_s=clock.seconds(sci.occupied_ticks),
            calibration_id=built.calibration_id,
            valid_science=valid,
        )
        return row.payload()


def _truth_planet(truth, subject, built):
    """The true planet of `subject` in photomancy's physical names."""
    mp = truth.Mp_kg[subject]
    return {
        "a": period_to_sma(truth.period_d[subject], built.star_mass_kg + mp),
        "e": truth.e[subject],
        "cos_i": truth.cos_i[subject],
        "W": truth.W_rad[subject],
        "cos_w": truth.cos_w[subject],
        "sin_w": truth.sin_w[subject],
        "tp": truth.tp_jd[subject] - built.epoch_jd,
        "Lambda": truth.Ag[subject] * truth.Rp_AU[subject] ** 2,
        "Mp": mp,
    }
