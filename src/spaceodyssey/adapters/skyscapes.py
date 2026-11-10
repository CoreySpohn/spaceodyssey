"""Hand-built skyscapes scenes, the physical world source and truth-side diagnosis.

`SceneSpec` is the plain-data description of one target: the host star (mass,
distance, coordinates, luminosity, effective temperature, radius, angular
diameter) and one planet (period, eccentricity, the radial-velocity argument of
periastron, periastron epoch, inclination, node, mass, radius, geometric albedo).
`build_scene` turns it into a skyscapes `System`: a tabulated `Star` whose
single-epoch flux table is a skyscapes `BlackbodyStar` spectrum over the band, and
one `Planet` with an orbix `KeplerianOrbit` and a grey `LambertianPhysicalModel`.

Conventions are the libraries' own: positions in the right-handed (east, north,
away) frame, the east offset already multiplied by cos(declination), the node
measured from north through east, the ascending node receding, and the standard
phase angle (0 at superior conjunction). A radial-velocity solution gives the
star's argument of periastron; the planet's is that plus 180 degrees, so a scene
reproduces the published velocity curve of the star. Kepler's equation is solved
with orbix's Markley solver.

`PhysicalWorldSource` draws each subject's truth from photomancy's orbit prior
(`build_model` with the planet-scale sites active): period, eccentricity and
argument of periastron, isotropic orientation, phase, planet mass, and
log-uniform radius and albedo. Each subject is keyed by its index, so adding a
subject leaves the others' draws unchanged. photomancy's periastron time is
relative to its time origin, which the source places at `epoch_jd`.

`ExoVistaWorldSource` reads stored ExoVista universes (one file per star per
universe) into `PopulationTruth`: each star has `n_slots` planet slots with a
`present` mask, so the truth of one world is a fixed-shape array tree whatever
the number of planets a generator drew. Slots hold the planets inside a declared
`SupportRange`, ranked by peak flux ratio; `klass` is 1 for the planets
`skyscapes.io.get_earth_like_planet_indices` selects. The albedo of a stored planet
is the geometric albedo whose Lambert phase curve reproduces the file's contrast at
the band center at the first epoch (the file's own spectra are not carried).

`diagnose(truth, record, ...)` recomputes truth-side geometry for a record from
the truth and the record identity (subject, start and end ticks): the science
midpoint epoch, the sky offsets, the separation, the phase angle, the
star-planet distance and the contrast. It is for scoring and checks, never for a
policy, observer or refit.
"""

import dataclasses
import functools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from hwoutils.constants import (
    G,
    Mearth2kg,
    Mjup2kg,
    Msun2kg,
    Rearth2AU,
    Rearth2m,
    Rsun2AU,
    Rsun2m,
    deg2rad,
    mas2arcsec,
    pc2AU,
    rad2arcsec,
)
from hwoutils.conversions import photons_per_nm_per_m2_to_jy
from numpyro import handlers
from orbix.equations.orbit import mean_motion, period_a
from orbix.kepler import markley_solve_trig
from orbix.orbit import KeplerianOrbit
from photomancy.orbit.data import OrbitData
from photomancy.orbit.forward import predict_photometry
from photomancy.orbit.model import build_model
from skyscapes.io import from_exovista, get_earth_like_planet_indices
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.scene import BlackbodyStar, Planet, Star, System

from spaceodyssey.core.enums import F64
from spaceodyssey.core.kinds import AbstractWorldSource

# Wavelength samples of the stellar flux table across the band (inclusive).
FLUX_TABLE_POINTS = 21


@dataclasses.dataclass(frozen=True)
class SceneSpec:
    """One target: host star and planet, as plain data with units in the names.

    `omega_star_deg` is the star's argument of periastron as a radial-velocity
    solution reports it; the planet's is 180 degrees more. `diameter_mas` of
    None derives the angular diameter from the radius and distance.
    """

    name: str
    mass_msun: float
    dist_pc: float
    ra_deg: float
    dec_deg: float
    luminosity_lsun: float
    teff_k: float
    radius_rsun: float
    diameter_mas: float | None
    period_d: float
    ecc: float
    omega_star_deg: float
    tp_jd: float
    inclination_deg: float
    node_deg: float
    planet_mass_mearth: float
    planet_radius_m: float
    geometric_albedo: float
    band_nm: tuple[float, float]
    wavelength_nm: float

    @classmethod
    def from_profile(cls, profile, index):
        """The scene of target `index` in a profile table."""
        target = profile["targets"][index]
        planet = target["planet"]
        declared = planet["declared"]
        inclination = declared["inclination_deg"]
        if planet.get("msini_mjup") is not None:
            sin_i = math.sin(math.radians(inclination))
            mass_mearth = planet["msini_mjup"] * Mjup2kg / Mearth2kg / sin_i
        else:
            mass_mearth = declared["mass_mearth"]
        response = profile["response"]
        return cls(
            name=target["name"],
            mass_msun=target["mass_msun"],
            dist_pc=target["dist_pc"],
            ra_deg=target["ra_deg"],
            dec_deg=target["dec_deg"],
            luminosity_lsun=10.0 ** target["luminosity_log_lsun"],
            teff_k=target["teff_k"],
            radius_rsun=target["radius_rsun"],
            diameter_mas=target["diameter_mas"],
            period_d=planet["period_d"],
            ecc=planet["ecc"],
            omega_star_deg=planet["omega_star_deg"],
            tp_jd=planet["tp_jd"],
            inclination_deg=inclination,
            node_deg=declared["node_deg"],
            planet_mass_mearth=mass_mearth,
            planet_radius_m=declared["radius_m"],
            geometric_albedo=declared["geometric_albedo"],
            band_nm=tuple(response["band_nm"]),
            wavelength_nm=response["central_wavelength_nm"],
        )

    @property
    def omega_planet_deg(self):
        """The planet's argument of periastron: the star's plus 180 degrees."""
        return (self.omega_star_deg + 180.0) % 360.0

    @property
    def angular_diameter_mas(self):
        """The stated angular diameter, or twice the radius over the distance."""
        if self.diameter_mas is not None:
            return self.diameter_mas
        radians = 2.0 * self.radius_rsun * Rsun2AU / (self.dist_pc * pc2AU)
        return radians * rad2arcsec / mas2arcsec


@dataclasses.dataclass(frozen=True)
class PriorSpec:
    """photomancy's orbit-prior ranges for one subject (log10 of the stated units).

    `log_P_range` is log10 days, `log_Mp_range` log10 Earth masses, `log_Rp_range`
    log10 AU and `log_Ag_range` log10 geometric albedo; `ecc_prior` names a
    photomancy eccentricity prior.
    """

    log_P_range: tuple[float, float]
    ecc_prior: str
    log_Mp_range: tuple[float, float]
    log_Rp_range: tuple[float, float]
    log_Ag_range: tuple[float, float]

    @classmethod
    def from_profile(cls, profile, index):
        """The prior of target `index` in a profile table."""
        prior = profile["targets"][index]["planet"]["prior"]
        lo, hi = prior["period_range_d"]
        return cls(
            log_P_range=(math.log10(lo), math.log10(hi)),
            ecc_prior=prior["ecc_prior"],
            log_Mp_range=tuple(prior["log_mp_range"]),
            log_Rp_range=tuple(prior["log_rp_range"]),
            log_Ag_range=tuple(prior["log_ag_range"]),
        )


def build_star(spec):
    """A skyscapes `Star` for the host: blackbody flux table over the band."""
    ms_kg = spec.mass_msun * Msun2kg
    blackbody = BlackbodyStar(
        Ms_kg=ms_kg,
        dist_pc=spec.dist_pc,
        teff_K=spec.teff_k,
        radius_m=spec.radius_rsun * Rsun2m,
    )
    wavelengths = jnp.linspace(spec.band_nm[0], spec.band_nm[1], FLUX_TABLE_POINTS)
    times = jnp.asarray([spec.tp_jd], F64)
    photons = blackbody.spec_flux_density(wavelengths, times[0])
    flux_jy = photons_per_nm_per_m2_to_jy(photons, wavelengths)[:, None]
    return Star(
        Ms_kg=ms_kg,
        dist_pc=spec.dist_pc,
        wavelengths_nm=wavelengths,
        times_jd=times,
        flux_density_jy=flux_jy,
        ra_deg=spec.ra_deg,
        dec_deg=spec.dec_deg,
        diameter_arcsec=spec.angular_diameter_mas * mas2arcsec,
        luminosity_lsun=spec.luminosity_lsun,
        metadata={"name": spec.name},
        interpolation_method="linear",
    )


def _planet(
    star, period_d, ecc, cos_i, node_rad, cos_w, sin_w, tp_jd, mass_kg, r_au, ag
):
    """A one-planet skyscapes `Planet` from period-parameterized elements."""
    orbit = KeplerianOrbit.from_period(
        period_d,
        ecc,
        cos_i,
        node_rad,
        cos_w,
        sin_w,
        tp_jd,
        Ms_kg=star.Ms_kg + mass_kg,
    )
    return Planet(
        Rp_Rearth=jnp.atleast_1d(jnp.asarray(r_au / Rearth2AU, F64)),
        Mp_Mearth=jnp.atleast_1d(jnp.asarray(mass_kg / Mearth2kg, F64)),
        orbit=orbit,
        physical_model=LambertianPhysicalModel(Ag=jnp.atleast_1d(jnp.asarray(ag, F64))),
    )


def build_scene(spec):
    """The skyscapes `System` of a `SceneSpec`: its star and its one planet."""
    star = build_star(spec)
    w = spec.omega_planet_deg * deg2rad
    planet = _planet(
        star,
        spec.period_d,
        spec.ecc,
        jnp.cos(spec.inclination_deg * deg2rad),
        spec.node_deg * deg2rad,
        jnp.cos(w),
        jnp.sin(w),
        spec.tp_jd,
        spec.planet_mass_mearth * Mearth2kg,
        spec.planet_radius_m / Rearth2m * Rearth2AU,
        spec.geometric_albedo,
    )
    return System(star=star, planets=(planet,), trig_solver=markley_solve_trig)


class PhysicalTruth(eqx.Module):
    """One world's latent planets, one entry per subject.

    Attributes:
        period_d: Orbital period, days.
        e: Eccentricity.
        cos_i: Cosine of the inclination.
        W_rad: Longitude of the ascending node, radians.
        cos_w: Cosine of the planet's argument of periastron.
        sin_w: Sine of the planet's argument of periastron.
        tp_jd: Periastron epoch, TDB Julian Date.
        Mp_kg: Planet mass, kg.
        Rp_AU: Planet radius, AU.
        Ag: Geometric albedo.
    """

    period_d: jax.Array
    e: jax.Array
    cos_i: jax.Array
    W_rad: jax.Array
    cos_w: jax.Array
    sin_w: jax.Array
    tp_jd: jax.Array
    Mp_kg: jax.Array
    Rp_AU: jax.Array
    Ag: jax.Array


@functools.lru_cache(maxsize=16)
def _prior_model(prior):
    """Photomancy's prior-only orbit model for `prior`, built once per prior."""
    return build_model(
        sample_planet_scale=True,
        log_P_range=prior.log_P_range,
        log_Mp_range=prior.log_Mp_range,
        log_Rp_range=prior.log_Rp_range,
        log_Ag_range=prior.log_Ag_range,
        ecc_prior=prior.ecc_prior,
    )


class PhysicalWorldSource(AbstractWorldSource):
    """Truth drawn from photomancy's orbit prior, one subject per scene.

    The host of each subject is its scene's star (known, not drawn). Subject `s`
    draws with ``fold_in(key, s)``.

    Attributes:
        scenes: One `SceneSpec` per subject (the host star of each).
        priors: One `PriorSpec` per subject.
        epoch_jd: TDB Julian Date of photomancy's time origin.
    """

    scenes: tuple[SceneSpec, ...] = eqx.field(static=True)
    priors: tuple[PriorSpec, ...] = eqx.field(static=True)
    epoch_jd: float = eqx.field(static=True)

    def __check_init__(self):
        """Require one prior per scene and at least one subject."""
        if not self.scenes or len(self.scenes) != len(self.priors):
            raise ValueError("a world source needs one prior per scene, at least one")

    @classmethod
    def from_profile(cls, profile, targets=None):
        """The world source for the profile's targets (default all), study epoch."""
        chosen = range(len(profile["targets"])) if targets is None else targets
        return cls(
            scenes=tuple(SceneSpec.from_profile(profile, i) for i in chosen),
            priors=tuple(PriorSpec.from_profile(profile, i) for i in chosen),
            epoch_jd=profile["time"]["study_epoch_jd"],
        )

    def draw(self, key, capacities):
        """Draw one world's `PhysicalTruth`; the capacities fix the subject count."""
        if capacities.n_subjects != len(self.scenes):
            raise ValueError(
                f"capacities hold {capacities.n_subjects} subjects; "
                f"the world source has {len(self.scenes)} scenes"
            )
        draws = [
            self._draw_subject(jr.fold_in(key, s), s) for s in range(len(self.scenes))
        ]
        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *draws)

    def _draw_subject(self, key, subject):
        scene = self.scenes[subject]
        model = _prior_model(self.priors[subject])
        trace = handlers.trace(handlers.seed(model, key)).get_trace(
            scene.mass_msun * Msun2kg, scene.dist_pc, OrbitData()
        )

        def site(name):
            return jnp.squeeze(trace[name]["value"]).astype(F64)

        return PhysicalTruth(
            period_d=site("T"),
            e=site("e"),
            cos_i=site("cos_i"),
            W_rad=site("W"),
            cos_w=site("cos_w"),
            sin_w=site("sin_w"),
            tp_jd=self.epoch_jd + site("tp"),
            Mp_kg=site("Mp"),
            Rp_AU=site("Rp"),
            Ag=site("Ag"),
        )


def truth_scene(source, truth, subject):
    """The skyscapes `System` of one subject of a drawn world (truth side)."""
    star = build_star(source.scenes[subject])
    planet = _planet(
        star,
        truth.period_d[subject],
        truth.e[subject],
        truth.cos_i[subject],
        truth.W_rad[subject],
        truth.cos_w[subject],
        truth.sin_w[subject],
        truth.tp_jd[subject],
        truth.Mp_kg[subject],
        truth.Rp_AU[subject],
        truth.Ag[subject],
    )
    return System(star=star, planets=(planet,), trig_solver=markley_solve_trig)


class Geometry(eqx.Module):
    """Sky geometry of one planet at a set of epochs.

    Attributes:
        epoch_jd: TDB Julian Dates.
        east_arcsec: East offset Delta alpha cos(delta), arcsec.
        north_arcsec: North offset, arcsec.
        separation_arcsec: Projected separation, arcsec.
        phase_angle_rad: Standard phase angle, radians.
        distance_au: Star-planet distance, AU.
        contrast: Planet-to-star flux ratio at the scene's central wavelength.
    """

    epoch_jd: jax.Array
    east_arcsec: jax.Array
    north_arcsec: jax.Array
    separation_arcsec: jax.Array
    phase_angle_rad: jax.Array
    distance_au: jax.Array
    contrast: jax.Array


def geometry(system, epoch_jd, wavelength_nm):
    """`Geometry` of a one-planet `System` at `epoch_jd` (shape ``(T,)``)."""
    t = jnp.atleast_1d(jnp.asarray(epoch_jd, F64))
    planet = system.planets[0]
    _, phase, distance = planet.propagate(system.trig_solver, t, star=system.star)
    position = system.positions(t)
    east, north = position[0, 0], position[1, 0]
    contrast = system.contrasts(jnp.asarray(wavelength_nm, F64), t)[0]
    return Geometry(
        epoch_jd=t,
        east_arcsec=east,
        north_arcsec=north,
        separation_arcsec=jnp.hypot(east, north),
        phase_angle_rad=phase[0],
        distance_au=distance[0],
        contrast=contrast,
    )


class SeparationSupport(eqx.Module):
    """Response support of each target's nominal planet: separation in a band.

    Calling it as ``support(target, jd, margin_d=m)`` gives, per epoch, whether
    the nominal planet's projected separation stays in ``[iwa_arcsec,
    owa_arcsec]`` for every epoch within `m` days of it. The band is narrowed at
    each edge by `m` times an upper bound on the separation's rate of change, the
    largest sky-plane speed of the orbit, ``n a sqrt((1 + e) / (1 - e))`` (the
    speed at periastron) over the distance. With `m` half the sample spacing a
    pass at every sample is a pass for the whole sampled interval; it is the
    `support` predicate `ObservingContext.certify` takes.
    """

    systems: tuple[System, ...]
    wavelengths_nm: tuple[float, ...] = eqx.field(static=True)
    iwa_arcsec: float
    owa_arcsec: float

    def separation_rate_bound(self, target):
        """Upper bound on ``|d separation / dt|`` of `target`'s planet, arcsec/d."""
        system = self.systems[target]
        planet = system.planets[0]
        a_au, e = planet.orbit.a_AU[0], planet.orbit.e[0]
        mu = G * planet.central_mass_kg(star=system.star)[0]
        speed_au_per_d = mean_motion(a_au, mu) * a_au * jnp.sqrt((1 + e) / (1 - e))
        return speed_au_per_d * rad2arcsec / (system.star.dist_pc * pc2AU)

    def __call__(self, target, jd, margin_d=0.0):
        """Whether target `target`'s nominal planet stays in the support near `jd`."""
        g = geometry(self.systems[target], jd, self.wavelengths_nm[target])
        sep = g.separation_arcsec
        pad = self.separation_rate_bound(target) * margin_d
        return (sep >= self.iwa_arcsec + pad) & (sep <= self.owa_arcsec - pad)


class TruthDiagnosis(eqx.Module):
    """Truth-side geometry of one record at its science midpoint.

    Attributes:
        subject: The record's subject.
        has_science: Whether the record's occupancy left any science time.
        geometry: `Geometry` at the science-midpoint epoch (one epoch).
    """

    subject: int = eqx.field(static=True)
    has_science: jax.Array
    geometry: Geometry


def diagnose(truth, record, *, source, context):
    """Recompute truth-side geometry for `record` from `truth` and its identity.

    `record` is a core `Record`, or anything with `subject`, `start_tick` and
    `end_tick`; `context` is the `ObservingContext` that maps its ticks to the
    science-midpoint epoch. Only the record's identity is read, never its
    payload.
    """
    subject = int(record.subject)
    _, _, has_science = context.science_interval(record.start_tick, record.end_tick)
    epoch = context.science_midpoint_jd(record.start_tick, record.end_tick)
    system = truth_scene(source, truth, subject)
    return TruthDiagnosis(
        subject=subject,
        has_science=has_science,
        geometry=geometry(system, epoch, source.scenes[subject].wavelength_nm),
    )


# -- Stored populations: planet slots with a present mask ------------------------------

# The class number of the planets `get_earth_like_planet_indices` selects.
CLASS_ONE = 1
# Albedo of a stored planet whose Lambert phase is too small to read the file's
# contrast through (a planet at the crescent limit of its first epoch).
FALLBACK_ALBEDO = 0.3
_MIN_PHASE_FUNCTION = 0.05


@dataclasses.dataclass(frozen=True)
class SupportRange:
    """The orbits a belief can represent: a period range and an eccentricity ceiling.

    Planets outside it are dropped from the stored truth and counted.
    """

    period_d: tuple[float, float] = (50.0, 5000.0)
    e_max: float = 0.9

    def contains(self, period_d, e):
        """Whether a planet of period `period_d` and eccentricity `e` is inside."""
        low, high = self.period_d
        return bool(low <= period_d <= high and e < self.e_max)


@dataclasses.dataclass(frozen=True)
class StarSpec:
    """A host star as the study uses it: plain numbers, one per subject.

    `flux_density_jy` is the stellar flux density at the band center, read from the
    star the universe files hold.
    """

    name: str
    mass_msun: float
    dist_pc: float
    ra_deg: float
    dec_deg: float
    luminosity_lsun: float
    diameter_mas: float
    flux_density_jy: float


class PopulationTruth(PhysicalTruth):
    """One world's planets: `n_slots` per star, with a present mask and a class.

    Every `PhysicalTruth` leaf has shape ``[n_subjects, n_slots]`` (a batch of worlds
    adds a leading axis). An absent slot (`present` false) holds an inert filler
    planet: a circular one-year orbit with zero radius, so every kernel stays finite
    and the mask alone says whether it exists.

    Attributes:
        present: Whether the slot holds a planet.
        klass: 1 for a planet `skyscapes.io.get_earth_like_planet_indices` selects,
            0 for any other planet and for an absent slot.
    """

    present: jax.Array
    klass: jax.Array


_FILLER = {
    "period_d": 365.25,
    "e": 0.0,
    "cos_i": 0.0,
    "W_rad": 0.0,
    "cos_w": 1.0,
    "sin_w": 0.0,
    "Mp_kg": 0.0,
    "Rp_AU": 0.0,
    "Ag": 0.0,
}
_TRUTH_FIELDS = tuple(f.name for f in dataclasses.fields(PhysicalTruth))


def _lambert_albedo(system, planet, wavelength_nm):
    """Geometric albedo that gives the file's contrast at `wavelength_nm`, first epoch.

    Photomancy's Lambert phase curve with the planet's own radius, read at the
    planet's first epoch; `FALLBACK_ALBEDO` where the phase function is too small.
    """
    one = eqx.tree_at(lambda s: s.planets, system, (planet,))
    t0 = jnp.atleast_1d(planet.orbit.t0_d[:1])
    measured = float(geometry(one, t0, wavelength_nm).contrast[0])
    orbit = planet.orbit
    mass_kg = float(planet.Mp_Mearth[0]) * Mearth2kg
    star_kg = float(system.star.Ms_kg)
    a_au = float(orbit.a_AU[0])
    n = mean_motion(a_au, G * (star_kg + mass_kg))
    rp_au = float(planet.Rp_Rearth[0]) * Rearth2AU

    _, dmag = predict_photometry(
        jnp.zeros(1),
        a_au,
        float(orbit.e[0]),
        math.cos(float(orbit.i_rad[0])),
        float(orbit.W_rad[0]),
        math.cos(float(orbit.w_rad[0])),
        math.sin(float(orbit.w_rad[0])),
        -float(orbit.M0_rad[0]) / float(n),
        star_kg,
        rp_au**2,
        float(system.star.dist_pc),
        Mp=mass_kg,
    )
    unit = float(10.0 ** (-0.4 * dmag[0]))
    if not unit > 0.0 or not math.isfinite(unit):
        return FALLBACK_ALBEDO
    albedo = measured / unit
    return albedo if albedo > 1e-3 and math.isfinite(albedo) else FALLBACK_ALBEDO


def _stored_planets(path, system, wavelength_nm):
    """Each planet of one file: the ten truth leaves, `klass` and a ranking `peak`."""
    earth = set(get_earth_like_planet_indices(path))
    star_kg = float(system.star.Ms_kg)
    out = []
    for k, planet in enumerate(system.planets):
        orbit = planet.orbit
        a_au, e = float(orbit.a_AU[0]), float(orbit.e[0])
        mass_kg = float(planet.Mp_Mearth[0]) * Mearth2kg
        mu = G * (star_kg + mass_kg)
        n = float(mean_motion(a_au, mu))
        rp_au = float(planet.Rp_Rearth[0]) * Rearth2AU
        albedo = _lambert_albedo(system, planet, wavelength_nm)
        w = float(orbit.w_rad[0])
        out.append(
            {
                "period_d": float(period_a(a_au, mu)),
                "e": e,
                "cos_i": math.cos(float(orbit.i_rad[0])),
                "W_rad": float(orbit.W_rad[0]),
                "cos_w": math.cos(w),
                "sin_w": math.sin(w),
                "tp_jd": float(orbit.t0_d[0]) - float(orbit.M0_rad[0]) / n,
                "Mp_kg": mass_kg,
                "Rp_AU": rp_au,
                "Ag": albedo,
                "klass": CLASS_ONE if k in earth else 0,
                # flux ratio at full phase and periastron: an upper bound, for ranking
                "peak": albedo * (rp_au / (a_au * (1.0 - e))) ** 2,
            }
        )
    return out


def _star_spec(system, hip_name, wavelength_nm, epoch_jd):
    """The `StarSpec` of a loaded system's star."""
    star = system.star
    photons = star.spec_flux_density(jnp.asarray(wavelength_nm, F64), epoch_jd)
    flux_jy = photons_per_nm_per_m2_to_jy(jnp.ravel(photons)[0], wavelength_nm)
    return StarSpec(
        name=hip_name,
        mass_msun=float(star.Ms_kg) / Msun2kg,
        dist_pc=float(star.dist_pc),
        ra_deg=float(star.ra_deg),
        dec_deg=float(star.dec_deg),
        luminosity_lsun=float(star.luminosity_lsun),
        diameter_mas=float(star.diameter_arcsec) / mas2arcsec,
        flux_density_jy=float(flux_jy),
    )


class ExoVistaWorldSource(AbstractWorldSource):
    """Stored ExoVista universes as a world source: one stored universe per draw.

    The arrays hold every universe: `population` is a `PopulationTruth` whose leaves
    are ``[n_worlds, n_subjects, n_slots]``. `stars` are the hosts (the same in every
    universe, checked at construction). `dropped` counts, per universe and star, the
    planets that did not reach a slot (outside the support, or more than `n_slots`
    inside it); `dropped_class1` counts the class-1 ones among them; `occupancy`
    counts the class-1 planets that did.

    Attributes:
        population: The stored `PopulationTruth`, world axis first.
        dropped: ``int[n_worlds, n_subjects]``.
        dropped_class1: ``int[n_worlds, n_subjects]``.
        occupancy: ``int[n_worlds, n_subjects]`` class-1 planets in slots.
        stars: One `StarSpec` per subject.
        epoch_jd: Julian Date of the files' first epoch (the orbit epoch).
        support: The `SupportRange` the slots were filled under.
    """

    population: PopulationTruth
    dropped: jax.Array
    dropped_class1: jax.Array
    occupancy: jax.Array
    stars: tuple[StarSpec, ...] = eqx.field(static=True)
    epoch_jd: float = eqx.field(static=True)
    support: SupportRange = eqx.field(static=True)

    @property
    def n_worlds(self):
        """Number of stored universes."""
        return int(self.population.period_d.shape[0])

    @property
    def n_slots(self):
        """Planet slots per star."""
        return int(self.population.period_d.shape[2])

    @classmethod
    def from_files(
        cls,
        files,
        *,
        n_slots,
        support=None,
        wavelength_nm=1000.0,
        epoch_jd=None,
    ):
        """Read `files[w][s]`, star `s` of universe `w`, into a world source.

        Each file is read with `skyscapes.io.from_exovista(path, load_disk=False)`,
        which loads one system. Slots hold the planets inside `support`, the first
        `n_slots` by descending peak flux ratio (full phase at periastron). The slot
        index therefore carries the existence ordering, so slots are exchangeable to
        a belief, which must not read meaning into the index. The stars must agree
        across universes.
        """
        support = SupportRange() if support is None else support
        n_worlds, n_stars = len(files), len(files[0])
        if any(len(row) != n_stars for row in files):
            raise ValueError("every universe needs one file per star")
        leaves = {n: np.zeros((n_worlds, n_stars, n_slots)) for n in _TRUTH_FIELDS}
        for name, value in _FILLER.items():
            leaves[name][...] = value
        leaves["tp_jd"][...] = 0.0
        present = np.zeros((n_worlds, n_stars, n_slots), bool)
        klass = np.zeros((n_worlds, n_stars, n_slots), np.int64)
        dropped = np.zeros((n_worlds, n_stars), np.int64)
        dropped1 = np.zeros((n_worlds, n_stars), np.int64)
        occupancy = np.zeros((n_worlds, n_stars), np.int64)
        stars = [None] * n_stars
        epoch = epoch_jd
        for w in range(n_worlds):
            for s in range(n_stars):
                path = str(files[w][s])
                system = from_exovista(path, load_disk=False)
                t0 = float(system.planets[0].orbit.t0_d[0]) if system.planets else 0.0
                epoch = t0 if epoch is None else epoch
                spec = _star_spec(system, _hip_name(path), wavelength_nm, epoch)
                if stars[s] is None:
                    stars[s] = spec
                elif stars[s] != spec:
                    raise ValueError(f"star {s} differs between universes")
                planets = _stored_planets(path, system, wavelength_nm)
                inside = [p for p in planets if support.contains(p["period_d"], p["e"])]
                inside.sort(key=lambda p: -p["peak"])
                kept = inside[:n_slots]
                dropped[w, s] = len(planets) - len(kept)
                dropped1[w, s] = sum(p["klass"] for p in planets) - sum(
                    p["klass"] for p in kept
                )
                occupancy[w, s] = sum(p["klass"] for p in kept)
                for k, p in enumerate(kept):
                    present[w, s, k] = True
                    klass[w, s, k] = p["klass"]
                    for name in _TRUTH_FIELDS:
                        leaves[name][w, s, k] = p[name]
        population = PopulationTruth(
            **{n: jnp.asarray(v, F64) for n, v in leaves.items()},
            present=jnp.asarray(present),
            klass=jnp.asarray(klass),
        )
        return cls(
            population=population,
            dropped=jnp.asarray(dropped),
            dropped_class1=jnp.asarray(dropped1),
            occupancy=jnp.asarray(occupancy),
            stars=tuple(stars),
            epoch_jd=float(epoch),
            support=support,
        )

    def take(self, indices):
        """The `PopulationTruth` of universes `indices` (an int or an index array)."""
        index = jnp.asarray(indices)
        return jax.tree_util.tree_map(lambda x: x[index], self.population)

    def draw(self, key, capacities):
        """One stored universe, chosen uniformly by `key`."""
        if capacities.n_subjects != len(self.stars):
            raise ValueError(
                f"capacities hold {capacities.n_subjects} subjects; "
                f"the world source has {len(self.stars)} stars"
            )
        return self.take(jr.randint(key, (), 0, self.n_worlds))


def _hip_name(path):
    """`HIP <number>` from an ExoVista file name (``<id>-HIP_<number>-...``)."""
    stem = str(path).rsplit("/", 1)[-1]
    return "HIP " + stem.split("HIP_", 1)[1].split("-", 1)[0]


def population_system(source, truth, subject):
    """The skyscapes `System` of one star's slots (truth side); absent are fillers.

    `truth` is one world's `PopulationTruth` (no world axis). Planet `k` of the
    system is slot `k`.
    """
    spec = source.stars[subject]
    ms_kg = spec.mass_msun * Msun2kg
    star = Star(
        Ms_kg=ms_kg,
        dist_pc=spec.dist_pc,
        wavelengths_nm=jnp.asarray([900.0, 1100.0], F64),
        times_jd=jnp.asarray([source.epoch_jd], F64),
        flux_density_jy=jnp.full((2, 1), spec.flux_density_jy, F64),
        ra_deg=spec.ra_deg,
        dec_deg=spec.dec_deg,
        diameter_arcsec=spec.diameter_mas * mas2arcsec,
        luminosity_lsun=spec.luminosity_lsun,
        metadata={"name": spec.name},
        interpolation_method="linear",
    )
    planets = tuple(
        _planet(
            star,
            truth.period_d[subject, k],
            truth.e[subject, k],
            truth.cos_i[subject, k],
            truth.W_rad[subject, k],
            truth.cos_w[subject, k],
            truth.sin_w[subject, k],
            truth.tp_jd[subject, k],
            truth.Mp_kg[subject, k],
            truth.Rp_AU[subject, k],
            truth.Ag[subject, k],
        )
        for k in range(truth.period_d.shape[1])
    )
    return System(star=star, planets=planets, trig_solver=markley_solve_trig)
