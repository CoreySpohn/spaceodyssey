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
from orbix.equations.orbit import mean_motion
from orbix.kepler import markley_solve_trig
from orbix.orbit import KeplerianOrbit
from photomancy.orbit.data import OrbitData
from photomancy.orbit.model import build_model
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
        """Upper bound on |d separation / dt| of target `target`'s planet, arcsec/d."""
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
