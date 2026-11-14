"""Hand-built skyscapes scenes, the physical world source and truth-side diagnosis.

`SceneSpec` is the plain-data description of one target: the host star (mass,
distance, coordinates, luminosity, effective temperature, radius, angular
diameter) and one planet (period, eccentricity, the radial-velocity argument of
periastron, periastron epoch, inclination, node, mass, radius, geometric albedo).
`build_scene` turns it into a skyscapes `System`: a tabulated `Star` whose
single-epoch flux table is a skyscapes `BlackbodyStar` spectrum over the band, and
one `Planet` with an orbix `KeplerianOrbit` and a gray `LambertianPhysicalModel`.

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
the number of planets a generator drew. Slots hold every planet up to the capacity
`n_slots`, ranked by peak flux ratio, whether or not a belief can represent its orbit;
Every source declares the `PopulationModel` its worlds come from (`generator`), which
a decision view hands to beliefs in place of the worlds. Every wavelength is a declared
`Band`, and the truth carries one geometric albedo per declared band
(`PopulationTruth.albedo`). The albedo of a stored planet in a band is the geometric
albedo whose Lambert phase curve reproduces the file's contrast at the band center at
the first epoch (the file's own spectra are not carried); a band the file does not
cover is refused.

`ExoVistaWorldSource.from_draws` builds the same arrays without files: it draws each
world's planets with `skyscapes.population.draw_population` (the occurrence-rate
generator, pure JAX) and fills slots by the same rules. The albedo there is the
planet type's geometric albedo at each band center, and the star's flux comes from a
PHOENIX model atmosphere (`star_spec_from_model`), not from a scene file.

`diagnose(truth, record, ...)` recomputes truth-side geometry for a record from
the truth and the record identity (subject, start and end ticks): the science
midpoint epoch, the sky offsets, the separation, the phase angle, the
star-planet distance and the contrast. It is for scoring and checks, never for a
policy, observer or refit.
"""

import dataclasses
import functools
import itertools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from hwoutils.constants import (
    J2000_JD,
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
from skyscapes.io import from_exovista, from_phoenix
from skyscapes.physical_model import LambertianPhysicalModel
from skyscapes.population import (
    OccurrenceTable,
    PlanetTypes,
    PopulationConfig,
    draw_population,
    load_occurrence_table,
    load_planet_types,
    mass_to_radius,
    sky_elements,
)
from skyscapes.scene import BlackbodyStar, Planet, Star, System

from spaceodyssey.adapters.parameters import PlanetParameters
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
        luminosity_Lsun=spec.luminosity_lsun,
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

# Albedo of a stored planet whose Lambert phase is too small to read the file's
# contrast through (a planet at the crescent limit of its first epoch).
FALLBACK_ALBEDO = 0.3
_MIN_PHASE_FUNCTION = 0.05


@dataclasses.dataclass(frozen=True)
class SupportRange:
    """The orbits a belief can represent: a period range and an eccentricity ceiling.

    It belongs to a `PopulationModel` and bounds a refit's prior. The truth keeps the
    planets outside it, and a refit fits such a planet against the support.
    """

    period_d: tuple[float, float] = (50.0, 5000.0)
    e_max: float = 0.9

    def contains(self, period_d, e):
        """Whether a planet of period `period_d` and eccentricity `e` is inside."""
        low, high = self.period_d
        return bool(low <= period_d <= high and e < self.e_max)


@dataclasses.dataclass(frozen=True)
class Band:
    """A declared observing band: its center and full width, nm.

    Every wavelength the truth or a host star is read at is a declared band; none
    is a default. Construction refuses a non-finite or nonpositive center or width.
    """

    center_nm: float
    width_nm: float

    def __post_init__(self):
        """Refuse a non-finite or nonpositive center or width."""
        for name in ("center_nm", "width_nm"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"a band needs a finite positive {name}, got {value}")


# Half width added on each side of a population's range of ln(albedo radius^2), so a
# planet at the edge of the population sits inside a refit's scale prior.
LN_LAM_PAD = 0.5
# Log-spaced masses at which the forecaster radius is read to find the radii an
# occurrence grid can draw (the radius is not monotonic in mass above the Jovian
# break, so the edges alone do not bound it).
RADIUS_GRID_POINTS = 4097


class PopulationModel(eqx.Module):
    """A declared planet population: the generator's tables and a belief's support.

    `occurrence`, `types` and `config` are the inputs of
    `skyscapes.population.draw_population`: a world source draws its worlds from them
    and a belief starts from them. `support` is the range of orbits a belief can
    represent, which bounds a refit's period prior; the truth keeps the planets
    outside it.

    Attributes:
        occurrence: The occurrence grid planets are drawn from.
        types: The planet types and their albedo spectra.
        config: The settings of the draw.
        support: The orbits a belief can represent.
    """

    occurrence: OccurrenceTable
    types: PlanetTypes
    config: PopulationConfig = eqx.field(static=True)
    support: SupportRange = eqx.field(static=True)
    n_pool_systems: int = eqx.field(static=True, default=256)

    def __check_init__(self):
        """Require a nonempty bounded draw pool."""
        if self.n_pool_systems < 1:
            raise ValueError("n_pool_systems is at least one")

    @classmethod
    def exovista(cls):
        """The ExoVista generator: its occurrence grid and types, `SupportRange()`."""
        return cls(
            occurrence=load_occurrence_table(),
            types=load_planet_types(),
            config=PopulationConfig(),
            support=SupportRange(),
        )

    @property
    def period_support_d(self):
        """`(low, high)` of the period a belief can represent, days."""
        low, high = self.support.period_d
        return float(low), float(high)

    def _supported_systems(self, key, star, n_systems):
        """Read host scalars before entering the compiled population draw."""
        return self._draw_supported_systems(
            key,
            jnp.asarray(star.mass_msun),
            jnp.asarray(star.luminosity_lsun),
            n_systems,
        )

    @eqx.filter_jit
    def _draw_supported_systems(self, key, mass_msun, luminosity_lsun, n_systems):
        """Draw a fixed pool of identical hosts and mask planets outside support."""
        mass = jnp.full(n_systems, mass_msun)
        population = draw_population(
            key,
            self.occurrence,
            self.types,
            jnp.full(n_systems, luminosity_lsun),
            mass,
            self.config,
        )
        sky = sky_elements(population, mass)
        low, high = self.period_support_d
        valid = (
            population.present
            & (sky.period_d >= low)
            & (sky.period_d <= high)
            & (sky.e < self.support.e_max)
        )
        return population, sky, valid

    def draw_planets(self, key, star: "StarSpec", n, bands):
        """Return physical draws [n] and a validity mask from a bounded system pool.

        Keep the first n supported present planets in system/slot order. When the
        pool is exhausted, cycle its supported planets to fill the fixed shape;
        the mask marks only the original entries. An empty pool returns finite
        fillers with an all-false mask. The pool size is n_pool_systems.
        """
        population, sky, valid = self._supported_systems(key, star, self.n_pool_systems)
        count = valid.sum()
        order = jnp.argsort(~valid.ravel(), stable=True)
        selected = order[jnp.arange(n) % jnp.maximum(count, 1)]

        def take(x):
            return x.ravel()[selected]

        period_d = take(sky.period_d)
        albedo = jnp.stack(
            [
                take(self.types.geometric_albedo(population.type_index, b.center_nm))
                for b in bands
            ],
            axis=-1,
        )
        p = PlanetParameters(
            ln_period_d=jnp.log(period_d),
            ecc=take(sky.e),
            cos_i=jnp.cos(take(sky.i_rad)),
            node_rad=take(sky.W_rad),
            argp_rad=take(sky.w_rad),
            tau_d=-take(sky.M0_rad) * period_d / (2 * jnp.pi),
            star_mass_msun=jnp.full(n, star.mass_msun),
            planet_mass_mearth=take(population.mass_Mearth),
            radius_rearth=take(population.radius_Rearth),
            albedo=albedo,
        )
        # Positive filler scale keeps the physical-prior likelihood finite.
        p = eqx.tree_at(
            lambda x: (x.radius_rearth, x.albedo),
            p,
            (
                jnp.where(count > 0, p.radius_rearth, 1.0),
                jnp.where(count > 0, p.albedo, 1.0),
            ),
        )
        return p, jnp.arange(n) < count

    def expected_planets(self, star, key, n_systems):
        """Mean supported present count per host over n_systems drawn systems."""
        _, _, valid = self._supported_systems(key, star, n_systems)
        return valid.sum() / jnp.asarray(n_systems, jnp.float64)

    def albedo_prior(self, key, n, bands):
        """Draw [n, B] spectra from the occurrence prior conditioned on selectability.

        Cells have probability proportional to their nonnegative occurrence rates
        (only binned cells). Scaled semimajor axis and mass are log-uniform within
        a cell. At that location the eligible types' weights, including the configured
        candidate probability, are normalized. Locations without a selectable type
        are rejected. This is the pre-stability occurrence mixture; it is not
        conditioned on a host, the belief's orbital support, or observed reports.
        One type supplies all bands of a draw.
        """
        t, o = self.types, self.occurrence
        log_a, log_m = jnp.log(o.a_edges_AU), jnp.log(o.mass_edges_Mearth)
        rates = jnp.where(o.bin_id >= 0, o.rate, 0.0).ravel()
        cumulative = jnp.cumsum(rates)

        def draw(carry):
            key, indices, done = carry
            key, kc, ka, km, kt = jr.split(key, 5)
            cell = jnp.searchsorted(
                cumulative, jr.uniform(kc, (n,)) * cumulative[-1], side="right"
            )
            ia, im = cell // o.rate.shape[1], cell % o.rate.shape[1]
            a = jnp.exp(log_a[ia] + jr.uniform(ka, (n,)) * (log_a[ia + 1] - log_a[ia]))
            mass = jnp.exp(
                log_m[im] + jr.uniform(km, (n,)) * (log_m[im + 1] - log_m[im])
            )
            radius = mass_to_radius(mass)
            lower = jnp.where(
                t.earth_candidate,
                t.radius_min_Rearth / jnp.sqrt(a[:, None]),
                t.radius_min_Rearth,
            )
            fits = (
                (a[:, None] > t.a_min_AU)
                & (a[:, None] < t.a_max_AU)
                & (radius[:, None] > lower)
                & (radius[:, None] < t.radius_max_Rearth)
            )
            candidate = jnp.any(fits & t.earth_candidate, axis=-1, keepdims=True)
            p = self.config.earth_candidate_probability
            factor = jnp.where(t.earth_candidate, p, 1.0 - p)
            weights = fits * t.weight * jnp.where(candidate, factor, 1.0)
            cum = jnp.cumsum(weights, axis=-1)
            selected = jnp.sum(cum <= jr.uniform(kt, (n, 1)) * cum[:, -1:], axis=-1)
            valid = cum[:, -1] > 0.0
            return key, jnp.where(~done & valid, selected, indices), done | valid

        _, indices, _ = jax.lax.while_loop(
            lambda c: ~jnp.all(c[2]),
            draw,
            (key, jnp.zeros(n, jnp.int64), jnp.zeros(n, bool)),
        )
        return jnp.stack(
            [t.geometric_albedo(indices, b.center_nm) for b in bands], axis=-1
        )

    def _selectable_types(self):
        """Host-side reachability through positive-rate cells and type eligibility."""
        o, t = self.occurrence, self.types
        ia, im = np.nonzero((np.asarray(o.rate) > 0) & (np.asarray(o.bin_id) >= 0))
        a_edges = np.asarray(o.a_edges_AU)
        m_edges = np.asarray(o.mass_edges_Mearth)
        m_lo, m_hi = m_edges[im], m_edges[im + 1]
        r0, r1 = mass_to_radius(m_lo), mass_to_radius(m_hi)
        r_lo, r_hi = np.minimum(r0, r1), np.maximum(r0, r1)
        r_hi = np.where((m_lo < 131.6) & (m_hi > 131.6), mass_to_radius(131.6), r_hi)
        candidates = np.asarray(t.earth_candidate)
        p = float(self.config.earth_candidate_probability)
        selected = np.zeros(len(t.names), bool)
        for k in range(len(t.names)):
            if float(t.weight[k]) <= 0 or (candidates[k] and p == 0):
                continue
            low = np.maximum(a_edges[ia], float(t.a_min_AU[k]))
            high = np.minimum(a_edges[ia + 1], float(t.a_max_AU[k]))
            lower = float(t.radius_min_Rearth[k])
            radius_low = np.maximum(
                r_lo, lower / np.sqrt(high) if candidates[k] else lower
            )
            radius_high = np.minimum(r_hi, float(t.radius_max_Rearth[k]))
            eligible = (low < high) & (radius_low < radius_high)
            if p < 1 or candidates[k]:
                selected[k] = np.any(eligible)
                continue
            # At p=1 ordinary types are selectable only outside candidate coverage.
            for cell in np.flatnonzero(eligible):
                cuts = np.unique(
                    np.clip(
                        np.concatenate(
                            (
                                [low[cell], high[cell]],
                                np.asarray(t.a_min_AU)[candidates],
                                np.asarray(t.a_max_AU)[candidates],
                            )
                        ),
                        low[cell],
                        high[cell],
                    )
                )
                for left, right in itertools.pairwise(cuts):
                    mid = np.sqrt(left * right)
                    active = (
                        candidates
                        & (np.asarray(t.a_min_AU) < mid)
                        & (np.asarray(t.a_max_AU) > mid)
                    )
                    intervals = sorted(
                        zip(
                            np.asarray(t.radius_min_Rearth)[active] / np.sqrt(left),
                            np.asarray(t.radius_max_Rearth)[active],
                            strict=True,
                        )
                    )
                    edge = radius_low[cell]
                    for lo, hi in intervals:
                        if lo > edge:
                            break
                        edge = max(edge, hi)
                    if edge < radius_high[cell]:
                        selected[k] = True
                        break
                if selected[k]:
                    break
        return selected

    def check_bands(self, bands):
        """Refuse a declared band with a nonpositive or non-finite selectable albedo."""
        bands = _declared(bands)
        selectable = self._selectable_types()
        if not selectable.any():
            if not np.any(np.asarray(self.occurrence.rate) > 0):
                return
            raise ValueError(
                "the population has no selectable types in positive-rate cells"
            )
        for band in bands:
            albedo = np.asarray(
                self.types.geometric_albedo(
                    np.arange(len(self.types.names)), band.center_nm
                )
            )
            bad = selectable & (~np.isfinite(albedo) | (albedo <= 0))
            if bad.any():
                names = [
                    name
                    for name, invalid in zip(self.types.names, bad, strict=True)
                    if invalid
                ]
                raise ValueError(
                    f"nonpositive or non-finite albedo at {band.center_nm:g} nm "
                    f"for selectable types {names}; repair the population explicitly"
                )

    def radius_range_rearth(self):
        """`(low, high)` of the planet radius the population can draw, Earth radii.

        A drawn radius is the forecaster radius of a mass inside the occurrence grid
        (read at `RADIUS_GRID_POINTS` log-spaced masses and the grid edges), and lies
        inside some type's radius range; an exoEarth-candidate type's lower edge is
        divided by ``sqrt(a_max_AU)``, its smallest value over the type's range.
        """
        edges = np.asarray(self.occurrence.mass_edges_Mearth, np.float64)
        masses = np.concatenate(
            [edges, np.geomspace(edges.min(), edges.max(), RADIUS_GRID_POINTS)]
        )
        drawn = np.asarray(mass_to_radius(masses))
        t = self.types
        lower = np.where(
            np.asarray(t.earth_candidate),
            np.asarray(t.radius_min_Rearth) / np.sqrt(np.asarray(t.a_max_AU)),
            np.asarray(t.radius_min_Rearth),
        )
        low = max(float(lower.min()), float(drawn.min()))
        high = min(float(np.asarray(t.radius_max_Rearth).max()), float(drawn.max()))
        return low, high

    def scale_range_au2(self, band, pad=LN_LAM_PAD):
        """`(low, high)` of ``ln(albedo * radius^2)``, radius in AU, padded by `pad`.

        Radius bounds intersect the type bounds with radii reached by the forecaster
        mass-radius relation over the occurrence mass grid.
        The low end is the smallest geometric albedo of any type at the band center
        with the smallest radius (`radius_range_rearth`), the high end the largest
        albedo with the largest radius. It is a refit's scale prior: every planet the
        population can draw in `band` lies inside it.
        """
        self.check_bands((band,))
        types = self.types
        selectable = self._selectable_types()
        if not selectable.any():
            # Only the inert first-tier filler is evaluated for a zero-rate prior.
            center = math.log(Rearth2AU**2)
            return center - pad, center + pad
        albedo = np.asarray(
            types.geometric_albedo(np.arange(len(types.names)), band.center_nm)
        )[selectable]
        r_lo, r_hi = (r * Rearth2AU for r in self.radius_range_rearth())
        low = math.log(float(albedo.min()) * r_lo**2) - pad
        high = math.log(float(albedo.max()) * r_hi**2) + pad
        return low, high


def _declared(bands):
    """`bands` as a tuple of distinct `Band`s, at least one; refuse anything else."""
    bands = tuple(bands)
    if not bands or not all(isinstance(b, Band) for b in bands):
        raise ValueError("bands must be one or more `Band`s")
    centers = [b.center_nm for b in bands]
    if len(set(centers)) != len(centers):
        raise ValueError(f"declared bands share a center: {centers}")
    return bands


@dataclasses.dataclass(frozen=True)
class StarSpec:
    """A host star as the study uses it: plain numbers, one per subject.

    `band_fluxes_jy` holds the mean flux density of every declared band as
    ``(center_nm, flux_density_jy)`` pairs, in the declared order (`flux_at` reads
    them).
    """

    name: str
    mass_msun: float
    dist_pc: float
    ra_deg: float
    dec_deg: float
    luminosity_lsun: float
    diameter_mas: float
    band_fluxes_jy: tuple = ()

    def flux_at(self, wavelength_nm):
        """Flux density [Jy] in the band centered on `wavelength_nm`.

        The wavelength must be the center of a band the spec was built with
        (`band_fluxes_jy`).
        """
        for center, flux in self.band_fluxes_jy:
            if math.isclose(center, float(wavelength_nm), rel_tol=1e-9):
                return flux
        have = [c for c, _ in self.band_fluxes_jy]
        raise ValueError(
            f"star {self.name!r} has no flux at {wavelength_nm} nm; "
            f"band fluxes were built for {have}"
        )


class PopulationTruth(eqx.Module):
    """One world's planets: `n_slots` per star, with a present mask.

    Every leaf but `albedo` has shape ``[n_subjects, n_slots]``; `albedo` adds a last
    axis, one entry per declared band (a batch of worlds adds a leading axis to all).
    An absent slot (`present` false) holds an inert filler planet: a circular
    one-year orbit with zero radius and zero albedo, so every kernel stays finite and
    the mask alone says whether it exists.

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
        present: Whether the slot holds a planet.
        albedo: Geometric albedo in each declared band, ``[..., n_bands]``.
        bands: The declared bands, in the order of the last axis of `albedo`.
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
    present: jax.Array
    albedo: jax.Array
    bands: tuple[Band, ...] = eqx.field(static=True)

    def __check_init__(self):
        """Require declared bands and one albedo entry per band."""
        _declared(self.bands)
        if self.albedo.shape[-1:] != (len(self.bands),):
            raise ValueError(
                f"albedo has last axis {self.albedo.shape[-1:]}; "
                f"{len(self.bands)} bands are declared"
            )

    def band_index(self, center_nm):
        """Position of the declared band centered on `center_nm`; refuse any other."""
        for b, band in enumerate(self.bands):
            if math.isclose(band.center_nm, float(center_nm), rel_tol=1e-9):
                return b
        have = [b.center_nm for b in self.bands]
        raise ValueError(f"no declared band at {center_nm} nm; declared: {have}")


_FILLER = {
    "period_d": 365.25,
    "e": 0.0,
    "cos_i": 0.0,
    "W_rad": 0.0,
    "cos_w": 1.0,
    "sin_w": 0.0,
    "Mp_kg": 0.0,
    "Rp_AU": 0.0,
}
_ORBIT_FIELDS = (
    "period_d",
    "e",
    "cos_i",
    "W_rad",
    "cos_w",
    "sin_w",
    "tp_jd",
    "Mp_kg",
    "Rp_AU",
)
PHOENIX_LOGG_MAX = 5.0  # the model grid ends here; compact dwarfs are read at its edge
G_CGS = 6.6743e-8  # cm^3 g^-1 s^-2, for the surface gravity handed to PHOENIX


def _lambert_albedo(system, planet, wavelength_nm):
    """Geometric albedo that gives the file's contrast at `wavelength_nm`, first epoch.

    Photomancy's Lambert phase curve with the planet's own radius, read at the
    planet's first epoch. The wavelength must lie inside the file's tables
    (`from_files` checks). `FALLBACK_ALBEDO` stands in only for degenerate geometry:
    a Lambert phase too small to divide by, or an albedo that comes out non-finite
    or under 1e-3.
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


def _covered_nm(system):
    """``(low, high)`` [nm] that every table of a loaded file covers.

    The intersection of the star's flux table and each planet's contrast table.
    """
    tables = [np.asarray(system.star.wavelength_range_nm)]
    tables += [np.asarray(p.physical_model.wavelengths_nm) for p in system.planets]
    return max(float(t.min()) for t in tables), min(float(t.max()) for t in tables)


def _require_covered(path, system, bands):
    """Refuse a declared band that the file's tables do not cover from edge to edge."""
    low, high = _covered_nm(system)
    for band in bands:
        lo = band.center_nm - band.width_nm / 2
        hi = band.center_nm + band.width_nm / 2
        if lo < low or hi > high:
            raise ValueError(
                f"{path} covers {low:g} to {high:g} nm; the declared band at "
                f"{band.center_nm:g} nm (width {band.width_nm:g} nm) is outside it"
            )


def _stored_planets(path, system, bands):
    """Each planet of one file: the orbit leaves, `albedo` per band and `peak`.

    `peak`, the ranking key, uses the first declared band.
    """
    star_kg = float(system.star.Ms_kg)
    out = []
    for planet in system.planets:
        orbit = planet.orbit
        a_au, e = float(orbit.a_AU[0]), float(orbit.e[0])
        mass_kg = float(planet.Mp_Mearth[0]) * Mearth2kg
        mu = G * (star_kg + mass_kg)
        n = float(mean_motion(a_au, mu))
        rp_au = float(planet.Rp_Rearth[0]) * Rearth2AU
        albedo = [_lambert_albedo(system, planet, b.center_nm) for b in bands]
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
                "albedo": albedo,
                # flux ratio at full phase and periastron: an upper bound, for ranking
                "peak": albedo[0] * (rp_au / (a_au * (1.0 - e))) ** 2,
            }
        )
    return out


BAND_POINTS = 9  # wavelengths averaged across a band


def _band_flux_jy(star, epoch_jd, center_nm, width_nm):
    """Flux density [Jy] of `star` in a band: its mean photon flux, read at the center.

    The photon flux density per nm is averaged over `BAND_POINTS` wavelengths across
    the band, then expressed in Jy at `center_nm`.
    """
    points = jnp.linspace(
        center_nm - width_nm / 2, center_nm + width_nm / 2, BAND_POINTS
    )
    photons = jnp.ravel(star.spec_flux_density(points, epoch_jd))
    n = points.shape[0]
    photons = photons.reshape(n, -1)[:, 0]
    if not bool(jnp.all(jnp.isfinite(photons))):
        raise ValueError(
            f"the star's flux table does not cover the {center_nm} nm band "
            f"(width {width_nm} nm)"
        )
    return float(photons_per_nm_per_m2_to_jy(jnp.mean(photons), center_nm))


def _spec_of_star(star, hip_name, bands, epoch_jd, luminosity_lsun=None):
    """The `StarSpec` of a skyscapes `Star` with its mean flux in each of `bands`."""
    band_fluxes = tuple(
        (b.center_nm, _band_flux_jy(star, epoch_jd, b.center_nm, b.width_nm))
        for b in bands
    )
    return StarSpec(
        name=hip_name,
        mass_msun=float(star.Ms_kg) / Msun2kg,
        dist_pc=float(star.dist_pc),
        ra_deg=float(star.ra_deg),
        dec_deg=float(star.dec_deg),
        luminosity_lsun=float(
            star.luminosity_Lsun if luminosity_lsun is None else luminosity_lsun
        ),
        diameter_mas=float(star.diameter_arcsec) / mas2arcsec,
        band_fluxes_jy=band_fluxes,
    )


def star_spec_from_model(
    name,
    *,
    mass_msun,
    dist_pc,
    luminosity_lsun,
    teff_k,
    radius_rsun,
    bands,
    ra_deg=0.0,
    dec_deg=0.0,
    epoch_jd=J2000_JD,
):
    """A `StarSpec` whose flux is a PHOENIX model atmosphere, not a scene file.

    The model is `skyscapes.io.from_phoenix` at `teff_k` and the surface gravity of
    `mass_msun` and `radius_rsun` (solar metallicity, gravity capped at the grid edge
    of log g = 5), scaled to `dist_pc`. The spec carries the mean flux density in
    each of `bands` (`Band`s, at least one).
    `luminosity_lsun` is the catalog value, which is what the exoEarth-candidate box
    uses, and is kept as given.
    """
    bands = _declared(bands)
    radius_m = radius_rsun * Rsun2m
    logg = math.log10(G_CGS * mass_msun * Msun2kg * 1e3 / (radius_m * 1e2) ** 2)
    logg = min(logg, PHOENIX_LOGG_MAX)
    star = from_phoenix(
        teff_k,
        logg,
        Ms_kg=mass_msun * Msun2kg,
        dist_pc=dist_pc,
        radius_m=radius_m,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
    )
    return _spec_of_star(star, name, bands, epoch_jd, luminosity_lsun)


class ExoVistaWorldSource(AbstractWorldSource):
    """Stored ExoVista universes as a world source: one stored universe per draw.

    The arrays hold every universe: `population` is a `PopulationTruth` whose leaves
    are ``[n_worlds, n_subjects, n_slots]``. `stars` are the hosts (the same in every
    universe, checked at construction). `dropped` counts, per universe and star, the
    planets past the capacity `n_slots`, the faintest by peak flux ratio. `generator`
    is the population the worlds come from; `declared_population` returns it, so a
    belief starts from the population and never from the stored worlds.

    Attributes:
        population: The stored `PopulationTruth`, world axis first.
        dropped: ``int[n_worlds, n_subjects]`` planets past the capacity.
        generator: The `PopulationModel` the worlds are drawn from.
        stars: One `StarSpec` per subject.
        epoch_jd: Julian Date of the files' first epoch (the orbit epoch).
    """

    population: PopulationTruth
    dropped: jax.Array
    generator: PopulationModel
    stars: tuple[StarSpec, ...] = eqx.field(static=True)
    epoch_jd: float = eqx.field(static=True)

    def declared_population(self):
        """The `PopulationModel` the worlds come from (`generator`)."""
        return self.generator

    @property
    def n_worlds(self):
        """Number of stored universes."""
        return int(self.population.period_d.shape[0])

    @property
    def n_slots(self):
        """Planet slots per star."""
        return int(self.population.period_d.shape[2])

    @classmethod
    def from_files(cls, files, *, n_slots, bands, generator, epoch_jd=None):
        """Read `files[w][s]`, star `s` of universe `w`, into a world source.

        Each file is read with `skyscapes.io.from_exovista(path, load_disk=False)`,
        which loads one system. Slots hold the first `n_slots` planets by descending
        peak flux ratio (full phase at periastron, first declared band); the rest are
        `dropped`. The slot index therefore carries the existence ordering, so slots
        are exchangeable to a belief, which must not read meaning into the index. The
        stars must agree across universes.

        `bands` are the declared `Band`s: each planet's albedo and each star's flux
        are read in every one. A band the file's tables do not cover from edge to
        edge is refused, naming the file and its covered range. `generator` is the
        `PopulationModel` the files were drawn from.
        """
        bands = _declared(bands)
        generator.check_bands(bands)
        n_worlds, n_stars = len(files), len(files[0])
        if any(len(row) != n_stars for row in files):
            raise ValueError("every universe needs one file per star")
        shape = (n_worlds, n_stars, n_slots)
        leaves = {n: np.full(shape, _FILLER.get(n, 0.0)) for n in _ORBIT_FIELDS}
        albedo = np.zeros((*shape, len(bands)))
        present = np.zeros(shape, bool)
        dropped = np.zeros((n_worlds, n_stars), np.int64)
        stars = [None] * n_stars
        epoch = epoch_jd
        for w in range(n_worlds):
            for s in range(n_stars):
                path = str(files[w][s])
                system = from_exovista(path, load_disk=False)
                _require_covered(path, system, bands)
                t0 = float(system.planets[0].orbit.t0_d[0]) if system.planets else 0.0
                epoch = t0 if epoch is None else epoch
                spec = _spec_of_star(system.star, _hip_name(path), bands, epoch)
                if stars[s] is None:
                    stars[s] = spec
                elif stars[s] != spec:
                    raise ValueError(f"star {s} differs between universes")
                planets = _stored_planets(path, system, bands)
                planets.sort(key=lambda p: -p["peak"])
                kept = planets[:n_slots]
                dropped[w, s] = len(planets) - len(kept)
                for k, p in enumerate(kept):
                    present[w, s, k] = True
                    albedo[w, s, k] = p["albedo"]
                    for name in _ORBIT_FIELDS:
                        leaves[name][w, s, k] = p[name]
        population = PopulationTruth(
            **{n: jnp.asarray(v, F64) for n, v in leaves.items()},
            present=jnp.asarray(present),
            albedo=jnp.asarray(albedo, F64),
            bands=bands,
        )
        return cls(
            population=population,
            dropped=jnp.asarray(dropped),
            generator=generator,
            stars=tuple(stars),
            epoch_jd=float(epoch),
        )

    @classmethod
    def from_draws(cls, keys, stars, *, n_slots, bands, generator, epoch_jd=J2000_JD):
        """Draw one world per key from the occurrence-rate generator.

        `keys` is ``[n_worlds]`` PRNG keys (world `w` depends on `keys[w]` alone) and
        `stars` the `StarSpec` of each host; the draw is
        `skyscapes.population.draw_population` of `generator` (its occurrence grid,
        types and config) over the stars' masses and luminosities. See
        `from_population` for the slot rules and `bands`.
        """
        occurrence, types = generator.occurrence, generator.types
        config = generator.config
        mass = jnp.asarray([s.mass_msun for s in stars], F64)
        lum = jnp.asarray([s.luminosity_lsun for s in stars], F64)
        population = eqx.filter_jit(
            lambda ks: jax.lax.map(
                lambda k: draw_population(k, occurrence, types, lum, mass, config), ks
            )
        )(jnp.asarray(keys))
        return cls.from_population(
            population,
            stars,
            n_slots=n_slots,
            bands=bands,
            generator=generator,
            epoch_jd=epoch_jd,
        )

    @classmethod
    def from_population(
        cls, population, stars, *, n_slots, bands, generator, epoch_jd=J2000_JD
    ):
        """Build a world source from drawn `PlanetPopulation` leaves ``[W, S, P]``.

        The slot and drop rules are those of `from_files`: slots hold the first
        `n_slots` planets by descending peak flux ratio (first declared band), the
        rest are `dropped`, and slots past the drawn planets hold the filler.
        Orbits are at `epoch_jd` (the epoch of
        the ExoVista files is J2000). The geometric albedo in each declared `Band` is
        the planet type's (of `generator`) at the band center.
        """
        bands = _declared(bands)
        generator.check_bands(bands)
        types = generator.types
        n_stars = len(stars)
        if population.present.shape[1] != n_stars:
            raise ValueError("the population needs one star axis entry per star")
        mass_msun = jnp.asarray([s.mass_msun for s in stars], F64)
        sky = jax.vmap(lambda p: sky_elements(p, mass_msun))(population)
        present = np.asarray(population.present)
        a_au = np.where(present, np.asarray(sky.a_AU, np.float64), 1.0)
        e = np.where(present, np.asarray(sky.e, np.float64), 0.0)
        mass_kg = np.asarray(population.mass_Mearth, np.float64) * Mearth2kg
        star_kg = (np.asarray(mass_msun) * Msun2kg)[None, :, None]
        mu = G * (star_kg + mass_kg)
        period_d = np.asarray(period_a(a_au, mu), np.float64)
        n = np.asarray(mean_motion(a_au, mu), np.float64)
        rp_au = np.asarray(population.radius_Rearth, np.float64) * Rearth2AU
        albedo = np.stack(
            [
                np.asarray(
                    types.geometric_albedo(population.type_index, b.center_nm),
                    np.float64,
                )
                for b in bands
            ],
            axis=-1,
        )
        w_rad = np.asarray(sky.w_rad, np.float64)
        planets = {
            "period_d": period_d,
            "e": e,
            "cos_i": np.cos(np.asarray(sky.i_rad, np.float64)),
            "W_rad": np.asarray(sky.W_rad, np.float64),
            "cos_w": np.cos(w_rad),
            "sin_w": np.sin(w_rad),
            "tp_jd": epoch_jd - np.asarray(sky.M0_rad, np.float64) / n,
            "Mp_kg": mass_kg,
            "Rp_AU": rp_au,
        }
        peak = albedo[..., 0] * (rp_au / (a_au * (1.0 - e))) ** 2
        # Absent planets past the drawn ones, so a capacity above the draw fills.
        extra = max(n_slots - present.shape[-1], 0)
        if extra:
            pad = [(0, 0)] * (present.ndim - 1) + [(0, extra)]
            present, peak = (np.pad(x, pad) for x in (present, peak))
            planets = {k: np.pad(v, pad) for k, v in planets.items()}
            albedo = np.pad(albedo, [*pad, (0, 0)])
        order = np.argsort(np.where(present, -peak, np.inf), axis=-1, kind="stable")
        order = order[..., :n_slots]
        kept = np.take_along_axis(present, order, axis=-1)
        leaves = {}
        for name in _ORBIT_FIELDS:
            taken = np.take_along_axis(planets[name], order, axis=-1)
            leaves[name] = np.where(kept, taken, _FILLER.get(name, 0.0))
        taken = np.take_along_axis(albedo, order[..., None], axis=-2)
        population_truth = PopulationTruth(
            **{k: jnp.asarray(v, F64) for k, v in leaves.items()},
            present=jnp.asarray(kept),
            albedo=jnp.asarray(np.where(kept[..., None], taken, 0.0), F64),
            bands=bands,
        )
        return cls(
            population=population_truth,
            dropped=jnp.asarray(present.sum(-1) - kept.sum(-1)),
            generator=generator,
            stars=tuple(stars),
            epoch_jd=float(epoch_jd),
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
