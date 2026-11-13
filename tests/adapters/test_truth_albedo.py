"""Declared bands and the per-band truth albedo of the world sources.

Every wavelength the truth is read at is a declared `Band`; the truth carries one
geometric albedo per band, and the observing-mode channel reads the albedo of the band
each report is taken in.
"""

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Mearth2kg, Msun2kg, two_pi
from orbix.kepler import markley_solve_trig
from skyscapes.io import from_exovista
from skyscapes.population import PlanetPopulation

from spaceodyssey.adapters.jaxedith import ObservingModeChannel
from spaceodyssey.adapters.physical_profile import JATIS_EAC1_V1
from spaceodyssey.adapters.skyscapes import (
    Band,
    ExoVistaWorldSource,
    PopulationModel,
    StarSpec,
    geometry,
    star_spec_from_model,
)
from spaceodyssey.core.kinds import RealizeContext
from tests.studies.bracket_helpers import SURVEY_BANDS, fixture_files

BANDS = (Band(500.0, 100.0), Band(940.0, 6.7))
# Bands inside the fixture files' 900 to 1094 nm tables.
FILE_BANDS = (Band(910.0, 6.5), Band(940.0, 6.7))
STARS = tuple(
    StarSpec(f"HIP {k}", 1.0, 5.0 + k, 10.0, 20.0, 1.0, 1.0, 1.0e3) for k in range(2)
)
# The same hosts with their flux in each survey band, written out (no model atmosphere).
SURVEY_STARS = tuple(
    StarSpec(
        f"HIP {k}",
        1.0,
        5.0 + k,
        10.0,
        20.0,
        1.0,
        1.0,
        60.0,
        ((500.0, 60.0), (910.0, 90.0)),
    )
    for k in range(2)
)
KEYS = jax.random.split(jax.random.PRNGKey(0), 2)
GEN = PopulationModel.exovista()
# Two planet types whose albedo spectra differ between the bands.
TYPE_INDEX = ((0, 5), (3, 8))


def _per_planet(value, n_planets):
    """`value` with its last axis cycled to `n_planets` entries."""
    value = np.asarray(value)
    if value.ndim == 0:
        return np.full(n_planets, value)
    return np.take(value, np.arange(n_planets) % value.shape[-1], axis=-1)


def hand_population(
    n_planets=2, periods_d=None, radii_rearth=None, stars=STARS, **overrides
):
    """One world with `n_planets` per host in `stars`, a `PlanetPopulation`.

    By default planet 0 of each star is a giant at 1 AU and planet 1 an Earth-sized
    planet at 2 AU, both circular, with the types `TYPE_INDEX`, so the slot order
    (descending peak flux ratio) is the drawn order. `overrides` replace a field with
    one value per planet (the same for both stars) or one row per star; a default
    cycles to `n_planets`. `radii_rearth` sets `radius_rearth`, and `periods_d` sets
    `a_au` from the period about the star of `stars` with the planet's mass.
    """
    fields = {
        "mass_mearth": [300.0, 1.0],
        "radius_rearth": [11.0, 1.0],
        "a_au": [1.0, 2.0],
        "e": 0.0,
        "inc_deg": 30.0,
        "node_deg": 10.0,
        "argperi_deg": 40.0,
        "mean_anomaly_deg": 70.0,
        "type_index": TYPE_INDEX,
    }
    fields.update(overrides)
    if radii_rearth is not None:
        fields["radius_rearth"] = radii_rearth
    shape = (1, len(stars), n_planets)
    out = {
        k: np.broadcast_to(_per_planet(v, n_planets), shape) for k, v in fields.items()
    }
    if periods_d is not None:
        star_kg = np.asarray([s.mass_msun for s in stars])[None, :, None] * Msun2kg
        mu = G * (star_kg + out["mass_mearth"] * Mearth2kg)
        period = np.broadcast_to(_per_planet(periods_d, n_planets), shape)
        out["a_au"] = np.cbrt(mu * (period / two_pi) ** 2)
    floats = {k: jnp.asarray(v, jnp.float64) for k, v in out.items()}
    floats["type_index"] = jnp.asarray(out["type_index"], jnp.int32)
    return PlanetPopulation(
        present=jnp.ones(shape, bool),
        midplane_inc_deg=jnp.zeros(shape[:2]),
        midplane_pa_deg=jnp.zeros(shape[:2]),
        **floats,
    )


def _lambert_phase(alpha):
    """Lambert phase function of phase angle `alpha` (0 at full phase)."""
    return (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi


def file_contrast(source, bands):
    """Per slot of world 0: ``(contrast [S, K, B], phase [S, K], distance [S, K])``.

    Each slot is matched to its stored planet by node and eccentricity; the contrast
    is the file's own at each band center at the first epoch, the phase is the
    Lambert phase function there and the distance is in AU (exact Kepler solver).
    """
    truth = source.take(0)
    n_stars, n_slots = truth.present.shape
    contrast = np.full((n_stars, n_slots, len(bands)), np.nan)
    phase = np.zeros((n_stars, n_slots))
    distance = np.ones((n_stars, n_slots))
    for s, path in enumerate(fixture_files()):
        system = from_exovista(str(path), load_disk=False)
        system = dataclasses.replace(system, trig_solver=markley_solve_trig)
        for k in range(n_slots):
            if not bool(truth.present[s, k]):
                continue
            (j,) = [
                j
                for j, p in enumerate(system.planets)
                if float(p.orbit.W_rad[0]) == float(truth.W_rad[s, k])
                and float(p.orbit.e[0]) == float(truth.e[s, k])
            ]
            one = eqx.tree_at(lambda x: x.planets, system, (system.planets[j],))
            t0 = jnp.atleast_1d(system.planets[j].orbit.t0_d[:1])
            for b, band in enumerate(bands):
                g = geometry(one, t0, band.center_nm)
                contrast[s, k, b] = float(g.contrast[0])
            phase[s, k] = _lambert_phase(float(g.phase_angle_rad[0]))
            distance[s, k] = float(g.distance_au[0])
    return contrast, phase, distance


def lambert_contrast(source, phase, distance):
    """``albedo[..., b] * (Rp / r)**2 * phase`` of world 0, ``[S, K, B]``."""
    truth = source.take(0)
    scale = (np.asarray(truth.Rp_AU) / distance) ** 2 * phase
    return np.asarray(truth.albedo) * scale[..., None]


def test_a_band_refuses_a_nonpositive_or_nonfinite_value():
    """Fails if a band can be declared with no width or an infinite center."""
    for center, width in ((500.0, 0.0), (-1.0, 10.0), (math.inf, 10.0)):
        with pytest.raises(ValueError, match="finite positive"):
            Band(center, width)


def test_drawn_truth_albedo_is_the_type_spectrum_at_each_band():
    """Fails if a band reads another band's albedo or one gray value."""
    population = hand_population()
    truth = ExoVistaWorldSource.from_population(
        population, STARS, n_slots=2, bands=BANDS, generator=GEN
    ).population
    assert truth.albedo.shape == (1, 2, 2, len(BANDS))
    assert truth.bands == BANDS
    for b, band in enumerate(BANDS):
        want = GEN.types.geometric_albedo(population.type_index, band.center_nm)
        np.testing.assert_allclose(
            truth.albedo[..., b], np.where(truth.present, want, 0.0), rtol=1e-12
        )
    assert not np.allclose(truth.albedo[..., 0], truth.albedo[..., 1], rtol=0.05)


@pytest.mark.medium(reason="reads both fixture files and every planet in two bands")
def test_stored_albedo_reproduces_the_file_contrast_in_each_band():
    """Fails if a stored planet's albedo is read at one band and reused at another.

    The bands are the two the files cover; the stored albedos differ between them by
    up to 30 percent, so a reused band misses the file's contrast.
    """
    source = ExoVistaWorldSource.from_files(
        [fixture_files()], n_slots=6, bands=FILE_BANDS, generator=GEN
    )
    contrast, phase, distance = file_contrast(source, FILE_BANDS)
    lit = np.asarray(source.population.present[0]) & (phase > 0.05)
    assert lit.sum() >= 4
    model = lambert_contrast(source, phase, distance)
    np.testing.assert_allclose(model[lit], contrast[lit], rtol=1e-6)


def test_a_band_the_files_do_not_cover_is_refused():
    """Fails if an uncovered band is read through a fallback instead of refused."""
    with pytest.raises(ValueError, match=r"HIP_27913.*covers 900 to 1093.96 nm.*500"):
        ExoVistaWorldSource.from_files(
            [fixture_files()],
            n_slots=6,
            bands=(Band(910.0, 6.5), Band(500.0, 100.0)),
            generator=GEN,
        )


@pytest.mark.parametrize(
    "build",
    [
        lambda: ExoVistaWorldSource.from_files(
            [fixture_files()], n_slots=6, generator=GEN
        ),
        lambda: ExoVistaWorldSource.from_draws(KEYS, STARS, n_slots=6, generator=GEN),
        lambda: ExoVistaWorldSource.from_population(
            hand_population(), STARS, n_slots=2, generator=GEN
        ),
        lambda: star_spec_from_model(
            "x",
            mass_msun=1.0,
            dist_pc=10.0,
            luminosity_lsun=1.0,
            teff_k=5800.0,
            radius_rsun=1.0,
        ),
    ],
)
def test_every_constructor_needs_declared_bands(build):
    """Fails if any constructor falls back to a wavelength default."""
    with pytest.raises(TypeError, match="bands"):
        build()


@pytest.fixture(scope="module")
def survey_channel():
    return ObservingModeChannel.from_stars(
        SURVEY_STARS,
        n_slots=2,
        epoch_jd=2451545.0,
        profile=JATIS_EAC1_V1,
        n_grid=3,
    )


@pytest.fixture(scope="module")
def truth():
    source = ExoVistaWorldSource.from_population(
        hand_population(), SURVEY_STARS, n_slots=2, bands=SURVEY_BANDS, generator=GEN
    )
    return source.take(0)


def _one_day(channel):
    """The context of a record that holds one science day after the settling."""
    built = channel.assets().context
    start = 10 * int(built.clock.ticks_per_day)
    end = start + int(built.settle_ticks) + int(built.clock.ticks_per_day)
    zero = jnp.asarray(0)
    return RealizeContext(
        jnp.asarray(start), jnp.asarray(end), zero, jnp.asarray(1.0), zero, zero
    )


@pytest.mark.medium(reason="builds the survey channel's two reports and two rows")
def test_the_channel_reads_the_albedo_of_each_report_band(survey_channel, truth):
    """Fails if detection and characterization share one albedo.

    The detection rows read the 500 nm albedo and the block rows the 910 nm one:
    changing only the 910 nm albedo changes the block row and leaves the visit row.
    """
    assert truth.bands == SURVEY_BANDS
    twice = eqx.tree_at(
        lambda t: t.albedo, truth, truth.albedo.at[..., 1].set(2 * truth.albedo[..., 0])
    )
    det = survey_channel._slot_planets(twice, 0, twice.band_index(500.0))["Lambda"]
    char = survey_channel._slot_planets(twice, 0, twice.band_index(910.0))["Lambda"]
    np.testing.assert_allclose(char, 2.0 * det, rtol=1e-12)
    assert survey_channel.detection_band == SURVEY_BANDS[0]
    block = survey_channel._block_band
    assert block.center_nm == SURVEY_BANDS[1].center_nm
    assert block.width_nm == pytest.approx(SURVEY_BANDS[1].width_nm, rel=1e-12)
    context = _one_day(survey_channel)
    key = jax.random.key(0)

    @eqx.filter_jit
    def rows(t):
        return (
            survey_channel._visit_row(t, 0, 0, key, context),
            survey_channel._block_row(t, 0, 0, key, context),
        )

    visit, block_row = (np.asarray(x) for x in rows(truth))
    visit_twice, block_twice = (np.asarray(x) for x in rows(twice))
    np.testing.assert_array_equal(visit, visit_twice)
    assert not np.array_equal(block_row, block_twice), "the block ignored 910 nm"
    with pytest.raises(ValueError, match="no declared band at 1000"):
        truth.band_index(1000.0)
