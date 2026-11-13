"""Named parameters preserve physical units and the catalog layout."""

import dataclasses
import types

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Msun2kg, Rearth2AU
from orbix.equations.orbit import period_a
from orbix.kepler import markley_solve_trig
from skyscapes.io import from_exovista
from skyscapes.population import (
    PlanetTypes,
    PopulationConfig,
    mass_to_radius,
    occurrence_table_from_grid,
)

from spaceodyssey.adapters import photomancy as ph
from spaceodyssey.adapters.parameters import (
    catalog_template,
    from_photomancy,
    planet_mass_mearth,
    scale_au2,
    to_photomancy,
    truth_parameters,
)
from spaceodyssey.adapters.skyscapes import (
    Band,
    ExoVistaWorldSource,
    PopulationModel,
    SupportRange,
    geometry,
)
from spaceodyssey.core.records import layout_size, pack, unpack
from spaceodyssey.criteria.claims import EarthCandidateClaim
from spaceodyssey.studies import bracket as br
from tests.studies.bracket_helpers import (
    N_SLOTS,
    build,
    fixture_channel,
    fixture_files,
    fixture_source,
)

PLANET_EPOCHS = [(0, k, t) for k in range(3) for t in (0.0, 100.0, 400.0)]


def photomancy_cloud(n):
    """Seed-zero photomancy orbits, radii 0.5 to 20 Earth radii."""
    rng = np.random.default_rng(0)
    return jnp.asarray(
        np.column_stack(
            [
                rng.uniform(np.log(50), np.log(5000), n),
                rng.uniform(0, 0.8, n),
                rng.uniform(-1, 1, n),
                rng.uniform(0, 2 * np.pi, n),
                rng.uniform(-np.pi, np.pi, n),
                rng.uniform(0, 1, n),
                rng.uniform(-0.2, 0.2, n),
                np.log(0.25 * (np.geomspace(0.5, 20, n) * Rearth2AU) ** 2),
            ]
        )
    )


def assert_trees_close(a, b, **kw):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, **kw)


@pytest.fixture(scope="module")
def source():
    return fixture_source()


@pytest.fixture(scope="module")
def channel(source):
    return fixture_channel(source)


@pytest.mark.parametrize("jit", [False, True])
def test_photomancy_round_trip_is_exact(jit):
    """Fails if a unit or field order is wrong, or the conversion cannot trace."""
    el = photomancy_cloud(n=64)
    albedo = jnp.full((64, 2), 0.25).at[:, 1].set(0.4)

    def trip(e, a):
        return to_photomancy(from_photomancy(e, a, band_index=0), band_index=0)

    back = (jax.jit(trip) if jit else trip)(el, albedo)
    np.testing.assert_allclose(back, el, atol=1e-12)


def test_the_planet_mass_inverts_the_mass_radius_relation_and_is_capped():
    """Fails if the inverse misses the relation, monotonicity, or the mass cap."""
    r = jnp.geomspace(0.3, 30.0, 200)
    m = jax.jit(planet_mass_mearth)(r)
    assert bool(jnp.all(jnp.diff(m) >= 0)) and float(m.max()) <= 131.6
    below = r < mass_to_radius(131.6)
    np.testing.assert_allclose(mass_to_radius(m[below]), r[below], rtol=1e-12)


def true_report_position(channel, source, s, k, t):
    """True sky position computed directly from the channel's stored-truth path."""
    truth = source.take(0)
    planet = jax.tree.map(lambda x: x[k], channel._slot_planets(truth, s, 0))
    report = channel.assets().report
    rates = report.rates(
        channel._prepared(s, nominal=False), planet, source.epoch_jd + t
    )
    return jnp.stack([rates.ra_arcsec[0], rates.dec_arcsec[0]])


@pytest.mark.slow(reason="loads and compares named, exact and tabulated fixture orbits")
def test_truth_parameters_place_each_planet_where_the_channel_does(source, channel):
    """Fails if the named conversion moves a planet on the sky."""
    p = truth_parameters(
        source.take(0), channel.catalog_mass_kg / Msun2kg, channel.epoch_jd
    )
    for s, k, t in PLANET_EPOCHS:
        got = channel.sky_position(
            s, t, jax.tree.map(lambda x, s=s, k=k: x[s, k][None], p)
        )[0]
        np.testing.assert_allclose(
            got, true_report_position(channel, source, s, k, t), atol=1e-9
        )

    stored = from_exovista(str(fixture_files()[0]), load_disk=False)
    exact = dataclasses.replace(stored, trig_solver=markley_solve_trig)
    times_d = jnp.asarray([0.0, 91.3, 400.7, 20000.0])

    def separation(system, slot):
        one = eqx.tree_at(lambda s: s.planets, system, (system.planets[slot],))
        return np.asarray(
            geometry(one, source.epoch_jd + times_d, 1000.0).separation_arcsec
        )

    for slot in np.flatnonzero(np.asarray(source.take(0).present[0])):
        params = jax.tree.map(lambda x, slot=slot: x[0, slot][None], p)
        xy = jax.vmap(lambda t, params=params: channel.sky_position(0, t, params))(
            times_d
        )
        ours = np.asarray(jnp.linalg.norm(xy[:, 0], axis=-1))
        for system, tolerance in ((exact, 1e-9), (stored, 1e-6)):
            assert (
                min(
                    np.max(np.abs(ours - separation(system, j)))
                    for j in range(len(stored.planets))
                )
                < tolerance
            )


def slot_params(source, subject):
    t = source.take(0)
    return jax.tree.map(
        lambda x: x[subject],
        truth_parameters(
            t, jnp.asarray([s.mass_msun for s in source.stars]), source.epoch_jd
        ),
    )


def realize_catalog(channel, truth, subject):
    return channel._catalog_row(truth, subject, 0, jax.random.key(0), None)


@pytest.mark.medium(reason="loads the fixture and packs its catalog")
def test_the_catalog_row_round_trips_and_unpacks_to_the_truth_parameters(
    source, channel
):
    """Fails if the catalog writer and reader disagree on a column."""
    row = realize_catalog(channel, source.take(0), subject=0)
    template = catalog_template(N_SLOTS, channel.n_bands)
    back = unpack(template, row)
    assert_trees_close(back.params, slot_params(source, 0), atol=1e-12)
    np.testing.assert_array_equal(back.present, source.take(0).present[0])
    np.testing.assert_array_equal(pack(back), row[: layout_size(template)])


def test_the_radius_follows_the_detection_scale_and_the_drawn_albedo():
    """Fails if radius uses a fixed albedo or the wrong band."""
    el = photomancy_cloud(n=8)
    a = jnp.stack([jnp.full(8, 0.2), jnp.full(8, 0.5)], -1)
    p0, p1 = from_photomancy(el, a, band_index=0), from_photomancy(el, a, band_index=1)
    np.testing.assert_allclose(scale_au2(p0, 0), jnp.exp(el[:, 7]), rtol=1e-12)
    np.testing.assert_allclose(
        p0.radius_rearth / p1.radius_rearth, jnp.sqrt(0.5 / 0.2), rtol=1e-12
    )


def mixture_population():
    """Two occurrence cells and three types, including overlapping unequal weights."""

    occurrence = occurrence_table_from_grid(
        [[1.0], [3.0]], [1.0, 2.0, 4.0], [1.0, 2.0], [[0], [1]]
    )
    types = PlanetTypes(
        names=("a", "b", "c"),
        radius_min_Rearth=jnp.zeros(3),
        radius_max_Rearth=jnp.full(3, 10.0),
        a_min_AU=jnp.array([1.0, 2.0, 2.0]),
        a_max_AU=jnp.array([2.0, 4.0, 4.0]),
        weight=jnp.array([100.0, 1.0, 3.0]),
        earth_candidate=jnp.zeros(3, bool),
        wavelength_nm=jnp.array([500.0, 1000.0]),
        albedo=jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
    )
    return PopulationModel(occurrence, types, PopulationConfig(), SupportRange())


def test_albedo_prior_combines_occurrence_and_conditional_type_weights():
    """Fails if selection weights become frequencies or bands draw independent types.

    1024 draws resolve a probability error of 0.063 at four standard deviations.
    """

    population = mixture_population()
    bands = (Band(500.0, 10.0), Band(1000.0, 10.0))
    draw = eqx.filter_jit(lambda key: population.albedo_prior(key, 1024, bands))(
        jax.random.key(0)
    )
    np.testing.assert_allclose(draw[:, 1] - draw[:, 0], 0.1, atol=1e-15)
    fractions = np.array(
        [(np.asarray(draw[:, 0]) == a).mean() for a in (0.1, 0.3, 0.5)]
    )
    np.testing.assert_allclose(fractions, [0.25, 0.1875, 0.5625], atol=0.063, rtol=0)


@pytest.mark.parametrize("bad", [0.0, -0.1, np.inf, np.nan])
def test_declared_population_bands_refuse_invalid_selectable_albedo(bad):
    """Fails if a declared band can send a selectable invalid albedo into an inverse."""

    population = mixture_population()
    population = eqx.tree_at(
        lambda p: p.types.albedo, population, population.types.albedo.at[1, 0].set(bad)
    )
    with pytest.raises(ValueError, match="albedo"):
        population.check_bands((Band(500.0, 10.0),))
    ignored = eqx.tree_at(
        lambda p: p.types.weight, population, population.types.weight.at[1].set(0)
    )
    ignored.check_bands((Band(500.0, 10.0),))


def test_cached_cloud_identity_distinguishes_albedo_mixtures():
    """Fails if equal support bounds let a different albedo mixture reuse a cloud."""
    population = mixture_population()
    changed = eqx.tree_at(
        lambda p: p.types.weight, population, population.types.weight.at[1].set(2.0)
    )
    bands = (Band(500.0, 10.0), Band(1000.0, 10.0))
    st = types.SimpleNamespace(
        ln_lam_range=(-30.0, -10.0), period_support_d=(10.0, 1000.0)
    )
    item = {name: np.zeros(1) for name in ph._FIT_INPUTS}
    assert ph._fit_digest(item, st, population, bands) != ph._fit_digest(
        item, st, changed, bands
    )
    assert ph._fit_digest(item, st, population, bands) == ph._fit_digest(
        item, st, population, bands
    )


def test_earth_candidate_uses_the_physical_semimajor_axis_convention():
    """Fails if the claim uses a Julian-year approximation at either box edge."""
    el = photomancy_cloud(4).at[:, 6].set(0.0)
    axes = jnp.array(
        [0.95 * (1 - 1e-7), 0.95 * (1 + 1e-7), 1.67 * (1 - 1e-7), 1.67 * (1 + 1e-7)]
    )
    el = el.at[:, 0].set(jnp.log(period_a(axes, G * Msun2kg)))
    el = el.at[:, 7].set(jnp.log(0.3 * Rearth2AU**2))
    params = from_photomancy(el, jnp.full((4, 2), 0.3), 0)
    np.testing.assert_array_equal(
        EarthCandidateClaim().evaluate(params, 1.0), [False, True, True, False]
    )


def test_channel_and_source_bands_must_agree_at_study_construction(source, channel):
    """Fails if a channel silently reads different bands than its source declares."""
    mismatched = dataclasses.replace(channel, bands=tuple(reversed(channel.bands)))
    with pytest.raises(ValueError, match="declared bands"):
        br.bracket_study(source, mismatched, arm="oracle", horizon_d=1.0, budget_d=1.0)


def test_actual_invalid_spectrum_is_rejected_at_source_and_refit_construction(
    source, channel
):
    """Fails if the clipped-zero 1842 nm spectrum reaches a radius inverse."""
    population = source.generator
    bad_bands = (Band(1842.0, 1.0),)
    with pytest.raises(ValueError, match="Warm_Neptune_1AU_Clouds"):
        ExoVistaWorldSource.from_files(
            [fixture_files()], n_slots=N_SLOTS, bands=bad_bands, generator=population
        )
    _, study, *_ = build(run=False, source=source, channel=channel, arm="replan")
    bad_channel = dataclasses.replace(channel, bands=(*channel.bands, *bad_bands))
    study = dataclasses.replace(study, planning_model=bad_channel)
    with pytest.raises(ValueError, match="Warm_Neptune_1AU_Clouds"):
        ph.ReportRefit(kernel="posterior")._post_settings(study)


@pytest.mark.parametrize("candidate_probability", [0.0, 1.0])
def test_albedo_validation_and_draws_follow_candidate_probability(
    candidate_probability,
):
    """Fails if disabled candidates or covered ordinary types stay selectable."""
    population = mixture_population()
    population = eqx.tree_at(
        lambda p: (
            p.types.a_min_AU,
            p.types.a_max_AU,
            p.types.earth_candidate,
        ),
        population,
        (
            jnp.ones(3),
            jnp.full(3, 4.0),
            jnp.array([True, False, False]),
        ),
    )
    population = dataclasses.replace(
        population,
        config=dataclasses.replace(
            population.config, earth_candidate_probability=candidate_probability
        ),
    )
    bad_type = 0 if candidate_probability == 0 else 1
    population = eqx.tree_at(
        lambda p: p.types.albedo,
        population,
        population.types.albedo.at[bad_type, 0].set(-0.1),
    )
    bands = (Band(500.0, 10.0),)
    population.check_bands(bands)
    draws = population.albedo_prior(jax.random.key(0), 32, bands)
    assert bool(jnp.all(draws > 0))
    if candidate_probability == 1:
        np.testing.assert_allclose(draws, 0.1)
