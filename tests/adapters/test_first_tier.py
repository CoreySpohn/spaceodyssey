"""Population first tiers and claims at the decision/scoring boundary."""

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Rearth2AU
from scipy.stats import poisson
from skyscapes.population import (
    PopulationConfig,
    earth_like,
    occurrence_table_from_grid,
)

from spaceodyssey._threads import _in_thread, _thread_config
from spaceodyssey.adapters import photomancy as ph
from spaceodyssey.adapters.parameters import (
    PlanetParameters,
    scale_au2,
    semimajor_axis_au,
    truth_parameters,
)
from spaceodyssey.adapters.skyscapes import (
    Band,
    ExoVistaWorldSource,
    PopulationModel,
    StarSpec,
)
from spaceodyssey.core.enums import EffectKind
from spaceodyssey.criteria.claims import EarthCandidateClaim, claims_on_truth
from spaceodyssey.studies import bracket as br
from tests.adapters.test_report_posterior_refit import _null_lane
from tests.studies.bracket_helpers import (
    decide_once,
    effect_kind,
    state_with,
    summary_selecting_node,
    with_grants,
)

KEY = jax.random.PRNGKey(0)


def population_scaled(population, rate_factor):
    """Scale the declared occurrence table, preserving its cell boundaries."""
    occ = population.occurrence
    return dataclasses.replace(
        population,
        occurrence=occurrence_table_from_grid(
            rate_factor * occ.rate, occ.a_edges_AU, occ.mass_edges_Mearth, occ.bin_id
        ),
    )


@pytest.fixture(scope="module")
def boundary_inputs(bracket_study):
    """A small exact population: one cell, no stability removal, no misses."""
    population = dataclasses.replace(
        PopulationModel.exovista(),
        occurrence=occurrence_table_from_grid(
            jnp.asarray([[0.25]]),
            jnp.asarray([1.0, 1.1]),
            jnp.asarray([1.0, 1.1]),
            jnp.asarray([[0]]),
        ),
        config=PopulationConfig(max_planets=2, max_iterations=1, mutual_hill_min=0.0),
        n_pool_systems=8,
    )
    study = with_grants(bracket_study, br.GRANT_PRESETS["replan"])
    boundary = ph.ReportRefit(
        kernel="posterior",
        n_first=8,
        n_particles=8,
        n_estimate_systems=8,
        grid_points=16,
        n_grid=2,
        ess_floor=1.0,
        lanes=1,
    )
    return study, boundary, population


def first_tier(boundary_inputs, population, claims=()):
    """Fit one star's dark slots with the declared population and no reports."""
    study, boundary, _ = boundary_inputs
    study = dataclasses.replace(study, belief_population=population)
    study = dataclasses.replace(
        study,
        observer=dataclasses.replace(
            study.observer,
            criterion=dataclasses.replace(
                study.observer.criterion, claims=claims, target_claim=None
            ),
        ),
    )
    st = boundary._post_settings(study)
    lane = _null_lane(study.channel_model, st, 0, 1, key=KEY)
    lane = eqx.tree_at(lambda x: x.valid, lane, jnp.zeros_like(lane.valid))
    ph._photomancy_refit()
    return ph._first_tier_lane(study.channel_model, population, lane, st, 1), lane, st


@pytest.mark.slow(reason="two cold population first tiers, 15 s focused on this host")
def test_the_first_tier_draws_the_belief_population(boundary_inputs):
    """Fails if first-tier existence or joint draws differ from the population."""
    population = boundary_inputs[2]
    for p, expected in ((population, 0.25), (population_scaled(population, 2.0), 0.5)):
        out, lane, st = first_tier(boundary_inputs, p)
        n_dark = int(lane.n_undetected)
        n_detected = st.n_slots - n_dark
        want = poisson.sf(n_detected + np.arange(n_dark), expected).sum()
        want /= poisson.sf(n_detected - 1, expected) * n_dark
        assert float(out.result.p_exist) == pytest.approx(want)
        elements = out.result.draws
        draws, _ = p.draw_planets(
            ph._slot_key(lane, 1),
            boundary_inputs[0].world_source.stars[0],
            st.n_first,
            boundary_inputs[0].channel_model.bands,
        )
        assert np.isin(
            np.asarray(elements.ln_period_d), np.asarray(draws.ln_period_d)
        ).all()
        np.testing.assert_array_equal(
            jnp.sort(elements.radius_rearth), jnp.sort(draws.radius_rearth)
        )
        np.testing.assert_array_equal(
            jnp.sort(scale_au2(elements, 0)), jnp.sort(scale_au2(draws, 0))
        )
        got = p.expected_planets(
            boundary_inputs[0].world_source.stars[0], KEY, st.n_estimate_systems
        )
        assert float(got) == expected


def hand_cloud():
    """Four draws with two inside the claim and unequal weights."""
    zeros = jnp.zeros(4)
    p = PlanetParameters(
        ln_period_d=jnp.log(jnp.full(4, 365.25)),
        ecc=zeros,
        cos_i=zeros,
        node_rad=zeros,
        argp_rad=zeros,
        tau_d=zeros,
        star_mass_msun=jnp.ones(4),
        planet_mass_mearth=jnp.ones(4),
        radius_rearth=jnp.asarray([1.0, 3.0, 1.0, 3.0]),
        albedo=jnp.full((4, 2), 0.3),
    )
    return p, jnp.log(jnp.asarray([0.1, 0.2, 0.3, 0.4])), jnp.asarray([1, 0, 1, 0])


def test_without_claims_a_dark_slot_values_any_planet():
    """Fails if no claims or an all-false claim produces zero search value or NaN."""
    params, log_w, _ = hand_cloud()
    none = ph.summarize_claims(params, log_w, (), 1.0)
    earth = ph.summarize_claims(params, log_w, (EarthCandidateClaim(),), 1.0)
    false = ph.summarize_claims(
        dataclasses.replace(params, radius_rearth=jnp.full(4, 3.0)),
        log_w,
        (EarthCandidateClaim(),),
        1.0,
    )
    p_exist = 0.25
    v_none = p_exist * none.p_claim.prod(-1)
    v_earth = p_exist * earth.p_claim[..., 0]
    assert bool(jnp.isfinite(v_none)) and float(v_none) >= float(v_earth)
    assert float(v_none) > 0 and float(false.p_claim[0]) == 0


@pytest.fixture(scope="module")
def drawn_source(boundary_inputs):
    """A source drawn from the same tiny population."""
    study, _, population = boundary_inputs
    return ExoVistaWorldSource.from_draws(
        jax.random.split(KEY, 1),
        study.world_source.stars,
        n_slots=2,
        bands=study.channel_model.bands,
        generator=population_scaled(population, 4.0),
    )


def a_au_of(source):
    """True semimajor axes, including inert slots."""
    return jax.vmap(
        lambda t: semimajor_axis_au(
            truth_parameters(
                t, jnp.asarray([s.mass_msun for s in source.stars]), source.epoch_jd
            )
        )
    )(source.population)


def radius_of(source):
    """True radii in Earth radii."""
    return source.population.Rp_AU / Rearth2AU


@pytest.mark.medium(reason="one real two-host population draw and scorer evaluation")
def test_a_claim_on_truth_is_the_catalog_rule(drawn_source):
    """Fails if scorer claims differ from the population's Earth-candidate rule."""
    mass = jnp.asarray([s.mass_msun for s in drawn_source.stars])
    lum = jnp.asarray([s.luminosity_lsun for s in drawn_source.stars])
    got = claims_on_truth(
        drawn_source.population,
        (EarthCandidateClaim(),),
        mass,
        lum,
        drawn_source.epoch_jd,
    )[..., 0]
    want = earth_like(a_au_of(drawn_source), radius_of(drawn_source), lum[:, None])
    assert bool(got.any()) and bool((~drawn_source.population.present).any())
    np.testing.assert_array_equal(got, want & drawn_source.population.present)
    empty = claims_on_truth(
        drawn_source.population, (), mass, lum, drawn_source.epoch_jd
    )
    np.testing.assert_array_equal(empty.prod(-1), np.ones(got.shape))


def test_p_claim_is_the_weighted_claim_mean_of_the_draws():
    """Fails if a claim leaf is summarized with uniform weights or the wrong axis."""
    params, log_w, inside = hand_cloud()
    leaves = ph.summarize_claims(params, log_w, (EarthCandidateClaim(),), 1.0)
    assert float(leaves.p_claim[0]) == pytest.approx(
        float(jax.nn.softmax(log_w) @ inside)
    )


@pytest.mark.medium(reason="cold first-tier fits with and without a null report")
@pytest.mark.parametrize("padded", [False, True])
def test_first_tier_padding_has_no_prior_weight(boundary_inputs, monkeypatch, padded):
    """Fails if row scales change, misses favor bright rows, or padding has mass.

    Systematic resampling has a discrepancy of at most one particle per row;
    without misses, twelve particles exactly preserve the three/four-row prior.
    Equal periods force row identity to include radius, mass and both albedos.
    """
    study, boundary, population = boundary_inputs
    st = dataclasses.replace(boundary._post_settings(study), n_first=4, n_particles=12)
    params, _, _ = hand_cloud()
    params = dataclasses.replace(
        params,
        cos_i=jnp.ones(4),
        radius_rearth=jnp.asarray([0.5, 1.0, 3.0, 4.0]),
        albedo=jnp.asarray([[0.1, 0.4], [0.2, 0.3], [0.3, 0.2], [0.4, 0.1]]),
    )
    valid = jnp.asarray([True, True, True, not padded])
    n_valid = int(valid.sum())
    monkeypatch.setattr(PopulationModel, "draw_planets", lambda *args: (params, valid))
    # The forecast cache keys code by class name; a patched method needs its own.
    monkeypatch.setattr(ph, "_POPULATION_CACHE", {})
    channel = study.channel_model
    ph._photomancy_refit()
    lane = _null_lane(channel, st, 0, 1, key=KEY)
    lane = eqx.tree_at(
        lambda x: (x.sep, x.dmag0, x.limit),
        lane,
        (
            jnp.linspace(0.0, 100.0, st.grid_points)[None],
            jnp.full_like(lane.dmag0, 23.0),
            jnp.full_like(lane.limit, 10.0),
        ),
    )
    band = ph._detection_band_index(channel)
    # Circular face-on rows have a constant Lambert phase of 1 / pi.
    contrast = scale_au2(params, band) / (jnp.pi * semimajor_axis_au(params) ** 2)
    counts_pred = float(channel.law[0]) * 10.0 * contrast * 10.0 ** (0.4 * 23.0)
    null_likelihood = jax.scipy.special.ndtr(
        (float(channel.law[0]) * 10.0 - counts_pred) / jnp.sqrt(100.0 + counts_pred)
    )[:n_valid]
    for has_miss in (False, True):
        lane = eqx.tree_at(lambda x: x.valid, lane, jnp.asarray([has_miss]))
        likelihood = null_likelihood if has_miss else jnp.ones(n_valid)
        cloud, elements = ph._first_tier_cloud(channel, population, lane, st, 1)
        assert float(jnp.exp(cloud.log_evidence)) == pytest.approx(
            float(likelihood.mean())
        )
        weights = likelihood / likelihood.sum()
        counts = jnp.sum(
            elements.radius_rearth[:, None] == params.radius_rearth[None, :n_valid],
            axis=0,
        )
        selected = jnp.argmax(
            elements.radius_rearth[:, None] == params.radius_rearth[None, :], axis=1
        )
        for got, source in zip(
            jax.tree.leaves(elements), jax.tree.leaves(params), strict=True
        ):
            np.testing.assert_array_equal(got, source[selected])
        np.testing.assert_allclose(counts / st.n_particles, weights, atol=1 / 12)
        if not has_miss:
            np.testing.assert_array_equal(counts, np.full(n_valid, 12 // n_valid))
        else:
            assert float(weights[0]) > float(weights[2])
            assert counts[0] > counts[2]
        assert float(cloud.ess) == pytest.approx(float(1 / jnp.sum(weights**2)))
        assert float(cloud.max_weight) == pytest.approx(float(weights.max()))


@pytest.mark.slow(reason="cold bounded population fits for three empty-support cases")
def test_empty_support_and_exhaustion_are_bounded(boundary_inputs):
    """Fails if empty support retries forever or padding invents supported planets."""
    study, _, p = boundary_inputs
    star, bands = study.world_source.stars[0], study.channel_model.bands
    draws, valid = p.draw_planets(KEY, star, 8, bands)
    assert int(valid.sum()) == 8
    again, _ = p.draw_planets(KEY, star, 8, bands)
    changed, _ = p.draw_planets(jax.random.fold_in(KEY, 1), star, 8, bands)
    np.testing.assert_array_equal(draws.ln_period_d, again.ln_period_d)
    assert not np.array_equal(draws.ln_period_d, changed.ln_period_d)
    assert np.all(np.asarray(draws.planet_mass_mearth)[np.asarray(valid)] < 1.1)
    for empty in (
        population_scaled(p, 0.0),
        dataclasses.replace(p, support=dataclasses.replace(p.support, e_max=0.0)),
        dataclasses.replace(
            p, support=dataclasses.replace(p.support, period_d=(1.0, 2.0))
        ),
    ):
        _, valid = empty.draw_planets(KEY, star, 8, bands)
        assert not bool(valid.any())
        assert float(empty.expected_planets(star, KEY, 8)) == 0.0
        out, _, _ = first_tier(boundary_inputs, empty)
        assert float(out.result.p_exist) == 0.0
    params, _, _ = hand_cloud()
    st = boundary_inputs[1]._post_settings(study)
    for n_dark, expected, want in ((0, 100.0, 0.0), (2, 0.0, 0.0)):
        got = ph._first_leaves(
            study.channel_model, 0, 0, n_dark, params, 0.0, expected, st
        )[0]
        assert float(got) == want


@pytest.mark.medium(reason="one policy decision using the prepared two-star channel")
def test_a_dark_slot_with_zero_existence_does_not_schedule_followup(boundary_inputs):
    """Fails if an empty claim tuple makes an impossible dark slot look known."""
    study, _, _ = boundary_inputs
    summary = summary_selecting_node(study, node=0, detected=False)
    for probability, n_claims, expected_kind, searched in (
        (0.0, 0, EffectKind.WAIT, False),
        (0.1, 0, EffectKind.OBSERVE, True),
        (1.0, 1, EffectKind.WAIT, False),
    ):
        held = dataclasses.replace(
            summary,
            p_exist=jnp.full_like(summary.p_exist, probability),
            p_claim=jnp.zeros((*summary.p_exist.shape, n_claims)),
        )
        criterion = dataclasses.replace(
            study.observer.criterion,
            claims=() if n_claims == 0 else (EarthCandidateClaim(),),
            target_claim=None if n_claims == 0 else "earth_candidate",
        )
        local = dataclasses.replace(
            study, observer=dataclasses.replace(study.observer, criterion=criterion)
        )
        decision = decide_once(local, summary=held, state=state_with(local))
        assert effect_kind(decision) == int(expected_kind)
        assert bool(decision.state.n_imaging.sum() > 0) == searched


@pytest.mark.slow(reason="physical channel plus two conditional dark-slot summaries")
def test_dark_existence_stays_below_one_and_falls_after_misses(boundary_inputs):
    """Fails if a high expected count saturates occupancy and makes misses inert."""
    study, boundary, _ = boundary_inputs
    params, _, _ = hand_cloud()
    st = boundary._post_settings(study)
    # Six slots, four detected tracks, two dark: condition Poisson(8) on N >= 4.
    before = ph._first_leaves(study.channel_model, 0, 0, 2, params, 0.0, 8.0, st)[0]
    after = ph._first_leaves(
        study.channel_model, 0, 0, 2, params, jnp.log(0.1), 8.0, st
    )[0]
    assert 0.0 < float(after) < float(before) < 1.0
    # P(N >= 5 | N >= 4) + P(N >= 6 | N >= 4), divided among two slots.
    assert float(before) == pytest.approx(0.8923851513715046)
    assert float(after) == pytest.approx(0.45332481716385115)


def test_dark_count_prior_handles_zero_and_roundoff_near_saturation():
    """Fails if an empty or very abundant population gives an invalid dark prior."""
    assert float(ph._dark_slot_prior(0.0, jnp.asarray(2), 6)) == 0.0
    assert float(ph._dark_slot_prior(8.0, jnp.asarray(0), 6)) == 0.0
    assert 0.0 < float(ph._dark_slot_prior(100.0, jnp.asarray(2), 6)) < 1.0


@pytest.mark.slow(reason="draws 4096 and 16384 distinct supported population rows")
def test_solar_first_tier_and_retry_reach_their_declared_distinct_sizes():
    """Fails if the first tier or retry reuses an exhausted fixed system pool."""
    population = PopulationModel.exovista()
    star = StarSpec("Sun", 1.0, 10.0, 0.0, 0.0, 1.0, 4.8)
    bands = (Band(500.0, 100.0), Band(940.0, 6.7))
    lane = SimpleNamespace(key=KEY, slot=jnp.asarray(6))
    original = None
    for factor in (1, ph.RETRY_FACTOR):
        size = ph.FIRST_TIER_DRAWS * factor
        draws, valid = population.draw_planets(
            ph._slot_key(lane, factor), star, size, bands
        )
        assert int(valid.sum()) == size
        periods = np.asarray(draws.ln_period_d)
        assert np.unique(periods).size == size
        if original is not None:
            assert not np.isin(periods, original).any()
        original = periods


@pytest.mark.slow(
    reason="two physical first tiers with independently keyed report fits"
)
def test_expected_host_count_is_stable_across_refit_keys(boundary_inputs):
    """Fails if a star's count prior is re-estimated from each report-set key."""
    population = dataclasses.replace(PopulationModel.exovista(), n_pool_systems=8)
    first, lane, st = first_tier(boundary_inputs, population)
    lane = eqx.tree_at(lambda x: x.key, lane, jax.random.fold_in(lane.key, 1))
    second = ph._first_tier_lane(
        boundary_inputs[0].channel_model, population, lane, st, 1
    )
    assert float(first.expected) == float(second.expected)
    host_key = jax.random.fold_in(jax.random.PRNGKey(0), lane.subject)
    expected = population.expected_planets(
        boundary_inputs[0].world_source.stars[0],
        jax.random.fold_in(host_key, ph._EXPECTED_TAG),
        st.n_estimate_systems,
    )
    assert float(first.expected) == float(expected)


@pytest.mark.slow(reason="compares serial and threaded population forecast programs")
def test_parallel_population_prior_preserves_all_leaves_and_host_order(
    boundary_inputs, monkeypatch
):
    """Fails if partitioning changes bytes or identical forecasts are recomputed.

    Changing the population must miss the cache; a hit must keep every leaf and
    the caller's prior state, as well as the existing host-order/precision guards.
    """
    study, boundary, population = boundary_inputs
    # Ten keyed subjects put one real host and two pads in the fourth partition.
    indices = jnp.arange(10) % study.caps.n_subjects
    channel = dataclasses.replace(
        study.channel_model,
        band_scenes=jax.tree.map(lambda x: x[indices], study.channel_model.band_scenes),
        band_leakage=tuple(x[indices] for x in study.channel_model.band_leakage),
        catalog_mass_kg=study.channel_model.catalog_mass_kg[indices],
    )
    prior = jax.tree.map(lambda x: x[indices], study.observer.prior)
    # Repeated physical hosts retain subject-specific draw keys.
    study = eqx.tree_at(
        lambda s: (s.channel_model, s.observer.prior),
        study.decision_view(),
        (channel, prior),
    )
    study = eqx.tree_at(lambda s: s.belief_population, study, population)
    with jax.default_device(None), jax.default_matmul_precision(None):
        serial = jax.block_until_ready(boundary.initial_summary(study))
        config = _thread_config()
    lock, active, peak, calls = Lock(), 0, 0, 0
    leaves = ph._population_leaves

    def counted(*args):
        nonlocal active, peak, calls
        assert _thread_config() == config
        with lock:
            active += 1
            calls += 1
            peak = max(peak, active)
        try:
            return jax.block_until_ready(leaves(*args))
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(ph, "_population_leaves", counted)
    device, precision = (
        jax.config.jax_default_device,
        jax.config.jax_default_matmul_precision,
    )
    try:
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
        jax.config.update("jax_default_matmul_precision", "highest")
        with jax.default_device(None), jax.default_matmul_precision(None):
            assert _thread_config() == config
            parallel = jax.block_until_ready(boundary.initial_summary(study, workers=4))
    finally:
        jax.config.update("jax_default_device", device)
        jax.config.update("jax_default_matmul_precision", precision)
    assert peak <= 2  # default-size forecasts need several GiB per live map

    for a, b in zip(jax.tree.leaves(serial), jax.tree.leaves(parallel), strict=True):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    assert parallel.fc_counts.dtype == jnp.float64
    assert not np.array_equal(parallel.fc_counts[0], parallel.fc_counts[1])

    assert calls == 4
    changed_prior = dataclasses.replace(prior, state=prior.state + 1.0)
    repeated_study = eqx.tree_at(lambda s: s.observer.prior, study, changed_prior)
    with jax.default_device(None), jax.default_matmul_precision(None):
        repeated = jax.block_until_ready(
            boundary.initial_summary(repeated_study, workers=4)
        )
    assert calls == 4
    for name in ph.SUMMARY_LEAVES:
        assert (
            np.asarray(getattr(repeated, name)).tobytes()
            == np.asarray(getattr(parallel, name)).tobytes()
        )
    assert (
        np.asarray(repeated.state).tobytes()
        == np.asarray(changed_prior.state).tobytes()
    )

    changed_study = eqx.tree_at(
        lambda s: s.belief_population, study, population_scaled(population, 2.0)
    )
    with jax.default_device(None), jax.default_matmul_precision(None):
        changed = jax.block_until_ready(
            boundary.initial_summary(changed_study, workers=4)
        )
    assert calls == 8
    assert not np.array_equal(changed.p_exist, parallel.p_exist)


def test_population_cache_key_separates_weak_and_strong_arrays():
    """Fails if a weakly typed leaf shares a forecast with its strong twin."""
    weak, strong = jnp.asarray(1.0), jnp.asarray(1.0, jnp.float64)
    assert weak.weak_type and not strong.weak_type
    key = ph._population_signature((), weak, None, 1, 1)
    assert key is not None
    assert key == ph._population_signature((), jnp.asarray(1.0), None, 1, 1)
    assert key != ph._population_signature((), strong, None, 1, 1)


def test_population_worker_inherits_prng_configuration(monkeypatch):
    """Fails if workers lose caller settings or the cache aliases array ranks."""

    def draw():
        key = jax.random.PRNGKey(0)
        return _thread_config(), jax.random.uniform(jax.random.split(key)[0], (4,))

    for partitionable, implementation in ((False, "threefry2x32"), (True, "rbg")):
        with (
            jax.threefry_partitionable(partitionable),
            jax.default_prng_impl(implementation),
        ):
            config, expected = draw()
            with ThreadPoolExecutor(max_workers=1) as pool:
                actual_config, actual = pool.submit(
                    _in_thread, draw, (), config
                ).result()
            assert actual_config == config
            assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()

    def execution_mode(_channel, _population, subjects, _summary):
        if _summary is not None:
            return jnp.full(subjects.shape, _summary.weights[500])
        if _channel is not None:
            return jnp.full(subjects.shape, _channel.ndim, dtype=int)
        return jnp.full(subjects.shape, jax.config.jax_disable_jit, dtype=bool)

    monkeypatch.setattr(ph, "_population_leaves", execution_mode)
    monkeypatch.setattr(ph, "_install_population", lambda _prior, _st, result: result)
    for mode in (True, False):
        with jax.disable_jit(mode):
            actual = ph._parallel_population_summary(
                None, None, SimpleNamespace(n_subjects=2), None, 4
            )
        assert np.asarray(actual).tolist() == [mode, mode]

    for value in (jnp.asarray(0.0), jnp.asarray([0.0])):
        actual = ph._parallel_population_summary(
            value, None, SimpleNamespace(n_subjects=2), None, 4
        )
        assert np.asarray(actual).tolist() == [value.ndim, value.ndim]

    @dataclasses.dataclass(frozen=True)
    class Settings:
        weights: jax.Array

    for value in (0.0, 1.0):
        settings = Settings(jnp.zeros(1024).at[500].set(value))
        actual = ph._parallel_population_summary(
            None, None, SimpleNamespace(n_subjects=2), settings, 4
        )
        assert np.asarray(actual).tolist() == [value, value]
