"""The declared population: the truth keeps every planet, beliefs read the population.

A world source keeps every drawn planet up to its capacity, whatever orbit it has,
and declares the `PopulationModel` it draws from. A refit reads its prior bounds from
the belief population of the study's decision view, never from the stored worlds.
"""

import dataclasses
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Rearth2AU, d2s
from skyscapes.population import mass_to_radius

from spaceodyssey.adapters import photomancy as ph
from spaceodyssey.adapters.skyscapes import (
    RADIUS_GRID_POINTS,
    ExoVistaWorldSource,
    PopulationModel,
    StarSpec,
    SupportRange,
)
from spaceodyssey.core.records import unpack
from tests.adapters.posterior_helpers import realize_visits, visit_request
from tests.adapters.test_truth_albedo import STARS, hand_population
from tests.studies.bracket_helpers import (
    BUDGET_D,
    FIXTURE_BANDS,
    HORIZON_D,
    N_SLOTS,
    build,
    fixture_channel,
    fixture_files,
    fixture_source,
    prior_summary,
)
from tests.studies.bracket_helpers import br as bracket

GEN = PopulationModel.exovista()
BANDS = FIXTURE_BANDS
# A Sun-like host at 1.3 pc, so a 20-day orbit (0.14 AU) clears the inner working
# angle; the flux is the 1000 nm flux of the nearer fixture star scaled to 1.3 pc.
NEAR_STARS = (StarSpec("HIP 0", 1.0, 1.3, 10.0, 20.0, 1.0, 7.2),)
SMALL = dict(
    n_round=2**9,
    n_rounds=1,
    n_centers=16,
    n_constants=1,
    max_rounds=2,
    n_particles=128,
    grid_points=128,
    n_first=64,
)
TICK = round(250 * d2s * 1000)


def type_radius_edges(types):
    """`(type, (r_lo, r_hi))` of every type the population can draw, Earth radii.

    A type's edges are its radius range (an exoEarth-candidate lower edge divided by
    the square root of its largest scaled semimajor axis) cut to the forecaster
    radii of `GEN`'s occurrence grid, read on a grid twice as fine as the model's.
    """
    edges = np.asarray(GEN.occurrence.mass_edges_Mearth)
    masses = np.geomspace(edges.min(), edges.max(), 2 * RADIUS_GRID_POINTS)
    drawn = np.asarray(mass_to_radius(masses))
    for t in range(len(types.names)):
        lo = float(types.radius_min_Rearth[t])
        if bool(types.earth_candidate[t]):
            lo /= float(np.sqrt(types.a_max_AU[t]))
        lo = max(lo, float(drawn.min()))
        hi = min(float(types.radius_max_Rearth[t]), float(drawn.max()))
        if lo <= hi:
            yield t, (lo, hi)


@pytest.fixture(scope="module")
def worlds():
    """`(WORLDS_A, WORLDS_B)`: the fixture worlds, and the same with radii tripled."""
    a = ExoVistaWorldSource.from_files(
        [[fixture_files()[0]]], n_slots=N_SLOTS, bands=FIXTURE_BANDS, generator=GEN
    )
    b = eqx.tree_at(lambda s: s.population.Rp_AU, a, a.population.Rp_AU * 3.0)
    return a, b


@pytest.fixture(scope="module")
def bounds_channel(worlds):
    """One channel for the common hosts of both stored populations."""
    return fixture_channel(worlds[0])


def build_study(worlds, belief_population=None, *, channel):
    """The oracle bracket study of `worlds`, with `belief_population` when given."""
    return bracket.bracket_study(
        worlds,
        channel,
        arm="oracle",
        horizon_d=HORIZON_D,
        budget_d=BUDGET_D,
        observer=bracket.SummaryObserver(
            prior=prior_summary(channel, len(worlds.stars))
        ),
        policy=bracket.BracketPolicy(BUDGET_D, N_SLOTS),
        belief_population=belief_population,
    )


def boundary_bounds(study):
    """`(scale range, period support)` a posterior `ReportRefit` reads from `study`."""
    boundary = ph.ReportRefit(kernel="posterior", truth_groups=())
    st = boundary._post_settings(study)
    return st.ln_lam_range, st.period_support_d


def test_the_truth_keeps_planets_outside_the_belief_support():
    """Fails if the world source deletes a planet the belief cannot fit."""
    pop = hand_population(periods_d=[20.0, 400.0], e=[0.0, 0.95])  # both outside
    for period, e in zip([20.0, 400.0], [0.0, 0.95], strict=True):
        assert not GEN.support.contains(period, e)
    source = ExoVistaWorldSource.from_population(
        pop, STARS, n_slots=4, bands=BANDS, generator=GEN
    )
    assert int(source.population.present.sum()) == 4  # two planets on each star
    assert int(source.dropped.sum()) == 0
    np.testing.assert_allclose(
        np.sort(np.asarray(source.population.period_d[0, 0, :2])), [20.0, 400.0]
    )
    assert source.declared_population() is GEN


def test_planets_past_capacity_are_dropped_brightest_first():
    """Fails if the capacity keeps a faint planet over a bright one."""
    # equal a, e and type (so equal albedo): the peak flux ratio orders by radius alone
    pop = hand_population(
        n_planets=3,
        radii_rearth=[1.0, 4.0, 2.0],
        a_AU=[1.0] * 3,
        e=[0.0] * 3,
        type_index=[0] * 3,
    )
    source = ExoVistaWorldSource.from_population(
        pop, STARS, n_slots=2, bands=BANDS, generator=GEN
    )
    assert int(source.dropped[0, 0]) == 1
    radii = np.asarray(source.population.Rp_AU[0, 0]) / Rearth2AU
    assert sorted(radii) == pytest.approx([2.0, 4.0])


def test_the_generator_is_required():
    """Fails if a world source silently inherits a population."""
    with pytest.raises(TypeError, match="generator"):
        ExoVistaWorldSource.from_population(
            hand_population(), STARS, n_slots=2, bands=BANDS
        )


def test_refit_bounds_come_from_the_belief_population_not_the_worlds(
    worlds, bounds_channel
):
    """Fails if a refit setting is derived from stored or realized worlds."""
    worlds_a, worlds_b = worlds
    a = build_study(worlds_a, belief_population=GEN, channel=bounds_channel)
    b = build_study(worlds_b, belief_population=GEN, channel=bounds_channel)
    assert boundary_bounds(a) == boundary_bounds(b)
    narrow = dataclasses.replace(GEN, support=SupportRange((100.0, 1000.0)))
    assert boundary_bounds(
        build_study(worlds_a, belief_population=narrow, channel=bounds_channel)
    ) != boundary_bounds(a)


def test_without_a_belief_population_the_bounds_are_the_generators(
    worlds, bounds_channel
):
    """Fails if a study with no belief population falls back to anything but the
    world source's declared generator."""
    narrow = dataclasses.replace(GEN, support=SupportRange((100.0, 1000.0)))
    source = eqx.tree_at(lambda s: s.generator, worlds[0], narrow)
    study = build_study(source, channel=bounds_channel)
    assert study.belief_population is None
    band = study.channel_model.detection_band
    want = (narrow.scale_range_au2(band), narrow.period_support_d)
    assert boundary_bounds(study) == want
    assert want != boundary_bounds(
        build_study(worlds[0], belief_population=GEN, channel=bounds_channel)
    )


def test_the_scale_range_holds_every_type_at_its_radius_edges():
    """Fails if a planet the population can draw lies outside the refit's scale
    prior."""
    lo, hi = GEN.scale_range_au2(BANDS[0])
    for t, (r_lo, r_hi) in type_radius_edges(GEN.types):
        for r in (r_lo, r_hi):
            v = np.log(
                GEN.types.geometric_albedo(t, BANDS[0].center_nm) * (r * Rearth2AU) ** 2
            )
            assert lo <= v <= hi


def refit_one_planet(period_d, a_quadrature=True):
    """`(summary, rows)` of a posterior refit of one star holding one planet.

    The planet is a Jupiter-sized giant at `period_d` around a Sun-like star at
    1.3 pc (`NEAR_STARS`). With `a_quadrature` the orbit is circular and face-on, so
    it sits at quadrature at every visit; otherwise it is edge-on. One visit of the
    longest detection mode on day 30 is refit with the small proposal settings.
    `summary` is the observer's `DecisionSummary` after the refit and `rows.detected`
    the slot-0 detection flag of each visit.
    """
    template = fixture_source(1).stars[0]
    flux_scale = 3555.0 / template.flux_at(FIXTURE_BANDS[0].center_nm)
    hosts = tuple(
        dataclasses.replace(
            star,
            band_fluxes_jy=tuple(
                (band.center_nm, template.flux_at(band.center_nm) * flux_scale)
                for band in FIXTURE_BANDS
            ),
        )
        for star in NEAR_STARS
    )
    pop = hand_population(
        n_planets=1,
        periods_d=[period_d],
        radii_rearth=[11.0],
        stars=hosts,
        mass_Mearth=[300.0],
        inc_deg=0.0 if a_quadrature else 90.0,
        type_index=[8],
    )
    pop = jax.tree.map(lambda x: jnp.concatenate([x, x]), pop)  # two worlds
    source = ExoVistaWorldSource.from_population(
        pop, hosts, n_slots=N_SLOTS, bands=FIXTURE_BANDS, generator=GEN
    )
    _, study, *_ = build(run=False, arm="replan", source=source)
    channel = study.channel_model
    truth = source.take(0)
    visits = realize_visits(
        channel, truth, 0, channel.mode_table.channel("detection", 1), [30.0]
    )
    boundary = ph.ReportRefit(
        kernel="posterior",
        truth_groups=(),
        lanes=1,
        caps=study.caps,
        ess_floor=1.0,
        **SMALL,
    )
    result = boundary.refit_batch(study, visit_request(study, {0: visits}, tick=TICK))
    names = [name for name, _, _ in ph.slice_spec(study)]
    leaves = {n: x[0, 0] for n, x in zip(names, result.slices, strict=True)}
    summary = ph.DecisionSummary(**leaves, aux=jnp.zeros(0))
    rows = SimpleNamespace(
        detected=np.asarray(
            [
                unpack(channel.measurement_template, r).detected[0] > 0.5
                for r, _ in visits
            ]
        )
    )
    return summary, rows


@pytest.mark.slow(
    reason=(
        "one posterior refit of a planet outside the belief support; "
        "about 20 s on a quiet Mac"
    )
)
def test_an_unsupported_planet_is_detected_and_fits_finitely():
    """Fails if a planet outside the belief support crashes the refit, returns
    non-finite leaves, or never detects.
    """
    summary, rows = refit_one_planet(period_d=20.0, a_quadrature=True)
    assert rows.detected.any()
    assert int(summary.status[0, 0]) in {
        ph.SLOT_OK,
        ph.SLOT_NO_SUPPORT,
        ph.SLOT_DEGRADED,
    }
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(summary))
