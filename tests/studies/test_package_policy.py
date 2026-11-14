"""Package selection, ledger capacity and packed policy state."""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.adapters.photomancy import ReportRefit
from spaceodyssey.core.enums import AdmissionCode, EffectKind
from spaceodyssey.core.records import pack, unpack
from spaceodyssey.studies import bracket as br
from tests.studies.bracket_helpers import (
    assert_trees_equal,
    build,
    decide_once,
    effect_kind,
    n_refreshes,
    policy_trace,
    random_state,
    spec_row,
    starts_spectrograph,
    state_with,
    summary_selecting_node,
)

pytestmark = pytest.mark.medium(
    reason="builds a physical study and compiles package selection"
)


@pytest.fixture(scope="module")
def fixture():
    return build(run=False).study


def test_a_package_starts_only_at_its_selected_node(fixture):
    """Fails if a selected option at a later node is started now (the mission waits)."""
    d = decide_once(
        fixture,
        summary=summary_selecting_node(fixture, node=3),
        state=state_with(fixture),
    )
    assert effect_kind(d) == EffectKind.WAIT
    now = decide_once(
        fixture,
        summary=summary_selecting_node(fixture, node=0),
        state=state_with(fixture),
        balance_d=10.0,
    )
    assert effect_kind(now) == EffectKind.OBSERVE
    assert int(now.state.n_spectra.sum()) == 1
    short_study = dataclasses.replace(
        fixture,
        horizon_tick=jnp.asarray(30 * fixture.observer.ticks_per_day),
    )
    past_end = decide_once(
        short_study,
        summary=summary_selecting_node(fixture, node=3),
        state=state_with(fixture),
        balance_d=10.0,
    )
    assert past_end.selected_d == 0


def test_the_policy_reads_the_ledger_balance(fixture):
    """Fails if spending is tracked in a private copy instead of the ledger balance."""
    summary = summary_selecting_node(fixture, node=0)
    modes = fixture.channel_model.mode_table
    channel = modes.spectrograph_channels[1]
    summary = dataclasses.replace(
        summary, available=jnp.zeros_like(summary.available).at[:, 0, channel].set(True)
    )
    state = state_with(fixture)
    a = decide_once(fixture, summary=summary, state=state, balance_d=10.0)
    b = decide_once(fixture, summary=summary, state=state, balance_d=0.5)
    assert a.selected_d > b.selected_d and b.selected_d <= 0.5
    at_end = decide_once(
        fixture,
        summary=dataclasses.replace(
            summary,
            ref_tick=jnp.full_like(
                summary.ref_tick,
                int(fixture.horizon_tick) - round(0.1 * fixture.observer.ticks_per_day),
            ),
        ),
        state=state,
        tick_d=float(fixture.horizon_tick) / fixture.observer.ticks_per_day - 0.1,
        balance_d=10.0,
    )
    assert at_end.selected_d <= 0.1
    assert effect_kind(at_end) == EffectKind.WAIT
    modes = fixture.channel_model.mode_table
    channel = modes.spectrograph_channels[1]
    only_one = dataclasses.replace(
        summary,
        available=jnp.zeros_like(summary.available).at[0, 0, channel].set(True),
    )
    cost_d = float(fixture.channels.cost[channel, modes.resource("telescope_days")])
    exact = decide_once(fixture, summary=only_one, state=state, balance_d=cost_d)
    assert effect_kind(exact) == EffectKind.OBSERVE
    assert exact.selected_d == cost_d
    assert int(exact.effects.channel_index[0]) == channel
    short = decide_once(
        fixture,
        summary=only_one,
        state=state,
        balance_d=cost_d - 0.01,
    )
    assert short.selected_d <= cost_d - 0.01
    near = decide_once(
        fixture,
        summary=only_one,
        state=state,
        balance_d=cost_d - fixture.caps.balance_quantum / 4,
    )
    assert effect_kind(near) == EffectKind.WAIT


def test_a_star_at_its_spectrum_cap_gets_no_spectrum(fixture):
    """Fails if the policy can exceed the declared spectrograph row capacity."""
    d = decide_once(
        fixture,
        summary=summary_selecting_node(fixture, node=0),
        state=state_with(fixture, n_spectra=br.MAX_SPECTRA),
    )
    assert not starts_spectrograph(d, fixture)


def test_a_star_at_its_imaging_cap_gets_no_imaging(fixture):
    """Fails if an imaging package can exceed the declared row cap."""
    summary = summary_selecting_node(fixture, node=0, detected=False)
    summary = dataclasses.replace(
        summary, n_obs=jnp.full_like(summary.n_obs, br.MAX_VISITS)
    )
    d = decide_once(
        fixture,
        summary=summary,
        state=state_with(fixture, n_imaging=br.MAX_VISITS),
    )
    assert effect_kind(d) == EffectKind.WAIT


def test_a_policy_state_round_trips_through_the_codec(fixture):
    """Fails if the policy state packs to another width than its declared layout."""
    policy = fixture.agenda.policies[0]
    t = policy.state_layout(fixture.caps)
    x = random_state(t, seed=0)
    assert_trees_equal(unpack(t, pack(x)), x)
    assert pack(x).size == policy.state_size(fixture.caps.n_subjects)


def test_the_initial_state_starts_the_catalog_kickoff(fixture):
    """Fails if decide_once silently bypasses the initial catalog kickoff."""
    d = decide_once(fixture)
    assert effect_kind(d) == EffectKind.OBSERVE
    assert (
        int(d.effects.channel_index[0])
        == fixture.channel_model.mode_table.catalog_channel
    )
    assert int(d.state.kick) == 1 and int(d.state.n_trace) == 0


def test_measurement_grants_gate_waiting_for_a_refit(fixture):
    """Fails if stale summaries are ignored or private rows stall the policy."""
    summary = summary_selecting_node(fixture, node=0)
    state = state_with(fixture, n_imaging=1)
    d = decide_once(fixture, summary=summary, state=state)
    assert effect_kind(d) == EffectKind.WAIT and int(d.state.n_trace) == 0
    observer = dataclasses.replace(
        fixture.observer, grants=br.GRANT_PRESETS["prior_planned"]
    )
    private = dataclasses.replace(fixture, observer=observer)
    d = decide_once(private, summary=summary, state=state)
    assert int(d.state.n_trace) == 1


def test_exhausted_trace_capacity_is_an_explicit_failure(fixture):
    """Fails if an exhausted trace silently drops or overwrites a refresh row."""
    state = state_with(fixture, n_trace=fixture.agenda.policies[0].trace_capacity)
    with pytest.raises(Exception, match="trace capacity"):
        decide_once(
            fixture, summary=summary_selecting_node(fixture, node=0), state=state
        )


def test_every_audience_starts_with_population_forecasts(fixture):
    """Fails if a dark slot lacks population probabilities or forecast leaves."""
    prior = fixture.observer.init(fixture.caps)
    assert np.any(np.asarray(prior.p_exist) > 0)
    assert np.any(np.asarray(prior.p_det) > 0)
    assert np.any(np.asarray(prior.p_claim) > 0)
    assert np.any(np.asarray(prior.fc_counts) > 0)
    private = dataclasses.replace(
        fixture.observer, grants=br.GRANT_PRESETS["prior_planned"]
    )
    assert_trees_equal(prior, private.init(fixture.caps))
    model = fixture.channel_model
    model = eqx.tree_at(
        lambda x: x.band_scenes[0].star_flux_density_jy,
        model,
        model.band_scenes[0].star_flux_density_jy / 2,
    )
    planned_study = br.bracket_study(
        fixture.world_source,
        fixture.channel_model,
        horizon_d=float(fixture.horizon_tick) / fixture.observer.ticks_per_day,
        budget_d=float(fixture.budget[-1, model.mode_table.resource("telescope_days")]),
        observer=fixture.observer,
        policy=fixture.agenda.policies[0],
        planning_model=model,
    )
    planned = planned_study.observer.prior
    assert not np.array_equal(planned.fc_counts, prior.fc_counts)
    np.testing.assert_array_equal(planned.p_claim, prior.p_claim)


@pytest.fixture(scope="module")
def built():
    return build()


@pytest.mark.slow(reason="compiles the chunked refresh driver once")
def test_no_admission_is_refused_for_funds(built):
    """Fails if the policy asks for an observation the ledger cannot pay for."""
    assert (
        int((np.asarray(built.run.trace.admit_code) == AdmissionCode.NO_FUNDS).sum())
        == 0
    )
    assert not np.any(np.asarray(built.run.trace.admit_code) == AdmissionCode.LANE_BUSY)
    resource = built.study.channel_model.mode_table.resource("telescope_days")
    for world in range(built.roots.shape[0]):
        trace = policy_trace(built, world)
        assert np.any(trace[:, 4] > 0)
        charged_d = float(
            built.study.budget[-1, resource]
            - built.run.carry.ledger.balance[world, -1, resource]
        )
        np.testing.assert_allclose(trace[-1, 1], charged_d, atol=1e-10, rtol=1e-12)


@pytest.mark.slow(reason="compiles the chunked refresh driver once")
def test_every_refresh_writes_a_price_trace_row(built):
    """Fails if a refresh leaves no trace row or the cumulative columns decrease."""
    for world in range(built.roots.shape[0]):
        trace = policy_trace(built, world)
        assert trace.shape[0] == n_refreshes(built, world)
        assert bool(np.all(np.diff(trace[:, 1]) >= 0)) and bool(
            np.all(np.diff(trace[:, 5]) >= 0)
        )
        assert jnp.isfinite(trace).all()


def test_the_study_refuses_an_undersized_named_policy_state(fixture):
    """Fails if Study accepts a vector narrower than the policy's codec layout."""
    with pytest.raises(ValueError, match="n_pstate"):
        dataclasses.replace(fixture, caps=dataclasses.replace(fixture.caps, n_pstate=1))


def test_a_pending_wait_does_not_start_a_second_telescope_row(fixture):
    """Fails if a queued wait starts a package while the telescope is occupied."""
    summary = summary_selecting_node(fixture, node=0)
    first = decide_once(
        fixture, summary=summary, state=state_with(fixture), balance_d=10.0
    )
    assert effect_kind(first) == EffectKind.OBSERVE
    second = decide_once(
        fixture, summary=summary, state=first.state, tick_d=0.01, balance_d=9.0
    )
    assert effect_kind(second) == EffectKind.WAIT
    assert int(second.state.n_spectra.sum()) == 1


@pytest.mark.slow(
    reason="compiles an empty-population forecast with a separate support range"
)
def test_an_empty_belief_population_has_finite_zero_forecasts(fixture):
    """Fails if a population with no valid draws initializes NaN probabilities."""
    population = fixture.world_source.declared_population()
    population = eqx.tree_at(
        lambda p: p.occurrence.rate,
        population,
        jnp.zeros_like(population.occurrence.rate),
    )
    summary = ReportRefit(kernel="posterior").initial_summary(
        dataclasses.replace(fixture, belief_population=population)
    )
    for name in ("p_exist", "p_det", "p_claim", "fc_counts", "fc_variance", "ess"):
        value = np.asarray(getattr(summary, name))
        assert np.isfinite(value).all() and not value.any()


@pytest.mark.slow(reason="physical study and package selection take 16 s cold locally")
def test_a_package_cannot_promise_more_spectra_than_remain(fixture):
    """Fails if a two-pass package is started with only one spectrum row left."""
    criterion = dataclasses.replace(fixture.observer.criterion, repeat=jnp.asarray([2]))
    study = dataclasses.replace(
        fixture, observer=dataclasses.replace(fixture.observer, criterion=criterion)
    )
    summary = summary_selecting_node(study, node=0)
    decision = decide_once(
        study,
        summary=summary,
        state=state_with(study, n_spectra=br.MAX_SPECTRA - 1),
    )
    assert effect_kind(decision) == EffectKind.WAIT
    assert not bool(decision.state.has_package.any())


@pytest.mark.slow(reason="physical study and repeated package decisions take 16 s cold")
def test_trace_value_is_the_current_label_after_failed_and_successful_rows(fixture):
    """Fails if unfinished forecast gain is repeatedly booked as realized value."""
    criterion = dataclasses.replace(fixture.observer.criterion, repeat=jnp.asarray([2]))
    study = dataclasses.replace(
        fixture, observer=dataclasses.replace(fixture.observer, criterion=criterion)
    )
    modes = study.channel_model.mode_table
    summary = summary_selecting_node(study, node=0)
    current = criterion.init_state(summary.n_det.shape[1], modes.n_bands)
    state = state_with(study)
    rows = [spec_row(0, -10.0, 1.0), spec_row(0, 100.0, 1.0), spec_row(0, 100.0, 1.0)]
    for index, (row, expected) in enumerate(zip(rows, (0.0, 0.0, 1.0), strict=True)):
        current = criterion.update(current, row, modes)
        summary = dataclasses.replace(
            summary, state=summary.state.at[0].set(pack(current))
        )
        decision = decide_once(study, summary=summary, state=state, tick_d=2.0 * index)
        assert float(decision.trace_row[5]) == expected
        state = decision.state
    assert decision.state.trace.shape[1] == 6
