"""Transition folds and probabilistic packages on small declared mode tables."""

import dataclasses
import json
from importlib.resources import files

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import integers
from jax.scipy.special import ndtr, ndtri

from spaceodyssey.adapters.measurement import measurement_template
from spaceodyssey.adapters.modes import ModeSpec, ModeTable, load_mode_specs
from spaceodyssey.adapters.photomancy import SLOT_OK, DecisionSummary
from spaceodyssey.adapters.skyscapes import Band
from spaceodyssey.core.enums import I64
from spaceodyssey.core.records import pack, unpack
from spaceodyssey.criteria import transition
from spaceodyssey.criteria.claims import EarthCandidateClaim
from spaceodyssey.criteria.transition import (
    TransitionSettings,
    TransitionTable,
    load_transition_table,
    pass_probability,
)
from spaceodyssey.studies.selection import select
from tests.criteria.options_reference import reference_options
from tests.studies.bracket_helpers import assert_trees_equal, random_state

TELESCOPE = 2
EXAMPLE = files("spaceodyssey.criteria") / "examples/h2o_then_o2.json"
EXAMPLE_MODES = files("spaceodyssey.adapters") / "examples/h2o_o2_modes.json"
MODES = ModeTable(
    (
        ModeSpec("image", "imaging", Band(500.0, 100.0), 0.2, 100.0, None, 7.0),
        ModeSpec("a", "spectrograph", Band(940.0, 6.7), 0.2, 300.0, 140.0, None),
        ModeSpec("b", "spectrograph", Band(760.0, 5.4), 0.2, 300.0, 140.0, None),
    ),
    (1.0,),
)


def one_step_table(repeat=1):
    return TransitionTable(
        step_names=("a",),
        labels=("met",),
        mode_names=("a",),
        claims=(),
        target_claim=None,
        step_modes=[[True]],
        requirement_snr=[5.0],
        margin=[0.0],
        repeat=[repeat],
        on_pass=[1],
        on_fail=[-1],
        label_values=[1.0],
        start=0,
        step_labels=[-1],
    )


def two_step_table():
    return TransitionTable(
        step_names=("a", "b"),
        labels=("none", "a", "ab"),
        mode_names=("a", "b"),
        claims=(),
        target_claim=None,
        step_modes=[[True, False], [False, True]],
        requirement_snr=[5.0, 5.0],
        margin=[0.0, 0.0],
        repeat=[1, 1],
        on_pass=[1, 4],
        on_fail=[-1, -1],
        label_values=[0.0, 0.0, 1.0],
        start=0,
        step_labels=[0, 1],
    )


def spectral_row(*, mode=1, counts=6.0, variance=1.0):
    row = jax.tree.map(
        lambda x: jnp.zeros(x.shape, x.dtype), measurement_template(2, 3)
    )
    band = MODES.band_index(MODES.modes[mode].name)
    return dataclasses.replace(
        row,
        mode=jnp.asarray(mode, I64),
        valid_science=jnp.asarray(1.0),
        assignment=jnp.asarray([0, -1], I64),
        counts=row.counts.at[0, band].set(counts),
        variance=row.variance.at[0, band].set(variance),
    )


_update = eqx.filter_jit(lambda table, state, row: table.update(state, row, MODES))


def fold_passing_rows(table, n_rows):
    state = table.init_state(1, 3)
    for _ in range(n_rows):
        state = _update(table, state, spectral_row())
    return state


def package_inputs(*, p=(0.75, 0.60), t_d=(10.0, 20.0), n_slots=1, n_grid=2):
    summary = DecisionSummary.zeros(1, n_slots, n_grid, 1, n_an=1, n_variants=3)
    summary = jax.tree.map(lambda x: x[0], summary)
    counts = jnp.asarray([0.0, 5.0 + ndtri(p[0]), 5.0 + ndtri(p[1])])
    summary = dataclasses.replace(
        summary,
        p_exist=jnp.ones(n_slots),
        n_det=jnp.full(n_slots, 3, I64),
        status=jnp.full(n_slots, SLOT_OK, I64),
        fc_counts=jnp.broadcast_to(
            counts[None, None, :, None], (n_slots, n_grid, 3, 3)
        ),
        fc_variance=jnp.ones((n_slots, n_grid, 3, 3)),
        p_det=jnp.full((n_slots, n_grid, 1), 0.8),
    )
    cost = jnp.zeros((4, 3)).at[:, TELESCOPE].set(jnp.asarray([2.0, *t_d, 0.0]))
    settings = TransitionSettings(
        candidate_detections=3,
        candidate_scatter_lod=1.0,
        imaging_lod_arcsec=0.1,
        channel_cost=cost,
        resources=MODES.resources,
    )
    return summary, settings


_options = eqx.filter_jit(
    lambda table, state, summary, settings: table.options(
        state, summary, 0, MODES, settings
    )
)
_options_with_modes = eqx.filter_jit(TransitionTable.options)


def options_of(table, *, p, t_d):
    summary, settings = package_inputs(p=p, t_d=t_d)
    return _options(table, table.init_state(1, 3), summary, settings)


def test_pass_probability_matches_the_closed_form_with_no_prior_data():
    """Fails if the law drops the margin or the prior state."""
    s, m = 7.0, 0.5
    for snr in (5.0, 7.5, 9.0):
        p = pass_probability(0.0, 0.0, jnp.full(3, snr), jnp.ones(3), s, m)
        assert float(p) == pytest.approx(float(ndtr(snr - s - m)), rel=1e-12)
    assert float(
        pass_probability(2.0, 3.0, jnp.full(3, 8.0), jnp.ones(3), 5.0, 0.5)
    ) == pytest.approx(float(ndtr(-1.0)))


def test_pass_probability_is_finite_at_zero_and_infinite_variance():
    """Fails if an empty band or an unreachable mode returns NaN, or no data passes."""
    assert (
        float(pass_probability(10.0, 1.0, jnp.zeros(3), jnp.zeros(3), 5.0, 0.0)) == 1.0
    )
    assert (
        float(pass_probability(0.0, 0.0, jnp.zeros(3), jnp.zeros(3), 5.0, 0.0)) == 0.0
    )
    assert (
        float(
            pass_probability(
                0.0, 0.0, jnp.full(3, jnp.inf), jnp.full(3, jnp.inf), 5.0, 0.0
            )
        )
        == 0.0
    )
    assert (
        float(pass_probability(4.0, 1.0, jnp.zeros(3), jnp.zeros(3), 5.0, 0.0)) == 0.0
    )
    assert (
        float(pass_probability(jnp.nan, 1.0, jnp.ones(3), jnp.ones(3), 5.0, 0.0)) == 0.0
    )


@settings(max_examples=25, deadline=None)
@given(repeat=integers(1, 3))
def test_a_step_needs_its_repeat_count_of_passes(repeat):
    """Fails if a repeat count is ignored or one pass is counted twice."""
    table = one_step_table(repeat=repeat)
    state = fold_passing_rows(table, n_rows=repeat - 1)
    assert int(state.node[0]) == 0 and int(state.passes[0]) == repeat - 1
    assert int(fold_passing_rows(table, n_rows=repeat).node[0]) == table.n_steps


def test_a_transition_state_round_trips_through_the_codec():
    """Fails if the state packs to another width than its template's."""
    t = two_step_table().state_template(4, 3)
    x = random_state(t, seed=0)
    assert_trees_equal(unpack(t, pack(x)), x)


def test_two_step_package_gain_and_time():
    """Fails if later steps are charged when they are not reached."""
    opts = options_of(two_step_table(), p=(0.75, 0.60), t_d=(10.0, 20.0))
    best = jnp.unravel_index(jnp.argmax(opts.gain), opts.gain.shape)
    assert float(opts.gain[best]) == pytest.approx(0.45)
    assert float(opts.time[best][TELESCOPE]) == pytest.approx(25.0)
    assert int(opts.first_channel[best]) == 1


def test_the_example_files_load_with_declared_values():
    """Fails if example bands, requirements, margins or claims are implicit."""
    table = load_transition_table(
        EXAMPLE, claims={"earth_candidate": EarthCandidateClaim()}
    )
    modes = load_mode_specs(EXAMPLE_MODES)
    assert [m.band.center_nm for m in modes] == [940.0, 760.0] and table.mode_names == (
        "ifs_940",
        "ifs_760",
    )
    assert table.target_claim == "earth_candidate" and float(table.margin[0]) == 0.0
    np.testing.assert_array_equal(table.requirement_snr, [5.0, 5.0])
    np.testing.assert_array_equal(table.margin, [0.0, 0.0])
    np.testing.assert_array_equal(table.repeat, [1, 1])
    np.testing.assert_array_equal(table.on_fail, [-1, -1])
    assert table.step_names == ("h2o", "o2") and table.labels == (
        "none",
        "h2o",
        "h2o_o2",
    )
    np.testing.assert_array_equal(table.label_values, [0.0, 0.5, 1.0])
    for mode in modes:
        assert mode.band.width_nm == pytest.approx(mode.band.center_nm / 140.0)
        assert (mode.optics_throughput, mode.frame_time_s, mode.resolving_power) == (
            0.29165,
            300.0,
            140.0,
        )
        assert mode.snr_threshold is None
    assert "assumption" in json.loads(EXAMPLE_MODES.read_text())
    assert json.loads(EXAMPLE.read_text())["citations"] == [
        "stark_lower_2015",
        "latouf_bayesian_2023",
        "latouf_bayesian_2023a",
        "stark_paths_2024",
    ]
    with pytest.raises(KeyError, match="earth_candidate"):
        load_transition_table(EXAMPLE, claims={})


def test_only_assigned_valid_rows_in_an_allowed_mode_advance_the_step():
    """Fails if other bands, invalid science or unassigned truth signal count."""
    table = two_step_table()
    state = table.init_state(2, 3)
    row = spectral_row(counts=3.0, variance=1.0)
    for ignored in (
        spectral_row(mode=2),
        dataclasses.replace(row, valid_science=jnp.asarray(0.0)),
        dataclasses.replace(row, assignment=jnp.full(2, -1, I64)),
    ):
        assert_trees_equal(_update(table, state, ignored), state)
    state = _update(table, state, row)
    assert int(state.node[0]) == 0
    state = _update(table, state, spectral_row(counts=5.0))
    np.testing.assert_array_equal(state.node, [1, 0])
    np.testing.assert_array_equal(state.sum_counts, 0.0)
    assert int(_update(table, state, spectral_row(mode=2)).node[0]) == 4
    table = dataclasses.replace(
        one_step_table(),
        mode_names=("a", "b"),
        step_modes=[[True, True]],
        margin=[1.0],
    )
    state = table.init_state(2, 3)
    state = dataclasses.replace(state, sum_counts=state.sum_counts.at[0, 2].set(2.0))
    state = _update(table, state, spectral_row(counts=5.0))
    assert int(state.node[0]) == 0
    state = _update(table, state, spectral_row(counts=4.0))
    assert int(state.node[0]) == 1 and float(state.sum_counts[0, 2]) == 2.0
    row = dataclasses.replace(spectral_row(mode=2), assignment=jnp.asarray([1, -1]))
    final = _update(table, state, row)
    np.testing.assert_array_equal(final.node, [1, 1])
    assert_trees_equal(_update(table, final, row), final)


def test_active_steps_retain_achieved_value_and_price_only_remaining_gain():
    """Fails if entering oxygen loses water credit or charges that credit twice."""
    table = dataclasses.replace(
        two_step_table(), label_values=jnp.asarray([0.0, 0.5, 1.0])
    )
    state = fold_passing_rows(table, 1)
    values, labels = table.value(state, jnp.empty((1, 0)), 0)
    np.testing.assert_array_equal(values, [0.5])
    np.testing.assert_array_equal(labels, [1])
    summary, settings = package_inputs()
    opts = _options(table, state, summary, settings)
    assert float(opts.gain[0, 2]) == pytest.approx(0.6 * 0.5)
    assert float(opts.time[0, 2, TELESCOPE]) == pytest.approx(20.0)
    initial = _options(table, table.init_state(1, 3), summary, settings)
    assert float(initial.gain[0, 1]) == pytest.approx(0.75 * (0.5 + 0.6 * 0.5))
    targeted = dataclasses.replace(
        table, claims=(EarthCandidateClaim(),), target_claim="earth_candidate"
    )
    np.testing.assert_allclose(targeted.value(state, jnp.asarray([[0.2]]), 0)[0], [0.1])
    opts = _options(
        targeted,
        state,
        dataclasses.replace(summary, p_claim=jnp.asarray([[0.2]])),
        settings,
    )
    assert float(opts.gain[0, 2]) == pytest.approx(0.2 * 0.6 * 0.5)


def test_imaging_gate_handles_dark_slots_bad_status_and_scatter():
    """Fails if occupancy is omitted or counts bypass the fit/scatter gate."""
    table = two_step_table()
    summary, settings = package_inputs()
    state = table.init_state(1, 3)
    for change in ({"status": jnp.asarray([-1])}, {"scatter": jnp.asarray([0.2])}):
        opts = _options(table, state, dataclasses.replace(summary, **change), settings)
        assert float(opts.gain[0, 1]) == 0.0
        assert float(opts.gain[0, 0]) == pytest.approx(0.8 * 0.75 * 0.6)
        assert float(opts.time[0, 0, TELESCOPE]) == pytest.approx(
            2 + 0.8 * (10 + 0.75 * 20)
        )
    dark = dataclasses.replace(
        summary, n_det=jnp.zeros(1, I64), p_exist=jnp.asarray([0.25])
    )
    opts = _options(table, state, dark, settings)
    assert float(opts.gain[0, 0]) == pytest.approx(0.25 * 0.8**3 * 0.75 * 0.6)
    assert float(opts.time[0, 0, TELESCOPE]) == pytest.approx(
        2 + 0.25 * 0.8 * 2 + 0.25 * 0.8**2 * 2 + 0.25 * 0.8**3 * 25
    )


def test_packages_use_charged_vectors_repeats_and_later_available_nodes():
    """Fails if costs lose overheads, duplicate the start or use unavailable nodes."""
    table = dataclasses.replace(two_step_table(), repeat=jnp.asarray([2, 1], I64))
    summary, settings = package_inputs(n_slots=2)
    summary = dataclasses.replace(
        summary, available=summary.available.at[0, 2].set(False)
    )
    settings = dataclasses.replace(
        settings,
        channel_cost=settings.channel_cost.at[:, 0].set(
            jnp.asarray([1.0, 4.0, 8.0, 0.0])
        ),
    )
    opts = _options(table, table.init_state(2, 3), summary, settings)
    assert float(opts.gain[0, 1]) == pytest.approx(2 * 0.75**2 * 0.6)
    assert float(opts.time[0, 1, TELESCOPE]) == pytest.approx(
        10 + 2 * (0.75 * 10 + 0.75**2 * 20)
    )
    assert float(opts.time[0, 1, 0]) == pytest.approx(4 + 2 * (0.75 * 4 + 0.75**2 * 8))
    blocked = dataclasses.replace(
        summary, available=summary.available.at[:, 2].set(False)
    )
    assert (
        float(_options(table, table.init_state(2, 3), blocked, settings).gain[0, 1])
        == 0.0
    )
    infinite = dataclasses.replace(
        settings, channel_cost=settings.channel_cost.at[2, TELESCOPE].set(jnp.inf)
    )
    assert (
        float(_options(table, table.init_state(2, 3), summary, infinite).gain[0, 1])
        == 0.0
    )
    partial = dataclasses.replace(table.init_state(2, 3), passes=jnp.ones(2, I64))
    assert float(
        _options(table, partial, summary, settings).gain[0, 1]
    ) == pytest.approx(2 * 0.75 * 0.6)
    modes = ModeTable(MODES.modes, (1.0, 2.0))
    summary = dataclasses.replace(
        summary,
        available=jnp.repeat(summary.available, 2, axis=-1),
        fc_counts=jnp.repeat(summary.fc_counts, 2, axis=2)
        .at[:, :, 5]
        .set(5 + ndtri(0.9)),
        fc_variance=jnp.repeat(summary.fc_variance, 2, axis=2),
        p_det=jnp.repeat(summary.p_det, 2, axis=-1),
    )
    costs = (
        jnp.zeros((7, 3))
        .at[:, TELESCOPE]
        .set(jnp.asarray([2.0, 4.0, 10.0, 50.0, 20.0, 60.0, 0.0]))
    )
    settings = dataclasses.replace(settings, channel_cost=costs)
    variants = _options_with_modes(
        table, table.init_state(2, 3), summary, 0, modes, settings
    )
    assert float(variants.gain[0, 2]) == pytest.approx(2 * 0.75**2 * 0.6)
    assert float(variants.time[0, 2, TELESCOPE]) == pytest.approx(
        10 + 2 * (0.75 * 10 + 0.75**2 * 20)
    )


def test_invalid_tables_are_refused_but_numeric_fields_can_be_traced():
    """Fails if cycles, invalid references or unknown claims survive construction."""
    table = two_step_table()
    for change in (
        {"on_pass": [1, 0]},
        {"on_fail": [0, -1]},
        {"on_pass": [1, 9]},
        {"requirement_snr": [0.0, 5.0]},
        {"requirement_snr": [jnp.inf, 5.0]},
        {"target_claim": "missing"},
        {"step_labels": [0, 8]},
        {"repeat": [0, 1]},
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(table, **change)
    result = eqx.filter_jit(
        lambda requirement: (
            dataclasses.replace(table, requirement_snr=requirement).requirement_snr
        )
    )(jnp.asarray([6.0, 7.0]))
    np.testing.assert_array_equal(result, [6.0, 7.0])
    _, package_settings = package_inputs()
    scatter = eqx.filter_jit(
        lambda value: (
            dataclasses.replace(
                package_settings, candidate_scatter_lod=value
            ).candidate_scatter_lod
        )
    )(jnp.asarray(2.0))
    assert float(scatter) == 2.0


def test_an_imaging_step_costs_the_row_that_completes_its_fold():
    """Fails if a completing image is charged again as a gate or table step."""
    table = dataclasses.replace(one_step_table(), mode_names=("image",))
    summary, settings = package_inputs(p=(1.0, 1.0))
    summary = dataclasses.replace(
        summary,
        n_det=jnp.zeros(1, I64),
        p_det=jnp.ones_like(summary.p_det),
        fc_counts=jnp.full_like(summary.fc_counts, 50.0),
        fc_variance=jnp.ones_like(summary.fc_variance),
    )
    state = table.init_state(1, 3)
    folded = _update(table, state, spectral_row(mode=0, counts=50.0))
    assert int(folded.node[0]) == 1
    opts = _options(table, state, summary, settings)
    assert float(opts.gain[0, 0]) == 1.0
    assert float(opts.time[0, 0, TELESCOPE]) == 2.0
    selection = select(
        opts.gain[None], opts.time[None], jnp.asarray([0.0, 0.0, 2.0]), 48
    )
    assert int(selection.choice[0]) == 0


def test_imaging_progress_requires_a_delivered_detection():
    """Fails if uncensored counts credit null images or their spectral continuations."""
    summary, settings = package_inputs(p=(0.6, 1.0))
    p_det = 0.5200394249187209  # Phi((50 - 7 * sqrt(50)) / sqrt(100)).
    summary = dataclasses.replace(
        summary,
        n_det=jnp.full(1, 2, I64),
        p_det=jnp.full_like(summary.p_det, p_det),
        fc_counts=summary.fc_counts.at[:, :, 0].set(50.0),
        fc_variance=summary.fc_variance.at[:, :, 0].set(100.0),
    )
    terminal = dataclasses.replace(
        one_step_table(), mode_names=("image",), requirement_snr=[1.0]
    )
    continuation = dataclasses.replace(
        two_step_table(),
        mode_names=("image", "a"),
        requirement_snr=[1.0, 5.0],
        label_values=[0.0, 0.25, 1.0],
    )
    image_pass = p_det * float(ndtr(4.0))
    for table, gain, time_d in (
        (terminal, image_pass, 2.0),
        (continuation, image_pass * (0.25 + 0.6 * 0.75), 2.0 + image_pass * 10.0),
    ):
        state = table.init_state(1, 3)
        detected = _update(
            table, state, spectral_row(mode=0, counts=50.0, variance=100.0)
        )
        assert int(detected.node[0]) == 1
        null = _update(table, state, spectral_row(mode=0, counts=0.0, variance=0.0))
        assert_trees_equal(null, state)
        opts = _options(table, state, summary, settings)
        assert float(opts.gain[0, 0]) == pytest.approx(gain)
        assert float(opts.gain[0, 0]) <= p_det
        assert float(opts.time[0, 0, TELESCOPE]) == pytest.approx(time_d)
        no_detection = dataclasses.replace(summary, p_det=jnp.zeros_like(summary.p_det))
        opts = _options(table, state, no_detection, settings)
        assert float(opts.gain[0, 0]) == 0.0
        assert float(opts.time[0, 0, TELESCOPE]) == 2.0


def test_row_allowances_keep_only_reachable_intermediate_credit():
    """Fails if continuation credit survives its row cap or loses a reachable label."""
    table = dataclasses.replace(two_step_table(), label_values=[0.0, 0.5, 1.0])
    summary, settings = package_inputs(p=(1.0, 1.0))
    summary = dataclasses.replace(
        summary, fc_counts=jnp.full_like(summary.fc_counts, 50.0)
    )
    limited = dataclasses.replace(settings, remaining_rows=jnp.asarray([3, 1, 1], I64))
    opts = _options(table, table.init_state(1, 3), summary, limited)
    assert float(opts.gain[0, 1]) == 0.5
    assert float(opts.time[0, 1, TELESCOPE]) == 10.0
    dark = dataclasses.replace(summary, n_det=jnp.zeros(1, I64))
    limited = dataclasses.replace(settings, remaining_rows=jnp.asarray([2, 5, 5], I64))
    opts = _options(table, table.init_state(1, 3), dark, limited)
    assert float(opts.gain[0, 0]) == 0.0


def test_tracks_share_the_star_packages_remaining_rows_in_order():
    """Fails if each track reuses a spectrum allowance or consumes the shared start."""
    table = dataclasses.replace(two_step_table(), on_pass=[4, 4], step_labels=[-1, -1])
    summary, settings = package_inputs(p=(1.0, 1.0), n_slots=2)
    summary = dataclasses.replace(
        summary,
        n_det=jnp.full(2, 2, I64),
        p_det=jnp.ones_like(summary.p_det),
        fc_counts=jnp.full_like(summary.fc_counts, 50.0),
    )
    state = dataclasses.replace(table.init_state(2, 3), node=jnp.asarray([0, 1], I64))
    limited = dataclasses.replace(settings, remaining_rows=jnp.asarray([1, 1, 1], I64))
    opts = _options(table, state, summary, limited)
    assert float(opts.gain[0, 0]) == 1.0
    assert float(opts.time[0, 0, TELESCOPE]) == 12.0

    image = dataclasses.replace(
        spectral_row(mode=0, counts=50.0),
        assignment=jnp.asarray([0, 1], I64),
    )
    after_image = _update(table, state, image)
    after_spectrum = _update(table, after_image, spectral_row(mode=1, counts=50.0))
    np.testing.assert_array_equal(after_spectrum.node, [4, 1])
    assert float(table.value(after_spectrum, jnp.empty((2, 0)), 0)[0].sum()) == 1.0
    reversed_state = dataclasses.replace(state, node=state.node[::-1])
    reversed_opts = _options(table, reversed_state, summary, limited)
    assert float(reversed_opts.gain[0, 0]) == 1.0
    assert float(reversed_opts.time[0, 0, TELESCOPE]) == 22.0
    enough = dataclasses.replace(settings, remaining_rows=jnp.asarray([1, 2, 2], I64))
    opts = _options(table, state, summary, enough)
    assert float(opts.gain[0, 0]) == 2.0
    assert float(opts.time[0, 0, TELESCOPE]) == 32.0
    ready = dataclasses.replace(summary, n_det=jnp.full(2, 3, I64))
    opts = _options(table, table.init_state(2, 3), ready, limited)
    assert float(opts.gain[0, 1]) == 2.0
    assert float(opts.time[0, 1, TELESCOPE]) == 10.0
    intermediate = dataclasses.replace(
        two_step_table(), mode_names=("image", "a"), label_values=[0.0, 0.25, 1.0]
    )
    opts = _options(intermediate, reversed_state, summary, limited)
    # The first track gains 0.75; the second retains only its 0.25 imaging credit.
    assert float(opts.gain[0, 0]) == 1.0
    assert float(opts.time[0, 0, TELESCOPE]) == 12.0


def test_equivalent_continuations_reuse_work_until_row_constraints_bind(monkeypatch):
    """Fails if repeated continuations run again or reuse crosses a row limit."""
    calls = []
    dense_searches = []
    argmax = jnp.argmax

    def counted_argmax(values, axis=None, **kwargs):
        if values.ndim == 2 and axis is None:
            jax.debug.callback(
                lambda _: dense_searches.append(1), values.sum(), ordered=True
            )
        return argmax(values, axis=axis, **kwargs)

    monkeypatch.setattr(jnp, "argmax", counted_argmax)
    probability = transition.pass_probability

    def counted(*args):
        jax.debug.callback(lambda _: calls.append(1), args[0].sum(), ordered=True)
        return probability(*args)

    monkeypatch.setattr(transition, "pass_probability", counted)
    table = one_step_table()
    summary, settings = package_inputs(n_slots=6)
    summary = dataclasses.replace(
        summary,
        n_det=jnp.zeros_like(summary.n_det),
        p_det=jnp.ones_like(summary.p_det),
        fc_counts=jnp.full_like(summary.fc_counts, 1000.0),
    )
    state = table.init_state(6, 3)
    for rows, gain, time_d in ((24, 6.0, 86.0), (3, 1.0, 16.0), (2, 0.0, 4.0)):
        calls.clear()
        dense_searches.clear()
        limited = dataclasses.replace(
            settings, remaining_rows=jnp.asarray([rows, 12, 12], I64)
        )
        options = _options(table, state, summary, limited)
        jax.effects_barrier()
        assert float(options.gain[0, 0]) == gain
        assert float(options.time[0, 0, TELESCOPE]) == time_d
        if rows == 24:
            assert len(calls) == 6
            assert len(dense_searches) == 6

    future = dataclasses.replace(summary, p_det=summary.p_det.at[:, 0, :].set(0.25))
    enough = dataclasses.replace(
        settings, remaining_rows=jnp.asarray([24, 12, 12], I64)
    )
    for allowed, gain, time_d in ((True, 1.5, 23.0), (False, 0.09375, 6.6875)):
        supplied = dataclasses.replace(
            future, available=future.available.at[1].set(allowed)
        )
        options = _options(table, state, supplied, enough)
        assert float(options.gain[0, 0]) == gain
        assert float(options.time[0, 0, TELESCOPE]) == time_d


def test_no_credit_star_preserves_costs_and_terminal_search_is_skipped(monkeypatch):
    """Fails if zero target credit erases active costs or terminal tracks search."""
    calls = []
    probability = transition.pass_probability

    def counted(*args):
        jax.debug.callback(lambda _: calls.append(1), args[0].sum(), ordered=True)
        return probability(*args)

    monkeypatch.setattr(transition, "pass_probability", counted)
    table = dataclasses.replace(
        one_step_table(),
        claims=(EarthCandidateClaim(),),
        target_claim="earth_candidate",
    )
    summary, settings = package_inputs(n_slots=2)
    summary = dataclasses.replace(summary, p_claim=jnp.zeros((2, 1)))
    state = table.init_state(2, 3)
    options = _options(table, state, summary, settings)
    expected = _reference_options(table, state, summary, 0, MODES, settings)
    for a, b in zip(jax.tree.leaves(options), jax.tree.leaves(expected), strict=True):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    jax.effects_barrier()
    calls.clear()
    terminal = dataclasses.replace(state, node=jnp.full_like(state.node, table.n_steps))
    _options(table, terminal, summary, settings).gain.block_until_ready()
    jax.effects_barrier()
    assert not calls
    np.testing.assert_array_equal(options.gain, 0.0)
    np.testing.assert_array_equal(options.track_gain, 0.0)


_reference_options = eqx.filter_jit(reference_options)


@pytest.mark.medium(
    reason="two independent multi-step package programs and fixed-shape cases"
)
def test_optimized_options_match_reference_bytes():
    """Fails if row bounds, zero-gain pruning or track reuse alter any option byte."""
    table = dataclasses.replace(
        two_step_table(),
        repeat=jnp.asarray([2, 3], I64),
        margin=jnp.asarray([0.25, 0.75]),
        label_values=jnp.asarray([0.0, 0.4, 1.0]),
        claims=(EarthCandidateClaim(),),
        target_claim="earth_candidate",
    )
    summary, settings = package_inputs(n_slots=3, n_grid=3)
    state = table.init_state(3, 3)
    rng = np.random.default_rng(0)
    for case in range(14):
        # Both sides of repeat + missing-image bound, including equal/zero tracks.
        rows = (9, 9, 9, 7, 1, 0, 8, 100, 100, 100, 100, 100, 100, 100)[case]
        current = dataclasses.replace(
            state,
            node=jnp.asarray([0, 0, case % 2], I64),
            passes=jnp.asarray([1, 1, case % 3], I64),
            sum_counts=jnp.full((3, 3), 0.5 * (case % 2)),
            sum_variance=jnp.full((3, 3), 0.25 * (case % 2)),
        )
        target = 0.0 if case == 6 else 0.75
        if case >= 8:
            current = dataclasses.replace(current, node=jnp.full(3, table.n_steps, I64))
            target = (0.75, -0.0, np.nan, np.inf, 0.75, 0.0)[case - 8]
        forecast = dataclasses.replace(
            summary,
            n_det=jnp.full(3, case % 4, I64),
            p_claim=jnp.full((3, 1), target),
            p_det=jnp.asarray(rng.uniform(0.1, 0.9, (3, 3, 1))),
            fc_counts=jnp.asarray(rng.uniform(3, 9, (3, 3, 3, 3))),
            available=jnp.asarray(rng.uniform(size=(3, 3)) > 0.2),
        )
        if case >= 12:
            forecast = dataclasses.replace(
                forecast,
                n_det=jnp.zeros(3, I64),
                p_exist=jnp.full(3, np.nan if case == 12 else np.inf),
            )
        # Adjacent identical tracks exercise reuse; zero forecasts exercise pruning.
        for name in ("p_det", "fc_counts"):
            value = getattr(forecast, name)
            forecast = dataclasses.replace(
                forecast, **{name: value.at[1].set(value[0])}
            )
        if case == 4:
            forecast = dataclasses.replace(
                forecast, fc_counts=jnp.zeros_like(forecast.fc_counts)
            )
        costs = settings.channel_cost
        if case == 8:
            costs = costs.at[:, 0].set(-0.0)
        limited = dataclasses.replace(
            settings, remaining_rows=jnp.full(3, rows, I64), channel_cost=costs
        )
        actual = _options_with_modes(table, current, forecast, 0, MODES, limited)
        expected = _reference_options(table, current, forecast, 0, MODES, limited)
        for a, b in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), case
