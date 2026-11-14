"""Cross-criterion credit on completed rows, including coverage and reset order."""

import dataclasses
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core.carry import init_carry
from spaceodyssey.core.enums import I64
from spaceodyssey.core.records import pack, unpack, write_log
from spaceodyssey.studies.scoring import (
    attribution,
    cross_yield,
    refold,
    row_log,
    verdict,
)
from tests.studies.bracket_helpers import needs_mode
from tests.studies.test_scoring import CRITERION, MODES, hand_scored_log


@pytest.fixture(scope="module")
def scored_case(bracket_study):
    return hand_scored_log(bracket_study)


def flight(scored_case, criterion, exposures):
    """Replay explicit (mode, counts_e, variance_e2) exposures after the catalog."""
    result, study = scored_case
    study = dataclasses.replace(
        study, observer=dataclasses.replace(study.observer, criterion=criterion)
    )
    carry = init_carry(study, None)
    catalog = jax.tree.map(lambda x: x[0, 0], result.carry.log)
    carry, _ = write_log(carry, catalog, study.caps)
    record = jax.tree.map(lambda x: x[0, 1], result.carry.log)
    template = study.channel_model.measurement_template
    row = unpack(template, record.payload)
    for tick, (mode, counts_e, variance_e2) in enumerate(exposures, 2):
        band = MODES.band_index(MODES.modes[mode].name)
        counts = jnp.zeros_like(row.counts).at[:4, band].set(counts_e)
        variance = jnp.zeros_like(row.variance).at[:4, band].set(variance_e2)
        measured = dataclasses.replace(
            row,
            mode=jnp.asarray(mode, I64),
            detected=jnp.where(
                MODES.modes[mode].kind == "imaging",
                (row.assignment >= 0).astype(row.detected.dtype),
                jnp.zeros_like(row.detected),
            ),
            counts=counts,
            variance=variance,
            true_counts=counts,
            true_variance=variance,
        )
        encoded = pack(measured)
        rec = dataclasses.replace(
            record,
            end_tick=jnp.asarray(tick, I64),
            payload=jnp.pad(encoded, (0, study.caps.payload_width - encoded.size)),
        )
        carry, _ = write_log(carry, rec, study.caps)
        carry = dataclasses.replace(
            carry, belief=study.observer.fold(carry.belief, rec, study.caps)
        )
    carry = dataclasses.replace(carry, tick=jnp.asarray(len(exposures) + 1, I64))
    return SimpleNamespace(carry=jax.tree.map(lambda x: x[None], carry)), study


@pytest.fixture(scope="module")
def scored_case_strict(scored_case):
    strict = eqx.tree_at(
        lambda t: t.requirement_snr, CRITERION, 2.0 * CRITERION.requirement_snr
    )
    return flight(scored_case, strict, [(1, 12.0, 1.0)])


def test_counting_a_flight_under_its_own_criterion_is_its_verdict(scored_case):
    """Fails if the cross yield refolds differently from the verdict."""
    result, study = scored_case
    np.testing.assert_array_equal(
        cross_yield(result, study, study.observer.criterion),
        verdict(result, study).yield_,
    )


def test_a_stricter_requirement_counts_no_more_than_the_flown_one(scored_case):
    """Fails if a stricter single-step requirement uses measured or unsupported data."""
    result, study = scored_case
    payload = result.carry.log.payload
    for index in (1, 2):
        row = unpack(study.channel_model.measurement_template, payload[0, index])
        encoded = pack(dataclasses.replace(row, counts=3.0 * row.counts))
        payload = payload.at[0, index, : encoded.size].set(encoded)
    result = SimpleNamespace(
        carry=dataclasses.replace(
            result.carry, log=dataclasses.replace(result.carry.log, payload=payload)
        )
    )
    strict = eqx.tree_at(
        lambda t: t.requirement_snr,
        study.observer.criterion,
        2.0 * study.observer.criterion.requirement_snr,
    )
    actual = cross_yield(result, study, strict)
    assert bool(np.all(actual <= verdict(result, study).yield_))
    np.testing.assert_array_equal(actual, [0.0])


def test_a_criterion_needing_an_unobserved_mode_has_no_cross_yield(scored_case):
    """Fails if a definition is counted on modes the flight never observed."""
    result, study = scored_case
    assert np.isnan(cross_yield(result, study, needs_mode("ifs_760", CRITERION))).all()
    assert np.isnan(
        cross_yield(result, study, needs_mode(MODES.modes[0].name, CRITERION))
    ).all()


def test_the_two_parts_sum_to_the_total(scored_case, scored_case_strict):
    """Fails if the decomposition drops a term or uses the wrong flight."""
    a = attribution(scored_case, scored_case_strict)
    np.testing.assert_allclose(
        a.definition_alone + a.planning_response, a.total, atol=1e-12
    )
    np.testing.assert_allclose(a.definition_alone, [-1.4], atol=1e-12)
    np.testing.assert_allclose(a.planning_response, [2.1], atol=1e-12)
    np.testing.assert_allclose(a.total, [0.7], atol=1e-12)


def test_empty_flights_keep_native_zero_but_foreign_yield_is_undefined(scored_case):
    """Fails if missing rows mask a native verdict or define foreign coverage."""
    result, study = flight(scored_case, CRITERION, [])
    native = dataclasses.replace(CRITERION)
    np.testing.assert_array_equal(cross_yield(result, study, native), [0.0])
    np.testing.assert_array_equal(
        cross_yield(result, study, native), verdict(result, study).yield_
    )
    strict = eqx.tree_at(
        lambda t: t.requirement_snr, native, 2.0 * native.requirement_snr
    )
    assert np.isnan(cross_yield(result, study, strict)).all()
    result.carry = dataclasses.replace(
        result.carry,
        log=dataclasses.replace(
            result.carry.log, valid=jnp.zeros_like(result.carry.log.valid)
        ),
    )
    np.testing.assert_array_equal(cross_yield(result, study, native), [0.0])


def test_coverage_is_per_world_and_accepts_an_observed_alternative(scored_case):
    """Fails if one world supplies another's modes or every alternative is required."""
    first, study = flight(scored_case, CRITERION, [(1, 6.0, 1.0)])
    second, _ = flight(scored_case, CRITERION, [(0, 6.0, 1.0)])
    empty, _ = flight(scored_case, CRITERION, [])
    result = SimpleNamespace(
        carry=jax.tree.map(
            lambda *xs: jnp.concatenate(xs), first.carry, second.carry, empty.carry
        )
    )
    alternative = dataclasses.replace(
        CRITERION,
        mode_names=tuple(m.name for m in MODES.modes),
        step_modes=[[True, True]],
    )
    np.testing.assert_allclose(
        cross_yield(result, study, alternative), [2.1, 2.1, np.nan]
    )
    absent_alternative = dataclasses.replace(
        alternative, mode_names=(MODES.modes[1].name, "ifs_760")
    )
    np.testing.assert_allclose(
        cross_yield(result, study, absent_alternative), [2.1, np.nan, np.nan]
    )
    two_step = dataclasses.replace(
        alternative,
        step_names=("first", "second"),
        step_modes=[[True, False], [False, True]],
        requirement_snr=[1.0, 1.0],
        margin=[0.0, 0.0],
        repeat=[1, 1],
        on_pass=[1, 2],
        on_fail=[-1, -1],
        step_labels=[-1, -1],
    )
    assert np.isnan(cross_yield(result, study, two_step)).all()


def test_multistep_cross_yield_follows_chronological_resets(scored_case):
    """Fails if refolding combines rows from before a step's reset."""
    loose = dataclasses.replace(
        CRITERION,
        mode_names=tuple(m.name for m in MODES.modes),
        step_names=("first", "second"),
        step_modes=[[True, False], [False, True]],
        requirement_snr=[1.0, 1.0],
        margin=[0.0, 0.0],
        repeat=[1, 1],
        on_pass=[1, 2],
        on_fail=[-1, -1],
        step_labels=[-1, -1],
    )
    result, study = flight(
        scored_case,
        loose,
        [(0, 1.5, 1.0), (1, 0.0, 100.0), (0, 1.5, 1.0), (1, 3.0, 1.0)],
    )
    strict = eqx.tree_at(
        lambda t: t.requirement_snr, loose, 2.0 * loose.requirement_snr
    )
    rows = row_log(result, study)
    for criterion, expected_node, expected_yield in ((loose, 1, 0.0), (strict, 2, 2.1)):
        state = refold(criterion, rows, 0, 0, MODES, columns="true")
        np.testing.assert_array_equal(state.node[:4], expected_node)
        np.testing.assert_allclose(
            cross_yield(result, study, criterion), [expected_yield]
        )


def test_missing_cross_coverage_propagates_only_to_attribution_parts(scored_case):
    """Fails if undefined cross data erase the defined native yield contrast."""
    counted = flight(
        scored_case, needs_mode(MODES.modes[0].name, CRITERION), [(0, 6.0, 1.0)]
    )
    a = attribution(scored_case, counted)
    np.testing.assert_allclose(a.total, [0.7], atol=1e-12)
    assert np.isnan(a.definition_alone).all()
    assert np.isnan(a.planning_response).all()
