"""Paired contrasts of the arms against closed forms, on hand-built tables."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.adapters.photomancy import N_GRID, DecisionSummary
from spaceodyssey.core.enums import RefitStatus
from spaceodyssey.studies.bracket import ArmsResult, OutcomeTable, claimed_slots

A = np.array([3.0, 5.0, 4.0, 8.0, 6.0])
B = np.array([2.0, 5.0, 1.0, 6.0, 6.0])


def _result():
    def table(x):
        return OutcomeTable(slot={}, star={}, world={"confirmed": x})

    return ArmsResult(table={"a": table(A), "b": table(B)})


def test_gap_matches_the_paired_closed_form():
    """Mean, standard error, correlation and n equal the textbook values."""
    mean, se, corr, n = _result().gap("a", "b", "confirmed")
    d = A - B
    assert n == 5
    assert mean == pytest.approx(d.sum() / 5, abs=1e-12)
    assert se == pytest.approx(np.sqrt(((d - d.mean()) ** 2).sum() / 4 / 5), abs=1e-12)
    assert corr == pytest.approx(np.corrcoef(A, B)[0, 1], abs=1e-12)


def test_the_log_ratio_has_a_known_answer_for_a_pure_rescaling():
    """Arm a is twice arm b world by world: value ln 2 and no paired error."""
    result = ArmsResult(
        table={
            "a": OutcomeTable(slot={}, star={}, world={"confirmed": 2.0 * B}),
            "b": OutcomeTable(slot={}, star={}, world={"confirmed": B}),
        }
    )
    value, se = result.log_ratio("a", "b", "confirmed")
    assert value == pytest.approx(np.log(2.0), abs=1e-12)
    assert se == pytest.approx(0.0, abs=1e-12)


def test_the_log_ratio_error_matches_a_bootstrap_of_the_worlds():
    """The delta-method error matches a seeded paired bootstrap within 10 percent."""
    rng = np.random.default_rng(5)
    common = rng.poisson(6.0, 300).astype(float)
    xa = common + rng.poisson(3.0, 300)
    xb = common + rng.poisson(1.0, 300)
    result = ArmsResult(
        table={
            "a": OutcomeTable(slot={}, star={}, world={"confirmed": xa}),
            "b": OutcomeTable(slot={}, star={}, world={"confirmed": xb}),
        }
    )
    _, se = result.log_ratio("a", "b", "confirmed")
    picks = rng.integers(0, 300, (4000, 300))
    boot = np.log(xa[picks].mean(axis=1) / xb[picks].mean(axis=1))
    assert se == pytest.approx(boot.std(ddof=1), rel=0.10)


def test_the_log_ratio_of_a_zero_mean_is_undefined():
    """An endpoint with a zero mean in either arm gives NaN, not inf."""
    result = ArmsResult(
        table={
            "a": OutcomeTable(slot={}, star={}, world={"false_claims": np.zeros(4)}),
            "b": OutcomeTable(slot={}, star={}, world={"false_claims": B[:4]}),
        }
    )
    value, se = result.log_ratio("a", "b", "false_claims")
    assert np.isnan(value) and np.isnan(se)


def test_a_claim_needs_detections_class_status_and_a_tight_position():
    """Each condition alone withholds the claim; all together grant it."""
    lod = 0.04

    def belief(**override):
        base = DecisionSummary.zeros(1, 1, N_GRID, 1, 1)
        leaves = {
            "n_det": jnp.asarray([[3]]),
            "p_class": jnp.asarray([[0.9]]),
            "status": jnp.asarray([[int(RefitStatus.OK)]]),
            "scatter": jnp.asarray([[0.5 * lod]]),
        }
        leaves.update({k: jnp.asarray(v) for k, v in override.items()})
        return dataclasses.replace(base, **leaves)

    assert bool(claimed_slots(belief(), lod)[0, 0])
    withheld = {
        "two detections": {"n_det": [[2]]},
        "class probability under 0.5": {"p_class": [[0.49]]},
        "degraded refit": {"status": [[int(RefitStatus.DEGRADED)]]},
        "no refit yet": {"status": [[int(RefitStatus.NO_SUPPORT)]]},
        "wide scatter": {"scatter": [[1.5 * lod]]},
    }
    for reason, change in withheld.items():
        assert not bool(claimed_slots(belief(**change), lod)[0, 0]), reason


def test_an_arm_paired_with_itself_has_zero_gap_and_zero_error():
    """A self pair is exactly zero with unit correlation."""
    mean, se, corr, _ = _result().gap("a", "a", "confirmed")
    assert (mean, se) == (0.0, 0.0)
    assert corr == pytest.approx(1.0, abs=1e-12)
    assert _result().log_ratio("a", "a", "confirmed") == (0.0, 0.0)
