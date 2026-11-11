"""Paired contrasts of the arms against closed forms, on hand-built tables."""

import dataclasses
import functools
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.adapters.photomancy import (
    N_GRID,
    SLOT_DEGRADED,
    SLOT_FAILED,
    SLOT_NEVER_FIT,
    SLOT_NO_SUPPORT,
    SLOT_OK,
    DecisionSummary,
)
from spaceodyssey.core.enums import RefitStatus
from spaceodyssey.studies.bracket import (
    ArmsResult,
    OutcomeTable,
    SummaryObserver,
    candidate_slots,
    confirmed_slots,
)

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


def test_a_candidate_needs_detections_class_status_and_a_tight_position():
    """Each condition alone withholds the nomination; all together grant it."""
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

    assert bool(candidate_slots(belief(), lod)[0, 0])
    withheld = {
        "two detections": {"n_det": [[2]]},
        "class probability under the bar": {"p_class": [[0.19]]},
        "degraded refit": {"status": [[int(RefitStatus.DEGRADED)]]},
        "no refit yet": {"status": [[int(RefitStatus.NO_SUPPORT)]]},
        "failed refit": {"status": [[int(RefitStatus.FAILED)]]},
        "never fit": {"status": [[SLOT_NEVER_FIT]]},
        "wide scatter": {"scatter": [[1.5 * lod]]},
    }
    for reason, change in withheld.items():
        assert not bool(candidate_slots(belief(**change), lod)[0, 0]), reason
    # host arrays: the rule is plain comparisons, so no program is compiled
    low = jax.tree_util.tree_map(np.asarray, belief(p_class=[[0.15]]))
    assert not bool(candidate_slots(low, lod)[0, 0])
    assert bool(candidate_slots(low, lod, 0.1)[0, 0]), "the bar is a parameter"


def test_only_a_revealed_class_one_slot_is_confirmed():
    """Confirmation is the revealed class, not the class probability."""
    # host arrays and only the leaves the rule reads: no program is compiled
    belief = SimpleNamespace(p_class=np.ones((1, 3)), revealed=np.asarray([[0, 1, 2]]))
    assert np.array_equal(np.asarray(confirmed_slots(belief))[0], [0, 0, 1])


def test_an_arm_paired_with_itself_has_zero_gap_and_zero_error():
    """A self pair is exactly zero with unit correlation."""
    mean, se, corr, _ = _result().gap("a", "a", "confirmed")
    assert (mean, se) == (0.0, 0.0)
    assert corr == pytest.approx(1.0, abs=1e-12)
    assert _result().log_ratio("a", "a", "confirmed") == (0.0, 0.0)


@functools.cache
def _install():
    """One compiled `(install, claims)` of a two-star, three-slot summary."""
    zeros = DecisionSummary.zeros(2, 3, N_GRID, 1, 1, n_an=1)
    observer = SummaryObserver(prior=zeros)

    def run(old, new):
        out = observer.install(old, new, 0)
        return out, candidate_slots(out, 1.0)

    return eqx.filter_jit(run)


def _summary(n_det, p_class, status, p_new=0.0):
    """A two-star, three-slot summary: star 0 holds the given slot leaves."""
    base = DecisionSummary.zeros(2, 3, N_GRID, 1, 1, n_an=1)
    host = jax.tree_util.tree_map(np.array, base)
    host.n_det[0] = n_det
    host.p_class[0] = p_class
    host.p_det[0] = np.asarray(p_class)[:, None, None]
    host.status[0] = status
    host.p_new[0] = p_new
    host.n_obs[0] = 4
    return jax.tree_util.tree_map(jnp.asarray, host)


def test_a_failed_slot_keeps_its_summary_and_the_fitted_slots_install():
    """A partial refit installs the slots that fit; a failed one keeps its leaves.

    Slot 0 (detected, fit), slot 1 (undetected, no support) and slot 2 (detected,
    degraded): slots 0 and 2 take the new leaves, slot 1 keeps the old ones with the
    new code, `p_new` keeps its old value because an undetected slot failed, the star
    counters are the new ones and the other star is untouched.
    """
    old = _summary([3, 0, 1], [0.7, 0.7, 0.7], [SLOT_OK, SLOT_OK, SLOT_OK], 0.4)
    new = _summary(
        [4, 0, 2],
        [0.2, 0.2, 0.2],
        [SLOT_OK, SLOT_NO_SUPPORT, SLOT_DEGRADED],
        0.9,
    )
    out, claim = _install()(old, new)
    assert np.allclose(out.p_class[0], [0.2, 0.7, 0.2])
    assert np.allclose(out.p_det[0, :, 0, 0], [0.2, 0.7, 0.2])
    assert np.array_equal(out.n_det[0], [4, 0, 2])
    assert np.array_equal(out.status[0], [SLOT_OK, SLOT_NO_SUPPORT, SLOT_DEGRADED])
    assert float(out.p_new[0, 0, 0]) == 0.4
    assert int(out.n_obs[0]) == 4
    assert np.array_equal(np.asarray(out.p_class[1]), np.asarray(old.p_class[1]))
    claim = np.asarray(claim)
    assert not claim[0, 1] and not claim[0, 2]


def test_a_failed_detected_slot_keeps_its_leaves_but_the_first_tier_installs():
    """When only a detected slot fails, `p_new` is the new one and the slot is kept."""
    old = _summary([3, 0, 0], [0.7, 0.1, 0.1], [SLOT_OK, SLOT_OK, SLOT_OK], 0.4)
    new = _summary([5, 0, 0], [0.3, 0.2, 0.2], [SLOT_FAILED, SLOT_OK, SLOT_OK], 0.9)
    out, _ = _install()(old, new)
    assert np.allclose(out.p_class[0], [0.7, 0.2, 0.2])
    assert np.array_equal(out.n_det[0], [3, 0, 0])
    assert int(out.status[0, 0]) == SLOT_FAILED
    assert float(out.p_new[0, 0, 0]) == 0.9
