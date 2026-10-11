"""Replicate uncertainty preserves correlations across planets and plans."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.engines.analytic import (
    CHANNELS,
    certified_yield,
    paired_difference,
    replicate_yields,
)
from spaceodyssey.metrics import CertifiedYield
from tests.test_plan import make_outcome


def outcome(pos=None, *, kind="batch", key="stream", **overrides):
    pos = jnp.asarray([[0.25], [0.5], [0.75]]) if pos is None else jnp.asarray(pos)
    reps = {
        "p_pos": pos,
        "p_wrong": pos / 4,
        "p_neg": 1 - pos,
        "p_neg_wrong": (1 - pos) / 4,
    }
    fields = {name: rep.mean(axis=0) for name, rep in reps.items()}
    fields.update({name + "_rep": rep for name, rep in reps.items()})
    fields.update(
        {
            name + "_se": jnp.std(rep, axis=0, ddof=1)
            / (np.sqrt(len(pos)) if kind == "batch" else 1)
            for name, rep in reps.items()
        }
    )
    fields.update(replicate_kind=kind, replicate_key=key)
    fields.update(overrides)
    return make_outcome(n_planets=pos.shape[1], **fields)


def report(value, **kwargs):
    return certified_yield(
        value, jnp.ones(value.p_pos.shape), certificate="test", **kwargs
    )


@pytest.mark.parametrize("kind", ["batch", "bootstrap"])
def test_shared_planet_error_scales_coherently(kind):
    single = report(outcome(kind=kind))
    double = report(
        outcome(jnp.tile(jnp.array([[0.25], [0.5], [0.75]]), (1, 2)), kind=kind)
    )
    for name in CHANNELS:
        assert double.standard_errors[name] == pytest.approx(
            2 * single.standard_errors[name]
        )
        assert getattr(double, name + "_se_independent_lower_bound") == pytest.approx(
            np.sqrt(2) * getattr(single, name + "_se_independent_lower_bound")
        )
        assert double.design_effects[name] == pytest.approx(
            2 * single.design_effects[name]
        )


@pytest.mark.parametrize("kind", ["batch", "bootstrap"])
def test_hand_intervals_and_ratio(kind):
    r = report(outcome(kind=kind))
    assert r.interval_kind == "replicate-" + kind
    for name, scale in zip(CHANNELS, [1, 0.75, 0.25, 1, 0.25], strict=True):
        point = getattr(r, name)
        if kind == "batch":
            # sd([1/4, 1/2, 3/4]) = 1/4; t_2(.975) = 4.3026527299.
            se = scale * 0.25 / np.sqrt(3)
            interval = (point - 4.3026527299 * se, point + 4.3026527299 * se)
        else:
            se = scale * 0.25
            interval = (scale * 0.2625, scale * 0.7375)
        assert r.standard_errors[name] == pytest.approx(se)
        assert r.intervals[name] == pytest.approx(interval)
    assert r.wrong_issued_ratio == 0.25
    assert r.wrong_issued_ratio_rep == (0.25, 0.25, 0.25)
    assert r.wrong_issued_ratio_interval == (0.25, 0.25)


@pytest.mark.parametrize("kind", ["batch", "bootstrap"])
def test_identical_paired_difference_exact_zero(kind):
    a = outcome(kind=kind)
    difference = paired_difference(a, a, jnp.ones(1))
    for name in CHANNELS:
        assert difference.yields[name] == 0
        assert difference.replicate_yields[name] == (0, 0, 0)
        assert difference.standard_errors[name] == 0
        assert difference.intervals[name] == (0, 0)


def test_coupled_difference_smaller_than_quadrature():
    a = report(outcome())
    b = report(outcome([[0.375], [0.625], [0.875]]))
    difference = paired_difference(a, b)
    assert difference.yields["y_issued"] == -0.125
    assert difference.standard_errors["y_issued"] == 0
    assert np.hypot(a.standard_errors["y_issued"], b.standard_errors["y_issued"]) > 0


@pytest.mark.parametrize(
    "other",
    [
        outcome(key="other"),
        outcome(kind="bootstrap"),
        outcome([[0.25], [0.5], [0.75], [0.5]]),
    ],
)
def test_pairing_metadata_rejected(other):
    with pytest.raises(ValueError, match="matching replicate"):
        paired_difference(report(outcome()), report(other))


@pytest.mark.parametrize(
    "overrides",
    [
        {"p_pos_rep": None},
        {"replicate_kind": "unknown"},
        {"replicate_key": None},
        {"replicate_key": ""},
        {"p_wrong_rep": jnp.zeros((3, 2))},
        {"p_pos_rep": jnp.ones((1, 1))},
        {"p_pos_rep": jnp.ones((3,))},
        {"p_pos_rep": jnp.full((3, 1), jnp.nan)},
        {"p_pos_rep": jnp.full((3, 1), -0.1)},
        {"p_pos_rep": jnp.full((3, 1), 1.1)},
        {"p_pos": jnp.array([0.6])},
    ],
)
def test_replicate_validation(overrides):
    with pytest.raises(ValueError):
        outcome(**overrides)


@pytest.mark.parametrize(
    "metadata", [{"replicate_key": "stream"}, {"replicate_kind": "batch"}]
)
def test_metadata_without_replicates_rejected(metadata):
    with pytest.raises(ValueError, match="metadata"):
        make_outcome(**metadata)


def test_bootstrap_main_need_not_equal_mean():
    assert report(outcome(kind="bootstrap", p_pos=[0.625])).y_issued == 0.625


def test_replicate_jit_and_grad():
    value = outcome()
    np.testing.assert_allclose(
        jax.jit(replicate_yields)(value, jnp.ones(1))[1], [0.1875, 0.375, 0.5625]
    )
    gradient = jax.grad(lambda w: jnp.sum(replicate_yields(value, w)[1]))(jnp.ones(1))
    np.testing.assert_allclose(gradient, [1.125])


def test_independent_design_effect():
    rng = np.random.default_rng(42)
    pos = rng.uniform(0.25, 0.75, (2000, 8))
    r = report(outcome(pos))
    assert r.design_effects["y_issued"] == pytest.approx(1, abs=0.1)


@pytest.mark.parametrize("kind", ["batch", "bootstrap"])
def test_json_systematic_and_forward_model_separation(kind):
    r = report(
        outcome(kind=kind),
        systematic={
            "interp": {"y_cert_pos": 0.1},
            "index": {"y_cert_pos": [-0.2, 0.3]},
            "fm": {"y_cert_pos": 10},
        },
    )
    assert r.systematic["interp"]["y_cert_pos"] == (-0.1, 0.1)
    lo, hi = r.intervals["y_cert_pos"]
    assert r.total_intervals["y_cert_pos"] == pytest.approx((lo - 0.3, hi + 0.4))
    assert CertifiedYield.from_json(r.to_json()) == r
    json.dumps(json.loads(r.to_json()), allow_nan=False)


def test_no_replicates_and_old_json():
    r = report(make_outcome())
    assert r.interval_kind == "independent-lower-bound"
    data = json.loads(r.to_json())
    old = {
        key: value
        for key, value in data.items()
        if key
        in (
            "operator",
            "certificate",
            *CHANNELS,
            "weights",
            "issued",
            "cert_pos",
            "wrong",
            "neg",
            "neg_wrong",
            "in_range",
            *[name + "_se_independent_lower_bound" for name in CHANNELS[1:]],
        )
    }
    restored = CertifiedYield.from_json(json.dumps(old))
    assert restored.y_cert_pos == r.y_cert_pos
    assert restored.interval_kind == "independent-lower-bound"


def test_zero_denominator_and_zero_independent_variance():
    r = report(outcome(jnp.zeros((3, 1))))
    assert r.wrong_issued_ratio_interval is None
    assert r.design_effects["y_issued"] is None
    json.dumps(json.loads(r.to_json()), allow_nan=False)


@pytest.mark.parametrize("confidence", [0, 1, float("nan")])
def test_invalid_confidence(confidence):
    with pytest.raises(ValueError, match="confidence"):
        report(outcome(), confidence=confidence)


@pytest.mark.parametrize(
    "systematic",
    [
        {"interp": {"bad": 1}},
        {"interp": {"y_issued": -1}},
        {"interp": {"y_issued": [2, 1]}},
    ],
)
def test_invalid_systematic(systematic):
    with pytest.raises(ValueError):
        report(outcome(), systematic=systematic)


@pytest.mark.parametrize("kind", ["batch", "bootstrap"])
def test_nonconstant_ratio_uses_replicates(kind):
    value = outcome(kind=kind, p_wrong_rep=[[0.125], [0.125], [0.125]], p_wrong=[0.125])
    r = report(value)
    ratios = np.array([0.5, 0.25, 1 / 6])
    assert r.wrong_issued_ratio_rep == pytest.approx(ratios)
    sd = ratios.std(ddof=1)
    if kind == "batch":
        se = sd / np.sqrt(3)
        half = 4.3026527299 * se
        expected = (0.25 - half, 0.25 + half)
    else:
        se = sd
        expected = (1 / 6 + 0.05 * (0.25 - 1 / 6), 0.5 - 0.05 * 0.25)
    assert r.wrong_issued_ratio_se == pytest.approx(se)
    assert r.wrong_issued_ratio_interval == pytest.approx(expected)


def test_custom_confidence():
    r = report(outcome(kind="bootstrap"), confidence=0.8)
    assert r.intervals["y_issued"] == pytest.approx((0.3, 0.7))
    assert r.confidence == 0.8


def test_construction_with_replicates_under_jit():
    value = jax.jit(lambda pos: outcome(pos))(jnp.array([[0.25], [0.5], [0.75]]))
    assert value.replicate_kind == "batch"
    assert value.replicate_key == "stream"
    np.testing.assert_allclose(value.p_pos, [0.5])


@pytest.mark.parametrize(
    "hashes", [(None, None), ("a" * 64, "a" * 64), ("a" * 64, "b" * 64)]
)
@pytest.mark.parametrize("as_reports", [False, True])
def test_paired_difference_records_both_profile_hashes(hashes, as_reports):
    a, b = (outcome(profile_hash=h) for h in hashes)
    comparison = (
        paired_difference(report(a), report(b))
        if as_reports
        else paired_difference(a, b, weights=[1])
    )
    assert (comparison.profile_hash_a, comparison.profile_hash_b) == hashes
