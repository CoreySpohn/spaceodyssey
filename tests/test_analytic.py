"""Tests for the weighted-yield report and the CertifiedYield container."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from spaceodyssey.metrics import CertifiedYield
from spaceodyssey.reporting import certified_yield, weighted_yields
from tests.test_plan import make_outcome


def hand_outcome():
    """Four planets with hand-chosen probabilities and Monte Carlo errors."""
    return make_outcome(
        n_planets=4,
        p_pos=jnp.array([0.9, 0.5, 0.2, 0.0]),
        p_neg=jnp.array([0.0, 0.1, 0.6, 0.9]),
        p_wrong=jnp.array([0.0, 0.1, 0.2, 0.0]),
        p_neg_wrong=jnp.array([0.0, 0.1, 0.0, 0.0]),
        p_pos_se=jnp.array([0.01, 0.02, 0.03, 0.0]),
        p_neg_se=jnp.array([0.0, 0.01, 0.02, 0.04]),
        p_wrong_se=jnp.array([0.0, 0.0, 0.03, 0.0]),
        p_neg_wrong_se=jnp.array([0.0, 0.01, 0.0, 0.0]),
        truth_positive=jnp.array([True, True, False, False]),
    )


WEIGHTS = jnp.array([1.0, 2.0, 0.5, 4.0])


class TestWeightedYields:
    """The five yields as weighted sums over a planet batch."""

    def test_hand_values(self):
        """Y_cert+ excludes wrong claims, including those on truth candidates.

        Tolerance: exact-algebra (float32 sums of a few products).
        """
        y_issued, y_pos, y_wrong, y_neg, y_neg_wrong = weighted_yields(
            hand_outcome(), WEIGHTS
        )
        assert float(y_issued) == pytest.approx(2.0)
        assert float(y_pos) == pytest.approx(1.7)
        assert float(y_wrong) == pytest.approx(0.3)
        assert float(y_neg) == pytest.approx(2.0 * 0.1 + 0.5 * 0.6 + 4.0 * 0.9)
        assert float(y_neg_wrong) == pytest.approx(2.0 * 0.1)

    def test_report_breakdown_sums_to_yields(self):
        """Per-planet columns sum to the yields, so a bootstrap re-sums rows."""
        report = certified_yield(hand_outcome(), WEIGHTS, certificate="hand")
        assert report.y_issued == pytest.approx(sum(report.issued))
        assert report.y_cert_pos == pytest.approx(sum(report.cert_pos))
        assert report.cert_pos[1] == pytest.approx(0.8)
        assert report.issued[2] == pytest.approx(0.1)
        assert report.wrong[2] == pytest.approx(0.1)
        assert report.y_wrong == pytest.approx(sum(report.wrong))
        assert report.y_neg == pytest.approx(sum(report.neg))
        assert report.y_neg_wrong == pytest.approx(sum(report.neg_wrong))
        assert report.cert_pos[2] == 0.0
        assert len(report.weights) == 4

    def test_report_standard_errors_in_quadrature(self):
        """Per-planet errors propagate in quadrature; Y+ uses se(p_pos) + se(p_wrong).

        The correct-certification probability is p_pos - p_wrong, whose error
        is bounded by the sum of the two channel errors, over all planets.
        """
        outcome = hand_outcome()
        report = certified_yield(outcome, WEIGHTS, certificate="hand")
        per_planet = np.asarray(WEIGHTS) * (
            np.asarray(outcome.p_pos_se) + np.asarray(outcome.p_wrong_se)
        )
        assert report.y_cert_pos_se_independent_lower_bound == pytest.approx(
            float(np.sqrt(np.sum(per_planet**2)))
        )
        assert report.y_wrong_se_independent_lower_bound == pytest.approx(0.5 * 0.03)
        assert report.y_neg_wrong_se_independent_lower_bound == pytest.approx(
            2.0 * 0.01
        )
        assert report.y_neg_se_independent_lower_bound == pytest.approx(
            np.sqrt((2.0 * 0.01) ** 2 + (0.5 * 0.02) ** 2 + (4.0 * 0.04) ** 2)
        )

    def test_report_json_round_trip(self):
        """The report survives serialization exactly."""
        report = certified_yield(hand_outcome(), WEIGHTS, certificate="hand")
        assert CertifiedYield.from_json(report.to_json()) == report
        assert report.all_in_range

    def test_legacy_standard_error_names(self):
        """Old JSON standard-error keys remain readable after the rename."""
        report = certified_yield(hand_outcome(), WEIGHTS, certificate="hand")
        data = json.loads(report.to_json())
        for name in list(data):
            if name.endswith("_se_independent_lower_bound"):
                data[name.removesuffix("_independent_lower_bound")] = data.pop(name)
        assert CertifiedYield.from_json(json.dumps(data)) == report

    def test_weight_shape_checked(self):
        """Weights must be one per planet."""
        with pytest.raises(ValueError, match="weights"):
            certified_yield(hand_outcome(), jnp.ones(3), certificate="hand")

    def test_differentiable_in_probabilities(self):
        """Y_cert+ is differentiable in the outcome probabilities.

        The gradient is w * (1 - r) for a wrong fraction r.

        Tolerance: floating-point; the yield is linear in p_pos, so check_grads
        agrees with the exact gradient to 1e-6 in float64.
        """
        weights = jnp.array([1.0, 2.0, 0.5])
        wrong_fraction = 0.25

        def y_pos(p_pos):
            outcome = make_outcome(p_pos=p_pos, p_wrong=wrong_fraction * p_pos)
            return weighted_yields(outcome, weights)[1]

        p_pos = jnp.array([0.9, 0.5, 0.2])
        check_grads(y_pos, (p_pos,), order=1, modes=["rev"], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(
            jax.grad(y_pos)(p_pos), weights * (1.0 - wrong_fraction), rtol=1e-12
        )
