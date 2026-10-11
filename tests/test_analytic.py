"""Tests for the analytic integrate engine and the CertifiedYield report."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from spaceodyssey import engines
from spaceodyssey.build import build
from spaceodyssey.engines.analytic import certified_yield, weighted_yields
from spaceodyssey.errors import OutOfRangeError, WiringError
from spaceodyssey.metrics import CertifiedYield
from spaceodyssey.plan import PlanPredictive
from spaceodyssey.spec import FidelityConfig, SeamRef
from tests.doubles import make_spec
from tests.test_plan import make_outcome, make_plan
from tests.test_toy_certifier import make_toy, predictive_for


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

    def test_matched_flat_prior_wrong_yield_bound(self):
        """A proper wide-window approximation to the toy's improper flat prior.

        Seed 2026, 40000 uniform truths within 10 noise sigmas of the
        boundary. The distant window edges affect the certification tail
        negligibly (normal tails beyond 8 sigma). Tolerance is five sample
        standard errors of p_wrong - alpha*p_pos, covering population Monte
        Carlo variation; noise is integrated analytically, not sampled.
        """
        toy = make_toy()
        rng = np.random.default_rng(2026)
        theta = rng.uniform(toy.boundary - 2.0, toy.boundary + 2.0, 40_000)
        outcome = toy.integrate(predictive_for(theta, np.full(theta.size, 10.0)))
        report = certified_yield(outcome, np.ones(theta.size), certificate="toy")
        residual = np.asarray(outcome.p_wrong) - toy.alpha * np.asarray(outcome.p_pos)
        tolerance = 5.0 * np.std(residual, ddof=1) * np.sqrt(theta.size)
        assert report.y_issued > 0.0
        assert report.y_wrong > 0.0
        assert report.y_wrong <= toy.alpha * report.y_issued + tolerance

    def test_weight_shape_checked(self):
        """Weights must be one per planet."""
        with pytest.raises(ValueError, match="weights"):
            certified_yield(hand_outcome(), jnp.ones(3), certificate="hand")

    def test_differentiable_in_snr(self):
        """Y_cert+ from the toy is differentiable in the toy adapter's SNR.

        Tolerance: floating-point; float32 central differences at eps 1e-2
        carry O(eps**2) truncation and round-off, so check_grads uses 2e-2.
        """
        toy = make_toy()
        theta = jnp.array([0.5, 0.9, 0.2])

        def y_pos(snr):
            plan = make_plan(n_planets=3, n_bins=2, noise_sigma=1.0 / snr)
            outcome = toy.integrate(PlanPredictive(plan=plan, truth=theta))
            return weighted_yields(outcome, jnp.ones(3))[1]

        snr = jnp.array([[2.0, 1.0], [1.5, 1.5], [3.0, 0.5]])
        check_grads(
            y_pos, (snr,), order=1, modes=["rev"], eps=1e-2, atol=2e-2, rtol=2e-2
        )
        assert float(jnp.sum(jax.grad(y_pos)(snr))) != 0.0


class TestIntegrateEngine:
    """spec + integrate config + predictive in, CertifiedYield out."""

    @staticmethod
    def config():
        """An integrate-operator configuration selecting the toy certifier."""
        return FidelityConfig.make(
            "integrate",
            certification=SeamRef.make(
                "certification",
                "toy_gaussian",
                boundary=0.3,
                alpha=0.05,
                sigma_unit_snr=2.0,
            ),
        )

    def test_end_to_end(self):
        """The engine reports the weighted sum of the certificate's outcome."""
        predictive = predictive_for([0.1, 0.6, 1.0], [3.0, 9.0, 5.0])
        weights = jnp.array([1.0, 0.5, 2.0])
        report = engines.run(
            make_spec(), self.config(), predictive=predictive, weights=weights
        )
        assert isinstance(report, CertifiedYield)
        assert report.operator == "integrate"
        assert report.certificate == "toy_gaussian"
        expected = weighted_yields(make_toy().integrate(predictive), weights)
        assert report.y_issued == pytest.approx(float(expected[0]))
        assert report.y_cert_pos == pytest.approx(float(expected[1]))
        assert report.y_wrong == pytest.approx(float(expected[2]))
        assert report.y_neg == pytest.approx(float(expected[3]))
        assert report.y_neg_wrong == pytest.approx(float(expected[4]))

    def test_out_of_range_raises_on_host(self):
        """The engine is the Python wrapper that turns the flag into an error."""
        predictive = PlanPredictive(
            plan=make_plan(n_planets=2, noise_known=[False, True]),
            truth=jnp.array([0.5, 0.5]),
        )
        with pytest.raises(OutOfRangeError, match=r"\[0\]"):
            engines.run(
                make_spec(), self.config(), predictive=predictive, weights=jnp.ones(2)
            )

    def test_out_of_range_reported_when_not_strict(self):
        """With strict off, the report carries the flags instead."""
        predictive = PlanPredictive(
            plan=make_plan(n_planets=2, noise_known=[False, True]),
            truth=jnp.array([0.5, 0.5]),
        )
        report = engines.run(
            make_spec(),
            self.config(),
            predictive=predictive,
            weights=jnp.ones(2),
            strict=False,
        )
        assert report.in_range == (False, True)
        assert not report.all_in_range

    def test_requires_a_certificate(self):
        """An integrate run with no certification choice is refused."""
        mission = build(make_spec(), FidelityConfig.make("integrate"))
        with pytest.raises(WiringError, match="certification"):
            engines.get_engine("integrate")(
                mission, predictive=predictive_for([0.5], [4.0]), weights=jnp.ones(1)
            )


@pytest.mark.parametrize("n_visits", [0, 4])
def test_missing_data_engine_yields_are_finite(n_visits):
    plan = make_plan(
        n_visits=n_visits,
        spectrum_visit=jnp.full(3, -1) if not n_visits else jnp.array([-1, 0, 0]),
        visit_success=jnp.zeros((3, n_visits), dtype=bool),
    )
    predictive = PlanPredictive(plan=plan, truth=jnp.array([0.1, 0.5, 1.0]))
    outcome = make_toy().integrate(predictive)
    yields = jax.jit(weighted_yields)(outcome, jnp.ones(3))
    np.testing.assert_array_equal(yields, np.zeros(5))
    report = engines.run(
        make_spec(),
        TestIntegrateEngine.config(),
        predictive=predictive,
        weights=jnp.ones(3),
    )
    assert report.all_in_range
    assert report.y_issued == report.y_cert_pos == report.y_wrong == 0.0
    assert report.y_neg == report.y_neg_wrong == 0.0
    json.dumps(json.loads(report.to_json()), allow_nan=False)


def test_certified_yield_se_bounds_the_correct_certification_channel():
    """Y_cert+ SE uses se(p_pos) + se(p_wrong) over all planets.

    The correct-certification probability is p_pos - p_wrong, so its error is
    bounded by the sum of the two channel errors; non-candidates contribute.
    """
    import jax.numpy as jnp

    from spaceodyssey.engines.analytic import certified_yield
    from spaceodyssey.plan import CertificationOutcome

    outcome = CertificationOutcome(
        p_pos=jnp.array([0.6, 0.2]),
        p_neg=jnp.array([0.1, 0.5]),
        p_wrong=jnp.array([0.1, 0.2]),
        p_neg_wrong=jnp.array([0.0, 0.1]),
        p_pos_se=jnp.array([0.03, 0.04]),
        p_neg_se=jnp.array([0.01, 0.01]),
        p_wrong_se=jnp.array([0.02, 0.04]),
        p_neg_wrong_se=jnp.array([0.0, 0.01]),
        truth_positive=jnp.array([True, False]),
        in_range=jnp.array([True, True]),
    )
    report = certified_yield(outcome, jnp.array([1.0, 2.0]), certificate="toy")
    expected = float(jnp.sqrt((1.0 * 0.05) ** 2 + (2.0 * 0.08) ** 2))
    assert report.y_cert_pos_se_independent_lower_bound == pytest.approx(
        expected, rel=1e-6
    )
