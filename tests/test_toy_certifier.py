"""Tests for the closed-form toy certifier, the integrate-interface reference.

Expected values are computed inside the tests from primitives (scipy's normal
distribution, the declared SNR mapping typed once here), never from the
certifier's own helpers.
"""

import ineedvalidation as vv
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import floats
from scipy.stats import norm

from spaceodyssey.certify import CertificateState, SnrThreshold, ToyGaussianCertifier
from spaceodyssey.errors import UnsupportedOperator
from spaceodyssey.plan import CertificationOutcome, PlanPredictive
from spaceodyssey.registry import REGISTRY
from spaceodyssey.testing import check_certificate, check_outcome, check_ports
from tests.test_data import make_dataset
from tests.test_plan import make_outcome, make_plan

ALPHA = 0.05
BOUNDARY = 0.3
SIGMA_UNIT_SNR = 2.0


def make_toy(**overrides) -> ToyGaussianCertifier:
    """The toy certifier at the reference settings."""
    fields = {"boundary": BOUNDARY, "alpha": ALPHA, "sigma_unit_snr": SIGMA_UNIT_SNR}
    fields.update(overrides)
    return ToyGaussianCertifier(**fields)


def predictive_for(theta, combined_snr) -> PlanPredictive:
    """A one-bin plan with the given combined SNR per planet."""
    theta = jnp.asarray(theta, dtype=float)
    snr = jnp.asarray(combined_snr, dtype=float)[:, None]
    plan = make_plan(n_planets=theta.shape[0], n_bins=1, noise_sigma=1.0 / snr)
    return PlanPredictive(plan=plan, truth=theta)


class TestClosedForm:
    """The exact law against an independent Monte Carlo of the same toy."""

    def test_matches_monte_carlo(self):
        """p_pos, p_neg, p_wrong, p_neg_wrong agree with simulated certification.

        Truths straddle the boundary at several SNRs. The Monte Carlo draws
        posterior means m = theta + sigma e and certifies from the posterior
        tail probability computed with scipy, independent of the toy's
        decision helper. Tolerance: monte-carlo, five binomial standard
        errors at N = 40000 draws per planet, seed 0.
        """
        theta = np.array([0.3, 0.5, 1.2, 0.1, -0.4, 0.3, 0.9])
        combined = np.array([4.0, 8.0, 3.0, 12.0, 6.0, 20.0, 1.5])
        n_draws = 40_000
        outcome = make_toy().integrate(predictive_for(theta, combined))

        sigma = SIGMA_UNIT_SNR / combined
        noise = np.asarray(jr.normal(jr.PRNGKey(0), (n_draws, theta.size)))
        posterior_mean = theta + sigma * noise
        p_above = norm.sf(BOUNDARY, loc=posterior_mean, scale=sigma)
        positive = p_above >= 1.0 - ALPHA
        negative = 1.0 - p_above >= 1.0 - ALPHA
        wrong = positive & (theta <= BOUNDARY)
        neg_wrong = negative & (theta > BOUNDARY)
        assert neg_wrong.any(), "some wrong negatives must be drawn"

        checked = 0
        for name, hits in (
            ("p_pos", positive),
            ("p_neg", negative),
            ("p_wrong", wrong),
            ("p_neg_wrong", neg_wrong),
        ):
            exact = np.asarray(getattr(outcome, name), dtype=float)
            standard_error = np.sqrt(exact * (1.0 - exact) / n_draws)
            np.testing.assert_array_less(
                np.abs(hits.mean(axis=0) - exact),
                5.0 * standard_error + 1e-9,
                err_msg=name,
            )
            checked += int(np.sum((exact > 0.01) & (exact < 0.99)))
        assert checked >= 5, "too few planets in the informative range"

    def test_boundary_planet_certifies_at_alpha(self):
        """A truth on the boundary certifies each way with probability alpha.

        Tolerance: floating-point (float32 normal cdf near 0.05).
        """
        outcome = make_toy().integrate(predictive_for([BOUNDARY], [5.0]))
        np.testing.assert_allclose(float(outcome.p_pos[0]), ALPHA, rtol=1e-5)
        np.testing.assert_allclose(float(outcome.p_neg[0]), ALPHA, rtol=1e-5)
        assert float(outcome.p_wrong[0]) == pytest.approx(float(outcome.p_pos[0]))
        assert not bool(outcome.truth_positive[0])

    def test_wrong_only_when_truth_below_boundary(self):
        """Positive certifications of true candidates are never wrong."""
        outcome = make_toy().integrate(predictive_for([1.0, -1.0], [2.0, 2.0]))
        assert float(outcome.p_wrong[0]) == 0.0
        assert float(outcome.p_wrong[1]) == pytest.approx(float(outcome.p_pos[1]))
        assert bool(outcome.truth_positive[0])
        assert not bool(outcome.truth_positive[1])

    def test_negative_wrong_only_when_truth_above_boundary(self):
        """Every negative verdict on a true candidate is wrong; none below."""
        outcome = make_toy().integrate(predictive_for([1.0, -1.0], [0.5, 0.5]))
        assert float(outcome.p_neg_wrong[0]) == float(outcome.p_neg[0])
        assert float(outcome.p_neg_wrong[0]) > 0.0
        assert float(outcome.p_neg_wrong[1]) == 0.0

    @settings(deadline=None, max_examples=50)
    @given(
        theta=floats(min_value=-5.0, max_value=5.0),
        snr=floats(min_value=0.05, max_value=50.0),
    )
    def test_wrong_verdicts_bounded_by_verdicts(self, theta, snr):
        """p_neg_wrong <= p_neg and p_wrong <= p_pos for any truth and SNR."""
        outcome = make_toy().integrate(predictive_for([theta], [snr]))
        assert float(outcome.p_neg_wrong[0]) <= float(outcome.p_neg[0])
        assert float(outcome.p_wrong[0]) <= float(outcome.p_pos[0])

    def test_truth_mask_matches_boundary(self):
        """The truth-candidate flag is theta > boundary."""
        outcome = make_toy().integrate(predictive_for([1.0, -1.0], [2.0, 2.0]))
        assert bool(outcome.truth_positive[0])
        assert not bool(outcome.truth_positive[1])

    def test_closed_form_uses_quadrature_snr(self):
        """Bins combine in quadrature: SNR (3, 4) equals a single bin of 5.

        Tolerance: exact-algebra (sqrt(9 + 16) = 5 in floating point).
        """
        toy = make_toy()
        two_bin = make_plan(
            n_planets=1, n_bins=2, noise_sigma=1.0 / jnp.array([[3.0, 4.0]])
        )
        one_bin = make_plan(n_planets=1, n_bins=1, noise_sigma=jnp.array([[0.2]]))
        theta = jnp.array([0.8])
        a = toy.integrate(PlanPredictive(plan=two_bin, truth=theta))
        b = toy.integrate(PlanPredictive(plan=one_bin, truth=theta))
        assert float(a.p_pos[0]) == float(b.p_pos[0])

    @settings(deadline=None, max_examples=50)
    @given(
        theta=floats(min_value=BOUNDARY + 1e-3, max_value=BOUNDARY + 5.0),
        low=floats(min_value=0.05, max_value=50.0),
        factor=floats(min_value=1.0, max_value=10.0),
    )
    def test_p_pos_nondecreasing_in_snr(self, theta, low, factor):
        """For a truth above the boundary, more SNR never lowers p_pos."""
        outcome = make_toy().integrate(
            predictive_for([theta, theta], [low, low * factor])
        )
        assert float(outcome.p_pos[1]) >= float(outcome.p_pos[0])


@vv.case("parity-limit", "code-verification", srq="snr-count")
class TestParityLimit:
    """Support only, known noise, one prior: the SNR-threshold indicator."""

    def test_verdict_equals_snr_threshold_per_realization(self):
        """On every realization the toy certifies exactly when SnrThreshold does.

        Detection claim: theta is the signal amplitude, boundary 0, so the
        expected SNR is theta / sigma and the observed statistic is m / sigma.
        The matching threshold is Phi^-1(1 - alpha). Tolerance: exact-algebra
        (both sides compare the same statistic with the same bar).
        """
        toy = make_toy(boundary=0.0, sigma_unit_snr=1.0)
        threshold = float(norm.isf(ALPHA))
        reference = SnrThreshold(threshold=threshold)
        expected_snr = 2.0
        sigma = 1.0 / expected_snr
        noise = jr.normal(jr.PRNGKey(0), (2000,))
        posterior_mean = 1.0 + sigma * noise
        positive, _ = toy.verdict(posterior_mean, sigma)
        observed_snr = posterior_mean / sigma
        crossed = reference.update(
            reference.initial(), make_dataset(snr=observed_snr)
        ).crossed
        decided = np.asarray(positive)
        assert 0 < decided.sum() < decided.size, "draws must land on both sides"
        np.testing.assert_array_equal(decided, np.asarray(crossed))

    def test_completeness_is_the_threshold_detection_probability(self):
        """C reduces to P(observed SNR >= threshold) = Phi(SNR - z).

        Tolerance: floating-point (float32 normal cdf).
        """
        toy = make_toy(boundary=0.0, sigma_unit_snr=1.0)
        snr = np.array([0.5, 1.645, 3.0, 7.0])
        outcome = toy.integrate(predictive_for(np.ones_like(snr), snr))
        expected = norm.cdf(snr - norm.isf(ALPHA))
        np.testing.assert_allclose(np.asarray(outcome.p_pos), expected, rtol=2e-5)


class TestRangeFlag:
    """Out-of-range is a returned flag, never an exception inside jit."""

    def test_unknown_noise_flags_out_of_range_under_jit(self):
        """Unknown noise is outside the known-noise toy, with finite outputs."""
        toy = make_toy()
        predictive = PlanPredictive(
            plan=make_plan(n_planets=2, noise_known=[False, True]),
            truth=jnp.ones(2),
        )
        outcome = jax.jit(toy.integrate)(predictive)
        assert not bool(outcome.in_range[0])
        assert bool(outcome.in_range[1])
        assert float(outcome.p_pos[0]) == pytest.approx(ALPHA, rel=1e-5)
        assert bool(jnp.all(jnp.isfinite(outcome.p_pos)))

    def test_jit_equals_eager(self):
        """The compiled and eager outcomes agree bitwise."""
        toy = make_toy()
        predictive = predictive_for([0.1, 0.4, 0.9], [2.0, 6.0, 10.0])
        eager = toy.integrate(predictive)
        compiled = jax.jit(toy.integrate)(predictive)
        np.testing.assert_array_equal(np.asarray(eager.p_pos), compiled.p_pos)
        np.testing.assert_array_equal(np.asarray(eager.p_neg), compiled.p_neg)


class TestSeamConformance:
    """Registry metadata and the certification conformance suite."""

    def test_registered_as_closed_form_integrate_peer(self):
        """The toy is a peer on the certification seam."""
        entry = REGISTRY.get("certification", "toy_gaussian")
        assert entry.cls is ToyGaussianCertifier
        assert entry.info.operators == frozenset({"integrate"})
        assert entry.info.fidelity == "closed_form"

    def test_conformance_suite_passes(self):
        """Declared operators are honest and the outcome meets its contract."""
        toy = make_toy()
        info = REGISTRY.get("certification", "toy_gaussian").info
        predictive = predictive_for([0.1, 0.6], [3.0, 9.0])
        state = check_certificate(toy, info, [make_dataset()], predictive=predictive)
        assert isinstance(state, CertificateState)
        check_ports(toy.integrate(predictive), info)

    def test_sample_operator_unsupported(self):
        """The toy evaluates the integrate operator only."""
        toy = make_toy()
        with pytest.raises(UnsupportedOperator):
            toy.update(toy.initial(), make_dataset())

    def test_outcome_check_catches_overlapping_verdicts(self):
        """p_pos + p_neg > 1 breaks the outcome contract."""
        bad = make_outcome(p_pos=jnp.full((3,), 0.7), p_neg=jnp.full((3,), 0.5))
        with pytest.raises(AssertionError, match="exceeds one"):
            check_outcome(bad)

    def test_outcome_check_ignores_out_of_range_values(self):
        """Untrusted out-of-range values are excluded, but never all of them."""
        flags = jnp.array([True, False, True])
        loose = make_outcome(p_pos=jnp.array([0.5, 2.0, 0.5]), in_range=flags)
        check_outcome(loose)
        with pytest.raises(AssertionError, match="no planet"):
            check_outcome(make_outcome(in_range=jnp.zeros(3, dtype=bool)))

    @pytest.mark.parametrize("alpha", [0.0, 0.6, -0.1])
    def test_invalid_alpha_rejected(self, alpha):
        """Alpha outside (0, 0.5] would let both verdicts overlap or vanish."""
        with pytest.raises(ValueError, match="alpha"):
            make_toy(alpha=alpha)

    def test_outcome_type(self):
        """Integrate returns the outcome contract."""
        outcome = make_toy().integrate(predictive_for([0.5], [4.0]))
        assert isinstance(outcome, CertificationOutcome)


@pytest.mark.parametrize("rank", [None, 0])
def test_white_noise_matches_v1_closed_form(rank):
    snr = np.array([[1.0, 2.0], [3.0, 4.0], [0.2, 0.3]])
    theta = np.array([0.1, 0.6, 1.0])
    factors = None if rank is None else jnp.zeros((3, 2, rank))
    plan = make_plan(noise_sigma=1.0 / snr, noise_corr_factors=factors)
    outcome = make_toy().integrate(PlanPredictive(plan=plan, truth=theta))
    d = (theta - BOUNDARY) * np.sqrt(np.sum(snr**2, axis=1)) / SIGMA_UNIT_SNR
    np.testing.assert_allclose(outcome.p_pos, norm.cdf(d - norm.isf(ALPHA)), rtol=2e-6)
    np.testing.assert_allclose(outcome.p_neg, norm.cdf(-d - norm.isf(ALPHA)), rtol=2e-6)
    plain = make_toy().integrate(
        PlanPredictive(plan=make_plan(noise_sigma=1.0 / snr), truth=theta)
    )
    np.testing.assert_array_equal(outcome.p_pos, plain.p_pos)


def test_correlated_noise_matches_dense_reference_under_jit():
    noise = np.array([[0.2, 0.4], [0.3, 0.1], [0.2, 0.3]])
    factors = np.array([[[0.4], [0.4]], [[0.1], [0.3]], [[0.2], [-0.1]]])
    theta = np.array([0.5, 0.8, 0.1])
    plan = make_plan(noise_sigma=noise, noise_corr_factors=factors)
    toy = make_toy()
    outcome = jax.jit(toy.integrate)(PlanPredictive(plan=plan, truth=theta))
    combined = np.array(
        [
            np.sqrt(np.ones(2) @ np.linalg.solve(np.diag(n**2) + f @ f.T, np.ones(2)))
            for n, f in zip(noise, factors, strict=True)
        ]
    )
    np.testing.assert_allclose(
        toy.posterior_sd(plan), SIGMA_UNIT_SNR / combined, rtol=2e-6
    )
    expected = norm.cdf(
        (theta - BOUNDARY) * combined / SIGMA_UNIT_SNR - norm.isf(ALPHA)
    )
    np.testing.assert_allclose(outcome.p_pos, expected, rtol=2e-6)


@pytest.mark.parametrize(
    "case", ["no_spectrum", "no_success", "failed_spectrum", "no_visits"]
)
def test_absent_data_has_zero_finite_verdicts(case):
    fields = {
        "no_spectrum": {"spectrum_visit": [-1, -1, -1]},
        "no_success": {"visit_success": jnp.zeros((3, 4), dtype=bool)},
        "failed_spectrum": {"visit_success": [[False, True, True, True]] * 3},
        "no_visits": {"n_visits": 0},
    }[case]
    predictive = PlanPredictive(
        plan=make_plan(**fields), truth=jnp.array([0.1, 0.5, 1.0])
    )
    for integrate in (make_toy().integrate, jax.jit(make_toy().integrate)):
        outcome = integrate(predictive)
        for leaf in jax.tree_util.tree_leaves(outcome):
            assert bool(jnp.all(jnp.isfinite(leaf)))
        np.testing.assert_array_equal(outcome.p_pos, 0.0)
        np.testing.assert_array_equal(outcome.p_neg, 0.0)
        assert bool(jnp.all(outcome.in_range))
