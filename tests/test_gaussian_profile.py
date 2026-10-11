"""Posterior certification probabilities under linear Gaussian experiments."""

from dataclasses import replace
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from photomancy.priors import Normal
from scipy.optimize import brentq
from scipy.stats import norm

from spaceodyssey import Claim, Edge, PlanPredictive, PriorSet, Profile, Truth
from spaceodyssey.errors import UnsupportedProfile
from spaceodyssey.testing import LinearGaussianTruthModel, check_outcome
from tests.test_plan import make_plan


@pytest.mark.parametrize(
    "width, sd, alpha", [(2.0, 0.3, 0.05), (1.0, 0.15, 0.01), (0.01, 0.001, 0.1)]
)
def test_interval_boundary_matches_brentq(width, sd, alpha):
    from spaceodyssey.gaussian_profile import _interval_radius

    with jax.enable_x64():
        for probability in (alpha, 1 - alpha):
            expected = brentq(
                lambda radius, probability=probability: (
                    norm.cdf((width / 2 - radius) / sd)
                    - norm.cdf((-width / 2 - radius) / sd)
                    - probability
                ),
                0,
                width / 2 + 10 * sd,
                xtol=5e-15,
            )
            radius = jax.jit(_interval_radius)(width, sd, probability)
            np.testing.assert_allclose(radius, expected, rtol=0, atol=1e-12)


def test_interval_too_narrow_is_empty():
    from spaceodyssey.gaussian_profile import _interval_radius

    with jax.enable_x64():
        assert float(_interval_radius(0.1, 1.0, 0.95)) < 0


def test_public_registered_certifier():
    import spaceodyssey
    from spaceodyssey.registry import REGISTRY

    assert hasattr(spaceodyssey, "GaussianProfileCertifier")
    assert (
        REGISTRY.get("certification", "gaussian_profile").cls
        is spaceodyssey.GaussianProfileCertifier
    )


class GaussianModel(LinearGaussianTruthModel):
    prior: Normal

    def __init__(self, matrix=None, scale=2.0, prior_set_id="normal@1"):
        self.spectral_matrix = (
            jnp.eye(len(self.param_names)) if matrix is None else jnp.asarray(matrix)
        )
        self.prior = Normal(
            jnp.zeros(len(self.param_names)), jnp.full(len(self.param_names), scale)
        )
        self.prior_sets = (PriorSet((self.prior,), prior_set_id, ("nominal",), 0),)


class ReferenceModel(GaussianModel):
    """Interface fixture with both radius names mapped to one toy scalar.

    These quantities test profile plumbing, not a physical orbit-radius model.
    """

    param_names: ClassVar = ("ln_a_eeid", "ln_rp_re", "log10_h2o_vmr")
    param_units: ClassVar = ("1", "1", "1")
    quantity_names: ClassVar = (*param_names, "ln_rp_sqrt_a_eeid")

    def quantities(self, theta, context):
        result = super().quantities(theta, context)
        result["ln_rp_sqrt_a_eeid"] = theta[..., 1]
        return result


def profile_for(claims, **kwargs):
    return Profile(
        "experiment",
        tuple(claims),
        0.05,
        support_rule=kwargs.get("rule", "separate"),
        prior_set_id="normal@1",
    )


def experiment(model, theta, **plan_kwargs):
    theta = jnp.asarray(theta, dtype=float)
    plan = make_plan(
        n_planets=len(theta), n_bins=model.spectral_matrix.shape[0], **plan_kwargs
    )
    truth = Truth(theta, {"offset": jnp.zeros(len(theta))}, model.model_id)
    return PlanPredictive(plan, truth)


def test_flat_threshold_matches_toy():
    from spaceodyssey import GaussianProfileCertifier, ToyGaussianCertifier

    with jax.enable_x64():
        model = GaussianModel()
        profile = profile_for(
            [Claim("x", (Edge("x", 0.3, "above"),), "positive")], rule="joint"
        )
        predictive = experiment(model, [[-0.5, 0], [0.1, 0], [0.8, 0]])
        outcome = GaussianProfileCertifier(profile, model, flat_prior=True).integrate(
            predictive
        )
        toy = ToyGaussianCertifier(0.3, 0.05, 1.0).integrate(
            PlanPredictive(make_plan(n_bins=1), predictive.truth.theta[:, 0])
        )
        for name in ("p_pos", "p_neg", "p_wrong", "p_neg_wrong"):
            np.testing.assert_allclose(
                getattr(outcome, name), getattr(toy, name), atol=1e-12, rtol=0
            )
        assert outcome.fidelity == "closed_form"
        check_outcome(outcome, profile)


def test_joint_rejected_before_model_work(monkeypatch):
    from spaceodyssey import GaussianProfileCertifier
    from spaceodyssey.profiles import EXOEARTH_CANDIDATE

    class FullReference(ReferenceModel):
        quantity_names: ClassVar = (*ReferenceModel.param_names, "ln_rp_sqrt_a_eeid")

        def __init__(self):
            self.spectral_matrix = jnp.eye(3)
            self.prior = Normal(jnp.zeros(3), jnp.ones(3))
            self.prior_sets = (
                PriorSet(
                    (self.prior,), EXOEARTH_CANDIDATE.prior_set_id, ("nominal",), 0
                ),
            )

    model = FullReference()

    def forbidden(*args, **kwargs):
        pytest.fail("joint rejection must precede model work")

    for name in ("quantities", "expected_data", "simulate"):
        monkeypatch.setattr(FullReference, name, forbidden)
    with pytest.raises(UnsupportedProfile, match=r"joint.*exact"):
        GaussianProfileCertifier(EXOEARTH_CANDIDATE, model)


@pytest.mark.parametrize("interval", [False, True])
@pytest.mark.parametrize("correlated", [False, True])
def test_separate_matches_data_monte_carlo(interval, correlated):
    from spaceodyssey import GaussianProfileCertifier

    with jax.enable_x64():
        model = GaussianModel(
            matrix=jnp.array([[1.0, 0.4 if correlated else 0], [0.0, 1.0]])
        )
        edges = (
            (Edge("x", -0.5, "above"), Edge("x", 0.7, "below"))
            if interval
            else (Edge("x", 0.0, "above"),)
        )
        profile = profile_for(
            [
                Claim("x", edges, "positive"),
                Claim("y", (Edge("y", 0.0, "above"),), "either"),
            ]
        )
        theta = np.array([[0.25, 0.15], [-0.1, -0.3], [0.6, 0.4]])
        predictive = experiment(model, theta)
        outcome = GaussianProfileCertifier(profile, model).integrate(predictive)
        assert outcome.fidelity == ("quadrature" if correlated else "closed_form")
        check_outcome(outcome, profile)
        a = np.asarray(model.spectral_matrix)
        post = np.linalg.inv(a.T @ a / 0.2**2 + np.eye(2) / 2**2)
        sd = np.sqrt(np.diag(post))
        rng = np.random.default_rng(821)
        data = theta @ a.T + rng.normal(0, 0.2, (200_000, 3, 2))
        means = data @ a @ post / 0.2**2
        mass_x = (
            norm.cdf((0.7 - means[..., 0]) / sd[0])
            - norm.cdf((-0.5 - means[..., 0]) / sd[0])
            if interval
            else norm.cdf(means[..., 0] / sd[0])
        )
        masses = np.stack([mass_x, norm.cdf(means[..., 1] / sd[1])], -1)
        pos, neg = masses >= 0.95, masses <= 0.05
        truth = np.stack(
            [
                ((theta[:, 0] >= -0.5) & (theta[:, 0] <= 0.7))
                if interval
                else theta[:, 0] >= 0,
                theta[:, 1] >= 0,
            ],
            -1,
        )
        counted = pos[..., 0] & (pos[..., 1] | neg[..., 1])
        wrong = np.any((pos & ~truth) | (neg & truth), -1)
        for name, hits in (
            ("p_pos", counted),
            ("p_wrong", counted & wrong),
            ("p_neg", neg[..., 0]),
            ("p_neg_wrong", neg[..., 0] & truth[:, 0]),
        ):
            expected = hits.mean(0)
            actual = np.asarray(getattr(outcome, name))
            se = np.sqrt(actual * (1 - actual) / len(hits))
            np.testing.assert_array_less(np.abs(actual - expected), 5 * se + 1e-12)


def test_reference_separate_and_orbit_run_integrate():
    from spaceodyssey import GaussianProfileCertifier
    from spaceodyssey.build import build
    from spaceodyssey.engines.analytic import run_integrate
    from spaceodyssey.profiles import EXOEARTH_CANDIDATE, HABITABLE_ZONE_ONLY
    from tests.doubles import make_spec
    from tests.test_analytic import TestIntegrateEngine

    with jax.enable_x64():
        reference = replace(EXOEARTH_CANDIDATE, support_rule="separate")
        model = ReferenceModel(prior_set_id=reference.prior_set_id, scale=5.0)
        predictive = experiment(
            model,
            [[0.2, 0.1, -3.0], [0.1, -0.1, -4.0], [0.5, 0.3, -5.0]],
            noise_sigma=jnp.full((3, 3), 0.05),
        )
        outcomes = []
        for profile in (reference, HABITABLE_ZONE_ONLY):
            cert = GaussianProfileCertifier(profile, model)
            outcome = cert.integrate(predictive)
            check_outcome(outcome, profile)
            mission = build(make_spec(), TestIntegrateEngine.config())
            mission = eqx.tree_at(lambda m: m.model.certification, mission, cert)
            report = run_integrate(
                mission, predictive=predictive, weights=[1.0, 1.0, 1.0]
            )
            assert report.profile_hash == profile.content_hash()
            outcomes.append(outcome)
        np.testing.assert_allclose(
            outcomes[1].p_pos, outcomes[0].claim_p_pos[:, 0], atol=1e-12, rtol=0
        )


def test_general_linear_quantity_and_nonlinear_rejection(monkeypatch):
    from spaceodyssey import GaussianProfileCertifier

    class Linear(GaussianModel):
        def quantities(self, theta, context):
            return {"x": 2 * theta[..., 0] - theta[..., 1], "y": theta[..., 1]}

    class Nonlinear(GaussianModel):
        def quantities(self, theta, context):
            return {"x": theta[..., 0] ** 2, "y": theta[..., 1]}

    profile = profile_for(
        [Claim("combination", (Edge("x", 0.1, "above"),), "positive")]
    )
    with jax.enable_x64():
        model = Linear()
        predictive = experiment(model, [[0.25, 0.1]])
        outcome = GaussianProfileCertifier(profile, model).integrate(predictive)
        posterior_variance = 1 / (25 + 0.25)
        post_sd = np.sqrt(5 * posterior_variance)
        sampling_sd = np.sqrt(5 * posterior_variance**2 * 25)
        expected_mean = 25 * posterior_variance * 0.4
        expected = norm.sf(
            (0.1 + norm.isf(0.05) * post_sd - expected_mean) / sampling_sd
        )
        assert float(outcome.p_pos[0]) == pytest.approx(expected, abs=1e-12)

        def forbidden(*args, **kwargs):
            pytest.fail("posterior computation must not run for nonlinear quantities")

        monkeypatch.setattr(Nonlinear, "expected_data", forbidden)
        with pytest.raises(UnsupportedProfile, match=r"nonlinear.*quantity"):
            GaussianProfileCertifier(profile, Nonlinear())


@pytest.mark.parametrize("rule", ["separate", "split"])
def test_negative_union_and_wrong_either_reports(rule):
    from spaceodyssey import GaussianProfileCertifier

    with jax.enable_x64():
        model = GaussianModel(matrix=jnp.array([[1.0, 0.5], [0.0, 1.0]]))
        claims = [
            Claim(name, (Edge(name, 0.0, "above"),), "positive") for name in ("x", "y")
        ]
        profile = profile_for(claims, rule=rule)
        predictive = experiment(model, [[0.1, 0.15]])
        result = GaussianProfileCertifier(profile, model).integrate(predictive)
        a = np.asarray(model.spectral_matrix)
        posterior = np.linalg.inv(a.T @ a / 0.04 + np.eye(2) / 4)
        data = np.array([0.1, 0.15]) @ a.T + np.random.default_rng(716).normal(
            0, 0.2, (200_000, 2)
        )
        means = data @ a @ posterior / 0.04
        mass = norm.cdf(means / np.sqrt(np.diag(posterior)))
        alpha = 0.05 / 2 if rule == "split" else 0.05
        for name, hits in (
            ("p_neg", np.any(mass <= alpha, -1)),
            ("p_pos", np.all(mass >= 1 - alpha, -1)),
        ):
            expected = hits.mean()
            assert abs(float(getattr(result, name)[0]) - expected) <= 5 * np.sqrt(
                expected * (1 - expected) / len(hits)
            )
        for k, claim in enumerate(claims):
            solo = GaussianProfileCertifier(
                replace(profile, claims=(claim,)), model
            ).integrate(predictive)
            np.testing.assert_allclose(result.claim_p_pos[:, k], solo.p_pos, atol=1e-12)
        check_outcome(result, profile)


def test_prior_evidence_and_missing_spectra():
    from spaceodyssey import GaussianProfileCertifier

    with jax.enable_x64():
        profile = profile_for([Claim("x", (Edge("x", -10.0, "above"),), "positive")])
        model = GaussianModel()
        result = GaussianProfileCertifier(profile, model).integrate(
            experiment(model, [[0.0, 0.0]])
        )
        assert float(result.p_pos[0]) == 0
        missing = GaussianProfileCertifier(profile, model).integrate(
            experiment(model, [[0.0, 0.0]], n_visits=0)
        )
        assert float(missing.p_pos[0]) == float(missing.p_neg[0]) == 0
        assert bool(missing.in_range[0])


def test_integer_cast_quantity_is_rejected():
    from spaceodyssey import GaussianProfileCertifier

    class Rounded(GaussianModel):
        def quantities(self, theta, context):
            x = theta[..., 0]
            return {"x": x + x.astype(jnp.int32).astype(x.dtype), "y": theta[..., 1]}

    profile = profile_for([Claim("x", (Edge("x", 0.0, "above"),), "positive")])
    with pytest.raises(UnsupportedProfile, match="nonlinear quantity"):
        GaussianProfileCertifier(profile, Rounded())


def test_threshold_boundary_uses_inclusive_profile_truth():
    from spaceodyssey import GaussianProfileCertifier

    with jax.enable_x64():
        model = GaussianModel()
        profile = profile_for([Claim("x", (Edge("x", 0.0, "above"),), "positive")])
        result = GaussianProfileCertifier(profile, model, flat_prior=True).integrate(
            experiment(model, [[0.0, 0.0]])
        )
        assert bool(result.truth_positive[0])
        np.testing.assert_allclose(result.p_pos, 0.05, atol=1e-12)
        np.testing.assert_allclose(result.p_neg_wrong, 0.05, atol=1e-12)
        assert float(result.p_wrong[0]) == 0
