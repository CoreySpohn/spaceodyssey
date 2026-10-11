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
    "dtype, width, sd, alpha",
    [
        (np.float64, 2.0, 0.3, 0.05),
        (np.float64, 1.0, 0.15, 0.01),
        (np.float64, 0.01, 0.001, 0.1),
        (np.float32, 2.0, 0.1, 1e-6),
        (np.float32, 2.0, 0.1, 1e-9),
    ],
)
def test_interval_boundary_matches_brentq(dtype, width, sd, alpha):
    from spaceodyssey.gaussian_profile import _interval_radius

    with jax.enable_x64():
        width, sd, alpha = map(dtype, (width, sd, alpha))
        w, s, a = map(float, (width, sd, alpha))
        for positive in (True, False):
            # Evaluate the reference in float64, including the tiny tails.
            def residual(radius, positive=positive):
                if positive:
                    return (
                        norm.cdf((radius - w / 2) / s)
                        + norm.cdf((-w / 2 - radius) / s)
                        - a
                    )
                return (
                    norm.cdf((w / 2 - radius) / s) - norm.cdf((-w / 2 - radius) / s) - a
                )

            expected = brentq(
                residual,
                0,
                w / 2 + 10 * s,
                xtol=5e-15,
            )
            radius, feasible = _interval_radius(width, sd, alpha, positive=positive)
            assert bool(feasible)
            assert radius.dtype == dtype
            tolerance = 10 * np.finfo(dtype).eps * w
            np.testing.assert_allclose(radius, expected, rtol=0, atol=tolerance)


def test_interval_too_narrow_is_empty():
    from spaceodyssey.gaussian_profile import _interval_radius

    with jax.enable_x64():
        radius, feasible = _interval_radius(0.1, 1.0, 0.05)
        assert not bool(feasible)
        assert float(radius) == 0


def test_public_registered_certifier():
    import spaceodyssey
    from spaceodyssey.registry import REGISTRY

    assert hasattr(spaceodyssey, "GaussianProfileCertifier")
    assert (
        REGISTRY.get("certification", "gaussian_profile").cls
        is spaceodyssey.GaussianProfileCertifier
    )
    assert REGISTRY.get("certification", "gaussian_profile").info.fidelity == ""


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


def profile_for(claims, *, rule="separate"):
    return Profile(
        "experiment",
        tuple(claims),
        0.05,
        support_rule=rule,
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
@pytest.mark.slow
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

        import spaceodyssey.gaussian_profile as gaussian_profile

        monkeypatch.setattr(gaussian_profile, "cho_solve", forbidden)
        monkeypatch.setattr(Nonlinear, "expected_data", forbidden)
        with pytest.raises(UnsupportedProfile, match=r"nonlinear.*quantity"):
            GaussianProfileCertifier(profile, Nonlinear())


@pytest.mark.parametrize("rule", ["separate", "split"])
@pytest.mark.slow
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


def _interval_support_probability(sd, width):
    """Positive support probability for a centered two-edge flat-prior claim."""
    from spaceodyssey.gaussian_profile import _normal_mass, _regions

    lower, upper = _regions(
        -width / 2, width / 2, sd, 0.05, jnp.asarray(True), jnp.asarray(True)
    )
    return _normal_mass(0.05, sd, lower[0], upper[0])


def _central_gradient(function, sd, width):
    values = np.array([sd, width], dtype=np.float64)
    gradient = []
    for i in range(2):
        delta = np.zeros(2)
        delta[i] = values[i] * 1e-5
        gradient.append(
            (float(function(*(values + delta))) - float(function(*(values - delta))))
            / (2 * delta[i])
        )
    return gradient


@pytest.mark.parametrize("sd, width", [(0.3, 2.0), (0.2, 1.0), (0.08, 0.5)])
@pytest.mark.parametrize("positive", [True, False])
def test_interval_radius_gradient_matches_finite_difference(sd, width, positive):
    from spaceodyssey.gaussian_profile import _interval_radius

    with jax.enable_x64():

        def radius(sd, width):
            return _interval_radius(width, sd, 0.05, positive=positive)[0]

        gradient = jax.grad(radius, argnums=(0, 1))(sd, width)
        np.testing.assert_allclose(
            gradient, _central_gradient(radius, sd, width), rtol=1e-5, atol=0
        )


@pytest.mark.parametrize("sd, width", [(0.3, 2.0), (0.2, 1.0), (0.08, 0.5)])
def test_interval_support_gradient_matches_finite_difference(sd, width):
    with jax.enable_x64():
        gradient = jax.grad(_interval_support_probability, argnums=(0, 1))(sd, width)
        np.testing.assert_allclose(
            gradient,
            _central_gradient(_interval_support_probability, sd, width),
            rtol=1e-5,
            atol=0,
        )


@pytest.mark.parametrize("positive", [True, False])
def test_interval_gradients_jit_and_vmap(positive):
    from spaceodyssey.gaussian_profile import _interval_radius

    with jax.enable_x64():
        scales = jnp.array([0.3, 0.2, 0.08])
        widths = jnp.array([2.0, 1.0, 0.5])

        def radius(sd, width):
            return _interval_radius(width, sd, 0.05, positive=positive)[0]

        for function in (radius, _interval_support_probability):
            gradient = jax.grad(function, argnums=(0, 1))
            expected = np.array(
                [
                    _central_gradient(function, float(sd), float(width))
                    for sd, width in zip(scales, widths, strict=True)
                ]
            ).T
            np.testing.assert_allclose(
                jax.jit(gradient)(scales[0], widths[0]),
                expected[:, 0],
                rtol=1e-5,
                atol=0,
            )
            for batched in (jax.vmap(gradient), jax.jit(jax.vmap(gradient))):
                np.testing.assert_allclose(
                    batched(scales, widths), expected, rtol=1e-5, atol=0
                )


@pytest.mark.parametrize("positive", [True, False])
def test_infeasible_interval_tangents_do_not_leak(positive):
    from spaceodyssey.gaussian_profile import _interval_radius

    with jax.enable_x64():
        scales = jnp.array([0.3, 1.0, 0.2])
        widths = jnp.array([2.0, 0.001, 1.0])

        def radius(sd, width):
            return _interval_radius(width, sd, 0.05, positive=positive)[0]

        gradient = jax.grad(radius, argnums=(0, 1))
        batched = jax.jit(jax.vmap(gradient))(scales, widths)
        assert np.all(np.isfinite(batched))
        np.testing.assert_array_equal(np.asarray(batched)[:, 1], 0.0)
        for i in (0, 2):
            np.testing.assert_allclose(
                np.asarray(batched)[:, i], gradient(scales[i], widths[i]), rtol=1e-12
            )

        # Exercise the array-valued root and its reverse-mode tangent solve too.
        array_gradient = jax.jit(
            jax.grad(
                lambda sd, width: jnp.sum(
                    _interval_radius(width, sd, 0.05, positive=positive)[0]
                ),
                argnums=(0, 1),
            )
        )(scales, widths)
        np.testing.assert_allclose(array_gradient, batched, rtol=1e-12, atol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("positive", [True, False])
def test_interval_radius_forward_bits_unchanged(dtype, positive):
    from jax.scipy.special import ndtr, ndtri

    from spaceodyssey.gaussian_profile import _interval_radius

    @jax.jit
    def reference(width, sd, alpha):
        half = width / 2
        high = half + sd * (-ndtri(alpha / 2) + 1)

        def within_boundary(mean):
            if positive:
                return ndtr((mean - half) / sd) + ndtr((-half - mean) / sd) <= alpha
            return ndtr((half - mean) / sd) - ndtr((-half - mean) / sd) >= alpha

        def step(_, bracket):
            low, high = bracket
            middle = (low + high) / 2
            within = within_boundary(middle)
            return jnp.where(within, middle, low), jnp.where(within, high, middle)

        low, high = jax.lax.fori_loop(0, 80, step, (jnp.zeros_like(high), high))
        feasible = within_boundary(jnp.zeros_like(high))
        return jnp.where(feasible, (low + high) / 2, jnp.zeros_like(high)), feasible

    with jax.enable_x64():
        widths = jnp.array([2.0, 1.0, 0.01, 0.001], dtype=dtype)
        scales = jnp.array([0.3, 0.2, 0.0001, 1.0], dtype=dtype)
        for alpha in (0.05, 1e-9):
            actual, feasible = _interval_radius(
                widths, scales, dtype(alpha), positive=positive
            )
            expected, expected_feasible = reference(widths, scales, dtype(alpha))
            bits = np.uint64 if dtype == np.float64 else np.uint32
            np.testing.assert_array_equal(
                np.asarray(actual).view(bits), np.asarray(expected).view(bits)
            )
            np.testing.assert_array_equal(feasible, expected_feasible)


def test_interval_bisection_has_fixed_iteration_count():
    from spaceodyssey.gaussian_profile import _INTERVAL_BISECTIONS, _interval_radius

    assert _INTERVAL_BISECTIONS == 80

    def scan_lengths(jaxpr):
        lengths = []
        for equation in jaxpr.eqns:
            if equation.primitive.name == "scan":
                lengths.append(equation.params["length"])
            for value in equation.params.values():
                if hasattr(value, "jaxpr"):
                    lengths.extend(scan_lengths(value.jaxpr))
        return lengths

    lengths = []
    for width, sd, alpha in ((2.0, 0.3, 0.05), (0.01, 0.0001, 1e-9)):
        program = jax.make_jaxpr(_interval_radius)(width, sd, alpha)
        lengths.append(scan_lengths(program.jaxpr))
        # ndtri also contains fixed scans for its polynomial evaluation.
        assert _INTERVAL_BISECTIONS in lengths[-1]
    assert lengths[0] == lengths[1]


def test_orthogonal_float32_design_uses_closed_form():
    from spaceodyssey import GaussianProfileCertifier

    with jax.enable_x64(False):
        matrix = np.array([[0.6, -0.8], [0.8, 0.6]], dtype=np.float32)
        assert (matrix.T @ matrix)[0, 1] != 0  # Rounding breaks exact diagonality.
        model = GaussianModel(matrix)
        profile = profile_for(
            [
                Claim(name, (Edge(name, 0.0, "above"),), "positive")
                for name in ("x", "y")
            ]
        )
        result = GaussianProfileCertifier(profile, model).integrate(
            experiment(model, [[0.1, 0.2]])
        )
        assert result.fidelity == "closed_form"
        assert bool(result.in_range[0])
        check_outcome(result, profile)


def test_nearly_collinear_flat_prior_never_returns_nan():
    from spaceodyssey import GaussianProfileCertifier

    with jax.enable_x64(False):
        model = GaussianModel(jnp.array([[1.0, 1.0], [1.0, 1.0001]]))
        profile = profile_for(
            [
                Claim(name, (Edge(name, 0.0, "above"),), "positive")
                for name in ("x", "y")
            ]
        )
        result = GaussianProfileCertifier(profile, model, flat_prior=True).integrate(
            experiment(model, [[0.1, 0.2]])
        )
        for name in (
            "p_pos",
            "p_neg",
            "p_wrong",
            "p_neg_wrong",
            "claim_p_pos",
            "claim_p_neg",
        ):
            assert np.all(np.isfinite(getattr(result, name)))
        if bool(result.in_range[0]):
            check_outcome(result, profile)
        else:
            assert not np.any(result.claim_in_range)
            np.testing.assert_array_equal(result.p_pos, 0.0)
            np.testing.assert_array_equal(result.p_neg, 0.0)


def test_nan_rectangle_probability_clears_in_range(monkeypatch):
    import spaceodyssey.gaussian_profile as gaussian_profile
    from spaceodyssey import GaussianProfileCertifier

    class EagerProbabilities(GaussianProfileCertifier):
        _probabilities = GaussianProfileCertifier._probabilities.__wrapped__

    monkeypatch.setattr(
        gaussian_profile,
        "normal_rectangle_probability",
        lambda *args, **kwargs: jnp.nan,
    )
    model = GaussianModel(jnp.array([[1.0, 0.5], [0.0, 1.0]]))
    profile = profile_for(
        [Claim(name, (Edge(name, 0.0, "above"),), "positive") for name in ("x", "y")]
    )
    result = EagerProbabilities(profile, model).integrate(
        experiment(model, [[0.1, 0.2]])
    )
    assert not bool(result.in_range[0])
    for name in (
        "p_pos",
        "p_neg",
        "p_wrong",
        "p_neg_wrong",
        "claim_p_pos",
        "claim_p_neg",
    ):
        np.testing.assert_array_equal(getattr(result, name), 0.0)


def test_proportional_float32_quantity_coefficients_are_accepted():
    from spaceodyssey import GaussianProfileCertifier

    class Proportional(GaussianModel):
        def quantities(self, theta, context):
            row = jnp.array([0.7, -0.3], dtype=theta.dtype)
            scaled = jnp.array([0.7 * 1.3, -0.3 * 1.3], dtype=theta.dtype)
            return {"x": theta @ row, "y": theta @ scaled}

    with jax.enable_x64(False):
        model = Proportional()
        profile = profile_for(
            [
                Claim(
                    "band",
                    (Edge("x", -1.0, "above"), Edge("y", 1.3, "below")),
                    "positive",
                )
            ]
        )
        result = GaussianProfileCertifier(profile, model).integrate(
            experiment(model, [[0.1, 0.2]])
        )
        assert bool(result.in_range[0])
        check_outcome(result, profile)


@pytest.mark.parametrize("opt_in", [False, True])
def test_linear_spectrum_requires_explicit_opt_in(opt_in, monkeypatch):
    from spaceodyssey import GaussianProfileCertifier

    class DeclaredModel(GaussianModel):
        LINEAR_GAUSSIAN_SPECTRUM: ClassVar[bool] = opt_in

    model = DeclaredModel()
    assert hasattr(model, "spectral_matrix")
    profile = profile_for([Claim("x", (Edge("x", 0.0, "above"),), "positive")])
    if opt_in:
        result = GaussianProfileCertifier(profile, model).integrate(
            experiment(model, [[0.1, 0.2]])
        )
        assert bool(result.in_range[0])
    else:

        def forbidden(*args, **kwargs):
            pytest.fail("missing opt-in must be rejected before model work")

        monkeypatch.setattr(DeclaredModel, "quantities", forbidden)
        with pytest.raises(UnsupportedProfile, match="explicitly opt in"):
            GaussianProfileCertifier(profile, model)


@pytest.mark.parametrize("restriction", ["five_claims", "dependent_claims"])
def test_claim_scope_rejected_before_posterior_work(restriction, monkeypatch):
    import spaceodyssey.gaussian_profile as gaussian_profile
    from spaceodyssey import GaussianProfileCertifier

    model = GaussianModel()
    count = 5 if restriction == "five_claims" else 2
    profile = profile_for(
        [
            Claim(f"claim_{i}", (Edge("x", float(i), "above"),), "positive")
            for i in range(count)
        ]
    )

    def forbidden(*args, **kwargs):
        pytest.fail("scope rejection must precede posterior or observation work")

    monkeypatch.setattr(gaussian_profile, "cho_solve", forbidden)
    monkeypatch.setattr(GaussianModel, "expected_data", forbidden)
    monkeypatch.setattr(GaussianModel, "simulate", forbidden)
    monkeypatch.setattr(GaussianModel, "quantities", forbidden)
    message = (
        "at most four claims"
        if restriction == "five_claims"
        else "linearly independent"
    )
    with pytest.raises(UnsupportedProfile, match=message):
        GaussianProfileCertifier(profile, model)


def test_distinct_dependent_quantities_rejected_before_posterior(monkeypatch):
    import spaceodyssey.gaussian_profile as gaussian_profile
    from spaceodyssey import GaussianProfileCertifier

    class DependentModel(GaussianModel):
        def quantities(self, theta, context):
            return {"x": theta[..., 0], "y": 2 * theta[..., 0]}

    model = DependentModel()
    profile = profile_for(
        [Claim(name, (Edge(name, 0.0, "above"),), "positive") for name in ("x", "y")]
    )

    def forbidden(*args, **kwargs):
        pytest.fail("dependent quantities must be rejected before posterior work")

    monkeypatch.setattr(gaussian_profile, "_posterior_moments", forbidden)
    with pytest.raises(UnsupportedProfile, match="linearly independent"):
        GaussianProfileCertifier(profile, model)


def test_mixed_quantity_covariance_projection_matches_numpy(monkeypatch):
    import spaceodyssey.gaussian_profile as gaussian_profile
    from spaceodyssey import GaussianProfileCertifier

    matrix = np.array(
        [[1.0, 0.4, -0.2], [0.2, 1.0, 0.3], [0.3, -0.2, 1.0], [0.5, 0.4, 0.2]]
    )
    q = np.array([[1.2, -0.4, 0.3], [0.2, 0.8, -0.5], [-0.3, 0.4, 1.1]])
    captured = []

    class MixedModel(GaussianModel):
        param_names: ClassVar = ("x", "y", "z")
        param_units: ClassVar = ("1", "1", "1")
        quantity_names: ClassVar = param_names

        def __init__(self):
            super().__init__(matrix)
            self.prior = Normal(jnp.array([0.1, -0.2, 0.3]), jnp.array([1.5, 0.8, 2.0]))
            self.prior_sets = (PriorSet((self.prior,), "normal@1", ("nominal",), 0),)

        def quantities(self, theta, context):
            values = theta @ jnp.asarray(q).T
            return {name: values[..., i] for i, name in enumerate(self.quantity_names)}

    original = gaussian_profile._posterior_moments

    def capture(*args):
        moments = original(*args)
        captured.append(tuple(np.asarray(value) for value in moments))
        return moments

    monkeypatch.setattr(gaussian_profile, "_posterior_moments", capture)
    with jax.enable_x64():
        model = MixedModel()
        profile = profile_for(
            [
                Claim(name, (Edge(name, 0.0, "above"),), "positive")
                for name in model.quantity_names
            ]
        )
        theta = np.array([[0.15, -0.1, 0.2]])
        noise_sigma = np.array([[0.2, 0.3, 0.25, 0.4]])
        factors = np.array([[[0.05], [-0.08], [0.1], [0.04]]])
        predictive = experiment(
            model, theta, noise_sigma=noise_sigma, noise_corr_factors=factors
        )
        result = GaussianProfileCertifier(profile, model).integrate(predictive)
        noise = np.diag(noise_sigma[0] ** 2) + factors[0] @ factors[0].T
        g = matrix.T @ np.linalg.solve(noise, matrix)
        prior_precision = np.diag(1 / np.asarray(model.prior.scale) ** 2)
        s = np.linalg.inv(g + prior_precision)
        mean, covariance, posterior_covariance = captured[0]
        np.testing.assert_allclose(
            covariance, q @ s @ g @ s.T @ q.T, rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(
            posterior_covariance, q @ s @ q.T, rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(
            mean,
            q @ s @ (g @ theta[0] + prior_precision @ np.asarray(model.prior.loc)),
            rtol=1e-12,
            atol=1e-14,
        )
        assert result.fidelity == "quadrature"
        check_outcome(result, profile)
