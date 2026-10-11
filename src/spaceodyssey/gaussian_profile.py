"""Posterior support regions for linear Gaussian profile experiments."""

import itertools
from functools import partial
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Literal
from jax.scipy.linalg import cho_solve
from jax.scipy.special import ndtr, ndtri
from photomancy.priors import Normal

from spaceodyssey.certify import AbstractProfileCertificate
from spaceodyssey.errors import UnsupportedProfile
from spaceodyssey.gaussian import normal_rectangle_probability
from spaceodyssey.plan import CertificationOutcome
from spaceodyssey.profile import Profile
from spaceodyssey.registry import SeamInfo, register
from spaceodyssey.truth import AbstractTruthModel, _plan_masks

_INTERVAL_BISECTIONS = 80


@partial(jax.jit, static_argnames=("positive",))
def _interval_radius(width, sd, alpha, *, positive=True):
    """Return a support radius and feasibility flag for an interval claim.

    Positive support requires outside mass <= alpha, evaluated directly on
    both tails without forming 1 - alpha. Negative support requires inside
    mass <= alpha beyond the returned radius. An infeasible positive region
    is empty; an infeasible negative boundary means all means qualify.
    The 80 fixed bisections are deterministic and JIT compatible, including
    for batched scales. The radius is differentiable through the implicit
    function theorem away from the feasibility boundary. Infeasible radii
    are constant zero, with finite zero tangents.
    """
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

    feasible = within_boundary(jnp.zeros_like(high))

    def residual(radius):
        if positive:
            mass = ndtr((radius - half) / sd) + ndtr((-half - radius) / sd)
        else:
            mass = ndtr((half - radius) / sd) - ndtr((-half - radius) / sd)
        # The zero sentinel for an empty interval solves r = 0 instead.
        return jnp.where(feasible, mass - alpha, radius)

    def solve(_, initial_guess):
        low, upper = jax.lax.fori_loop(
            0, _INTERVAL_BISECTIONS, step, (initial_guess, high)
        )
        return jnp.where(feasible, (low + upper) / 2, jnp.zeros_like(high))

    def tangent_solve(linear, tangent):
        # The Jacobian is diagonal for batched, independent scalar roots.
        slope = jnp.where(feasible, linear(jnp.ones_like(high)), 1.0)
        return jnp.where(feasible, tangent, 0.0) / slope

    radius = jax.lax.custom_root(residual, jnp.zeros_like(high), solve, tangent_solve)
    return radius, feasible


def _assert_affine(jaxpr, dependent):
    """Reject nonlinear parameter dependence from a traced quantity program."""
    dependencies = dict(zip(jaxpr.invars, dependent, strict=True))
    needed = set(v for v in jaxpr.outvars if not isinstance(v, Literal))
    live = []
    for equation in reversed(jaxpr.eqns):
        if any(v in needed for v in equation.outvars):
            live.append(equation)
            needed.update(v for v in equation.invars if not isinstance(v, Literal))
    linear = {
        "add",
        "sub",
        "neg",
        "slice",
        "squeeze",
        "reshape",
        "transpose",
        "broadcast_in_dim",
        "concatenate",
        "reduce_sum",
        "copy",
        "rev",
        "split",
    }
    for equation in reversed(live):
        inputs = [
            False if isinstance(v, Literal) else dependencies.get(v, False)
            for v in equation.invars
        ]
        name = equation.primitive.name
        if name in ("jit", "pjit"):
            outputs = _assert_affine(equation.params["jaxpr"].jaxpr, inputs)
        else:
            depends = any(inputs)
            safe = (
                not depends
                or name in linear
                or (
                    name == "convert_element_type"
                    and np.dtype(equation.params["new_dtype"]).kind == "f"
                    and np.dtype(equation.invars[0].aval.dtype).kind == "f"
                )
                or (name in ("mul", "dot_general") and sum(inputs) <= 1)
                or (name == "div" and not inputs[1])
                or (name == "integer_pow" and equation.params["y"] == 1)
                or (name == "gather" and not inputs[1])
            )
            if not safe:
                raise UnsupportedProfile(
                    f"nonlinear quantity: parameter-dependent {name!r} is unsupported"
                )
            outputs = [depends] * len(equation.outvars)
        dependencies.update(zip(equation.outvars, outputs, strict=True))
    return [
        False if isinstance(v, Literal) else dependencies.get(v, False)
        for v in jaxpr.outvars
    ]


def _posterior_moments(precision, prior_precision, rows, rhs):
    """Return claim means, sampling covariance, and posterior covariance.

    For parameter posterior covariance S and data precision G, the two claim
    covariances are Q S G S.T Q.T and Q S Q.T, respectively. Cholesky solves
    avoid forming the inverse of the posterior precision.
    """
    cholesky = jnp.linalg.cholesky(precision + prior_precision)
    projected = cho_solve((cholesky, True), rows.T)
    mean = rows @ cho_solve((cholesky, True), rhs)
    covariance = projected.T @ precision @ projected
    covariance = (covariance + covariance.T) / 2
    posterior_covariance = rows @ projected
    posterior_covariance = (posterior_covariance + posterior_covariance.T) / 2
    return mean, covariance, posterior_covariance


@register(
    "certification",
    "gaussian_profile",
    SeamInfo(
        operators={"integrate"},
        consumes=("spaceodyssey.PlanPredictive",),
        produces=("spaceodyssey.CertificationOutcome",),
        version="1",
        fidelity="",
        cost_hint="low",
    ),
)
class GaussianProfileCertifier(AbstractProfileCertificate):
    """A Gaussian toy for testing the profile integration interface.

    Certification is a posterior-risk condition; its probability is taken
    over noisy data at fixed truth. The model declares a full-column-rank
    ``spectral_matrix`` with response ``A @ theta`` plus a known offset, and
    explicitly opts in with ``LINEAR_GAUSSIAN_SPECTRUM = True``.
    Only spectral data are used. Noise is known, with covariance
    ``diag(noise_sigma**2) + F @ F.T``. A singleton independent Normal prior
    gives a Gaussian posterior. Explicit ``flat_prior=True`` instead uses
    the improper flat-prior experiment, ignoring the model's nominal prior;
    it is a support-only interface test, with no prior evidence or robustness
    interpretation. A bounded uniform posterior is not treated as Gaussian.

    Quantities may be arbitrary affine combinations of parameters. Their
    JAX programs are checked structurally before posterior computation.
    Each claim must reduce to an interval or half-line on one scalar;
    two quantities in one claim must have proportional coefficient vectors.
    More than four claims and linearly dependent claim scalars (singular
    claim covariance) raise ``UnsupportedProfile`` before posterior computation.

    Single thresholds and independent claim means have outcome fidelity
    ``closed_form``. Interval boundaries use 80 fixed bisections. Correlated
    claim means use fixed-node rectangle quadrature and report ``quadrature``.
    Standard errors are zero (there is no Monte Carlo); they do not estimate
    quadrature error. ``separate`` and ``split`` are supported. Stand-alone
    claim columns always use alpha, including for a split profile.

    Multi-claim ``joint`` is rejected by the base construction guard before
    any computation. Its region is convex for interval claims, but counting
    certification in either direction creates a union of such regions.
    A library point estimate for joint support needs a tabulated or approximate
    tier validated against the exact tier. This toy is not a production tier.

    ``integrate`` runs eagerly on the host with per-planet Python iteration.
    It cannot be jitted, vmapped, or differentiated with respect to the plan,
    truth, or model; no public integration argument supports differentiation.
    The numerical helpers can be traced separately; interval radii are
    differentiable through the implicit function theorem away from the
    feasibility boundary.

    Args:
        profile: Claims, support rule, and risk tolerance.
        truth_model: Linear spectral model supplying named quantities.
        flat_prior: Explicitly select the support-only flat-prior experiment.
        nodes: Fixed quadrature nodes per integration dimension. Each rectangle
            costs nodes**(D - 1) evaluations for D claims, with up to 81 counted
            rectangles plus one for the negative event per planet.
    """

    profile: Profile = eqx.field(static=True)
    truth_model: AbstractTruthModel
    flat_prior: bool = eqx.field(static=True, default=False)
    nodes: int = eqx.field(static=True, default=32)
    SUPPORT_RULES: ClassVar = frozenset({"joint", "separate", "split"})
    SUPPORTED_QUANTITIES: ClassVar = None

    @classmethod
    def check_profile_support(cls, profile, truth_model):
        """Reject unsupported profile geometry before model computation."""
        if profile.support_rule == "joint" and len(profile.claims) > 1:
            raise UnsupportedProfile(
                "joint with multiple claims needs a tabulated or approximate "
                "tier validated against the exact tier"
            )
        if len(profile.claims) > 4:
            raise UnsupportedProfile("Gaussian rectangles support at most four claims")
        scalar_names = [claim.edges[0].quantity for claim in profile.claims]
        if len(set(scalar_names)) != len(scalar_names):
            raise UnsupportedProfile(
                "claim scalars must be linearly independent for rectangle quadrature"
            )
        if getattr(truth_model, "LINEAR_GAUSSIAN_SPECTRUM", False) is not True:
            raise UnsupportedProfile(
                "linear Gaussian spectral model must explicitly opt in with "
                "LINEAR_GAUSSIAN_SPECTRUM = True"
            )
        if len(truth_model.prior_set(profile.prior_set_id).members) != 1:
            raise UnsupportedProfile("Gaussian toy requires a singleton prior set")

    def __check_init__(self):
        """Validate the Gaussian experiment and trace quantity linearity."""
        if type(self.nodes) is not int or self.nodes < 1:
            raise ValueError("nodes must be a positive integer")
        if not self.flat_prior and not isinstance(self.truth_model.prior, Normal):
            raise UnsupportedProfile(
                "Gaussian posterior requires a Normal prior or explicit flat_prior=True"
            )
        model = self.truth_model
        d = len(model.param_names)
        context = {
            name: jax.ShapeDtypeStruct((), float) for name in model.context_names
        }
        for name in self.profile.quantities:
            program = jax.make_jaxpr(
                lambda theta, ctx, name=name: model.quantities(theta, ctx)[name]
            )(jax.ShapeDtypeStruct((d,), float), context)
            _assert_affine(program.jaxpr, [True] + [False] * len(context))
        a = np.asarray(model.spectral_matrix)
        if a.ndim != 2 or a.shape[1] != d or not np.all(np.isfinite(a)):
            raise ValueError("spectral_matrix must be finite with shape (B, D)")
        if np.linalg.matrix_rank(a) != d:
            raise UnsupportedProfile("spectral_matrix must have full column rank")
        # Check claim independence before computing any posterior. Geometry is
        # also checked at each planet's actual context during integration.
        self._claim_geometry({name: jnp.asarray(0.0) for name in model.context_names})
        if not self.flat_prior:
            prior = model.prior
            if not np.all(np.isfinite(prior.loc)) or not np.all(
                np.isfinite(prior.scale) & (np.asarray(prior.scale) > 0)
            ):
                raise ValueError(
                    "Normal prior parameters must be finite with positive scale"
                )

    def _claim_geometry(self, context):
        """Return coefficient rows and bounds for independent affine claims."""
        model = self.truth_model
        zero = jnp.zeros(
            len(model.param_names), dtype=jnp.result_type(model.spectral_matrix, 0.0)
        )
        values = model.quantities(zero, context)
        gradients = jax.jacfwd(model.quantities)(zero, context)
        rows, lows, highs = [], [], []
        for claim in self.profile.claims:
            first = claim.edges[0].quantity
            row = gradients[first]
            norm2 = jnp.dot(row, row)
            if float(norm2) == 0:
                raise UnsupportedProfile(f"quantity {first!r} is constant")
            low, high = -jnp.inf, jnp.inf
            for edge in claim.edges:
                coefficient = gradients[edge.quantity]
                factor = jnp.dot(coefficient, row) / norm2
                rtol = 100 * jnp.finfo(row.dtype).eps
                atol = rtol * float(jnp.linalg.norm(row))
                if (
                    not np.allclose(coefficient, factor * row, rtol=rtol, atol=atol)
                    or float(factor) == 0
                ):
                    raise UnsupportedProfile(
                        f"claim {claim.name!r} needs a multidimensional posterior "
                        "region; its edges must reduce to one linear scalar"
                    )
                boundary = (edge.threshold - values[edge.quantity]) / factor
                above = (edge.side == "above") == (float(factor) > 0)
                if above:
                    low = jnp.maximum(low, boundary)
                else:
                    high = jnp.minimum(high, boundary)
            rows.append(row)
            lows.append(low)
            highs.append(high)
        c = jnp.stack(rows)
        if np.linalg.matrix_rank(np.asarray(c)) != len(rows):
            raise UnsupportedProfile(
                "claim scalars must be linearly independent for rectangle quadrature"
            )
        return c, jnp.asarray(lows), jnp.asarray(highs)

    def integrate(self, predictive):
        """Return Gaussian probabilities with claim columns and profile identity.

        Runs eagerly on the host, validates inputs and geometry, and records
        fidelity. It cannot be jitted, vmapped, or differentiated in the plan,
        truth, or model; no argument supports differentiation through this
        method. Numerical helpers are JIT compatible when called separately;
        interval radii use implicit-function derivatives away from the
        feasibility boundary.
        """
        self.validate_predictive(predictive)
        model, plan, truth = self.truth_model, predictive.plan, predictive.truth
        a = model.spectral_matrix
        if plan.noise_sigma.shape[1] != a.shape[0]:
            raise ValueError("plan B must match spectral_matrix")
        available, _ = _plan_masks(plan)
        results = []
        diagonal = True
        for i in range(plan.n_planets):
            context = {name: value[i] for name, value in truth.context.items()}
            c, lower, upper = self._claim_geometry(context)
            sigma = plan.noise_sigma[i]
            valid = bool(
                plan.noise_known[i]
                & jnp.all(jnp.isfinite(sigma) & (sigma > 0))
                & jnp.all(jnp.isfinite(truth.theta[i]))
            )
            noise = jnp.diag(
                jnp.where(jnp.isfinite(sigma) & (sigma > 0), sigma, 1.0) ** 2
            )
            if plan.noise_corr_factors is not None:
                f = plan.noise_corr_factors[i]
                valid = valid and bool(jnp.all(jnp.isfinite(f)))
                f = jnp.nan_to_num(f)
                noise = noise + f @ f.T
            precision = a.T @ jnp.linalg.solve(noise, a)
            prior_precision = (
                jnp.zeros_like(precision)
                if self.flat_prior
                else jnp.diag(1 / model.prior.scale**2)
            )
            rhs = precision @ jnp.nan_to_num(truth.theta[i])
            if not self.flat_prior:
                rhs = rhs + prior_precision @ model.prior.loc
            mean, covariance, posterior_covariance = _posterior_moments(
                precision, prior_precision, c, rhs
            )
            sd = jnp.sqrt(jnp.diag(posterior_covariance))
            valid = valid and bool(
                jnp.all(jnp.isfinite(mean))
                & jnp.all(jnp.isfinite(covariance))
                & jnp.all(jnp.isfinite(sd) & (sd > 0))
                & jnp.all(jnp.diag(covariance) > 0)
            )
            scale = np.sqrt(np.diag(np.asarray(covariance)))
            correlation = np.asarray(covariance) / (scale[:, None] * scale[None, :])
            off_diagonal = correlation - np.eye(len(correlation))
            is_diagonal = np.all(
                np.abs(off_diagonal) <= 100 * np.finfo(correlation.dtype).eps
            )
            diagonal = diagonal and is_diagonal
            if self.flat_prior:
                evidence_pos = evidence_neg = jnp.ones(
                    len(self.profile.claims), dtype=bool
                )
            else:
                prior_sd = jnp.sqrt(jnp.sum((c * model.prior.scale) ** 2, axis=1))
                prior_mean = c @ model.prior.loc
                prior_mass = ndtr((upper - prior_mean) / prior_sd) - ndtr(
                    (lower - prior_mean) / prior_sd
                )
                evidence_pos = 1 - prior_mass > self.profile.alpha
                evidence_neg = prior_mass > self.profile.alpha
            row = self._probabilities(
                mean,
                covariance,
                sd,
                lower,
                upper,
                evidence_pos,
                evidence_neg,
                bool(is_diagonal),
                truth.theta[i],
                context,
            )
            valid = valid and all(bool(jnp.all(jnp.isfinite(value))) for value in row)
            active = bool(available[i]) and valid
            results.append(
                (
                    tuple(jnp.where(active, value, 0.0) for value in row),
                    not bool(available[i]) or valid,
                )
            )
        probabilities = [jnp.stack([row[0][k] for row in results]) for k in range(6)]
        pos, neg, wrong, neg_wrong, claim_pos, claim_neg = probabilities
        quantities = model.quantities(truth.theta, truth.context)
        claim_truth = self.profile.claim_truth(quantities)
        in_range = jnp.asarray([row[1] for row in results])
        zeros = jnp.zeros_like(pos)
        return CertificationOutcome(
            p_pos=pos,
            p_neg=neg,
            p_wrong=wrong,
            p_neg_wrong=neg_wrong,
            p_pos_se=zeros,
            p_neg_se=zeros,
            p_wrong_se=zeros,
            p_neg_wrong_se=zeros,
            truth_positive=self.profile.truth_positive(quantities),
            in_range=in_range,
            claim_p_pos=claim_pos,
            claim_p_neg=claim_neg,
            claim_p_wrong=jnp.where(claim_truth, 0.0, claim_pos),
            claim_p_neg_wrong=jnp.where(claim_truth, claim_neg, 0.0),
            claim_truth_positive=claim_truth,
            claim_in_range=jnp.broadcast_to(in_range[:, None], claim_truth.shape),
            claim_names=self.profile.claim_names,
            profile_hash=self.profile.content_hash(),
            fidelity="closed_form" if diagonal else "quadrature",
        )

    @eqx.filter_jit
    def _probabilities(
        self,
        mean,
        covariance,
        sd,
        lower,
        upper,
        evidence_pos,
        evidence_neg,
        diagonal,
        theta,
        context,
    ):
        count = len(self.profile.claims)
        alpha = self.profile.alpha
        level = alpha / count if self.profile.support_rule == "split" else alpha
        low, high = _regions(lower, upper, sd, level, evidence_pos, evidence_neg)
        stand_low, stand_high = _regions(
            lower, upper, sd, alpha, evidence_pos, evidence_neg
        )
        scale = jnp.sqrt(jnp.diag(covariance))
        marginal = _normal_mass(mean[:, None], scale[:, None], stand_low, stand_high)
        claim_pos, claim_neg = marginal[:, 0], jnp.sum(marginal[:, 1:], -1)
        claim_truth = self.profile.claim_truth(
            self.truth_model.quantities(theta, context)
        )
        positive = [claim.counted == "positive" for claim in self.profile.claims]

        def rectangle(lo, hi):
            if diagonal:
                return jnp.prod(_normal_mass(mean, scale, lo, hi))
            return normal_rectangle_probability(
                mean, covariance, lo, hi, nodes=self.nodes
            )

        pos, wrong = jnp.asarray(0.0), jnp.asarray(0.0)
        choices = [(0,) if required else (0, 1, 2) for required in positive]
        for states in itertools.product(*choices):
            idx = jnp.asarray(states)
            mass = rectangle(low[jnp.arange(count), idx], high[jnp.arange(count), idx])
            pos = pos + mass
            incorrect = jnp.any((idx == 0) != claim_truth)
            wrong = wrong + jnp.where(incorrect, mass, 0.0)
        # No negative report is a connected interval between the two tails.
        no_neg_low = jnp.where(jnp.asarray(positive), high[:, 1], -jnp.inf)
        no_neg_high = jnp.where(jnp.asarray(positive), low[:, 2], jnp.inf)
        neg = (
            jnp.clip(1 - rectangle(no_neg_low, no_neg_high), 0.0, 1.0)
            if any(positive)
            else jnp.asarray(0.0)
        )
        truth_positive = jnp.all(claim_truth | ~jnp.asarray(positive))
        return (
            jnp.clip(pos, 0.0, 1.0),
            neg,
            jnp.clip(wrong, 0.0, pos),
            jnp.where(truth_positive, neg, 0.0),
            claim_pos,
            claim_neg,
        )


def _normal_mass(mean, sd, lower, upper):
    """Return univariate Gaussian interval mass, reflecting upper tails."""
    reflect = lower > mean
    low = jnp.where(reflect, mean - upper, lower - mean) / sd
    high = jnp.where(reflect, mean - lower, upper - mean) / sd
    return jnp.maximum(ndtr(high) - ndtr(low), 0.0)


def _regions(lower, upper, sd, alpha, evidence_pos, evidence_neg):
    """Return positive, left-negative, and right-negative support bounds."""
    z = -ndtri(alpha)
    finite_low, finite_high = jnp.isfinite(lower), jnp.isfinite(upper)
    interval = finite_low & finite_high
    width = jnp.where(interval, upper - lower, 1.0)
    center = (
        jnp.where(finite_low, lower, 0.0) + jnp.where(finite_high, upper, 0.0)
    ) / 2
    radius_pos, feasible_pos = _interval_radius(width, sd, alpha)
    radius_neg, _ = _interval_radius(width, sd, alpha, positive=False)
    pos_low = jnp.where(interval, center - radius_pos, lower + z * sd)
    pos_high = jnp.where(interval, center + radius_pos, upper - z * sd)
    neg_left = jnp.where(
        interval, center - jnp.maximum(radius_neg, 0.0), lower - z * sd
    )
    neg_right = jnp.where(
        interval, center + jnp.maximum(radius_neg, 0.0), upper + z * sd
    )
    feasible = ~interval | feasible_pos
    pos_low = jnp.where(evidence_pos & feasible, pos_low, jnp.inf)
    pos_high = jnp.where(evidence_pos & feasible, pos_high, -jnp.inf)
    neg_left = jnp.where(evidence_neg, neg_left, -jnp.inf)
    neg_right = jnp.where(evidence_neg, neg_right, jnp.inf)
    return (
        jnp.stack([pos_low, jnp.full_like(lower, -jnp.inf), neg_right], -1),
        jnp.stack([pos_high, neg_left, jnp.full_like(upper, jnp.inf)], -1),
    )
