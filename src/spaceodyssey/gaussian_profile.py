"""Posterior support regions for linear Gaussian profile experiments."""

import itertools
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr, ndtri
from photomancy.priors import Normal

from spaceodyssey.certify import AbstractProfileCertificate
from spaceodyssey.errors import UnsupportedProfile
from spaceodyssey.gaussian import normal_rectangle_probability
from spaceodyssey.plan import CertificationOutcome
from spaceodyssey.profile import Profile
from spaceodyssey.registry import SeamInfo, register
from spaceodyssey.truth import AbstractTruthModel, _plan_masks


@jax.jit
def _interval_radius(width, sd, probability):
    """Return the half-width of means with interval mass >= probability.

    A negative result denotes an empty region. The 80 fixed bisections are
    deterministic and JIT compatible, including for batched scales.
    """
    half = width / 2
    high = half + sd * (-ndtri(probability / 2) + 1)

    def step(_, bracket):
        low, high = bracket
        middle = (low + high) / 2
        mass = ndtr((half - middle) / sd) - ndtr((-half - middle) / sd)
        return jnp.where(mass >= probability, middle, low), jnp.where(
            mass >= probability, high, middle
        )

    low, high = jax.lax.fori_loop(0, 80, step, (jnp.zeros_like(high), high))
    feasible = 2 * ndtr(half / sd) - 1 >= probability
    return jnp.where(feasible, (low + high) / 2, -jnp.ones_like(high))


def _assert_affine(jaxpr, dependent):
    """Reject nonlinear parameter dependence from a traced quantity program."""
    from jax.extend.core import Literal

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


@register(
    "certification",
    "gaussian_profile",
    SeamInfo(
        operators={"integrate"},
        consumes=("spaceodyssey.PlanPredictive",),
        produces=("spaceodyssey.CertificationOutcome",),
        version="1",
        fidelity="closed_form",
        cost_hint="low",
    ),
)
class GaussianProfileCertifier(AbstractProfileCertificate):
    """A Gaussian toy for testing the profile integration interface.

    Certification is a posterior-risk condition; its probability is taken
    over noisy data at fixed truth. The model declares a full-column-rank
    ``spectral_matrix`` with response ``A @ theta`` plus a known offset.
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
    Up to four linearly independent claim scalars are supported.

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

    Args:
        profile: Claims, support rule, and risk tolerance.
        truth_model: Linear spectral model supplying named quantities.
        flat_prior: Explicitly select the support-only flat-prior experiment.
        nodes: Fixed quadrature nodes per integration dimension.
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
        if not hasattr(truth_model, "spectral_matrix"):
            raise UnsupportedProfile("linear spectral_matrix declaration is required")
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
        if not self.flat_prior:
            prior = model.prior
            if not np.all(np.isfinite(prior.loc)) or not np.all(
                np.isfinite(prior.scale) & (np.asarray(prior.scale) > 0)
            ):
                raise ValueError(
                    "Normal prior parameters must be finite with positive scale"
                )

    def _claim_geometry(self, context):
        model = self.truth_model
        zero = jnp.zeros(len(model.param_names))
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
                if (
                    not np.allclose(coefficient, factor * row, rtol=1e-12, atol=1e-14)
                    or float(factor) == 0
                ):
                    raise UnsupportedProfile(
                        f"claim {claim.name!r} needs a multidimensional posterior "
                        "region; "
                        "its edges must reduce to one linear scalar"
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

        This host-side wrapper validates geometry and records fidelity; the
        scalar root solver and rectangle integrator are JIT compatible.
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
            posterior = jnp.linalg.inv(precision + prior_precision)
            rhs = precision @ jnp.nan_to_num(truth.theta[i])
            if not self.flat_prior:
                rhs = rhs + prior_precision @ model.prior.loc
            mean = c @ posterior @ rhs
            covariance = c @ posterior @ precision @ posterior @ c.T
            sd = jnp.sqrt(jnp.diag(c @ posterior @ c.T))
            is_diagonal = np.all(
                np.asarray(covariance) == np.diag(np.diag(np.asarray(covariance)))
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
    reflect = lower > mean
    low = jnp.where(reflect, mean - upper, lower - mean) / sd
    high = jnp.where(reflect, mean - lower, upper - mean) / sd
    return jnp.maximum(ndtr(high) - ndtr(low), 0.0)


def _regions(lower, upper, sd, alpha, evidence_pos, evidence_neg):
    z = -ndtri(alpha)
    finite_low, finite_high = jnp.isfinite(lower), jnp.isfinite(upper)
    interval = finite_low & finite_high
    width = jnp.where(interval, upper - lower, 1.0)
    center = (
        jnp.where(finite_low, lower, 0.0) + jnp.where(finite_high, upper, 0.0)
    ) / 2
    radius_pos = _interval_radius(width, sd, 1 - alpha)
    radius_neg = _interval_radius(width, sd, alpha)
    pos_low = jnp.where(interval, center - radius_pos, lower + z * sd)
    pos_high = jnp.where(interval, center + radius_pos, upper - z * sd)
    neg_left = jnp.where(
        interval, center - jnp.maximum(radius_neg, 0.0), lower - z * sd
    )
    neg_right = jnp.where(
        interval, center + jnp.maximum(radius_neg, 0.0), upper + z * sd
    )
    pos_low = jnp.where(evidence_pos, pos_low, jnp.inf)
    pos_high = jnp.where(evidence_pos, pos_high, -jnp.inf)
    neg_left = jnp.where(evidence_neg, neg_left, -jnp.inf)
    neg_right = jnp.where(evidence_neg, neg_right, jnp.inf)
    return (
        jnp.stack([pos_low, jnp.full_like(lower, -jnp.inf), neg_right], -1),
        jnp.stack([pos_high, neg_left, jnp.full_like(upper, jnp.inf)], -1),
    )
