"""Conformance suites that ship with the seams, not the implementations.

Every registry entry, in-tree or external, is expected to instantiate its
seam's suite in its own tests. This is what keeps an unordered, open registry
trustworthy: declared operators must be honest (an undeclared operator raises
:class:`spaceodyssey.errors.UnsupportedOperator`, a declared one works), sampled
moments must match declared moments, and outputs must match declared contract
types. Checks raise plain AssertionError so any test framework can host them.
"""

import warnings
from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from photomancy.priors import Uniform

from spaceodyssey.contracts import resolve_contract
from spaceodyssey.errors import UnsupportedOperator
from spaceodyssey.plan import CertificationOutcome, _check_values
from spaceodyssey.profile import Profile
from spaceodyssey.truth import AbstractTruthModel, ObservedData, PriorSet, _plan_masks


def check_predictive(pred, *, key, n: int = 4096) -> None:
    """Check a predictive distribution's sampling and moment honesty.

    Draws ``n`` samples, verifies they and their log-density are finite, and,
    when the predictive exposes moments, verifies the sample mean lies within
    six standard errors of the declared mean.

    Args:
        pred: The predictive to check.
        key: PRNG key for the draws.
        n: Number of samples.

    Raises:
        AssertionError: On any conformance violation.
    """
    samples = jax.vmap(pred.sample)(jr.split(key, n))
    if not bool(jnp.all(jnp.isfinite(samples))):
        raise AssertionError("predictive produced non-finite samples")
    log_prob = pred.log_prob(samples[0])
    if not bool(jnp.isfinite(log_prob)):
        raise AssertionError("log_prob of a drawn sample is not finite")
    try:
        mean, cov = pred.moments()
    except UnsupportedOperator:
        return
    standard_error = jnp.sqrt(jnp.clip(jnp.diag(cov), min=0.0) / n)
    tolerance = float(6.0 * jnp.max(standard_error)) + 1e-12
    np.testing.assert_allclose(
        np.asarray(samples.mean(axis=0)),
        np.asarray(mean),
        atol=tolerance,
        rtol=0.0,
        err_msg="sample mean disagrees with declared moments",
    )


def check_certificate(cert, info, datasets, *, moments=None, predictive=None):
    """Check a certificate implementation against its declared operators.

    Args:
        cert: The certificate implementation.
        info: Its registered SeamInfo (source of the declared operators).
        datasets: Realized datasets to fold in when "sample" is declared.
        moments: Predictive moments to propagate when "propagate" is declared.
        predictive: Predictive to integrate when "integrate" is declared.
            When the certificate returns a CertificationOutcome, the outcome
            contract is checked with :func:`check_outcome`.

    Returns:
        The certificate state after any declared-sample updates, or None for
        integrate-only certificates.

    Raises:
        AssertionError: When a declared operator raises UnsupportedOperator,
            an undeclared one fails to, or an integrated outcome breaks its
            contract.
    """
    if info.operators & {"sample", "propagate"}:
        state = cert.initial()
    else:
        _expect_unsupported(cert.initial, "initial")
        state = None
    if "sample" in info.operators:
        for dataset in datasets:
            try:
                state = cert.update(state, dataset)
            except UnsupportedOperator as err:
                raise AssertionError(
                    "certificate declares 'sample' but update() raised"
                ) from err
    else:
        _expect_unsupported(lambda: cert.update(state, datasets[0]), "update")
    if "propagate" in info.operators:
        try:
            cert.propagate(cert.initial(), moments)
        except UnsupportedOperator as err:
            raise AssertionError(
                "certificate declares 'propagate' but propagate() raised"
            ) from err
    else:
        _expect_unsupported(lambda: cert.propagate(state, moments), "propagate")
    if "integrate" in info.operators:
        try:
            result = cert.integrate(predictive)
        except UnsupportedOperator as err:
            raise AssertionError(
                "certificate declares 'integrate' but integrate() raised"
            ) from err
        if isinstance(result, CertificationOutcome):
            check_outcome(result)
    else:
        _expect_unsupported(lambda: cert.integrate(predictive), "integrate")
    return state


def check_outcome(
    outcome: CertificationOutcome, profile: Profile | None = None
) -> None:
    """Check a certification outcome against its value contract.

    Over the in-range planets: every probability lies in [0, 1], the
    positive and negative verdicts are exclusive (p_pos + p_neg <= 1), a
    wrong counted-direction certification is a positive one
    (p_wrong <= p_pos), a wrong negative is a negative one
    (p_neg_wrong <= p_neg), and standard errors are non-negative. All flags must
    be boolean and at least one planet must be in range, so the check is
    never vacuous. When a profile is supplied, its hash and ordered claim
    names must match, and the profile positive probability cannot exceed
    any positive-counted claim probability or either-direction claim sum.
    Claim bounds use planets in range for both the profile and the claim.

    Args:
        outcome: The outcome to check.
        profile: Optional profile whose identity and claim bounds are checked.

    Raises:
        AssertionError: On any contract violation.
    """
    if profile is not None:
        if outcome.profile_hash != profile.content_hash():
            raise AssertionError("profile_hash does not match profile")
        if outcome.claim_names != profile.claim_names:
            raise AssertionError("claim_names do not match profile")
        for k, claim in enumerate(profile.claims):
            bound = np.asarray(outcome.claim_p_pos[:, k])
            if claim.counted == "either":
                bound = bound + np.asarray(outcome.claim_p_neg[:, k])
            keep_claim = np.asarray(outcome.in_range & outcome.claim_in_range[:, k])
            if np.any(np.asarray(outcome.p_pos)[keep_claim] > bound[keep_claim] + 1e-6):
                raise AssertionError(f"p_pos exceeds claim bound for {claim.name}")
    if outcome.in_range.dtype != jnp.bool_:
        raise AssertionError("in_range must be boolean")
    if outcome.truth_positive.dtype != jnp.bool_:
        raise AssertionError("truth_positive must be boolean")
    keep = np.asarray(outcome.in_range)
    if not keep.any():
        raise AssertionError("no planet is in range; the outcome checks nothing")
    p_pos = np.asarray(outcome.p_pos)[keep]
    p_neg = np.asarray(outcome.p_neg)[keep]
    p_wrong = np.asarray(outcome.p_wrong)[keep]
    p_neg_wrong = np.asarray(outcome.p_neg_wrong)[keep]
    slack = 1e-6
    for name, value in (
        ("p_pos", p_pos),
        ("p_neg", p_neg),
        ("p_wrong", p_wrong),
        ("p_neg_wrong", p_neg_wrong),
    ):
        if not np.all((value >= 0.0) & (value <= 1.0)):
            raise AssertionError(f"{name} leaves [0, 1]: {value}")
    if not np.all(p_pos + p_neg <= 1.0 + slack):
        raise AssertionError("p_pos + p_neg exceeds one")
    if not np.all(p_wrong <= p_pos + slack):
        raise AssertionError("p_wrong exceeds p_pos")
    if not np.all(p_neg_wrong <= p_neg + slack):
        raise AssertionError("p_neg_wrong exceeds p_neg")
    for name in ("p_pos_se", "p_neg_se", "p_wrong_se", "p_neg_wrong_se"):
        value = np.asarray(getattr(outcome, name))[keep]
        if not np.all(value >= 0.0):
            raise AssertionError(f"{name} is negative: {value}")


def check_ports(instance_output, info) -> None:
    """Check that an implementation's output matches its declared produces.

    Args:
        instance_output: A real output object produced by the implementation.
        info: Its registered SeamInfo.

    Raises:
        AssertionError: When the output is no instance of any declared
            produced contract type.
    """
    if not info.produces:
        return
    produced_types = tuple(resolve_contract(name) for name in info.produces)
    if not isinstance(instance_output, produced_types):
        names = ", ".join(info.produces)
        raise AssertionError(
            f"output of type {type(instance_output).__name__} matches none of "
            f"the declared produced types: {names}"
        )


def _expect_unsupported(call, name: str) -> None:
    try:
        call()
    except UnsupportedOperator:
        return
    except Exception as err:
        raise AssertionError(
            f"undeclared operator '{name}' raised {type(err).__name__} "
            "instead of UnsupportedOperator"
        ) from err
    raise AssertionError(
        f"undeclared operator '{name}' did not raise UnsupportedOperator"
    )


class LinearGaussianTruthModel(AbstractTruthModel):
    """Linear measurements of named parameters with a uniform box prior.

    The default parameters are dimensionless ``x`` and ``y``. Subclasses may
    declare other parameter names, units, and matching quantity names. The
    spectrum is ``theta @ spectral_matrix.T``; photometry repeats the first
    parameter, astrometry repeats the first and last parameters, and RV is
    the last parameter. A known per-planet ``offset`` is added to every
    channel in that channel's units. Missing measurements are zero-filled.
    ``LINEAR_GAUSSIAN_SPECTRUM`` explicitly opts into Gaussian certification:
    the spectral response is affine in theta with the declared matrix, and
    the offset is known. Subclasses changing this contract must disable it.

    Args:
        low: Uniform lower bounds (D,), default -10 in every dimension.
        high: Uniform upper bounds (D,), default 10 in every dimension.
        spectral_matrix: Linear spectral response (B, D), default identity.
        prior_set_id: Identifier of the singleton nominal prior set.
    """

    model_id: ClassVar[str] = "linear-gaussian@1"
    param_names: ClassVar[tuple[str, ...]] = ("x", "y")
    param_units: ClassVar[tuple[str, ...]] = ("1", "1")
    context_names: ClassVar[tuple[str, ...]] = ("offset",)
    quantity_names: ClassVar[tuple[str, ...]] = param_names
    flux_reference: ClassVar[str] = "unit_flux"
    LINEAR_GAUSSIAN_SPECTRUM: ClassVar[bool] = True
    prior: Uniform
    prior_sets: tuple[PriorSet, ...]
    spectral_matrix: jax.Array

    def __init__(
        self, low=None, high=None, spectral_matrix=None, prior_set_id="uniform@1"
    ):
        """Build the linear response and its uniform nominal prior."""
        d = len(self.param_names)
        low = jnp.full(d, -10.0) if low is None else jnp.asarray(low)
        high = jnp.full(d, 10.0) if high is None else jnp.asarray(high)
        if low.shape != (d,) or high.shape != (d,):
            raise ValueError("uniform bounds must have shape (D,)")
        _check_values(
            jnp.isfinite(low) & jnp.isfinite(high) & (low < high),
            "uniform bounds must be finite and low < high",
        )
        self.prior = Uniform(low, high)
        self.prior_sets = (PriorSet((self.prior,), prior_set_id, ("nominal",), 0),)
        self.spectral_matrix = (
            jnp.eye(d) if spectral_matrix is None else jnp.asarray(spectral_matrix)
        )
        if self.spectral_matrix.ndim != 2 or self.spectral_matrix.shape[1] != d:
            raise ValueError("spectral_matrix must have shape (B, D)")
        _check_values(
            jnp.isfinite(self.spectral_matrix), "spectral_matrix must be finite"
        )
        if self.quantity_names != self.param_names:
            raise ValueError("quantity_names must equal param_names")

    def quantities(self, theta, context):
        """Return the parameter columns as named scalar quantities."""
        if theta.shape[-1] != len(self.param_names):
            raise ValueError("theta has wrong D")
        missing = set(self.context_names) - context.keys()
        if missing:
            raise ValueError(f"missing context: {sorted(missing)}")
        return {name: theta[..., i] for i, name in enumerate(self.param_names)}

    def expected_data(self, truth, plan):
        """Return the linear response with the plan's availability masks."""
        self.validate_truth(truth, plan)
        if plan.noise_sigma.shape[1] != self.spectral_matrix.shape[0]:
            raise ValueError("plan B must match spectral_matrix")
        theta = truth.theta
        offset = truth.context["offset"]
        has, usable = _plan_masks(plan)
        visits = plan.visit_success.shape
        phot = jnp.broadcast_to((theta[:, 0] + offset)[:, None], visits)
        ast = jnp.broadcast_to(
            (theta[:, (0, -1)] + offset[:, None])[:, None, :], (*visits, 2)
        )
        return ObservedData(
            spectrum=jnp.where(
                has[:, None], theta @ self.spectral_matrix.T + offset[:, None], 0
            ),
            has_spectrum=has,
            photometry=jnp.where(usable & (plan.visit_photometric_snr > 0), phot, 0),
            astrometry_mas=jnp.where(usable[..., None], ast, 0),
            visit_usable=usable,
            rv_semi_amplitude_ms=jnp.where(
                plan.mass_measured, theta[:, -1] + offset, 0
            ),
        )


def __getattr__(name):
    """Resolve the deprecated truth-model name for existing callers."""
    if name == "LinearGaussianTruth":
        warnings.warn(
            "LinearGaussianTruth is deprecated; use LinearGaussianTruthModel; "
            "will be removed in 0.1.0",
            DeprecationWarning,
            stacklevel=2,
        )
        return LinearGaussianTruthModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
