"""The certification seam: when does an accumulating result count.

A certificate is a persistent per-target state that hardens as evidence
accumulates and gates what the mission announces; it is distinct from any
score a policy uses to rank actions, and the two are never converted into
each other. One certificate definition supports up to three evaluations,
matching the outcome operators: update against realized data, propagate
against predictive moments, and integrate to a certification probability.

Two meanings of "certified" are kept apart. An announcement is historical
and irreversible: once a certificate crosses, the mission said so and the
record stands. Scientific support is current and can weaken: a profile
certificate whose posterior moves back across its risk bound revokes support
without erasing the announcement.
"""

import abc
from typing import ClassVar, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri

from spaceodyssey.errors import UnsupportedOperator
from spaceodyssey.plan import CertificationOutcome, PlanPredictive, PlanSummary
from spaceodyssey.registry import SeamInfo, register


@final
class CertificateState(eqx.Module):
    """The running state of one certificate on one target.

    Attributes:
        statistic: The certificate's accumulated statistic.
        crossed: The historical announcement: whether the certificate has
            hardened past its threshold at any update. Never reverts.
        epoch_d: Epoch of the last update, in days of mission time.
        supported: Whether the evidence currently supports the certified
            claim. Defaults to ``crossed``; a certificate whose support can
            weaken sets it separately and may revoke it.
    """

    statistic: jax.Array = eqx.field(converter=jnp.asarray)
    crossed: jax.Array = eqx.field(converter=jnp.asarray)
    epoch_d: float
    supported: jax.Array = eqx.field(converter=jnp.asarray)

    def __init__(self, statistic, crossed, epoch_d: float = 0.0, supported=None):
        """Build a state; ``supported`` defaults to ``crossed``."""
        self.statistic = statistic
        self.crossed = crossed
        self.epoch_d = epoch_d
        self.supported = crossed if supported is None else supported

    def revise(self, statistic, supported, epoch_d: float) -> "CertificateState":
        """Return the state after evidence that sets current support.

        The announcement latches (``crossed`` stays true once set) while
        ``supported`` follows the latest evidence, so support can be revoked
        without un-announcing.

        Args:
            statistic: The certificate's statistic after the evidence.
            supported: Whether the evidence now supports the claim.
            epoch_d: Epoch of the evidence, in days of mission time.

        Returns:
            The revised state.
        """
        supported = jnp.asarray(supported)
        return CertificateState(
            statistic=statistic,
            crossed=jnp.logical_or(self.crossed, supported),
            epoch_d=epoch_d,
            supported=supported,
        )


class AbstractCertificate(eqx.Module):
    """One certificate definition, evaluable under multiple operators."""

    PROTOCOL_VERSION: ClassVar[int] = 1

    @abc.abstractmethod
    def initial(self) -> CertificateState:
        """Return the state before any evidence."""

    @abc.abstractmethod
    def update(self, state: CertificateState, dataset) -> CertificateState:
        """Advance the state with one realized dataset."""

    @abc.abstractmethod
    def propagate(self, state: CertificateState, moments) -> CertificateState:
        """Advance the state with predictive moments instead of a draw."""

    @abc.abstractmethod
    def integrate(self, predictive) -> jax.Array | CertificationOutcome:
        """Return the certification probability under a predictive.

        A certificate over one target returns a scalar probability. A
        profile certificate evaluated over a batch of planets takes a
        :class:`spaceodyssey.PlanPredictive` and returns a
        :class:`spaceodyssey.CertificationOutcome`; it never raises for inputs
        outside its validated range, it clears ``in_range`` instead.
        """


@final
@register(
    "certification",
    "snr_threshold",
    SeamInfo(
        operators={"sample"},
        consumes=("spaceodyssey.SummaryDataset",),
        produces=("spaceodyssey.CertificateState",),
        version="1",
        fidelity="threshold",
        cost_hint="low",
    ),
)
class SnrThreshold(AbstractCertificate):
    """Certify a detection when the best observed statistic crosses a bar.

    The running statistic is the maximum detection statistic seen so far;
    weaker later data never un-certifies a target.

    Attributes:
        threshold: The statistic value at which the certificate hardens.
    """

    threshold: float

    def initial(self) -> CertificateState:
        """Return an uncrossed state with a zero statistic."""
        return CertificateState(statistic=0.0, crossed=False, epoch_d=0.0)

    def update(self, state: CertificateState, dataset) -> CertificateState:
        """Fold one summary dataset's detection statistic into the state."""
        statistic = jnp.maximum(state.statistic, dataset.snr)
        return CertificateState(
            statistic=statistic,
            crossed=statistic >= self.threshold,
            epoch_d=dataset.epoch_d,
        )

    def propagate(self, state: CertificateState, moments) -> CertificateState:
        """Raise: this certificate evaluates realized data only."""
        raise UnsupportedOperator("snr_threshold supports only the sample operator")

    def integrate(self, predictive) -> jax.Array:
        """Raise: this certificate evaluates realized data only."""
        raise UnsupportedOperator("snr_threshold supports only the sample operator")


@final
@register(
    "certification",
    "toy_gaussian",
    SeamInfo(
        operators={"integrate"},
        consumes=("spaceodyssey.PlanPredictive",),
        produces=("spaceodyssey.CertificationOutcome",),
        version="1",
        fidelity="closed_form",
        cost_hint="low",
    ),
)
class ToyGaussianCertifier(AbstractCertificate):
    """A one-claim certificate with an exact closed-form certification law.

    The known-answer reference for the integrate interface. Each planet
    carries one scalar claim parameter theta (the truth passed in the
    predictive, shape (P,)) and the claim is "theta > boundary".

    Declared signal model: each bin has unit reference planet flux. For
    covariance C = diag(noise_sigma**2) + F F^T, the effective SNR is
    S = sqrt(1^T C^-1 1). White noise gives S = sqrt(sum_b noise_sigma_b^-2),
    reproducing v1 with snr_b = 1 / noise_sigma_b. The posterior standard
    deviation of theta is sigma = sigma_unit_snr / S. With a flat prior and
    known noise the posterior is N(m, sigma**2) with m = theta + e and
    e ~ N(0, sigma**2). The certificate is positive when
    P(theta > boundary | data) >= 1 - alpha, that is when
    m >= boundary + z sigma with z = Phi^-1(1 - alpha), and negative when
    m <= boundary - z sigma. Writing d = (theta - boundary) / sigma,

        p_pos = Phi(d - z),    p_neg = Phi(-d - z),

    and a positive certification is wrong exactly when theta <= boundary,
    a negative one exactly when theta > boundary.

    Parity limit: this is the support condition alone with known noise and
    one prior. For a detection claim (theta the signal amplitude,
    boundary = 0) the observed statistic m / sigma is the measured SNR, and
    the certificate is the indicator that it reaches z, identical to
    :class:`SnrThreshold` with ``threshold = Phi^-1(1 - alpha)``.

    A planet is in range when its combined SNR is positive and finite and
    its theta is finite, and noise_known is true. Missing spectra, failed
    spectrum visits, and zero visits give both verdict probabilities zero
    and are in range (a known absence of data). Other out-of-range planets
    get the zero-information limit d = 0 (both verdicts at probability
    alpha), flagged untrusted.

    Attributes:
        boundary: The claim's decision boundary, in the units of theta.
        alpha: The allowed risk of the certified verdict, in (0, 0.5].
        sigma_unit_snr: Posterior standard deviation of theta at a combined
            SNR of one, in the units of theta.
    """

    boundary: float
    alpha: float
    sigma_unit_snr: float

    def __check_init__(self):
        """Validate the risk level and the SNR mapping."""
        if not 0.0 < self.alpha <= 0.5:
            raise ValueError(f"alpha must lie in (0, 0.5], got {self.alpha}")
        if not self.sigma_unit_snr > 0.0:
            raise ValueError(
                f"sigma_unit_snr must be positive, got {self.sigma_unit_snr}"
            )

    def _combined_snr(self, plan: PlanSummary) -> jax.Array:
        signal = jnp.ones_like(plan.noise_sigma) / plan.noise_sigma
        factors = plan.noise_corr_factors
        if factors is None or factors.shape[-1] == 0:
            return jnp.sqrt(jnp.sum(jnp.square(signal), axis=-1))
        whitened = factors / plan.noise_sigma[..., None]
        covariance = jnp.eye(signal.shape[-1]) + whitened @ jnp.swapaxes(
            whitened, -1, -2
        )
        solved = jnp.linalg.solve(covariance, signal[..., None])[..., 0]
        return jnp.sqrt(jnp.maximum(jnp.sum(signal * solved, axis=-1), 0.0))

    def posterior_sd(self, plan: PlanSummary) -> jax.Array:
        """Return theta's posterior sigma for the toy's unit spectral signal.

        This noise mapping ignores visit availability; integrate gates the
        verdicts on usable spectra. The result has shape (P,).
        """
        return self.sigma_unit_snr / self._combined_snr(plan)

    def verdict(self, posterior_mean, posterior_sd) -> tuple[jax.Array, jax.Array]:
        """Return the (positive, negative) certified verdicts for realized data.

        Args:
            posterior_mean: Posterior mean of theta, any shape.
            posterior_sd: Posterior standard deviation, broadcastable.

        Returns:
            Boolean arrays: certified positive and certified negative.
        """
        z = ndtri(1.0 - self.alpha)
        score = (posterior_mean - self.boundary) / posterior_sd
        return score >= z, score <= -z

    def initial(self) -> CertificateState:
        """Return an uncrossed state with a zero statistic."""
        return CertificateState(statistic=0.0, crossed=False, epoch_d=0.0)

    def update(self, state: CertificateState, dataset) -> CertificateState:
        """Raise: this certificate evaluates the integrate operator only."""
        raise UnsupportedOperator("toy_gaussian supports only the integrate operator")

    def propagate(self, state: CertificateState, moments) -> CertificateState:
        """Raise: this certificate evaluates the integrate operator only."""
        raise UnsupportedOperator("toy_gaussian supports only the integrate operator")

    def integrate(self, predictive: PlanPredictive) -> CertificationOutcome:
        """Return the exact per-planet certification probabilities.

        Args:
            predictive: The plan summary and the truth theta, shape (P,).

        Returns:
            The closed-form outcome, with zero standard errors.
        """
        theta = jnp.asarray(predictive.truth)
        plan = predictive.plan
        combined = self._combined_snr(plan)
        if plan.visit_success.shape[1] == 0:
            available = jnp.zeros((plan.n_planets,), dtype=bool)
        else:
            index = jnp.maximum(plan.spectrum_visit, 0)
            available = (plan.spectrum_visit >= 0) & jnp.take_along_axis(
                plan.visit_success, index[:, None], axis=1
            )[:, 0]
        in_range = (
            (combined > 0)
            & jnp.isfinite(combined)
            & jnp.isfinite(theta)
            & plan.noise_known
        )
        safe_combined = jnp.where(in_range, combined, 1.0)
        safe_theta = jnp.where(in_range, theta, self.boundary)
        sigma = self.sigma_unit_snr / safe_combined
        d = jnp.where(in_range, (safe_theta - self.boundary) / sigma, 0.0)
        z = ndtri(1.0 - self.alpha)
        p_pos = jnp.where(available, ndtr(d - z), 0.0)
        p_neg = jnp.where(available, ndtr(-d - z), 0.0)
        truth_positive = theta > self.boundary
        zeros = jnp.zeros_like(p_pos)
        return CertificationOutcome(
            p_pos=p_pos,
            p_neg=p_neg,
            p_wrong=jnp.where(truth_positive, zeros, p_pos),
            p_neg_wrong=jnp.where(truth_positive, p_neg, zeros),
            p_pos_se=zeros,
            p_neg_se=zeros,
            p_wrong_se=zeros,
            p_neg_wrong_se=zeros,
            truth_positive=truth_positive,
            in_range=~available | in_range,
        )
