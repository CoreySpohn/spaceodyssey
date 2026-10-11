"""The analytic engine: integrate a certificate over a weighted planet batch."""

from spaceodyssey.build import Mission
from spaceodyssey.certify import AbstractProfileCertificate
from spaceodyssey.engines import register_engine
from spaceodyssey.errors import WiringError
from spaceodyssey.metrics import CertifiedYield
from spaceodyssey.plan import PlanPredictive
from spaceodyssey.reporting import CHANNELS as CHANNELS
from spaceodyssey.reporting import PairedDifference as PairedDifference
from spaceodyssey.reporting import _confidence_check as _confidence_check
from spaceodyssey.reporting import _contributions as _contributions
from spaceodyssey.reporting import _replicate_interval as _replicate_interval
from spaceodyssey.reporting import _systematic_bounds as _systematic_bounds
from spaceodyssey.reporting import certified_yield as certified_yield
from spaceodyssey.reporting import paired_difference as paired_difference
from spaceodyssey.reporting import replicate_yields as replicate_yields
from spaceodyssey.reporting import weighted_yields as weighted_yields


@register_engine("integrate")
def run_integrate(
    mission: Mission,
    *,
    seed: int = 0,
    predictive: PlanPredictive,
    weights,
    strict: bool = True,
    confidence: float = 0.95,
    systematic=None,
) -> CertifiedYield:
    """Integrate the model-side certificate over a weighted planet batch.

    Args:
        mission: The compiled mission; its model stack must hold a
            certificate supporting the integrate operator.
        seed: Unused; closed-form and tabulated tiers draw nothing.
        predictive: The plan summary and truth planets to integrate over.
        weights: Per-planet population weights, shape (P,).
        confidence: Coverage probability for random intervals.
        systematic: Component-to-channel bounds passed to the report.
        strict: Raise when any planet lies outside the certificate's
            validated range; when False the report carries the flags.

    Returns:
        The report with replicate intervals when available. Independent
        standard errors remain labeled lower bounds; numerical systematic
        bounds extend only total_intervals, and fm stays separate.

    Raises:
        WiringError: When no certificate is configured.
        OutOfRangeError: When ``strict`` and a planet is out of range.
    """
    certificate = mission.model.certification
    if certificate is None:
        raise WiringError("the integrate engine needs a certification seam choice")
    if isinstance(certificate, AbstractProfileCertificate):
        certificate.validate_predictive(predictive)
    outcome = certificate.integrate(predictive)
    if isinstance(certificate, AbstractProfileCertificate):
        if outcome.profile_hash != certificate.profile.content_hash():
            raise WiringError("outcome profile_hash does not match certificate profile")
    if strict:
        outcome.raise_if_out_of_range()
    name = mission.config.choice("certification").model_ref.name
    return certified_yield(
        outcome, weights, certificate=name, confidence=confidence, systematic=systematic
    )
