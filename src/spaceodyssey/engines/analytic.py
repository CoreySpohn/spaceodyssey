"""The analytic engine: expected certified yields as a weighted sum.

This engine evaluates the integrate operator over a synthetic population: a
finite batch of drawn planets, each with a weight, scored by a certificate's
per-planet certification probabilities. It is a weighted sum over that
batch, not the integral over the population prior; it approximates the
population integral only as well as the weighted batch represents the
population. Everything up to the report is pure JAX and differentiable.
"""

import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.build import Mission
from spaceodyssey.engines import register_engine
from spaceodyssey.errors import WiringError
from spaceodyssey.metrics import CertifiedYield
from spaceodyssey.plan import CertificationOutcome, PlanPredictive


def _contributions(outcome: CertificationOutcome, weights):
    weights = jnp.asarray(weights)
    return (
        weights * outcome.p_pos,
        weights * (outcome.p_pos - outcome.p_wrong),
        weights * outcome.p_wrong,
        weights * outcome.p_neg,
        weights * outcome.p_neg_wrong,
    )


def weighted_yields(
    outcome: CertificationOutcome, weights
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return (y_issued, y_cert_pos, y_wrong, y_neg, y_neg_wrong) over all planets.

    Args:
        outcome: Per-planet certification probabilities, shape (P,) each.
        weights: Per-planet population weights, shape (P,).

    Returns:
        The five expected yields as scalars: all issued positive certifications,
        correct counted-direction certifications,
        wrong counted-direction certifications, certified negatives, and
        wrong certified negatives.
    """
    return tuple(jnp.sum(c) for c in _contributions(outcome, weights))


def certified_yield(
    outcome: CertificationOutcome,
    weights,
    *,
    certificate: str,
    operator: str = "integrate",
) -> CertifiedYield:
    """Score per-planet outcomes into a report with a per-planet breakdown.

    Args:
        outcome: Per-planet certification probabilities, shape (P,) each.
        weights: Per-planet population weights, shape (P,).
        certificate: Name of the certificate that produced the outcome.
        operator: The operator label to record.

    Returns:
        The CertifiedYield report. Its independent-planet standard errors
        are LOWER BOUNDS: shared table and common-random-number errors
        are not included.
    """
    weights = jnp.asarray(weights)
    if weights.shape != outcome.p_pos.shape:
        raise ValueError(
            f"weights must have shape {outcome.p_pos.shape}, got {weights.shape}"
        )
    columns = [np.asarray(c, dtype=float) for c in _contributions(outcome, weights)]
    w = np.asarray(weights, dtype=float)
    candidate = np.asarray(outcome.truth_positive)
    errors = (
        w * np.where(candidate, np.asarray(outcome.p_pos_se, dtype=float), 0.0),
        w * np.asarray(outcome.p_wrong_se, dtype=float),
        w * np.asarray(outcome.p_neg_se, dtype=float),
        w * np.asarray(outcome.p_neg_wrong_se, dtype=float),
    )
    yields = [float(c.sum()) for c in columns]
    standard_errors = [float(np.sqrt(np.sum(e**2))) for e in errors]
    return CertifiedYield(
        operator=operator,
        certificate=certificate,
        y_issued=yields[0],
        y_cert_pos=yields[1],
        y_wrong=yields[2],
        y_neg=yields[3],
        y_neg_wrong=yields[4],
        y_cert_pos_se_independent_lower_bound=standard_errors[0],
        y_wrong_se_independent_lower_bound=standard_errors[1],
        y_neg_se_independent_lower_bound=standard_errors[2],
        y_neg_wrong_se_independent_lower_bound=standard_errors[3],
        weights=tuple(w.tolist()),
        issued=tuple(columns[0].tolist()),
        cert_pos=tuple(columns[1].tolist()),
        wrong=tuple(columns[2].tolist()),
        neg=tuple(columns[3].tolist()),
        neg_wrong=tuple(columns[4].tolist()),
        in_range=tuple(np.asarray(outcome.in_range).tolist()),
    )


@register_engine("integrate")
def run_integrate(
    mission: Mission,
    *,
    seed: int = 0,
    predictive: PlanPredictive,
    weights,
    strict: bool = True,
) -> CertifiedYield:
    """Integrate the model-side certificate over a weighted planet batch.

    Args:
        mission: The compiled mission; its model stack must hold a
            certificate supporting the integrate operator.
        seed: Unused; closed-form and tabulated tiers draw nothing.
        predictive: The plan summary and truth planets to integrate over.
        weights: Per-planet population weights, shape (P,).
        strict: Raise when any planet lies outside the certificate's
            validated range; when False the report carries the flags.

    Returns:
        The CertifiedYield report. Its independent-planet standard errors
        are LOWER BOUNDS: shared table and common-random-number errors
        are not included.

    Raises:
        WiringError: When no certificate is configured.
        OutOfRangeError: When ``strict`` and a planet is out of range.
    """
    certificate = mission.model.certification
    if certificate is None:
        raise WiringError("the integrate engine needs a certification seam choice")
    outcome = certificate.integrate(predictive)
    if strict:
        outcome.raise_if_out_of_range()
    name = mission.config.choice("certification").model_ref.name
    return certified_yield(outcome, weights, certificate=name)
