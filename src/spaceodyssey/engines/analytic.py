"""The analytic engine: expected certified yields as a weighted sum.

This engine evaluates the integrate operator over a synthetic population: a
finite batch of drawn planets, each with a weight, scored by a certificate's
per-planet certification probabilities. It is a weighted sum over that
batch, not the integral over the population prior; it approximates the
population integral only as well as the weighted batch represents the
population. Everything up to the report is pure JAX and differentiable.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm, t

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


CHANNELS = ("y_issued", "y_cert_pos", "y_wrong", "y_neg", "y_neg_wrong")


def replicate_yields(outcome: CertificationOutcome, weights) -> tuple[jax.Array, ...]:
    """Sum weighted (R, P) probabilities along P, preserving shared errors.

    This operation is differentiable and safe under JAX transformations.
    """
    if outcome.p_pos_rep is None:
        raise ValueError("replicate yields require replicate arrays")
    weights = jnp.asarray(weights)
    if weights.shape != outcome.p_pos.shape:
        raise ValueError("weights must have shape (P,)")
    values = (
        outcome.p_pos_rep,
        outcome.p_pos_rep - outcome.p_wrong_rep,
        outcome.p_wrong_rep,
        outcome.p_neg_rep,
        outcome.p_neg_wrong_rep,
    )
    return tuple(jnp.sum(weights * value, axis=1) for value in values)


def _replicate_interval(point, values, kind, confidence):
    sd = float(np.std(values, ddof=1))
    if kind == "batch":
        se = sd / np.sqrt(len(values))
        half = float(t.ppf((1 + confidence) / 2, len(values) - 1)) * se
        return float(se), (float(point - half), float(point + half))
    tail = (1 - confidence) / 2
    return sd, tuple(float(v) for v in np.quantile(values, [tail, 1 - tail]))


def _confidence_check(confidence):
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class PairedDifference:
    """First report minus second, with intervals preserving replicate pairing."""

    yields: dict[str, float]
    replicate_yields: dict[str, tuple[float, ...]]
    standard_errors: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    interval_kind: str
    replicate_key: str
    confidence: float


def paired_difference(a, b, weights=None, *, confidence=0.95) -> PairedDifference:
    """Compare reports, or two outcomes using common planet weights.

    Matching stream identifiers, kinds and replicate counts are required.
    Outcome inputs require ``weights``; reports already contain weighted sums.
    """
    _confidence_check(confidence)
    if isinstance(a, CertificationOutcome) and isinstance(b, CertificationOutcome):
        if weights is None:
            raise ValueError("outcome comparisons require weights")
        a = certified_yield(a, weights, certificate="a", confidence=confidence)
        b = certified_yield(b, weights, certificate="b", confidence=confidence)
    if not a.replicate_key or a.replicate_key != b.replicate_key:
        raise ValueError("paired comparison requires matching replicate_key")
    if a.replicate_kind != b.replicate_kind:
        raise ValueError("paired comparison requires matching replicate_kind")
    points, reps, errors, intervals = {}, {}, {}, {}
    for channel in CHANNELS:
        left = np.asarray(a.replicate_yields[channel])
        right = np.asarray(b.replicate_yields[channel])
        if left.shape != right.shape:
            raise ValueError("paired comparison requires matching replicate counts")
        values = left - right
        points[channel] = getattr(a, channel) - getattr(b, channel)
        reps[channel] = tuple(values.tolist())
        errors[channel], intervals[channel] = _replicate_interval(
            points[channel], values, a.replicate_kind, confidence
        )
    return PairedDifference(
        points, reps, errors, intervals, a.interval_kind, a.replicate_key, confidence
    )


def _systematic_bounds(systematic):
    result = {}
    for component, channels in (systematic or {}).items():
        result[component] = {}
        for channel, bound in channels.items():
            if channel not in CHANNELS:
                raise ValueError(f"unknown systematic channel: {channel}")
            if np.isscalar(bound):
                magnitude = float(bound)
                if not np.isfinite(magnitude) or magnitude < 0:
                    raise ValueError("symmetric bounds must be finite and nonnegative")
                lo, hi = -magnitude, magnitude
            else:
                lo, hi = map(float, bound)
                if not np.isfinite([lo, hi]).all() or lo > hi:
                    raise ValueError("systematic bounds must be finite and ordered")
            result[component][channel] = (lo, hi)
    return result


def certified_yield(
    outcome: CertificationOutcome,
    weights,
    *,
    certificate: str,
    operator: str = "integrate",
    confidence: float = 0.95,
    systematic=None,
) -> CertifiedYield:
    """Score per-planet outcomes into a report with a per-planet breakdown.

    Args:
        outcome: Per-planet certification probabilities, shape (P,) each.
        weights: Per-planet population weights, shape (P,).
        certificate: Name of the certificate that produced the outcome.
        operator: The operator label to record.
        confidence: Coverage probability for random intervals.
        systematic: Component-to-channel bounds, scalar +/- or [lo, hi];
            fm is reported separately from numerical total intervals.

    Returns:
        The report with replicate intervals when available. Independent
        standard errors remain labeled lower bounds; numerical systematic
        bounds extend only total_intervals, and fm stays separate.
    """
    weights = jnp.asarray(weights)
    if weights.shape != outcome.p_pos.shape:
        raise ValueError(
            f"weights must have shape {outcome.p_pos.shape}, got {weights.shape}"
        )
    _confidence_check(confidence)
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
    independent = [
        float(np.linalg.norm(w * np.asarray(outcome.p_pos_se))),
        *standard_errors,
    ]
    points = dict(zip(CHANNELS, yields, strict=True))
    errors = dict(zip(CHANNELS, independent, strict=True))
    z = float(norm.ppf((1 + confidence) / 2))
    intervals = {
        name: (points[name] - z * se, points[name] + z * se)
        for name, se in errors.items()
    }
    reps, deff = {}, {}
    kind = "independent-lower-bound"
    ratio = yields[2] / yields[0] if yields[0] != 0 else None
    ratio_rep = ratio_se = ratio_interval = None
    if outcome.p_pos_rep is not None:
        kind = "replicate-" + outcome.replicate_kind
        for name, values, indep in zip(
            CHANNELS, replicate_yields(outcome, weights), independent, strict=True
        ):
            values = np.asarray(values, dtype=float)
            reps[name] = tuple(values.tolist())
            errors[name], intervals[name] = _replicate_interval(
                points[name], values, outcome.replicate_kind, confidence
            )
            deff[name] = errors[name] ** 2 / indep**2 if indep > 0 else None
        issued = np.asarray(reps["y_issued"])
        # Ratios are undefined if any replicate issues no certificates.
        if ratio is not None and np.all(issued != 0):
            values = np.asarray(reps["y_wrong"]) / issued
            ratio_rep = tuple(values.tolist())
            ratio_se, ratio_interval = _replicate_interval(
                ratio, values, outcome.replicate_kind, confidence
            )
    bounds = _systematic_bounds(systematic)
    total = {}
    for name, (lo, hi) in intervals.items():
        numerical = [
            channels.get(name, (0.0, 0.0))
            for component, channels in bounds.items()
            if component != "fm"
        ]
        total[name] = (
            lo + sum(b[0] for b in numerical),
            hi + sum(b[1] for b in numerical),
        )
    return CertifiedYield(
        y_issued_se_independent_lower_bound=independent[0],
        interval_kind=kind,
        confidence=confidence,
        replicate_kind=outcome.replicate_kind,
        replicate_key=outcome.replicate_key,
        replicate_yields=reps,
        standard_errors=errors,
        intervals=intervals,
        design_effects=deff,
        wrong_issued_ratio=ratio,
        wrong_issued_ratio_rep=ratio_rep,
        wrong_issued_ratio_se=ratio_se,
        wrong_issued_ratio_interval=ratio_interval,
        systematic=bounds,
        total_intervals=total,
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
    outcome = certificate.integrate(predictive)
    if strict:
        outcome.raise_if_out_of_range()
    name = mission.config.choice("certification").model_ref.name
    return certified_yield(
        outcome, weights, certificate=name, confidence=confidence, systematic=systematic
    )
