"""The report core: the summary every engine emits.

A certified yield is never reported as a bare count: the summary carries the
false-certification accounting next to it, because a mission certifying
thirty targets with three false is not better than one certifying
twenty-seven clean. Two kinds of false certification are kept apart: a
certified target with no real source behind it, and a certified claim that
is wrong about a real planet. Engines of every operator emit the same
schemas, so cross-engine comparisons are table joins rather than bespoke
glue: :class:`YieldSummary` for realized runs, :class:`CertifiedYield` for
expected yields from per-planet certification probabilities.

This module is declarative: stdlib only, no JAX.
"""

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class YieldSummary:
    """What one run reports.

    Attributes:
        operator: The outcome operator the run evaluated.
        certificate: Name of the certificate that defines "certified".
        certified_targets: Target ids whose certificate crossed.
        false_certified_targets: Certified ids with no real underlying
            source; only truth-holding engines can fill this.
        epochs: Number of executed loop steps (zero for closed-form runs).
        resources_spent: Final ledger totals as (resource, amount) pairs.
        wrong_certified_targets: Certified ids with a real source whose
            certified claim is false about it; only truth-holding engines
            can fill this.
    """

    operator: str
    certificate: str
    certified_targets: tuple[int, ...]
    false_certified_targets: tuple[int, ...]
    epochs: int
    resources_spent: tuple[tuple[str, float], ...]
    wrong_certified_targets: tuple[int, ...] = ()

    @property
    def certified(self) -> int:
        """The certified count."""
        return len(self.certified_targets)

    @property
    def false_certifications(self) -> int:
        """The false-certification count."""
        return len(self.false_certified_targets)

    @property
    def wrong_certifications(self) -> int:
        """The count of certified claims that are wrong about a real planet."""
        return len(self.wrong_certified_targets)

    def to_json(self) -> str:
        """Serialize canonically (sorted keys)."""
        return json.dumps(
            {
                "operator": self.operator,
                "certificate": self.certificate,
                "certified_targets": list(self.certified_targets),
                "false_certified_targets": list(self.false_certified_targets),
                "epochs": self.epochs,
                "resources_spent": [[k, v] for k, v in self.resources_spent],
                "wrong_certified_targets": list(self.wrong_certified_targets),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, text: str) -> "YieldSummary":
        """Deserialize from :meth:`to_json` output."""
        data = json.loads(text)
        return cls(
            operator=data["operator"],
            certificate=data["certificate"],
            certified_targets=tuple(data["certified_targets"]),
            false_certified_targets=tuple(data["false_certified_targets"]),
            epochs=data["epochs"],
            resources_spent=tuple((k, v) for k, v in data["resources_spent"]),
            wrong_certified_targets=tuple(data.get("wrong_certified_targets", ())),
        )


def summarize(
    state,
    universe,
    *,
    operator: str,
    certificate: str = "detection",
    is_real=bool,
    is_wrong=None,
) -> YieldSummary:
    """Score a finished run against its drawn truth.

    Args:
        state: The final loop state.
        universe: The drawn ground truth the run played against.
        certificate: Which certificate defines "certified".
        operator: The operator label to record.
        is_real: Maps a target's truth scene to whether a real source is
            present. The default truthiness test suits simple containers;
            scene-aware implementations supply their own.
        is_wrong: Maps a real target's truth scene to whether the certified
            claim is false about it. None skips claim-level scoring, which
            suits a detection certificate whose claim is presence itself.

    Returns:
        The run's YieldSummary.
    """
    certified = state.belief.certified(certificate)
    false_certified = tuple(
        target_id
        for target_id in certified
        if not is_real(universe.systems.get(target_id))
    )
    wrong_certified = ()
    if is_wrong is not None:
        wrong_certified = tuple(
            target_id
            for target_id in certified
            if target_id not in false_certified
            and is_wrong(universe.systems.get(target_id))
        )
    return YieldSummary(
        operator=operator,
        certificate=certificate,
        certified_targets=certified,
        false_certified_targets=false_certified,
        epochs=state.epoch_index,
        resources_spent=tuple(sorted(state.ledger.spent.items())),
        wrong_certified_targets=wrong_certified,
    )


@dataclass(frozen=True)
class CertifiedYield:
    """Expected certified yields from per-planet certification probabilities.

    With w_p the planet weights, p_pos, p_neg, p_wrong and p_neg_wrong the
    per-planet probabilities from the certificate,

        y_issued    = sum over all planets of w_p p_pos
        y_cert_pos  = sum over all planets of w_p (p_pos - p_wrong)
        y_wrong     = sum over all planets of w_p p_wrong
        y_neg       = sum over all planets of w_p p_neg
        y_neg_wrong = sum over all planets of w_p p_neg_wrong

    They are always reported together. The per-planet columns hold each
    planet's weighted contribution, so every yield is the sum of its column
    and a bootstrap over planets resamples rows and re-sums. Standard errors
    propagate the per-planet Monte Carlo errors in quadrature, treating the
    planets' estimates as independent; they are zero for closed-form tiers.
    These are independent-planet LOWER BOUNDS: shared table or
    common-random-number errors are not included. The positive-yield error
    retains the candidate-masked p_pos_se approximation. With replicates,
    the random interval uses all planets' paired probability estimates,
    including covariance between positive and wrong certifications. Batch
    SEs are sample SD / sqrt(R), with Student t intervals; bootstrap SEs are
    sample SD, with percentile intervals. Systematic bounds remain explicit.

    Attributes:
        operator: The outcome operator the run evaluated.
        certificate: Name of the certificate that defines "certified".
        y_issued: Expected count of all issued positive certifications.
        y_cert_pos: Expected count of correct counted-direction certifications.
        y_wrong: Expected count of counted-direction certifications that are
            wrong.
        y_neg: Expected count of certified negative verdicts.
        y_neg_wrong: Expected count of certified negative verdicts that are
            wrong.
        y_cert_pos_se_independent_lower_bound: Independent-planet LOWER BOUND
            for the Monte Carlo standard error of ``y_cert_pos``.
        y_wrong_se_independent_lower_bound: Independent-planet LOWER BOUND
            for the Monte Carlo standard error of ``y_wrong``.
        y_neg_se_independent_lower_bound: Independent-planet LOWER BOUND
            for the Monte Carlo standard error of ``y_neg``.
        y_neg_wrong_se_independent_lower_bound: Independent-planet LOWER BOUND
            for the Monte Carlo standard error of ``y_neg_wrong``.
        weights: Per-planet weights w_p.
        issued: Per-planet contributions to ``y_issued``.
        cert_pos: Per-planet contributions to ``y_cert_pos``.
        wrong: Per-planet contributions to ``y_wrong``.
        neg: Per-planet contributions to ``y_neg``.
        neg_wrong: Per-planet contributions to ``y_neg_wrong``.
        in_range: Per-planet in-range flags of the evaluating tier.
        y_issued_se_independent_lower_bound: Independent quadrature for issued.
        interval_kind: replicate-batch, replicate-bootstrap, or
            independent-lower-bound.
        confidence: Random interval coverage probability.
        replicate_kind: Batch or bootstrap scheme, when available.
        replicate_key: Identifier shared by paired streams.
        replicate_yields: Per-channel weighted sums for each replicate.
        standard_errors: Replicate SEs, or independent lower bounds.
        intervals: Random intervals, without systematic bounds.
        design_effects: Replicate variance divided by independent variance;
            None when the independent variance is zero.
        wrong_issued_ratio: Wrong yield divided by issued yield, or None.
        wrong_issued_ratio_rep: Paired replicate ratios, or None if undefined.
        wrong_issued_ratio_se: Replicate ratio SE, when defined.
        wrong_issued_ratio_interval: Replicate ratio interval, when defined.
        systematic: Component-to-channel signed bounds in yield units.
        total_intervals: Random intervals plus numerical systematic bounds;
            the component fm is excluded and remains in systematic.
    """

    operator: str
    certificate: str
    y_issued: float
    y_cert_pos: float
    y_wrong: float
    y_neg: float
    y_neg_wrong: float
    y_cert_pos_se_independent_lower_bound: float
    y_wrong_se_independent_lower_bound: float
    y_neg_se_independent_lower_bound: float
    y_neg_wrong_se_independent_lower_bound: float
    weights: tuple[float, ...]
    issued: tuple[float, ...]
    cert_pos: tuple[float, ...]
    wrong: tuple[float, ...]
    neg: tuple[float, ...]
    neg_wrong: tuple[float, ...]
    in_range: tuple[bool, ...]

    y_issued_se_independent_lower_bound: float = 0.0
    interval_kind: str = "independent-lower-bound"
    confidence: float = 0.95
    replicate_kind: str | None = None
    replicate_key: str | None = None
    replicate_yields: dict[str, tuple[float, ...]] = field(default_factory=dict)
    standard_errors: dict[str, float] = field(default_factory=dict)
    intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    design_effects: dict[str, float | None] = field(default_factory=dict)
    wrong_issued_ratio: float | None = None
    wrong_issued_ratio_rep: tuple[float, ...] | None = None
    wrong_issued_ratio_se: float | None = None
    wrong_issued_ratio_interval: tuple[float, float] | None = None
    systematic: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)
    total_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def all_in_range(self) -> bool:
        """Whether every planet was evaluated inside the validated range."""
        return all(self.in_range)

    def to_json(self) -> str:
        """Serialize canonically (sorted keys)."""
        return json.dumps(
            {
                name: list(value) if isinstance(value, tuple) else value
                for name, value in self.__dict__.items()
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, text: str) -> "CertifiedYield":
        """Deserialize from :meth:`to_json` output."""
        data = json.loads(text)
        for name in ("issued", "cert_pos", "wrong", "neg", "neg_wrong"):
            old = f"y_{name}_se"
            if old in data:
                data.setdefault(f"{old}_independent_lower_bound", data.pop(old))
        # Older reports have no issued column; recover it from correct + wrong.
        if "issued" not in data:
            data["issued"] = [
                correct + wrong
                for correct, wrong in zip(data["cert_pos"], data["wrong"], strict=True)
            ]
        data.setdefault("y_issued", sum(data["issued"]))
        for name in ("replicate_yields", "intervals", "total_intervals"):
            if name in data:
                data[name] = {key: tuple(value) for key, value in data[name].items()}
        if "systematic" in data:
            data["systematic"] = {
                component: {key: tuple(bound) for key, bound in channels.items()}
                for component, channels in data["systematic"].items()
            }
        return cls(
            **{
                name: tuple(value) if isinstance(value, list) else value
                for name, value in data.items()
            }
        )
