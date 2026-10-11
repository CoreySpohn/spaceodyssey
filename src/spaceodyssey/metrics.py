"""The report core: the summary every engine emits.

A certified yield is never reported as a bare count: the summary carries the
false-certification accounting next to it, because a mission certifying
thirty targets with three false is not better than one certifying
twenty-seven clean. Engines of every operator emit this same schema, so
cross-engine comparisons are table joins rather than bespoke glue.

This module is declarative: stdlib only, no JAX.
"""

import json
from dataclasses import dataclass


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
    """

    operator: str
    certificate: str
    certified_targets: tuple[int, ...]
    false_certified_targets: tuple[int, ...]
    epochs: int
    resources_spent: tuple[tuple[str, float], ...]

    @property
    def certified(self) -> int:
        """The certified count."""
        return len(self.certified_targets)

    @property
    def false_certifications(self) -> int:
        """The false-certification count."""
        return len(self.false_certified_targets)

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
        )


def summarize(
    state,
    universe,
    *,
    operator: str,
    certificate: str = "detection",
    is_real=bool,
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

    Returns:
        The run's YieldSummary.
    """
    certified = state.belief.certified(certificate)
    false_certified = tuple(
        target_id
        for target_id in certified
        if not is_real(universe.systems.get(target_id))
    )
    return YieldSummary(
        operator=operator,
        certificate=certificate,
        certified_targets=certified,
        false_certified_targets=false_certified,
        epochs=state.epoch_index,
        resources_spent=tuple(sorted(state.ledger.spent.items())),
    )
