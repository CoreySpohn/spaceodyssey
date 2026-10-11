"""Immutable claims and certification profiles on named scalar quantities."""

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import jax.numpy as jnp
from jax import Array

Side = Literal["above", "below"]
Counted = Literal["positive", "either"]
SupportRule = Literal["separate", "joint", "split"]
EvidenceRule = Literal["every_claim"]


def _nonempty_string(value: object, field: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a nonempty string")


def _finite_number(value: object, field: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite real number")


@dataclass(frozen=True)
class Edge:
    """An inclusive threshold on a named quantity.

    Args:
        quantity: Name of the scalar quantity supplied by a truth model.
        threshold: Finite boundary in the quantity's units.
        side: Positive side, at or above, or at or below, the boundary.
    """

    quantity: str
    threshold: float
    side: Side

    def __post_init__(self) -> None:
        """Validate the quantity, finite threshold and side."""
        _nonempty_string(self.quantity, "quantity")
        _finite_number(self.threshold, "threshold")
        _nonempty_string(self.side, "side")
        if self.side not in ("above", "below"):
            raise ValueError("side must be 'above' or 'below'")
        object.__setattr__(self, "threshold", float(self.threshold))


@dataclass(frozen=True)
class Claim:
    """A conjunction of one or two edges, with a counted direction.

    Args:
        name: Unique claim label within a profile.
        edges: Tuple of one or two inclusive boundaries.
        counted: Whether the counted event requires a true report or either report.
    """

    name: str
    edges: tuple[Edge, ...]
    counted: Counted

    def __post_init__(self) -> None:
        """Validate immutable edges and nonempty same-quantity intervals."""
        _nonempty_string(self.name, "claim name")
        if type(self.edges) is not tuple or not 1 <= len(self.edges) <= 2:
            raise ValueError("edges must be a tuple of one or two Edge objects")
        if any(type(edge) is not Edge for edge in self.edges):
            raise ValueError("edges must contain Edge objects")
        _nonempty_string(self.counted, "counted")
        if self.counted not in ("positive", "either"):
            raise ValueError("counted must be 'positive' or 'either'")
        if len(self.edges) == 2:
            first, second = self.edges
            if first.quantity == second.quantity:
                if first.side == second.side:
                    raise ValueError("same-quantity edges must have opposite sides")
                lower, upper = (
                    (first, second) if first.side == "above" else (second, first)
                )
                if lower.threshold >= upper.threshold:
                    raise ValueError("lower threshold must be below upper threshold")


@dataclass(frozen=True)
class Profile:
    """A hashable certification configuration with a fixed claim axis.

    Args:
        name: Display label, excluded from the content digest.
        claims: Nonempty tuple of uniquely named claims, in output-axis order.
        alpha: Risk tolerance strictly between zero and one half.
        support_rule: Joint familywise risk, separate claim risks, or an equal
            split of alpha across claims. Joint is the default.
        prior_set_id: Nonempty identifier of the robustness prior set.
        evidence_rule: Require prior-only risk to exceed alpha for every claim.

    Raises:
        ValueError: A field violates the configuration invariants.
    """

    name: str
    claims: tuple[Claim, ...]
    alpha: float
    support_rule: SupportRule = "joint"
    prior_set_id: str = ""
    evidence_rule: EvidenceRule = "every_claim"

    def __post_init__(self) -> None:
        """Validate the profile, including after dataclasses.replace."""
        _nonempty_string(self.name, "name")
        _nonempty_string(self.prior_set_id, "prior_set_id")
        _finite_number(self.alpha, "alpha")
        if not 0 < self.alpha < 0.5:
            raise ValueError("alpha must be strictly between 0 and 0.5")
        object.__setattr__(self, "alpha", float(self.alpha))
        if type(self.claims) is not tuple or not self.claims:
            raise ValueError("claims must be a nonempty tuple of Claim objects")
        if any(type(claim) is not Claim for claim in self.claims):
            raise ValueError("claims must contain Claim objects")
        if len(set(self.claim_names)) != len(self.claims):
            raise ValueError("claim names must be unique")
        _nonempty_string(self.support_rule, "support_rule")
        if self.support_rule not in ("joint", "separate", "split"):
            raise ValueError("support_rule must be 'joint', 'separate' or 'split'")
        _nonempty_string(self.evidence_rule, "evidence_rule")
        if self.evidence_rule != "every_claim":
            raise ValueError("evidence_rule must be 'every_claim'")

    @property
    def claim_names(self) -> tuple[str, ...]:
        """Claim labels in the order of the final output axis."""
        return tuple(claim.name for claim in self.claims)

    @property
    def quantities(self) -> frozenset[str]:
        """Names of all quantities required to evaluate the claims."""
        return frozenset(edge.quantity for claim in self.claims for edge in claim.edges)

    def critical_value(self) -> float | None:
        """Return the one-sided normal critical value, or None for joint risk."""
        if self.support_rule == "joint":
            return None
        alpha = (
            self.alpha / len(self.claims)
            if self.support_rule == "split"
            else self.alpha
        )
        # Symmetry avoids rounding 1 - alpha to one for small alpha.
        return -NormalDist().inv_cdf(alpha)

    def issued_error_bound(self) -> float:
        """Return the matched-population bound on wrong over issued reports."""
        return self.alpha * (len(self.claims) if self.support_rule == "separate" else 1)

    def claim_truth(self, q: Mapping[str, Array]) -> Array:
        """Evaluate inclusive edges, returning boolean truth with shape (..., K).

        Args:
            q: Quantity arrays with mutually broadcastable planet batch shapes.

        Returns:
            Each claim's conjunction, stacked on the final axis in claim order.
        """
        truths = []
        for claim in self.claims:
            truth = jnp.asarray(True)
            for edge in claim.edges:
                value = jnp.asarray(q[edge.quantity])
                holds = (
                    value >= edge.threshold
                    if edge.side == "above"
                    else value <= edge.threshold
                )
                truth = truth & holds
            truths.append(truth)
        return jnp.stack(jnp.broadcast_arrays(*truths), axis=-1)

    def truth_positive(self, q: Mapping[str, Array]) -> Array:
        """Return whether every positive-counted claim is true, shape (...,).

        With no positive-counted claims this is vacuously true for every planet.
        """
        truth = self.claim_truth(q)
        required = jnp.asarray([claim.counted == "positive" for claim in self.claims])
        return jnp.all(truth | ~required, axis=-1)

    def content_hash(self) -> str:
        """Return the SHA-256 digest of canonical, versioned configuration JSON.

        Floats use hexadecimal strings. Claim order and claim names are content;
        the profile's display name is excluded. The digest is independent of
        Python's randomized in-process hash.
        """
        content = {
            "schema": "spaceodyssey.Profile/1",
            "claims": [
                {
                    "name": claim.name,
                    "counted": claim.counted,
                    "edges": [
                        {
                            "quantity": edge.quantity,
                            "threshold": edge.threshold.hex(),
                            "side": edge.side,
                        }
                        for edge in claim.edges
                    ],
                }
                for claim in self.claims
            ],
            "alpha": self.alpha.hex(),
            "support_rule": self.support_rule,
            "prior_set_id": self.prior_set_id,
            "evidence_rule": self.evidence_rule,
        }
        encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()
