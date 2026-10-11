"""Mission belief containers.

The belief is everything the agent may see: per-target posteriors,
accumulated observation records, and certificate states, plus reserved slots
for the shared population block and the instrument block so the container
does not need redesign when those land. The drawn truth is never part of the
belief; the only bridge between them is the observation function.

Updates are pure: every method returns a new container and leaves the
original untouched, so belief snapshots are free.
"""

from typing import Any, final

import equinox as eqx

from spaceodyssey.certify import CertificateState
from spaceodyssey.data import Observation


@final
class TargetBelief(eqx.Module):
    """The agent's state of knowledge about one target.

    Attributes:
        posterior: The target's scene posterior once inference runs; None
            until the belief-update layer lands.
        observations: Every observation record accumulated on this target.
        certificates: Certificate states keyed by certificate name.
    """

    posterior: Any | None = None
    observations: tuple = ()
    certificates: dict = eqx.field(default_factory=dict)


@final
class MissionBelief(eqx.Module):
    """The agent-visible state of knowledge for the whole mission.

    Attributes:
        per_target: TargetBelief entries keyed by target id.
        population: Reserved slot for a shared population-hyperparameter
            posterior; None until that layer lands.
        instrument: Reserved slot for an instrument-state posterior; None
            until that layer lands.
    """

    per_target: dict = eqx.field(default_factory=dict)
    population: Any | None = None
    instrument: Any | None = None

    def target(self, target_id: int) -> TargetBelief:
        """Return the belief about one target (empty if never observed)."""
        return self.per_target.get(target_id, TargetBelief())

    def record(self, obs: Observation) -> "MissionBelief":
        """Return a new belief with one observation appended to its target."""
        target_id = obs.action.target_id
        current = self.target(target_id)
        updated = TargetBelief(
            posterior=current.posterior,
            observations=(*current.observations, obs),
            certificates=dict(current.certificates),
        )
        per_target = dict(self.per_target)
        per_target[target_id] = updated
        return MissionBelief(
            per_target=per_target,
            population=self.population,
            instrument=self.instrument,
        )

    def with_certificate(
        self, target_id: int, name: str, state: CertificateState
    ) -> "MissionBelief":
        """Return a new belief with one certificate state replaced."""
        current = self.target(target_id)
        certificates = dict(current.certificates)
        certificates[name] = state
        updated = TargetBelief(
            posterior=current.posterior,
            observations=current.observations,
            certificates=certificates,
        )
        per_target = dict(self.per_target)
        per_target[target_id] = updated
        return MissionBelief(
            per_target=per_target,
            population=self.population,
            instrument=self.instrument,
        )

    def certificate(self, target_id: int, name: str) -> CertificateState | None:
        """Return one certificate state, or None if absent."""
        return self.target(target_id).certificates.get(name)

    def certified(self, name: str) -> tuple[int, ...]:
        """Return the sorted target ids whose named certificate has crossed."""
        crossed = []
        for target_id, belief in self.per_target.items():
            state = belief.certificates.get(name)
            if state is not None and bool(state.crossed):
                crossed.append(target_id)
        return tuple(sorted(crossed))
