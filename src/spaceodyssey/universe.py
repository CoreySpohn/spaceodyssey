"""Truth containers: the drawn universe and its sources.

A universe is the frozen ground truth of one simulated mission: per-target
scenes, per-target exozodi draws, and the instrument truth. It lives strictly
on the truth side of the firewall; it is never an argument to policies,
belief updates, or certification.
"""

import abc
from typing import Any, ClassVar, final

import equinox as eqx


@final
class Universe(eqx.Module):
    """The frozen ground truth for one simulated mission.

    Attributes:
        systems: Truth scene per target id.
        exozodi: Exozodiacal draw per target id.
        instrument: The instrument truth; a static instrument today, a
            time-dependent observatory state later, at exactly this slot.
    """

    systems: dict = eqx.field(default_factory=dict)
    exozodi: dict = eqx.field(default_factory=dict)
    instrument: Any | None = None


class AbstractUniverseSource(eqx.Module):
    """Draws universes from a population description."""

    PROTOCOL_VERSION: ClassVar[int] = 1

    @abc.abstractmethod
    def draw(self, key) -> Universe:
        """Draw one universe."""


@final
class FixedUniverse(AbstractUniverseSource):
    """A source that returns one pre-built universe regardless of key.

    For replay and testing, constructed programmatically; it is deliberately
    not a registry entry because a pre-built universe is not expressible as
    declarative parameters (a serialized-universe reference will be, once
    run-log persistence lands).

    Attributes:
        universe: The universe every draw returns.
    """

    universe: Universe

    def draw(self, key) -> Universe:
        """Return the fixed universe."""
        return self.universe
