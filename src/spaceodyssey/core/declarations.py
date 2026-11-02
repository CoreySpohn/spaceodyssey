"""Declarations.

A component states what it reads, which coordinates key its draws, which stops and
effects it asks for and what it requires from an executor, by defining a `declare`
method that returns a `Declaration`. A component without `declare` declares nothing
unusual. Pre-flight reads these statements; it never runs the study.
"""

import dataclasses

from spaceodyssey.core.enums import Disposition, EffectKind, Scope
from spaceodyssey.core.keys import COORDS, OPTIONAL_COORDS

TRUTH = "truth"
SCORER = "scorer"
READABLE = frozenset({TRUTH, SCORER})

COMPILED = "compiled"
HOST = "host"
EXECUTORS = frozenset({COMPILED, HOST})

# Named capabilities per executor. Both run the same effects; the rest differ by form.
EXECUTOR_CAPABILITIES = {
    COMPILED: frozenset({"batched_worlds", "gradients"}),
    HOST: frozenset({"python_policy"}),
}
SUPPORTED_EFFECTS = frozenset(
    {
        EffectKind.NONE,
        EffectKind.OBSERVE,
        EffectKind.WAIT,
        EffectKind.STOP,
        EffectKind.CANCEL,
        EffectKind.QUERY,
    }
)

DRAW_COORDS = tuple(c for c in COORDS if c not in OPTIONAL_COORDS)


@dataclasses.dataclass(frozen=True)
class Need:
    """A decision at `decide_tick` reads the product of a channel's acquisition.

    The acquisition starts at `start_tick`; its product is released no earlier than
    `start_tick` plus the channel's occupancy and latency.
    """

    channel_id: int
    start_tick: int
    decide_tick: int


@dataclasses.dataclass(frozen=True)
class Declaration:
    """What a component states about itself."""

    reads: tuple[str, ...] = ()  # beyond released records: `TRUTH` or `SCORER`
    key_coords: tuple[str, ...] = DRAW_COORDS  # coordinates that key its draws
    stops: tuple[tuple[Scope, Disposition], ...] = ()  # (scope, disposition) asked
    effects: frozenset = frozenset()  # `EffectKind` values it asks for
    needs: tuple[Need, ...] = ()  # products a decision reads at a fixed tick
    requires: frozenset = frozenset()  # executor capabilities it needs
    executors: frozenset = EXECUTORS  # executors it runs on
    disruptive: bool = False  # asks to break a frozen horizon (reserved)

    def __post_init__(self):
        """Reject unknown names so a typo cannot read as a clean declaration."""
        if set(self.reads) - READABLE:
            raise ValueError(f"reads must be drawn from {sorted(READABLE)}")
        if set(self.key_coords) - set(COORDS):
            raise ValueError("key_coords names an unknown coordinate")
        if set(self.executors) - EXECUTORS:
            raise ValueError(f"executors must be drawn from {sorted(EXECUTORS)}")


def declaration_of(component) -> Declaration:
    """The declaration of `component`, or the empty one."""
    declare = getattr(component, "declare", None)
    return declare() if callable(declare) else Declaration()
