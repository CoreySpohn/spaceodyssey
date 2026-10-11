"""The fixed seam vocabulary of the mission pipeline.

A seam is one pluggable stage of the simulation: implementations of a seam are
an open, unordered registry (see :mod:`spaceodyssey.registry`), but the seam names
themselves and the dataflow edges between them are fixed. Adding a seam is a
deliberate API event, not a runtime extension point.

The typed edges below cover only the data plane, where concrete dataset kinds
flow between stages and compatibility must be checked. The belief plane
(policies, cost models, belief updates) is kind-agnostic by construction:
every dataset exposes a likelihood factory, so nothing downstream of the
belief needs to know what kind of data produced it.
"""

SEAMS: tuple[str, ...] = (
    "scene",
    "instrument",
    "observation",
    "post_processing",
    "characterization",
    "certification",
    "policy",
    "population",
    "context",
    "cost",
)

DATA_PLANE_EDGES: tuple[tuple[str, str], ...] = (
    ("instrument", "observation"),
    ("scene", "observation"),
    ("observation", "post_processing"),
    ("observation", "characterization"),
    ("post_processing", "certification"),
)

KNOWN_OPERATORS: frozenset[str] = frozenset({"sample", "propagate", "integrate"})
"""Outcome operators with engines today.

The operator vocabulary is open: a new evaluation semantics enters as a new
engine registry entry (see :mod:`spaceodyssey.engines`), so nothing here or
elsewhere enumerates operators exhaustively.
"""
