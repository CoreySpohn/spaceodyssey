"""Study graph.

A `Study` compiles to a small typed graph: a source for the world and the process, one
measure per channel, an update for the observer, a decide per agenda phase and a score
for the objective. Each node states its side (truth, observer or scorer); an edge says
the reader takes input from the source. The only edges from the truth side to the
observer side leave a node marked as a crossing.
"""

import dataclasses
from enum import StrEnum

import numpy as np

from spaceodyssey.core.declarations import TRUTH, declaration_of
from spaceodyssey.core.enums import Audience
from spaceodyssey.core.study import Study


class NodeKind(StrEnum):
    """What a node does."""

    SOURCE = "source"
    MEASURE = "measure"
    UPDATE = "update"
    DECIDE = "decide"
    SCORE = "score"


class Side(StrEnum):
    """Which party owns a node."""

    TRUTH = "truth"
    OBSERVER = "observer"
    SCORER = "scorer"


@dataclasses.dataclass(frozen=True)
class Node:
    """One node: its role, side and the coordinates that key its draws."""

    name: str
    kind: NodeKind
    side: Side
    crossing: bool = False  # the sanctioned truth-to-observer boundary
    key_coords: tuple[str, ...] = ()
    required_coords: tuple[str, ...] = ()  # stable-id coordinates its draws need
    channel_id: int | None = None


@dataclasses.dataclass(frozen=True)
class Edge:
    """The reader takes input from the source."""

    reader: str
    source: str


@dataclasses.dataclass(frozen=True)
class StudyGraph:
    """Nodes by name and the edges between them."""

    nodes: dict
    edges: tuple[Edge, ...]

    @classmethod
    def from_study(cls, study: Study) -> "StudyGraph":
        """Compile the declarations of `study`; nothing runs."""
        nodes = {}
        edges = []

        def add(node):
            nodes[node.name] = node

        world = declaration_of(study.world_source)
        add(
            Node(
                "world",
                NodeKind.SOURCE,
                Side.TRUTH,
                key_coords=world.key_coords,
                required_coords=("world",),
            )
        )
        process = declaration_of(study.process)
        add(
            Node(
                "process",
                NodeKind.SOURCE,
                Side.TRUTH,
                crossing=True,
                key_coords=process.key_coords,
                required_coords=("world",),
            )
        )
        model = declaration_of(study.channel_model)
        ids = np.asarray(study.channels.id)
        audiences = np.asarray(study.channels.audience)
        for i, cid in enumerate(ids.tolist()):
            add(
                Node(
                    f"channel[{i}]",
                    NodeKind.MEASURE,
                    Side.TRUTH,
                    crossing=True,
                    key_coords=model.key_coords,
                    required_coords=("subject", "channel", "acquisition_index"),
                    channel_id=int(cid),
                )
            )
            edges.append(Edge(f"channel[{i}]", "world"))
        add(Node("observer", NodeKind.UPDATE, Side.OBSERVER))
        edges.append(Edge("observer", "process"))
        phases = []
        for i in range(len(study.agenda.policies)):
            name = f"policy[{i}]"
            phases.append(name)
            add(Node(name, NodeKind.DECIDE, Side.OBSERVER))
            edges.append(Edge(name, "observer"))
        add(Node("objective", NodeKind.SCORE, Side.SCORER))
        edges += [Edge("objective", "world"), Edge("objective", "observer")]

        for i, audience in enumerate(audiences.tolist()):
            if int(audience) & Audience.OBSERVER:
                edges.append(Edge("observer", f"channel[{i}]"))
            if int(audience) & Audience.POLICY:
                edges += [Edge(name, f"channel[{i}]") for name in phases]

        reading = {"observer": study.observer}
        reading.update({f"policy[{i}]": p for i, p in enumerate(study.agenda.policies)})
        for name, component in reading.items():
            for read in declaration_of(component).reads:
                edges.append(Edge(name, "world" if read == TRUTH else "objective"))
        return cls(nodes, tuple(edges))
