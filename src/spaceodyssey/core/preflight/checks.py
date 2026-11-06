"""Structural checks.

Each check reads the declarations of a `Study` and its `StudyGraph`; none runs the
study. A passing pre-flight says the declarations are consistent, never that the
science is right.
"""

import numpy as np

from spaceodyssey.core.declarations import (
    EXECUTOR_CAPABILITIES,
    EXECUTORS,
    SUPPORTED_EFFECTS,
    declaration_of,
)
from spaceodyssey.core.enums import (
    Audience,
    CoincidentOrder,
    Disposition,
    EffectKind,
    Scope,
)
from spaceodyssey.core.preflight.findings import Finding, FindingCode, Severity
from spaceodyssey.core.preflight.graph import NodeKind, Side, StudyGraph
from spaceodyssey.core.study import Study

ERROR, WARNING = Severity.ERROR, Severity.WARNING


def _components(study):
    """(path, component) pairs that carry declarations."""
    pairs = [
        ("world_source", study.world_source),
        ("process", study.process),
        ("channel_model", study.channel_model),
        ("observer", study.observer),
        ("objective", study.objective),
        ("design", study.design),
    ]
    pairs += [(f"policy[{i}]", p) for i, p in enumerate(study.agenda.policies)]
    return pairs


def _slot(name):
    """(slot, index) of a component name: `policy[2]` is the policy slot, index 2."""
    if name.startswith("policy["):
        return "policy", int(name[len("policy[") : -1])
    return name, None


def firewall(graph: StudyGraph):
    """F001 and F002: what an observer-side node may read."""
    out = []
    for edge in graph.edges:
        reader, source = graph.nodes[edge.reader], graph.nodes[edge.source]
        if reader.side != Side.OBSERVER:
            continue
        path = f"{edge.reader} <- {edge.source}"
        slot, index = _slot(edge.reader)
        where = dict(subject=slot, index=index, member="reads", entry=edge.entry)
        if source.side == Side.SCORER:
            out.append(
                Finding(
                    FindingCode.F002,
                    ERROR,
                    path,
                    f"{edge.reader} reads the scorer {edge.source}",
                    "remove the read; the scorer joins credits to truth, "
                    "never the reverse",
                    **where,
                )
            )
        elif source.side == Side.TRUTH and not source.crossing:
            out.append(
                Finding(
                    FindingCode.F001,
                    ERROR,
                    path,
                    f"{edge.reader} reads {edge.source} outside a measure crossing",
                    "route the read through a channel that realizes it as a record",
                    **where,
                )
            )
    return out


def causality(study: Study, graph: StudyGraph):
    """C001: a decision reads a product before it is released to the policy."""
    out = []
    table = study.channels
    ids = np.asarray(table.id).tolist()
    for i, policy in enumerate(study.agenda.policies):
        for j, need in enumerate(declaration_of(policy).needs):
            path = f"policy[{i}] <- channel id {need.channel_id}"
            where = dict(subject="policy", index=i, member="needs", entry=(j,))
            if need.channel_id not in ids:
                out.append(
                    Finding(
                        FindingCode.C001,
                        ERROR,
                        path,
                        f"the channel table has no channel {need.channel_id}",
                        "read a channel the study declares",
                        **where,
                    )
                )
                continue
            c = ids.index(need.channel_id)
            if not int(table.audience[c]) & Audience.POLICY:
                out.append(
                    Finding(
                        FindingCode.C001,
                        ERROR,
                        path,
                        f"channel {need.channel_id} is never released to the policy",
                        "release the channel to the policy audience, or read a "
                        "different channel",
                        **where,
                    )
                )
                continue
            latency = int(table.latency_ticks[c])
            completed = need.start_tick + int(table.occupancy_ticks[c]) + latency
            # A cancel can end an admitted occupancy even during its transition.
            lane = int(table.lane[c])
            transition = int(study.assets.transition_ticks[lane])
            admitted = max(0, need.start_tick - transition)
            ready = min(completed, admitted + max(latency, 1))
            if ready > need.decide_tick:
                out.append(
                    Finding(
                        FindingCode.C001,
                        ERROR,
                        path,
                        f"the product is released no earlier than tick {ready} but "
                        f"the decision is at tick {need.decide_tick}",
                        "decide later, or read an earlier product",
                        **where,
                    )
                )
    return out


def keys(graph: StudyGraph):
    """K001 and K002: stable-id coordinates, distinct per measure."""
    out, seen = [], {}
    for node in graph.nodes.values():
        if node.kind not in (NodeKind.MEASURE, NodeKind.SOURCE):
            continue
        missing = [c for c in node.required_coords if c not in node.key_coords]
        if missing:
            out.append(
                Finding(
                    FindingCode.K001,
                    ERROR,
                    node.name,
                    f"draws are not keyed by {', '.join(missing)}",
                    "key every draw by its stable ids; see spaceodyssey.core.keys",
                    subject=node.slot,
                    member="key_coords",
                )
            )
        elif node.kind == NodeKind.MEASURE:
            signature = (frozenset(node.key_coords), node.channel_id)
            if signature in seen:
                out.append(
                    Finding(
                        FindingCode.K002,
                        ERROR,
                        node.name,
                        f"same key coordinates as {seen[signature]}; the draws "
                        "would be correlated",
                        "give each channel its own id",
                        subject="channels",
                        member="id",
                        entry=node.entry,
                    )
                )
            else:
                seen[signature] = node.name
    return out


def record_channels(study: Study):
    """K002: admission, scripted, service and product channels are disjoint.

    A record's channel is part of its identity and of every key drawn for it, so two
    producers writing on one channel id would share identities. The owners, in order:
    the admission channels of the channel table, the channel a process writes scripted
    records on, the service channel each policy asks a `Query` on, and the two product
    channels of the refresh protocol (when it is on). Two admission channels with one id
    are `keys`' finding; here a channel id used by a later owner than an earlier one is
    reported at the later one, with the earlier one named.
    """
    owned = {}  # channel id -> name of its first owner
    for i, cid in enumerate(np.asarray(study.channels.id).tolist()):
        owned.setdefault(int(cid), f"admission channel {i}")
    out = []

    def claim(channel, name, **where):
        channel = int(channel)
        if channel in owned:
            out.append(
                Finding(
                    FindingCode.K002,
                    ERROR,
                    name,
                    f"channel id {channel} is already the {owned[channel]}; records "
                    "of two producers would share an identity",
                    "give admission, scripted, service and product channels "
                    "disjoint ids",
                    **where,
                )
            )
        else:
            owned[channel] = name

    for channel in declaration_of(study.process).channels:
        claim(channel, "scripted channel of the process", subject="process")
    for i, policy in enumerate(study.agenda.policies):
        for channel in declaration_of(policy).channels:
            claim(
                channel,
                f"service channel of policy[{i}]",
                subject="policy",
                index=i,
            )
    if study.caps.rows_per_subject > 0:
        sched = study.refresh
        claim(
            sched.observer_channel,
            "observer product channel",
            subject="refresh",
            member="observer_channel",
        )
        claim(
            sched.policy_channel,
            "policy product channel",
            subject="refresh",
            member="policy_channel",
        )
    return out


def capability(study: Study, executor: str | None = None):
    """X001: effects and capabilities the selected executors do not provide.

    With `executor` set, the study is checked against that executor; otherwise a
    requirement fails only when no executor provides it.
    """
    chosen = EXECUTORS if executor is None else frozenset({executor})
    out = []
    for path, component in _components(study):
        decl = declaration_of(component)
        slot, index = _slot(path)
        for kind in sorted(decl.effects, key=int):
            if EffectKind(kind) not in SUPPORTED_EFFECTS:
                out.append(
                    Finding(
                        FindingCode.X001,
                        ERROR,
                        path,
                        f"asks for {EffectKind(kind).name.capitalize()}, which no "
                        "executor supports",
                        "remove the effect; admission answers it as unsupported",
                        subject=slot,
                        index=index,
                        member="effects",
                    )
                )
        provided = set().union(*(EXECUTOR_CAPABILITIES[e] for e in chosen))
        for name in sorted(decl.requires - provided):
            out.append(
                Finding(
                    FindingCode.X001,
                    ERROR,
                    path,
                    f"requires {name!r}, which the selected executor "
                    f"({', '.join(sorted(chosen))}) does not provide",
                    "choose an executor that provides it, or drop the requirement",
                    subject=slot,
                    index=index,
                    member="requires",
                )
            )
        if not chosen & decl.executors:
            out.append(
                Finding(
                    FindingCode.X001,
                    ERROR,
                    path,
                    f"runs only on {', '.join(sorted(decl.executors))}, "
                    f"not on {', '.join(sorted(chosen))}",
                    "select an executor the component supports",
                    subject=slot,
                    index=index,
                    member="executors",
                )
            )
    return out


def reserved(study: Study):
    """X002: reserved fields hold their defaults."""
    out = []

    def flag(path, what, subject, member, index=None, entry=()):
        out.append(
            Finding(
                FindingCode.X002,
                ERROR,
                path,
                f"reserved field set to {what}; no rule reads it yet",
                "restore the default; the field is ignored until a rule reads it",
                subject=subject,
                index=index,
                member=member,
                entry=entry,
            )
        )

    ch, assets = study.channels, study.assets
    charge, cost = np.asarray(ch.charge), np.asarray(ch.cost)
    if not np.array_equal(charge, cost):
        differs = np.argwhere(charge != cost) if charge.shape == cost.shape else []
        entry = tuple(int(n) for n in differs[0]) if len(differs) else ()
        flag(
            "channels.charge",
            "a value that differs from cost",
            "channels",
            "charge",
            entry=entry,
        )
    for name, column, default, what in (
        ("freeze_ticks", assets.freeze_ticks, 0, "a nonzero horizon"),
        ("disruptive_resource", assets.disruptive_resource, -1, "a resource row"),
    ):
        found = np.argwhere(np.asarray(column) != default)
        if len(found):
            entry = tuple(int(n) for n in found[0])
            flag(f"assets.{name}", what, "assets", name, entry=entry)
    for i, policy in enumerate(study.agenda.policies):
        if declaration_of(policy).disruptive:
            flag(f"policy[{i}]", "a disruptive effect", "policy", "disruptive", index=i)
    return out


def stops(study: Study):
    """T001: a drain below campaign scope behaves as a close."""
    out = []
    for i, policy in enumerate(study.agenda.policies):
        for j, (scope, disposition) in enumerate(declaration_of(policy).stops):
            if disposition == Disposition.DRAIN and scope != Scope.CAMPAIGN:
                out.append(
                    Finding(
                        FindingCode.T001,
                        WARNING,
                        f"policy[{i}]",
                        f"Drain at {Scope(scope).name.lower()} scope behaves as Close",
                        "ask for Close, or stop at campaign scope",
                        subject="policy",
                        index=i,
                        member="stops",
                        entry=(j,),
                    )
                )
    return out


def channels(study: Study):
    """T002 and R001: coincident order and record kinds of the channel table."""
    out = []
    table = study.channels
    order = table.coincident_order
    n = np.asarray(table.id).shape[0]
    valid = {int(v) for v in CoincidentOrder}
    values = None if order is None else np.asarray(order)
    if values is None or values.shape != (n,) or not set(values.tolist()) <= valid:
        invalid = ()
        if values is not None and values.shape == (n,):
            invalid = (
                next(i for i, v in enumerate(values.tolist()) if v not in valid),
            )
        out.append(
            Finding(
                FindingCode.T002,
                ERROR,
                "channels.coincident_order",
                "every channel needs a coincident order of its own",
                "set SETTLE_FIRST or FAULT_FIRST per channel; there is no default",
                subject="channels",
                member="coincident_order",
                entry=invalid,
            )
        )
    known = len(study.caps.payload_audience)
    for i, kind in enumerate(np.asarray(table.kind).tolist()):
        if not 0 <= kind < known:
            out.append(
                Finding(
                    FindingCode.R001,
                    ERROR,
                    f"channels.kind[{i}]",
                    f"channel {i} declares record kind {kind}, but the registry "
                    f"holds {known} kinds",
                    "register the kind and lower the registry into the capacities",
                    subject="channels",
                    member="kind",
                    entry=(i,),
                )
            )
    return out


def preflight(study: Study, executor: str | None = None) -> list[Finding]:
    """Structural findings for `study`, errors first. Nothing runs."""
    graph = StudyGraph.from_study(study)
    found = [
        *firewall(graph),
        *causality(study, graph),
        *keys(graph),
        *record_channels(study),
        *capability(study, executor),
        *reserved(study),
        *stops(study),
        *channels(study),
    ]
    return sorted(found, key=lambda f: (f.severity != ERROR, f.code))
