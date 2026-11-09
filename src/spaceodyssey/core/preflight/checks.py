"""Structural checks.

Each check reads the declarations of a `Study` and its `StudyGraph`; none runs the
study. A passing pre-flight says the declarations are consistent, never that the
science is right.
"""

import dataclasses

import numpy as np

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import (
    EXECUTOR_CAPABILITIES,
    EXECUTORS,
    GATED_EFFECTS,
    SUPPORTED_EFFECTS,
    declaration_of,
)
from spaceodyssey.core.enums import (
    Audience,
    ChargePricing,
    CoincidentOrder,
    Disposition,
    EffectKind,
    Scope,
)
from spaceodyssey.core.ledger import CONFIG_PAD
from spaceodyssey.core.preflight.findings import Finding, FindingCode, Severity
from spaceodyssey.core.preflight.graph import NodeKind, Side, StudyGraph
from spaceodyssey.core.study import Study

ERROR, WARNING = Severity.ERROR, Severity.WARNING


@dataclasses.dataclass(frozen=True)
class RunBudget:
    """What a run may use: event slots, log rows and executor steps.

    `n_steps` is `None` when the step budget is not known (a document records none);
    a check of it then does not run.
    """

    n_events: int
    n_log: int
    n_steps: int | None = None

    @classmethod
    def of(cls, study, n_steps=None):
        """The budget of `study` run for `n_steps` steps, from its capacities."""
        steps = None if n_steps is None else int(n_steps)
        return cls(study.caps.n_events, study.caps.n_log, steps)


@dataclasses.dataclass(frozen=True)
class PreflightContext:
    """What a run supplies beside the study, for checks the study alone cannot make.

    `boundary` is the refit boundary the executor is given, `budget` the run's
    `RunBudget`, and `upstream_document` the document a declared upstream hash names.
    Each is optional; a check that needs a missing one does not run, or reports that
    it could not verify.
    """

    boundary: object = None
    budget: RunBudget | None = None
    upstream_document: object = None


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
    if study.caps.commitments and study.caps.awards:
        out.append(
            Finding(
                FindingCode.X001,
                ERROR,
                "caps",
                "holds commitments and awards together, which no executor supports",
                "turn one of them off; a commitment's debit and an award's window "
                "are not yet accounted together",
                subject="caps",
                member="n_commitments",
            )
        )
    for path, component in _components(study):
        decl = declaration_of(component)
        slot, index = _slot(path)
        for kind in sorted(decl.effects, key=int):
            if EffectKind(kind) not in SUPPORTED_EFFECTS or (
                EffectKind(kind) in GATED_EFFECTS and not study.caps.awards
            ):
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
    if not study.caps.refunds:
        # the charge and the refund declarations are read only while refunds are on
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
            (
                "pricing",
                ch.pricing,
                ChargePricing.OCCUPANCY,
                "a pricing other than occupancy",
            ),
            ("surcharge", ch.surcharge, 0, "a nonzero surcharge"),
            ("failure_kind", ch.failure_kind, -1, "a failure kind"),
            (
                "refund_deadline_ticks",
                ch.refund_deadline_ticks,
                0,
                "a nonzero deadline",
            ),
        ):
            found = [] if column is None else np.argwhere(np.asarray(column) != default)
            if len(found):
                entry = tuple(int(n) for n in found[0])
                flag(f"channels.{name}", what, "channels", name, entry=entry)
    if not study.caps.commitments:
        # the horizon and the disruptive resource are read only while commitments are on
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
                flag(
                    f"policy[{i}]",
                    "a disruptive effect",
                    "policy",
                    "disruptive",
                    index=i,
                )
    return out


def asset_columns(study: Study):
    """A001: a disruptive resource that is neither unset nor a resource column."""
    if not study.caps.commitments:
        return []
    column = np.asarray(study.assets.disruptive_resource)
    out = []
    for entry in np.argwhere((column < -1) | (column >= study.caps.n_resources)):
        out.append(
            Finding(
                FindingCode.A001,
                ERROR,
                "assets.disruptive_resource",
                "disruptive resource names a column the study does not hold",
                "use -1 for none, or a resource column below n_resources",
                subject="assets",
                member="disruptive_resource",
                entry=tuple(int(n) for n in entry),
            )
        )
    return out


def failure_kinds(study: Study):
    """W002: a declared failure kind that the policy cannot read.

    A refund is credited when its receipt is released to the policy, so a kind no
    payload column of which the policy may read could never settle one.
    """
    out = []
    kinds = study.channels.failure_kind
    if not study.caps.refunds or kinds is None:
        return out
    table = study.caps.payload_audience
    for i, kind in enumerate(np.asarray(kinds).tolist()):
        if kind < 0:
            continue
        seen = not table or (
            0 <= kind < len(table)
            and any(f & int(Audience.POLICY) for f in table[kind])
        )
        if not seen:
            out.append(
                Finding(
                    FindingCode.W002,
                    ERROR,
                    "channels.failure_kind",
                    f"failure kind {kind} of channel {i} is not readable by the policy",
                    "declare a policy-visible kind; a receipt settles when released "
                    "to the policy",
                    subject="channels",
                    member="failure_kind",
                    entry=(i,),
                )
            )
    return out


def balances(study: Study):
    """X002: a balance quantum set while the snapshot shows no balances."""
    caps = study.caps
    if caps.snapshot_balances or caps.balance_quantum == Capacities().balance_quantum:
        return []
    return [
        Finding(
            FindingCode.X002,
            ERROR,
            "caps.balance_quantum",
            f"balance_quantum is {caps.balance_quantum} but snapshot_balances is off; "
            "no rule reads it",
            "restore the default, or turn snapshot_balances on",
            subject="caps",
            member="balance_quantum",
        )
    ]


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


def programs(study: Study):
    """P001: a program id a policy names outside `0 <= program < n_programs`."""
    out = []
    n = study.caps.n_programs
    for i, policy in enumerate(study.agenda.policies):
        for j, program in enumerate(declaration_of(policy).programs):
            if not 0 <= program < n:
                out.append(
                    Finding(
                        FindingCode.P001,
                        ERROR,
                        f"policy[{i}]",
                        f"program {program} is outside the {n} the capacities hold",
                        f"name a program from 0 to {n - 1}, or raise n_programs",
                        subject="policy",
                        index=i,
                        member="programs",
                        entry=(j,),
                    )
                )
    return out


def bundles(study: Study):
    """B001: a bundle a policy declares that admission would refuse.

    A declared member is `(ordinal, lane, group, program, start offset)`; members that
    share an ordinal are one bundle. It needs bundles on, an ordinal below
    `max_bundles`, no more members than `max_bundle_size`, one lane, one admission
    group and one program, and an immediate start (offset 0).

    A declaration carries no decision index, so members are merged by ordinal across the
    whole policy declaration, while the run scopes a bundle to one decision: a policy
    that reuses an ordinal on different lanes in different decisions is flagged here
    although the run admits both. Declare one distinct ordinal per bundle shape to
    avoid it. The run treats any negative ordinal as not bundled; this check flags a
    declared negative ordinal as out of range.
    """
    out = []
    caps = study.caps

    def flag(i, j, why):
        out.append(
            Finding(
                FindingCode.B001,
                ERROR,
                f"policy[{i}]",
                f"bundle member {j} {why}",
                "give a bundle ordinal below max_bundles, at most max_bundle_size "
                "members on one lane, group and program, and an immediate start",
                subject="policy",
                index=i,
                member="bundles",
                entry=(j,),
            )
        )

    for i, policy in enumerate(study.agenda.policies):
        opener = {}  # ordinal -> (index, lane, group, program) of its first member
        size = {}
        for j, (ordinal, lane, group, program, offset) in enumerate(
            declaration_of(policy).bundles
        ):
            opened = opener.setdefault(ordinal, (j, lane, group, program))
            size[ordinal] = size.get(ordinal, 0) + 1
            top = caps.max_bundles - 1
            if not caps.bundles:
                flag(i, j, "is declared while max_bundles is 0")
            elif not 0 <= ordinal <= top:
                flag(i, j, f"names ordinal {ordinal}, outside 0 to {top}")
            elif (lane, group, program) != opened[1:]:
                flag(i, j, f"differs from member {opened[0]} in lane, group or program")
            elif offset != 0:
                flag(i, j, f"starts {offset} ticks after its decision")
            elif size[ordinal] > caps.max_bundle_size:
                flag(i, j, f"exceeds the {caps.max_bundle_size} members a bundle holds")
    return out


def config_timeline(study: Study):
    """S001: a configuration timeline the capacities and horizon cannot hold.

    With `max_config_switches` zero the assets declare no switch ticks. With it positive
    each asset declares a row of exactly that many ticks, strictly increasing (two
    switches at one tick would skip a configuration id), inside the horizon
    (0 to `horizon_tick`), the unused tail padded with `CONFIG_PAD`.
    """
    caps = study.caps
    ticks = study.assets.config_switch_ticks

    def flag(why, entry=()):
        return [
            Finding(
                FindingCode.S001,
                ERROR,
                "assets.config_switch_ticks",
                why,
                "declare max_config_switches and one strictly increasing row of switch ticks "
                "inside the horizon per asset, padded with CONFIG_PAD",
                subject="assets",
                member="config_switch_ticks",
                entry=entry,
            )
        ]

    if not caps.configs:
        return [] if ticks is None else flag("is set while max_config_switches is 0")
    shape = (caps.n_assets, caps.max_config_switches)
    if ticks is None or tuple(np.shape(ticks)) != shape:
        return flag(f"is not an array of shape {shape}")
    horizon = int(study.horizon_tick)
    out = []
    for a, row in enumerate(np.asarray(ticks)):
        real = row[row != CONFIG_PAD]
        if len(real) and (real[0] < 0 or real[-1] > horizon):
            out += flag(f"holds a tick outside 0 to {horizon}", (a,))
        elif (np.diff(row) < 0).any() or (np.diff(real) == 0).any():
            out += flag("is not strictly increasing with its padding last", (a,))
    return out


def ops_costs(study: Study):
    """O001: a per-ops-state cost table the capacities cannot hold.

    With `n_ops` 1 no table is declared. With more states a declared `cost_by_ops` has
    shape `(n_ops, n_channels, n_resources)` and finite entries that are not negative
    (a negative one would credit the ledger).
    """
    caps = study.caps
    table = study.channels.cost_by_ops

    def flag(why, entry=()):
        return [
            Finding(
                FindingCode.O001,
                ERROR,
                "channels.cost_by_ops",
                why,
                "declare n_ops and a finite, nonnegative cost table of shape "
                "(n_ops, n_channels, n_resources), or drop the table",
                subject="channels",
                member="cost_by_ops",
                entry=entry,
            )
        ]

    if table is None:
        return []
    if not caps.ops:
        return flag("is set while n_ops is 1")
    shape = (caps.n_ops, caps.n_channels, caps.n_resources)
    if tuple(np.shape(table)) != shape:
        return flag(f"is not an array of shape {shape}")
    values = np.asarray(table, dtype=np.float64)
    bad = np.argwhere(~(np.isfinite(values) & (values >= 0.0)))
    if len(bad):
        return flag("holds an entry that is negative or not finite", tuple(bad[0]))
    return []


def awards(study: Study):
    """W001: an award window a policy declares that cannot be admitted.

    A window is `(recipient, start, until)`: the recipient must be a balance row, the
    window nonempty (`0 <= start < until`) and inside the run's last tick, and no two
    windows on one row may overlap (a row holds one window at a time).
    """
    out = []
    caps = study.caps

    def flag(i, j, why):
        out.append(
            Finding(
                FindingCode.W001,
                ERROR,
                f"policy[{i}]",
                f"award {j} {why}",
                "give the award a balance row, a nonempty window inside the run and "
                "no overlap with another award on that row",
                subject="policy",
                index=i,
                member="awards",
                entry=(j,),
            )
        )

    for i, policy in enumerate(study.agenda.policies):
        taken = {}
        for j, (row, start, until) in enumerate(declaration_of(policy).awards):
            if not 0 <= row < caps.n_balance_rows:
                last = caps.n_balance_rows - 1
                flag(i, j, f"names balance row {row}, outside 0 to {last}")
            elif not 0 <= start < until <= caps.max_tick:
                flag(
                    i, j, f"has the window [{start}, {until}), empty or outside the run"
                )
            elif any(start < u and s < until for s, u in taken.get(row, ())):
                flag(i, j, f"overlaps an earlier award on balance row {row}")
            else:
                taken.setdefault(row, []).append((start, until))
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


def components(study: Study, context: PreflightContext | None = None):
    """Findings the process and the observer report on themselves.

    A component that defines `validate(study, context)` returns its own findings.
    A finding that leaves its slot empty is relative to the validating component: the
    slot is filled in and its member and entry are kept. A finding that names no slot
    of the study, or a policy that is missing or out of range, cannot be located
    there, so it is located at the validating component's slot alone (its member and
    entry dropped; its path and message kept). A member the slot does not hold is left
    to the document pointer walk, which stops at the deepest value it can resolve.
    `context` is passed as given, `None` included.
    """
    out = []
    for slot, component in (("process", study.process), ("observer", study.observer)):
        validate = getattr(component, "validate", None)
        if not callable(validate):
            continue
        for item in validate(study, context):
            if not item.subject:
                item = dataclasses.replace(item, subject=slot, index=None)
            elif not _locates(item, study):
                item = dataclasses.replace(
                    item, subject=slot, index=None, member="", entry=()
                )
            out.append(item)
    return out


# The fields of `Study`, which a finding's `subject` may name (the policy slot names
# `agenda.policies` by index); a test keeps the list equal to the dataclass's fields.
STUDY_SLOTS = frozenset(
    {
        "caps",
        "clock",
        "channels",
        "channel_model",
        "assets",
        "budget",
        "horizon_tick",
        "agenda",
        "process",
        "observer",
        "design",
        "world_source",
        "objective",
        "coupling",
        "fork",
        "refresh",
        "key_version",
    }
)


def _locates(item, study):
    """Whether a finding's slot exists: a field of the study, or a policy by index."""
    if item.subject == "policy":
        index = item.index
        return isinstance(index, int) and 0 <= index < len(study.agenda.policies)
    return item.subject in STUDY_SLOTS


def preflight(
    study: Study,
    executor: str | None = None,
    context: PreflightContext | None = None,
) -> list[Finding]:
    """Structural findings for `study`, errors first. Nothing runs.

    `context` carries what a run supplies beside the study (`PreflightContext`); the
    process's and observer's own checks receive it.
    """
    graph = StudyGraph.from_study(study)
    found = [
        *firewall(graph),
        *causality(study, graph),
        *keys(graph),
        *record_channels(study),
        *capability(study, executor),
        *reserved(study),
        *asset_columns(study),
        *failure_kinds(study),
        *balances(study),
        *stops(study),
        *programs(study),
        *awards(study),
        *bundles(study),
        *config_timeline(study),
        *ops_costs(study),
        *channels(study),
        *components(study, context),
    ]
    return sorted(found, key=lambda f: (f.severity != ERROR, f.code))
