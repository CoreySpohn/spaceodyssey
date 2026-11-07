"""Pre-mission tables: rows that describe the world before tick 0, and their import.

A `PreMissionTable` holds one row per datum known before the run starts: its source
identity (`SourceId`), the semantic index the run knows it by (`local_index`), its
channel, record kind and audience, its time span relative to its release, a payload,
and, for a prior, the pedigree of sources it summarizes.

`PreMissionImporter` is the process that brings a table into a run. Its `seed` queues
one process event per written row at tick 0 in the release stage; its `handle` writes
the row unset (`released_tick` and `accepted_tick` at -1), and the release event the
core queues behind it stamps the row at tick 0, so the observer folds a raw row once.
A public row of a derived kind (a prior) is written as two rows, one per audience,
with fixed metadata. The importer also moves the public kickoff wait into the release
stage, so the policy's first decision fires once, after the last import.

Identity rules:

- A source is named by `(namespace, producer, subject, source_index)`; the revision
  is recorded beside it and never enters overlap. Namespace 0 is this table, `k > 0`
  the k-th imported run, and `PAD` (-1) marks a padding row that is not imported.
- `local_index` is supplied, never computed from the imported rows: the record's
  `acquisition_index` is that value, so inserting or removing an observer-only row
  moves no public identity. It must be unique per channel.

The table's own checks run in pre-flight through `PreMissionImporter.validate`
(codes `D004` to `D007`).

Overlap (`resolve_overlap`): a prior is a present row of a derived kind, and its
pedigree names the sources it summarizes. For one audience, the evidence is the
present raw (non-derived) rows released to that audience. A prior is considered only
for the audiences it is released to, and its pedigree (external entries included) is
compared with that audience's evidence by source identity. `PedigreeRule` says which
side of an overlap is dropped.
"""

import dataclasses
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core import AbstractProcess, EventKind, Stage, enqueue, record_none
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import (
    F64,
    I64,
    PUBLIC,
    Audience,
    B,
    Capacity,
    RecordStatus,
)
from spaceodyssey.core.preflight import Finding, FindingCode, Severity

PAD = -1  # the namespace of a padding row
IDENTITY = ("namespace", "producer", "subject", "source_index")
AUDIENCES = (int(Audience.OBSERVER), int(Audience.POLICY), int(PUBLIC))


def _i64(x):
    return jnp.asarray(x, I64)


def _f64(x):
    return jnp.asarray(x, F64)


def _bool(x):
    return jnp.asarray(x, B)


def _concrete(*arrays):
    return not any(isinstance(a, jax.core.Tracer) for a in arrays)


class SourceId(eqx.Module):
    """Source identities, one per entry: who produced a datum and which one it is."""

    namespace: jax.Array = eqx.field(converter=_i64)
    producer: jax.Array = eqx.field(converter=_i64)  # its channel in its namespace
    subject: jax.Array = eqx.field(converter=_i64)
    source_index: jax.Array = eqx.field(converter=_i64)
    revision: jax.Array = eqx.field(converter=_i64)  # for lineage, never overlap

    def __check_init__(self):
        """Require one shape for every column."""
        shapes = {jnp.shape(self.namespace)}
        shapes |= {jnp.shape(x) for x in (self.producer, self.subject)}
        shapes |= {jnp.shape(x) for x in (self.source_index, self.revision)}
        if len(shapes) != 1:
            raise ValueError(f"SourceId columns need one shape, not {sorted(shapes)}")

    def keys(self):
        """The overlap identity of every entry as integer tuples (concrete arrays)."""
        cols = (self.namespace, self.producer, self.subject, self.source_index)
        stacked = np.stack([np.asarray(c) for c in cols], axis=-1)
        return [tuple(int(v) for v in row) for row in stacked.reshape(-1, 4)]


class PreMissionTable(eqx.Module):
    """Rows that describe the world before tick 0; see the module docstring.

    Every row column has shape `[N]`; `payload` is `[N, P]`, `pedigree` and
    `external` are `[N, W]`. Offsets are at or below 0 and relative to the row's
    release tick; windows are relative to the row's `released_tick`. A pedigree
    entry with `external` set names a source outside the table (provenance only);
    entries at or beyond `n_pedigree` are padding.
    """

    source: SourceId
    local_index: jax.Array = eqx.field(converter=_i64)
    channel: jax.Array = eqx.field(converter=_i64)
    kind: jax.Array = eqx.field(converter=_i64)
    audience: jax.Array = eqx.field(converter=_i64)
    start_offset: jax.Array = eqx.field(converter=_i64)
    end_offset: jax.Array = eqx.field(converter=_i64)
    window_lo_rel: jax.Array = eqx.field(converter=_i64)
    window_hi_rel: jax.Array = eqx.field(converter=_i64)
    payload: jax.Array = eqx.field(converter=_f64)
    pedigree: SourceId
    external: jax.Array = eqx.field(converter=_bool)
    n_pedigree: jax.Array = eqx.field(converter=_i64)
    upstream_hash: str = eqx.field(default="", static=True)

    def __check_init__(self):
        """Require consistent shapes and, on present rows, a known audience."""
        n = jnp.shape(self.local_index)
        if len(n) != 1:
            raise ValueError("local_index is one value per row")
        rows = {
            "source": jnp.shape(self.source.namespace),
            "channel": jnp.shape(self.channel),
            "kind": jnp.shape(self.kind),
            "audience": jnp.shape(self.audience),
            "start_offset": jnp.shape(self.start_offset),
            "end_offset": jnp.shape(self.end_offset),
            "window_lo_rel": jnp.shape(self.window_lo_rel),
            "window_hi_rel": jnp.shape(self.window_hi_rel),
            "n_pedigree": jnp.shape(self.n_pedigree),
        }
        for name, shape in rows.items():
            if shape != n:
                raise ValueError(f"{name} has shape {shape}; the table has {n[0]} rows")
        if jnp.ndim(self.payload) != 2 or jnp.shape(self.payload)[0] != n[0]:
            raise ValueError("payload is one row of values per table row")
        width = jnp.shape(self.pedigree.namespace)
        if len(width) != 2 or width[0] != n[0] or jnp.shape(self.external) != width:
            raise ValueError("pedigree and external are [rows, width] with one shape")
        if _concrete(self.audience, self.source.namespace):
            present = np.asarray(self.source.namespace) != PAD
            bad = ~np.isin(np.asarray(self.audience), AUDIENCES) & present
            if bad.any():
                i = int(np.argmax(bad))
                raise ValueError(
                    f"row {i} has audience {int(self.audience[i])}; a row is released "
                    "to the observer (1), the policy (2) or both (3)"
                )

    @property
    def n_rows(self):
        """Rows of the table, padding included."""
        return jnp.shape(self.local_index)[0]

    @property
    def present(self):
        """Boolean per row: the row is imported (not padding)."""
        return self.source.namespace != PAD


class PedigreeRule(IntEnum):
    """Which side of an overlap of a prior and its audience's evidence is dropped."""

    EXCLUDE_PRIOR = 0  # the prior, when its pedigree names any evidence row
    EXCLUDE_ROWS = 1  # every evidence row its pedigree names; the prior is kept


def _identity(ids):
    """The overlap identity `(namespace, producer, subject, source_index)` per entry."""
    cols = (ids.namespace, ids.producer, ids.subject, ids.source_index)
    return jnp.stack(cols, axis=-1)


def overlap_roles(table, audience, *, derived_kinds):
    """`(prior, evidence, names)` of `table` for one audience.

    `prior[p]`: row `p` is a prior released to `audience`; `evidence[r]`: row `r` is a
    raw row released to it; `names[p, r]`: prior `p`'s pedigree (its first
    `n_pedigree` entries, external ones included) names the source of evidence row
    `r`. Pure and traceable; the revision never enters.
    """
    if int(audience) not in (int(Audience.OBSERVER), int(Audience.POLICY)):
        raise ValueError(f"overlap is resolved for one audience, not {audience}")
    derived = jnp.isin(table.kind, jnp.asarray(tuple(derived_kinds), I64))
    released = table.present & ((table.audience & int(audience)) != 0)
    prior, evidence = released & derived, released & ~derived
    width = jnp.shape(table.external)[1]
    counted = jnp.arange(width)[None, :] < table.n_pedigree[:, None]
    same = _identity(table.pedigree)[:, :, None, :] == _identity(table.source)
    named = (same.all(axis=-1) & counted[:, :, None]).any(axis=1)
    return prior, evidence, named & prior[:, None] & evidence[None, :]


def resolve_overlap(table, rule, audience, *, derived_kinds):
    """Per-row weights in {0, 1} for one audience under `rule`; pure, order free.

    A prior or evidence row of `audience` keeps weight 1 unless `rule` drops it:
    `EXCLUDE_PRIOR` drops a prior whose pedigree names any of the audience's evidence,
    `EXCLUDE_ROWS` drops every evidence row a considered prior names and keeps the
    prior. A row the audience never receives (another audience's, or padding) has
    weight 0. `derived_kinds` are the study's derived kind indices
    (`Capacities.derived_kinds`), which tell a prior from a raw row.
    """
    rule = PedigreeRule(int(rule))
    prior, evidence, names = overlap_roles(table, audience, derived_kinds=derived_kinds)
    if rule == PedigreeRule.EXCLUDE_PRIOR:
        keep = jnp.where(prior, ~names.any(axis=1), evidence)
    else:
        keep = jnp.where(prior, True, evidence & ~names.any(axis=0))
    return (keep & (prior | evidence)).astype(I64)


def split_rows(table, caps):
    """Boolean per row: a public row of a derived kind, written once per audience."""
    derived = jnp.isin(table.kind, jnp.asarray(caps.derived_kinds, I64))
    return table.present & (table.audience == PUBLIC) & derived


@dataclasses.dataclass(frozen=True)
class ImportDemand:
    """What importing a table costs a run (concrete tables only).

    `rows` log rows are written (a split row counts twice), by `events` events (one
    process and one release per row) that take `ids` stable ids (the two events' and
    the row's own). `queue_peak` slots are held at once: every process row plus the
    kickoff wait, at tick 0. `steps` executor steps pass before the first decision.
    """

    rows: int
    events: int
    ids: int
    queue_peak: int
    steps: int

    @classmethod
    def of(cls, table, caps):
        """The demand of importing `table` under `caps`."""
        rows = int(np.sum(np.asarray(table.present))) + int(
            np.sum(np.asarray(split_rows(table, caps)))
        )
        return cls(rows, 2 * rows, 3 * rows, rows + 1, 2 * rows + 1)


def _retarget_kickoff(carry):
    """Move the public kickoff wait from the admit stage into the release stage.

    Imports wake the policy in the release stage; with the wait there too, every wake
    of tick 0 is pending when the last import is released, and one decision fires.
    """
    ev = carry.ev
    kick = ev.valid & (ev.kind == EventKind.WAIT) & (ev.tick == 0)
    kick = kick & (ev.stage == Stage.ADMIT)
    stage = jnp.where(kick, jnp.asarray(Stage.RELEASE, I64), ev.stage)
    return eqx.tree_at(lambda c: c.ev.stage, carry, stage)


class PreMissionImporter(AbstractProcess):
    """The process that imports a `PreMissionTable` at tick 0.

    Row `i` is written by the process event of group `i`. A public row of a derived
    kind is written twice, once for the observer and once for the policy, so the
    metadata of each audience's row never depends on the other audience. A written
    row takes the next stable id as its acquisition id (each row is its own
    acquisition), and the next process row is released at tick 0 by the core.
    """

    table: PreMissionTable

    def declare(self):
        """The record channels the imported rows are written on, for pre-flight."""
        present = np.asarray(self.table.present)
        channels = np.asarray(self.table.channel)[present]
        return Declaration(channels=tuple(sorted({int(c) for c in channels})))

    def seed(self, carry, caps):
        """Retarget the kickoff wait and queue one process event per written row."""
        carry = _retarget_kickoff(carry)
        table = self.table
        present = table.present
        split = split_rows(table, caps)
        for i in range(table.n_rows):
            first = jnp.where(split[i], Audience.OBSERVER, table.audience[i])
            carry = enqueue(
                carry,
                present[i],
                kind=EventKind.PROCESS,
                tick=0,
                stage=Stage.RELEASE,
                group=i,
                audience=first,
                caps=caps,
            )
            carry = enqueue(
                carry,
                split[i],
                kind=EventKind.PROCESS,
                tick=0,
                stage=Stage.RELEASE,
                group=i,
                audience=Audience.POLICY,
                caps=caps,
            )
        return carry

    def handle(self, carry, ev, study):
        """Write row `ev.group` unset; the core queues the release that stamps it."""
        caps = study.caps
        table = self.table
        if table.n_rows == 0:
            # nothing is queued for an empty table, whatever its payload width; the
            # handler is still traced, so it returns before reading any column
            return carry, record_none(caps)
        width = jnp.shape(table.payload)[1]
        if width > caps.payload_width:
            raise ValueError(
                f"the table's payload has {width} columns; the capacities hold "
                f"{caps.payload_width}"
            )
        g = ev.group
        room = carry.next_sid < caps.max_sid
        internal = carry.next_sid
        carry = eqx.tree_at(
            lambda c: (c.next_sid, c.overflow, c.overflow_mask),
            carry,
            (
                carry.next_sid + room.astype(I64),
                carry.overflow | ~room,
                carry.overflow_mask | jnp.where(room, 0, Capacity.STABLE_IDS),
            ),
        )
        base = record_none(caps)
        rec = dataclasses.replace(
            base,
            kind=table.kind[g],
            subject=table.source.subject[g],
            channel=table.channel[g],
            acquisition_index=table.local_index[g],
            start_tick=ev.tick + table.start_offset[g],
            end_tick=ev.tick + table.end_offset[g],
            available_tick=ev.tick,
            status=jnp.asarray(RecordStatus.COMPLETED, I64),
            acquisition_id=internal,
            revision=table.source.revision[g],
            released_tick=jnp.asarray(-1, I64),
            accepted_tick=jnp.asarray(-1, I64),
            superseded_by=jnp.asarray(-1, I64),
            payload=base.payload.at[:width].set(table.payload[g]),
            valid=room,
        )
        return carry, rec

    def validate(self, study, context):
        """Pre-flight findings of the table: `D004` to `D007`."""
        table = self.table
        out = table_findings(table, study.caps)
        out += lineage_findings(table, context)
        budget = None if context is None else context.budget
        if budget is not None:
            out += capacity_findings(table, study.caps, budget)
        return out


# ---- pre-flight -------------------------------------------------------------------


def _finding(code, member, entry, message, fix, severity=Severity.ERROR):
    where = "".join(f"[{i}]" for i in entry)
    return Finding(
        code,
        severity,
        f"process.{member}{where}",
        message,
        fix,
        subject="process",
        member=member,
        entry=tuple(entry),
    )


def table_findings(table, caps):
    """`D005`, `D006` and `D007` of a concrete table under the study's capacities."""
    present = np.asarray(table.present)
    keys = table.source.keys()
    out = []
    namespace = np.asarray(table.source.namespace)
    for i in np.flatnonzero((namespace < 0) & (namespace != PAD)):
        out.append(
            _finding(
                FindingCode.D006,
                "table.source.namespace",
                (int(i),),
                f"row {i} has namespace {int(namespace[i])}; a namespace is at least "
                f"0, or {PAD} for padding",
                "give the row its source's namespace, or pad it with -1",
            )
        )
    out += _identity_findings(table, present, keys)
    out += _range_findings(table, present, caps)
    out += _pedigree_findings(table, present, keys)
    out += _offset_findings(table, present)
    return out


def _identity_findings(table, present, keys):
    """`D006`: a repeated source identity, or a repeated index on one channel."""
    out, seen, indices = [], {}, {}
    channel = np.asarray(table.channel)
    local = np.asarray(table.local_index)
    for i in np.flatnonzero(present):
        i = int(i)
        if keys[i] in seen:
            out.append(
                _finding(
                    FindingCode.D006,
                    "table.source.source_index",
                    (i,),
                    f"row {i} has the source identity of row {seen[keys[i]]}; a "
                    "datum is one row, its latest revision",
                    "keep one row per (namespace, producer, subject, source index)",
                )
            )
        seen.setdefault(keys[i], i)
        pair = (int(channel[i]), int(local[i]))
        if pair in indices:
            out.append(
                _finding(
                    FindingCode.D006,
                    "table.local_index",
                    (i,),
                    f"row {i} repeats local index {pair[1]} of row {indices[pair]} "
                    f"on channel {pair[0]}",
                    "give each row of a channel its own local index",
                )
            )
        indices.setdefault(pair, i)
    return out


def _range_findings(table, present, caps):
    """`D006`: a row the study's capacities cannot hold."""
    out = []
    width = np.asarray(table.n_pedigree)
    limit = np.shape(table.external)[1]
    for i in np.flatnonzero(present & ((width < 0) | (width > limit))):
        out.append(
            _finding(
                FindingCode.D006,
                "table.n_pedigree",
                (int(i),),
                f"row {i} names {int(width[i])} pedigree entries; the width is {limit}",
                "name at most the pedigree width, and never fewer than none",
            )
        )
    subject = np.asarray(table.source.subject)
    for i in np.flatnonzero(present & ((subject < 0) | (subject >= caps.n_subjects))):
        out.append(
            _finding(
                FindingCode.D006,
                "table.source.subject",
                (int(i),),
                f"row {i} is about subject {int(subject[i])}; the study holds "
                f"{caps.n_subjects}",
                "map the source's subject to one of the study's subjects",
            )
        )
    known = len(caps.payload_audience)
    kind = np.asarray(table.kind)
    if known:
        for i in np.flatnonzero(present & ((kind < 0) | (kind >= known))):
            out.append(
                _finding(
                    FindingCode.D006,
                    "table.kind",
                    (int(i),),
                    f"row {i} has record kind {int(kind[i])}; the registry holds "
                    f"{known} kinds",
                    "register the kind and lower the registry into the capacities",
                )
            )
    columns = np.shape(table.payload)[1]
    if present.any() and columns > caps.payload_width:
        out.append(
            _finding(
                FindingCode.D006,
                "table.payload",
                (),
                f"the payload has {columns} columns; a record holds "
                f"{caps.payload_width}",
                "register a kind as wide as the payload, or narrow the payload",
            )
        )
    return out


def _pedigree_findings(table, present, keys):
    """`D005`: an in-table pedigree entry that names no row of the table."""
    out = []
    rows = {keys[int(i)] for i in np.flatnonzero(present)}
    named = table.pedigree.keys()
    width = np.shape(table.external)[1]
    external = np.asarray(table.external)
    count = np.clip(np.asarray(table.n_pedigree), 0, width)
    for i in np.flatnonzero(present):
        i = int(i)
        for j in range(int(count[i])):
            if external[i, j] or named[i * width + j] in rows:
                continue
            out.append(
                _finding(
                    FindingCode.D005,
                    "table.pedigree.source_index",
                    (i, j),
                    f"pedigree entry {j} of row {i} names source "
                    f"{list(named[i * width + j])}, which the table does not hold",
                    "import that source, or mark the entry external",
                )
            )
    return out


def _offset_findings(table, present):
    """`D007`: an offset above zero, or a start after its end."""
    out = []
    start = np.asarray(table.start_offset)
    end = np.asarray(table.end_offset)
    for name, values in (("start_offset", start), ("end_offset", end)):
        for i in np.flatnonzero(present & (values > 0)):
            out.append(
                _finding(
                    FindingCode.D007,
                    f"table.{name}",
                    (int(i),),
                    f"row {i} has {name} {int(values[i])}; a row ends at or before "
                    "its release",
                    "give offsets at or below zero",
                )
            )
    for i in np.flatnonzero(present & (start > end) & (start <= 0) & (end <= 0)):
        out.append(
            _finding(
                FindingCode.D007,
                "table.start_offset",
                (int(i),),
                f"row {i} starts at offset {int(start[i])}, after its end at "
                f"{int(end[i])}",
                "give a start at or before the end",
            )
        )
    return out


def lineage_findings(table, context):
    """`D004`: the declared upstream hash against the supplied upstream document.

    No declared hash, no finding. A declared hash with no upstream document is a
    warning, "declared, unverified"; one that differs from the document's content
    hash is an error.
    """
    if not table.upstream_hash:
        return []
    document = None if context is None else context.upstream_document
    if document is None:
        return [
            _finding(
                FindingCode.D004,
                "table.upstream_hash",
                (),
                f"upstream hash {table.upstream_hash} is declared, unverified",
                "supply the upstream document to verify the lineage",
                Severity.WARNING,
            )
        ]
    actual = getattr(document, "content_hash", None)
    if actual == table.upstream_hash:
        return []
    problem = getattr(document, "problem", None)
    found = f"document {problem}" if problem else f"document's content hash {actual}"
    return [
        _finding(
            FindingCode.D004,
            "table.upstream_hash",
            (),
            f"upstream hash {table.upstream_hash} differs from the supplied upstream "
            f"{found}",
            "export the table from that document, or supply the document it names",
        )
    ]


def capacity_findings(table, caps, budget):
    """`D006`: an import larger than the run's queue, log, stable ids or steps."""
    need = ImportDemand.of(table, caps)
    out = []
    checks = (
        ("queue slots", need.queue_peak, budget.n_events, "raise n_events"),
        ("log rows", need.rows, budget.n_log, "raise n_log"),
        ("stable ids", need.ids + 1, caps.max_sid, "raise max_sid"),
        ("executor steps", need.steps, budget.n_steps, "run for more steps"),
    )
    for what, wanted, held, fix in checks:
        if held is not None and wanted > held:
            out.append(
                _finding(
                    FindingCode.D006,
                    "table",
                    (),
                    f"the import needs {wanted} {what} before the first decision; "
                    f"the run has {held}",
                    f"{fix}, or import fewer rows",
                )
            )
    return out


PREMISSION_KINDS = {
    ("process", "premission.importer", 1): PreMissionImporter,
    ("table", "premission.table", 1): PreMissionTable,
    ("table", "premission.source_id", 1): SourceId,
}
"""The kind references of the importer and its table, for `dump_study` and
`load_study` (merge them into the mapping or resolver a document is read with)."""


def _payload_rows(payload, n):
    """The payload as `[n, P]`, refusing any other rank or row count.

    An empty table may give an empty payload; it is one column wide.
    """
    values = np.asarray(payload, np.float64)
    if n == 0 and values.size == 0:
        return values.reshape(0, values.shape[1] if values.ndim == 2 else 1)
    if values.ndim != 2 or values.shape[0] != n:
        raise ValueError(
            f"payload has shape {values.shape}; it needs one row of values per "
            f"table row, [{n}, P]"
        )
    return values


def authored_table(
    *,
    subject,
    local_index,
    channel,
    kind,
    audience,
    payload,
    start_offset=None,
    end_offset=None,
    window_lo_rel=None,
    window_hi_rel=None,
    namespace=None,
    producer=None,
    source_index=None,
    revision=None,
    pedigree=(),
    width=1,
    upstream_hash="",
):
    """A table written by hand: one entry per row in each column.

    The source identity defaults to namespace 0, producer equal to the channel and
    source index equal to the local index. `pedigree` holds, per row, a sequence of
    `(namespace, producer, subject, source_index, external)` entries (at most `width`
    each); an empty `pedigree` gives every row none. Offsets and windows default to 0.
    """
    n = len(local_index)
    zeros = [0] * n

    def col(value, default):
        return default if value is None else value

    entries = list(pedigree) or [()] * n
    if len(entries) != n:
        raise ValueError(f"pedigree has {len(entries)} rows; the table has {n}")
    grid = np.full((n, width, 4), PAD, np.int64)
    external = np.zeros((n, width), bool)
    counts = np.zeros(n, np.int64)
    for i, row in enumerate(entries):
        if len(row) > width:
            raise ValueError(f"row {i} has {len(row)} pedigree entries; width {width}")
        counts[i] = len(row)
        for j, (*ident, outside) in enumerate(row):
            grid[i, j] = ident
            external[i, j] = bool(outside)
    return PreMissionTable(
        source=SourceId(
            namespace=col(namespace, zeros),
            producer=col(producer, channel),
            subject=subject,
            source_index=col(source_index, local_index),
            revision=col(revision, zeros),
        ),
        local_index=local_index,
        channel=channel,
        kind=kind,
        audience=audience,
        start_offset=col(start_offset, zeros),
        end_offset=col(end_offset, zeros),
        window_lo_rel=col(window_lo_rel, zeros),
        window_hi_rel=col(window_hi_rel, zeros),
        payload=_payload_rows(payload, n),
        pedigree=SourceId(
            namespace=grid[..., 0],
            producer=grid[..., 1],
            subject=grid[..., 2],
            source_index=grid[..., 3],
            revision=np.zeros((n, width), np.int64),
        ),
        external=external,
        n_pedigree=counts,
        upstream_hash=upstream_hash,
    )


__all__ = [
    "IDENTITY",
    "PAD",
    "PREMISSION_KINDS",
    "ImportDemand",
    "PedigreeRule",
    "PreMissionImporter",
    "PreMissionTable",
    "SourceId",
    "authored_table",
    "capacity_findings",
    "lineage_findings",
    "overlap_roles",
    "resolve_overlap",
    "split_rows",
    "table_findings",
]
