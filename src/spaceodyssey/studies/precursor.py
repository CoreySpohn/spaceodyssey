"""Precursor tables: rows exported from an upstream run, and their lineage.

`export_records` turns the record log of a finished upstream run into a
`PreMissionTable` that a downstream run imports. A datum of the upstream run is named
by its channel (the producer), its subject and its acquisition index (the source
index). For each datum the table holds one row, the latest revision released to the
chosen audience by the end of the upstream run; earlier revisions, and data never
released to that audience, are kept only in the export's revision list, so a
correction is one datum, not two measurements. The upstream run ends at the
downstream's tick 0: a row's offsets are its ticks minus the upstream horizon.

The table travels as a field of the downstream study. Large columns are written to
sha256-named files by the codec's generic walk (`dump_study(..., files=...)`), and the
document's hash covers their references. The table declares the content hash of the
upstream document (`upstream_hash`).
"""

import dataclasses

import numpy as np

from spaceodyssey.core.enums import PUBLIC, Audience, RecordStatus
from spaceodyssey.core.records import payload_columns
from spaceodyssey.studies.premission import (
    PAD,
    PreMissionTable,
    SourceId,
)

EPOCHS = ("end_at_zero",)


# ---- export -------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExportedRevision:
    """One upstream row of an exported datum, and what the export did with it.

    `producer`, `subject` and `source_index` are the upstream channel, subject and
    acquisition index. `released` says the row reached every chosen audience by the
    end of the upstream run. `row` is the table row it became, or -1; `reason` is
    `"exported"`, `"superseded"` (a later released revision was exported),
    `"withdrawn"` (the latest released revision is not a completed record, so the
    datum is not exported) or `"not released"`.
    """

    producer: int
    subject: int
    source_index: int
    revision: int
    status: int
    released: bool
    row: int
    reason: str


@dataclasses.dataclass(frozen=True)
class Export:
    """The exported `table` and every upstream revision it considered."""

    table: PreMissionTable
    revisions: tuple[ExportedRevision, ...]


def _log_of(run_result):
    """The record log of one world: a `RunResult`, a `Carry` or a `RecordLog`."""
    log = getattr(run_result, "carry", run_result)
    log = getattr(log, "log", log)
    if not hasattr(log, "released_tick"):
        raise TypeError("export_records needs a RunResult, a Carry or a RecordLog")
    if np.ndim(log.kind) != 1:
        raise ValueError("export_records reads one world; index the world first")
    return {
        name: np.asarray(getattr(log, name))
        for name in (
            "kind",
            "subject",
            "channel",
            "acquisition_index",
            "start_tick",
            "end_tick",
            "status",
            "revision",
            "released_tick",
            "accepted_tick",
            "audience",
            "payload",
            "valid",
        )
    }


def _audiences(audience):
    """The single audiences a chosen audience mask names (observer, policy, both)."""
    if audience not in (int(Audience.OBSERVER), int(Audience.POLICY), int(PUBLIC)):
        raise ValueError(
            f"audience {audience} is not the observer (1), the policy (2) or both (3)"
        )
    return [a for a in (int(Audience.OBSERVER), int(Audience.POLICY)) if audience & a]


def _released_to(log, single, horizon):
    """Boolean per row: released to one audience at a tick from 0 to the horizon.

    The observer's release is the row's `accepted_tick` and the policy's its
    `released_tick`; the core's release handler stamps each only for a row addressed
    to that audience.
    """
    stamp = log["accepted_tick" if single == Audience.OBSERVER else "released_tick"]
    return (stamp >= 0) & (stamp <= horizon)


def export_records(
    run_result,
    kind,
    audience,
    upstream_document_hash,
    subject_map,
    producer_map,
    epoch="end_at_zero",
    *,
    study,
    namespace=1,
):
    """Export the rows of one record kind from a finished upstream run as an `Export`.

    `run_result` is the upstream run (a `RunResult`, `Carry` or `RecordLog` of one
    world) and `study` the upstream study, which gives the horizon and the payload
    columns each audience may read. `kind` is the upstream record kind exported, and
    the kind index of the table rows; a derived kind is refused (a table prior needs
    a pedigree an export cannot supply). `audience` is the observer, the policy or
    both: a revision qualifies when it was released to each of them by the end of
    the run, and the exported row is released to exactly that audience, never to one
    the upstream row did not reach. Payload columns that some exported audience could
    not read upstream are written as zero.

    `producer_map` maps each exported upstream channel to its downstream channel
    (rows on other channels are not exported); `subject_map` maps every upstream
    subject of an exported row to a downstream subject. `namespace` is the import's
    namespace (at least 1; 0 is a table written by hand). `upstream_document_hash`
    is the upstream document's content hash, declared as the table's
    `upstream_hash`. `epoch="end_at_zero"` (the only epoch) places the upstream
    horizon at the downstream's tick 0.

    The table holds one row per datum, ordered by (channel, subject, acquisition
    index); its `local_index` is the upstream acquisition index and its source
    identity is `(namespace, upstream channel, downstream subject, upstream
    acquisition index)` with the revision exported.
    """
    if epoch not in EPOCHS:
        raise ValueError(f"epoch {epoch!r} is not one of {EPOCHS}")
    if type(upstream_document_hash) is not str or not upstream_document_hash:
        raise ValueError("upstream_document_hash is the upstream document's hash")
    if int(namespace) < 1:
        raise ValueError("an imported run's namespace is at least 1")
    caps = study.caps
    kind, audience = int(kind), int(audience)
    if kind in caps.derived_kinds:
        raise ValueError(f"kind {kind} is derived; an export holds raw rows only")
    singles = _audiences(audience)
    horizon = int(study.horizon_tick)
    log = _log_of(run_result)
    mapped = np.isin(log["channel"], np.asarray(sorted(producer_map), np.int64))
    chosen = log["valid"] & (log["kind"] == kind) & mapped
    released = chosen.copy()
    for single in singles:
        released &= _released_to(log, single, horizon)

    data = {}
    for i in np.flatnonzero(chosen):
        key = (int(log["channel"][i]), int(log["subject"][i]))
        key += (int(log["acquisition_index"][i]),)
        data.setdefault(key, []).append(int(i))
    rows, revisions = [], []
    for key in sorted(data):
        members = sorted(data[key], key=lambda i: int(log["revision"][i]))
        shown = [i for i in members if released[i]]
        latest = shown[-1] if shown else None
        if latest is not None:
            same = [i for i in shown if log["revision"][i] == log["revision"][latest]]
            if len(same) > 1:
                raise ValueError(f"datum {key} has two released rows of one revision")
        keep = latest is not None and log["status"][latest] == RecordStatus.COMPLETED
        if keep:
            rows.append(latest)
        for i in members:
            if not released[i]:
                reason = "not released"
            elif i != latest:
                reason = "superseded"
            else:
                reason = "exported" if keep else "withdrawn"
            revisions.append(
                ExportedRevision(
                    *key,
                    revision=int(log["revision"][i]),
                    status=int(log["status"][i]),
                    released=bool(released[i]),
                    row=len(rows) - 1 if reason == "exported" else -1,
                    reason=reason,
                )
            )
    table = _table(log, rows, kind, audience, singles, horizon, caps, namespace)
    table = dataclasses.replace(table, upstream_hash=upstream_document_hash)
    table = _mapped(table, log, rows, subject_map, producer_map)
    return Export(table, tuple(revisions))


def _table(log, rows, kind, audience, singles, horizon, caps, namespace):
    """The table of the selected log rows, in upstream coordinates."""
    rows = np.asarray(rows, np.int64)
    n = len(rows)
    readable = np.ones(caps.payload_width, bool)
    for single in singles:
        readable &= np.asarray(payload_columns(caps, kind, single))
    payload = np.where(readable, log["payload"][rows], 0.0)
    pad = np.full((n, 1), PAD, np.int64)
    return PreMissionTable(
        source=SourceId(
            namespace=np.full(n, int(namespace), np.int64),
            producer=log["channel"][rows],
            subject=log["subject"][rows],
            source_index=log["acquisition_index"][rows],
            revision=log["revision"][rows],
        ),
        local_index=log["acquisition_index"][rows],
        channel=log["channel"][rows],
        kind=np.full(n, kind, np.int64),
        audience=np.full(n, audience, np.int64),
        start_offset=log["start_tick"][rows] - horizon,
        end_offset=log["end_tick"][rows] - horizon,
        window_lo_rel=np.zeros(n, np.int64),
        window_hi_rel=np.zeros(n, np.int64),
        payload=payload.reshape(n, caps.payload_width),
        pedigree=SourceId(pad, pad, pad, pad, np.zeros((n, 1), np.int64)),
        external=np.zeros((n, 1), bool),
        n_pedigree=np.zeros(n, np.int64),
    )


def _mapped(table, log, rows, subject_map, producer_map):
    """The table with downstream subjects and channels, refusing a gap or a clash."""
    subjects, channels = [], []
    for i in rows:
        s, c = int(log["subject"][i]), int(log["channel"][i])
        if s not in subject_map:
            raise ValueError(f"upstream subject {s} has no downstream subject")
        subjects.append(int(subject_map[s]))
        channels.append(int(producer_map[c]))
    seen = {}
    local = np.asarray(table.local_index).tolist()
    for j, pair in enumerate(zip(channels, local, strict=True)):
        if pair in seen:
            raise ValueError(
                f"rows {seen[pair]} and {j} share local index {pair[1]} on downstream "
                f"channel {pair[0]}; map their producers to different channels"
            )
        seen[pair] = j
    source = dataclasses.replace(
        table.source, subject=np.asarray(subjects, np.int64).reshape(-1)
    )
    return dataclasses.replace(
        table, source=source, channel=np.asarray(channels, np.int64).reshape(-1)
    )


__all__ = [
    "EPOCHS",
    "Export",
    "ExportedRevision",
    "export_records",
]
