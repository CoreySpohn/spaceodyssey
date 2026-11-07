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
upstream document (`upstream_hash`). `verify_lineage` reads a downstream document,
decodes each table in it (so every file reference is checked: a path in a name, a
missing file, a symlink or a wrong digest is refused at its pointer) and compares the
declared hash with the supplied upstream document's content hash. A mismatch is `D004`
at the pointer of the declared hash; with no upstream document the lineage is
"declared, unverified".
"""

import dataclasses
import hashlib

import numpy as np

from spaceodyssey.codec.document import DocumentError, pointer
from spaceodyssey.codec.files import DEFAULT_MAX_BYTES, make_store, using_store
from spaceodyssey.codec.kinds import KindResolver
from spaceodyssey.codec.load import decode
from spaceodyssey.codec.preflight import (
    DocumentCode,
    DocumentFinding,
    read_document,
    read_upstream,
)
from spaceodyssey.core.enums import PUBLIC, Audience, RecordStatus
from spaceodyssey.core.preflight import FindingCode, Severity
from spaceodyssey.core.preflight.checks import PreflightContext
from spaceodyssey.core.records import payload_columns
from spaceodyssey.studies.premission import (
    PAD,
    PREMISSION_KINDS,
    PreMissionTable,
    SourceId,
    lineage_findings,
)

EPOCHS = ("end_at_zero",)
_TABLE_KINDS = {ref: cls for ref, cls in PREMISSION_KINDS.items() if ref[0] == "table"}
_TABLE_REF = next(ref for ref, cls in _TABLE_KINDS.items() if cls is PreMissionTable)
_RESOLVER = KindResolver(_TABLE_KINDS)


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

    `producer_map` maps each exported upstream channel to its downstream channel and
    `subject_map` each exported upstream subject to a downstream subject; rows on
    other channels or of other subjects are not exported. `namespace` is the import's
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
    mapped &= np.isin(log["subject"], np.asarray(sorted(subject_map), np.int64))
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
    """The table with downstream subjects and channels, refusing an index clash."""
    subjects, channels = [], []
    for i in rows:
        s, c = int(log["subject"][i]), int(log["channel"][i])
        subjects.append(int(subject_map[s]))
        channels.append(int(producer_map[c]))
    seen = {}
    local = np.asarray(table.local_index).tolist()
    for j, pair in enumerate(zip(channels, local, strict=True)):
        if pair in seen:
            raise ValueError(
                f"rows {seen[pair]} and {j} share local index {pair[1]} on downstream "
                f"channel {pair[0]}: a local index is the upstream acquisition index, "
                "which is counted per subject, and it must be unique per channel; "
                "select one subject per downstream channel with subject_map"
            )
        seen[pair] = j
    source = dataclasses.replace(
        table.source, subject=np.asarray(subjects, np.int64).reshape(-1)
    )
    return dataclasses.replace(
        table, source=source, channel=np.asarray(channels, np.int64).reshape(-1)
    )


def table_digest(table):
    """The sha256 of a table's columns in row order, and its declared upstream hash.

    It names one table whatever its encoding (inline or file-backed): reordering
    rows, or changing any value, gives another digest.
    """
    h = hashlib.sha256()
    for name, leaf in _columns(table, ""):
        arr = np.ascontiguousarray(np.asarray(leaf))
        h.update(f"{name}:{arr.dtype.name}:{list(arr.shape)};".encode("ascii"))
        h.update(arr.tobytes())
    h.update(f"upstream_hash:{table.upstream_hash}".encode())
    return h.hexdigest()


def _columns(module, prefix):
    """`(name, array)` for every array field of a table, sources included."""
    for field in dataclasses.fields(module):
        value = getattr(module, field.name)
        if isinstance(value, SourceId):
            yield from _columns(value, f"{prefix}{field.name}.")
        elif field.name != "upstream_hash":
            yield f"{prefix}{field.name}", value


# ---- lineage ------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FileReference:
    """An array file a table names: its pointer, file name, sha256 and size."""

    pointer: str
    name: str
    sha256: str
    bytes: int


@dataclasses.dataclass(frozen=True)
class TableLineage:
    """One table of a document: where it is, what it declares and what it holds."""

    pointer: str
    upstream_hash: str
    digest: str
    files: tuple[FileReference, ...]


@dataclasses.dataclass(frozen=True)
class LineageReport:
    """What `verify_lineage` found: the upstream hash, each table and the findings.

    `upstream` is the supplied upstream document as read (`None` when none was
    supplied). `status` is `"failed"` when any finding is an error, `"no lineage"`
    when no table declares an upstream hash, `"declared, unverified"` when no upstream
    document was supplied, and `"verified"` otherwise.
    """

    upstream: object
    tables: tuple[TableLineage, ...]
    findings: tuple[DocumentFinding, ...]

    @property
    def status(self):
        """The verdict, as described in the class docstring."""
        if any(f.severity == Severity.ERROR for f in self.findings):
            return "failed"
        if not any(t.upstream_hash for t in self.tables):
            return "no lineage"
        if self.upstream is None:
            return "declared, unverified"
        return "verified"

    @property
    def verified(self):
        """Whether every declared upstream hash was checked and matched."""
        return self.status == "verified"


def _table_nodes(node, path):
    """`(path, node)` of every encoded `PreMissionTable` under a document node."""
    if isinstance(node, list):
        for i, item in enumerate(node):
            yield from _table_nodes(item, (*path, i))
    elif isinstance(node, dict):
        ref = node.get("$kind")
        if isinstance(ref, list) and ref[:2] == list(_TABLE_REF[:2]):
            yield path, node
        elif "$array" not in node:
            for key, item in node.items():
                yield from _table_nodes(item, (*path, key))


def _file_refs(node, path):
    """Every array file reference under an encoded node, with its pointer."""
    if isinstance(node, list):
        for i, item in enumerate(node):
            yield from _file_refs(item, (*path, i))
    elif isinstance(node, dict):
        if "$array" in node and isinstance(node.get("file"), dict):
            ref = node["file"]
            where = pointer((*path, "file"))
            yield FileReference(where, ref["name"], ref["sha256"], ref["bytes"])
        else:
            for key, item in node.items():
                yield from _file_refs(item, (*path, key))


def _refused(err):
    """The finding of a table that does not decode: its own code, or `L001`."""
    return DocumentFinding(
        DocumentCode(err.code) if err.code else DocumentCode.L001,
        Severity.ERROR,
        err.pointer or "",
        err.detail,
        "restore the table and its array files as they were written",
    )


def verify_lineage(
    document,
    upstream_document=None,
    *,
    expected=None,
    files=None,
    max_file_bytes=DEFAULT_MAX_BYTES,
):
    """Check every table's declared upstream hash in a study or run document.

    `document` is JSON text, a parsed object, a `StudyDocument` or a `RunDocument`;
    `files` is the directory of its array files. Each `PreMissionTable` in the study
    is decoded as the loader decodes it, so a file reference whose name holds a path,
    or that names a missing file, a symlink or bytes that do not match its sha256, is
    refused (`L001` at the pointer of the reference). A stored `content_hash` that
    does not match the document is `L001` too.

    `upstream_document` is the document the tables name (any form
    `read_upstream` takes). Its content hash is compared with each declared
    `upstream_hash`: a mismatch, or an upstream document that does not read, is
    `D004` at the pointer of the declared hash; with no upstream document, a declared
    hash is a `D004` warning, declared, unverified. `expected` is the table the
    upstream run exports, or its `table_digest`: a table with a declared hash whose
    digest differs (rows reordered or changed) is `D004` at the table's pointer.

    Never raises on document content; returns a `LineageReport`.
    """
    data, found = read_document(document)
    upstream = None if upstream_document is None else read_upstream(upstream_document)
    if data is None:
        return LineageReport(upstream, (), tuple(found))
    want = expected
    if expected is not None and not isinstance(expected, str):
        want = table_digest(expected)
    context = PreflightContext(upstream_document=upstream)
    store = make_store(files, 0, max_file_bytes)
    tables = []
    for path, node in _table_nodes(data.get("study"), ("study",)):
        try:
            with using_store(store):
                table = decode(node, path, _RESOLVER)
        except DocumentError as err:
            found.append(_refused(err))
            continue
        for item in lineage_findings(table, context):
            found.append(
                DocumentFinding(
                    item.code,
                    item.severity,
                    pointer((*path, "upstream_hash")),
                    item.message,
                    item.fix,
                    in_memory=item,
                )
            )
        digest = table_digest(table)
        if want is not None and table.upstream_hash and digest != want:
            found.append(
                DocumentFinding(
                    FindingCode.D004,
                    Severity.ERROR,
                    pointer(path),
                    f"the table's digest {digest} differs from the expected export's "
                    f"{want}: its rows were changed or reordered",
                    "import the table exactly as the upstream run exports it",
                )
            )
        refs = tuple(_file_refs(node, path))
        tables.append(TableLineage(pointer(path), table.upstream_hash, digest, refs))
    found.sort(key=lambda f: (f.severity != Severity.ERROR, f.code, f.pointer))
    return LineageReport(upstream, tuple(tables), tuple(found))


__all__ = [
    "EPOCHS",
    "Export",
    "ExportedRevision",
    "FileReference",
    "LineageReport",
    "TableLineage",
    "export_records",
    "table_digest",
    "verify_lineage",
]
