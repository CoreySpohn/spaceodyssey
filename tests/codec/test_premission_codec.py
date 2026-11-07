"""B5: pre-mission tables through the codec, with file references and lineage.

A table is a study field: the codec's generic walk writes its large columns to
sha256-named files, and the document hash covers their references. Its declared
upstream hash is checked against a supplied upstream document (`D004`), in memory and
through the JSON pre-flight, at the pointer of the declared hash. No existing corpus
document hash and no schema version moves.
"""

import dataclasses
import json
import os
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.codec import (
    SCHEMA_VERSION,
    DocumentError,
    KindResolver,
    dump_study,
    load_study,
    preflight_document,
)
from spaceodyssey.codec.load import same_value
from spaceodyssey.codec.preflight import (
    UpstreamDocument,
    read_upstream,
    resolve_pointer,
)
from spaceodyssey.core.enums import RecordStatus
from spaceodyssey.core.preflight import preflight
from spaceodyssey.core.preflight.checks import PreflightContext
from spaceodyssey.core.records import empty_log
from spaceodyssey.studies import toy
from spaceodyssey.studies.conformance import CORPUS
from spaceodyssey.studies.precursor import export_records
from spaceodyssey.studies.premission import (
    PREMISSION_KINDS,
    PreMissionImporter,
    authored_table,
)
from tests.codec.test_document import TOY_KINDS
from tests.codec.test_run_document_runs import EXECUTORS, _corpus_document, _hash_pin

RESOLVER = KindResolver({**TOY_KINDS, **PREMISSION_KINDS})
UPSTREAM = dump_study(toy.survey_1(), TOY_KINDS)
OTHER = dump_study(toy.survey_1(horizon_tick=900), TOY_KINDS)
DECLARED = "/study/process/table/upstream_hash"
PAYLOAD = "/study/process/table/payload/file"
N_ROWS = 6


def _table(declared=UPSTREAM.content_hash, order=None):
    """Six public measurement rows on channel 60, one per local index."""
    rows = np.arange(N_ROWS) if order is None else np.asarray(order)
    return authored_table(
        subject=(rows % 4).tolist(),
        local_index=rows.tolist(),
        channel=[60] * N_ROWS,
        kind=[toy.RECORD_MEASUREMENT] * N_ROWS,
        audience=[3] * N_ROWS,
        payload=np.stack([rows * 0.5, rows * 0.0, -rows - 1.0, rows * 0.0], 1),
        start_offset=(-10 - rows).tolist(),
        end_offset=(-rows).tolist(),
        upstream_hash=declared,
    )


def _study(table):
    return toy.survey_1(process=PreMissionImporter(table=table))


def _stored(tmp_path, table=None):
    """The import study's document text, its table arrays in files under tmp_path."""
    table = _table() if table is None else table
    doc = dump_study(_study(table), RESOLVER, files=tmp_path, max_inline=2)
    return doc.to_json()


def _codes(found):
    return [(str(f.code), str(f.severity), f.pointer) for f in found]


# ---- the table through the codec ----------------------------------------------------


def test_the_table_round_trips_through_file_references(tmp_path):
    """Fails if a table column is not stored as a sha256 file reference, if the
    reloaded table differs in any value, dtype or its declared hash, if a reload
    re-dumps to other text, or if a document with references loads without its files.
    """
    table = _table()
    text = _stored(tmp_path, table)
    node = json.loads(text)["study"]["process"]["table"]
    for column in ("local_index", "payload", "start_offset"):
        ref = node[column]["file"]
        raw = (tmp_path / ref["name"]).read_bytes()
        assert ref["bytes"] == len(raw) and ref["name"] == f"{ref['sha256']}.bin"
    assert node["upstream_hash"] == UPSTREAM.content_hash
    again = load_study(text, RESOLVER, files=tmp_path)
    assert same_value(again.process.table, table)
    redump = dump_study(again, RESOLVER, files=tmp_path, max_inline=2).to_json()
    assert redump == text
    with pytest.raises(DocumentError) as info:
        load_study(text, RESOLVER)
    assert info.value.pointer.endswith("/file")


def test_every_corpus_document_keeps_its_pinned_hash_and_schema_version():
    """Fails if any corpus run document's content hash, built without running, moves
    from `hash_pin.json`, or the schema version differs from the pinned one (K4).
    """
    pin = _hash_pin()
    assert pin["schema_version"] == SCHEMA_VERSION
    for case in CORPUS:
        for executor in EXECUTORS:
            doc = _corpus_document(case, executor)
            assert doc.content_hash == pin["documents"][f"{case.name}/{executor}"]


# ---- D004 through the JSON pre-flight ----------------------------------------------


def test_the_json_preflight_reaches_d004_at_the_declared_hash(tmp_path):
    """Fails if the JSON pre-flight does not pass with the upstream document the table
    names, report exactly one `D004` error at the declared hash (equal to the in-memory
    finding) with another, a warning with none, and an error naming the failure with
    an upstream document that does not read.
    """
    text = _stored(tmp_path)
    study = load_study(text, RESOLVER, files=tmp_path)

    def found(upstream):
        return preflight_document(text, RESOLVER, files=tmp_path, upstream=upstream)

    assert found(UPSTREAM) == []
    assert found(UPSTREAM.to_json()) == []
    wrong = found(OTHER)
    assert _codes(wrong) == [("D004", "error", DECLARED)]
    assert resolve_pointer(json.loads(text), DECLARED) == UPSTREAM.content_hash
    context = PreflightContext(upstream_document=OTHER)
    assert wrong[0].in_memory in preflight(study, context=context)
    assert OTHER.content_hash in wrong[0].message
    assert _codes(found(None)) == [("D004", "warning", DECLARED)]
    assert "declared, unverified" in found(None)[0].message
    broken = found("{not json")
    assert _codes(broken) == [("D004", "error", DECLARED)]
    assert "does not read as a document" in broken[0].message


@pytest.mark.parametrize(
    "edit",
    ["value", "reordered", "stale_hash"],
)
def test_a_tampered_or_reordered_upstream_document_has_another_hash(edit):
    """Fails if a changed value, reordered array elements or a stored hash that no
    longer matches leaves the upstream document's hash as declared, or if the same
    document written with other whitespace and key order does not keep it.
    """
    data = json.loads(UPSTREAM.to_json())
    same = json.dumps(dict(reversed(list(data.items()))), indent=2)
    assert read_upstream(same) == UpstreamDocument(UPSTREAM.content_hash)
    if edit == "value":
        data["study"]["horizon_tick"]["data"] = 999
        data.pop("content_hash")
    elif edit == "reordered":
        ids = data["study"]["channels"]["id"]["data"]
        data["study"]["channels"]["id"]["data"] = ids[::-1]
        data.pop("content_hash")
    else:
        data["study"]["horizon_tick"]["data"] = 999
    read = read_upstream(json.dumps(data))
    assert read.content_hash != UPSTREAM.content_hash
    if edit == "stale_hash":
        assert read.content_hash is None and "/content_hash" in read.problem


# ---- file references are safe ------------------------------------------------------


def _outside(tmp_path, name, raw):
    """A file of the right bytes outside the files directory."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(exist_ok=True)
    path = elsewhere / name
    path.write_bytes(raw)
    return path


def _damage(how, data, files, tmp_path):
    """Damage the payload reference or its file; return the pointer it is refused at."""
    ref = resolve_pointer(data, PAYLOAD)
    path = files / ref["name"]
    raw = path.read_bytes()
    if how == "traversal":
        _outside(tmp_path, ref["name"], raw)
        ref["name"] = f"../elsewhere/{ref['name']}"
        return f"{PAYLOAD}/name"
    if how == "absolute":
        ref["name"] = str(_outside(tmp_path, ref["name"], raw))
        return f"{PAYLOAD}/name"
    if how == "symlink":
        target = _outside(tmp_path, ref["name"], raw)
        path.unlink()
        os.symlink(target, path)
        return f"{PAYLOAD}/name"
    if how == "missing":
        path.unlink()
        return f"{PAYLOAD}/name"
    if how == "digest":
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        return f"{PAYLOAD}/sha256"
    # rows reordered in the file: same size, same bytes, another order
    row = len(raw) // N_ROWS
    path.write_bytes(raw[row:] + raw[:row])
    return f"{PAYLOAD}/sha256"


DAMAGE = ["traversal", "absolute", "symlink", "missing", "digest", "reordered"]


@pytest.mark.parametrize("how", DAMAGE)
def test_an_unsafe_or_damaged_file_reference_is_refused_at_its_pointer(tmp_path, how):
    """Fails if a reference whose name holds a relative or absolute path, a symlink
    out of the directory, a missing file, or bytes that do not match the sha256 (also
    the same rows reordered) is read, or is reported anywhere but at its pointer, or
    raises from the pre-flight instead of an `L001` finding.
    """
    files = tmp_path / "files"
    data = json.loads(_stored(files))
    data.pop("content_hash")
    where = _damage(how, data, files, tmp_path)
    text = json.dumps(data)
    found = preflight_document(text, RESOLVER, files=files, upstream=UPSTREAM)
    assert _codes(found) == [("L001", "error", where)]
    with pytest.raises(DocumentError) as info:
        load_study(text, RESOLVER, files=files)
    assert info.value.pointer == where


# ---- export from an upstream run ----------------------------------------------------

MEASUREMENT = toy.RECORD_MEASUREMENT
SOURCE = toy.survey_1()  # the upstream study: horizon 1000
SUBJECTS = {0: 1, 1: 2, 2: 3}
PRODUCERS = {10: 60}
ROW = {
    "kind": MEASUREMENT,
    "channel": 10,
    "status": int(RecordStatus.COMPLETED),
    "valid": True,
}


def _row(subject, index, revision, tick=200, audience=3, **fields):
    """A measurement released at `tick` to the audiences it names (-1: never)."""
    return {
        **ROW,
        "subject": subject,
        "acquisition_index": index,
        "revision": revision,
        "audience": audience,
        "accepted_tick": tick if audience & 1 else -1,
        "released_tick": tick if audience & 2 else -1,
        "start_tick": tick - 20,
        "end_tick": tick - 5,
        # value, a scorer-only variate, a public column, a column nobody reads
        "payload": [10.0 * subject + index + 0.1 * revision, 7.0, -1.0 - revision, 5.0],
        **fields,
    }


def _log(*rows):
    base = empty_log(SOURCE.caps)
    cols = {f.name: np.array(getattr(base, f.name)) for f in dataclasses.fields(base)}
    for i, row in enumerate(rows):
        for name, value in row.items():
            cols[name][i] = value
    return dataclasses.replace(base, **{k: jnp.asarray(v) for k, v in cols.items()})


LOG = _log(
    _row(0, 0, 0, 200),  # corrected: revision 1 at 300 is exported
    _row(0, 0, 1, 300),
    _row(1, 1, 0, 200),  # revision 1 after the end, revision 2 never released
    _row(1, 1, 1, 1200),
    _row(1, 1, 2, -1),
    _row(2, 3, 0, 200),  # revision 1 released to the observer only
    _row(2, 3, 1, 250, audience=1),
    _row(0, 4, 0, -1),  # never released
    _row(0, 2, 0, 200),  # the latest released revision failed: withdrawn
    _row(0, 2, 1, 260, status=int(RecordStatus.FAILED)),
    _row(3, 5, 0, 200, kind=2),  # another kind
    _row(3, 6, 0, 200, channel=11),  # a channel the export does not map
    _row(3, 7, 0, 200, valid=False),  # not a row
)


def _export(audience, study=SOURCE, log=LOG, **kw):
    return export_records(
        SimpleNamespace(carry=SimpleNamespace(log=log)),
        MEASUREMENT,
        audience,
        UPSTREAM.content_hash,
        kw.pop("subject_map", SUBJECTS),
        kw.pop("producer_map", PRODUCERS),
        study=study,
        **kw,
    )


def _ints(x):
    return np.asarray(x).tolist()


def test_export_keeps_the_latest_revision_released_to_the_audience():
    """Fails if a datum is exported at another revision than the latest released to
    the policy by the end of the run (a correction twice, a revision released after
    the end, an observer-only revision), if a withdrawn, unreleased, other-kind or
    unmapped row is exported, or if the dropped revisions are not listed with why.
    """
    export = _export(2)
    table = export.table
    assert _ints(table.source.namespace) == [1, 1, 1]
    assert _ints(table.source.producer) == [10, 10, 10]
    assert _ints(table.source.subject) == [1, 2, 3]
    assert _ints(table.source.source_index) == [0, 1, 3]
    assert _ints(table.source.revision) == [1, 0, 0]
    assert _ints(table.local_index) == [0, 1, 3]
    assert _ints(table.channel) == [60, 60, 60]
    assert _ints(table.kind) == [MEASUREMENT] * 3
    assert _ints(table.n_pedigree) == [0, 0, 0]
    assert table.upstream_hash == UPSTREAM.content_hash
    reasons = {
        (r.subject, r.source_index, r.revision): r.reason for r in export.revisions
    }
    assert reasons == {
        (0, 0, 0): "superseded",
        (0, 0, 1): "exported",
        (0, 2, 0): "superseded",
        (0, 2, 1): "withdrawn",
        (0, 4, 0): "not released",
        (1, 1, 0): "exported",
        (1, 1, 1): "not released",
        (1, 1, 2): "not released",
        (2, 3, 0): "exported",
        (2, 3, 1): "not released",
    }
    rows = {r.source_index: r.row for r in export.revisions if r.reason == "exported"}
    assert rows == {0: 0, 1: 1, 3: 2}


def test_export_never_widens_an_audience_or_a_payload_column():
    """Fails if an exported row is released to an audience the upstream row did not
    reach by the end, if a revision released to only one audience is exported to
    both, or if a payload column some exported audience could not read is copied.
    """
    policy, observer, public = _export(2), _export(1), _export(3)
    assert _ints(policy.table.audience) == [2, 2, 2]
    assert _ints(observer.table.audience) == [1, 1, 1]
    assert _ints(public.table.audience) == [3, 3, 3]
    # the observer-only correction of subject 2 reaches the observer's export only
    assert _ints(observer.table.source.revision) == [1, 0, 1]
    assert _ints(public.table.source.revision) == [1, 0, 0]
    expected = [[0.1, 0.0, -2.0, 0.0], [11.0, 0.0, -1.0, 0.0], [23.0, 0.0, -1.0, 0.0]]
    assert np.asarray(policy.table.payload).tolist() == expected
    assert np.asarray(public.table.payload)[:, 1:].tolist() == [
        row[1:] for row in expected
    ]


def test_export_places_the_upstream_end_at_tick_zero():
    """Fails if an offset is not the row's tick minus the upstream horizon, or if a
    revision released after the horizon counts as released by the end of the run.
    """
    table = _export(2).table
    assert _ints(table.start_offset) == [280 - 1000, 180 - 1000, 180 - 1000]
    assert _ints(table.end_offset) == [295 - 1000, 195 - 1000, 195 - 1000]
    later = _export(2, study=toy.survey_1(horizon_tick=1300)).table
    assert _ints(later.source.revision) == [1, 1, 0]
    assert _ints(later.start_offset) == [280 - 1300, 1180 - 1300, 180 - 1300]


def test_export_refuses_what_it_cannot_write():
    """Fails if an export that would misstate its rows goes through: a derived kind,
    an unknown epoch or audience, no upstream hash, namespace 0, an unmapped subject,
    two rows on one downstream channel with one local index, or a batched log.
    """
    derived = dataclasses.replace(
        SOURCE, caps=dataclasses.replace(SOURCE.caps, derived_kinds=(MEASUREMENT,))
    )
    run = SimpleNamespace(carry=SimpleNamespace(log=LOG))
    refused = [
        lambda: _export(2, study=derived),
        lambda: _export(4),
        lambda: _export(2, namespace=0),
        lambda: _export(2, subject_map={0: 1, 1: 2}),
        lambda: _export(2, log=_log(_row(0, 0, 0), _row(1, 0, 0))),
        lambda: export_records(
            run, MEASUREMENT, 2, "", SUBJECTS, PRODUCERS, study=SOURCE
        ),
        lambda: export_records(
            run, MEASUREMENT, 2, "h", SUBJECTS, PRODUCERS, "start", study=SOURCE
        ),
        lambda: _export(2, log=_batched(LOG)),
    ]
    for call in refused:
        with pytest.raises(ValueError):
            call()
    with pytest.raises(TypeError):
        export_records(object(), MEASUREMENT, 2, "h", SUBJECTS, PRODUCERS, study=SOURCE)


def _batched(log):
    return dataclasses.replace(log, kind=jnp.stack([log.kind, log.kind]))
