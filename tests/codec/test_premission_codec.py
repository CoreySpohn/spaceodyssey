"""B5: pre-mission tables through the codec, with file references and lineage.

A table is a study field: the codec's generic walk writes its large columns to
sha256-named files, and the document hash covers their references. Its declared
upstream hash is checked against a supplied upstream document (`D004`), in memory and
through the JSON pre-flight, at the pointer of the declared hash. No existing corpus
document hash and no schema version moves.
"""

import dataclasses
import hashlib
import json
import os
import shutil
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.codec import (
    SCHEMA_VERSION,
    DocumentError,
    KindResolver,
    StudyDocument,
    dump_study,
    load_study,
    preflight_document,
)
from spaceodyssey.codec.files import FileStore, UnsafeDirectory
from spaceodyssey.codec.load import same_value
from spaceodyssey.codec.preflight import (
    UpstreamDocument,
    read_upstream,
    resolve_pointer,
)
from spaceodyssey.core import HostExecutor
from spaceodyssey.core.enums import RecordStatus
from spaceodyssey.core.preflight import preflight
from spaceodyssey.core.preflight.checks import PreflightContext
from spaceodyssey.core.records import empty_log
from spaceodyssey.studies import toy
from spaceodyssey.studies.conformance import CORPUS
from spaceodyssey.studies.precursor import export_records, table_digest, verify_lineage
from spaceodyssey.studies.premission import (
    PREMISSION_KINDS,
    PreMissionImporter,
    authored_table,
    table_findings,
)
from tests.codec.test_document import TOY_KINDS
from tests.codec.test_run_document_runs import EXECUTORS, _corpus_document, _hash_pin
from tests.core.helpers import one_world

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
    raises from the pre-flight or the lineage check instead of an `L001` finding.
    """
    files = tmp_path / "files"
    data = json.loads(_stored(files))
    data.pop("content_hash")
    where = _damage(how, data, files, tmp_path)
    text = json.dumps(data)
    found = preflight_document(text, RESOLVER, files=files, upstream=UPSTREAM)
    assert _codes(found) == [("L001", "error", where)]
    report = verify_lineage(text, UPSTREAM, files=files)
    assert report.status == "failed"
    assert _codes(report.findings) == [("L001", "error", where)]
    with pytest.raises(DocumentError) as info:
        load_study(text, RESOLVER, files=files)
    assert info.value.pointer == where


# when and what is replaced by a link to an outside copy of the same bytes -> refused
SWAPS = {
    ("before", "parent"): False,  # the trusted directory as named, its alias resolved
    ("before", "directory"): True,
    ("during", "parent"): True,
    ("during", "directory"): True,
}


def _tree(root):
    """A file-backed document under root/doc/files and an outside copy of root/doc."""
    base = root / "doc"
    files = base / "files"
    text = _stored(files)
    copy = root / "outside"
    shutil.copytree(base, copy)
    return text, base, files, copy


def _swap(swapped, base, files, copy):
    if swapped == "parent":
        shutil.rmtree(base)
        os.symlink(copy, base)
    else:
        shutil.rmtree(files)
        os.symlink(copy / "files", files)


@pytest.mark.parametrize(("when", "swapped"), list(SWAPS))
def test_a_link_below_the_trusted_directory_is_refused(
    tmp_path, monkeypatch, when, swapped
):
    """The trusted directory is the files directory's parent, resolved once when the
    store is made. Fails if a files directory replaced by a link before the call, or
    the files directory or its parent replaced by a link while the document is read
    (after the store resolved its path), is read through by `load_study`, the
    pre-flight or the lineage check, or reported as anything but `L001` at a
    reference's name; or if a parent that is a link before the call (an alias of the
    trusted directory, as macOS's `/tmp` is) is not read.
    """
    original = FileStore.get
    armed = []

    def get(self, ref, path, expected=None):
        if armed:
            armed.pop()()
        return original(self, ref, path, expected)

    monkeypatch.setattr(FileStore, "get", get)
    refused = SWAPS[when, swapped]
    for name in ("load", "preflight", "lineage"):
        text, base, files, copy = _tree(tmp_path / name)

        def swap(base=base, files=files, copy=copy):
            _swap(swapped, base, files, copy)

        if when == "before":
            swap()
        else:
            armed.append(swap)
        if name == "load":
            if not refused:
                load_study(text, RESOLVER, files=files)
                continue
            with pytest.raises(DocumentError) as info:
                load_study(text, RESOLVER, files=files)
            assert info.value.pointer.endswith("/file/name")
        elif name == "preflight":
            found = preflight_document(text, RESOLVER, files=files, upstream=UPSTREAM)
            if not refused:
                assert found == []
                continue
            assert [str(f.code) for f in found] == ["L001"]
            assert found[0].pointer.endswith("/file/name")
        else:
            report = verify_lineage(text, UPSTREAM, files=files)
            if not refused:
                assert report.status == "verified"
                continue
            assert report.status == "failed"
            assert [str(f.code) for f in report.findings] == ["L001"]
            assert report.findings[0].pointer.endswith("/file/name")
        assert not armed


def test_a_link_below_the_trusted_directory_is_refused_whoever_owns_it(
    tmp_path, monkeypatch
):
    """Fails if a link below the trusted directory is followed because it is (or is
    reported as) owned by root: a user can replace a root-owned link in a directory
    they can write, so ownership is no reason to follow one.
    """
    text, base, files, copy = _tree(tmp_path)
    _swap("directory", base, files, copy)
    real = os.stat

    def as_root(path, *, dir_fd=None, follow_symlinks=True):
        info = real(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if follow_symlinks:
            return info
        return os.stat_result((info.st_mode, *info[1:4], 0, *info[5:]))

    monkeypatch.setattr(os, "stat", as_root)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {as_root})
    with pytest.raises(DocumentError) as info:
        load_study(text, RESOLVER, files=files)
    assert info.value.pointer.endswith("/file/name")
    assert "link" in info.value.detail


def test_the_writer_creates_nothing_outside_when_an_ancestor_becomes_a_link(tmp_path):
    """Fails if, after a store is made for a files directory whose parent does not
    exist yet and an existing ancestor is then replaced by a link to an outside
    directory, `put` creates anything outside, or does not refuse the link. Control:
    without the link, `put` makes the missing directories and the file.
    """
    ancestor = tmp_path / "ancestor"
    ancestor.mkdir()
    store = FileStore(ancestor / "newdoc" / "files", 0)
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.rmtree(ancestor)
    os.symlink(outside, ancestor)
    with pytest.raises(UnsafeDirectory):
        store.put(b"payload")
    assert list(outside.iterdir()) == []
    os.unlink(ancestor)
    ancestor.mkdir()
    ref = store.put(b"payload")
    assert (ancestor / "newdoc" / "files" / ref["name"]).read_bytes() == b"payload"


def test_the_writer_refuses_a_files_directory_reached_through_a_link(tmp_path):
    """Fails if `dump_study` writes array files through a files directory that is a
    link to another tree, or refuses without saying so. Control: a real directory
    takes the files.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    base = tmp_path / "doc"
    base.mkdir()
    os.symlink(outside, base / "files")
    study = _study(_table())
    with pytest.raises(DocumentError) as info:
        dump_study(study, RESOLVER, files=base / "files", max_inline=2)
    assert "link" in str(info.value)
    assert list(outside.iterdir()) == []
    dump_study(study, RESOLVER, files=base / "real" / "files", max_inline=2)
    assert any((base / "real" / "files").iterdir())


# ---- export from an upstream run ----------------------------------------------------

MEASUREMENT = toy.RECORD_MEASUREMENT
SOURCE = toy.survey_1()  # the upstream study: horizon 1000
SUBJECTS = {0: 1, 1: 2, 2: 3}
ONE = {0: 0, 1: 0}  # two upstream subjects mapped to one downstream subject
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
    cols = {
        f.name: np.array(getattr(base, f.name))
        for f in dataclasses.fields(base)
        if getattr(base, f.name) is not None  # a column the capacities do not hold
    }
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
    unmapped row (channel or subject) is exported, or if the dropped revisions are
    not listed with why.
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
    # the maps select: a subject outside the subject map is not exported
    fewer = _export(2, subject_map={0: 1, 2: 3})
    assert _ints(fewer.table.source.source_index) == [0, 3]
    assert {r.subject for r in fewer.revisions} == {0, 2}


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
    an unknown epoch or audience, no upstream hash, namespace 0, two rows of one
    downstream subject and channel with one local index, or a batched log.
    """
    derived = dataclasses.replace(
        SOURCE, caps=dataclasses.replace(SOURCE.caps, derived_kinds=(MEASUREMENT,))
    )
    run = SimpleNamespace(carry=SimpleNamespace(log=LOG))
    refused = [
        lambda: _export(2, study=derived),
        lambda: _export(4),
        lambda: _export(2, namespace=0),
        lambda: _export(2, log=_log(_row(0, 0, 0), _row(1, 0, 0)), subject_map=ONE),
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


def test_two_subjects_of_one_channel_export_with_one_local_index():
    """Fails if two subjects measured on one channel, each at acquisition index 0 as
    the core counts it, cannot be exported to one downstream channel, or lose their
    own identities (local index unique per channel and subject, D006).
    """
    log = _log(_row(0, 0, 0), _row(1, 0, 0), _row(0, 1, 0, 300))
    table = _export(2, log=log, subject_map={0: 0, 1: 1}).table
    assert _ints(table.channel) == [60, 60, 60]
    assert _ints(table.source.subject) == [0, 0, 1]
    assert _ints(table.local_index) == [0, 1, 0]
    assert table_findings(table, _study(table).caps) == []


@pytest.mark.medium  # one compiled host run of the toy survey
def test_a_toy_survey_channel_of_several_subjects_exports_to_one_channel():
    """Fails if the measurements a real toy survey run writes on its one channel, two
    subjects each counted from acquisition index 0, cannot be exported to one
    downstream channel and imported without a pre-flight finding.
    """
    world = one_world()
    run = HostExecutor().run(SOURCE, *world, 400)
    log = run.carry.log
    valid = np.asarray(log.valid)
    assert set(np.asarray(log.subject)[valid].tolist()) == {0, 1}
    table = export_records(
        run,
        MEASUREMENT,
        3,
        UPSTREAM.content_hash,
        {0: 0, 1: 1},
        {10: 60},
        study=SOURCE,
    ).table
    assert table.n_rows == int(valid.sum())
    assert sorted(_ints(table.local_index)) == [0, 0, 1, 1, 2]
    context = PreflightContext(upstream_document=UPSTREAM)
    assert preflight(_study(table), context=context) == []


def _batched(log):
    return dataclasses.replace(log, kind=jnp.stack([log.kind, log.kind]))


# ---- lineage ------------------------------------------------------------------------


def _exported(tmp_path, table=None):
    """The exported policy table imported by survey 1, as file-backed document text."""
    return _stored(tmp_path, _export(2).table if table is None else table)


def test_an_exported_table_imports_and_verifies_its_lineage(tmp_path):
    """Fails if the exported table does not pass pre-flight with its upstream
    document, if the lineage report does not name the declared hash and each file's
    sha256 (as the file's bytes hash), or if `verify_lineage` does not pass with the
    upstream document, fail with `D004` at the declared hash (as the JSON pre-flight
    does) with another one, and report declared, unverified with none.
    """
    table = _export(2).table
    context = PreflightContext(upstream_document=UPSTREAM)
    assert preflight(_study(table), context=context) == []
    text = _exported(tmp_path, table)
    report = verify_lineage(text, UPSTREAM.to_json(), files=tmp_path)
    assert report.status == "verified" and report.findings == ()
    (lineage,) = report.tables
    assert lineage.pointer == "/study/process/table"
    assert lineage.upstream_hash == UPSTREAM.content_hash
    assert lineage.digest == table_digest(table)
    named = {ref.pointer: ref for ref in lineage.files}
    assert f"{PAYLOAD}" in named
    for ref in named.values():
        raw = (tmp_path / ref.name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == ref.sha256 and len(raw) == ref.bytes
    wrong = verify_lineage(text, OTHER, files=tmp_path)
    assert wrong.status == "failed"
    assert _codes(wrong.findings) == [("D004", "error", DECLARED)]
    json_found = preflight_document(text, RESOLVER, files=tmp_path, upstream=OTHER)
    assert wrong.findings[0].message == json_found[0].message
    unverified = verify_lineage(text, files=tmp_path)
    assert unverified.status == "declared, unverified"
    assert _codes(unverified.findings) == [("D004", "warning", DECLARED)]


def test_lineage_fails_on_a_reordered_or_changed_table(tmp_path):
    """Fails if a table whose rows were reordered, or one value changed, after export
    passes the lineage check against the expected export, or if the table as exported
    does not.
    """
    table = _export(2).table
    reordered = _take(table, [2, 0, 1])
    changed = dataclasses.replace(table, payload=table.payload.at[0, 0].add(1.0))
    assert table_digest(reordered) != table_digest(table)
    right = verify_lineage(
        _exported(tmp_path, table), UPSTREAM, expected=table, files=tmp_path
    )
    assert right.status == "verified"
    for other in (reordered, changed):
        text = _exported(tmp_path, other)
        report = verify_lineage(text, UPSTREAM, expected=table, files=tmp_path)
        assert _codes(report.findings) == [("D004", "error", "/study/process/table")]
        by_digest = verify_lineage(
            text, UPSTREAM, expected=table_digest(table), files=tmp_path
        )
        assert by_digest.findings == report.findings


def _take(table, order):
    """The table with its rows in `order` (columns of `[N]` and `[N, ...]` alike)."""
    order = jnp.asarray(order)
    return jax.tree_util.tree_map(lambda x: x[order], table)


BROKEN = {
    "malformed": ("L001", ""),
    "array": ("L001", ""),
    "missing_field": ("L001", "/study/process/table"),
    "version": ("D001", "/study/process/table/$kind"),
    "stale": ("L001", "/content_hash"),
    "hash_array": ("L001", DECLARED),
    "digits": ("L001", ""),
    "deep": ("L001", ""),
    "format": ("L001", "/format"),
    "schema": ("D002", "/schema_version"),
}


@pytest.mark.parametrize("how", list(BROKEN))
def test_the_lineage_check_never_raises_on_a_broken_document(tmp_path, how):
    """Fails if a document that does not parse (also a 4,500-digit number or 1,100
    nested arrays), a table missing a field, at an unknown kind version or with an
    array for its declared hash, a stored hash that no longer matches, or a wrong
    format or a newer schema version (no stored hash) raises from the lineage check,
    the pre-flight or `read_upstream`, or does not fail the report with a finding at
    its pointer before any table is interpreted; or if text that does not parse
    raises anything but `DocumentError` from `load_study`.
    """
    data = json.loads(_exported(tmp_path))
    if how != "stale":
        data.pop("content_hash")
    node = data["study"]["process"]["table"]
    expected = BROKEN[how]
    text = None
    if how == "malformed":
        text = "{not json"
    elif how == "array":
        text = "[]"
    elif how == "missing_field":
        node.pop("local_index")
    elif how == "version":
        node["$kind"][2] = 9
    elif how == "stale":
        node["upstream_hash"] = OTHER.content_hash
    elif how == "hash_array":
        node["upstream_hash"] = {"$array": "int64", "shape": [2], "data": [1, 2]}
    elif how == "digits":
        data["schema_version"] = 7
        text = json.dumps(data).replace(
            '"schema_version": 7', '"schema_version": ' + "9" * 4500
        )
    elif how == "deep":
        text = json.dumps({**data, "study": None})
        text = text.replace("null", "[" * 1100 + "]" * 1100, 1)
    elif how == "format":
        data["format"] = "bogus"
    else:
        data["schema_version"] = 999
    text = json.dumps(data) if text is None else text
    report = verify_lineage(text, UPSTREAM, files=tmp_path)
    assert report.status == "failed"
    assert expected in {(str(f.code), f.pointer) for f in report.findings}
    if how in ("format", "schema"):
        assert report.tables == ()
    found = preflight_document(text, RESOLVER, files=tmp_path, upstream=UPSTREAM)
    assert expected in {(str(f.code), f.pointer) for f in found}
    read = read_upstream(text)
    if how in ("malformed", "array", "digits", "deep", "format", "schema"):
        assert read.content_hash is None and read.problem
    if how in ("malformed", "digits", "deep"):
        with pytest.raises(DocumentError):
            load_study(text, RESOLVER, files=tmp_path)


@pytest.mark.parametrize("where", ["schema", "study", "kind"])
def test_a_parsed_document_holding_a_huge_integer_is_a_finding(tmp_path, where):
    """Fails if a parsed document whose schema version, a value in its study (with a
    stored hash) or a table's kind version is an integer too long to write raises
    from the lineage check, the pre-flight or `read_upstream`, instead of an `L001`
    finding at its pointer; or if the same value in a document object raises anything
    but a finding or a `DocumentError` at its pointer from hashing, the lineage check,
    the pre-flight or `load_study`.
    """
    data = json.loads(_exported(tmp_path))
    if where == "schema":
        data = {"format": data["format"], "schema_version": 10**4499, "study": {}}
        where_ = "/schema_version"
    elif where == "study":
        data["study"]["horizon_tick"]["data"] = 10**4499
        where_ = "/study/horizon_tick/data"
    else:
        data["study"]["process"]["table"]["$kind"][2] = 10**4499
        where_ = "/study/process/table/$kind/2"
    report = verify_lineage(data, UPSTREAM, files=tmp_path)
    assert report.status == "failed"
    assert ("L001", where_) in {(str(f.code), f.pointer) for f in report.findings}
    found = preflight_document(data, RESOLVER, files=tmp_path, upstream=UPSTREAM)
    assert ("L001", where_) in {(str(f.code), f.pointer) for f in found}
    read = read_upstream(data)
    assert read.content_hash is None and where_ in read.problem
    # the same value in a document object: a finding or a DocumentError, never a
    # ValueError, on every entry
    if where == "schema":
        doc = StudyDocument(study={}, schema_version=10**4499)
        expected = ("L001", where_)
    else:
        doc = StudyDocument(study=data["study"])
        expected = ("L001", where_)
        with pytest.raises(DocumentError) as info:
            doc.canonical_bytes()
        assert info.value.pointer == where_
    report = verify_lineage(doc, UPSTREAM, files=tmp_path)
    assert report.status == "failed"
    assert expected in {(str(f.code), f.pointer) for f in report.findings}
    found = preflight_document(doc, RESOLVER, files=tmp_path, upstream=UPSTREAM)
    assert expected in {(str(f.code), f.pointer) for f in found}
    with pytest.raises(DocumentError) as info:
        load_study(doc, RESOLVER, files=tmp_path)
    assert info.value.pointer == where_


@pytest.mark.parametrize("how", ["file", "zero"])
def test_an_oversize_shape_is_refused_at_its_pointer(tmp_path, how):
    """Fails if an array whose shape no array can hold ([10**3000, 10**3000] on a
    file reference, or [0, 10**3000] with no data), in a document with a correct
    hash, raises anything but `L001` at the shape's pointer from the lineage check
    or the pre-flight, or anything but a `DocumentError` there from `load_study`.
    """
    data = json.loads(_exported(tmp_path))
    node = data["study"]["process"]["table"]["payload"]
    if how == "file":
        node["shape"] = [10**3000, 10**3000]
    else:
        node.pop("file")
        node["shape"] = [0, 10**3000]
        node["data"] = []
    text = StudyDocument(study=data["study"]).to_json()
    where = "/study/process/table/payload/shape"
    report = verify_lineage(text, UPSTREAM, files=tmp_path)
    assert report.status == "failed"
    assert [(str(f.code), f.pointer) for f in report.findings] == [("L001", where)]
    found = preflight_document(text, RESOLVER, files=tmp_path, upstream=UPSTREAM)
    assert [(str(f.code), f.pointer) for f in found] == [("L001", where)]
    with pytest.raises(DocumentError) as info:
        load_study(text, RESOLVER, files=tmp_path)
    assert info.value.pointer == where
