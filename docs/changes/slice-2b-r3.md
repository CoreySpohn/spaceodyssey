---
orphan: true
---

# Slice 2b R3: precursor export, the table codec and lineage

R3 builds D6 on the R1 table contract (`slice-2b-r1.md`): `export_records` turns a
finished upstream run into a `PreMissionTable` at the latest revision released to the
chosen audience; the table travels as a study field through the codec's existing
sha256 file references; `verify_lineage` checks a document's tables, their files and
the declared upstream hash; and the JSON pre-flight takes the upstream document, so
`D004` is reached at the pointer of the declared hash. B5 passes. There is no new
section, no `schema_version` change and no migration, and no pinned hash moved.

## Commits

1. `feat(codec): the JSON pre-flight takes an upstream document, so D004 is reached at the declared hash` (`cb67e03`)
2. `feat(studies): export a precursor table from an upstream run, at the latest revision released to the audience` (`7b433b9`)
3. `feat(studies): verify_lineage checks a document's tables, their files and declared upstream hash (D004)` (`b3f5a05`)
4. `fix(studies): the subject map selects exported subjects, so one subject of a shared channel can be exported` (`0c94a4b`)
5. this record

## What changed

- `codec/preflight.py`: `preflight_document(..., upstream=None)`. The upstream
  document (JSON text, a parsed object, a `StudyDocument` or a `RunDocument`) is read
  with the new `read_upstream` and passed as `PreflightContext.upstream_document`, so
  the importer's lineage check runs as it does in memory and its finding gets the
  pointer `/study/process/table/upstream_hash`. `UpstreamDocument(content_hash,
  problem)` and `read_upstream(upstream)` never raise on content: the hash is taken
  over the canonical bytes (whitespace and object key order do not change it; a
  changed value, reordered array elements or reordered `$dict` entries do), and a
  stored `content_hash` that no longer matches makes the document unreadable.
  `read_document(document)` returns a document's parsed object and its load findings.
- `studies/premission.py` (shared, R1-owned; the only edit, 4 lines): in
  `lineage_findings`, when the supplied upstream document carries a `problem` (it did
  not read), the `D004` message names that problem instead of "content hash None".
  Nothing else in the table, the importer or the overlap code changed.
- `studies/precursor.py` (new, R3-owned): `export_records`, `Export`,
  `ExportedRevision`, `table_digest`, `verify_lineage`, `LineageReport`,
  `TableLineage`, `FileReference`.
- No core file, no `codec/load.py`, `codec/dump.py`, `codec/files.py` or
  `codec/kinds.py` change. `studies/prior_observer.py` is untouched.

### Export as built

`export_records(run_result, kind, audience, upstream_document_hash, subject_map,
producer_map, epoch="end_at_zero", *, study, namespace=1)`:

- A datum is `(upstream channel, upstream subject, acquisition index)`. A revision
  qualifies when it was released to each chosen audience at a tick from 0 to the
  upstream horizon: the observer's release is the row's `accepted_tick`, the
  policy's its `released_tick`, which the core stamps only for an addressed row. The
  latest qualifying revision is exported once. If it is not a completed record (a
  failed or withdrawn correction), the datum is not exported.
- Every revision considered is listed in `Export.revisions` with its reason:
  `exported`, `superseded`, `withdrawn` or `not released`.
- Offsets are the row's ticks minus `study.horizon_tick`. The exported row's
  audience is the chosen audience, which every exported revision reached. A payload
  column that some chosen audience could not read upstream (`payload_columns`) is
  written as zero, so a scorer-only column never leaves the run.
- Source identity: `(namespace, upstream channel, downstream subject, upstream
  acquisition index)` with the exported revision; `local_index` is the upstream
  acquisition index (the R0a rule); `channel` is `producer_map[upstream channel]`.
  Rows are ordered by (channel, subject, acquisition index); there is no pedigree
  (width 1, `n_pedigree` 0); `upstream_hash` is the declared hash.
- Refused (`ValueError`): a derived kind, an epoch other than `end_at_zero`, an
  audience other than 1, 2 or 3, an empty upstream hash, namespace 0, two rows with
  one local index on one downstream channel, a batched log; `TypeError` for an object
  that holds no record log.

### Lineage as built

`verify_lineage(document, upstream_document=None, *, expected=None, files=None,
max_file_bytes=...)` returns a `LineageReport` and never raises on document content:

- The document is read with `read_document` (a stale stored `content_hash` is `L001`
  at `/content_hash`). Every `PreMissionTable` node in the study (the importer's and,
  in a prior study, the observer's copy) is decoded as the loader decodes it, with the
  file store, so every file reference is checked at its pointer. A failure is a
  finding with the loader's code (`L001`, or `D001` for an unknown kind version).
- Each decoded table's declared hash goes through `lineage_findings` with the read
  upstream document: a mismatch, or an upstream document that does not read, is a
  `D004` error at `<table>/upstream_hash`; with no upstream document, a `D004`
  warning, declared, unverified. The finding equals the JSON pre-flight's.
- `expected` (the table the upstream run exports, or its `table_digest`): a table
  with a declared hash whose digest differs is a `D004` error at the table's pointer.
  `table_digest` hashes every column's dtype, shape and bytes in row order and the
  declared hash, so it names one table whether stored inline or in files.
- `status` is `failed`, `no lineage`, `declared, unverified` or `verified`; each
  `TableLineage` names the table's pointer, declared hash, digest and every file
  reference (pointer, name, sha256, bytes).

### File references

Unchanged codec behavior, now pinned for the table and through both new paths: a file
name is `<sha256>.bin` and nothing else, so a relative path (`../x.bin`) and an
absolute path are refused at `.../file/name` before any file is opened; a symlink
(even to a file of the right bytes outside the directory) is refused by `O_NOFOLLOW`;
a missing file is refused at `.../file/name`; a digest mismatch, including the same
rows reordered in the file, at `.../file/sha256`. `load_study` raises `DocumentError`
at that pointer, and `preflight_document` and `verify_lineage` report it as one
finding.

## Tests

Tests added: 12 functions (23 nodes), all in `tests/codec/test_premission_codec.py`,
all unit tier; retired: none. Cold times, each node alone in a fresh process
(`SPACEODYSSEY_JAX_CACHE=off`, `JAX_ENABLE_COMPILATION_CACHE=false`, wall time
including the about 1.5 s import):

| Test | Tier | Cold s | Mutant it kills |
|---|---|---|---|
| `test_the_table_round_trips_through_file_references` | unit | 2.4 | store writes reversed bytes (C1); table integer columns kept inline (C2) |
| `test_every_corpus_document_keeps_its_pinned_hash_and_schema_version` | unit | 2.7 | every kind node gains an `imports` key (C3); `SCHEMA_VERSION` bumped (C4) |
| `test_the_json_preflight_reaches_d004_at_the_declared_hash` | unit | 2.4 | JSON pre-flight ignores `upstream` (C5); unreadable-upstream message dropped (C6) |
| `test_a_tampered_or_reordered_upstream_document_has_another_hash` (3) | unit | 2.2 each | hash over the raw text (C7); stored hash not checked (C8); arrays hashed as sets (C9) |
| `test_an_unsafe_or_damaged_file_reference_is_refused_at_its_pointer` (6) | unit | 2.3 each | names may hold a path (C10); symlinks followed (C11); digest not checked (C12); missing file read as zeros (C13); lineage reads tables without decoding (P11); a table that does not decode raises (P15) |
| `test_export_keeps_the_latest_revision_released_to_the_audience` | unit | 2.6 | earliest released revision (P1); release after the horizon counts (P3); withdrawn datum exported (P7); subject not mapped (P8); subject map does not select (P17) |
| `test_export_never_widens_an_audience_or_a_payload_column` | unit | 2.6 | P1, P3, P7; every payload column copied (P5); audience copied from the log (P6) |
| `test_export_places_the_upstream_end_at_tick_zero` | unit | 2.9 | offsets from tick 0 (P4); P3 |
| `test_export_refuses_what_it_cannot_write` | unit | 2.8 | local index clash not refused (P9) |
| `test_an_exported_table_imports_and_verifies_its_lineage` | unit | 2.8 | lineage findings dropped (P10); status ignores a missing upstream (P16); P3, P11 |
| `test_lineage_fails_on_a_reordered_or_changed_table` | unit | 2.8 | digest ignores row order (P12); expected export not compared (P13) |
| `test_the_lineage_check_never_raises_on_a_broken_document` (5) | unit | 2.4 to 2.6 | stored document hash not checked (P14); P11, P15 |

Each mutant was applied to a copy of `src/` and the file run against it (`$TMPDIR`
script, not committed); all 29 are killed. A mutant that removed the audience test in
the release check survived because the core stamps `accepted_tick` and
`released_tick` only for an addressed row; that test was redundant and is removed, not
covered.

## Checks run

- `tests/core/test_golden_histories.py` with `tests/codec/test_run_document_runs.py`
  (golden pin and corpus document hash pin) at `0c94a4b`, `-n 2 --dist loadfile`:
  189 passed in 327.3 s. `git diff be2e439 -- tests/core/golden_pin.json
  tests/codec/hash_pin.json` is empty.
- Quick tier (`-m "not slow" -n 2 --dist loadfile`, entry-point test deselected as in
  R0a) at `0c94a4b`: 2,097 passed in 257.2 s. The budget plugin reported one unit over
  5 s under the parallel load, the existing
  `tests/adapters/test_jaxedith_summary.py::test_detector_spec_and_report_refuse_undeclared_choices`
  (6.5 s), not touched here.
- The codec test files with the touched pre-flight and lint files
  (`test_premission_codec.py`, `test_preflight_json.py`, `test_run_document.py`,
  `test_document.py`, `tests/core/test_preflight.py`, `test_premission.py`,
  `test_lints.py`): 595 passed before the precursor commits; all of them are in the
  quick tier above at the final tree.
- `ruff check`, `ruff format --check`, `tools/check_internal_refs.py` and
  `pre-commit run` on every touched file: clean. Every touched Python file is ASCII.
- The full suite and the Sphinx docs build were not run.

## Pins

No pinned golden digest, fingerprint or corpus document hash moved
(`tests/core/golden_pin.json` and `tests/codec/hash_pin.json` untouched), and
`schema_version` is unchanged. The new quick test rebuilds all 58 corpus run documents
without running them (0.5 s) and compares them with the pin, so K4 is also checked in
the quick tier. No core file changed.

## Spec silences decided here

1. **`export_records` takes the upstream study** (keyword `study`). A `RunResult`
   holds no horizon and no payload column audiences; both are needed (offsets, and
   never copying a column an audience could not read).
2. **It returns an `Export`** (`table`, `revisions`), not a bare table: "earlier
   revisions are kept only in the lineage report" needs a place to live at export
   time.
3. **The chosen audience is the exported audience.** A revision qualifies only when it
   reached every chosen audience by the end, so the row is never released downstream
   to an audience it did not reach upstream; a public upstream row exported for the
   policy becomes a policy row. Rejected: copying the log's audience (it would widen a
   row released only to the observer by the end into a public one).
4. **A datum whose latest released revision is not completed is not exported.** The
   importer writes every row as completed, so exporting the earlier revision of a
   withdrawn correction would resurrect it.
5. **Payload columns are masked by the strictest chosen audience**: a column is kept
   only when every chosen audience could read it upstream.
6. **`kind` selects the upstream rows and is the table's kind index**; a derived kind
   is refused (an exported prior would have no pedigree).
7. **Both maps select.** Rows on channels outside `producer_map` or of subjects outside
   `subject_map` are not exported. The subject map selected nothing at first (an
   unmapped subject raised); it selects since commit 4, so one subject of a shared
   channel can be exported (see the open point below).
8. **`verify_lineage` decodes each table with the loader** instead of reading the JSON,
   so a file reference is checked exactly as `load_study` checks it, at the same
   pointer, and a table node that does not decode is a finding with the loader's code.
9. **"Fails on a tampered or reordered precursor"** is met in three places: a changed,
   reordered or stale-hashed upstream document has another hash (`D004`); a tampered
   or reordered table file is refused at its pointer (`L001`); and a table re-dumped
   with its rows reordered or one value changed fails against the expected export
   (`expected=`, `D004` at the table). Without `expected`, a re-dumped reordered
   table passes: the declared upstream hash names the upstream document, not the
   table, and row order carries no meaning for the import (K3).
10. **`D004` for an unreadable upstream document is an error** naming the reason; the
    JSON pre-flight reads the upstream through `read_upstream`, so it never raises.
11. **One `D004` rule for both paths.** D10 has the JSON `D004` use `verify_lineage`;
    both the JSON pre-flight and `verify_lineage` call the importer's
    `lineage_findings` with the upstream read by `read_upstream`, so the two give the
    same finding (a test compares the messages). The pre-flight does not call
    `verify_lineage` itself: its file and table checks are already the loader's, and
    the codec does not import `studies/`.
12. **The upstream document's own hash is the canonical one.** A run document's hash
    covers its run record and registry, so the declared hash names the whole run
    document when the upstream is one.

## Spec points not satisfied, and open points

- **The local index rule and D006 conflict for a shared channel.** The core counts
  acquisition indices per subject (the first measurement of every subject is index
  0), while R0a fixes `local_index` to the upstream acquisition index and D006
  requires it unique per channel. Exporting a toy survey's measurements of several
  subjects to one channel is therefore refused (`ValueError`); one subject per
  downstream channel works (checked on a real toy run: subject 0's three rows export
  with offsets -990, -730 and -470). Fixing this is a contract change, not an R3
  choice: either D006 becomes unique per (channel, subject), which also needs the
  prior observer's raw-row lookup (by channel and local index, in
  `studies/prior_observer.py`) to add the subject, or the exporter writes a frozen
  injective index such as `acquisition_index * n_subjects + subject`. R5 (survey 4)
  needs one of the two, or a precursor of one subject per channel.
- `load_run` takes no upstream document; D004 through JSON is reached with
  `preflight_document(..., upstream=)`, which `load_run` does not call with one.
- Survey 4's paired-difference, correction and tamper checks on a real run are R5's;
  R3's export tests use a synthetic record log of one world (the export reads only the
  log, so a `RunResult`, `Carry` or `RecordLog` are read alike).
- The review rounds (tier 2) are not run here.
