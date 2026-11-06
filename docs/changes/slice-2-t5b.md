---
orphan: true
---

# Slice 2 T5b: the run document and the remaining document-form items

T5a made a `Study` a document. T5b adds what a study does not hold: the `Query`
service, the refit boundary and the record-kind registry. A `RunDocument` binds them
to a study, so a run reloaded from text reproduces the original run on both
executors (gate B3). It also adds file references for large arrays, a named refusal
for callables and the `{"no_switch": true}` phase boundary. No file under `core/`
changed and no run behavior changed.

## The run document

`dump_run(study, registry, *, executor, bindings, service=None, refit=None,
kinds=None, files=None, max_inline=1024)` returns a `RunDocument`, a subclass of
`StudyDocument`; `load_run(document_or_text, bindings, kinds=None, *, files=None)`
returns a `Run`. A run document is a JSON object with these keys:

| Key | Content |
|---|---|
| `format` | `"spaceodyssey.run_document"` |
| `schema_version` | 1 |
| `study` | the encoded study, exactly as in a study document |
| `run` | the run record: `executor`, `service`, `refit` |
| `records` | the record-kind registry, in registry order |
| `content_hash` | sha256 of the canonical bytes of the five keys above |

The refresh schedule is a field of the study (`Study.refresh`), so it is already in
`study` and in the hash. The run record is the one the JSON pre-flight reads:
`service` is `{kind, version, configuration}` and `refit` is `{host, compiled}`, each
side `{kind, version, configuration, model_hash}`. A refit boundary may be one object
(bound on both executors) or a dict with a boundary for each executor; the two must
carry the same inference-model hash, which dump refuses otherwise (`D003`).

`records` is a list of `{name, derived, columns: [[name, audience]]}`. `Study` keeps
only the lowered tables and `KindRegistry.lower` drops column names, so the registry
is passed to `dump_run` and rebuilt by `load_run`. Both ends check that
`registry.lower(study.caps, study.observer)` equals the study's capacities, so a
registry that does not describe the study is refused at `/records`.

Code is never stored. A service or boundary is named by a kind in a `Bindings` table,
passed explicitly as the study kinds are. A `BindingKind` gives its name and
version, `describe(obj, registry)` (the configuration, plain JSON), `build(config,
registry)`, a `matches` predicate (default `type(obj) is cls`) and an optional
`migrate(from_version, configuration)`. A service names its record kind by name in
its configuration, which is why `build` receives the registry. A boundary's
`model_hash` is read from the instance on dump and compared with the stored one after
`build`, so a hand-edited hash is refused at its pointer. An unknown kind or a version
newer than the table is `D001`; an older version without `migrate` is `D001`; a
configuration `build` rejects fails at `/run/.../configuration`.

`Run` holds `study`, `registry`, `executor`, the service record, the rebuilt `refit`
boundaries by executor and the table. `Run.execute(truths, roots, n_steps,
executor=None, **kw)` runs the study on the named executor (default the document's)
with a fresh service from `make_service()` and that executor's boundary.

## How pre-flight consumes it

`preflight_document` accepts a `RunDocument`, its text or a parsed object whose
`format` is `spaceodyssey.run_document`. The run record is read from `run`, where T6 already
looked for it, with the same pointers (`/run/...`), so every existing check and every
existing study-document test is unchanged. For a run document:

- the envelope allows `run` and `records`, and both are part of the hash it checks;
- an unknown member of `run` is `D003` at its pointer, and a missing `executor` too;
  a member named `wake_fn`, `decide_py`, `activation`, `kernel` or `fit` is reported
  as a callable;
- `records` must decode and lower to the study's capacities, else `L001` at
  `/records` (or the entry that fails);
- with `bindings=`, a service or boundary naming an unknown kind or a newer version
  is `D001` at `/run/.../kind` or `/version`.

`load_run` runs the same record and service checks, so the first problem pre-flight
would list is the error `load_run` raises.

## File references

`dump_study` and `dump_run` take `files=<directory>` and `max_inline` (default 1024
elements). An array with more elements is written as its raw bytes (C order, native
byte order) to `<sha256>.bin` in that directory, and the node keeps the dtype and
shape: `{"$array": dtype, "shape": [...], "file": {"name", "sha256", "bytes"}}`, with
`weak` and `type` as before. The reference is inside the hashed study, so the
document hash covers the file content. Names are content hashes, so a repeated dump is
idempotent and two arrays with equal bytes share a file. The loader takes
`files=<directory>` and refuses, each with a pointer to the reference member: a
document with references and no directory, a name that is not `<sha256>.bin` or that
does not match its `sha256` (so no path can be named), a missing file (`name`), a
file of another size (`bytes`), bytes that do not hash to the stored value
(`sha256`), a shape that does not fit the byte count, and a node with both `data` and
`file`. NaN, infinities and `-0.0` are preserved bit for bit.

## Callable refusal

`CallableRefused` (a `DocumentError`) is raised with the callable's name and the
pointer of the value, and says to register a kind or a binding instead. It covers:

- a function, lambda or activation function in a module field, at the field
  (`/study/design/act`), on dump;
- `wake_fn` passed to `dump_run`, at `/run/wake_fn`;
- a boundary holding a function that no binding kind covers (`HostLoopRefit.fit`,
  `MaskedBatchRefit.kernel`), at `/run/refit/<side>`;
- a callable inside a configuration, at its path;
- on load, a document value tagged `$callable`, `$function`, `$lambda` or `$fn`, and a
  run member `wake_fn`, `decide_py`, `activation`, `fit` or `kernel`.

The spec lists the three new codes D001 to D003; the refusal is a distinct exception
and the in-band pre-flight form is the `D003` unknown-member finding above, so no
fourth code is added.

## No switch

`phase_until` is `I64` and compares integer ticks. A boundary entry may be written
`{"no_switch": true}` in the array data; the loader lowers it to `caps.max_tick` and
the dumper writes the tag back for every entry equal to `caps.max_tick` (so a literal
integer at that value dumps as the tag). Any other dict, or `false`, is refused at the
entry. The loader refuses a study whose `horizon_tick` is at or above `caps.max_tick`
at `/study/horizon_tick`, and the dumper refuses the same study, so no document is
written that cannot be loaded.

## Public API

Added to `spaceodyssey.codec`: `dump_run`, `load_run`, `RunDocument`, `Run`,
`Bindings`, `BindingKind`, `CallableRefused`, `RUN_FORMAT`. `dump_study` and
`load_study` gain `files` (and `max_inline` on dump). `preflight_document` gains
`bindings`. `codec.preflight` exposes `record_findings` and `service_findings`
(renamed from private helpers).

## Tests

`tests/codec/test_run_document.py` (unit tier, 82 cases; no executor run):

- round trip of the oracle refresh run (service, masked boundary, registry,
  executor), including a second dump equal to the first;
- the record in pre-flight form; a boundary per executor; models that differ are
  refused on dump; a bare run;
- pre-flight on a run document: clean, `D003` hashes and missing service, `D001` and
  `L001` for a bad binding kind and registry, the hash, a callable member;
- registry failures (one fixture each): dropped kind, dropped column, changed
  audience, entry shape, column shape, zero audience, repeated name, not a list, and
  a registry that does not lower on dump;
- binding failures (15 fixtures): unknown and newer kinds, version zero, an edited
  model hash, differing hashes, executor missing or unknown, a refit side missing, a
  bad configuration, an unknown member, a newer schema; migrate success, migrate
  failure, an older version without migrate, an instance no kind covers, a
  non-JSON configuration, duplicate registration;
- callables: activation function, lambda, `wake_fn`, unregistered boundary
  functions, a function in a configuration, five run members and four tags;
- file references: reference and sha256, hash covering the content, all 28 corpus
  studies through files with `max_inline=0`, and failure fixtures for a missing file,
  a changed byte, a truncated file, no directory, five malformed references, a wrong
  shape and data plus file; a run document through files;
- no switch: written back and lowered, a mixed list, a literal capacity tick, four
  malformed tags, the horizon refused on load and dump, and inside a run document.

`tests/codec/test_run_document_runs.py` (slow tier, listed in `SLOW_FILES`: it runs
the chunked driver and the masked kernel on both executors, as the refresh runs do):
B3. A reloaded run has an equal history, world by world, to the original run on the
host executor and on the compiled executor; a run with a host-loop boundary on the
host and a masked boundary on the compiled executor agrees across executors and with
the original; a run through array files is equal; and a corrupted boundary gives a
different history, so the equality is not vacuous.

## Timing

Local runs on the four-thread compute launcher.

| Run | Result |
|---|---|
| `test_run_document.py` | 82 passed, 1.8 s wall; no case near 5 s |
| `test_run_document_runs.py` | 5 passed, 24.2 s wall; compiled case 13.9 s, host case 7.7 s (cold compile) |
| Full suite (entry-point install test deselected in the worktree), `-n 2 --dist loadfile` | 2,915 passed, 1,329 s wall, peak footprint 10.5 GiB; the two unit-budget reports are the two known tests outside the codec |

## Remaining

- `KindResolver` and entry points discover study kinds; the `Bindings` table is
  explicit. A plugin family for services and boundaries, with an entry-point check,
  would follow the same pattern as `spaceodyssey.plugins`.
- A kind-version `migrate` for `Bindings` exists; no real binding has a second
  version yet, so only synthetic migrations are tested.
