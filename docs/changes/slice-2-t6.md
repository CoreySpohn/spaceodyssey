---
orphan: true
---

# Slice 2 T6: JSON pre-flight

`spaceodyssey.codec.preflight_document` runs every pre-flight check on a JSON study
document and returns all findings, each with the JSON pointer of the offending value.
It reports the ten in-memory codes with the same code the core gives, and the three
document codes `D001`, `D002` and `D003`. No file under `core/` changed and no run
behavior changed.

## Commits

On `slice2-t6` (first version):

1. `feat(codec): report all thirteen pre-flight codes on a JSON document`
2. `docs(changes): record Slice 2 T6`

On `slice2-t6-r`, the same two commits cherry-picked onto the main that followed, then
the review fixes (see "Review fixes").

## Interface

```python
preflight_document(document, kinds=None, *, run=None, executor=None)
    -> list[DocumentFinding]
```

- `document` is JSON text, a parsed object or a `StudyDocument`. `kinds` is a
  `KindResolver` or a mapping. `run` is the run record described below; a document
  object may carry it under the key `run` instead (passing both is a `ValueError`).
  `executor` selects the executor for the in-memory capability check and defaults to
  the record's `executor`.
- `DocumentFinding` has `code`, `severity`, `pointer`, `message`, `fix` and
  `in_memory`, the core `Finding` it translates (`None` for the document codes).
  Findings are ordered errors first, then by code and pointer.
- `resolve_pointer(root, pointer)` returns the value a pointer names, and raises
  `KeyError` when there is none.
- `preflight_document` and `DocumentFinding` are exported from `spaceodyssey.codec`
  (`resolve_pointer` and `DocumentCode` from `spaceodyssey.codec.preflight`).
- `RegistrationError` now lives in `spaceodyssey.codec.kinds` and is re-exported from
  `spaceodyssey.plugins`, which still raises it.

Document content never raises. A broken registration (an injected mapping the resolver
refuses) is a `RegistrationError` raised when the resolver is built, with the original
`ValueError` as its cause; it is a broken installation, not a finding.

### What runs, and what blocks what

Independent checks always run, whatever else is wrong with the document:

- the schema version (`D002`);
- every `$kind` and `$enum` reference through `KindResolver.resolve` (`D001`, all of
  them);
- the envelope, the content hash and the text itself (`L001`);
- the run record (`D003`): member presence and form, the executor, and the host and
  compiled model hashes.

The checks that need a decoded study run only when the document decodes: the ten
in-memory codes, and the part of `D003` that asks which components declare a `Query`.
A document with a `D001` or a `D002` does not decode (the loader cannot build an
unknown or newer kind), so it has no in-memory findings; its independent findings are
still all reported.

A document that does not decode for any other reason is reported as `L001` at the
pointer of the failure, beside the independent findings: malformed JSON, a key written
twice and a non-finite number (refused exactly as `load_study` refuses them, through
the shared parser `codec.document.parse_text`), a wrong envelope, a hand-edited hash, a
wrong shape, a module whose validation fails. `L001` is not one of the thirteen finding
codes; it is the in-band form of the `DocumentError` the loader raises. One field-level
failure is reported with its own code instead of aborting the load: a `null` coincident
order decodes (a pre-flight-only tolerance in `codec/load.py`), so `T002` is reported
with the other in-memory findings.

## The thirteen codes

Every pointer names the value itself and resolves in the document.

| Code | Raised when | Pointer |
|---|---|---|
| `F001` | an observer or policy reads truth outside a measure crossing | the read: `/study/observer/reads/0` |
| `F002` | an observer or policy reads a scorer | the read |
| `C001` | a decision reads a product not yet released to it | the need: `/study/agenda/policies/<i>/needs/<j>` |
| `K001` | draws without stable-id coordinates | the coordinates: `/study/channel_model/key_coords` |
| `K002` | two measures with identical key coordinates | the repeated id: `/study/channels/id/data/<i>` |
| `X001` | a capability or effect the selected executor does not provide | the effect, requirement or executor list: `.../effects/<j>`, `.../requires/<j>`, `.../executors` |
| `X002` | a reserved field set to a non-default value | the first offending element: `/study/channels/charge/data/<r>/<c>`, `/study/assets/freeze_ticks/data/<i>`, `/study/assets/disruptive_resource/data/<i>`, or the policy's `.../disruptive` |
| `T001` | a drain below campaign scope | the pair: `/study/agenda/policies/<i>/stops/<j>` |
| `T002` | a channel with no valid coincident order | the first order outside the enum, `/study/channels/coincident_order/data/<i>`; the field when null or of the wrong shape |
| `R001` | a channel declares a record kind the registry lacks | the kind element: `/study/channels/kind/data/<i>` |
| `D001` | a kind id or version the resolver does not know | the reference: `.../$kind` or `.../$enum` |
| `D002` | a schema version newer than the loader | `/schema_version` |
| `D003` | a binding missing or ill-typed, no service for a `Query`, or refit model hashes that differ | `/run`, a binding, a member such as `/run/service/kind`, or `/run/refit/compiled/model_hash` |

How a declaration is located: the value is found in the decoded component (the string
`"truth"` among its fields, the need triple, the key-coordinate tuple, the effect
number, the drain pair, the executor set), then followed through the document's own
encoding. A field written as `{"$list": [...]}`, `{"$dict": [...]}` or an array gets
the pointer that holds it (`/study/agenda/policies/$list/0/reads/$list/0`). The
follow stops at the deepest value the encoding holds, so a declaration a class computes
instead of storing points at the component, and every pointer resolves. The core
finding carries no source position, so the match is by value, and a repeated finding of
one kind in one component is matched by position (the second drain to the second pair).
The X001 kind is read from the start of the core message (`asks for`, `requires`, `runs
only on`).

An in-memory finding keeps its code, severity, message and fix; only the path becomes a
pointer. K001 is raised once per channel by the core, so a channel model that omits a
coordinate gives three findings at one pointer.

## The run record and D003

A study holds no service, so the Query service and the refit boundary are bound in a
run record, a JSON object next to the study document:

```json
{
  "executor": "host",
  "service": {"kind": "...", "version": 1, "configuration": {}},
  "refit": {
    "host": {"kind": "...", "version": 1, "configuration": {}, "model_hash": 7},
    "compiled": {"kind": "...", "version": 1, "configuration": {}, "model_hash": 7}
  }
}
```

Binding schema: `kind` a non-empty string, `version` a positive integer (not a bool),
`configuration` an object, `model_hash` an integer or a non-empty string. `D003` is
raised when:

- a component the core pre-flight reads (the same inventory: world source, process,
  channel model, observer, objective, design and each policy) declares a `Query` effect
  and the record has no `service`, at `/run`;
- a binding is not an object, lacks a member (at the binding) or holds a null or
  ill-typed member (at the member), a refit side is absent, or `executor` is not
  `host` or `compiled`;
- the host and compiled `model_hash` differ, at `/run/refit/compiled/model_hash`. Each
  side's hash is compared when it is valid, whatever else is wrong with its binding,
  and two null hashes are two defects, not a match.

Without a record the check is skipped, since a study document alone binds nothing.

The refit boundary types of the refresh task are not imported: the check compares plain
members of the record, whose names follow the boundary's `binding()` (`kind`, `version`,
`model_hash`) plus `configuration`. The seam is the schema tuples in
`codec/preflight.py` and the `run` key; when the run document lands it adds its envelope
around the same record and these names stay.

## Decisions

- `kinds=None` means the core kinds only, as in `dump_study` and `load_study`. The
  codec never reads installed entry points itself (T5c), and a pre-flight that treated
  `None` differently would say clean on a document `load_study(doc)` then refuses. A
  caller that wants installed plugin kinds passes `installed_resolver()`.
- Document content is reported, not raised (`L001`), so a caller always gets the
  independent findings with the failure.
- A `null` coincident order is tolerated by the pre-flight decode only; `load_study`
  still refuses it.

## Tests

`tests/codec/test_preflight_json.py` (71 cases), unit tier:

- `test_seeded_document_reports_its_code_at_the_offending_value`: 14 cases, one seeded
  document per code (`D003` twice). Each asserts the one code, the exact pointer, that
  the pointer resolves to the offending value, and for the ten in-memory codes that the
  finding wraps the core pre-flight's finding on the loaded study. F001, F002 and C001
  use the toy kinds `DocObserver` and `DocPolicy` in `tests/core/preflight_cases.py`,
  whose declarations read plain fields so a document can state them; `DocChannelModel`
  does the same for key coordinates and effects.
- `test_repaired_document_is_clean_and_the_check_discriminates`: the vacuity control,
  for every seeded case.
- `test_each_declaration_is_located_at_its_own_value` (7 cases: X001 requirement and
  executor list, X002 charge, resource and policy, T002 shape, a second C001 need) and
  `test_second_drain_is_located_at_the_second_pair`.
- `test_values_are_located_through_container_tags` (`$list` around the policies and the
  reads) and `test_follow_reads_dict_entries_and_array_rows`.
- `test_null_order_is_reported_beside_independent_findings`,
  `test_document_and_binding_findings_are_reported_together`,
  `test_documents_that_cannot_decode_are_reported_not_raised` (4 cases),
  `test_malformed_text_is_one_load_finding`,
  `test_repeated_key_and_non_finite_number_are_refused_like_the_loader`.
- `test_every_component_that_asks_a_query_needs_a_service`,
  `test_binding_members_are_checked_against_the_schema` (10 cases),
  `test_two_null_hashes_are_not_equal_hashes`,
  `test_an_incomplete_binding_does_not_hide_a_hash_mismatch`,
  `test_incomplete_bindings_are_findings`.
- `test_all_findings_are_reported_not_only_the_first`, `test_every_unknown_reference_is_reported`,
  `test_unknown_enum_reference_is_reported`, `test_clean_document_has_no_findings`,
  `test_document_forms_agree`, `test_run_record_selects_the_executor`,
  `test_service_is_required_only_when_a_component_asks`, `test_kinds_none_means_the_core_kinds`,
  `test_missing_id_is_a_finding_not_an_import_error`,
  `test_broken_installation_is_a_registration_error_not_a_finding` (exact type and
  cause) and `test_resolve_pointer_follows_escapes_and_refuses_missing_values`.

Retired: `test_newer_schema_reports_only_d002_even_with_bad_kinds` (the behavior it
pinned is the defect the review found) and `test_documents_that_cannot_load_raise_with_a_pointer`
(replaced by the reported-not-raised cases).

## Review fixes

A review of the first version found eight gaps, fixed together on `slice2-t6-r` with a
test for each:

- Pointers stopped at the containing module or the whole column; they now name the
  value (table above).
- Pointers assumed plain tuple encoding; they are now followed through `$list`,
  `$dict` and array encodings.
- An unknown kind or newer schema returned only that finding, and a field-level decode
  failure hid the core findings; independent checks always run and a decode failure is
  an `L001` beside them (see "What runs").
- The `Query` check read only the observer, the process and the policies; it now uses the
  core's component inventory.
- Null-filled bindings passed; members are validated against a schema.
- An incomplete binding hid a hash mismatch; hashes compare independently.
- Duplicate keys and non-finite numbers were not refused; the document parser is shared
  with `StudyDocument.from_json`.
- A broken registration raised a plain `ValueError`; it is a `RegistrationError` with
  its cause.

## Timing

The 71 cases run in 1.9 s together; the slowest is 0.07 s. Pre-flight loads and checks a
study and never compiles it.
