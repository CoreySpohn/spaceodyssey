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

1. `feat(codec): report all thirteen pre-flight codes on a JSON document`
2. `docs(changes): record Slice 2 T6`

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
  `in_memory`, the core `Finding` it translates (`None` for `D001` to `D003`).
  Findings are ordered errors first, then by code and pointer.
- `resolve_pointer(root, pointer)` returns the value a pointer names, and raises
  `KeyError` when there is none.
- Both are exported from `spaceodyssey.codec` (`resolve_pointer` and `DocumentCode`
  from `spaceodyssey.codec.preflight`).

Order of work: the schema version (`D002`), then every `$kind` and `$enum` reference
through `KindResolver.resolve` (`D001`, all of them), then, if no reference failed, the
load and the in-memory pre-flight on the loaded study, then the run record (`D003`).
A newer schema stops after `D002`, and unknown kinds stop after the `D001` findings,
because the in-memory checks need a study. A document that fails to load for another
reason (malformed JSON, a wrong shape, a hand-edited hash, a module whose validation
fails) raises `DocumentError` with its pointer, as `load_study` does. A
`RegistrationError` or other resolver-build error is a broken installation and is
raised, never a finding. A missing id is `D001`, never `ImportError`.

## The thirteen codes

| Code | Raised when | Pointer |
|---|---|---|
| `F001` | an observer or policy reads truth outside a measure crossing | the reading component: `/study/observer` or `/study/agenda/policies/<i>` |
| `F002` | an observer or policy reads a scorer | the reading component |
| `C001` | a decision reads a product not yet released to it | the policy: `/study/agenda/policies/<i>` |
| `K001` | draws without stable-id coordinates | the declaring component: `/study/channel_model`, `/study/world_source` or `/study/process` |
| `K002` | two measures with identical key coordinates | the repeated id: `/study/channels/id/data/<i>` |
| `X001` | a capability or effect the selected executor does not provide | the component: `/study/<component>` or the policy |
| `X002` | a reserved field set to a non-default value | the field: `/study/channels/charge`, `/study/assets/freeze_ticks`, `/study/assets/disruptive_resource`, or the policy |
| `T001` | a drain below campaign scope | the policy |
| `T002` | a channel with no valid coincident order | `/study/channels/coincident_order` |
| `R001` | a channel declares a record kind the registry lacks | the kind element: `/study/channels/kind/data/<i>` |
| `D001` | a kind id or version the resolver does not know | the reference: `.../$kind` or `.../$enum` |
| `D002` | a schema version newer than the loader | `/schema_version` |
| `D003` | a service binding missing, or refit model hashes that differ | `/run`, `/run/service`, `/run/refit` or `/run/refit/compiled/model_hash` |

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

`D003` is raised when a component of the study declares a `Query` effect and the record
has no `service` (at `/run`); when `service` or a refit binding lacks `kind`, `version`,
`configuration` or (refit) `model_hash`, or a refit side is absent (at that binding); and
when the host and compiled `model_hash` differ (at `/run/refit/compiled/model_hash`).
Without a record the check is skipped, since a study document alone binds nothing.

The refit boundary types of the refresh task are not imported: the check compares plain
members of the record, whose names follow the boundary's `binding()` (`kind`, `version`,
`model_hash`) plus `configuration`. The seam is the member list in
`codec/preflight.py` (`_SERVICE_FIELDS`, `_REFIT_FIELDS`) and the `run` key; when the
run document lands it adds its envelope around the same record and these names stay. The
tests build the record by hand (`tests/codec/test_preflight_json.py`, `_run`).

## Decisions

- `kinds=None` means the core kinds only, as in `dump_study` and `load_study`. The
  codec never reads installed entry points itself (T5c), and a pre-flight that treated
  `None` differently would say clean on a document `load_study(doc)` then refuses. A
  caller that wants installed plugin kinds passes `installed_resolver()`.
- The pointer of a finding names the component or value the in-memory path names, at the
  finest level the path locates: a table element for `K002` and `R001`, a column for
  `X002` and `T002`, the module otherwise. A finer pointer into a declaration would need
  the declaration to be a field, which it is not in general.
- `T002` for a missing order (`None`) cannot occur in a document: the codec refuses a
  `None` array on dump and load. The document case is an order outside the enum.
- Findings that no check of the thirteen covers are not invented: a loadable document
  with no finding returns an empty list.

## Tests

`tests/codec/test_preflight_json.py` (43 cases, 16 functions), unit tier:

- `test_seeded_document_reports_its_code_at_the_offending_value`: 14 cases, one
  seeded document per code (`D003` twice, for the missing service and for the hash
  mismatch). Each asserts the one code, the pointer, that the pointer resolves to the
  offending value, and for the ten in-memory codes that the finding wraps the core
  pre-flight's finding on the loaded study. F001, F002 and C001 use the toy kinds
  `DocObserver` and `DocPolicy` in `tests/core/preflight_cases.py`, whose declarations
  read plain fields so a document can state them; `DocChannelModel` does the same for
  key coordinates.
- `test_repaired_document_is_clean_and_the_check_discriminates`: the vacuity control.
  The repaired document of each case has no findings, and the value check fails on the
  repaired value at the same pointer.
- `test_all_findings_are_reported_not_only_the_first` (six findings, five codes, one
  document), `test_every_unknown_reference_is_reported`,
  `test_unknown_enum_reference_is_reported`, `test_clean_document_has_no_findings`.
- `test_document_forms_agree`, `test_run_record_selects_the_executor`,
  `test_service_is_required_only_when_a_component_asks`,
  `test_incomplete_bindings_are_findings`,
  `test_newer_schema_reports_only_d002_even_with_bad_kinds`,
  `test_kinds_none_means_the_core_kinds`,
  `test_missing_id_is_a_finding_not_an_import_error`,
  `test_broken_installation_is_an_error_not_a_finding`,
  `test_documents_that_cannot_load_raise_with_a_pointer` and
  `test_resolve_pointer_follows_escapes_and_refuses_missing_values`.

Retired: none.

## Timing

The 43 cases run in 1.5 s together; the slowest is 0.06 s. Pre-flight loads and checks a
study and never compiles it.
