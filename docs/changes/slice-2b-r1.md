---
orphan: true
---

# Slice 2b R1: the pre-mission table, its importer and the pre-flight hook

R1 builds the table contract frozen in R0a (`slice-2b-r0a.md`): `SourceId`,
`PreMissionTable` with a supplied `local_index`, the importer with fixed per-audience
rows and kickoff retargeting, codec registration of the new kinds, the pre-flight
context and component hook, and findings `D004` to `D007`.

## Commits

1. `feat(core): pre-flight context and a validate hook for component findings`
2. `feat(studies): pre-mission table, source identities and importer with findings D004 to D007`
3. `test(core): import protocol, kickoff and identity checks for pre-mission tables`
4. this record

Commit 2 is about 800 lines (the module and the finding cases of both pre-flight test
files land together so every commit is green), above the 600-line guide.

## What changed

- `core/preflight/checks.py`: `RunBudget(n_events, n_log, n_steps)` (with
  `RunBudget.of(study, n_steps)`), `PreflightContext(boundary, budget,
  upstream_document)`, and `preflight(study, executor=None, context=None)`. After the
  fixed checks, `components(study, context)` calls `validate(study, context)` on the
  process and the observer when they define it (looked up by its literal name, the way
  `declaration_of` looks up `declare`) and fills in the slot of each finding. Every
  existing call keeps its meaning.
- `core/preflight/findings.py`: `D004` to `D007`. A locator's `member` may be a dotted
  path (`table.local_index`) naming a field of a module the slot holds.
- `codec/preflight.py`: the pointer walk follows a dotted member field by field, so a
  table finding points at its element (`/study/process/table/local_index/data/1`). The
  docstring counts seventeen codes. The JSON path passes no context yet: `D005` to
  `D007` and the unverified `D004` warning are reported; the boundary (R2) and the
  upstream document (R3) are not wired.
- `studies/premission.py` (new): `SourceId`, `PreMissionTable`, `PAD`,
  `split_rows`, `ImportDemand`, `PreMissionImporter` (`declare`, `seed`, `handle`,
  `validate`), the finding functions, `authored_table` and `PREMISSION_KINDS`.

### The import protocol as built

- `seed` retargets the kickoff wait to the release stage, then queues, per present
  row, one `PROCESS` event at tick 0 in the release stage with group equal to the row
  index; a public row of a derived kind gets a second event, the first with audience
  `OBSERVER` and the second `POLICY`. Padding rows (namespace -1) queue nothing and take
  no id.
- `handle` writes row `ev.group` with `released_tick` and `accepted_tick` at -1,
  `acquisition_index = local_index`, `subject = source.subject`, `revision =
  source.revision`, `start_tick` and `end_tick` from the offsets, `available_tick` the
  event tick, and a fresh stable id as its acquisition id. `h_process` then queues the
  release at tick 0, which stamps the row.
- Cost for E written rows (a public prior counts twice): 2E events, 3E stable ids, E log
  rows, E + 1 queue slots at tick 0 (every process row and the kickoff), and 2E + 1
  executor steps before the first decision. `ImportDemand.of(table, caps)` reports
  these; the run test checks the step and id counts exactly.

### Findings

| Code | Raised when | Locator member |
|---|---|---|
| D004 | a declared `upstream_hash` differs from the supplied document's `content_hash` (error); declared with no document: warning "declared, unverified" | `table.upstream_hash` |
| D005 | an in-table (`external` false) pedigree entry names no present row | `table.pedigree.source_index`, entry `(row, entry)` |
| D006 | a repeated source identity, a `local_index` repeated on a channel, `n_pedigree` outside `[0, W]`; also a namespace below -1, a subject or kind outside the study, a payload wider than a record, and, with a budget, an import over the queue, log, stable ids or steps | the column, or `table` for capacity |
| D007 | a positive offset, or a start after its end | `table.start_offset`, `table.end_offset` |

## Tests

Tests added: 32, retired: none. `tests/core/test_premission.py` 11 (8 unit, 3
medium); `tests/core/test_preflight.py` 13 (four seeded cases, each in its mistake and
locator tests, one correct import study, the lineage test and three budget cases);
`tests/codec/test_preflight_json.py` 8 (four cases, each in its seeded and repaired
tests).

Cold times, each test alone in a fresh process (`SPACEODYSSEY_JAX_CACHE=off`, four
threads, wall time including the about 2.5 s import):

| Test | Tier | Cold s |
|---|---|---|
| `test_import_rows_are_stamped_folded_once_and_seen_by_the_one_first_decision` | medium | 16.2 |
| `test_one_initial_decision_with_and_without_a_public_zero_information_import` | medium | 16.0 |
| `test_an_observer_only_row_moves_no_public_identity_and_order_moves_none` | medium | 13.6 |
| slowest unit: `test_a_row_is_written_unset_...`, `test_an_overflow_...` | unit | 4.7 |
| other units (6) | unit | 2.6 to 4.4 |

The new pre-flight and JSON cases run in under 0.1 s each. The three medium tests
compile two study signatures (the import study and plain `survey_1` with a derived
kind) on the shared transition and both executors; they run in the quick CI job.

## Checks run

- `tests/core/test_golden_histories.py` with `tests/codec/test_run_document_runs.py`
  (the golden pin and the corpus reload with its document hash pin): 189 passed in
  426.6 s.
- Quick tier (`-m "not slow" -n 2 --dist loadfile`, deselecting
  `tests/codec/test_entry_points_installed.py` for the stale environment metadata
  noted in R0a): 1,997 passed in 224.3 s. The budget plugin reported one unit over
  5 s, the existing `test_document.py::test_reloaded_study_runs_and_reuses_the_compiled_program`.
- `ruff check`, `ruff format --check`, `tools/check_internal_refs.py`, and the
  domain-word and core-import lints: clean. Every touched file is ASCII.
- The full suite and the Sphinx docs build were not run.

## Mutants and controls

Each mutant was applied to a copy of `src/` and the three touched test files run
against it (`$TMPDIR` script, not committed). All are killed.

| Mutant | Failing tests |
|---|---|
| kickoff not retargeted | seed test, B1 run test, initial-decision test (2 decisions at tick 0) |
| release at tick 1 (row delivered unstamped at 0) | handle test, B1 run test, initial-decision test |
| every row queued twice (folded twice) | seed, overflow, B1 run, hidden-row tests |
| acquisition index from the row position | handle test, hidden-row and permutation test |
| no per-audience split of a public prior | demand, seed, B1 run, initial-decision tests |
| validate hook removed (findings unreachable) | 18 tests: every seeded D case, the report counts, identity and lineage tests |
| dotted locator ignored by the JSON walk | the four JSON D cases |
| lineage always passes | lineage test |
| queue peak counted without the kickoff | demand, overflow, budget tests |
| external pedigree entries checked against the table | pedigree test |

Shipped control: the initial-decision test steps the same study with the kickoff left
in the admit stage and asserts a second decision and a different plan state.

## Pins

No pinned golden digest, fingerprint or corpus document hash moved
(`golden_pin.json` and `hash_pin.json` untouched and passing), and `schema_version` is
unchanged. No core runtime file changed: the only core edits are
`core/preflight/checks.py` and `core/preflight/findings.py`, the paths K2 allows.

## Spec silences decided here

1. Capacity and range refusals are `D006`. The contract froze `D004` to `D010`, and a
   new code would collide with the parallel Slice 3a additions; `D006` already covers
   "a row the study cannot hold" (`n_pedigree` over the width). Rejected: a new code.
2. `D004` with no upstream document is a warning, "declared, unverified". Rejected:
   no finding (the report would not say it was unverified).
3. The kickoff wait moves to the release stage at microstep 0 and is handled before
   the imports, not after them. Wakes coalesce until nothing is queued ahead of the
   decide stage, so the one decision fires after the last release either way; placing
   the wait after the releases would push the decision to microstep 2 and bring the
   microstep guard closer. The retarget is unconditional (static), also for
   observer-only or padded tables, so it depends on nothing the policy cannot see.
4. A prior is a row of a derived kind (`caps.derived_kinds`); only a public one is
   split. The record subject is `source.subject` (the schema has no separate local
   subject) and the record revision is `source.revision`.
5. The policy copy of a split prior keeps `accepted_tick = -1`: `h_release` accepts
   only observer rows, and changing that is a core change. Its metadata is fixed
   whatever the exclusion, which is what D5 needs; the spec's "accepted tick 0" for
   both rows is not met.
6. The two copies of a split prior share `(channel, subject, acquisition index,
   revision)` and differ in audience and acquisition id. R2 should confirm the kit
   invariants accept that.
7. Registration is the mapping `PREMISSION_KINDS` (`process/premission.importer`,
   `table/premission.table`, `table/premission.source_id`), merged into the mapping a
   document is dumped and loaded with. No entry points: `test_entry_points_installed`
   requires every installed kind to round-trip in a corpus study, and the corpus is
   frozen; the plugin families also have no data-table family. `codec/kinds.py` is
   unchanged.
8. Window columns stay in the table; writing them into a record's payload depends on
   exclusion and the window kind (R2, R4b). `PedigreeRule` and `resolve_overlap` are
   left to R2.
9. `PreflightContext` and `RunBudget` are imported from
   `spaceodyssey.core.preflight.checks`; `core/preflight/__init__.py` and
   `core/__init__.py` are not edited (K2 lists only `findings.py` and `checks.py`).
10. The capacity check runs only with a budget, as D10 states, although `n_events` and
    `n_log` are also in the study's capacities.

## Spec points not satisfied

- The JSON pre-flight builds no context: no budget or boundary from the run record and
  no upstream document argument (R2 and R3 own those mappings).
- The policy copy of a split prior is not accepted at tick 0 (silence 5).
- Windows are not written into imported records (silence 8).
