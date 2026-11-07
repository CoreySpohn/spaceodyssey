---
orphan: true
---

# Slice 3a S2: programs per activity, per-program Stop and P001

S2 gives every activity a program, charges the program's balance row, lets a PROGRAM Stop
close one program without closing lanes, and adds the pre-flight finding P001 for a program
out of range. It is tier 2 (core handlers, records and pre-flight). With one program (the
default) nothing changes: the compiled program, every history digest and every pin are those
of `main`.

## What is in

- **Program column.** `EffectTable.program` is copied onto the queued DONE row and onto the
  record (`Record.program`, `RecordLog.program`). The column exists only when
  `caps.n_programs > 1`: with one program the field is `None`, so it adds no pytree leaf and
  the default scan is unchanged. Code that touches it is guarded by a static
  `caps.n_programs > 1` branch.
- **Ledger rows.** Balance rows are the assets, then the programs, then one shared row.
  `_resource_rows(study, lane, program)` charges row `n_assets + program`; a cancel's refund
  goes to the owner's row, read from the pending DONE row's program.
- **PROGRAM Stop.** With `n_programs > 1` a PROGRAM-scope Stop sets
  `ledger.program_closed[program]`. It does not close any lane. Drain and abort act only on
  live DONE rows whose program matches, across all lanes. Admission of an activity of a closed
  program is refused with `AdmissionCode.LANE_CLOSED`. With one program the old path runs.
- **Leaves.** `program` (event, record log, view) and `ledger.program_closed` are conditional
  leaves, listed in S1's `CONDITIONAL_FIELDS` and `CONDITIONAL_GROUPS` and present only where
  `n_programs > 1`. The codec rule is unchanged: nothing new is a document field.
- **Visibility.** The policy sees program ids; the observer shows none (the view carries no
  program column for the observer audience).
- **P001.** A program outside `[0, n_programs)` is reported by pre-flight in memory and, for a
  JSON document, at its pointer. It follows the T001 pattern: `Declaration.programs`, with a
  policy field named `programs` so the pointer resolves.
- **Battery.** Five battery sites iterated every column of the record log and would meet the
  absent program column. They now iterate `records.log_columns(log)`, which returns the
  columns a log holds.

## Moves

None. `tests/core/golden_histories.json` and every pin file are unchanged.

| Check | Result |
|---|---|
| Default scan StableHLO | byte-identical to `main` (a2d72ce): module text compared after the same lowering |
| Golden integer digests, all histories including the twelve `stop_*` | equal to the pins |
| Golden float digests | 18 differ on this Linux machine, plus the 2 fault-fixture entries, as on `main`; the Mac run is the check |
| `tests/codec/test_run_document_runs.py` (hash pin, 58 documents) | passes |
| Fingerprint pins, canonical-fields, guard and lint tests | pass |

## Sibling sites checked

- `_resource_rows` callers: handlers admission, completion and cancel paths all pass the
  program; the change to the function itself is one parameter and one added row index.
- Every `dataclasses.fields` / `__dataclass_fields__` walk over `EventTable`, `Record`,
  `RecordLog` and `Ledger`: `events.py`, `records.py`, `snapshot.py`, `handlers.py` skip or
  tolerate the program column; the battery sites above use `log_columns`; the two test
  walkers (`test_stop_equivalence.py`, `test_planit_adapter.py`) skip `None`.
- `STUDY_SLOTS` is unchanged: no `Study` field was added.
- The dispatch lint (K1a) forbids `getattr`, `vars` and `type` on study or caps objects in
  core. Replacing the ten battery lines removed their allowances from
  `tests/core/gate_helpers.py`; the stale-entry test confirms none is left.
- - Shared-file edit: `.pre-commit-config.yaml` adds `program` to the `name-tests-test` exclude
  list (support modules), for `tests/core/program_helpers.py`.
- Not touched, as the card requires: `studies/premission.py`, `prior_observer.py`,
  `precursor.py`, `codec/files.py`.

## Tests

| File | Tests | Tier | Cold time (alone, cache off) |
|---|---|---|---|
| `tests/core/test_programs.py` | 20 | unit and medium | 72 s in all; the four mutants about 13 s each, the shared stop run 13.6 s setup |
| `tests/core/test_slice3_guard.py` | 18 (existing file) | unit and medium | 33.3 s, medium tests 9.8, 9.7 and 9.0 s (the three key-version tests share one compiled run) |

Controls and mutants: a ledger that ignores the program (charges the first row), five extra
leaves against the one-program pin, boundary cases for P001, a view that shows the label (the
observer leak control), and four mutants: a Stop that closes every lane, a Stop that never
sets the flag, an abort that ignores the program, and a refund to the first program row.

Two existing tests exceed the budget on this machine cold
(`test_epoch_relation...negative_times...` 33.8 s and `test_snapshot_fingerprint_recorded`
21.2 s). They take the same time on `main` (33.75 s and 21.19 s), so S2 did not move them.

## Spec silences decided here

1. The program columns are `None` when `n_programs == 1`, not size-0 or zero-filled arrays.
   Rejected: size-0 or zero-filled columns, which would change the compiled program.
2. A closed program's admission refusal reuses `AdmissionCode.LANE_CLOSED`. Rejected: a new
   code, which is a wider change than the card names.
3. P001 is a policy declaration (`Declaration.programs`), as T001 is.
4. `EFFECT_FIELDS` excludes `program`; `empty_event_row(caps)` builds the row for a given
   capacity.
5. No `Study` field, so `STUDY_SLOTS` is unchanged.
6. The scripted program policy lives in `tests/core/program_helpers.py` only.
7. Battery iteration goes through `records.log_columns` rather than allowlisting `getattr`
   lines, so the dispatch lint's allowlist shrinks by ten entries.

## K3 and K4 attribution

`handlers.py` gained a branch, so K3 was rerun locally on this Linux machine, four threads,
cache off, two worlds, 400 steps, 30 repeats, `main` (a2d72ce) and S2 alternating over two
rounds:

| Round | `main` min per step | S2 min per step | `main` mutant ratio | S2 mutant ratio |
|---|---|---|---|---|
| 1 | 277.3 us | 264.2 us | 1.506 | 1.459 |
| 2 | 269.9 us | 264.2 us | 1.466 | 1.450 |

S2 is not slower than `main` (the default scan is the same program), and the calibrated
mutant still fails K3 as a control must. These figures are not comparable with the S0 macOS
baseline (about 100 us per step); the Mac and CI runner remain the baseline of record. Core
edits (`events.py`, `records.py`, `snapshot.py`, `ledger.py`, `carry.py`, `service.py`,
`refresh.py`, `executors/`, `conformance/observer_kit.py`, `transaction.py`, `handlers.py`,
`declarations.py`, `preflight/`, `battery/` log walks) belong to S2.

## Review round 1

- `EffectTable.program` is an optional document field (`OPTIONAL` metadata, default `None`;
  the codec's plain-default rule now accepts `None`). A one-program table is written without
  the key, a document from before programs loads with `None`, `"program": null` is refused
  as two documents for one table, and a populated column round-trips. The four codec tests
  fail without the marker.
- A producer that leaves `program` as `None` (`FixedPlanPolicy`) is program 0 when
  `n_programs > 1`: `_program_of` returns 0 for a missing column, and the single-program
  static path is unchanged. `toy.survey_1(caps=Capacities(n_programs=2))` is the test (it
  raised `TypeError` before).
- The four compiled mutants are slow tier. The relabelling check and the observer-leak
  control share one module fixture. `test_programs.py` quick tier (unit and medium): 34.4 s
  cold, alone, cache off (was 72 s); the mutants take 12.7 to 14.1 s each in the slow tier.
- Default scan StableHLO re-checked under this machine's current JAX: `main`, 7431b06 and the
  fixed tree lower to the identical module.
