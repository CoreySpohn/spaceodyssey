---
orphan: true
---

# Slice 3a survey 11: targets of opportunity

Survey 11 adds `src/spaceodyssey/studies/survey_11.py` (the study, the alert process and the
policy), `tests/studies/test_survey_11.py` (assertions only) and one budget line in
`pyproject.toml`. It changes no core file, no golden and no pin.

Asset 0 owns lane 0, which runs channel 10: 100 ticks of occupancy, 10 of transition, a
release delay of 20. Its price differs from its cost (10 in every column) in each resource
column: asset column SURCHARGE (charge 20, surcharge 2), program column OCCUPANCY (charge 12),
shared column FLAT_FRACTION (charge 8). A scripted alert is written at tick 40 and released to
the policy 15 ticks later. A level-1 array policy scans the released rows of its snapshot for a
row on the alert channel and, once, emits a live `Cancel` of lane 0; it starts the subject at
ticks 0 and 120, so the second start shows what became of the cancelled index. The log ends
with three rows: the alert, the partial record and the completed second exposure.

The alert is `ScriptedAlerts`, a `ScriptedRecords` with a `delay` field (written at its fire
tick, available `delay` ticks later), addressed to the policy audience alone so the observer
folds nothing from it. `ScriptedRecords` itself has no release delay.

## What it shows

- **(a) the cancelled generation settles nothing.** The first exposure would finish at tick
  110. The log holds one record for it, `PARTIAL`, and none `COMPLETED` carries its index. The
  only `DONE` the run handles is the second exposure's, at tick 230; none is handled at tick
  110, because the cancel invalidates the lane's pending `DONE` row. A separate test also
  delivers the cancelled generation's `DONE` to the post-cancel carry through the real
  handler path (the row read from the queue before the Cancel ran, with the generation it was
  queued with): the log, the ledger and the acquisition counts are unchanged and no double
  settlement is flagged. That test is what holds the stale-generation guard and the
  generation retirement in `_cancel_lane`. It delivers in the tick of the cancel, not tick
  110: the delivery helper enqueues at the carry's tick, and the partial record's queued
  release (tick 75) would be handled first at a later tick.
- **(b) the partial fraction.** Started at tick 10, cancelled at tick 55: 45 of 100 planned,
  fraction 0.45 to the go tolerance.
- **(c) charge and credit, every column and payer row.** The expected values are worked by
  hand in the test file. Cancelled at 0.45: column 0 keeps (22 - 2) * 0.45 + 2 = 11 (2 fixed)
  and is credited 11; column 1 keeps 5.4 and is credited 6.6; column 2 (flat) keeps all 8 and
  is credited 0. The completed second exposure keeps its whole price. The test compares the
  full debited, refunded and balance tables (asset row, program row, shared row, and the
  untouched second asset row): 67, 82.6 and 84 against 100.
- **(d) the index.** The partial record has index 0, the next exposure has index 1, and the
  subject's admission count is 2.
- **(e) the latency.** The alert is written at tick 40 and available 15 ticks later, the
  declared `delay`; it is released at tick 55, the policy answers in that tick, and the one
  `CANCEL` event handled lands at tick 55. The partial record is available and released at
  tick 75 (the channel's latency of 20 after the cancel) and the completed record 20 ticks
  after its `DONE`.
- **D8, live cancel is not frozen.** A variant run (commitments on, awards off, asset 0 with a
  30-tick horizon and a disruptive resource, the policy's starts disruptive) admits the live
  `Cancel` inside the horizon. Only the two starts spend an activation (`disruptive_used` is
  2), the cancel spends none.

Evidence counts, all required nonzero: 1 alert row, 1 cancel effect emitted and 1 admitted, 1
cancel event handled, 1 partial and 1 completed record, 1 `DONE` handled, 1 process event, 1
retained fixed part, 2 credited cells; the late `DONE` test counts its one delivery and the
queue row it came from; the variant counts its one admitted `Cancel`. No refusal is asserted.

## Source mutants

The checks are plain functions, so a mutant is held to the same assertions. Each replaces one
piece of source with `mutate`, clears the JAX caches and runs the study; the late `DONE`
delivery and the frozen variant run under it too. The ids M02 to M18 are the review's.
Old code: everything passes.

| Mutant | Source change | Fails |
|---|---|---|
| a | `_cancel_lane` leaves the pending `DONE` row valid | (a), late `DONE` |
| b | record gets fraction 1 | (b) |
| c | retained charge without the fixed part | (c) |
| d | cancel decrements the subject's acquisition count | (a), (c), (d), (e), evidence |
| e | policy-only rows released one tick late | (b), (c), (e) |
| M02 | `h_done` drops the stale-generation guard | late `DONE` |
| M03 | cancel does not retire the lane generation | late `DONE` |
| M09 | credit is cost times unspent instead of charge less kept | (c) |
| M14 | completed record loses the channel's release latency | (e) |
| M15 | partial record released after one tick | (e) |
| M16 | the freeze also applies to a live cancel | frozen variant |
| M17 | credit only resource column 0 | (c) |
| M18 | pricing reads the cost, not the declared charge | (c) |

Mutants d and e move ticks or indices the other checks pin, so they fail more than their own
check. They are run by a scratch script, not kept as tests: each is a cold recompile.

## Finding

The `stale` guard in `h_done` is not what stops the cancelled generation's `DONE` in a live
run: `_cancel_lane` invalidates the pending row, so the event never reaches `h_done`. The
guards behind it are reached only by a late delivery, which the late `DONE` test makes. No core
change is needed. `ScriptedRecords` has no release delay; a `delay` field on it would replace
the subclass.

## Executors and tiers

Row 11 names no executor. The file runs the host executor once in a module-scoped store, and
every check reads that run. One compiled run adds a parity check (ints exact, floats in the go
band). All nine tests are medium, with reasons. The budget plugin splits the host fixture
setup among the eight tests that use it, about 0.4 s each, so no one test carries it.

| Measure | Value |
|---|---|
| File, cold, caches off (one Mac) | 16 s to 25 s wall, 9 tests |
| Host run (shared), compiled run, frozen variant | about 6.5 s, 6.2 s, 8.0 s |
| Compiles attributed to the file (session total) | 155 (159) |
| Budget line | `"tests/studies/test_survey_11.py" = 189` |
