---
orphan: true
---

# Slice 3a survey 11: targets of opportunity

Survey 11 adds one test file, `tests/studies/test_survey_11.py`, and one budget line in
`pyproject.toml`. It changes no core file, no golden and no pin.

Asset 0 owns lane 0, which runs channel 10: 100 ticks of occupancy, 10 of transition, a
release delay of 20 and a priced surcharge (charge 10, surcharge 2, so 12 is debited and 2
of it can never come back). A scripted alert fires at tick 40 and is released to the policy
15 ticks later. A level-1 array policy scans the released rows of its snapshot for a row on
the alert channel and, once, emits a live `Cancel` of lane 0. The script, keyed by tick,
starts the same subject again at tick 120 so the next acquisition shows what became of the
cancelled index. The log ends with three rows: the alert, the partial record and the
completed second exposure.

The alert comes from `ScriptedRecords`, extended in the test file by `ScriptedAlerts`
with one `delay` leaf. `ScriptedRecords` releases a record in the tick it is written and has
no way to declare a delay, so the delayed release is a test-local subclass, not a change to
the study module. The alert is addressed to the policy audience alone, so the observer
folds nothing from it.

## What it shows

- **(a) the cancelled generation settles nothing.** The first exposure would finish at tick
  110. The log holds one record for it, `PARTIAL`, and no `COMPLETED` record carries its
  index. The only `DONE` the run handles is the second exposure's, at tick 230, and it
  writes its row; none is handled at tick 110. `double_settle` stays false.
- **(b) the partial fraction.** The exposure started at tick 10 and was cancelled at tick 55:
  45 elapsed of 100 planned. The fraction in the partial record is 0.45 to the go tolerance.
- **(c) charge and credit.** The pricing table gives charge 12 and a fixed part of 2. The
  partial record retains `(12 - 2) * 0.45 + 2 = 6.5`, of which 2 is nonrefundable, and the
  ledger credits `12 - 6.5 = 5.5`. The completed second exposure keeps its whole 12. The
  balance is the budget less 24 debited plus 5.5 refunded. The expected values are read from
  `charge_parts`, not typed in.
- **(d) the index.** The partial record has acquisition index 0, the next exposure of the
  subject has index 1, and the subject's admission count is 2.
- **(e) the latency.** The alert is written at tick 40 and its availability is 15 ticks later,
  the declared `delay`. It is released at tick 55, the policy answers in that tick, and the
  one `CANCEL` event handled lands at tick 55. Alert to cancel is 15 ticks.

Evidence counts, all required nonzero by `test_the_study_reads_and_cancels_on_the_alert`:
1 alert row, 1 cancel effect emitted and 1 admitted, 1 cancel event handled, 1 partial
record, 1 completed record, 1 `DONE` event handled, 1 process event handled, a retained
surcharge of 2 and a credit of 5.5. No refusal is asserted: the row has none to count.

## Source mutants

The checks are plain functions (`check_a` to `check_e`), so a mutant is held to the same
assertions. Each mutant replaces one piece of handler source with `mutate`, clears the JAX
caches and runs the study. Old code: all five checks and the evidence test pass.

| Mutant | Source change | Checks that fail |
|---|---|---|
| a | `_cancel_lane` leaves the pending `DONE` row valid | (a) only |
| b | `_cancel_lane` passes fraction 1 to the record instead of the elapsed fraction | (b) only |
| c | `_cancel_lane` drops the fixed part from the retained charge (`kept` without `+ fixed`) | (c) only |
| d | `_cancel_lane` decrements the subject's acquisition count | (a), (c), (d) and the evidence test |
| e | `_release_row` releases a policy-only row one tick late | (a) to (e) and the evidence test |

Mutants a, b and c are killed by their own check alone. Mutant d lets the second exposure
reuse index 0, so the shared `_split` of the records no longer finds one record per index
and (a) and (c) fail with (d). Mutant e moves the cancel from tick 55 to tick 56, and every
other check pins the script's ticks, so each fails with (e). The mutants are run once by a
scratch script, not kept as tests: five cold recompiles would cost far more than the file
does, and each mutant edits a line a refactor would also touch.

## Finding

The `stale` guard in `h_done` is not the line that stops the cancelled generation's `DONE`.
A live cancel invalidates the lane's pending `DONE` row in the queue (`_cancel_lane`), so the
event never reaches `h_done`. The generation guard and the duplicate guard behind it are
reached by other paths, not by this one. The spec's "settles nothing and writes no record"
holds, by the invalidation. No core change is needed, and the test asserts the outcome
(no handled `DONE` at the planned tick, no record, no settlement) and not which guard held.

`ScriptedRecords` has no release delay. If another survey wants one, a `delay` field on the
study module would replace the subclass here.

## Executors and tiers

Row 11 names no executor. The file runs the host executor once, in a module-scoped store,
and every check reads that run. One compiled run adds a parity check (ints exact, floats in
the go band): it is cheap here, about 6 s of added compile on this machine. All seven tests are medium, with reasons.
The host fixture setup is charged to the first test.

| Measure | Value |
|---|---|
| File, cold, caches off (this machine) | 14.2 s wall |
| Host run alone | 5.8 s cold |
| Compiled run added | 7.5 s cold |
| Compiles in the file | 131 |
| Budget line | `"tests/studies/test_survey_11.py" = 157` |
