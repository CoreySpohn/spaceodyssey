---
orphan: true
---

# Slice 3a survey 9: interleaved programs

Survey 9 adds one test file, `tests/studies/test_survey_9.py`, and changes no core file, no
golden and no pin. All studies are scripted (`ScriptedPrograms`, at most eight rows), so the
decision cadence is part of the evidence: a public release and a wait each wake one decision.

## What it shows

- **(b) independent balances.** Programs 0 and 1 alternate on channel 10. Before program 0 is
  stopped each is admitted twice; afterwards program 1 is admitted once more and program 0 is
  refused `LANE_CLOSED` once. The program rows end at 98 and 97 on the program-scoped resource.
- **(d) conservation per row.** The retained records hold five charges that, with the budget, give the
  final balances (a records-and-budget reconciliation local to the test), and the conformance
  invariant `balance_conserved` passes on the same run.
- **(c) the credit-free program.** Channel 13 shares lane 2 with channel 12 and costs nothing;
  program 1 holds no credit. It starts on an idle lane, is refused `LANE_BUSY` once, and the
  funded start ticks equal those of a run without it.
- **(a) one program equals `survey_1`.** The `survey_1` plan written as a script, with one
  program, makes the same history: ints exact, floats within the go band. Two leaves differ and
  are set aside by name because the policy itself owns them (`carry.pstate`, the policy state,
  and `trace.pol_u`, its draws); the test asserts they differ and that every other float leaf
  is compared.
- **Parity.** (b) and (c) make the same history on the host and compiled executors.

## Controls

- One pool: with every request in program 0, program 0 alone pays for its four admissions
  before the stop, the program-1 row stays at 100 and its later requests are refused.
- A credit-free start placed across the funded request leaves that request refused and moves
  its start tick; a priced channel 13 is refused `NO_FUNDS`, so credit-freeness was the reason
  the program ran.

## Tiers

The merge-time guard is (b) and (d): one study, one host run, medium (7 s cold). Everything
else is slow: the controls, (a), (c) and the two-executor parity.

## Notes for review

- Survey 9 does not run controls on both executors; the spec asks for that only where a
  survey names it.
- The test-local reconciliation in the guard reads the records and the budget rather than the
  ledger counters, so it stays independent of `balance_conserved`.
