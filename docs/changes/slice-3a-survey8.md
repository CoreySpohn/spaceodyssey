---
orphan: true
---

# Slice 3a survey 8: a second suppression system

Survey 8 adds one test file, `tests/studies/test_survey_8.py`, and changes no core file, no
golden and no pin. Two assets each own their lanes: asset 0 owns lanes 0 and 1, asset 1 owns
lane 2. Fuel is resource 0, held per asset; every channel costs one unit of it and nothing
else, so the program and shared rows never move. Asset 0 starts with 2 units, asset 1 with 10,
and balances are shown to the policy.

The script is keyed by tick, not by decision count, so a different number of releases in two
runs cannot move it, and each scripted tick is reached by a wait. The policy is a level-1
array policy: it reads the balance table, finds the fuel column of asset 0, and while that
balance is spent it emits one `Stop(ASSET)`, remembered in its state so it fires once.

## What it shows

- **(a) the policy stops the dry asset.** Asset 0 takes its last fuel at tick 120. The policy
  sees the empty balance at the next decision and emits the asset Stop at tick 130. The run
  holds exactly one Stop before the final campaign drain, and its tick is after the exhaustion
  tick.
- **(b) the refusal code changes across the Stop.** At tick 120 lane 1 of asset 0 asks for fuel
  and is refused `NO_FUNDS`; that is the only request refused before the Stop. At tick 230 both
  lanes of asset 0 ask again and are both refused `LANE_CLOSED`, although lane 0 is idle by
  then, so the Stop closed the asset and not one lane. Six of the nine requests are admitted.
- **(c) the other asset carries on.** Asset 1 is admitted at ticks 0, 120, 140 and 230, two of
  them after the Stop. Its records equal those of the run in which asset 0 has ten units and
  never runs dry, column for column, except the acquisition id, which is the queue's internal
  number and moves with asset 0's events. Every balance row except asset 0's is equal as well
  (asset 1 ends at 6); asset 0 ends at 0 against 6, so the two runs did differ.
- **(d) a scripted fault on the other asset.** A fault at tick 140 on lane 2 fails the work
  admitted at tick 120: one `FAILED` record on lane 2 ending at tick 140, charged its full unit
  of fuel. Asset 0's two records, including the acquisition under way when the fault fires,
  complete. Lane 2 is admitted again at tick 140, so the fault cancelled the work and did not
  close the lane. That is the behavior of `FAULT` today and the study does not turn on lane
  closure on a fault.
- **Parity.** Both executors make the same history; the host run is the one inspected.

## Control

The same study with the policy's Stop removed emits no Stop before the campaign drain and
refuses every request on asset 0 after the fuel is gone `NO_FUNDS`: three refusals and no
`LANE_CLOSED`. The assertion function that checks (a) and (b) raises on that run.

## Tiers

The merge-time guard covers (a) to (d) and parity: one study on both executors plus one more
host run for the reference, medium (about 22 s cold, nearly all of it compiling the two
executors). The control is slow; it reuses the same compiled shapes and takes under a second
once they are built, but it needs them built when run alone.

## Notes for review

- The reference run differs from the study only in the fuel budget, a leaf, so it recompiles
  nothing. The control differs only in a boolean leaf of the policy.
- The policy and the tick-keyed script live in the test file, as survey 9's helpers do.
- At the end of the run the campaign drain closes every lane, so the closed flags at the end
  say nothing about the Stop; the refusal codes at tick 230 are the evidence.
