---
orphan: true
---

# Slice 3a survey 8: a second suppression system

Survey 8 adds one test file, `tests/studies/test_survey_8.py`, and changes no core file, no
golden and no pin. Two assets each own their lanes: asset 0 owns lanes 0 and 1, asset 1 owns
lane 2. Fuel is resource 0, held per asset. Channels 10 and 11 (asset 0) cost one unit of it and
channel 12 (asset 1) two, and nothing else is charged, so the program and shared rows never
move and a debit booked to the wrong asset shows in the balances. Asset 0 starts with 2 units,
asset 1 with 10, and balances are shown to the policy. The agenda is at level 1, and the test
asserts it.

The script is keyed by tick, not by decision count, so a different number of releases in two
runs cannot move it, and each scripted tick is reached by a wait. It does not end with a campaign
drain, so the closed flags at the end of the run belong to the Stop alone. The policy is a level-1
array policy: it reads the balance table, finds the fuel column of asset 0, and while that
balance is spent it emits one `Stop(ASSET)`, remembered in its state so it fires once.

## What it shows

- **(a) the policy stops the dry asset.** Asset 0 takes its last fuel at tick 120. The policy
  sees the empty balance at the next decision and emits the asset Stop at tick 130. The run
  holds exactly one Stop effect, it is admitted, and its tick is after the exhaustion tick. At
  the end exactly lanes 0 and 1 are closed and lane 2 is not, so the Stop had asset scope and
  named the dry asset; two lane Stops or a Stop of another asset would leave a different set
  closed or a different effect count.
- **(b) the refusal code changes across the Stop.** At tick 120 lane 1 of asset 0 asks for fuel
  and is refused `NO_FUNDS`; that is the only request refused before the Stop. At tick 230 both
  lanes of asset 0 ask again and are both refused `LANE_CLOSED`, although lane 0 is idle by
  then. Six of the nine requests are admitted.
- **Per-asset fuel.** Each fuel row equals its budget less the charges of the records on its
  own lanes: asset 0 paid 1 for each of two records and ends at 0, asset 1 paid 2 for each of
  four and ends at 2. Because the two assets pay different amounts at tick 0, a debit booked to
  the other asset there leaves a gap.
- **(c) the other asset carries on.** Asset 1 is admitted at ticks 0, 120, 140 and 230, two of
  them after the Stop. Its records equal those of the run in which asset 0 has ten units and
  never runs dry, column for column, except the acquisition id, which is the queue's internal
  number and moves with asset 0's events. Every balance row except asset 0's is equal as well;
  asset 0 ends at 0 against 6, so the two runs did differ.
- **(d) a scripted fault on the other asset.** A fault at tick 140 on lane 2 fails the work
  admitted at tick 120: one `FAILED` record on lane 2 ending at tick 140, charged its full two
  units of fuel. Asset 0's two records, including the acquisition under way when the fault
  fires, complete. Lane 2 is admitted again at ticks 140 and 230, so the fault cancelled the
  work and did not close the lane. In the run with fuel, where no Stop closes anything, no
  lane is closed at the end and lane 1 is still refused `LANE_BUSY` at tick 230, so the fault
  left asset 0's lanes untouched. That is the behavior of `FAULT` today and the study does not
  turn on lane closure on a fault.
- **Parity.** Both executors make the same history; the host run is the one inspected.

## Control

The same study with the policy's Stop removed emits no Stop, closes no lane, and refuses
every request on asset 0 after the fuel is gone `NO_FUNDS`: three refusals and no
`LANE_CLOSED`. The assertion function that checks (a) and (b) raises on that run.

## Tiers

The merge-time guard covers (a) to (d): the study and its reference run on the host
executor only, medium (about 10 s cold on a lightly loaded machine, almost all of it compiling
the host programs, the same cost as survey 9's guard). The reference run recompiles nothing.
The control and the two-executor parity are slow: the control takes under a second once the
host programs are built, and parity compiles the compiled executor again.

## Notes

- The reference run differs from the study only in the fuel budget, a leaf, so it recompiles
  nothing. The control differs only in a boolean leaf of the policy.
- The policy and the tick-keyed script live in the test file, as survey 9's helpers do.
