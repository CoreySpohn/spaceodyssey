# Slice 1, T4: transaction semantics

## Behaviour changes

- Wakes coalesce per tick. An event that wakes the policy marks `Carry.pending_wake`; the
  decision fires once, on the event after which no queued row remains at or before the
  current tick at a stage ahead of `Stage.DECIDE` (`decision_due`). Rows that a decision
  enqueues (stage ADMIT) run after it and may wake a further decision at the next
  microstep, bounded by the microstep guard. The peek reads the whole queue, private rows
  included, so a private row at the same tick delays the decision to the last row of the
  tick instead of firing ahead of it and sending the position backwards. The decision's
  view is unchanged by private rows, and the leak scenarios still show zero extra wakes.
- `ChannelTable.coincident_order` is a required column (`CoincidentOrder.SETTLE_FIRST`,
  `FAULT_FIRST`). `EventTable` gains a `tie` column and the packed order key becomes
  `(tick, stage, micro, tie, sid)`: a fault row ranks below the neutral tie when its
  channel orders faults first and above it otherwise (`fault_tie`); every other row is
  neutral, so the tie decides only between a fault and its coincident rows, after stage
  and microstep and before the stable id. `EventKind.FAULT` and `h_fault` give the rank
  something to order: a fault fails the lane's live occupancy (record status `FAILED`,
  charged in full, no refund, generation retired) and does nothing if the settlement
  already ran. There is still no failure process; tests enqueue the row directly.
- Stop dispositions: `Ledger.lane_draining` is gone. A drain closes exactly as a close
  does below campaign scope (`test_stop_equivalence_below_campaign`); at campaign scope it
  still ends the run when the running occupancies settle. The pre-flight warning for a
  study that asks DRAIN where it equals CLOSE is left to the pre-flight task.
- `Agenda.permitted_effects` holds one bitmask over `EffectKind` per phase (`effect_mask`,
  `ALL_EFFECTS`). Admission refuses an effect whose bit is clear with
  `AdmissionCode.FORBIDDEN` and changes nothing. The toy builder permits every effect.
- Group admission is all or nothing. Observe rows sharing a `group` id are judged
  together on summed demand per balance cell, lane availability (two rows on one lane
  cannot both run) and queue room for every row. A refused group takes no acquisition
  index, every row of it carries the code of the first refused row, and groups with
  different ids are still admitted one after another against the balance the earlier
  ones left. The members of an admitted group are queued consecutively at its first row.
- `Allocate` stays declared and admission answers `AdmissionCode.UNSUPPORTED` (as does
  `Query`); the pre-flight check that rejects a study using it is left to the pre-flight
  task.
- `AdmissionCode` gains `OUTSIDE_VALIDITY`, `FROZEN_HORIZON` and `FORBIDDEN`. The first
  two are declared and unreachable.

## Reserved fields (no behaviour)

Present in the schema, the carry, `canonical_history` and both executors, defaults
round-trip unchanged: `ChannelTable.charge` (equals `cost`), `AssetTable.freeze_ticks`
(0), `AssetTable.disruptive_resource` (-1), `Ledger.program_charge_rule`
(`ChargeRule.RECEIPT`), `Ledger.valid_from_tick` (0), `Ledger.valid_until_tick` (the
largest tick), `Ledger.remainder_rule` (`RemainderRule.NONE`), `Ledger.disruptive_used`
(0), `Record.charged` (equals `cost`), `Record.refund_of` (-1), `EffectTable.disruptive`
(false, also recorded in `Trace.eff_disruptive`). Setting a non-default value changes no
history (`test_reserved_table_field_has_no_behaviour`); failing pre-flight on a
non-default value is left to the pre-flight task.

## Acquisition index

`Carry.acquisition_counts` already existed. It now holds under group admission (a refused
group takes no index) and is covered by the six tests of `test_acquisition_index.py`,
including a property test over random effect sequences and a decrement mutant that lives
in the test tree.

## Rebaselined cases

No corpus hand history had two wakes at one tick, so none changed: all 22 cases keep
their events, rows, admission codes and decision counts exactly (compared case by case
before and after the change). One case was retimed on purpose:

| Case | Old | New | Reason |
|---|---|---|---|
| `bundle` | Channel 12 occupies its lane 50 ticks. The bundle releases at 185 and 260, the decision at 185 meets the busy lane, and the plan entry is spent on a refusal. | Channel 12 occupies its lane 125 ticks, so the bundle's two records release together at 260 and one decision sees both. Hand history and rows added (14 handled rows, 6 records, no refusal). | The old extra wake sits at a different tick from the other release, so no coalescing rule can remove it. The case now exercises coalescing; the old timing stays as `test_wakes_at_distinct_ticks_stay_separate`. |

Test expectations that changed:

| Test | Old | New | Reason |
|---|---|---|---|
| `test_releasing_process_wakes_once_and_is_coincident` | one extra wake per world | renamed `test_releasing_process_coalesces_with_the_release_it_coincides_with`: no extra wake | the process event and a record release share tick 130 and now give one decision |
| `test_admission_is_cumulative_within_a_decision` | two observations of one lane, same group: second `LANE_BUSY`, first admitted | renamed `test_admission_is_cumulative_across_groups_of_a_decision`: the rows get different groups | rows of one group are now judged together; same-group behaviour is `test_group_on_one_lane_is_refused_whole` |
| `order_key_np` (test twin of `order_key`) | `(tick, stage, micro, sid)` | adds the tie rank | the packed key has a tie field |

## Deviations from the spec

- The `bundle` case is retimed rather than only re-baselined (see above): the spec
  expects coalescing to stop the plan entry being spent, but that wake is at a distinct
  tick, so it cannot.
- Group atomicity covers Observe rows only. Wait, Stop and Cancel rows of the same group
  id keep being admitted independently; they carry no demand.
- The decision's wake fires on the last row of the tick even when that row is private
  (see the first behaviour change); the spec's "peek must not move a wake" holds in
  count and in what the policy sees, not in which handled row carries the wake flag.
