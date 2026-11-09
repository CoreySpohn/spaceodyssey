# Slice 1, T5: executors

## Behaviour changes

- Run status. `RunStatus` (`COMPLETE`, `IDLE`, `TRUNCATED`, `STALLED`, `OVERFLOW`) is
  derived per world from the final carry by `run_status`; `RunResult.status` and
  `RunResult.complete` expose it and `RunHistory` carries `status`, `overflow_mask` and a
  `complete` property. Only `COMPLETE` means the run reached its own end. An overflow
  outranks a stall, a stall outranks a run that used up its step budget.
- Overflow names what ran out. `Carry.overflow_mask` holds `Capacity` bits: `QUEUE_SLOTS`
  (no free queue row), `STABLE_IDS` (`max_sid` exhausted) and `LOG_ROWS` (record log
  full). The microstep guard keeps its own flag and reports as `STALLED`. The `overflow`
  flag itself is unchanged.
- One run contract. `AbstractExecutor.run_worlds(study, truths, roots, n_steps)` returns a
  `RunResult` with a leading world axis on both executors (`HostExecutor` loops over
  worlds and stacks; `CompiledExecutor.run_worlds` is `run`). `RunResult.history(world)`
  gives the canonical history of one world. The existing `run` signatures are unchanged.
- `Trace` gains `tie` (the handled row's tie rank) and `pending_wake` (the carry's pending
  wake after the step). `canonical_history` records both, the final `pending_wake`,
  `overflow_mask` and `status`, next to the reserved ledger fields, admission codes and
  acquisition counts it already carried.
- `ParityReport.go` (`max_abs <= GO_TOLERANCE`, 1e-12) sits beside `ok` (kill tolerance
  1e-9, `KILL_TOLERANCE`). `ok` is unchanged.
- `replay_belief(study, carry, trace)` rebuilds the observer belief of a world from its
  recorded log: each row is folded once, in the order of the handled release steps, through
  the study's observer.

## Carry changes (shared with the record task)

- Added `Carry.overflow_mask` (int64, zero at start).
- No field was removed or retyped. `Trace` gained the two fields above.

## Tests

- `test_run_status.py`: `test_overflow_sets_status` (stable ids, log rows, queue slots, on
  both executors, mask checked), `test_truncated_run_not_reported_complete`, the microstep
  guard case, executor agreement on status, and status priority.
- `test_parity_corpus.py`: every corpus case, both executors and both policy forms, ints
  exact and floats in the go band.
- `test_replay_log.py`: `test_replay_reproduces_belief_bitwise` over the corpus cases that
  do not overflow, and a check that the replay reads the log.

## Rebaselined cases

None. No hand history, row or flag changed.

## Deviations from the spec

- Overflow of the queue is not reachable through the toy policies (admission refuses a row
  that would not fit), so its executor test seeds more process rows than the queue holds.
- `STALLED` stays a separate status rather than an overflow bit; the stall case in the
  corpus asserts the `overflow` flag is clear, and keeping that avoids re-baselining it.
- The host executor runs one world per call; `run_worlds` is a loop, not a batched run.

## Left for later tasks

- Pre-flight checks that reject a study whose capacities are too small.
- Executor-level host service (compiled run yielding to the host) is not part of this task.
