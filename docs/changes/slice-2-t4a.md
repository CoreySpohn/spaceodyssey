---
orphan: true
---

# Slice 2 T4a: the core refresh protocol

T4a gives a refit-capable observer a way to change its belief at semantic
checkpoints, in both executors, without folding. A checkpoint is a core event on a
tick grid. It holds the world, a boundary (an array kernel or host Python) answers
for the frozen world, the core installs the answer for each audience and the
decision owed at that tick runs afterwards on the refreshed snapshot. A study with
no refresh schedule is untouched: all 112 golden digests match, and the compiled
program of every Slice 1 study differs only by the new carry leaves, which have zero
size.

## Commits

1. `feat(core): refresh protocol with checkpoints, hold and install, masked and host-loop boundaries`
2. `test(core): refresh checks on both executors, leak scenarios, drain, mutants and field parity`
3. `merge: sync with main` (the legacy deletion; no conflicts)
4. `fix(core): requeue copies rows without dynamic getattr; release handler keeps its fold lines`
   (the dispatch scan refused the first `requeue`; the release handler's fold lines
   are matched by text by two existing mutant tests, so they keep their form)
5. `docs(changes): record Slice 2 T4a refresh protocol`

## What the protocol is

- **Schedule.** `RefreshSchedule(every_ticks, offset_ticks, observer_channel,
  policy_channel, kind)` is a `Study` field. A checkpoint is an `EventKind.REFRESH`
  row (code 10) at `(t_k, Stage.ANALYZE, micro 0, REFRESH_TIE)` with
  `t_k = offset + k * every`, on an unclamped grid. `every_ticks = 0` enqueues
  nothing. `next_grid_tick(schedule, tick, stage)` gives the checkpoint a row accepted
  at `(tick, stage)` falls in: the smallest grid tick at or after `tick`, or the next
  one when the position is already past `ANALYZE` of a grid tick (an `ADMIT`-stage
  result). A grid tick above `max_tick` sets a new overflow bit (`GRID_TICK`).
- **Eligibility and coalescing.** `h_release` marks `Carry.refresh_dirty[audience,
  subject]` for each audience that accepted (observer) or was released (policy) a row
  of a kind in `Capacities.refit_kinds`, and queues one checkpoint when none is
  queued. Later marks coalesce.
- **Hold, answer, complete.** The first handling of the row copies it into
  `Carry.refresh` and puts it back in the queue unchanged (`events.requeue`); a
  decision pending at that tick stays pending behind it and the world takes no step
  while `refresh.valid`. The executor gathers a `RefitRequest` with the jitted
  `gather_refit_rows` (exact, since a frozen log does not change), calls the bound
  boundary and installs through the jitted `commit_refresh`, which also marks the
  queued row installed. The second handling consumes the row as a no-op reporting the
  product rows written, and the ordinary wake logic runs the owed decision at `t_k`
  before any later event. Both handlings are handled steps of kind `REFRESH` and cost
  one step of the budget each, on both executors; waiting costs none.
- **Installation, per audience.** Dirty entries with status `OK` or `DEGRADED` are
  installed (`install` for the observer, `install_summary` of the refit belief's
  `summary` for the policy); nothing is folded. A status that does not install leaves
  the slice unchanged. The status is stored in `Carry.refresh_status[2, n_subjects]`,
  the dirty flag clears, and one derived product row per refit entry is written to the
  audience's product channel; the policy row counts in the policy release version.
  The observer's whole belief is analyzed once after its installs.
- **Deduplication.** Both audiences' canonically masked rows (zero where a row does
  not count; ints and float bits), keys and dirty flags decide whether the policy
  entry copies the observer's computation (`RefitRequest.source`). An observer-only
  column makes the inputs differ, so they are computed twice.
- **Drain and run status.** `REFRESH` is a drainable kind and `~refresh.valid` joins
  the drain predicate. A run with refresh and no boundary stops at its first hold and
  reports `RunStatus.AWAITING`. The compiled executor uses the chunked driver
  whenever there is a `service` or `study.refresh.every_ticks > 0`.
- **Boundaries.** `AbstractRefitBoundary.refit_batch(study, requests)` returns a
  `RefitResult` with axes `[world, 2, n_subjects, ...]`. `MaskedBatchRefit(kernel)`
  runs a fixed `refit_lanes` batch of world lanes through one jitted vmapped call,
  masks entries that are not dirty or are shared, and pads a batch of worlds to whole
  lanes, so the host executor's single world and the compiled executor's worlds run
  one program. Its default kernel is the observer's own `refit`
  (`observer_refit_kernel`). `HostLoopRefit(fit)` is a Python loop over dirty entries
  that is never an exception path: a failure is a `RefitStatus`.

## The six design inputs

1. **Refit results are subject slices.** A refit returns the leaves the observer names
   in `subject_leaves`, taken at the refit subject (`slice_spec`, `take_subject`),
   not a whole belief. A batch over `2 * n_subjects` entries holds that many slices.
   The core rebuilds a full-shaped belief with `put_subject` before it calls
   `install`.
2. **A non-incremental observer folds nothing.** `h_release` skips both folds when the
   observer's capabilities say it is not incremental, and the `Study` refusal of such
   an observer is lifted when the capacities carry the protocol and the schedule
   checkpoints. Such an observer changes only at checkpoints.
3. **`refit_key(root, subject, digest)` has no audience input.** One key per subject
   and revision-set digest serves both audiences, so a deduplicated computation is
   one computation. The digest is of the `(channel, acquisition index, revision)`
   triples of the counting rows in canonical order (`revision_set_digest`). The
   revision-set work itself is T4b; the key and the digest are delivered early
   because the request needs a key.
4. **Refit rows are canonical.** `refit_rows` order is what every refit sees. An
   observer whose refit depends on the order declares `order_sensitive` and is graded
   statistically by the kit; there is no release ordinal in the request.
5. **Policy-side parity.** `LeastSeenPolicy` observes the least-seen of two subjects
   by the policy belief's count, which a refit-only observer changes only at
   checkpoints, so every decision reads what a refresh installed. Both executors
   agree on its histories, and a mutant boundary that shifts the installed mean
   changes them, so the policy does read the install.
6. **The kit under the new install path.** The kit's `refit_beliefs` over the final
   log of a drained refresh run, on both executors, equals the belief the
   checkpoints installed. The kit still keys its own refits with `fold_in(fold_in(
   root, seed), subject)`; moving it to `refit_key` is T3b.

## Deviations from the spec text

- `refit_batch(study, requests)` takes the study, so one boundary instance serves any
  study and cannot be bound to the wrong one.
- `RefitStatus` is an enum (`OK`, `DEGRADED`, `NO_SUPPORT`, `FAILED`, ...), and
  `INSTALLING_STATUSES` names those that install.
- `RefreshSchedule` carries `kind`, the registered index of the derived product kind
  (`REFRESH_PRODUCT_KIND`), so the core never names a study's registry.
- Product rows carry no pedigree column. The `product[n_log, product_width]` column
  and the revision-set read from the log are T4b; here a product row holds its
  status and row count in two public payload columns.
- `refit_key` and the revision-set digest arrive here instead of in T4b (design
  input 3).
- Studies that checkpoint need `Capacities.rows_per_subject` and `refit_lanes`;
  `Study` refuses the combination otherwise, with a message that names both.

## Compiled signatures

New programs: the refresh toy study on each executor (the chunked driver, with the
`REFRESH` branch in the handler switch and the held-world guard in the scan) and the
toy masked kernel (`MaskedBatchRefit`), plus the jitted `gather_refit_rows` and
`commit_refresh` (each once per study structure and once vmapped over worlds).
Every Slice 1 program recompiles once per session for the new zero-size leaves and
the extra switch branch.

Measured row (`two worlds, 60 steps, chunk 7`, four threads through the compute
launcher, a shared machine, first call includes compilation):

| Program | First call (s) | Second call (s) |
|---|---|---|
| refresh toy study, compiled executor | 11.2 | 0.10 |
| refresh toy study, host executor | 8.7 | 0.31 |
| same study without refresh, compiled (reference) | 7.9 | 0.01 |
| same study without refresh, host (reference) | 4.9 | 0.26 |
| masked kernel alone, two worlds (after a cache clear) | 0.1 | 0.001 |

The refresh study adds about 3 s of cold compile on the compiled executor and about 4
s on the host executor (the gather and commit programs and the masked call, which the
host executor runs through lane 0). The kernel itself is small; a persistent compile
cache made its recompile 0.1 s.

## Tests

`tests/core/test_refresh.py`, 11 quick tests (0.3 s together in the quick tier, pure checks with no run):
`next_grid_tick` before the offset, on the grid and after `ADMIT`, between grid
ticks; `refit_key` has no audience input and moves with the subject and the digest;
subject slices round trip; a study without refresh has zero-size refresh leaves and
one with it does not; the structural refusals (no capacity, equal product channels,
an unregistered or non-derived product kind, negative ticks, an incremental observer
without subject leaves).

`tests/core/test_refresh_runs.py`, 49 slow tests (compile of the chunked
driver and the masked kernel per executor exceeds the unit budget; the file is in
`SLOW_FILES` with that reason):

| Test | Cases | Fails if |
|---|---|---|
| `test_host_and_compiled_histories_equal_at_every_chunk` | chunks 1, 7, 32 | the executors disagree on histories or step counts, in a study with refresh and no `Query` service |
| `test_refresh_ran_and_installed_something` | 1 | no checkpoint is handled or no belief moves |
| `test_host_loop_matches_masked_batch` | 1 | the host-loop and masked-kernel boundaries disagree |
| `test_awaiting_without_a_boundary_on_both_executors` | 1 | a run with refresh and no boundary does not report `AWAITING` |
| `test_no_refresh_study_ignores_a_boundary` | 1 | a boundary changes a study that does not checkpoint |
| `test_batched_worlds_equal_single_runs` | 1 | heterogeneous batched worlds differ from their single runs |
| `test_corrupted_install_changes_the_history` | 1 | an installed mean shifted by one leaves the history unchanged |
| `test_failed_status_installs_nothing_for_that_audience` | 1 | a failed observer entry installs a belief |
| `test_dedup_computes_once_when_rows_match` | 1 | equal masked inputs are computed twice |
| `test_refresh_leaf_mutation_fails_parity` | one per conditional refresh leaf | a compiled-only change to `carry.refresh.*`, `refresh_dirty` or `refresh_status` is not caught |
| `test_private_events_leave_wakes_views_and_product_rows_unchanged` | heartbeat, private_writer | a private event changes policy wakes, views or product rows |
| `test_release_at_a_checkpoint_then_query_owes_one_decision_after_refresh` | alone, with a coincident process row | the owed decision runs twice, before the install, or on the pre-refresh snapshot |
| `test_request_and_refresh_never_wait_together` | 1 | a row is handled between a `Query` and its answer, or both holds are set at the end |
| `test_oracle_refresh_parity_across_executors_and_chunks` | chunks 1, 7 | the executors disagree with a `Query` service and a boundary together |
| `test_release_past_the_horizon_schedules_a_later_checkpoint_and_completes` | 1 | a release after the horizon is lost, the grid is clamped, or the run does not complete |
| `test_admit_stage_result_at_a_checkpoint_tick_falls_in_the_next_checkpoint` | 1 (with its control) | an `ADMIT`-stage result is installed at the checkpoint that already ran |
| `test_checkpoints_do_not_wake_the_policy` | 1 | a checkpoint wakes the policy where nothing was released |
| `test_private_events_do_not_move_checkpoints` and `test_a_transition_count_schedule_is_caught` | 1, 1 | a log-length schedule (the mutant) is not caught by the tick comparison |
| `test_belief_reading_policy_agrees_across_executors_and_reads_the_install` | 1 | a belief-reading policy differs across executors, or ignores the install |
| `test_order_sensitive_observer_agrees_across_executors` | 1 | the order-sensitive sampling observer differs across executors |
| `test_observer_only_column_is_computed_twice_and_public_rows_once` | 1 | an observer-only column is deduplicated, or public rows are not |

`tests/core/test_canonical_fields.py`: the 23 conditional refresh leaves are removed
from the Slice 1 pair (their leaves are zero-size there, so a perturbation has
nothing to change) and are covered by the refresh-study mutation parity above.

Tiers: 11 quick tests added (11), 49 slow tests added.

Retired: none. The spec's list is covered except for the cases that need T4b or T4c
(revision sets, same-tick corrections around a checkpoint, keyed replay).

## Time

Quick tier (`-m "not slow" -n 2 --dist loadfile`, local, four threads, junit wall time),
alternating the merged main (`551530c`) and the branch on a shared machine with other
compute jobs running:

| Run | Tests | Wall (s) | Summed (s) |
|---|---|---|---|
| main | 1,556 | 110.9 | 199.4 |
| branch | 1,567 | 65.8 | 107.8 |
| main | 1,556 | 57.0 | 97.0 |
| branch | 1,567 | 59.0 | 100.0 |

The first main run overlapped another job; the quiet pair is 57.0 s against 59.0 s.
The quick tier gains 11 tests that take 0.3 s and is flat within the machine's noise.

Full suite (`-n 2 --dist loadfile`, through the compute launcher, the installed
entry-point test deselected in the worktree, on the merged tree): 2,695 passed in
1,499 s wall under heavy shared load, including the two slow kit files once
(`test_observer_kit_runs.py` 12 tests, 12.5 s summed; `test_observer_kit_statistics.py`
20 tests, 23.0 s summed) and the golden manifest. The three budget reports in that run
are existing tests (`test_core.py::test_gain_enters_measurement`,
`test_core.py::test_every_hand_study_reproduces_its_history` and
`test_record_statuses.py::test_statuses_agree_between_executors_and_replay`, 5.4 s,
15.1 s and 6.9 s on cold compiles under load); none is a T4a test.

The new slow file is 49 tests and 212 s summed in that run (the slowest four are the
`Query`-plus-refresh parity, the order-sensitive observer, the past-the-horizon drain
and the `ADMIT`-stage result, 17 to 22 s each under load, because each compiles the
chunked driver and the masked kernel for a different study structure). An earlier
full run before the merge had 7 failures: the dispatch scan on the first `requeue`
(2 tests), four mutant tests and one kit control that match the release handler's
fold lines by text; commit 4 fixes all seven.

## Golden status

All 112 golden digests match (`tests/core/test_golden_histories.py`, part of the full
suite). The compatibility run before the merge failed 23 tests of
`test_canonical_fields.py` ("unexercised field" on the zero-size refresh leaves of
the Slice 1 pair); they are now covered as above.

## Deferred

- T4b: the pedigree column `product[n_log, product_width]`, revision sets read from
  the log, derived-kind exclusion from expected folds and refit rows, replacement of a
  corrected row, semantic indices for scripted and `Query` products, reserved
  internal ids per product row, and the K002 disjointness of admission, scripted,
  service and product channels. Until then the product rows' identity is
  `(product channel, subject, checkpoint ordinal)` and their internal id is
  `next_sid` at the checkpoint, which `commit_refresh` advances once per row.
- T4c: keyed replay of refreshes.
- T4d: the OFTI host-loop refit in the photomancy adapter (`HostLoopRefit` is the
  seam and is exercised on the toy).
- T3b: kit extras (a delayed correction, a duplicate through a summary) and moving
  the kit's refit key to `refit_key`.
- Semantic indices (T4b) change the keys of repeated refreshes and answers; none of
  them is a Slice 1 history.
