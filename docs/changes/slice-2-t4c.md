---
orphan: true
---

# Slice 2 T4c: keyed replay with refresh

T4c lets a replay rebuild what a refreshing run did, intermediates included. It
walks the run's log and trace, refits at each checkpoint through the same refit
boundary with the same keys, installs, and checks every product row, the trace
projection of each checkpoint and the final beliefs. Nothing in the run changed: no
golden history moved.

## Commits

1. `feat(core): keyed replay with refresh and roots on run results`
2. `test(core): replay with refresh on both executors, corrections, queries, truncation and mutants`
3. `docs(changes): record Slice 2 T4c keyed replay`

## What changed

- **`replay_run(run, roots, carry, trace, *, refit=None, executor=None)`** in
  `core/executors/replay.py`, exported from `spaceodyssey.core` with its result type
  `Replayed(belief, pbelief, checkpoints, products)`. `run` is a loaded run (its
  `study` and its `refit` boundary, or the boundary of the named executor) or a study
  with `refit=` given. The name differs from the spec's `replay_beliefs` because the
  observer kit already exports a different `replay_beliefs(study, run, roots)`.
- **The walk.** Handled `RELEASE`, `RESULT` and `REFRESH` steps in trace order.
  Release steps keep, per audience, the slots received so far, the acquisitions
  received and the dirty flags, exactly as the release handler marks them. The final
  `superseded_by` is never read: at a hold step the log is presented as the run held
  it (rows written so far, each audience's received rows only) and the run's own
  `gather_refit_rows` rebuilds the revision set and the keys from it.
- **At a checkpoint.** The hold step refits through the boundary and installs through
  the run's own `commit_refresh` on that rebuilt world. The completion step verifies the
  rows written (every column, payload and pedigree), the written count and log length
  against the trace, and the trace projection of the belief, then adopts the install. A
  hold with no completion (a run cut by its budget) installs nothing.
- **Incremental observers** fold the first accepted revision between checkpoints
  (`fold` for the observer, `fold_summary` for the policy, with the replaced revision the
  policy fold saw); derived rows are never folded.
- **Final check.** The replayed `belief` and `pbelief` leaves equal the run's: ints
  exactly, floats within the kill tolerance. Any mismatch raises `ValueError` naming it.
- **`RunResult.roots`** holds the world root keys of the run (both executors).
- **Seeded fields.** The trace holds neither the internal id of a product row nor the
  phase, so the replay reads those two (and the vintage) from the run's row and checks
  every other column.

## Tests

`tests/core/test_replay_refresh.py` (slow tier, in `SLOW_FILES`: every test runs a
refresh study on an executor, so compile exceeds the 5 s unit budget), 13 cases:

| Test | Cases | Fails if |
|---|---|---|
| `test_replay_equals_the_run_on_the_toy_refit` | host, compiled | a replay misses a checkpoint, a product row, the trace projection or the final belief |
| `test_replay_with_same_tick_corrections_around_a_checkpoint` | host, compiled | a correction at the checkpoint tick or the tick after reaches the wrong revision set |
| `test_replay_with_two_questions_on_one_channel` | 1 | two answers on one service channel collide in identity, pedigree or revision set |
| `test_replay_of_a_truncated_run_keeps_the_stale_belief` | cut on a hold, after a completion, mid-run, on both executors | a hold with no completion installs, or a truncated final belief differs |
| `test_a_replayer_that_reads_the_final_superseded_by_is_caught` | 1 (with control) | a replayer hiding rows the final log marks superseded still matches |
| `test_a_replayer_that_keys_a_refit_by_a_count_is_caught` | 1 (with control) | a replay keyed from the log length still matches a key-reading refit |

`tests/core/test_replay_log.py` gains `test_refresh_cases_replay_with_their_checkpoints`,
which replays the corpus `refresh` case (the case the older replay tests skip). The
conformance invariants take a canonical history only, with no boundary, so the corpus
check lives with the replay tests rather than as an invariant.

The spec planned 5 quick tests. All of these need a run on an executor, so none fits the
quick tier; two same-tick corrections are interpreted as corrections released at the
checkpoint tick (before its stage) and at the following tick.

Remaining case, not covered: a correction released at a stage after the checkpoint on
the checkpoint's own tick (an `ADMIT`-stage `Query` result), which falls in the next
checkpoint; T4a covers its scheduling but not its replay.

## Timing

New file alone (four threads, shared machine, cold): 13 cases, 84 s summed; the heaviest
cases are the first run of each study structure (8 to 22 s, compile), the rest under 1 s
(the second run of a structure and every replay: 0.05 to 0.4 s). Replay itself adds
about 0.4 s cold and 0.05 s warm per world.

Full suite (`-n 2 --dist loadfile`, four threads, `tests/codec/test_entry_points_installed.py`
deselected): 3081 passed in 1956 s, on a machine busy with other jobs (about 15 min was
expected on an idle one). Lints: ruff check and format, the dispatch lints in
`test_lints.py`, internal references.

## Golden moves

None. `tests/core/golden_histories.json` is unchanged and `test_golden_histories.py`
passes.
