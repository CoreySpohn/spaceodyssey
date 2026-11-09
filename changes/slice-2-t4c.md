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
4. Review fixes: `fix(core): replay folds, ids, phase and refusals`,
   `test(core): quick-tier replay and refusal regressions`, and a docs update.

## What changed

- **`replay_run(run, roots, carry, trace, *, refit=None, executor=None)`** in
  `core/executors/replay.py`, exported from `spaceodyssey.core` with its result type
  `Replayed(belief, pbelief, checkpoints, products, refused, unfinished, overflow_mask)`. `run` is a loaded run (its
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
- **Fold-time rows.** An incremental fold receives the row as the handler saw it:
  `accepted_tick`, `released_tick` and `superseded_by` are rebuilt from the preceding
  trace steps (unset until the release stamps them; the policy fold sees the
  `superseded_by` of the higher revisions released so far).
- **Phase and ids.** The phase is rebuilt from the woken steps and the run's phase
  bounds, so a product row's `phase` and `vintage` are checked, not read. The reserved
  `acquisition_id` of a product row is checked to be above every earlier row id, to
  rise consecutively through the checkpoint and to be held by no other row. The first
  id is still seeded from the run's row: the trace carries no id-counter state
  (`next_sid`). A per-step `next_sid` in the trace would make replay fully independent
  of the run's rows; it is not added here and needs a decision.
- **Refusals.** A checkpoint refused for rows, ids or log room replays as a refusal:
  `Replayed.refused` lists `(ordinal, capacity bits)`, `unfinished` is true only for a
  hold with no completion, and `overflow_mask` is the run's. A refusal the replay
  reaches that the run did not raise is an error. A derived-row correction refused with
  `INVALID_CORRECTION` replays with neither field set and the bit in `overflow_mask`.
  The replay seeds a refused checkpoint's id counter with the run's final `next_sid`.

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

Three further slow cases: incremental folds that read the three lifecycle fields (host
and compiled; a replay that folds the final row fails), and an `INVALID_CORRECTION`
replay reported distinctly from a refusal or a cut.

`tests/core/test_replay_refresh_quick.py` (quick tier, about 2 s cold, no executor run)
replays a hand-built log: one subject, one row per subject, four log rows, revision 0
released, a checkpoint with two product rows from a boundary that writes the request key
into the mean, then revision 1 released at the checkpoint tick's `ADMIT` stage. Cases:
the control; both mutants (final `superseded_by`, a key from the log length); corrupted
product id, duplicated id, phase and vintage; refusals for exhausted ids and a full log;
a run cut on a hold.

No corpus case that reads its key was added: the corpus histories are goldens, and a
key-reading boundary needs new hand rows, which would move them. The key is covered by
the quick key-reading boundary and the slow `ResampleSample` mutant.

## Timing

New file alone (four threads, shared machine, cold): 13 cases, 84 s summed; the heaviest
cases are the first run of each study structure (8 to 22 s, compile), the rest under 1 s
(the second run of a structure and every replay: 0.05 to 0.4 s). Replay itself adds
about 0.4 s cold and 0.05 s warm per world.

Full suite after the review fixes: see the report. Earlier full suite (`-n 2 --dist loadfile`, four threads, `tests/codec/test_entry_points_installed.py`
deselected): 3081 passed in 1956 s, on a machine busy with other jobs (about 15 min was
expected on an idle one). Lints: ruff check and format, the dispatch lints in
`test_lints.py`, internal references.

## Golden moves

None. `tests/core/golden_histories.json` is unchanged and `test_golden_histories.py`
passes.
