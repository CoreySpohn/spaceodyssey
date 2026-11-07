---
orphan: true
---

# Slice 2b R4a: the epoch relation with negative times

R4a extends the epoch relation (D7) so it holds for data observed before the run
starts. Both comparators, the log comparator in `core/battery/metamorphic.py` and the
decision-view comparator in `core/battery/views.py`, now know tick columns by name
instead of by sign, and `shift_epoch` runs on either executor. B6 passes on a
synthetic study with negative observation times and a relative window. The four
survey studies are checked by the tasks that build them (R4b, R5).

## Commits

1. `fix(core): epoch comparators know tick columns by name, not by sign` (`3edbf15`)
2. `feat(core): shift_epoch runs on either executor and refuses a shift that is not positive`
   (`f55e643`)
3. this record

## What changed

- `core/battery/views.py`: `TICKS` (start, end, available, released and accepted
  tick), `STAMP_TICKS` (released and accepted tick) and `moved_ticks(name, column,
  valid, shift)`. A tick column moves on every valid row, whatever its sign; only a
  release or acceptance stamp still at -1 (unset) stays. A column outside `TICKS`
  never moves, so payloads, windows included, stay under exact comparison.
  `compare_views` uses it for the epoch and the release delay alike.
- `core/battery/metamorphic.py`: the log comparator in `_epoch_world` moves the
  shifted run's tick columns back with `moved_ticks(name, b, valid, -delta)`.
  `TICK_LOG_FIELDS` is gone (`views.TICKS` replaces it). `shift_epoch` gains
  `executor=` (the compiled executor by default) and runs both runs through
  `run_worlds`: a battery process wrapper, `_StartedLater`, seeds as the study's
  process does and then moves every queued event and the clock by `delta`. Both runs
  are wrapped, the base one with `delta = 0`, and `delta` is an array, so the two
  runs share one compiled program per executor. The bare scan `_scan_from` is
  removed. A shift that is not positive is refused (`ValueError`). The result's
  detail adds `n_before_start` (valid rows with a negative start or end tick in the
  base run) and `n_views` (decision views compared), so a vacuous pass shows.
- `tests/core/test_visible_relations.py`: the epoch case of
  `test_relation_rejects_changed_policy_view` patched the module's `transition`, which
  the relation no longer calls; it now passes the same view-changing executor the
  other two relations take (`executor=HiddenRows()`), so it still fails for a changed
  policy view.

The old rule shifted a tick where it was nonnegative. With negative times it fails
both ways: the view comparator left a base time of -12 in place and compared it with
the shifted -7, and the log comparator left a shifted time that was still negative in
place. Taking -1 as unset in every column was the other half of that rule: an end
tick at -1 is a time.

## The synthetic fixture

`tests/core/battery_helpers.py`, section "a study with times before the run starts"
(`epoch_table`, `EpochImporter`, `WindowAgePolicy`, `epoch_study`):
the toy survey 1 with a `PreMissionImporter` (R1) of four rows, a registry with a
derived kind `windowed_prior` whose payload holds `value` and `window_lo_rel`, and
`WindowAgePolicy`.

| Row | Kind | Audience | Start | End | Covers |
|---|---|---|---|---|---|
| 0 | measurement | public | -12 | -6 | under a shift of 5 the end lands on -1 |
| 1 | measurement | public | -5 | -1 | an end tick at -1 in the base run, seen by the policy |
| 2 | measurement | observer | -8 | -3 | a release stamp that stays -1 |
| 3 | prior, window 30 | public (two rows) | -20 | 0 | under 19 the start lands on -1; the policy copy's acceptance stamp stays -1 |

The policy's first decision waits until the prior's `released_tick` plus
`window_lo_rel`, plus the age of the newest imported raw row (decision tick less its
end tick); it then observes subjects 0 and 1 in turn three times and stops. The
shifts are 5, 19 and 1000; the oldest time is 20 ticks before the start, so the last
shift moves every time above zero.

## Tests

Tests added: 6, retired: none, changed: 1 (the visible-relations epoch case above). Cold times, each test alone in a fresh process
(`SPACEODYSSEY_JAX_CACHE=off JAX_ENABLE_COMPILATION_CACHE=false`, four threads, wall
time including the about 1.5 s import):

| Test | File | Tier | Cold s |
|---|---|---|---|
| `test_tick_columns_move_by_name_whatever_their_sign` | `test_epoch_relation.py` | unit | 2.2 |
| `test_shift_epoch_refuses_a_shift_that_is_not_positive` | `test_epoch_relation.py` | unit | 2.7 |
| `test_the_fixture_covers_negative_times_minus_one_and_a_shift_past_the_oldest` | `test_epoch_relation.py` | unit | 2.2 |
| `test_the_epoch_relation_holds_with_negative_times_on_both_executors` | `test_epoch_relation.py` | medium | 14.9 |
| `test_the_epoch_relation_catches_rows_written_at_absolute_ticks` | `test_epoch_relation.py` | medium | 9.3 |
| `test_shift_epoch_catches_a_value_that_reads_an_absolute_observation_tick` | `test_metamorphic.py` | slow | 9.8 |

- The relation test runs the three shifts on both executors (one compiled program
  per executor) and asserts, besides the pass, ten rows before the start and ten
  decision views over the two worlds.
- Control, medium: the fixture's importer with `absolute` set writes each start tick
  as if the run started at tick 0. The policy never reads start ticks, so the only
  failures allowed, and required, are the start-tick columns of the log and of the
  policy views. `absolute` is an array, so the control shares the relation test's
  program.
- Control, slow (the R0a inventory's absolute-tick value control): the policy with
  `rule="absolute"` takes the age as minus the end tick, as if the run started at 0.
  Both readings agree on the unshifted run; under a shift past the oldest time the
  wait moves and the relation fails on the trace and view ticks. The second policy
  type is a second compiled program, on the compiled executor only.

## Mutants and controls

Each mutant was applied to a copy of `src/` and the new tests, with the existing
`test_shift_epoch`, run against it (`$TMPDIR` script, not committed). All are killed.

| Mutant | Failing tests |
|---|---|
| both comparators ignore the shift (compare absolute ticks) | unit `moved_ticks`, relation, absolute-start control, `test_shift_epoch` |
| log comparator compares absolute ticks | relation, absolute-start control, `test_shift_epoch` |
| view comparator compares absolute ticks | relation, absolute-start control, `test_shift_epoch` |
| ticks taken by sign in both (the old rule) | unit `moved_ticks`, relation, absolute-start control |
| log comparator by sign (the old code) | relation, absolute-start control |
| view comparator by sign (the old code) | relation, absolute-start control |
| -1 taken as unset in every tick column | unit `moved_ticks`, relation, absolute-start control |
| an unset release or acceptance stamp moves | unit `moved_ticks`, relation, absolute-start control |
| both comparators skip tick columns | absolute-start control |
| the shifted run is not moved | relation, absolute-start control, `test_shift_epoch` |
| the trace tick is not compared | absolute-value control (slow) |
| a non-positive shift is run | refusal test |
| the fixture's middle shift 19 changed to 18 (no start lands on -1) | fixture test |

The two "old code" mutants pass the existing `test_shift_epoch`, whose survey has no
negative time; that is the gap B6 closes.

## Checks run

- `tests/core/test_golden_histories.py`, `tests/codec/test_run_document_runs.py`
  (golden pin and corpus document hash pin) and `tests/test_slice1_gate.py`, which
  imports the changed helper module: 355 passed in 574.9 s (`-n 2 --dist loadfile`).
  No pin moved.
- The battery and metamorphic files and the files using the changed helpers
  (`test_epoch_relation.py`, `test_metamorphic.py`, `test_battery_guards.py`,
  `test_fork.py`, `test_latency_shift.py`, `test_null_reference.py`,
  `test_truth_swap.py`, `test_visible_relations.py`, `test_belief_layout.py`,
  `test_declarations_match_run.py`, `test_lints.py`, `test_premission.py`,
  `test_survey_25.py`, `test_survey_2_static.py`, `test_survey_5.py`): 289 passed in
  90.3 s.
- Commit 1 alone (the comparator fix with its unit test, before the executor change):
  `test_epoch_relation.py`, `test_metamorphic.py`, `test_visible_relations.py`,
  `test_latency_shift.py`, `test_lints.py`: 208 passed.
- Quick tier at the final tree (`-m "not slow" -n 2 --dist loadfile`, deselecting
  `tests/codec/test_entry_points_installed.py` for the stale environment metadata
  noted in R0a): 2,018 passed in 207.5 s. The budget plugin reported one unit over
  5 s, the existing and untouched `test_snapshot_fingerprint.py::test_snapshot_fingerprint_recorded`
  (8.8 s, on a machine shared with another job).
- `ruff check`, `ruff format --check`, `tools/check_internal_refs.py`, the domain-word
  and core-import lints (`test_lints.py`) and `pre-commit` on the touched files:
  clean. Every touched file is ASCII.
- The full suite and the Sphinx docs build were not run.

## Pins

No pinned golden digest, fingerprint or corpus document hash moved
(`golden_pin.json` and `hash_pin.json` untouched and passing), and `schema_version` is
unchanged. The only library files touched are `core/battery/metamorphic.py` and
`core/battery/views.py`, the paths K2 allows.

## Spec silences decided here

1. **Where B6 lives.** The spec names `tests/core/test_metamorphic.py`, but
   `tests/conftest.py` lists that whole file in `SLOW_FILES`, so a test there never
   runs in the quick job, and `conftest.py` is R6's (Q9). The relation test and the
   absolute-start control, compile-bound and under 30 s, are medium in a new file,
   `tests/core/test_epoch_relation.py`; the absolute-value control, slow by the R0a
   inventory, is in `test_metamorphic.py`. The fixture is in `battery_helpers.py`,
   which the `name-tests-test` hook already exempts; a new helper module would need
   a `.pre-commit-config.yaml` edit. Rejected: editing `SLOW_FILES` (R6's file).
2. **Both executors.** The relation ran a bare compiled scan of the transition, not
   either executor. It now runs through `run_worlds` with the start moved by a
   process wrapper in `core/battery/`. Rejected: an initial-carry argument on the
   executors (a core runtime change, outside K2).
3. **Which -1 is unset.** Only `released_tick` and `accepted_tick` take -1 as unset.
   `available_tick` is set on every valid row (a write sets it to the event tick or
   later), so it moves like the span ticks.
4. **Positive shifts only.** A zero shift checks nothing; a negative one could move a
   release onto -1, where the comparators would read it as unset. No caller used
   either.
5. **The window in the fixture.** R1 does not write the table's window columns into
   records (R1 silence 8), so the synthetic prior carries `window_lo_rel` as a public
   payload column of its own derived kind. The policy reads `released_tick` plus that
   column, so a correct run shifts its decisions and leaves the payload equal.
6. **The release delay shares the rule.** `compare_views` applies `moved_ticks` to the
   delay of the latency relation too. For nonnegative times nothing changes; a
   negative observation time under a delay is now compared correctly as well.
7. **The fixture finds imported raw rows by channel id** (60). This is a fixture
   policy, not a pattern for studies.

## Spec points not satisfied

- B6 "on the four studies": R4a covers the synthetic study. Surveys 3, 4, 6 and 18
  run the relation when R4b and R5 build them.
- The relation's assumptions (no `Query` request, no refresh schedule) are stated in
  the docstring, not enforced; a refresh grid under the shift is Slice 2's own check
  (D7, out of scope).
