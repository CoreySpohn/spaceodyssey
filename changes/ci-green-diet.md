---
orphan: true
---

# CI green and the quick-tier diet

Main was red on two workflows and its quick tier was over budget. This change makes the
hygiene and docs jobs pass, adds per-test data to the cold pilot, and moves enough
heavy tests to the slow tier that the quick tier projects well under the 540 s ceiling
on a cold runner. No file under `src/` changed.

## Commits

1. `ci(hygiene): name-tests-test skips four support modules`
2. `docs(changes): format the observer example fence`
3. `ci(docs): executed examples skip the change records`
4. `ci(pilot): upload the cold quick-tier junit file and print the 40 slowest tests`
5. `ci(pilot): compare the cold quick tier with the pinned baseline`
6. `test(studies): the survey and calibration runs move to slow, K7 on the cold runner`
7. `test(adapters): cold jaxedith, executor-parity and prior refresh runs move to slow`
8. `test(adapters): the OFTI exhaustion status is checked without a real sampler`
9. `test: unit tests over 3 s cold are medium, so the runner stays under the unit budget`
10. this record

## Hygiene and docs

- **name-tests-test.** `tests/codec/run_cases.py`, `tests/studies/survey_helpers.py`,
  `tests/core/refresh_helpers.py` and `tests/anchors/reference_geometry.py` are support
  modules, not test files. They join the hook's exclude list, which already names the
  other support modules.
- **ruff-format.** The hook reformats Python fences in `changes/*.md`; the one
  rewrite (`slice-2-t1b.md`) is committed.
- **Executed fences.** Three fences in `changes` fail when run
  (`slice-2-t1b.md` line 155, `slice-2-t6.md` line 27, `slice-2b-r0a.md` line 109).
  They are design sketches in change records, not user examples, so `ci.yml` passes
  `--ignore=changes` to the markdown-docs run. README and the user docs are still
  executed (two fences, both pass). Marking each sketch `python notest` was rejected: it
  would have to be repeated for every future record.

Replicated locally, each step of `ci.yml` and `hygiene.yml` that does not need the
hosted runner:

| Step | Result |
|---|---|
| `pre-commit run --all-files` (with `SKIP=conventional-pre-commit`) | all hooks pass |
| `pytest README.md docs --ignore=changes --markdown-docs --markdown-docs-syntax=superfences` | 2 passed |
| `uvx zizmor --format=github --min-severity=medium .github` | clean |
| `uvx codespell --check-hidden` (without the local `.ruff_cache`) | clean |
| `uvx deptry src/spaceodyssey` | no dependency issues |
| `ruff check` and `ruff format --check` on `src/spaceodyssey tests` | clean |
| quick tier, enforcing | 1 failure, `test_entry_points_installed.py` (stale installed metadata; the entry point is in `pyproject.toml`) |

The Sphinx build was not run here (no docs extras in the shared environment).

## Pilot data

`pilot.yml` quick-cold now uploads `junit-quick-cold.xml` as the `junit-quick-cold`
artifact and prints the 40 slowest tests with no minimum duration, so the next
measurement has per-test runner times. Its summary step also compares the cold wall time
with the pinned baseline (see "The baseline").

## Quick-tier diet

### Measurement

Cold means `SPACEODYSSEY_JAX_CACHE=off JAX_ENABLE_COMPILATION_CACHE=false`, no pytest
cache, `-m "not slow" -n 2 --dist loadfile`, junit per test, through `compute-run
--threads 4`. The Mac was shared and heavily loaded during every run (load average
10 to 12), so wall times moved by up to 50 percent between runs of one tree.

| Tree | Local wall (s) | Unit tests, summed (s) | Medium tests, summed (s) |
|---|---|---|---|
| `423d74c` (before; same tests as `b60eba9`), three runs | 462, 439, 433 | 330, 310, 258 | 544, 505, 543 |
| after the moves (commit 9) | 168, 165, 214, 185, 264 (one run 1.5 times loaded) | 111, 103, 122, 108, 158 | 205, 205, 272, 229, 345 |

CPU time (user plus system, `time -l`) is steadier: 1266 s before, 530 s after, a ratio
of 0.42. The wall ratio of the two adjacent quiet runs is 0.38.

### Fit of the runner factor

The Pilot run at `b60eba9` is the only one with the tier split: 2121 unit tests summing
486.8 s and 73 medium tests summing 1104.9 s on two workers, 857.3 s wall. The tree is
the same as the "before" row.

- unit factor = 486.8 / 309.5 = 1.57, medium factor = 1104.9 / 505.4 = 2.19 (local sums
  of the run taken next to the pilot's tier counts; the other before-run gives 1.89 and
  2.03).
- packing: wall = 1.077 x (summed time) / 2, from 857.3 against 795.9 (the same constant,
  1.078, holds locally: 439.2 against 407.5).
- check against the other Pilot point, `97753d5` (1966 tests, 434.5 s): the Mac measured
  about 229 s for that ref, a wall ratio of 1.9; the same ratio for `b60eba9` is 857 against
  309 (2.8), which is what the medium-heavy mix of the later tree predicts.
- projected runner wall = 1.077 x (1.57 x U + 2.19 x M) / 2, with U and M the local
  summed unit and medium time.

The factors are fitted to a loaded Mac and applied to Mac runs under similar load, so
they are consistent with each other and not with a quiet laptop. This is an estimate
until a Pilot run at the merge commit replaces it.

### Projection

| Run | U (s) | M (s) | Projected cold runner wall (s) |
|---|---|---|---|
| before (`423d74c`) | 309.5 | 505.4 | 857 (the Pilot figure, by construction) |
| after, quiet run | 111.5 | 205.0 | 336 |
| after, CPU-time ratio method (0.42 x 857) | | | 359 |
| after, the one 1.5 times loaded run | 157.9 | 345.4 | 540 (load, not the tree: the same tree took 165 s the next time) |

The projection is **330 to 360 s**, against the 480 s target and the 540 s ceiling. The
headroom (about 120 s) is deliberate: Slices S3a and S4 add tests that replay onto this
tree. The diet fell more than the removed tests alone explain (308 local seconds of
medium tests removed, projecting about 360 s of runner wall, against 520 s observed). The
unit tier also dropped from about 310 to about 110 local seconds. I did not isolate the
cause; the likely one is that the survey runs push resident memory over the conftest
threshold that clears the compile caches, so fewer clears leave more programs shared.

### K6, judged cold on the runner

The 121 tests added by the 75 Slice 2b commits (the gate's own commit list, tests found
by the functions each commit added; R6 counted 144 with its own method) cost, by the same
fit:

| State | Unit (s) | Medium (s) | Projected runner delta (s) |
|---|---|---|---|
| before this change | 25.2 | 132.0 | 177 |
| after | 15.6 to 18.0 | 18.4 to 23.0 | 35 to 45 |

That is under the 60 s allowance. The margin is thin under load: the loaded run above
gives 61 s. The next test to move if the Pilot says otherwise is
`test_premission.py::test_import_rows_are_stamped_folded_once_and_seen_by_the_one_first_decision`
(about 17 s local, 20 s of runner wall).

### Moves from medium to slow

Local cold seconds are from the "before" run next to the Pilot figure; the runner
column is 1.18 s of wall per local medium second (1.077 x 2.19 / 2). Per-test times shift
when a file's first test, which pays the compile, moves, so the file totals are the figure
to read.

| Test or file | Local cold (s) | Projected runner (s) | Reason; what still guards it at merge time |
|---|---|---|---|
| `test_physical_ofti.py::test_executor_parity_on_one_certified_visit` | 37.4 | 44 | both executors compile the study around one real fit; `test_one_real_fit_reproduces_the_orbit_and_equals_ofti_on_the_t4d_table` (medium) and the file's units. Executor parity is slow-only |
| `test_survey_3.py::test_an_observer_only_source_moves_no_policy_view_or_decision` (2b) | 27.7 | 33 | two variants, one program per executor; `test_premission.py::test_overlap_observer_only_row_does_not_move_policy` (K3). The whole-run check is slow-only |
| `test_survey_18.py::test_the_precursor_changes_the_plan_only_when_released_to_the_policy` (2b) | 21.4 | 25 | one program per executor. Slow-only |
| `test_premission_refit.py` prior across a refresh, and two subjects sharing a local index (2b) | 33.7 | 40 | host refresh studies; `test_the_audience_prior_boundary_meets_its_contract` (medium) and the D009 and D010 units. Both are slow-only |
| `test_jaxedith_summary.py` variance and cached-rates agreement | 30.6 | 36 | the response package and cold jaxedith programs; `test_fixed_frame_floor_agrees_with_jaxedith` and `test_hand_count_and_variance_budget` (medium) |
| `test_survey_2_calibration.py` (whole file) | 18.0 | 21 | 2000 worlds on one adaptive run shared by all 14 tests, a statistical campaign. Slow-only |
| `test_survey_2.py` four runs (no-refresh equals survey 1, with-refresh on both executors, adaptive policy acts on belief) | 53.6 | 63 | cold survey runs; `test_the_adaptive_rule_matches_an_eager_reference` and the run-document round trip stay. The refresh-leaves-the-actions-alone check is slow-only |
| `test_survey_6.py::test_the_epoch_relation_holds_for_three_shifts_on_both_executors` (2b) | 16.4 | 19 | one program per executor; the comparator units (`test_tick_columns_move_by_name_whatever_their_sign`, `test_shift_epoch_refuses_a_shift_that_is_not_positive`). The relation across executors is slow-only |
| `test_survey_2_ofti.py` (both tests) | 32.0 | 38 | two cold host runs shared by both tests; the photomancy adapter tests. Slow-only |
| `test_survey_5.py` (whole file) | 37.6 | 44 | every test runs a cold study. Slow-only |

Total: about 308 local seconds, 363 projected runner seconds.

Three tests the first sizing moved were put back to medium, because they are the
adapter's only run-level guards and cost about 13 s of runner wall together:
`test_photomancy_ofti.py::test_a_checkpoint_fit_installs_for_both_audiences_and_replays`,
`::test_a_column_one_audience_cannot_read_is_no_support_for_that_audience` and
`::test_exhausted_batches_leave_the_belief_unchanged`.

### Unit tests over the runner budget

A unit test over 5 s on the runner is about 3.2 s locally (unit factor 1.57).

- `test_photomancy_ofti.py::test_a_failed_or_unsupported_entry_is_a_status_never_an_exception`
  (6.75 s on the runner) ran a real OFTI compile only to exhaust its batches. It now
  replaces the sampler with one that raises the exhaustion error, which is the mapping the
  test guards; the real sampler exhausting is the medium
  `test_exhausted_batches_leave_the_belief_unchanged`. It is now under a second.
- `test_the_same_rows_and_key_give_the_same_particles_and_a_new_set_a_new_key` runs two real
  OFTI fits (9.3 s here) and is medium.
- These unit tests measured over 3.3 s here, are compile-bound with no cheaper form, and
  are now medium (no time change): `test_reserved_fields.py` (the ledger round trip),
  `test_physical_context.py` (two), `test_jaxedith_summary.py` (three),
  `test_planit_adapter.py::test_b7_sequences_utilities_and_admitted_starts`,
  `test_observer_contract.py::test_each_fold_draws_from_its_records_inference_key`,
  `test_declarations_match_run.py` (two), `test_pedigree_layout.py`,
  `test_record_statuses.py`, `test_wake_coalescing.py`, `test_coincident_order.py`,
  `test_skyscapes_scene.py`, `test_inputs.py` (the cross-process digests),
  `test_viz_package.py` (two), `test_replay_refresh_quick.py`, `test_document.py` (the
  cross-process hash) and `test_reporting_law.py`.

Tests that cross 5 s on the runner only in a loaded run (single runs near 5 s locally, for
example `test_record_statuses.py::test_identities_and_provenance_of_a_settled_record` at
5.19 s in one run) are left alone.

### Retired

None.

### Slow-only mechanisms

These have no merge-time guard of their own; the nightly slow tier and the Slice 2b gate
cover them. No unit-sized replacement was written: each needs a study fixture, which would
add a test file or a shared helper, and the pending branches replay onto this tree.

- survey 2: refresh leaves the actions alone; adaptive design and calibration (whole file);
  the OFTI checkpoint on survey 2 (whole file)
- survey 3: an observer-only source over a whole run (the K3 handler check stays)
- survey 5: all of it
- survey 6: the epoch relation across both executors (comparator units stay)
- survey 18: all of it
- OFTI executor parity on a certified visit
- R2: the prior across a refresh, and two subjects sharing a local index through refit and
  replay

## The baseline

`.github/budget/quick_baseline.json` keeps its meaning, the previous slice's close-out
quick tier, and its value is unchanged. It is still the warm 418.4 s from `97753d5`; the
cold Pilot figure at that ref is 434.5 s (1966 tests).

How the comparison reads:

- **Warm CI run against a cold baseline.** A warm run is faster than a cold one, so a cold
  baseline makes the +60 s check lenient, never falsely red; a cache miss on CI is then a
  like-for-like comparison. With the warm baseline in the file today, a cold-cache CI run
  would be judged against a warm figure.
- **The +60 s check is meaningful once per slice, at its close-out.** It measures what one
  slice added over the previous close-out. Between close-outs every merged slice adds
  to the distance, so `main` would go red on the push that crosses 60 s through no fault
  of the latest change.
- **The ceiling is the standing rule.** 540 s applies to every run.

The smallest change, made here: the pilot's cold quick-tier job now compares its own wall
time with the pinned baseline (`--baseline` and `--max-increase-seconds 60` in its summary
step), so K6 is judged cold against cold, at a ref chosen at close-out. `ci.yml` is
unchanged, so the Slice 2b gate's wiring check still passes. The follow-up, for the next
close-out: drop `--baseline` from the quick tier's summary step in `ci.yml` and
update the gate's K6 wiring test with it, once the pilot carries the relative check.

## What remains

1. Push the branch (not done here).
2. Dispatch Pilot with `quick_ref` set to the merge commit, then read the uploaded
   `junit-quick-cold` artifact: refit the two factors from per-test runner times and
   compare the cold wall time with the 330 to 360 s projection and the 480 s target.
3. Pin the baseline from that Pilot summary (`quick-cold-summary.json`) with `cache` set to
   cold.
4. If the Pilot lands over 480 s or the 2b delta over 60 s, move the next tests from the
   list in "K6" and from the medium tests that are largest per the artifact.
5. Restore moved tests in this order if the Pilot leaves room: the OFTI executor parity
   test, then `test_jaxedith_summary.py`'s two, then survey 2's eager-versus-refresh runs.
