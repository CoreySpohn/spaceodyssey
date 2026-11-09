---
orphan: true
---

# Slice 2b R6: close-out

R6 closes Slice 2b. It adds the gate that runs kill rules K1 to K5 as one command,
turns on the test-budget enforcement of decision Q9 (option A), measures the cold
quick-tier cost the slice added and moves tests to the slow tier until it fits (K6,
Corey's 2026-11-07 decision), retires three subsumed tests, and records the status
of B1 to B8 and K1 to K7.

The branch starts from `main` at `794a715` (Slice 3a S2), so every figure below is
for the tree that merges.

## Commits

1. `build(test): require the budget plugin, enforce the test tiers and record each
   test's tier` (`70aad42`)
2. `ci(budget): quick tier held to 540 s and 60 s above a pinned baseline, with a
   cold quick-tier pilot` (`3067bcb`)
3. `test(core): the slice 2b gate runs K1 to K5 as one command and checks the K6 and
   K7 wiring` (`862fae8`)
4. `test(studies): five slice 2b medium tests move to slow and three subsumed tests
   are retired (K6)` (`1cee2c1`)
5. `test(codec): the reloaded-study run test is medium, one host compile of about 9 s
   cold` (`6116c33`)
6. this record, first version (`bd3a846`)
7. `test(core): the slice 2b gate lists the slice 3a S2 commits on main as other
   work` (`77e5af4`)
8. `test(core): three unit tests over 5 s cold are medium, so the enforced unit
   budget holds` (`6f04c13`)
9. this record, completed

## What changed

- `tests/test_slice2b_gate.py` (new, slow tier): K1 to K5, the K6 and K7 wiring and
  the named interface checks. One command: `pytest tests/test_slice2b_gate.py`.
- `tests/conftest.py`: the budget plugin is loaded unconditionally; every test
  records its tier (`unit`, `medium`, `slow`) as a junit property; with
  `SPACEODYSSEY_BUDGET_SCALE` set, every tier budget is multiplied by it; the gate
  file joins `SLOW_FILES`.
- `pyproject.toml`: `[tool.testbudget] mode = "enforce"`; the `test` extra requires
  `hwoutils>=1.11.0`, the first release with the budget plugin.
- `.github/scripts/ci_summary.py`: `--baseline <summary.json>` with
  `--max-increase-seconds`, `--require-tier`, `--write-summary` and `--commit`; the
  summary reports tests and summed time per tier.
- `.github/budget/quick_baseline.json` (new): the pinned, provisional baseline.
- `.github/workflows/ci.yml`: the quick tier's summary step enforces 540 s, the
  baseline with a 60 s limit and the presence of medium tests;
  `SPACEODYSSEY_BUDGET_SCALE: "3"`; the slow job checks out the full history (the
  gate's K2 reads it).
- `.github/workflows/pilot.yml`: a `quick_ref` input runs the whole quick tier cold
  at that ref (budget plugin in report mode) and prints its summary in the baseline
  format; without it the pilot runs as before.
- `.github/workflows/upstream.yml`: the canary runs the budget plugin in report mode
  (it looks for upstream breakage; `ci.yml` enforces the budgets).
- Test tiers: five Slice 2b tests from medium to slow, three Slice 2b tests retired,
  four tests outside Slice 2b from unit to medium (below).

## The gate

`tests/test_slice2b_gate.py` re-exports the named checks under `test_k<n>_` names, so
one command runs them, and adds the checks no other file makes.

| Rule | What the gate runs |
|---|---|
| K1 | the golden-history pin tests (pin coverage, manifest equals pin, 58 history digests, 58 fingerprint cases, the fault fixture on both executors); a git check that both pins name `96a6447` and that no commit of the slice touched a pin file except the two R0a captures (`5b9fbd5`, `fc0e2e4`) |
| K2 | `test_k2_commits_touch_only_allowed_paths`, with the range and exclusion checks below and a control |
| K3 | `test_overlap_permutation_invariant` (both rules), `test_overlap_observer_only_row_does_not_move_policy`, `test_retained_priors_disjoint` |
| K4 | the hash pin's commit and coverage, every corpus document hash and the schema version, the changed-study control, the table round trip through file references, tampering refused, lineage passing and failing |
| K5 | the domain-word lint and the import lint over the core, and the scan-coverage check |
| K6, K7 | the quick-tier summary step carries `--budget-seconds 540`, `--baseline`, `--max-increase-seconds 60` and `--require-tier medium`; the quick tier selects `-m "not slow"`; the plugin is mandatory and enforcing; the baseline names its source and status; the script itself fails 61 s over a baseline, over 540 s and with no medium test, and passes 59 s over |
| named | `test_replay_reproduces_prior_run` (T4c) and `test_a_refit_observer_keeps_its_prior_across_a_refresh` (the T3b derived-summary fixture); a check whose dependency is not importable is listed as pending and fails the gate. Both dependencies are merged: none is pending |

### Which commits are Slice 2b (K2)

- **Range.** From the pins' source commit `96a6447` (the R0a record) to the newest
  commit that touches a `changes/slice-2b-*.md` record. Every task ends with its
  record, so the slice ends there; later work on `main` is not judged.
- **Other work in the range** is listed in the gate by full sha with its subject
  prefix and its change record: physical PT4 (8 commits, which landed between the pin
  commit and the R0a commits), PT6 (5), Slice 3a S1 (6) and Slice 3a S2 (3). A stale
  entry, an entry outside the range, or an entry that touches a file only Slice 2b
  writes fails; the one exception is listed with its reason (`794a715` adapts the
  export test's log helper to a log column the capacities do not hold).
- **Merges.** The one merge in the range (`536803f`, `main` into `s2b-r5`, bringing
  R3) has an empty combined diff: it carries no change of its own, and the commits it
  brought in are judged one by one. The gate requires that of every merge.
- **Allowed paths.** `core/battery/`, `core/conformance/`,
  `core/preflight/findings.py`, `core/preflight/checks.py`, `studies/`, `codec/`,
  `tests/`, the Slice 2b records, and the CI files of the budget enforcement
  (`pyproject.toml`, `ci.yml`, `upstream.yml`, `pilot.yml`, `ci_summary.py`,
  `pilot_summary.py`, `.github/budget/`). Reviewed one-commit allowances: R0a's
  `tools/step_cost.py`, `.codespellrc` and `slice-3a-s0.md` (shared with 3a S0), and
  R2's `.pre-commit-config.yaml` (`93a12af`).
- **Control.** Slice 3a S1's core commit `f155d0b`, judged as if it were Slice 2b, is
  caught at `core/capacities.py`; `17976f4` is caught at `.gitignore`.
- **Result.** 75 Slice 2b commits (R0a to R5, A10 and R6), no path outside the list.

## K6: the cold quick-tier cost of Slice 2b

### How it was measured

The whole quick tier (`-m "not slow" -n 2 --dist loadfile`) on this Mac, cold
(`SPACEODYSSEY_JAX_CACHE=off JAX_ENABLE_COMPILATION_CACHE=false`, no pytest cache,
`compute-run --threads 4`), on exported trees, in one session, interleaved. The
Slice 2b delta is the same tree run with and without the 144 Slice 2b quick-tier
tests (deselected by node id), so the other slices' tests and machine drift cancel.
The machine was shared with other jobs throughout; repeated runs of one tree differ
by up to 30 s. Wall times are the junit testsuite times.

| Tree | Runs (s) | Mean (s) |
|---|---|---|
| `97753d5` (before Slice 2b) | 220.7, 224.5, 241.4 (a fourth, 338.6, ran under heavy load) | 228.9 |
| `b2c7e03` (all of Slice 2b, before R6) | 362.7, 338.2, 363.0 | 354.6 |
| `b2c7e03` without the Slice 2b tests | 258.5 | 258.5 |
| `794a715` with R6 | 310.6, 308.0 | 309.3 |
| `794a715` with R6, without the Slice 2b tests | 250.7, 280.2 | 265.5 |

- **Before the retiering:** Slice 2b added about **96 s** (354.6 against 258.5), with
  152 tests summing 181 to 193 s over the two workers.
- **After:** about **44 s** (309.3 against 265.5; the three adjacent pairs give 60,
  31 and 28 s), with 144 tests summing 106 to 108 s. Within the 60 s allowance.
- **Not Slice 2b**, reported separately: everything else since `97753d5` adds about
  37 s cold here (265.5 against 228.9): PT6 (`test_physical_ofti.py`, about 35 s
  summed), Slice 3a S1 (`test_slice3_guard.py` and others, about 14 s summed) and
  Slice 3a S2 (`test_programs.py`, about 16 to 18 s summed; `test_refresh_runs.py` is
  a slow file and adds nothing to the quick tier). They count toward K7, not toward
  the Slice 2b delta.

### The Slice 2b quick-tier tests after the retiering

Cold, mean of the two full runs on the merge tree (junit time, setup included):

| Test | Tier | Guards | Cold s |
|---|---|---|---|
| `test_survey_3.py::test_an_observer_only_source_moves_no_policy_view_or_decision` | medium | survey 3 (d), K3, D2 replay | 22.1 |
| `test_survey_18.py::test_the_precursor_changes_the_plan_only_when_released_to_the_policy` | medium | survey 18 (a), (c), (e) | 16.0 |
| `test_premission_refit.py::test_a_refit_observer_keeps_its_prior_across_a_refresh` | medium | R2 prior and refit (B3), host replay, T3b fixture | 14.3 |
| `test_premission.py::test_import_rows_are_stamped_folded_once_and_seen_by_the_one_first_decision` | medium | R1 importer (B1) | 14.0 |
| `test_survey_6.py::test_the_epoch_relation_holds_for_three_shifts_on_both_executors` | medium | survey 6 (a), R4a epoch relation with negative times | 13.4 |
| `test_premission_refit.py::test_the_audience_prior_boundary_meets_its_contract` | medium | R2 boundary contract | 5.0 |
| `test_survey_4.py::test_a_precursor_exports_imports_and_verifies_its_lineage` | unit | survey 4, R3 codec and lineage | 2.7 |
| `test_premission.py::test_overlap_observer_only_row_does_not_move_policy` | medium | K3 | 1.7 |
| the other 136 tests | unit | | 18.3 together |

### Moves (medium to slow)

| Test | Cold s before | What still guards it at merge time |
|---|---|---|
| `test_epoch_relation.py::test_the_epoch_relation_holds_with_negative_times_on_both_executors` | 29.0 | the survey 6 guard runs the relation with negative times, a time landing on -1 and a shift past the oldest, on both executors; the comparator unit tests (`test_tick_columns_move_by_name_whatever_their_sign`, `test_shift_epoch_refuses_a_shift_that_is_not_positive`) |
| `test_epoch_relation.py::test_the_epoch_relation_catches_rows_written_at_absolute_ticks` (control) | under 1 (shares the program above) | the comparator unit test; the control runs nightly |
| `test_premission.py::test_one_initial_decision_with_and_without_a_public_zero_information_import` | 10.2 | the R1 guard (one first decision) and the seed unit test (the kickoff retargeted) |
| `test_premission_refit.py::test_replay_reproduces_prior_run` (named T4c check) | 18.1 | the R2 guard replays the host run; the compiled run and its replay run nightly and in the gate |
| `test_premission_refit.py::test_equal_resolutions_refit_alike_on_both_boundaries` (A10) | 7.1 | the D010 unit test (the observer-init boundary accepted with equal resolutions, refused otherwise) |

Every mechanism and survey keeps a merge-time guard: R1 importer, R2 prior and refit,
R3 codec and lineage (the unit tests of `test_premission_codec.py` and survey 4's
unit guard), R4a epoch (survey 6's guard and the comparator units), surveys 3, 4, 6
and 18.

### Pruning (retired)

A fresh-eyes reviewer read the Slice 2b tests against the admission and pruning
rules and agreed to these three; each guard stays covered:

| Retired | Subsumed by | Guard kept |
|---|---|---|
| `test_premission_codec.py::test_a_toy_survey_channel_of_several_subjects_exports_to_one_channel` | `::test_two_subjects_of_one_channel_export_with_one_local_index` (same mutants R7, R9) | local index unique per channel and subject (R3 review round 1) |
| `test_premission.py::test_two_subjects_sharing_a_local_index_keep_their_own_weights[folded]` | the `handled` node of the same test (identical assertions; the handlers call the same fold) | R7, R8 |
| `test_premission_refit.py::test_without_a_boundary_d009_and_d010_are_reported_unverified` | `test_preflight_json.py::test_the_run_record_boundary_reaches_the_prior_checks` (unbound branch) and the seeded `prior_refit_boundary_unverified` case | B4 for the D010 warning, mutant M12 |

Two weaker candidates were kept (the importer's inline codec round trip, and the
corpus hash test that is the named B5 and K4 item).

Tests: added 152 collected in the gate (re-exports, plus 11 of its own checks);
retired 3; net in the quick tier: 5 medium tests moved to slow, 3 retired, 4
non-2b tests relabelled unit to medium.

### Tests outside Slice 2b relabelled for the enforced unit budget

With the plugin enforcing, a unit test over 5 s fails. These were over it in every
cold run here and are now medium (no quick-tier time changes):
`test_document.py::test_reloaded_study_runs_and_reuses_the_compiled_program` (about
9 s), `test_stop_equivalence.py::test_stop_equivalence_below_campaign` (10 to 17 s on
its first scope), `test_wake_coalescing.py::test_per_wake_rule_would_decide_twice_and_the_first_misses_a_record`
(about 7 s), `test_snapshot_fingerprint.py::test_snapshot_fingerprint_recorded`
(about 7 s). Tests that crossed 5 s only now and then in a cold run on the loaded
machine (5.0 to 5.7 s: two in `test_coincident_order.py`, two in
`test_declarations_match_run.py`, `test_reserved_fields.py` `ChannelTable.charge`,
`test_record_statuses.py`, `test_wake_coalescing.py::test_two_releases_one_tick_one_decision`,
one each in `test_photomancy_ofti.py` and `test_jaxedith_summary.py`) and the two
Slice 3a S2 unit tests in `test_programs.py` measured at 6.6 and 7.8 s with setup are
left to their owners.

## Q9 enforcement

- **Plugin.** `pytest_plugins = ["hwoutils.testbudget.plugin"]` unconditionally and
  `mode = "enforce"`: a unit test over 5 s, a medium over 30 s or a slow over 300 s
  fails. The budgets are laptop figures; one process on the hosted runner is about
  three times slower (E0), so `ci.yml` sets `SPACEODYSSEY_BUDGET_SCALE=3`, which
  conftest turns into a `time_budget` marker on every test without one. The cold
  pilot and the upstream canary pass `--testbudget-mode report`.
- **Medium tests counted.** CI selects `-m "not slow"`, which includes them; each test
  now carries its tier in the junit file, the summary prints counts and time per tier,
  and `--require-tier medium` fails a quick tier that has none.
- **Ceiling.** `--budget-seconds 540` (was 600).
- **Relative limit.** `--baseline .github/budget/quick_baseline.json
  --max-increase-seconds 60`.

### The baseline

The decision names the Slice 2 close-out (T11) quick-tier summary, which does not
exist yet. The pinned file is provisional and says so:

- Source: CI run `42325472621`, job `128259618434` (Quick tier, py3.12,
  ubuntu-latest), the push of `97753d5`, the last commit before the Slice 2b build:
  junit testsuite time 418.37 s, 1,966 tests (1 failed, 1,965 passed).
- **Warm, not cold.** That job restored the JAX compile cache (key
  `jax-Linux-py3.12-...-031c074...`). The CI quick tier at `b2c7e03` restored the same
  key and took 712.6 s (2,173 tests): the cache on `main` has not been saved since
  `031c074`, because `actions/cache` saves only when the job passes and the quick
  tier has failed since then (`test_entry_points_installed.py`: `ToyRefitMean`, used
  by a corpus study, has no entry point in `pyproject.toml`). So both CI figures are
  partly warm, and K6, judged cold, cannot be read from them.
- The cold pilot run `42325497216` at `97753d5` measured three medium study files
  cold (survey 5: 44.6 s, survey 2: 86.3 s, survey 2 calibration: 16.1 s), not the
  whole quick tier, so it cannot serve as a cold baseline either.
- The clean fix: `pilot.yml` now runs the whole quick tier cold at a given ref
  (`quick_ref`). Dispatched at `97753d5` and at the merge commit, the two printed
  wall times give the cold Slice 2b plus other-work delta on the runner, and the
  `97753d5` summary can replace the warm figure in the baseline file (with `cache`
  set to cold) until T11 replaces it.

## Status: B1 to B8

| # | Status | Evidence |
|---|---|---|
| B1 | met | `test_premission.py`: stamping, one fold and one first decision (merge-time guard), capacity 2N events and 3N ids, overflow of the queue, log and ids (units); one initial decision with a public zero-information import for a stateful policy (slow). All pass at the final tree |
| B2 | met | `test_premission.py` disjoint external prior, exclusion rules against sufficient statistics, per-audience resolution, the three K3 checks; `test_observer_kit.py` prior checks and their controls |
| B3 | met | `test_premission_refit.py`: prior kept across a refresh with the flat-start control (medium), the boundary contract (medium), D009 and D010 (units), compiled replay (slow, named T4c check) |
| B4 | met | `test_preflight.py` and `test_preflight_json.py` seeded and repaired D004 to D010 cases at their pointers |
| B5 | met | `test_premission_codec.py`; the gate's K4 |
| B6 | met | the relation runs on the synthetic study and surveys 3, 4, 6 and 18 with both comparators; at merge time only survey 6's guard and the comparator units, the rest slow (K6) |
| B7 | met, partly at merge time | every survey row passes on both executors in the slow tier; the merge-time guards cover survey 3 (d), (e), survey 4's lineage, survey 6 (a) and survey 18 (a), (c), (e) |
| B8 | met | K1 |

## Status: K1 to K7

| Rule | Status | Evidence |
|---|---|---|
| K1 | pass | gate run at `77e5af4`: the golden suite (124) and the gate's K1 tests pass; no pin changed |
| K2 | pass | 75 Slice 2b commits, no path outside the list; the control is caught |
| K3 | pass | the three named checks in the gate |
| K4 | pass | schema version 1 and all 58 document hashes as pinned |
| K5 | pass | `test_lints.py` (187) and the gate's K5 tests |
| K6 | pass locally, cold: about 44 s (was 96 s) | the table above. On CI the step is wired against a provisional warm baseline; see below |
| K7 | wired; not met on CI yet | the cold local quick tier at the merge tree is about 309 s; the last CI quick tier on `main` (`b2c7e03`) took 712.6 s with a stale cache, over 540 s |

## Checks run

- Gate, golden suite and lints in one process, at `77e5af4` (cache directory fresh):
  `tests/core/test_golden_histories.py tests/test_slice2b_gate.py
  tests/core/test_lints.py`: 463 passed in 441.9 s.
- Every slow test of the Slice 2b files (and of `test_metamorphic.py`,
  `test_observer_kit.py`, `test_preflight.py`, `test_preflight_json.py`), `-n 2`: 34
  passed in 194.6 s.
- Quick tier, enforcing, fresh cache (cold) at `77e5af4`: 2,189 passed, 4 failed:
  `test_entry_points_installed.py` (stale installed metadata in the shared
  environment; on CI it fails for the missing `ToyRefitMean` entry point, as on
  `main`), and three over-budget failures in that cold run: two of the four tests
  since relabelled medium, and the survey 3 guard at 34.0 s on the loaded machine
  (22.1 s in the measurement runs).
- Quick tier, enforcing, warm, at `6f04c13` (the final code): 2,192 passed, 1 failed
  (`test_entry_points_installed.py`, as above) in 205.7 s; no budget failure.
  `ci_summary.py` with the CI flags on its junit file: 72 medium tests, 2,121 unit
  tests, exit 0.
- The K2, K6 and named-check tests of the gate again after this record's last
  commit, so the range ends at R6: 11 passed.
- `ruff check`, `ruff format --check`, `tools/check_internal_refs.py`, `pre-commit run
  --from-ref 794a715 --to-ref HEAD` (all hooks), `uvx zizmor .github`: clean.

## Spec silences decided here

1. **Range end.** The slice ends at its newest change-record commit, so the gate keeps
   passing after other work lands on `main`. Rejected: `HEAD` as the end (any later
   core commit would fail K2).
2. **Budget scale.** The plugin's budgets are laptop figures; an environment factor
   scales them on the runner instead of a second budget table. Rejected: report mode
   on CI (not enforcing), and unscaled enforcement (27 failures at `b2c7e03`'s warm CI
   timings).
3. **Medium tests counted** is made checkable with a tier property per test and
   `--require-tier medium`.
4. **The cold baseline** is a pilot job, not a change to the regular quick tier, which
   keeps its compile cache.

## Spec points not satisfied

- The baseline is not the Slice 2 close-out summary (it does not exist yet) and is
  warm. Provisional, as recorded in the file.
- K7 is wired but not met on CI until the cache on `main` is saved again, which needs
  the `ToyRefitMean` entry point (outside Slice 2b).
- `ci_summary.py` now differs from the copy the other libraries carry; the template
  change is not propagated here.
- The K6 delta is a laptop measurement; the runner figures wait on two pilot
  dispatches.

## Core paths changed (K4)

The exact core paths that Slice 2b (R1 to R6, attributed here) changed after the S0 close-out, which the Slice 3a gate matches against this record:

- `src/spaceodyssey/core/battery/metamorphic.py`
- `src/spaceodyssey/core/battery/result.py`
- `src/spaceodyssey/core/battery/views.py`
- `src/spaceodyssey/core/conformance/__init__.py`
- `src/spaceodyssey/core/conformance/observer_kit.py`
- `src/spaceodyssey/core/preflight/checks.py`
- `src/spaceodyssey/core/preflight/findings.py`

Core tests that cover this change: `tests/core/test_epoch_relation.py`, `tests/core/test_golden_histories.py`, `tests/core/test_lints.py`, `tests/core/test_metamorphic.py`, `tests/core/test_observer_kit_corrections.py`, `tests/core/test_preflight.py`.
