# Quick tier timing and compile reuse

Measured on 2026-11-04 against baseline `ba9008f`. The final quick tier keeps
all 1,684 original checks and adds nine memory-guard regressions. No tests moved
between tiers, no assertions were removed or weakened, and no library code or
CI budget changed.

## Measurement method

Python 3.12.12, JAX/jaxlib 0.12.0.1, Equinox 0.13.8, pytest 9.1.1,
and hwoutils 1.10.1.dev24 on macOS 26.7.2 arm64, using the shared environment.
Each command runs exactly one file through the four-thread launcher. The
launcher and interpreter below stand for the provided shared tools:

```sh
PYTHONPATH=src <four-thread-launcher> --threads 4 -- <shared-python> -m pytest -q \
  -p no:cacheprovider <one-file> -m "not slow" --durations=25 --budget-report \
  --junitxml=<per-file-report>
```

Each pass starts with an empty persistent JAX cache, shared by that pass's
subsequent file processes. The before and final cache directories are separate.
No xdist workers or whole-suite command are used. The local budget comparison
uses the sum of launcher-to-exit elapsed times, including interpreter startup.
Pytest wall time is the plugin's session measurement; individual test times
below include setup, call, and teardown from JUnit.

Compile counts are the testbudget plugin's compile/load events, including
persistent cache loads; they are not counts of distinct XLA programs. RSS is
per-process peak resident memory. The plugin labels its values GB but divides
by `2**30`, so the tables correctly label those values GiB. The acceptance
ceiling is 6,000,000,000 bytes (about 5.588 GiB).

## Changes and coverage

- Compile the event helper's enqueue, selection, and handler dispatch together.
  Pass varying integer event values as int64 arrays so values share a program.
- Build the immutable bundle starting state once per core test module. Normalize
  the direct transition step index to the same dynamic scalar used elsewhere.
- Reuse queue, selection, log-writing, and snapshot jitted functions instead of
  creating fresh closures. Keep capacities and audience dispatch static.
- Release freed allocator pages before rechecking either memory watermark.
  macOS uses its existing pressure-relief API; Linux optionally uses
  `malloc_trim(0)`. Both fixed watermarks remain 3500 MiB RSS and 1750 MiB
  live heap, with RSS as the heap fallback. Cache clearing, garbage collection,
  and final allocator relief still run whenever either fresh reading is high.

A diagnostic baseline core run cleared caches 11 times; many late clears
occurred near 3500 MiB RSS with only 294-1022 MiB live heap. An optimized
diagnostic run cleared once. The committed guard also covers the Linux RSS
fallback at the lower watermark. Nine regression cases verify release/recheck
ordering, unchanged hard thresholds, and unavailable Linux trimming. The new
cases failed on the prior behavior and passed after the changes.

All 124 pre-existing assertion ASTs in the two refactored test modules are
unchanged. All original parameter cases, histories, worlds, and numerical
inputs remain. The independent code review found no correctness issues.

## Totals

| Measurement | Before | Final |
| --- | ---: | ---: |
| Files | 54 | 55 |
| Passed checks | 1684 | 1693 |
| Launcher-to-exit sum (s) | 354.49 | 317.65 |
| Pytest wall sum (s) | 276.5 | 240.4 |
| Compile/load events | 9268 | 7516 |
| Largest file peak RSS (GiB) | 3.84 | 3.44 |

## Per file results

Rows are ranked by baseline elapsed time. Values are before -> final; elapsed
includes launcher/startup overhead. The new guard file has no baseline.

| File under tests/ | Passed | Elapsed s | Pytest s | Peak RSS GiB | Compile/load events |
| --- | ---: | ---: | ---: | ---: | ---: |
| `core/test_core.py` | 39 | 84.90 -> 55.66 | 83.3 -> 54.4 | 3.84 -> 3.44 | 1998 -> 404 |
| `test_gaussian_profile.py` | 65 | 28.11 -> 29.17 | 26.4 -> 27.4 | 1.55 -> 1.55 | 1709 -> 1709 |
| `core/test_declarations_match_run.py` | 7 | 15.95 -> 15.40 | 14.1 -> 13.6 | 2.64 -> 2.68 | 212 -> 212 |
| `core/test_record_statuses.py` | 6 | 14.96 -> 14.89 | 12.9 -> 13.0 | 1.94 -> 1.93 | 274 -> 274 |
| `core/test_reserved_fields.py` | 15 | 14.93 -> 14.89 | 12.6 -> 12.9 | 2.80 -> 2.79 | 154 -> 154 |
| `core/test_lints.py` | 169 | 13.35 -> 13.36 | 11.9 -> 12.0 | 0.15 -> 0.15 | 0 -> 0 |
| `core/test_snapshot_fingerprint.py` | 4 | 12.84 -> 12.89 | 11.0 -> 11.1 | 2.34 -> 2.34 | 303 -> 303 |
| `core/test_stop_equivalence.py` | 4 | 11.35 -> 5.18 | 9.7 -> 3.5 | 2.84 -> 1.56 | 153 -> 67 |
| `studies/test_survey_1.py` | 1 | 9.85 -> 9.84 | 8.4 -> 8.3 | 1.16 -> 1.14 | 122 -> 122 |
| `test_viz_plots.py` | 138 | 8.79 -> 8.77 | 7.4 -> 7.2 | 0.45 -> 0.47 | 155 -> 155 |
| `core/test_coincident_order.py` | 8 | 7.80 -> 7.23 | 6.0 -> 5.9 | 0.95 -> 0.95 | 99 -> 99 |
| `core/test_wake_coalescing.py` | 3 | 7.29 -> 7.29 | 5.5 -> 5.6 | 1.45 -> 1.41 | 205 -> 205 |
| `test_gaussian_rectangle.py` | 10 | 6.25 -> 5.73 | 4.7 -> 4.5 | 0.45 -> 0.45 | 407 -> 407 |
| `core/test_schema.py` | 28 | 5.77 -> 3.71 | 4.2 -> 2.1 | 0.57 -> 0.36 | 222 -> 151 |
| `test_habitable_zone_parity.py` | 5 | 5.25 -> 4.70 | 3.5 -> 3.3 | 0.50 -> 0.50 | 43 -> 43 |
| `test_toy_certifier.py` | 30 | 5.24 -> 5.23 | 3.7 -> 3.8 | 0.45 -> 0.45 | 736 -> 736 |
| `test_analytic.py` | 26 | 5.23 -> 5.22 | 3.9 -> 4.0 | 0.39 -> 0.39 | 407 -> 407 |
| `test_inputs.py` | 339 | 5.22 -> 5.16 | 4.1 -> 4.1 | 0.27 -> 0.27 | 54 -> 54 |
| `test_profile.py` | 69 | 4.70 -> 5.23 | 3.6 -> 3.7 | 0.27 -> 0.27 | 54 -> 54 |
| `test_viz_package.py` | 12 | 4.70 -> 4.74 | 3.2 -> 3.2 | 0.23 -> 0.23 | 2 -> 2 |
| `test_truth.py` | 33 | 4.20 -> 4.22 | 2.8 -> 2.9 | 0.41 -> 0.41 | 404 -> 404 |
| `test_engines.py` | 11 | 3.69 -> 3.71 | 2.3 -> 2.4 | 0.27 -> 0.27 | 74 -> 74 |
| `core/test_keys.py` | 31 | 3.65 -> 3.70 | 2.2 -> 2.2 | 0.38 -> 0.38 | 57 -> 57 |
| `test_replicates.py` | 46 | 3.18 -> 3.18 | 1.8 -> 1.8 | 0.33 -> 0.33 | 175 -> 175 |
| `test_dist.py` | 7 | 2.68 -> 2.66 | 1.4 -> 1.3 | 0.31 -> 0.31 | 80 -> 80 |
| `test_viz_extract.py` | 144 | 2.68 -> 2.67 | 1.3 -> 1.3 | 0.29 -> 0.29 | 146 -> 146 |
| `test_plan.py` | 65 | 2.67 -> 2.69 | 1.3 -> 1.3 | 0.30 -> 0.30 | 162 -> 162 |
| `test_profile_certificate.py` | 31 | 2.67 -> 2.68 | 1.2 -> 1.2 | 0.31 -> 0.31 | 306 -> 306 |
| `core/test_preflight.py` | 60 | 2.65 -> 2.69 | 1.2 -> 1.2 | 0.27 -> 0.27 | 94 -> 94 |
| `test_claim_outcome.py` | 57 | 2.63 -> 2.70 | 1.3 -> 1.3 | 0.28 -> 0.28 | 90 -> 90 |
| `test_public_api.py` | 8 | 2.19 -> 2.15 | 0.8 -> 0.8 | 0.23 -> 0.23 | 2 -> 2 |
| `test_belief.py` | 5 | 2.17 -> 2.17 | 0.8 -> 0.8 | 0.25 -> 0.25 | 10 -> 10 |
| `test_context.py` | 3 | 2.17 -> 2.13 | 0.8 -> 0.8 | 0.26 -> 0.26 | 18 -> 18 |
| `test_data.py` | 5 | 2.17 -> 2.15 | 0.9 -> 1.0 | 0.26 -> 0.26 | 37 -> 37 |
| `test_loop.py` | 11 | 2.17 -> 2.15 | 0.9 -> 1.0 | 0.24 -> 0.24 | 86 -> 86 |
| `test_package.py` | 1 | 2.17 -> 2.17 | 0.8 -> 0.8 | 0.23 -> 0.23 | 2 -> 2 |
| `test_policy.py` | 1 | 2.17 -> 2.15 | 0.8 -> 0.8 | 0.23 -> 0.23 | 2 -> 2 |
| `test_qualname_aliases.py` | 67 | 2.17 -> 2.16 | 0.8 -> 0.9 | 0.23 -> 0.23 | 2 -> 2 |
| `test_rng.py` | 4 | 2.17 -> 2.17 | 0.8 -> 0.8 | 0.23 -> 0.23 | 14 -> 14 |
| `test_universe.py` | 3 | 2.17 -> 2.15 | 0.8 -> 0.8 | 0.23 -> 0.23 | 6 -> 6 |
| `test_wiring.py` | 12 | 2.17 -> 2.18 | 0.8 -> 0.8 | 0.23 -> 0.23 | 2 -> 2 |
| `core/test_stop_grid.py` | 1 | 2.16 -> 2.18 | 0.8 -> 0.8 | 0.23 -> 0.23 | 4 -> 4 |
| `studies/test_toy.py` | 1 | 2.16 -> 2.17 | 1.0 -> 1.0 | 0.28 -> 0.28 | 43 -> 43 |
| `test_build.py` | 10 | 2.16 -> 2.15 | 0.8 -> 0.8 | 0.23 -> 0.23 | 2 -> 2 |
| `test_io.py` | 7 | 2.16 -> 2.17 | 0.8 -> 0.8 | 0.23 -> 0.23 | 2 -> 2 |
| `test_registry.py` | 11 | 2.16 -> 2.11 | 0.9 -> 0.9 | 0.23 -> 0.23 | 2 -> 2 |
| `core/test_layout.py` | 1 | 2.15 -> 2.19 | 0.9 -> 0.9 | 0.23 -> 0.23 | 2 -> 2 |
| `core/test_parity_values.py` | 19 | 2.15 -> 2.16 | 0.8 -> 0.8 | 0.23 -> 0.23 | 2 -> 2 |
| `test_account.py` | 9 | 2.15 -> 2.17 | 0.9 -> 0.9 | 0.27 -> 0.27 | 52 -> 52 |
| `test_seams.py` | 3 | 2.15 -> 2.17 | 0.8 -> 0.8 | 0.23 -> 0.23 | 2 -> 2 |
| `test_certify.py` | 11 | 2.14 -> 2.16 | 0.8 -> 0.8 | 0.25 -> 0.25 | 13 -> 13 |
| `test_spec.py` | 13 | 2.13 -> 2.18 | 0.8 -> 0.8 | 0.23 -> 0.23 | 2 -> 2 |
| `core/test_record_kinds.py` | 11 | 2.11 -> 2.16 | 1.0 -> 1.0 | 0.27 -> 0.27 | 64 -> 63 |
| `test_internal_refs_guard.py` | 4 | 1.64 -> 1.64 | 0.3 -> 0.3 | 0.14 -> 0.14 | 0 -> 0 |
| `test_memory_budget.py` | 9 | new -> 1.12 | new -> 0.0 | new -> 0.14 | new -> 0 |

## File compile ranking

Highest ten file totals, ranked independently for each pass.

| Rank | Baseline file | Events | Final file | Events |
| ---: | --- | ---: | --- | ---: |
| 1 | `core/test_core.py` | 1998 | `test_gaussian_profile.py` | 1709 |
| 2 | `test_gaussian_profile.py` | 1709 | `test_toy_certifier.py` | 736 |
| 3 | `test_toy_certifier.py` | 736 | `test_analytic.py` | 407 |
| 4 | `test_analytic.py` | 407 | `test_gaussian_rectangle.py` | 407 |
| 5 | `test_gaussian_rectangle.py` | 407 | `core/test_core.py` | 404 |
| 6 | `test_truth.py` | 404 | `test_truth.py` | 404 |
| 7 | `test_profile_certificate.py` | 306 | `test_profile_certificate.py` | 306 |
| 8 | `core/test_snapshot_fingerprint.py` | 303 | `core/test_snapshot_fingerprint.py` | 303 |
| 9 | `core/test_record_statuses.py` | 274 | `core/test_record_statuses.py` | 274 |
| 10 | `core/test_schema.py` | 222 | `core/test_declarations_match_run.py` | 212 |

## Slowest individual tests

Ranked by baseline JUnit time, including setup and teardown. Shared-fixture
setup is charged to its first requesting test.

| Test | Before s | Final s |
| --- | ---: | ---: |
| `core.test_core::test_every_hand_study_reproduces_its_history` | 14.362 | 13.743 |
| `core.test_core::test_threshold_stop_policy_drains_after_threshold` | 8.326 | 8.529 |
| `studies.test_survey_1::test_survey_1_hand_count_and_executor_parity` | 7.559 | 7.462 |
| `core.test_core::test_phase_switch_changes_policy_at_boundary` | 7.183 | 7.993 |
| `core.test_record_statuses::test_statuses_agree_between_executors_and_replay` | 6.545 | 6.565 |
| `core.test_snapshot_fingerprint::test_snapshot_fingerprint_recorded` | 5.551 | 5.583 |
| `core.test_reserved_fields::test_reserved_ledger_field_roundtrips_through_both_executors[Ledger.program_charge_rule]` | 5.071 | 5.165 |
| `core.test_snapshot_fingerprint::test_the_fingerprint_hashes_the_masked_view_not_the_raw_log` | 4.496 | 4.635 |
| `core.test_core::test_gain_enters_measurement` | 4.303 | 4.437 |
| `core.test_coincident_order::test_fault_first_fails_the_acquisition_and_the_settlement_goes_stale` | 4.235 | 4.211 |
| `core.test_core::test_close_scope_membership` | 4.159 | 0.019 |
| `core.test_record_statuses::test_every_status_is_logged_and_only_completed_values_are_folded` | 3.520 | 3.668 |
| `core.test_core::test_cancel_bumps_lane_generation` | 3.492 | 0.006 |
| `core.test_core::test_stale_completion_settles_nothing` | 3.474 | 2.468 |
| `core.test_stop_equivalence::test_stop_equivalence_below_campaign[0]` | 3.374 | 2.571 |
| `core.test_core::test_done_stops_all_writes` | 3.339 | 0.006 |
| `core.test_reserved_fields::test_reserved_table_field_has_no_behaviour[AssetTable.freeze_ticks]` | 3.323 | 3.374 |
| `core.test_declarations_match_run::test_the_plan_declares_the_stop_it_emits` | 3.298 | 3.222 |
| `core.test_reserved_fields::test_reserved_table_field_has_no_behaviour[AssetTable.disruptive_resource]` | 3.221 | 3.332 |
| `core.test_core::test_partial_draw_key_ignores_status` | 2.697 | 0.014 |
| `core.test_core::test_cancel_partial_joins_observer_not_belief_mean` | 2.686 | 2.488 |
| `core.test_core::test_cancelled_occupancy_never_completes` | 2.684 | 0.042 |
| `core.test_core::test_abort_partial_is_never_released_or_folded` | 2.673 | 0.857 |
| `core.test_core::test_zero_latency_cancel_does_not_violate` | 2.669 | 2.574 |
| `core.test_core::test_zero_latency_abort_does_not_violate` | 2.636 | 2.458 |

## Baseline highest per test compile counts

Global top five, obtained by merging the plugin's five highest counts per file.

| Test | Compile/load events |
| --- | ---: |
| `test_gaussian_profile.py::test_flat_threshold_matches_toy` | 346 |
| `core/test_snapshot_fingerprint.py::test_snapshot_fingerprint_recorded` | 231 |
| `core/test_wake_coalescing.py::test_two_releases_one_tick_one_decision` | 196 |
| `test_gaussian_profile.py::test_orthogonal_float32_design_uses_closed_form` | 181 |
| `test_gaussian_rectangle.py::test_rectangle_reference_and_monte_carlo[1.2-0.0]` | 159 |

## Final highest per test compile counts

Global top five, obtained by merging the plugin's five highest counts per file.

| Test | Compile/load events |
| --- | ---: |
| `test_gaussian_profile.py::test_flat_threshold_matches_toy` | 346 |
| `core/test_snapshot_fingerprint.py::test_snapshot_fingerprint_recorded` | 231 |
| `core/test_wake_coalescing.py::test_two_releases_one_tick_one_decision` | 196 |
| `test_gaussian_profile.py::test_orthogonal_float32_design_uses_closed_form` | 181 |
| `test_gaussian_rectangle.py::test_rectangle_reference_and_monte_carlo[1.2-0.0]` | 159 |

## Validation and limits

Every listed final file passed, and every measured process stayed below the
6 GB ceiling. Ruff check and format check pass on all touched Python files.
The two slow-tier tests that call the changed event helper also passed in
separate fresh-cache processes, selected with `-k` and without excluding slow:

| Slow-tier caller | Passed | Elapsed s | Pytest s | Peak RSS GiB |
| --- | ---: | ---: | ---: | ---: |
| `tests/core/test_acquisition_index.py::test_cancel_keeps_its_index` | 1 | 11.83 | 10.0 | 1.35 |
| `tests/core/test_revisions.py::test_a_second_release_of_one_row_is_counted_as_a_double_fold` | 1 | 9.79 | 8.3 | 1.08 |

No tests or files moved to slow. The existing slow selection still contains
every previously slow check. The full slow tier was not rerun for this quick-tier
performance change.

These are local isolated-file measurements. The unchanged CI summary still
enforces 600 seconds on its single-process quick job. No new Ubuntu CI run was
triggered; its actual wall time remains to be verified after integration.
