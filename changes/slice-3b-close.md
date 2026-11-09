---
orphan: true
---

# Slice 3b close-out

Slice 3b is the realize context, bundles, the configuration timeline, asset ops state with
ops-dependent cost, world-drawn trajectory tables, and surveys 4 (full), 7, 12, 13, 19, 22 and
24. This record is the merge receipt: the acceptance checks run once, the kill rules judged,
the findings the builders left open, and the retirement of the Slice 3a gate. Nothing here is a
standing test.

## Slice 3a gate retired

The next slice has merged, so the Slice 3a gate goes, the way the earlier gates went. Its file,
`tests/test_slice3a_gate.py`, was an inventory: it ran named tests of the regular suite in a
child process and required each to pass with the stated case count, plus a few checks of its
own. It cost one worker about 22 minutes of the slow tier on the hosted runner (setup alone, 1348
s in one run) and about 6 minutes on a laptop. Every rule and item it listed is held by the
regular test it named, so the child run re-ran work the tiers already do.

| Gate check | Disposition |
| --- | --- |
| K1 golden histories, fingerprints, FAULT fixture, document hash pins (12 ids) | Covered: `tests/core/test_golden_histories.py`, `tests/codec/test_run_document_runs.py`, `tests/codec/test_premission_codec.py` |
| K1 pin files equal the blob of their capture commit; no commit after the capture touched a pin | Dropped as a merge-time process check (history walk); the goldens job and the pin tests judge the values |
| K1 mutant: asset-key change moves a pinned history | Covered in kind: `tests/core/test_metamorphic.py` (the `asset * n_assets` control) and `tests/studies/test_survey_19.py` (key and payload comparison). The pin-sensitivity form is dropped; its mutant code is deleted from `tests/core/mutants.py` |
| K1 mutant: an always-on empty leaf moves the pins | Covered: `tests/core/test_slice3_guard.py` (`test_control_an_open_gate_puts_the_empty_leaf_in_the_history`, `test_an_empty_conditional_view_leaf_is_not_hashed`). Mutant code deleted |
| K1 mutant: a changed `N_TIES` moves a pinned history | Dropped: it checked that exact digests move when the event order keys repack, which `test_a_one_bit_change_moves_the_digest` and the `N_TIES` assertion in `test_awards.py` already pin. Not rerun in this pass |
| K2 conservation and awards (17 ids), A1, A2, A3, A6 | Covered: `test_conservation.py`, `test_awards.py`, `test_refund.py`, `test_charging.py` |
| K3 step cost wiring, recorded ratio, extra-pass probe | Dropped as a merge-time check (receipt below). The one lasting piece, that the same-job comparison separates its limit, moved to `tests/test_ci_summary.py` (`test_the_step_cost_comparison_separates_its_limit`) |
| K4 attribution of every core commit; survey commits do not touch core | Dropped as a merge-time check (receipt below) |
| K5 domain-word and import lints | Covered: `tests/core/test_lints.py` |
| K6, K8 truth discipline and replay | Covered: `test_refund.py`, `test_survey_8.py`, `test_survey_10.py`, `test_allocator.py`, `test_core.py`, `test_audience_firewall.py` |
| K7 quick-tier CI step text, ceiling constants | Dropped as a process check; the summary script's limits stay tested in `tests/test_ci_summary.py` |
| A4, A5, A7, A8, A9 (frozen horizon, programs, balances, pre-flight, codec, surveys 8, 9, 10, 11) | Covered: the files the gate listed, all in the regular suite |
| A5 the twelve `stop_*` goldens are pinned | Moved: `tests/core/test_golden_histories.py::test_the_twelve_stop_cases_stay_pinned` (reads JSON, no compile) |
| Inventory, clean-child-environment and completeness meta checks | Dropped with the gate machinery |

Also removed: `tests/pin_helpers.py` (the gate's pin constants and git helpers, nothing else
used them), its entry in the test naming exclusion (committed first, alone), the gate's slow
tier file entry in `tests/conftest.py`, its program budget line in `pyproject.toml`, the
`skip_gate` handling and the `fetch-depth: 0` of the slow job in `ci.yml` (nothing else reads
history; `pilot.yml` keeps its own checkout of the base tree), and the unused `EVENT` variable
of that step. `pilot.yml` held no gate wiring. `tests/test_ci_summary.py` now defines its own
`REPO`.

## Acceptance, run once at the final commit

All through `compute-run` (4 threads, 8000 MiB), `PYTHONPATH=src`, never `-n auto`.

- Quick tier, `tests -m "not slow" -n 2 --dist loadfile`: 2018 passed, no skip, no budget
  violation, 327.5 s wall (5146 programs, peak 2.9 GB per process) while a golden run held the
  other slot. Quick ceiling is 540 s.
- Slow-marked and survey files, `-m ""`: surveys 4, 7, 12, 13, 19, 22, 24 with
  `test_conservation.py` and `test_ci_summary.py`: 44 passed, 80.7 s, no violation.
- Goldens: see K1.

| # | Check | Covering tests | Result |
| --- | --- | --- | --- |
| B1 | Realize context: version 1 unchanged, version 2 gets the context, delay moves the epoch, fork battery rebuilds it | `tests/core/test_realize_context.py` (8); version-1 program and histories by the goldens | passed |
| B3 | Bundles: whole admission and refusal, one transition, one generation, busy window, pre-flight refusals, cancel of any member | `tests/core/test_bundles.py` (15) | passed |
| B4 | Configuration timeline: stamping, commitments keep their admission config, traced leaf | `tests/core/test_config_timeline.py` (6) | passed |
| B5 | Ops state: transitions, charge at admission, Done and cancel from the stored state, conservation | `tests/core/test_ops_state.py` (9), `tests/core/test_conservation.py` (18) | passed |
| B6 | Trajectory sampling at the start tick, keyed by purpose, both executors, fork battery | `tests/core/test_world_tables.py` (6) | passed |
| B7 | New fields pass pre-flight and the codec round trip; corpus document hashes unchanged | `tests/core/test_preflight.py` (193), `tests/codec/test_preflight_json.py` (160), `tests/codec/test_optional_fields.py` (73), `tests/core/test_slice3_guard.py` (24), the hash pin test | passed |
| B8 | Surveys 4, 7, 12, 13, 19, 22, 24 | `tests/studies/test_survey_{4,7,12,13,19,22,24}.py` (5, 3, 3, 2, 3, 2, 2) | passed |
| B9 | No golden digest, fingerprint, FAULT fixture or corpus hash moved | `tests/core/test_golden_histories.py`, hash pin tests; no pin or manifest file differs from the base | passed |

## Kill rules

- **K1, earlier slices stay true.** `tests/core/test_golden_histories.py -m ""`: 125 passed
  (124 existing and the moved stop-case check), in 159 s and in 97 s, in two runs one after the
  other. A second pair of golden processes, started by mistake, overlapped these; the two result
  files each ended with a clean `125 passed`. The earlier whole-file failure of
  `test_history_matches_its_golden_digest[refresh-compiled]` (one builder, passing alone) did not
  reproduce: no failure in any of these runs. The history of that case is a bit-exact digest
  under the pinned jaxlib, so a failure means the history changed, not a tolerance. The builder
  run happened under machine load with other jobs, and the module clears JAX caches between
  tests and by a memory watermark, so a cache clear mid-file under memory pressure is the
  suspect; it is unconfirmed and left open. If it recurs, run the file alone and with `-x`
  and compare the first moved entry before touching the pin.
- **K2, conservation.** `tests/core/test_conservation.py`: 18 passed, including the two
  ops-priced checks added with the ops state.
- **K3, step cost.** Hosted-runner pilot, run 42718863863, one job, five interleaved rounds,
  this branch at e83e421 against fc48651: per-step minimum 126.62 us for the base and 126.60 us
  for this branch, ratio 1.00 against the 1.10 limit (within the limit); the extra-pass
  calibration mutant ratio is 1.57 in both trees, so the probe resolves a regression of that
  size; 13 handlers in both. No `core/` file changed after e83e421.
- **K4, no loop fork.** The only loop added is the member loop in `_cancel_lane`, bounded by the
  static `max_bundle_size` and absent from the program when bundles are off; the mutant with an
  unconditional member loop moves the `survey_1` `transition` jaxpr hash from `465cc8909391c8c9`
  to `04fb1eb9ceb3641c`. Receipt and measurements: `slice-3b-u3.md`. The later ops and
  configuration changes were checked the same way: the default `step` jaxpr is identical
  (`slice-3b-u4.md`, `slice-3b-u5.md`).
- **K5, a survey needs no core patch.** Commits since fc48651 that change `src/spaceodyssey/core/`,
  with the core test that covers each:

  | Commit | Change | Core test |
  | --- | --- | --- |
  | 045c050, 25ebf9d (main) | realize context | `test_realize_context.py` |
  | 27185b0, 82d0226, bd187a6, bf9c2b1 (main) | checkpoint and resume | `test_resume.py`, `test_resume_compiled.py`, `test_resume_kill.py`, `tests/codec/test_checkpoint_store.py` |
  | 8eb562b, 86bad72 (main) | legacy module removal, digest moved to codec | `tests/codec/test_digest.py`, `test_public_api.py`, the goldens |
  | 63f6361 (main) | world trajectory tables | `test_world_tables.py` |
  | feffafc, e36a08c | bundles | `test_bundles.py` |
  | c00ee4c, 1a718e6, c872a99 | configuration timeline | `test_config_timeline.py`, pre-flight cases |
  | 7b15ee6 | ops state and cost | `test_ops_state.py`, `test_conservation.py` |

  `git log --name-only fc48651..HEAD` shows eleven survey commits (`feat(studies)`,
  `test(studies)`, `style(studies)`: e7d081e, 1762a8a, 3a1b904, f3432cb, 1dc15af, 98e4490,
  69d6427, de8eb4c, 9c2fdb5, 5bb676c, e83e421) and none touches `core/`. K5 holds.
- **K6, truth discipline.** The controls of surveys 12, 22 and 24 fail as designed, each asserted
  in the slow tier: survey 12 `test_the_policy_acts_only_after_the_notice_is_released` (per the
  survey record: the doubled delay moves the action later by the same amount; the
  balance-reading policy acts before the release),
  survey 22 `test_a_finder_that_releases_the_position_at_tick_zero_breaks_the_invariance`, survey
  24 `test_no_view_changes_before_the_first_realization_and_one_does_after`. All passed in the
  run above.

## Known limits and findings

- Pre-flight merges bundle ordinals across decisions. A declaration has no decision index, but a
  run scopes a bundle id to one decision, so a policy that reuses an ordinal for different lanes
  in different decisions may be refused by pre-flight although the run admits both. Declare a
  distinct ordinal per bundle shape.
- The conservation check raises `NotImplementedError` for a completion the log refused when a
  per-state cost table is declared, because the trace holds no ops state to price it from. It
  does not guess.
- The snapshot fingerprint hashes integer content only. A leak that appears in float payloads
  (survey 22's control) is invisible to a fingerprint comparison, so the survey compares full
  views and documents that the fingerprints agree in the control.
- Scripted tests numbered by decision tick are brittle: a change to wake semantics shifts which
  tick a decision lands on and moves their expected values without a defect. Prefer expectations
  stated in terms of records and release ticks.
- A bundle is refused when any member starts after the decision tick (no bundled commitment).
- The `N_TIES` pin-sensitivity mutant and the pin capture-history walk are no longer run.

## Test work

Test-work share of the slice, `fc48651..e83e421` (before the retirement): 10517 of 22893
changed lines are under `tests/` (45.9 percent; 53.4 percent of tests plus `src/`). Of the 53
commits, 16 are `test:` (30 percent). Both figures are above the target of about 20 percent. Most of the test lines are feature tests that
shipped inside their cards (bundles, configuration, ops state, tables, context, checkpoint),
and survey tests are 861 lines. With the retirement (`fc48651..HEAD`) tests are 11929 of 24315
changed lines (49.1 percent) because the 1.4k deleted lines count as changed; by net lines the
slice removes 2539 test lines and 1590 `src` lines, because the legacy-module removal and the
gate retirement are larger than the additions.

## Test budget

- Quick tier: 327.5 s wall here under load for 2018 tests (the last full CI quick run failed
  only on one lint line, since fixed; its time is not recorded here). Ceiling 540 s.
- Slow tier on the runner, before this change: 38.9 to 50.5 min on three workers against a 45
  min cap, with the gate in it (its setup alone about 1348 s on one worker). Removing the gate
  takes about 22 min of worker time out of the dispatched and manual slow runs; the scheduled
  nightly run already skipped the gate, so the nightly figure does not move. The seven
  surveys add about 80 s on a laptop (about 3 to 4 min of worker time on the runner). The slow
  tier after retirement is estimated at 17 to 20 minutes of worker time less than a run with the
  gate, about 7 to 22 minutes less wall depending on how the files fall across the three workers;
  the next dispatched run will measure it.
- Slowest tests in the survey and conservation run: survey 4 value test 17.3 s,
  survey 4 epoch relation 7.7 s, survey 19 setup 6.4 s.
