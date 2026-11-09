---
orphan: true
---

# Slice 2b R5: surveys 18 and 4

R5 builds the two remaining Slice 2b surveys: survey 18 (exozodi-first, prescribed:
a star-level precursor chooses between two plans, after R1) and survey 4 (a precursor
exported from an upstream run, after R3, merged into this branch from `main` at
`751f658`).

## Commits

1. `feat(studies): survey 18, a star-level precursor that chooses between two plans`
   (`a90a85d`)
2. `test(studies): survey 18 assertions, one medium guard and the rest slow (K6)`
   (`55c459d`)
3. this record (`6a1c2bc`, with its commit list in `65ef971`)
4. `Merge branch 'main' into s2b-r5` (`536803f`), bringing R3 (`751f658`)
5. `test(studies): survey 18 evidence counts, the control on both executors, a host
   parity mutant` (`8966b0b`; review round 1)
6. `feat(studies): survey 4, a precursor exported from an upstream run and imported
   with its lineage` (`de6f07d`)
7. `test(studies): survey 4 assertions, a unit guard and the full runs slow (K6)`
   (`bc03094`)
8. this record's survey 4, review and checks sections

## What changed

- `studies/survey_18.py` (new): `STAR_LEVEL_KIND`, `survey_18_registry`,
  `Survey18Truth`, `draw_survey_18_truth`, `Survey18WorldSource`,
  `Survey18Photometry`, `Survey18Belief`, `Survey18Observer`, `Survey18Policy`,
  `Survey18Importer`, `survey_18_table` and `survey_18`.
- `tests/studies/test_survey_18.py` (new).
- Survey 4's files are listed in its own section below.
- No shared file changed: `studies/premission.py`, `studies/prior_observer.py`,
  `studies/toy.py`, `core/`, `codec/` and `tests/conftest.py` are untouched, and so
  are the golden and document hash pins.

### Survey 18 (D9)

- **World.** `Survey18Truth` holds the planets' signals and one exozodi level per star,
  the group latent shared by the star's planets (`STAR_OF = (0, 0, 1, 1)`: two stars
  of two planets). `Survey18WorldSource.draw` splits the world key it is given (the
  core derives it under `Purpose.WORLD`) into the signals and the levels. The toy
  truth, its draw and the toy photometry are untouched.
- **Channel.** `Survey18Photometry` returns `gain * signal[s] + coupling *
  level[star_of[s]] + noise * z`, so every measurement of a planet carries its star's
  level. `coupling` is an array field; at 0 the measurements carry none of it.
- **Precursor.** One raw row of the new kind `star_level` (payload: the measured
  level), filed under the target star's first planet, start and end before tick 0.
- **Importer.** `Survey18Importer` is the R1 importer with a `release_delay` per row:
  the row is written at tick 0 as R1 writes it, and its `available_tick` is the import
  tick plus the delay, so its release event and stamps come that many ticks later.
  The delay is relative to the import, so the epoch relation holds.
- **Observer.** `Survey18Observer` keeps planet running means (`mean`, `n`, folded from
  measurement rows) and a Gaussian posterior of each star's level in `aux[k]`, from a
  declared prior (mean 1, precision one row) updated by star-level rows only;
  `n_level` counts them. The summary has the belief's type and is folded the same way
  from the rows released to the policy, which is the core's default.
- **Policy.** `Survey18Policy` chooses at every decision between two plans by
  `view.belief.aux[target]` against the threshold 0.8: below, the target star's
  planets (the faint plan); otherwise the other star's. It observes the chosen plan's
  planet at the position of the measurements folded so far and drains once four are
  folded. The precursor values `FAINT = 0.2` and `BRIGHT = 1.8` give posteriors 0.6
  and 1.4; with no precursor the summary holds the prior, 1, and the bright plan.

### The survey 18 checks

A plan is what the policy did: each decision's tick, effect kinds and admission
codes, and each measurement's subject, channel, acquisition index and start tick
(payloads, which carry the world, are left out). Runs use pairs of worlds drawn from
their world keys, with the stars' levels set so the target star is faint (0.1) in one
world and bright (1.9) in the other. Each check compares plans across every pair of
worlds, so a plan that read the truth would differ there.

| Row | Check |
|---|---|
| (a) | withheld precursor (released after the campaign drained), payloads `FAINT` and `BRIGHT`: every plan of every world equal |
| (b) | released at tick 60: plans before tick 60 equal the withheld run's in every pair of worlds, and every world's plan differs after it |
| (c) | observer-only precursor: every plan equals every withheld plan; the observer folded it, the summary did not |
| (d) | the two worlds' levels swapped: with coupling 1 a released row differs and the plan is equal before the tick it is released at; with coupling 0 no released row differs and the whole plan is equal; the platform's `battery.truth_swap` also holds |
| (e) | released with `FAINT` and `BRIGHT`: the plans differ in every world, each payload gives the same plan on unrelated worlds (other keys, levels reversed), and the faint plan observes the target star's planets, the bright one the other star's |
| (f) | control: `TruthReadingPolicy` (the survey policy choosing by the world's true level, one world per run) fails (a), (b), (c), (d) at both couplings, and (e) |

## Tests

Tests added: 6, in `tests/studies/test_survey_18.py`; retired: none.

### Tiers (K6)

Every variant (payload, audience, release delay, coupling) is an array value of one
study shape, so the survey has one compiled program per executor. The merge-time
guard is (a), (c) and (e) on both executors, with pre-flight and host-compiled plan
agreement: it reaches the release delay, the audience split, the observer's level
posterior and the policy's choice in one program. Everything else is slow.

Cold times, each test alone in a fresh process, serially
(`SPACEODYSSEY_JAX_CACHE=off JAX_ENABLE_COMPILATION_CACHE=false`, four threads, wall
time including the about 1.5 s import), on a machine shared with other jobs:

| Test | Spec | Tier | Cold s |
|---|---|---|---|
| `test_the_precursor_changes_the_plan_only_when_released_to_the_policy` | 18 (a), (c), (e) | medium (guard) | 16.0 |
| `test_decisions_before_a_delayed_release_are_the_withheld_runs` | 18 (b) | slow | 14.6 |
| `test_a_swapped_group_latent_moves_the_plan_only_after_a_differing_release` | 18 (d) | slow | 16.6 |
| `test_the_epoch_relation_holds_with_released_and_delayed_precursors` | B6 | slow | 16.3 |
| `test_a_policy_reading_the_truth_latent_fails_rows_a_to_e` | 18 (f) | slow | 8.8 |
| `test_every_variant_passes_preflight_and_agrees_between_executors` | B7 executors | slow | 15.7 |

Quick-tier cold time survey 18 adds: 16.0 s of wall time (about 14.5 s without the
process import); 18.0 s when re-timed after review round 1, on the loaded machine,
with `--threads 4`. The medium tests of R1, R2, R4a and R3 also count toward the
Slice 2b allowance, which the slice already exceeds; R6 reconciles the total.

## Mutants and controls

Each mutant was applied to a copy of `src/` and the test file run against it
(`$TMPDIR` script, not committed). All 10 are killed.

| Mutant | Where | Failing tests |
|---|---|---|
| r1 release delay ignored (released at the import) | `Survey18Importer.handle` | guard, (b) |
| r2 delay counted from tick 0, not the import | `Survey18Importer.handle` | B6 |
| r3 every completed row updates the level | `Survey18Observer.fold` | guard |
| r4 plan choice reversed | `Survey18Policy.decide` | guard |
| r5 channel ignores the level | `Survey18Photometry.realize` | (d) |
| r6 level posterior ignores the declared prior | `Survey18Observer.fold` | guard |
| r7 plan chosen by a planet mean | `Survey18Policy.level` | guard |
| r8 every imported row public (audience ignored) | `PreMissionImporter.seed` (copy only) | guard |
| r9 a star-level row updates the other star | `Survey18Observer.fold` | guard |
| r10 the star-level kind not registered | `survey_18_registry` | guard, B7 executors (pre-flight `D006`) |

The guard kills every claim mutant except r2 (B6) and r5 (the channel, which only
(d) sees). Shipped control: `TruthReadingPolicy` (f). The executor-agreement half of
the B7 test is a parity check with no seeded mutant.

## Checks run

- `tests/core/test_golden_histories.py` with `tests/codec/test_run_document_runs.py`
  (golden pin and corpus document hash pin, `-n 2 --dist loadfile`): 189 passed in
  417.7 s. No pin file changed.
- Quick tier (`-m "not slow" -n 2 --dist loadfile`, deselecting
  `tests/codec/test_entry_points_installed.py` for the stale environment metadata
  noted in R0a): 2,126 passed in 372.4 s. The budget plugin reported three existing,
  untouched tests over budget on the loaded machine
  (`test_wake_coalescing.py::test_per_wake_rule_would_decide_twice_and_the_first_misses_a_record`
  5.4 s, `test_jaxedith_summary.py::test_detector_spec_and_report_refuse_undeclared_choices`
  5.2 s, `test_physical_ofti.py::test_executor_parity_on_one_certified_visit` 32.7 s);
  no survey 18 test.
- The touched file: `test_survey_18.py` 6 passed (each alone, above).
- `ruff check`, `ruff format --check`, `tools/check_internal_refs.py`,
  `tests/core/test_lints.py` with `tests/test_internal_refs_guard.py` (191 passed) and
  `pre-commit` on the touched files: clean. Every touched file is ASCII.
- The full suite and the Sphinx docs build were not run.

## Spec silences decided here

1. **Withheld means released after the campaign has drained.** The table contract
   has no audience that receives nothing, and a padded row would make (a) vacuous
   (nothing of it enters the run). A delay of `WITHHELD` (10^6 ticks) writes the row
   at tick 0 and releases it after the last decision, so a mechanism that leaked an
   unreleased row would show in (a).
2. **Delayed release is a survey-18 importer subclass** (`Survey18Importer`), not an
   option of the shared importer: the row is still written at tick 0 (it is
   pre-mission data), and only its `available_tick`, hence its release and stamps,
   moves. Survey 18 is the only consumer; `studies/premission.py` is untouched.
3. **The star-level row is filed under the target star's first planet.** A record
   has a subject and no star field; the observer maps a subject to its star through
   `star_of`.
4. **The level is learned from star-level rows only.** The measurements carry the
   level, but the toy observer does not infer it from them, so the plan depends on
   the world only through precursor rows. That is the prescribed variant: the
   optimized precursor is Slice 4.
5. **Checks compare plans across worlds.** On one world a truth-reading policy gives
   the same plan with any precursor, so (f) could not fail (a) to (c). Every check
   therefore compares across worlds whose target levels straddle the threshold. The
   levels are set in the fixtures, over drawn signals, because a draw does not
   guarantee straddling.
6. **(d) "with downstream released rows replayed identically"** is checked both ways:
   with coupling 0 the released rows are identical and the whole plan must hold;
   with coupling 1 they differ and the plan must hold before the first differing
   release (`first_truth_dependent_release`). The platform's `truth_swap` also runs.
7. **The truth-reading control holds the world's level as a policy field**, since a
   policy receives no truth; it therefore runs one world per run.
8. **A plan leaves payloads out**, since they carry the world; decisions, admission
   codes and the acquisitions made are the plan.

## Spec points not satisfied

- B7 at merge time covers survey 18 (a), (c) and (e); (b), (d), (f), B6 and the
  executor agreement pass in the slow tier only.
- The quick-tier allowance: Slice 2b already exceeds its about 60 s with the medium
  tests of R1 to R4b; survey 18's guard adds 16 to 18 s and survey 4 adds a unit
  test only. R6 reconciles the total.
- Survey 18 is not round-tripped through the codec (the importer subclass has no
  registered kind); B7 does not require it.
- A negative release delay is refused at construction (`ValueError`), not by a
  pre-flight finding.

## Review round 1 (survey 18)

The tier-1 review returned "fix first" on two P2s and asked for a parity mutant. Each
fix has a test that failed on a seeded mutant before it.

1. **Vacuous relations.** At coupling 0 `swap_check` accepted empty decision and
   acquisition projections, and B6 asserted only the count of rows before the
   start. `swap_check` now requires of every run its five decisions, four
   acquisitions and four folded measurements (`_missing_evidence`), and at least one
   decision before the first differing release; B6 requires the unshifted run's
   evidence, as many compared decision views as the run's decisions, and the five
   records of each world. Mutant r11 (the policy drains at its first decision) is
   killed by the guard, (b), (d), B6 and the control.
2. **The control on one executor.** `TruthReadingRunner` ran the compiled executor
   only; (f) now runs on both, still slow.
3. **Parity mutant.** r12 shifts, on the host executor only, the `available_tick` of
   a delayed precursor by one tick (`HostExecutor.advance` in a copy of `src/`); the
   executor-agreement test rejects it (also B6).

| Test | Tier | Cold s (after the fixes) |
|---|---|---|
| `test_the_precursor_changes_the_plan_only_when_released_to_the_policy` | medium (guard) | 18.0 |
| `test_a_swapped_group_latent_moves_the_plan_only_after_a_differing_release` | slow | 16.1 |
| `test_the_epoch_relation_holds_with_released_and_delayed_precursors` | slow | 41.2 |
| `test_a_policy_reading_the_truth_latent_fails_rows_a_to_e` | slow | 17.0 |

Timed alone, serially, caches off, `compute-run --threads 4`, on a machine shared with
other jobs (load average about 8). B6 grew because its evidence reads the unshifted
run, one more program per executor. The reviewer's one-thread cold run of the guard
took 42.75 s; at four threads, as CI's two workers see it, it is 18.0 s, well under
the 30 s medium budget, so it is unchanged.

## Survey 4

### What changed

- `studies/survey_4.py` (new): `survey_4_upstream`, `export_precursor`,
  `without_precursor`, `zero_information`, `LeastKnownPolicy`, `LeastKnownValue`,
  `survey_4` and `SURVEY_4_KINDS`.
- `tests/studies/test_survey_4.py` (new).
- No shared file changed: `studies/precursor.py`, `studies/premission.py`, `codec/`
  and `core/` are untouched.

### As built

- **Upstream.** The toy survey (`survey_4_upstream`) observes planets 0 and 1; planet
  1 twice. `export_precursor` calls R3's `export_records` with its measurement kind,
  the chosen audience (public by default), the upstream document's content hash,
  the subject map `{1: 0}` (the upstream planet 1 is the downstream target 0) and
  the producer map `{10: 60}`.
- **Downstream.** `survey_4(table)` is the toy survey with the pre-mission importer
  of the exported table and the toy running mean, which folds the imported
  measurement rows like its own. `LeastKnownPolicy` observes, at each of its first
  four decisions, the target (0 or 1) its summary holds the fewest measurements of,
  the first on a tie, then drains. `LeastKnownValue` is the observer's count for its
  least-known target. With `k` precursor rows of one target and four observations
  the value is `(4 + k) // 2` against 2 without, so the toy's sign is positive (+1
  for the two exported rows).
- **Variants of one shape.** `without_precursor` pads every row (nothing imported),
  `zero_information` makes every row an auxiliary row of value 0, and an export to
  the observer only differs from the public one in its audience column. All share
  one compiled program per executor.
- **Kinds.** `SURVEY_4_KINDS` registers the policy and the value for `dump_study`, so
  the downstream study is stored as a document and its lineage checked.

### Tests

Tests added: 4; retired: none. Cold times, each alone, serially, caches off,
`compute-run --threads 4`, wall time with the about 1.5 s import:

| Test | Spec | Tier | Cold s |
|---|---|---|---|
| `test_a_precursor_exports_imports_and_verifies_its_lineage` | 4 (d), (e), (f), the import half of (b), the first decision | unit (guard) | 4.5 (3.5 in the test) |
| `test_the_precursor_raises_the_value_by_its_known_amount` | 4 (a), pre-flight with the upstream document, executor agreement | slow | 18.5 |
| `test_the_precursor_moves_only_decisions_that_saw_it` | 4 (b), (c) | slow | 18.6 |
| `test_the_epoch_relation_holds_with_an_exported_precursor` | B6 | slow | 19.3 |

The guard compiles no executor: it exports a synthetic upstream log of five rows (a
correction, a datum released after the upstream horizon, a planet not exported),
stores the downstream study with its table in files and verifies the lineage
(right upstream, wrong upstream, a tampered file), handles the import's tick-0
events eagerly and compares the written rows with the table, and calls the
policy on its snapshot with and without the precursor. Quick-tier time survey 4 adds:
4.5 s of wall time. The slow tests share one real upstream run on the compiled
executor and the survey's runs (four variants, both executors).

### Mutants

Each mutant was applied to a copy of `src/` and the test file run against it
(`$TMPDIR` script, not committed). All 13 are killed.

| Mutant | Where | Failing tests |
|---|---|---|
| s1 the earliest released revision exported | `precursor.export_records` | guard |
| s2 a release after the upstream horizon counts | `precursor._released_to` | guard |
| s3 lineage always passes | `premission.lineage_findings` | guard |
| s4 an array file's sha256 not checked on read | `codec/files.py` | guard |
| s5 the acquisition index from the row position | `PreMissionImporter.handle` | guard |
| s6 the imported span from tick 0, not the import | `PreMissionImporter.handle` | B6 |
| s7 the policy alternates, ignoring its summary | `LeastKnownPolicy.decide` | guard, (a), (b) |
| s8 the value of the best-known target | `LeastKnownValue.evaluate` | (a) |
| s9 `without_precursor` imports every row | `survey_4.without_precursor` | guard, (a), (b) |
| s10 the zero-information rows keep kind and payload | `survey_4.zero_information` | (b) and (c) |
| s11 the export widens the audience to both | `precursor._table` | (b) and (c) |
| s12 offsets from tick 0, not the upstream horizon | `precursor._table` | guard, (a) |
| s13 `zero_information` pads every row (no import at all) | `survey_4.zero_information` | (b) and (c) (review round 1) |

s4 was first applied to the write-side check (`FileStore._intact`), where it is
inert; applied to the read-side check it is killed. s8 first survived: with two
precursor rows the final counts are equal, so a maximum and a minimum agree. Test
(a) now also compares the value with the least-known count read from the log on
the observer-only variant, whose counts are unequal (4 and 2).

### Spec silences decided here

1. **The guard is a unit test** (coordinator, 2026-11-07): export, store, lineage,
   tamper, import and first decision on a synthetic upstream log, with no executor
   compile. The paired difference needs whole runs and is slow.
2. **The value is the least-known target's measurement count**, a known-answer
   oracle (`(n_obs + k) // 2`), not the toy's threshold count, whose difference
   depends on the drawn signals.
3. **The upstream document is the upstream study's document**
   (`dump_study(survey_4_upstream())`); the upstream run itself is a toy run on the
   compiled executor (slow) or a synthetic log (guard).
4. **(b) "decisions whose snapshot showed a precursor row"** are the decisions at or
   after the first policy release of an imported row, read from the log. A public
   precursor is released at tick 0, so every decision sees it; the projection must
   then differ (counted). The observer-only export is the case no decision sees,
   and its projection must equal the run without the precursor.
5. **(c) zero information** is the exported table with every row an auxiliary row of
   value 0: present, public and imported, so a second initial decision would show,
   but folded as nothing by the toy observer and never read by the policy.
6. **(f) is checked on the synthetic log**, since the toy survey writes no
   correction; R3's export tests cover the same rule more widely.

### Spec points not satisfied

- B7 at merge time covers survey 4 (d), (e), (f) and the import half of (b); (a),
  (b), (c) and B6 on whole runs pass in the slow tier only.
- The full survey 4 (an RV channel on a second asset) is Slice 3, as the memo puts
  it.

## Checks run at the final tree (both surveys, review round 1)

- `tests/core/test_golden_histories.py` with `tests/codec/test_run_document_runs.py`
  (`-n 2 --dist loadfile`): 189 passed in 191.4 s. No pin file changed.
- Quick tier (`-m "not slow" -n 2 --dist loadfile`, entry-point test deselected): 2,172
  passed in 380.2 s. The budget plugin reported four existing, untouched unit tests
  over 5 s on the loaded machine (`test_reserved_fields.py` `freeze_ticks` 8.3 s,
  `test_coincident_order.py` fault-first 9.4 s, `test_profile.py` hash 6.1 s,
  `test_declarations_match_run.py` undeclared stop 7.6 s); none of R5's.
- `test_survey_4.py`, `test_survey_18.py`, `tests/core/test_lints.py` and
  `tests/test_internal_refs_guard.py`: 201 passed in 63.4 s.
- `ruff check`, `ruff format --check`, `tools/check_internal_refs.py` and `pre-commit`
  on the touched files: clean. Every touched file is ASCII.

## Review round 1 (survey 4)

The review reproduced s1, s5 and s8, confirmed the guard runs no executor, and found
one P2: the zero-information control compared projections only, so a
`zero_information` that imported nothing (the `without_precursor` table) passed.
Test (b) and (c) now first requires, in every world and on both executors, the two
imported auxiliary rows on the precursor channel, public, with `released_tick` and
`accepted_tick` at 0. Mutant s13 (`zero_information` returns the padded table) is
killed by it; s10 still is. Checks rerun: `test_survey_4.py` with
`tests/core/test_lints.py` and `tests/test_internal_refs_guard.py`, 195 passed in
23.1 s; ruff and `pre-commit` on the touched files clean. No source file changed, so
the golden and quick-tier results above stand.
