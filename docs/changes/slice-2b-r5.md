---
orphan: true
---

# Slice 2b R5: surveys 18 and 4

R5 builds the two remaining Slice 2b surveys: survey 18 (exozodi-first, prescribed:
a star-level precursor chooses between two plans, after R1) and survey 4 (a precursor
exported from an upstream run, after R3). Survey 18 is recorded here; survey 4 follows
once R3 is on `main`.

## Commits

1. `feat(studies): survey 18, a star-level precursor that chooses between two plans`
2. `test(studies): survey 18 assertions, one medium guard and the rest slow (K6)`
3. this record

## What changed

- `studies/survey_18.py` (new): `STAR_LEVEL_KIND`, `survey_18_registry`,
  `Survey18Truth`, `draw_survey_18_truth`, `Survey18WorldSource`,
  `Survey18Photometry`, `Survey18Belief`, `Survey18Observer`, `Survey18Policy`,
  `Survey18Importer`, `survey_18_table` and `survey_18`.
- `tests/studies/test_survey_18.py` (new).
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
process import). With R4b's 42.6 s, Slice 2b has added 58.6 s; survey 4's guard
would take it past the 60 s allowance.

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

- Survey 4 waits for R3 on `main`.
- B7 at merge time covers survey 18 (a), (c) and (e); (b), (d), (f), B6 and the
  executor agreement pass in the slow tier only.
- The quick-tier allowance: survey 18's guard brings Slice 2b to 58.6 s of the about
  60 s, so survey 4's guard will exceed it unless the allowance moves or a guard
  shrinks.
- Survey 18 is not round-tripped through the codec (the importer subclass has no
  registered kind); B7 does not require it.
- A negative release delay is refused at construction (`ValueError`), not by a
  pre-flight finding.
