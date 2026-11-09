---
orphan: true
---

# Slice 2 T3a: the observer kit

The observer kit is a reusable set of checks that any observer runs through before
its refit is trusted. T3a adds its first four checks, each with negative controls:
calibration on a toy prior, a record folded twice, promotion parity and declared
margins. Every Slice 1 history is unchanged: all 112 golden digests match.

## Commits

1. `feat(core): observer kit with calibration, single-fold, promotion-parity and margin checks`
2. `test(core): observer kit checks on the toy observers with negative controls`
3. `docs(changes): record Slice 2 T3a observer kit`
4. `fix(core): kit judges margins per stratum with t intervals, challenges install, gates refit_rows order and bands calibration by world`
   (the review fixes below)
5. `test(core): kit review mutants on one replayed 2,000-world batch`
6. `fix(core): kit text avoids the denied core words` (the domain-word lint)
7. `docs(changes): T3a review fixes`
8. `fix(core): kit declares stochastic and subject-owned leaves, replays presented records and calibrates refits`
   (the second review fixes below)
9. `test(core): kit second-review mutants; run-based kit tests move to the slow tier`
10. `fix(core): kit validates declarations without numbers ABCs (dispatch scan)`
11. `docs(changes): T3a second review fixes`

## Where it lives

`spaceodyssey.core.conformance.observer_kit`, beside the corpus and its invariants,
and exported from `spaceodyssey.core.conformance`. It imports only the core. Every
check takes a batched `RunResult` (a leading world axis, as `CompiledExecutor.run`
and `HostExecutor.run_worlds` return it) and returns a `CheckResult`. An input the
check cannot judge is refused with `ValueError`.

| Function | Check |
|---|---|
| `check_calibration(run, truths, pit, *, margin)` | 1. the PITs are within `margin` of uniform (Kolmogorov distance) |
| `check_folded_once(study, run, truths, roots)` | 2. no world sets `dup_fold`; a second release of an accepted row is refused |
| `check_promotion_parity(study, run, roots, *, observer=None)` | 3. incremental belief equals the refit installed into it, in release order and through `refit_rows`; `install` writes its subject only |
| `check_declared_margins(observer)` | 4. an observer judged in a margin declares margin, seeds, standard error and stochastic leaves |
| `check_refit_calibration(study, run, truths, roots, pit, *, margin, observer=None)` | 1 applied to the beliefs a refresh installs, per declared seed |

Helpers: `declared_margin(observer)`, `calibration_size(margin, n_slots)`,
`refit_beliefs(study, run, roots, *, observer=None, seed=0)`,
`release_order(carry, trace, audience)`, `replay_beliefs(study, run, roots,
observer=None)` (the beliefs another observer would hold over the same records,
each row presented with the stamps it carried when the handler folded it; a log
with a revised acquisition is refused),
`t_quantile(df)` and `double_fold_failures(carry)`. The corpus invariant
`one_fold_per_acquisition` calls `double_fold_failures`, so the kit's flag check
runs on every corpus case on both executors through the existing corpus tests.

## The checks

- **Calibration.** `pit(belief, truth)` gives one posterior CDF value at a true
  value per PIT slot (a subject) and whether the slot counts; the kit maps it over
  the world axis of one batched run. It bounds the Kolmogorov distance between the
  true distribution of the pooled PITs and the uniform one by the empirical
  distance plus a slack, which must stay within `margin`. Slots of one world may
  share randomness, so the slack comes from worlds: by the Dvoretzky-Kiefer-Wolfowitz
  inequality each slot's empirical CDF over its N_j worlds is within
  sqrt(log(2 m / alpha) / (2 N_j)) of its true CDF for all m slots at once with
  probability 1 - alpha (alpha = 0.05), and the pooled empirical CDF is the
  N_j-weighted mean of the slot CDFs. The kit refuses a slack wider than half the
  margin; `calibration_size` gives the worlds needed.
- **Folded twice.** `dup_fold` must be clear in every world. The kit then hands
  the transaction's own `step` a second release of the first row of world 0 that
  both audiences received: both beliefs must be bitwise unchanged and `dup_fold`
  set.
- **Promotion parity.** For each world and each prefix of the rows the observer
  folded, the subject of the prefix's last row is refit and installed into the
  incremental belief after that prefix, then analyzed, twice: from the prefix's
  counting rows in release order (counting rows first), and from `refit_rows` of
  the prefix's log (canonical order, counting rows first, the rows a refresh
  installs). Integer leaves must match exactly in both. Floats must match within
  the parity tolerances (1e-12), except the observer's declared
  `stochastic_leaves`, which need only be finite in both paths for a `statistical`
  observer and in the `refit_rows` path for an `order_sensitive` one; each such
  path is then also judged in the observer's margin. Deterministic float state
  (the sampling observer's `mean`) is compared exactly on both paths. A
  margin is judged per stratum (the refit subject and how many of its rows the
  prefix holds): the trace-projection difference at the refit subject, averaged
  over the declared seeds within each world, gives a Student t interval over
  worlds with the larger of the observed and declared standard errors, which must
  lie inside the margin, over ten worlds at least. `install` is challenged: the
  release-order refit installed into a copy of the incremental belief with every
  boolean leaf inverted and every numeric value shifted enough to differ must
  give the refit subject the refit's values in
  every leaf the observer declares in `subject_leaves` (each with a leading
  subject axis), keep every other subject's values there, and keep every other
  leaf, before analysis. The
  kit's own incremental path is anchored to the run: after the last folded row it
  must equal the run's final belief. The refit key is `fold_in(fold_in(root, seed),
  subject)` until T4b defines `refit_key`.
- **Declared margins.** `AbstractObserver` gains seven class attributes:
  `statistical` and `order_sensitive` (both False), `margin`, `seeds`,
  `standard_error`, `stochastic_leaves` and `subject_leaves`. They are
  `ClassVar`s, not fields, so no pytree, compiled program or history changes. An
  observer that is statistical or order-sensitive must declare the first four,
  with a positive finite margin and standard error, distinct integer seeds in
  [0, 2^32) and a margin of at least four standard errors (the D11 sizing); an
  exact observer must declare none (a subclass that turns both flags off must also
  clear inherited declarations). A refit-capable observer declares
  `subject_leaves`; the toy observers do. `check_promotion_parity` refuses an
  observer that fails this.
- **Refit calibration.** Parity holds a statistical refit to the incremental
  belief in mean only, so `check_refit_calibration` holds its draws to the truth:
  the canonical refit at every promotion prefix is installed and analyzed under
  each declared seed. Only the refitted subject's PIT counts at that prefix, and
  each subject row depth must pass calibration separately, with the DKW slack
  built from worlds. `refit_beliefs` still gives the final-log refresh beliefs.

## Row order in promotion parity

Both orders are gated. Release order, the order the incremental folds ran, holds
every observer to its incremental belief exactly (or, if statistical, in its
margin), so it isolates whether `refit` and `fold` compute the same update even for
an order-sensitive sampler. `refit_rows` order is what a refresh installs: an
order-insensitive observer must match it exactly too, and an order-sensitive one
declares `order_sensitive` and is held to exact integers and its margin there. In
the kit study the running mean matches in both orders (`canonical_equal` true); the
toy sampling observer is declared order-sensitive, matches release order exactly
and `refit_rows` order within its margin (`canonical_equal` false). A refit that is
wrong only in canonical order now fails.

## Tests

(As of the second review fixes; the first version's single file is described by
the review sections below.)

The kit study (`tests/core/kit_helpers.py`) is the toy survey with four subjects
observed twice each on channels of one noise level, so a flat-prior Gaussian
posterior with that noise is exact; subjects 0 and 2 are released out of canonical
order, and the log holds exactly the eight records (`n_log = 8`). Two runs of the
running mean are the batches, each built in a module fixture: four worlds on the
host executor for the exact checks, and 2,000 compiled worlds for calibration and
margins. Every other observer's beliefs come from `replay_beliefs` over a batch
(the plan reads no belief, so the records do not depend on the observer); each
batch is checked once against a replay of its own observer. The kit runs on
`ToyRunningMean` and `ToySampleObserver`, each declared refit-capable on
measurement rows by a subclass (the sampling one also declares itself
order-sensitive, with margin 0.1, seed 0, standard error 0.02 and `sample` as its
stochastic leaf). The mutants are two classes with array-valued faults, so each
class compiles once; the running-mean batch itself runs the mean mutant with every
fault off, so its mutants share the batch's programs.

One executor compile alone is near 5 s cold, over the unit budget, so the tests
that need a run are in the slow tier (`SLOW_FILES`, with that reason). The quick
tier keeps the declarations and refusals, which need no run.

`tests/core/test_observer_kit.py`, 18 quick tests (6 functions), 1.2 s cold:

| Test | Cases | Fails if |
|---|---|---|
| `test_promotion_parity_refuses_undeclared_leaf_roles` | 4 | parity judges an observer with no `subject_leaves`, a subject leaf its belief lacks, stochastic leaves on an exact observer, or an integer stochastic leaf |
| `test_declared_margins` | 1 | an exact observer needs a margin, a declared observer is refused, or the t quantile is off |
| `test_declared_margins_controls` | 8 | a missing, undersized, inconsistent or stochastic-less declaration, seeds 0.5 and 1.5, 0 and 2^32, booleans, or a NaN margin is accepted |
| `test_margin_interval_at_its_boundary` | 1 | the interval uses 1.96 or the degrees of freedom of every seed of every world |
| `test_calibration_margin_must_be_positive_and_finite` | 4 | calibration, its sizing or refit calibration accepts a NaN, zero, negative or infinite margin |

`tests/core/test_observer_kit_runs.py`, 11 slow tests on the four host worlds:

| Test | Cases | Fails if |
|---|---|---|
| `test_no_record_is_folded_twice` | mean, sample | `dup_fold` is set, or a second release changes a belief or goes unflagged |
| `test_folded_twice_control_the_dup_fold_mutants` | 1 | the planted `dup_fold` mutant of Slice 1, or a release handler without its first-acceptance guard, passes |
| `test_replay_presents_records_as_the_handler_did` | 1 | a replay of an observer that reads `accepted_tick` differs from its real run, or a revised log is replayed |
| `test_promotion_parity_exact` | 1 | the running mean's promoted refit differs in either order, or `install` misses or leaks |
| `test_promotion_parity_controls` | 5 | late twice, no-op install (alone and beside late twice), canonical-only bug, or unanchored folds pass |
| `test_declared_margins_need_ten_worlds` | 1 | a margin is judged over four worlds |

`tests/core/test_observer_kit_statistics.py`, 19 slow tests on the 2,000 compiled
worlds:

| Test | Cases | Fails if |
|---|---|---|
| `test_calibration_on_a_toy_prior` | mean, sample | the posterior is miscalibrated (the sampling observer is judged on its posterior draws under its declared noise) or worlds are dropped |
| `test_calibration_controls` | 4 | ignores half, NaN draws, wrong noise or four-point mass passes |
| `test_calibration_slack_counts_worlds_not_pits` | 1 | the slack treats PITs of one world as independent, or too few worlds are accepted |
| `test_promotion_parity_order_sensitive` | 1 | the sampling observer differs in release order, in its deterministic leaves in canonical order, or outside its margin |
| `test_promotion_parity_statistical` | 1 | a refit that redraws from the refit key is judged outside its margin |
| `test_promotion_parity_controls` | 7 | late twice, NaN refit, cancelling errors, corrupt draw install, canonical-only mean bug, stale mean install or leaked mean install passes |
| `test_refit_calibration` | statistical, order-sensitive | the draws a refresh installs are miscalibrated under a declared seed |
| `test_refit_calibration_control_a_refit_twice_as_wide` | 1 | refit calibration passes draws of twice the posterior's width (which parity alone passes) |

Measured: calibration distance 0.008 (mean) and 0.006 (sample draws) against a
slack of 0.036 and a margin of 0.075; the control that ignores half its records sits
at 0.083 and the wrong noise at 0.153. Statistical parity: the worst stratum's
`|estimate| + t * se` is 0.060 against the margin 0.1; the sampling observer's
`refit_rows` path 0.050.

Retired: none. Net: 18 quick and 30 slow tests (the spec listed about 8: each
check runs on both toy observers, and two reviews asked for a mutant per finding).

Candidates for the slice's pruning pass (not deleted here; the rule asks for a
fresh-eyes reviewer): `test_revisions.py::test_a_second_release_of_one_row_is_counted_as_a_double_fold`
(slow tier) checks the duplicate-release refusal that `check_folded_once` now
checks on two observers with every belief leaf compared.

## Review fixes

A review of the first three commits found nine defects. Each is fixed, with a
mutant that passed the first version and fails now.

1. NaN refits passed statistical parity (a NaN interval compared false). Floats of
   a statistical observer must now be finite wherever the incremental belief is,
   and a non-finite stratum difference fails. Mutant: `SampleMutant(nan_refit)`.
2. Subject and prefix errors could cancel when pooled. The margin is judged per
   stratum (refit subject, rows of it in the prefix), seeds averaged within a
   world. Mutant: `SampleMutant(cancelling)`, +1 on even subjects and -1 on odd.
3. `install` was not established: a no-op install passed (even beside a doubled
   record), and a statistical install corrupting another subject passed. The
   install challenge above catches both. Mutants: `MeanMutant(noop_install)`, with
   and without `late_twice`; `SampleMutant(corrupt_install)`.
4. Two PIT summaries could not show uniformity (equal mass at 0.1, 0.4, 0.6, 0.9
   passed). Calibration now bounds the Kolmogorov distance with a DKW slack and is
   re-sized: margin 0.075, 2,000 worlds (1,805 needed). Control: the four-point PITs.
5. A refit broken only in canonical order passed, and the canonical rows lacked
   `refit_rows`' counting-rows-first compaction. The `refit_rows` path is now
   gated (exact, or in the margin for an order-sensitive observer) and uses
   `refit_rows` itself on the prefix's log; the release path compacts too. Mutant:
   `MeanMutant(canonical_bug)`, wrong when its first row is not its subject's first
   acquisition.
6. Calibration treated a world's correlated PITs as independent. The slack is now
   built from per-slot world counts. Test: four identical PITs per world pass, and
   the slack equals the per-world width.
7. A 1.96 standard-error interval had poor coverage at the two worlds then
   allowed. The interval uses a Student t quantile and needs ten worlds per
   stratum. Test: five worlds fail; `t_quantile(9)` is 2.262.
8. The sampling observer's calibration read only its mean and count under a
   hard-coded noise. It is now judged on its posterior draws under its declared
   noise. Controls: NaN draws (`SampleMutant(nan_sample)`) and a wrong declared
   noise (0.2 against 0.5).
9. Two full executor runs over-ran the unit budget. One run of the running mean is
   the batch, the other observers are replayed over it, `n_log` is 8, and the run
   is built in a module fixture: every test call is under 5 s.

The reviewer confirmed the subject selection, the install-then-analyze boundary
and that the class attributes stay outside the pytree.

Deciding tests: after the review fixes, on top of `dbef55a`, the full suite passes
through the compute launcher, 2,835 tests in 806 s wall at `-n 2 --dist loadfile`,
golden manifest unchanged (it runs in that suite). `ruff`, the domain-word and
core-import lints, the Slice 1 gate's dispatch scan and the internal-reference
check are clean. Before the review fixes the full suite had passed with 2,803
tests.

## Second review fixes

A second review found eight defects. Each is fixed with a mutant (or boundary case)
that passed the first fixed version and fails now; on that version 20 of the new
tests fail.

1. Canonical parity exempted every float leaf of an order-sensitive observer, so a
   refit adding 100 to its mean in canonical order only passed. Observers now
   declare `stochastic_leaves`; only those are loose, and deterministic float
   state is compared exactly on both paths. Mutant: `SampleMutant(canonical_mean)`.
2. The install challenge covered only the projection and the count, so a stale
   subject mean, or +100 on another subject's mean, passed. Observers declare
   `subject_leaves` (each with a leading subject axis); install must write the
   refit's values there at its subject, keep them at the others and keep every
   other leaf. Mutants: `SampleMutant(stale_install)`, `SampleMutant(leak_install)`.
3. `replay_beliefs` folded the final log rows, but the handler folds a row before
   it stamps `accepted_tick` and `released_tick`; an observer accumulating
   `accepted_tick` gave -8 in the run and 5,580 in the replay. The replay now
   presents each row with the stamps it carried at its fold, and refuses a log
   with a revised acquisition (whose earlier `superseded_by` the final log does not
   hold). Mutant: `MeanMutant(tick_aux)`, run on the host executor against its
   replay; a planted revision is refused.
4. Mean parity accepted a refit whose draws have twice the posterior's standard
   deviation (worst bound 0.0651 inside the 0.1 margin). `check_refit_calibration`
   now holds a refresh's draws to the truth per declared seed. Control:
   `SampleMutant(wide_refit)` passes parity and fails refit calibration (Kolmogorov
   distance 0.11 and 0.12 against the margin 0.075; the correct refits sit at 0.008
   to 0.010).
5. The interval tests survived a 1.96 multiplier and degrees of freedom taken from
   every seed. A boundary case through `_stratum_failures` (ten worlds, twenty
   seeds, difference 0.055, declared standard error 0.02) gives 0.1002 with nine
   degrees of freedom and fails; both mutants give under 0.0945 and pass.
6. The kit file's cold shared setup (6.0 s) was over the unit budget. One executor
   compile is near 5 s cold by itself (4.7 s for the host executor on four worlds,
   5.0 s for the compiled one), so no run fits a quick-tier test. The tests that need a
   run move to two slow files with that reason: exact checks on four host worlds,
   and the judged checks on the 2,000 compiled worlds. The quick file keeps the
   declarations and refusals and takes 1.2 s cold.
7. Seed declarations accepted (0.5, 1.5) as (0, 1) and (0, 2^32) as distinct keys
   that fold in identically. Seeds must be integers (not booleans) in [0, 2^32),
   distinct. Controls: both pairs, and (False, True).
8. A NaN calibration margin passed arbitrary PITs. Calibration, its sizing and
   refit calibration refuse a margin that is not positive and finite. Controls:
   NaN, zero, negative and infinite margins.

Measured with the fixes: the sampling observer's canonical-order bound 0.050, the
statistical refit's 0.060 in both orders, against the margin 0.1.

Deciding tests: on top of `8c4d307`, the full suite passes through the compute
launcher, 2,900 tests in 725 s wall at `-n 2 --dist loadfile` (the installed
entry-point test deselected in the worktree: it reads the main checkout's
metadata), golden manifest unchanged. `ruff`, the domain-word and core-import
lints, the Slice 1 gate's dispatch scan and the internal-reference check are clean.
The one budget report in that run is an existing hand-study test in `test_core.py`.

## Post-merge follow-ups

The install challenge now inverts boolean state and shifts floats enough to
change even large values, while integers still increase by one. The
`ReadyMean(stale_ready=True)` mutant leaves its subject-owned `ready` boolean
stale and fails with `install did not take the refit's ready`.

Refit calibration now checks every subject row depth exercised by canonical
promotion parity, including one-row prefixes, separately under each seed. It
keeps the world-clustered DKW slack. The
`SampleMutant(one_row_wide_refit=True)` mutant doubles the posterior draw standard
deviation only at one row, passes parity, and fails calibration with
`after 1 of its rows` and `from uniform`; correct refits pass at both depths.
Run-based regressions remain in the existing slow files, and the numeric
challenge regression needs no run.

## Time

Quick tier (`-m "not slow" -n 2 --dist loadfile`, local, four threads, a fresh
compile cache each run, junit wall time) after the second review fixes, on top of
`8c4d307`, alternating, on a shared machine with other compute jobs running:

| Run | Tests | Wall (s) | Summed (s) |
|---|---|---|---|
| `8c4d307` | 1,791 | 115.3 | 202.3 |
| branch | 1,809 | 107.8 | 192.3 |
| `8c4d307` | 1,791 | 114.3 | 206.0 |
| branch | 1,809 | 122.8 | 220.2 |

(Both branch runs carried the dispatch-scan failure fixed in commit 10, one fast
lint test.) The quick tier is flat within the machine's noise: the kit adds 18
tests that take 1.2 s cold together. The two slow files take 17 s (four host
worlds; the module fixture 5.8 to 7.3 s cold) and 35 s (2,000 compiled worlds; the
fixture 9.1 s under load), with no call over 5 s in either.

After the first review fixes, the same measurement alternating `dbef55a` and the
branch (the other compute slot was busy):

| Run | Tests | Wall (s) | Summed (s) |
|---|---|---|---|
| branch | 1,773 | 155.1 | 282.5 |
| `dbef55a` | 1,750 | 145.9 | 260.6 |
| branch | 1,773 | 162.4 | 298.6 |
| branch | 1,773 | 124.4 | 231.3 |
| `dbef55a` | 1,750 | 104.3 | 189.2 |

(The first two branch runs carried the domain-word failure fixed in commit 6;
two fast lint tests, no effect on time.) The branch adds about 10 to 20 s of wall
time and 20 to 40 s summed with a cold cache, on a noisy machine. The kit file
alone takes 33 s cold and 13.5 to 17 s with a warm persistent cache (peak 1.8 GB);
its cold cost is one compiled run of 2,000 worlds (about 7 to 8 s, in the module
fixture), the kit's vmapped promotion and replay programs for each observer class,
and the redelivery programs. No call in the file is over its 5 s unit budget; the
slowest calls are `test_promotion_parity_controls[statistical_late_twice]` (4.0 s
cold) and `test_promotion_parity[sample]` (3.5 s). CI runs warm (E0: 85 s against
the 540 s limit).

The first version (before the review) measured 77.0 s on `37fe6cc` against 90.0
and 90.1 s on the branch, with the file at 20.5 s cold and 9.4 s warm.

Actual time spent: about 3.5 hours for the task, 2.5 hours for the first review
fixes and 2.5 hours for the second, of which about 3 hours in all was spent waiting
on timing and full-suite runs.
