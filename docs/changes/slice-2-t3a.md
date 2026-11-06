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
| `check_declared_margins(observer)` | 4. an observer judged in a margin declares margin, seeds and standard error |

Helpers: `declared_margin(observer)`, `calibration_size(margin, n_slots)`,
`release_order(carry, trace, audience)`, `replay_beliefs(study, run, roots,
observer=None)` (the beliefs another observer would hold over the same records),
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
  the parity tolerances (1e-12), except that a `statistical` observer's floats need
  only be finite in both paths, and an `order_sensitive` observer's in the
  `refit_rows` path; each such path is then judged in the observer's margin. A
  margin is judged per stratum (the refit subject and how many of its rows the
  prefix holds): the trace-projection difference at the refit subject, averaged
  over the declared seeds within each world, gives a Student t interval over
  worlds with the larger of the observed and declared standard errors, which must
  lie inside the margin, over ten worlds at least. `install` is challenged: the
  release-order refit installed into a copy of the incremental belief with every
  numeric leaf shifted by one must give the refit subject the refit's projection
  and count leaf and leave every other subject's unchanged, before analysis. The
  kit's own incremental path is anchored to the run: after the last folded row it
  must equal the run's final belief. The refit key is `fold_in(fold_in(root, seed),
  subject)` until T4b defines `refit_key`.
- **Declared margins.** `AbstractObserver` gains five class attributes:
  `statistical` and `order_sensitive` (both False), `margin`, `seeds` and
  `standard_error`. They are `ClassVar`s, not fields, so no pytree, compiled program
  or history changes. An observer that is statistical or order-sensitive must
  declare all three, positive and finite, with distinct nonnegative seeds and a
  margin of at least four standard errors (the D11 sizing); an exact observer must
  declare none (a subclass that turns both flags off must also clear inherited
  declarations). `check_promotion_parity` refuses an observer that fails this.

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

The kit study (`tests/core/kit_helpers.py`) is the toy survey with four subjects
observed twice each on channels of one noise level, so a flat-prior Gaussian
posterior with that noise is exact; subjects 0 and 2 are released out of canonical
order, and the log holds exactly the eight records (`n_log = 8`). One compiled run
of the running mean over 2,000 worlds is the measurement batch, built in a module
fixture. Every other observer's beliefs come from `replay_beliefs` over that batch
(the plan reads no belief, so the records do not depend on the observer); the batch
is checked once against a replay of its own observer. The kit runs on
`ToyRunningMean` and `ToySampleObserver`, each declared refit-capable on
measurement rows by a subclass (the sampling one also declares itself
order-sensitive, with margin 0.1, seed 0 and standard error 0.02). The mutants are
two classes with array-valued faults, so each class compiles once.

`tests/core/test_observer_kit.py`, 23 quick tests (10 functions):

| Test | Cases | Fails if |
|---|---|---|
| `test_calibration_on_a_toy_prior` | mean, sample | the posterior is miscalibrated (the sampling observer is judged on its posterior draws under its declared noise) or worlds are dropped |
| `test_calibration_controls` | ignores half, NaN draws, wrong noise, four-point mass | calibration passes any of them |
| `test_calibration_slack_counts_worlds_not_pits` | 1 | the slack treats PITs of one world as independent, or too few worlds are accepted |
| `test_no_record_is_folded_twice` | mean, sample | `dup_fold` is set, or a second release changes a belief or goes unflagged |
| `test_folded_twice_control_the_dup_fold_mutants` | 1 | the planted `dup_fold` mutant of Slice 1, or a release handler without its first-acceptance guard, passes |
| `test_promotion_parity` | mean, sample | a promoted refit differs in either order, `install` misses or leaks, or an order changes |
| `test_promotion_parity_controls` | nine mutants | any passes (see the review fixes) |
| `test_declared_margins` | 1 | an exact observer needs a margin, a declared statistical observer is refused or judged outside its margin, or the t quantile is off |
| `test_declared_margins_control_missing_inconsistent_or_too_few_worlds` | 1 | an undeclared, undersized or inconsistent declaration is judged, or a margin over five worlds passes |

Measured: calibration distance 0.008 (mean) and 0.006 (sample draws) against a
slack of 0.036 and a margin of 0.075; the control that ignores half its records sits
at 0.083 and the wrong noise at 0.153. Statistical parity: the worst stratum's
`|estimate| + t * se` is 0.060 against the margin 0.1; the sampling observer's
`refit_rows` path 0.050.

Retired: none. Net: 23 quick tests (the spec listed about 8: each check runs on
both toy observers, and the review asked for a mutant per finding).

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

## Time

Quick tier (`-m "not slow" -n 2 --dist loadfile`, local, four threads, a fresh
compile cache each run, junit wall time), alternating `dbef55a` and the branch
after the review fixes, on a shared machine (the other compute slot was busy):

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

Actual time spent: about 3.5 hours for the task and 2.5 hours for the review fixes,
of which about 2 hours in all was spent waiting on timing and full-suite runs.
