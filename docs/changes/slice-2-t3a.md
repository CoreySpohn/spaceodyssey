---
orphan: true
---

# Slice 2 T3a: the observer kit

The observer kit is a reusable set of checks that any observer runs through before
its refit is trusted. T3a adds its first four checks, each with a negative control:
calibration on a toy prior, a record folded twice, promotion parity and declared
margins. Every Slice 1 history is unchanged: all 112 golden digests match.

## Commits

1. `feat(core): observer kit with calibration, single-fold, promotion-parity and margin checks`
2. `test(core): observer kit checks on the toy observers with negative controls`
3. `docs(changes): record Slice 2 T3a observer kit`

## Where it lives

`spaceodyssey.core.conformance.observer_kit`, beside the corpus and its invariants,
and exported from `spaceodyssey.core.conformance`. It imports only the core. Every
check takes a batched `RunResult` (a leading world axis, as `CompiledExecutor.run`
and `HostExecutor.run_worlds` return it) and returns a `CheckResult`. An input the
check cannot judge is refused with `ValueError`.

| Function | Check |
|---|---|
| `check_calibration(run, truths, pit, *, margin)` | 1. PITs uniform within `margin` |
| `check_folded_once(study, run, truths, roots)` | 2. no world sets `dup_fold`; a second release of an accepted row is refused |
| `check_promotion_parity(study, run, roots, *, observer=None)` | 3. incremental belief equals the refit installed into it |
| `check_declared_margins(observer)` | 4. a statistical observer declares margin, seeds and standard error |

Helpers: `declared_margin(observer)` (the `ParityMargin` of a statistical
observer, None for an exact one, `ValueError` when the declaration is missing or
inconsistent), `calibration_size(margin)` (the PITs a margin needs) and
`release_order(carry, trace)` (the log slots the observer folded, in fold order).
`double_fold_failures(carry)` reads the flag; the corpus invariant
`one_fold_per_acquisition` now calls it, so the kit's flag check runs on every
corpus case on both executors through the existing corpus tests.

## The checks

- **Calibration.** `pit(belief, truth)` gives each subject's posterior CDF at its
  true value and whether the subject has data; the kit maps it over the world
  axis of one batched run. Two statistics are compared with 1/2: the mean PIT (a
  biased posterior moves it) and the fraction of PITs in the central half [1/4,
  3/4] (a posterior too narrow or too wide moves it). Each passes when its 95
  percent interval lies inside 1/2 +/- `margin`. The kit refuses fewer PITs than
  N = (4 sigma / margin)^2 with sigma = 1/2.
- **Folded twice.** `dup_fold` must be clear in every world. The kit then hands
  the transaction's own `step` a second release of the first row of world 0 that
  both audiences received: both beliefs must be bitwise unchanged and `dup_fold`
  set.
- **Promotion parity.** For each world and each prefix of the rows the observer
  folded, the subject of the prefix's last row is refit from the prefix's counting
  rows (`caps.refit_kinds`, `caps.refit_statuses`) with their fold keys, installed
  into the incremental belief after that prefix, and analyzed. An exact observer
  must reproduce the incremental belief within the parity tolerances (integers
  exact, floats within 1e-12). A statistical observer must reproduce its integer
  leaves exactly, and the difference of its trace projection at the installed
  subject, averaged per world over the prefixes and its declared seeds, must have
  a 95 percent interval inside its margin, with the standard error taken as the
  larger of the observed and the declared one. The kit's own incremental path is
  anchored to the run: after the last folded row it must equal the run's final
  belief, so an observer whose folds differ from the run's is caught rather than
  judged against itself. The refit key is `fold_in(fold_in(root, seed), subject)`
  until T4b defines `refit_key`; the default refit does not read it.
- **Declared margins.** `AbstractObserver` gains four class attributes:
  `statistical` (False), `margin`, `seeds` and `standard_error`. They are
  `ClassVar`s, not fields, so no pytree, compiled program or history changes. A
  statistical observer must declare all three, positive and finite, with distinct
  nonnegative seeds and a margin of at least four standard errors (the D11
  sizing); an exact observer must declare none. `check_promotion_parity` refuses
  an observer that fails this.

## Row order in promotion parity

The refit of a prefix reads its rows in **release order**, the order the
incremental folds ran. An order-sensitive observer (the sampling observer keeps the
draw of its latest fold) then matches its incremental belief exactly, so the check
isolates whether `refit` and `fold` compute the same update; under canonical order
every sampler would have to be judged statistically, which weakens the check from
bitwise to a margin. `refit_rows` hands a refresh its rows in canonical order, so
the check also refits every prefix in canonical order and reports, without failing,
whether that agrees (`canonical_equal`). In the kit study the running mean agrees
in both orders and the sampling observer agrees only in release order; the
promotion-parity test asserts both, so a change of order in the check fails it.

## Tests

The kit study (`tests/core/kit_helpers.py`) is the toy survey with four subjects
observed twice each on channels of one noise level, so a flat-prior Gaussian
posterior with that noise is exact, and with subjects 0 and 2 released out of
canonical order. Each observer runs once, 400 worlds in one compiled `vmap` run,
cached for the process. The kit runs on `ToyRunningMean` and `ToySampleObserver`,
each declared refit-capable on measurement rows by a subclass. The running-mean
subclass also carries the calibration control as an array-valued switch,
`ignore_odd`, off by default (its fold is then the toy running mean's, bit for
bit), so the control shares the honest observer's compiled program. The kit
checks read only the run they are given; that the two executors produce the same
run is the Slice 1 gate's property, so the kit tests use the compiled executor
alone. A first version also ran the host executor and a separate control
observer; that cost 40 s of cold compile in this file against 20 s now.

Added, `tests/core/test_observer_kit.py`, 13 quick tests (8 functions):

| Test | Cases | Fails if |
|---|---|---|
| `test_calibration_on_a_toy_prior` | mean, sample | the posterior is miscalibrated or worlds are dropped (1,600 PITs; N = (4 x 0.5 / 0.07)^2 = 817 needed) |
| `test_calibration_control_an_observer_ignoring_half_the_records` | 1 | calibration passes an observer that counts records it does not use (central half 0.35), or accepts too few PITs |
| `test_no_record_is_folded_twice` | mean, sample | `dup_fold` is set in any of the 400 worlds, or a second release changes a belief or goes unflagged |
| `test_folded_twice_control_the_dup_fold_mutants` | 1 | the planted `dup_fold` mutant of Slice 1, or a release handler without its first-acceptance guard, passes |
| `test_promotion_parity_in_release_order` | mean, sample | a promoted refit differs from the incremental belief (8 worlds, 64 prefixes), or the check changes order |
| `test_promotion_parity_control_a_late_record_weighted_twice` | exact, statistical, unanchored | a refit that folds its latest record twice passes (exact, and statistical through its counts), or an observer whose folds do not reproduce the run is judged |
| `test_declared_margins` | 1 | an exact toy observer must declare a margin, or a declared statistical observer (refit draw from the refit key, 400 worlds, two seeds) is refused or judged outside its margin |
| `test_declared_margins_control_missing_or_inconsistent` | 1 | an observer without its declarations, one sized below four standard errors, or an exact one declaring a margin is judged |

Measured in the statistical case: projection difference 0.0084, observed
standard error 0.0107 against the declared 0.02, margin 0.08.

The spec listed about 8 tests; the three checks that run on both toy observers are
parametrized over them, and the parity control carries a third case for the
anchor. Retired: none. Net: 13 quick tests.

Candidates for the slice's pruning pass (not deleted here; the rule asks for a
fresh-eyes reviewer): `test_revisions.py::test_a_second_release_of_one_row_is_counted_as_a_double_fold`
(slow tier) checks the duplicate-release refusal that `check_folded_once` now
checks on two observers with every belief leaf compared.

Deciding tests: the full suite passed before the rebase onto `dbef55a` (T5a),
2,803 tests in 1,064 s wall at `-n 2 --dist loadfile`, through the compute
launcher, golden manifest unchanged. After the rebase (no conflicts; T5a touches
neither the core nor `tests/core`) the kit, corpus invariant, parity value, Slice 1
gate (dispatch scan included), golden manifest, lint, observer contract, invariant
mutant and codec files pass: 672 tests, manifest unchanged. `ruff`, the
domain-word and core-import lints and the internal-reference check are clean.

## Time

Quick tier (`-m "not slow" -n 2 --dist loadfile`, local, four threads, a fresh
compile cache each run, junit wall time), run alternately on `37fe6cc` and this
branch on a shared machine (another session held the second compute slot):

| Run | Tests | Wall (s) | Summed (s) |
|---|---|---|---|
| branch | 1,741 | 90.0 | 163.3 |
| `37fe6cc` | 1,728 | 77.0 | 137.3 |
| branch | 1,741 | 90.1 | 163.1 |

The branch adds about 13 s of wall time and 26 s summed with a cold compile
cache. The new file alone takes 20.5 s cold and 9.4 s with a warm persistent
cache (peak resident memory 1.4 to 1.9 GB); its cold cost is two compiled
programs (one per toy observer type, 400 worlds each, about 5.5 s apiece), the
kit's refit and redelivery programs, and the 400-world runs. CI runs warm (E0:
85 s against the 540 s limit), where the file's cost is the warm figure. An
earlier draft that also ran the host executor and a
separately compiled control observer took 40 s cold in this file and added 13 to
25 s of quick-tier wall time in two earlier alternating pairs (124 and 119 s on
`37fe6cc`, 150 and 133 s on that version, a busier machine).

The spec's compile list expected no new compiled signature for T3a. Calibration
needs one batched run per observer, and a 400-world batch is a new shape for any
existing program, so each toy observer type adds one compiled program.

Three slowest quick tests in the last branch run: `test_observer_kit.py::test_calibration_on_a_toy_prior[mean]`
(5.9 s call), `test_reserved_fields.py::test_reserved_ledger_field_roundtrips_through_both_executors`
(5.4 s setup), `test_observer_contract.py::test_each_fold_draws_from_its_records_inference_key`
(4.9 s setup).

Actual time spent: about 3.5 hours of wall time, of which about 1.5 hours was spent
waiting on timing and full-suite runs.
