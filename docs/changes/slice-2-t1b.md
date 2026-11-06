---
orphan: true
---

# Slice 2 T1b: the observer contract

T1b settles the methods and capabilities of `AbstractObserver` that the refresh
protocol (T4a), the observer kit (T3a) and the pedigree work (T4b) build on. Every
fold now receives the record's inference key, an observer declares the record kinds
and statuses it refits, and the registry lowers that declaration into `Capacities`.
The defaults keep every Slice 1 history unchanged: all 112 golden digests match.

## Commits

1. `feat(core): observer refit, summary and install methods; folds take their inference key`
2. `test(core): observer contract and fold-key invariance`
3. `docs(changes): record Slice 2 T1b observer contract`
4. `feat(core)!: observer fold and fold_summary take key=; keyed, padding-safe refit over refit_rows` (review fixes 1 to 4, and the release marker of fix 6)
5. `test(core): keyed promotion parity, analysis boundary, padding gradients and retraction`
6. `fix(core): refit_rows reads the log without a dynamic attribute read` (the
   Slice 1 gate's dispatch scan refused the first version's `getattr` over column
   names; the rows now come from the sorted log's leaves, and
   `test_canonical_fields.py` now requires `Record` and `RecordLog` to list their
   columns in one order, which that relies on)
7. `docs(changes): T1b review fixes, release notes and refit result size`

## The observer contract

| Method | Signature | Default |
|---|---|---|
| `capabilities` | `capabilities() -> ObserverCapabilities` | incremental only |
| `init` | `init(caps) -> belief` | zero `MeanBelief` |
| `init_summary` | `init_summary(caps) -> summary` | `init(caps)` |
| `layout` | `layout(caps) -> BeliefLayout` | `MeanBelief` layout of `init` |
| `fold` | `fold(belief, rec, caps, key=None) -> belief` | none (abstract) |
| `fold_summary` | `fold_summary(summary, rec, caps, key=None) -> summary` | `analyze(fold(summary, rec, caps, key=key))` |
| `summary` | `summary(belief) -> summary` | identity |
| `refit` | `refit(belief0, rows, caps, key, row_keys) -> belief` | folds each valid row in order with its fold key, analyzing between folds |
| `install` | `install(belief, fitted, subject) -> belief` | none (raises) |
| `install_summary` | `install_summary(summary, fitted, subject) -> summary` | `analyze(install(summary, fitted, subject))` |
| `analyze` | `analyze(belief) -> belief` | identity |

- **Analysis.** The full-belief methods (`fold`, `refit`, `install`) return the
  belief before analysis, and the caller applies `analyze` to the whole belief after
  each change, as the release handler already did after a fold. Analysis may couple
  subjects (the toy solve does), so an install of one subject must be followed by
  `analyze`. The summary methods (`fold_summary`, `install_summary`) return the
  finished summary: summary analysis is not a separate method. This keeps T1a's
  `fold_summary` unchanged.
- **Keys.** `fold` and `fold_summary` receive the same key, `keys.fold_key(root,
  rec, study)`: `draw_key` of the record's own identity (subject, channel,
  acquisition index, phase, revision, status) under `Purpose.INFERENCE`, with the
  study's fork branch and coupling, as a measurement draw has. No count, queue
  position or private event enters it (D3). The key is computed from the unmasked
  log row; masking changes no identity field. An observer that draws nothing
  ignores it.
- **`refit` takes `caps` and the row keys.** The spec table writes `refit(belief0,
  rows, key)`. The settled signature is `refit(belief0, rows, caps, key,
  row_keys)`. The default refit folds, and `fold` needs the capacities, as every
  other method that reads records does. `belief0` is the starting belief (normally
  `init(caps)`). `rows` is a `Record` with a leading row axis as
  `records.refit_rows` selects it (below). `row_keys` holds each row's fold key,
  `keys.fold_keys(root, rows, study)`: the world root key with the study's fork and
  coupling, exactly the keys the incremental folds of the same rows received, so a
  sampling observer refits bit for bit to its incremental belief over the same rows
  in the same order. `key` is the subject's refit key (T4b's `refit_key`), kept
  separate for an observer whose refit draws beyond its row folds; the default does
  not read it.
- **Refit analysis boundary.** `refit` returns before the final analysis, like
  `fold`, and the default analyzes between folds as the incremental path does, so
  `analyze(refit(...))` equals folding the rows one by one even when `analyze` is
  not idempotent.
- **Padding.** Before a row that does not count is folded (and the result
  discarded), the default replaces it with the first counting row marked invalid,
  so a fold that divides by a padding value cannot put a NaN into a gradient. A
  refit with no counting row returns `belief0`; its discarded fold then sees a
  padding row, so an observer whose fold is singular at a zero payload should not be
  differentiated through an empty refit.
- **Row selection.** `records.refit_rows(log, caps, audience, subject)` selects the
  rows: for each acquisition of the subject it takes the latest revision the
  audience has received (accepted by the observer, released to the policy) first,
  and that row counts only when its kind is in `caps.refit_kinds` and its status in
  `caps.refit_statuses`, so a correction with an unlisted status retracts the
  acquisition. Counting rows come first in canonical order (channel, acquisition
  index, revision) with `valid` true; the rest follow with `valid` false; all are
  masked for the audience. T4a truncates to `rows_per_subject` and T4b adds the
  pedigree; the selection rule is fixed here.
- **Order.** Refit rows are in canonical order, incremental folds in release order.
  The two agree for an order-insensitive observer; for an order-sensitive one
  (the sampling observer's last draw), promotion parity holds when the rows are
  given in release order, which is how the parity test feeds them. Whether the kit's
  promotion-parity check (T3a) uses canonical or release order is a T3a decision.
- **`install` and `summary`.** `install(belief, fitted, subject)` writes the
  subject's part of a refit belief into the current one; it has no default, since
  which leaves are per subject is the observer's knowledge. For the policy, D2's
  installation is `install_summary(pbelief, summary(fitted), subject)`.
- **Capabilities.** `ObserverCapabilities(incremental=True, refit=False,
  refit_kinds=(), refit_statuses=(COMPLETED, PARTIAL))`. It refuses an observer
  that is neither incremental nor refit-capable, `refit_kinds` that are empty when
  `refit` is true or present when it is false, a repeated kind and an unknown
  status. `refit_kinds` are registered record-kind names.
- **Refit result size.** The current contract: `refit` returns a belief of
  `belief0`'s full shape, every subject's leaves included, and `install` takes the
  subject's part. A boundary batched over audiences and subjects
  (`[2, n_subjects, ...]`) therefore holds `n_subjects` full beliefs per audience,
  quadratic in `n_subjects`. The open alternative is a per-subject slice type,
  declared in the layout, that `refit` returns and `install` consumes. T4a decides.
- **Projections** stay `eqx.Module` callables or module-level functions (answer 3);
  `BeliefLayout.project_summary` is kept (answer 2); every caller passes the layout
  (answer 4).

## Refit declarations

- `RecordKind` gains `derived: bool = False` (D5); T4b uses it to keep products out
  of folds.
- `Capacities` gains `kind_names`, `derived_kinds`, `refit_kinds` and
  `refit_statuses`, all empty by default. `KindRegistry.lower(caps, observer=None)`
  records the kind names and derived kinds and lowers the observer's refit kinds to
  their indices through `lower_refit`, which refuses a derived or unregistered name.
- `Study` refuses a refit declaration that names a derived or unregistered kind, or
  that the capacities do not hold (a registry lowered without the observer). It also
  refuses an observer that is not incremental, since no refresh schedule exists to
  change its belief; T4a lifts that refusal.
- Every Slice 1 observer is incremental only, so every Slice 1 study lowers to empty
  refit fields and builds as before.

## Changed call sites

- The release handler passes the key to both folds. `replay_belief(study, carry,
  trace, root=None)` passes the same keys when given the world's root key; without
  it the folds get no key, which only an observer that draws nothing accepts.
- `CountingObserver` delegates `capabilities`, `fold` and `fold_summary` with the
  key, `summary`, `refit`, `install` and `install_summary`. A refit or an install is
  not a fold: the counters of the belief are kept unchanged.
- The in-repository observers (`ToyRunningMean`, `CatalogObserver` and five test
  observers) accept `key`. `ToyRunningMean` gains `install` (mean and count of the
  subject; `aux` kept), which `ToySolveAnalysis` and `CatalogObserver` inherit.
- Toy sample observer: `ToySampleObserver` in `studies/toy.py`, whose full belief
  (`SampleBelief`) holds a posterior draw per subject and the standard-normal
  variate each fold drew, by subject and acquisition index, and whose summary is the
  running mean's `MeanBelief`. Its fold refuses a missing key.

## Public API

New in `spaceodyssey.core`: `ObserverCapabilities`, `fold_key`, `fold_keys`,
`lower_refit`, `refit_rows`; in `spaceodyssey.studies.toy`: `ToySampleObserver`,
`SampleBelief`, `project_sample`. `KindRegistry.lower` gains `observer`,
`RecordKind` gains `derived`, `Capacities` gains four fields, and `replay_belief`
gains `root`, all with defaults.

Breaking: the core now calls `fold` and `fold_summary` with `key=`, so an observer
subclass that overrides either without a `key` parameter raises `TypeError`.
Migration, for release notes:

```python
class MyObserver(AbstractObserver):
    def fold(self, belief, rec, caps, key=None):
        ...

    def fold_summary(self, summary, rec, caps, key=None):
        ...
```

An observer that draws nothing ignores `key`; one that calls `super().fold` passes
`key=key` on. A subclass that overrides `refit` takes `refit(self, belief0, rows,
caps, key, row_keys)`.

Release labelling: commit 4 carries the breaking marker, `feat(core)!:`. The
release-please configuration sets `bump-minor-pre-major`, so on 0.x the marker
bumps the minor version (0.1.0 to 0.2.0) and lists the change under breaking
changes, without graduating the library to 1.0.

## Tests

Added:

- `tests/core/test_observer_contract.py`: 7 functions, 13 unit tests. An
  inconsistent capability declaration is refused (four cases, with the consistent
  declaration as control); lowering stores the index and refuses derived and
  unregistered names (with the lowered index as control); `Study` refuses four bad
  declarations through its own check, with the lowered one as control; each fold
  draws from its record's inference key, not from the measurement
  key, and the policy summary equals `summary(belief)` (host executor); replay with the world key
  reproduces the keyed belief, and refuses without it; the default refit and
  `install` reproduce the incremental belief while masking padding rows, with a
  refit missing a row as control; the fold counter delegates every method, keeps
  the key and counts no refit or install.
- `tests/core/test_draws.py` (slow tier): `test_fold_key_with_n_dec_is_caught`, 2
  tests (host and compiled), the negative control of the fold-key invariance: a fold
  key folded with the decision count is caught, and moves no measurement draw; and
  `test_compiled_fold_variates_equal_the_host_variates`, which reuses the
  invariance runs to check that the compiled executor passes the same keys as the
  host (a first version ran this in the quick tier, at 11 s under load, over the
  unit budget).

Changed:

- `test_draws_invariant` (10 tests) runs both studies of each perturbation through
  `ToySampleObserver` and checks the fold variates beside the measurement variates:
  the `INFERENCE` invariance under queue order, an inserted wait, renumbered kinds,
  a cancel and re-observation, a padded handler table and a wider payload.
  `test_policy_draws_invariant` uses the same studies so the host programs are
  shared.
- The two source-patching mutants (`test_conformance_guards.py`) name the new fold
  call and assert it is present; five test observers accept `key`.

Added by the review fixes (`test_observer_contract.py`, unit tier, 5 tests): keyed
promotion parity of the sampling observer against the run's incremental belief,
bitwise, with padding before and between rows and the last row valid, and with a
reversed order and a dropped final row as controls; the analysis boundary under a
non-idempotent analysis; finite, exact gradients of a refit batched over both
audiences whose fold divides by padding values; the retraction rule of
`refit_rows` (a failed correction retracts, a completed one replaces, the policy
reads only what it received), two cases. Retired: the first refit test
(`test_default_refit_then_install_equals_the_incremental_belief`), which the
keyed parity test subsumes; the reviewer showed it passed with the final row
dropped or the rows reversed. Each fix was checked against a mutant: the original
refit, a refit that analyzes after its last fold, one without the padding stand-in,
one that drops the final counting row, one that reverses the rows, and a selection
without the current-revision rule each fail at least one of these tests.

Retired: one (above). Net: 17 quick tests, 3 slow tests.

The deciding tests `test_corpus_invariants`, `test_parity_values`,
`test_slice1_gate` and `test_golden_histories` pass with the golden manifest
unchanged.

## Time

Quick tier (`-m "not slow" -n 2 --dist loadfile`, local, four threads, a fresh
compile cache each run, junit wall time), run alternately on `ab85828` and this
branch under the same machine load (another session's suite and a photomancy job
were running):

| Run | Tests | Wall (s) | Summed (s) |
|---|---|---|---|
| `ab85828` | 1,701 | 102.0 | 183.7 |
| branch | 1,714 | 86.5 | 155.5 |
| `ab85828` | 1,701 | 83.1 | 148.1 |
| branch | 1,714 | 76.2 | 135.9 |

The difference is within run-to-run noise; the new quick file sums 5.5 s. No new
compiled signature enters the quick tier beyond the toy sample observer's host
programs. The slow `test_draws.py` now runs its perturbation studies through the
sampling observer, a different program from before but the same number of programs.

The full suite on the first two commits passed: 2,776 tests in 1,316 s wall at
`-n 2` (the machine was shared; T1a measured 686 s unloaded). The compiled fold
check then moved from the quick file to `test_draws.py`; both files pass on the
final branch (45 tests). The deciding tests and the golden manifest ran in that
full suite, with the manifest unchanged.

After rebasing onto `bd3ff31` (T2, Agenda levels; no conflicts), the deciding
tests, the golden manifest, the contract, draw, lint, layout, conformance-guard and
Agenda-level files pass: 663 tests, manifest unchanged.

After the review fixes: the deciding tests, the golden manifest (unchanged), the
contract, draw, lint, layout, conformance-guard, replay and revision files pass
(703 tests), with the gate's dispatch scan and its allowance test passing after
commit 6 (500 tests in the rerun of the gate, lint, contract and canonical-field
files).

Actual time spent: about 1 hour of wall time for the task and 45 minutes for the
review fixes, of which about 40 minutes and 20 minutes were spent
waiting on test runs.
