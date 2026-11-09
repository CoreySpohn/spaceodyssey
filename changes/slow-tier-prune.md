---
orphan: true
---

# Slow-tier prune (U0b)

The slow tier took 64 minutes on the runner in the last measured dispatched run, 53
minutes without the slice 3a gate, against a 45 minute budget. This change shrinks or
merges the most expensive slow tests where the named failure stays detectable. No test
moved to a less frequent tier, no file under `src/`, no golden and no workflow changed.
The shared helper `tests/core/gradients.py` changed, so the merge needs a full suite.

Times are the Mac (cold: `SPACEODYSSEY_JAX_CACHE=off`, one file, `compute-run`) before
and after; the runner figure is the last dispatched run scaled by the Mac ratio.

## Changes

| Change | Mac before, after | Runner before, after (estimate) |
| --- | --- | --- |
| `test_gradients.py::test_d3_per_world_gap_monotone_in_width` | 36 s, 19 s | 98 s, about 50 s |
| `test_physical_ofti.py::test_five_refreshes...` now `test_three_refreshes...` | 57 s, 19 s | 111 s, about 37 s |
| `test_physical_ofti.py::test_survey_posterior_contraction...` | 73 s, 41 s | 156 s, about 90 s |
| `test_reporting_law.py::test_survey_detection_and_null_sequence...` | 40 s, 32 s | 78 s, about 62 s |
| `test_refresh_runs.py::test_a_budget_that_ends_on_the_hold...` | not measured, about 25 s less | about 55 s less |
| `test_host_service.py::test_host_service_parity` | about 4 s less | about 14 s less |

### Gradient widths are arrays

`d3_pathwise_and_analytic` and `d1_reverse_and_fd` passed each smoothing width as a
Python float into an `eqx.filter_jit` function, which treats it as static, so each of
the four widths compiled the per-world gradient and the gap program again. The widths
are now float64 arrays, so one program serves every width. Still catches: a per-world
gap that is not monotone in the width, and a reverse-mode gradient that disagrees with
central differences (the assertions and sizes are unchanged).

### Five refreshes become three

The parity test ran ten OFTI fits (five visits on each executor); the fits of the
fourth and fifth visit cost most of it. It now runs the first three visits. The
contraction over all five visits stays in the survey test. Still catches: a refresh
that does not install particles, a record epoch that is not its science midpoint, and a
compiled executor that installs other particles than the host executor, on visits one
to three. Dropped: parity on visits four and five, where the fit and the install path
are the same code as on visit three.

### Survey contraction with 32 particles

The fit of the fifth visit costs 43 s on the Mac and scales with the particle count.
The survey test now uses 32 particles. At 64 and at 32 the spread after five visits is
0.26 of the spread after one (threshold 0.4), every visit narrows it, and the true track
stays within 4 predictive standard deviations. Still catches: a visit that does not
narrow the posterior, a spread that fails to contract below 0.4, and a true orbit
outside the predictive spread.

### Faint-world information in one jitted call

The reporting-law survey computed the dark-sky log likelihood of each faint world in a
Python loop of eager vmaps. It is one `filter_jit(vmap)` over the faint worlds, the same
arithmetic. Programs for the test fell from 1259 to 974. Still catches: nulls that carry
information for a planet that never reaches half the limit.

### Fewer budget and chunk cases

`test_a_budget_that_ends_on_the_hold_installs_nothing` ran four budgets at three chunk
sizes (12 compiles). It runs the four budgets at chunk 7 and two more cases that put a
one-world chunk and a padded chunk on a budget. The chunk sweep itself is the first test
of the file. Still catches: a checkpoint that installs when no step is left for its
completion, at every budget and at both chunk extremes.

`test_host_service_parity` swept chunks 1, 3 and 32 over three worlds; chunk 3 holds
all worlds in one exact chunk, which chunk 32 also does with padding. It sweeps 1 and
32. Still catches: a hosted run whose answers depend on the chunking.

## Budgets

Per-file program budgets by the rule `ceil(n x 1.15) + 10`, from `--budget-json`:
`test_reporting_law.py` 1944 to 1602 (1384 programs), `test_physical_ofti.py` 791 to 772
(662). Other changed files compile no more programs than before and keep their budget.

## Not cut

- `test_corpus_invariants.py`, `test_invariant_mutants.py`: each case is a distinct
  study or mutant; the whole files cost 203 s and 71 s on the Mac.
- The three epoch relation tests (core, survey 3, survey 6): each runs its own study.
- `test_photomancy_ofti.py`: sharing the host run between its two slow tests saved
  about 1 s, so it was reverted.
- `test_refresh_runs.py::test_refresh_leaf_mutation_fails_parity`: one cold compile of
  a two-program study, shared by every field.

## Result

About 275 s of summed runner test time (about 2.5 minutes of wall time at two workers)
is removed, short of the 10 minute target.
