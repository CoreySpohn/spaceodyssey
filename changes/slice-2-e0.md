# Slice 2 E0: baseline measurements

Measured on 2026-11-05 against `main` at `9b02fa0`. This record changes no
library code. It adds one scan tool, `tools/legacy_inventory.py`, and its unit
test, `tests/test_legacy_inventory.py`. Every other number below comes from a
CI log or a throwaway probe that is not committed.

Local environment: Python 3.12.12, JAX and jaxlib 0.12.0.1, Equinox 0.13.8,
pytest 9.1.1, macOS 26.7.2 arm64, the shared development environment, and the
four-thread compute launcher. CI numbers come from the hosted Linux runner.

## Summary

| Item | Measurement |
|---|---|
| 1. CI quick-tier wall time | 85.2 s (1,686 tests, `-n 2 --dist loadfile`), CI run 42129619884 on `9b02fa0`; warm compile cache |
| 2. Legacy closure (three modules) | 14 source files, 2,870 lines; 19 test modules, 231 test functions |
| 2. Legacy closure (expanded) | 19 source files, 3,507 lines with the package `__init__`; 26 test modules, 410 test functions (24 and 292 by direct import) |
| 3. Legacy test wall time, cold cache, `-n 2` | 26 expanded files: 52.2 s wall, 77.2 s summed, 838 cases; 19 closure files: 45.1 s wall, 61.6 s summed, 439 cases |
| 4. First call, one corpus study (`baseline`) | Host executor 3.4 to 3.6 s (23 XLA compiles); compiled executor 5.6 to 5.7 s (1 compile); through `run_case`: 9.0 to 9.4 s host, 11.0 to 11.4 s compiled |
| 4. First call, two-wide belief proxy | Host 3.9 to 4.7 s; compiled 5.5 to 5.9 s; same compile counts as the one-wide study |
| 5. Reconstructed `Study` reuses the program | Yes, in process: equal treedef, equal static part, zero new compiles on both executors; a changed static field recompiles |
| 6. Nightly slow tier | Red. One scheduled run so far, 42055076555 on `b2f086f`: 1 failed, 28 errors, 971 passed in 67.8 min |

## 1. CI quick-tier wall time

The latest `main` CI run is 42129619884 (push of `9b02fa0`, 2026-11-05 23:28
UTC), job "Quick tier (py3.12)", conclusion success. Its pytest line reads
`1686 passed in 85.16s`, and the budget plugin reports `wall=85.2s`. The junit
`testsuite` time that `ci_summary.py` reads is this session time; the junit file
is not uploaded as an artifact and the job summary is not exposed through the
REST API, so the value is read from the job log.

The persistent compile cache matters to this figure. This run restored the
cache saved by the previous commit (`0ae7a67`), whose compiled programs are the
same, so it is a warm run. The run for `0ae7a67` itself (42126915489) took
182.1 s: it restored a cache saved by `fa0a538`, before the compile-reuse
changes, so most of its programs were new. The run for `fa0a538` took 349.1 s.
A change that alters many compiled programs should be expected to pay roughly
the 182 s figure on its first CI run and the 85 s figure afterwards.

## 2. Legacy closure inventory

Command: `python tools/legacy_inventory.py --list`.

The tool parses every module under `src/spaceodyssey` and every file under
`tests` with `ast`, records each explicit import (top-level and inside
functions), and resolves `from spaceodyssey import name` and
`spaceodyssey.name` through the package `__init__` to the module that defines
the name. The package `__init__` itself is excluded from the closure, since it
imports every public module, and is reported separately. Test helpers
(`tests.*` imports) are followed transitively. Test functions are counted as
definitions (`test_*` functions and `Test*` methods), not parametrized cases.
String imports through `importlib` are not followed.

Results, exactly as printed:

| Seed set | Source | Test modules (direct) | Test modules (with helpers) |
|---|---|---|---|
| `spec`, `registry`, `wiring` | 14 files, 2,870 lines | 19 files, 231 functions | 19 files, 231 functions |
| plus `seams`, `contracts`, `aliases` | 18 files, 3,352 lines; 19 files, 3,507 lines with `__init__` (155) | 24 files, 292 functions | 26 files, 410 functions |

The three named modules are 839 lines (`spec.py` 332, `registry.py` 250,
`wiring.py` 257). The three-module closure adds `account`, `belief`, `build`,
`certify`, `context`, `engines` (with `analytic` and `mc`),
`gaussian_profile`, `io` and `loop`. The expanded set adds `seams` (42),
`contracts` (91), `aliases` (20) and `testing` (329), which imports
`contracts`. Every count matches the planned inventory; no figure was adjusted.

Two notes for whoever plans the deletion:

- `engines/__init__.py` imports `aliases.resolve_qualname_alias` as well as
  `registry`, so the engines package is in the closure by two routes.
- "Affected" is wider than "retired". Five of the 26 expanded test modules test
  retained modules and enter the set only through `spaceodyssey.testing` or the
  `tests.test_plan` helpers: `test_plan.py` (imports `contracts`),
  `test_truth.py`, `test_dist.py`, `test_viz_extract.py` and
  `test_viz_plots.py`. Together they hold 167 of the 410 functions.

## 3. Wall time of the legacy test files

Command, with the persistent cache disabled (`SPACEODYSSEY_JAX_CACHE=off`) so
the times include every compile:

```sh
<compute-launcher> --threads 4 -- <worktree pytest> <files> \
  -n 2 --dist loadfile --durations=0 -q -p no:randomly --junitxml=<report>
```

No marker filter was applied, so the run includes the nine slow-marked tests of
`tests/test_contracts.py` (marked slow because its file name matches the core
`test_contracts.py` entry in `SLOW_FILES`).

| Set | Cases | Wall | Summed | Peak footprint |
|---|---|---|---|---|
| Expanded, 26 files | 838 | 52.2 s | 77.2 s | 2.4 GiB |
| Three-module closure, 19 files | 439 | 45.1 s | 61.6 s | 1.9 GiB |

Per-file summed time, slowest first (the remaining 16 files are each under
1 s): `test_gaussian_profile.py` 46.8 s (71 cases), `test_viz_plots.py` 8.6 s,
`test_toy_certifier.py` 5.2 s, `test_truth.py` 3.9 s, `test_engines.py` 3.3 s,
`test_analytic.py` 2.1 s, `test_dist.py` 1.3 s, `test_profile_certificate.py`
1.1 s, `test_replicates.py` 1.0 s.

`test_gaussian_profile.py` is 76 percent of the closure's summed time. The
other 18 closure files together take 14.9 s summed.

## 4. First-call compile time

A throwaway probe ran each case in a fresh process with no persistent
compilation cache, x64 enabled, `N_STEPS = 40`, and two worlds from
`make_worlds`. Each case ran twice (two fresh processes); both values are
given. "First" is the wall time of the first call, blocked until ready;
"second" is the same call again in the same process. The trace, lowering and
XLA compile columns sum JAX's own compile-duration monitoring events during the
first call.

| Case | First | Trace | Lower | XLA compile | Compiles | Second |
|---|---|---|---|---|---|---|
| `baseline`, host executor, one world | 3.62 / 3.41 s | 0.38 / 0.34 s | 0.24 / 0.21 s | 2.85 / 2.71 s | 23 | 0.08 s |
| `baseline`, compiled executor, two worlds | 5.66 / 5.58 s | 1.54 / 1.54 s | 0.15 / 0.14 s | 3.93 / 3.86 s | 1 | 0.005 s |
| two-wide belief, host | 3.90 / 4.68 s | 0.41 / 0.47 s | 0.28 / 0.31 s | 3.07 / 3.73 s | 23 | 0.08 s |
| two-wide belief, compiled | 5.52 / 5.87 s | 1.66 / 1.64 s | 0.21 / 0.22 s | 3.60 / 3.97 s | 1 | 0.005 s |
| `run_case(baseline, "host")` | 9.44 / 9.00 s | 0.91 / 0.92 s | 0.52 / 0.52 s | 7.22 / 6.77 s | 92 | 0.63 to 0.73 s |
| `run_case(baseline, "compiled")` | 11.36 / 11.04 s | 2.94 / 2.69 s | 0.45 / 0.46 s | 7.71 / 7.63 s | 72 | 0.11 to 0.12 s |

What was measured for the two-type layout: no two-type belief exists yet, and
the carry, snapshot and trace read `belief.mean`, `belief.n` and `belief.aux`
by name, so a belief of a different type does not run. The proxy is a
`MeanBelief` subclass with a second block of the same three fields
(`mean2`, `n2`, `aux2`) and a `ToyRunningMean` subclass that folds every
completed record into both blocks (the second block folds the squared payload).
The belief pytree has six leaves instead of three and twice the bytes, in both
`belief` and `pbelief`, on the same `survey_1` study. It measures the cost of a
wider belief in the existing programs. It does not measure a new record kind, a
second fold dispatch, or new carry leaves outside the belief.

Reading: a wider belief changed neither the number of compiled programs nor,
beyond run-to-run noise, the first-call time. One new study structure costs
about 3.5 s (host) and 5.6 s (compiled) on this machine when run directly, and
9 to 11 s when it goes through `run_case`, which also compiles the counting
observer, the Python policy form, the decision-view capture and the private
probe variant. The hosted runner has been about three times slower than this
machine in a single process.

## 5. Does a reconstructed Study reuse the compiled program?

The serialization codec does not exist yet, so the probe used the nearest
reconstructions available in process, each compared with the `baseline` study
`a`:

- `b`: `case.make_study()` called again (re-construction from the same
  arguments);
- `c`: `eqx.tree_at` replacing `horizon_tick` with an equal fresh array;
- `d`: `tree_unflatten(treedef, leaves)` of `a`.

For all three: the treedef equals `a`'s, the non-array part from
`eqx.partition(study, eqx.is_array)` has an equal structure, hash and value,
and `eqx.tree_equal` holds. After `a` ran once on the compiled executor (one
XLA compile, about 4.7 s), runs of `b`, `c` and `d` took 5 ms each with zero
trace, lowering or compile events. The host executor likewise ran `b` with no
compile after `a` (0.08 s). The negative control, `a` with a
`ToyRunningMean(measurement_kind=99)` observer, compiled again (one compile,
4.4 to 4.8 s over three runs), so the probe can detect a miss.

What this shows: `eqx.filter_jit` keys the executors' programs on the treedef,
the non-array leaf values and the array shapes and dtypes, not on object
identity. A loaded study that reproduces those reuses the in-process program,
so a round trip costs one corpus compile, not two.

What it does not show: a decoder must reproduce the same classes, the same
Python types for non-array fields (a list in place of a tuple changes the
treedef), and the same array dtypes and weak-type flags; none of those was
exercised, because no decoder exists. Reuse across processes depends on the
persistent cache, which keys on the lowered program, and was not tested.

## 6. Nightly slow tier

The CI workflow's schedule (`17 7 * * *`) has produced one run so far:
42055076555, created 2026-11-05 15:26 UTC on `b2f086f`, job "Slow tier",
conclusion failure. Result: `1 failed, 971 passed, 1684 deselected, 28 errors
in 4065.77s (1:07:45)`. No later scheduled or dispatched slow-tier run exists.

Failing tests:

- 28 setup errors, `tests/test_slice1_gate.py::test_executor_parity[<case>-K4-parity]`
  for every corpus case. The test requests pytest's `record_property` fixture.
  The slow tier runs with `--junitxml`, whose default family `xunit2` makes
  `record_property` emit `PytestWarning: record_property is incompatible with
  junit_family 'xunit2'`, and `filterwarnings = ["error"]` turns it into a setup
  error. A local run without `--junitxml` does not see it.
- 1 failure, `tests/core/test_contracts.py::test_collection_preserves_callers_x64_setting`.
  The test runs `pytest.main(["--collect-only", ...])` in a subprocess with x64
  off and asserts that collection leaves it off. Collection succeeds (32 tests)
  but x64 is left on, because `tests/conftest.py` calls
  `jax.config.update("jax_enable_x64", True)` at import (since `ba9008f`). The
  test predates that change and still assumes the scoped approach of
  `tests/core/conftest.py`, whose `_X64Module` overrides the private
  `pytest.Module._getobj` to enable x64 only while a module is imported. With
  the global setting in place, that override no longer changes behavior. This
  failure reproduces locally in 3 s.

Both files are slow only because their names are in `SLOW_FILES`, which is
why the quick tier never reports them. Neither is fixed here.

## Tests added and retired

- Added: `tests/test_legacy_inventory.py::test_closure_follows_reexports_lazy_imports_and_helpers`
  (no JAX, under 0.1 s). Fails if the scan misses a re-exported name, a
  function-level import or a test helper, or if the package `__init__` pulls
  every importer into the closure.
- Retired: none.
- Time spent: about two hours, mostly waiting on timing runs.
