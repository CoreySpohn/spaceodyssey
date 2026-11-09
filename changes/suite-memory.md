# Test suite resident memory

Measurements against baseline `c059668`, on macOS 26.7.2 with Python 3.12.12,
JAX and jaxlib 0.10.2, Equinox 0.13.8, CPU execution, and x64 enabled for the
transaction tests. Each file runs in its own process with four threads and a
14000 MiB job ceiling. No xdist workers are used. The quick tier is measured in
one process.
Final validation uses an isolated virtual environment with these same versions.
`UV_PROJECT_ENVIRONMENT` selects that environment, `UV_NO_SYNC=1` prevents
dependency changes during measurement, and `PYTHONPATH` includes `src:.` so
imports resolve to the measured worktree.

An intervening shared-environment upgrade to JAX 0.12.0.1 produced two
Gaussian-profile compatibility failures in unchanged code: the fixed-iteration
Jaxpr inspection and the zero-covariance warning guard. Those experimental runs
are excluded from the comparison. No warning or assertion guard was relaxed;
both checks pass in the isolated baseline environment.

RSS is measured with `resource.getrusage(RUSAGE_SELF).ru_maxrss`, not the job
supervisor's physical footprint (which is a different macOS measurement). Wall
time excludes waiting for a compute slot. The persistent cache starts empty for
the first baseline file and is shared by subsequent files. Each final file and the
final quick tier start with a separate empty persistent-cache directory. These
cache conditions make final memory measurements conservative, but wall times are
not a controlled speed comparison. The environment already sets `MallocNanoZone=0`;
this change adds no allocator or XLA options.

Distinct programs are unique XLA persistent-cache keys logged by JAX, including
small eager operations. Compile/load requests also include repeated loads after
cache clearing. This distinction matters: a newly created scan body can reload
the same program hundreds of times. The optional budget plugin's compile total
double-counts persistent hits on this JAX version, so it is not used here.
Measurement instrumentation disables that plugin's secondary RSS ceiling and
trend-file writes; the requested 14000 MiB supervisor ceiling remains enforced.
This is local profiling configuration, not a change to suite guards.

The diagnostic profile also measures cleanup after the tests: JAX cache clearing,
garbage collection, Equinox cache clearing, and allocator pressure relief. Peak
RSS includes that cleanup, making the bound conservative.
The acceptance bound is 6,000,000,000 bytes (about 5722 MiB), rather than 6 GiB.

## Baseline findings

- `test_executors.py`: 33 passing tests, 4927 MiB peak, 252 distinct programs,
  405 compile/load requests, 272 seconds.
- `test_corpus_invariants.py`: 165 passing tests, 6963 MiB peak including cleanup
  (6578 MiB during tests), 369 distinct programs, 916 requests, 558 seconds.
- JAX clearing reduced live executables to zero. Extra garbage collection,
  Equinox clearing and allocator pressure relief did not reduce the remaining
  RSS. The final corpus-invariants clear actually raised RSS from 5273 to 6923 MiB.
- `malloc_zone_statistics` confirms freed allocator capacity: acquisition-index
  cleanup reduced malloc's in-use bytes from 4801 to 61 MiB, but 5667 MiB remained
  reserved and RSS remained 5918 MiB. The parity corpus retained only 9.36 MiB of
  JAX arrays. Removing the cached histories would not recover gigabytes.
- The previous RSS growth gate permitted another 1000 MiB above the retained
  level before clearing again; it did not enforce a fixed upper bound.
- The fingerprint digest creates a fresh scan body on each eager call. Corpus
  invariants made 270 scan compile/load requests for only two distinct programs.
- Necessary specializations include overflow capacities, phase counts, policy and
  observer types, instrumented callbacks, executor mutants, world batch sizes,
  scan lengths, and hosted scan chunks. Overflow capacities must remain small to
  test exhaustion. Corpus results contain arrays and histories, not executables;
  the decision-view capture registry uses weak references.

## Changes

The only library change moves the fingerprint digest's scan body to module scope.
The integer operations, order, seed and final mix are identical. Reusing the same
callable avoids retracing an equivalent scan for every eager fingerprint call.
Early cache eviction can increase total compile/load requests for other programs;
the final measurements distinguish those reloads from distinct XLA programs.
In corpus invariants, scan requests fell from 270 to 11, still for two distinct
scan programs. Total requests rose from 916 to 3010 as other programs were
reloaded, while the distinct-program count remained 369.

`WaitingReactivePolicy` in the test helpers now stores its wait location and
length as scalar int64 arrays. Equinox can pass the values dynamically instead
of specializing for each Python integer. The schedules and perturbations are
unchanged.

Test teardown now clears JAX caches at fixed watermarks: 3500 MiB current RSS or
1750 MiB live malloc allocations. macOS's
[malloc_zone_statistics(NULL, ...)](https://github.com/apple-oss-distributions/libmalloc/blob/main/src/malloc.c)
reports live allocations across all zones, excluding freed pages retained by the
allocator. This signal also tracks live allocations when macOS compression
reduces their resident footprint. Other platforms fall back to RSS. Both
watermarks derive from `SPACEODYSSEY_RSS_LIMIT_MB`; overriding it changes the
measured budget. The previous growth-since-last-clear condition is removed.
Early eviction leaves space for the next compile and for cleanup itself; it does
not promise that the
allocator will return freed memory immediately. Reusing freed allocator space
keeps subsequent compiles from growing its retained capacity indefinitely.

Replay-log checks join the existing `slow` file marker. The quick tier contains
1674 checks and the slow tier contains 1000. All 2674 checks remain collected by
the default suite. No assertions, worlds, corpus cases, capacity boundaries or
canonical histories were removed or reduced. Padding overflow capacities would
change the boundary under test, so those shapes remain distinct.

## Running within the measured budget

Run the quick tier in one process, then each file containing slow checks in its
own process. `SLOW_FILES` in `tests/conftest.py` defines whole-file markers;
`tests/test_gaussian_profile.py` also has six individually marked slow checks.
Run that entire file too when verifying its standalone peak. Ordinary pytest
selection is sufficient. Do not combine all slow files into one process
when relying on this budget. No custom test runner is required.

```sh
compute-run --threads 4 --mem 14000 -- uv run python -m pytest -q -p no:cacheprovider -m "not slow"
compute-run --threads 4 --mem 14000 -- uv run python -m pytest -q -p no:cacheprovider tests/test_slice1_gate.py
compute-run --threads 4 --mem 14000 -- uv run python -m pytest -q -p no:cacheprovider tests/test_gaussian_profile.py
# Repeat the second command for each remaining slow file.
```

The final validation runs use precisely this process/thread/memory arrangement,
without xdist. The memory limit is a measured result on the stated CPU platform,
not a hard in-process limiter: cache eviction happens between tests. Necessary
shape, policy-type, executor and callback specializations still incur tracing
and compilation costs, and earlier eviction adds reload work.

## Results

All 2674 distinct checks passed. The quick tier ran 1674 checks in one process.
All 1000 slow checks were covered by 32 separate file runs; the Gaussian-profile
file also reran its quick checks to verify its standalone peak. Every final run
stayed below 6,000,000,000 bytes, including diagnostic cleanup. The highest peak
was 4,600 MiB (`tests/core/test_run_status.py`). Every distinct program's first
cache request was a miss in these final runs; compile-start and cache-key event
counts matched.

Values below are **before -> after**. RSS uses MiB; wall time uses pytest's
reported seconds and excludes scheduler waiting. Distinct programs include eager
kernels as well as large executor programs. For the contract guard that launches
a collection subprocess, RSS is the maximum per-process peak and program keys
are deduplicated across the parent and child; both processes satisfy the bound.

| File / tier | Peak RSS (MiB) | Distinct programs | Wall time (s) |
| --- | ---: | ---: | ---: |
| `tests/core/test_corpus_invariants.py` | 6,963 -> 3,955 | 369 -> 369 | 558 -> 345 |
| `tests/test_slice1_gate.py` | >=6,998* -> 3,974 | 498* -> 604 | 744* -> 957 |
| `tests/core/test_corpus.py` | 5,170 -> 3,800 | 325 -> 325 | 232 -> 181 |
| `tests/core/test_parity_corpus.py` | 5,616 -> 3,538 | 319 -> 319 | 81 -> 240 |
| `tests/core/test_scorer_columns.py` | 5,180 -> 3,321 | 230 -> 230 | 237 -> 159 |
| `tests/core/test_acquisition_index.py` | 5,958 -> 3,385 | 174 -> 174 | 166 -> 244 |
| `tests/core/test_replay_log.py` | 3,420 -> 2,792 | 167 -> 167 | 42 -> 55 |
| `tests/core/test_draws.py` | 6,043 -> 3,452 | 162 -> 158 | 262 -> 198 |
| `tests/core/test_executors.py` | 4,927 -> 3,624 | 252 -> 252 | 272 -> 260 |
| `tests/core/test_conformance_guards.py` | 4,648 -> 3,122 | 205 -> 205 | 68 -> 146 |
| `-m "not slow"` | >=6,150* -> 3,662 | 1,008* -> 2,347 | 322* -> 619 |

`*` The baseline gate and quick-tier processes were killed (exit 137) at supervisor
footprints of 14005 and 14022 MiB respectively. Their RSS values are the last
observed peak lower bounds, not those footprint measurements. Their program
counts are partial requests, including the in-flight compile. The gate completed
115 of 163 checks; the old quick tier completed 590 of 1707. The final quick tier
has 33 fewer selected checks because replay-log moved to `slow`; that file passes
separately with all 33 checks intact. Killed-run wall time is elapsed time until
the kill, recovered from the process log.

The following remaining slow files were also run individually, each with a fresh
persistent cache. They have no before comparison because they were additional
coverage of the per-file bound, beyond the ten requested heavy baselines.

| Additional file | Passed | Peak RSS (MiB) | Distinct programs | Wall time (s) |
| --- | ---: | ---: | ---: | ---: |
| `tests/test_contracts.py` | 9 | 262 | 1 | 2 |
| `tests/core/test_admission_rules.py` | 12 | 1,960 | 106 | 38 |
| `tests/core/test_audience_firewall.py` | 12 | 2,811 | 177 | 27 |
| `tests/core/test_battery_guards.py` | 8 | 2,785 | 147 | 85 |
| `tests/core/test_canonical_fields.py` | 144 | 1,570 | 66 | 20 |
| `tests/core/test_contracts.py` | 54 | 3,629 | 104 | 50 |
| `tests/core/test_fork.py` | 7 | 2,319 | 97 | 57 |
| `tests/core/test_host_service.py` | 14 | 3,028 | 332 | 165 |
| `tests/core/test_invariant_mutants.py` | 39 | 3,172 | 258 | 213 |
| `tests/core/test_latency_shift.py` | 4 | 1,648 | 76 | 42 |
| `tests/core/test_leak.py` | 6 | 2,649 | 100 | 35 |
| `tests/core/test_metamorphic.py` | 8 | 3,644 | 155 | 78 |
| `tests/core/test_null_reference.py` | 6 | 1,782 | 82 | 20 |
| `tests/core/test_query_ordering.py` | 19 | 3,724 | 228 | 145 |
| `tests/core/test_revisions.py` | 12 | 3,663 | 206 | 131 |
| `tests/core/test_run_status.py` | 15 | 4,600 | 204 | 193 |
| `tests/core/test_second_kind.py` | 4 | 1,870 | 147 | 38 |
| `tests/core/test_truth_swap.py` | 3 | 1,649 | 86 | 37 |
| `tests/core/test_visible_relations.py` | 8 | 2,724 | 115 | 88 |
| `tests/test_gaussian_profile.py` | 67 | 2,337 | 1,026 | 52 |
| `tests/studies/test_gradients.py` | 16 | 4,366 | 157 | 342 |
| `tests/studies/test_survey_25.py` | 6 | 1,836 | 190 | 34 |
