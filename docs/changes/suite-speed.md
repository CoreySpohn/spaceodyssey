# Test suite speed and memory

This is the historical speed-pass report. The follow-up
[resident-memory report](suite-memory.md) supersedes its cache watermarks and run
guidance. Use one process for the quick tier and one process per slow file when
enforcing the 6 GB process budget.

## What was slow

Nearly all test time is JAX tracing and XLA compilation. Each distinct program class
(structure plus shapes) costs 20 to 45 s cold across four program families (counted and
instrumented 2-world run, probed run, host jits, 4-world parity run), and about 1 GB of
resident memory that `jax.clear_caches()` does not return to the operating system.

The causes of repeated compilation were:

- The decision-view recorder was a bound method of a fresh object per run, hence a new
  static jit key per run. It now delivers through a module-level function and a slot array.
- Policy, agenda and study knobs (plan, wait, bundle, cancel, stop, horizon, heartbeat
  period and audience) were Python ints and tuples, which `eqx.filter_jit` treats as
  static. They are now int64 arrays, and the plan is padded to 8 rows with -1 sentinels, so
  studies that differ only in those values share one compiled program.
- Gate, corpus invariants and parity corpus each re-ran the corpus. They now share one
  cached run store (`tests/core/parity_helpers.py::corpus_runs`).

## What changed

- `src/spaceodyssey/core/battery/views.py`, `studies/toy.py`, `core/study.py`,
  `core/transaction.py`: stable static configuration as above, no behavior change; every
  hand history and the canonical history are unchanged.
- `tests/conftest.py`: persistent compilation cache (`SPACEODYSSEY_JAX_CACHE`, default
  `.jax-cache/`, `off` disables), a resident-memory bound (`SPACEODYSSEY_RSS_LIMIT_MB`,
  default 4500: clear caches when above it and grown 1000 MB since the last clear), the
  `slow` marking by file (`SLOW_FILES`), and xdist groups for the gate.
- `tests/core/test_stop_grid.py`: the 12 `test_stop_case` tests were exact duplicates of
  `test_corpus.py::test_corpus_three_way_parity[stop_*]` (criterion a) and are removed.
- `tests/core/test_keys.py`: two assertions added (the `ordinal` coordinate is refused in
  a coupling; the unknown-coordinate error names the coordinate).

## How to run

- Quick run, `uv run python -m pytest -m "not slow"`: legacy tests, hand histories, survey 1
  and the light core files.
- Gate, `pytest -n 2 --dist loadgroup tests/test_slice1_gate.py`: the compiled and host
  tests land on separate workers.
- Full suite: `pytest -n 2 --dist loadgroup` on a machine with at least 32 GB, or in
  sequential chunks (gate, `tests/core` in three file groups, `tests/studies`, top-level
  files), one process each. Never above `-n 2`.
- The slow set is `SLOW_FILES` in `tests/conftest.py`: gate, corpus, parity, executors,
  conformance guards, scorer columns, acquisition index, query ordering, revisions,
  metamorphic, host service, draws, visible relations, battery guards, invariant mutants,
  run status, fork, latency shift, truth swap, null reference, contracts, audience
  firewall, admission rules, leak, second kind, canonical fields, gradients and survey 25.
  All of them run in the full suite.

## Timings

Measured on the development Mac, cold persistent cache unless noted.

| Run | Before | After |
| --- | --- | --- |
| Full suite | about 43 min at `-n 4` (OOM on two files at about 40 GB) | 16.6 min at `-n 2 --dist loadgroup` (2674 passed), 18.2 min on an earlier run |
| Gate alone | out of memory (about 40 GB) | 4.1 min at `-n 2 --dist loadgroup`, 10.9 min single process |
| `-m "not slow"` | not available (3 slow marks) | 3.2 min, 1707 tests |
| Corpus invariants file | 18.6 min | 7.7 min |

Peak resident memory of the largest single process is 12 to 13 GB in the full and gate
runs, and 8.8 GB in the quick run, so the 6 GB target is not met. Each distinct compiled
program class keeps about 1 GB that `jax.clear_caches()` does not return, and the files
that exercise many classes (gate, corpus, contracts, audience firewall) pass that point
within one process. The persistent compilation cache removes only the XLA part of a
compile and does not help a cold run. The 10-minute full-suite target is not reached
without dropping checks: what remains is tracing and compiling about 40 distinct
program classes.

Test counts: the baseline ran 2356 passed and 2 failed, with the gate and corpus
invariants files (about 330 tests) lost to out-of-memory, so about 2685 in all; after, 2674
pass, with the 2 failures fixed upstream and 12 duplicate tests removed.
