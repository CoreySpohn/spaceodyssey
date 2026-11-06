---
orphan: true
---

# Slice 2 T1a: observer-declared belief types

An observer now declares two belief types to the core through a `BeliefLayout`:
the full belief, held in `Carry.belief`, and the policy summary, held in
`Carry.pbelief` and the only belief a policy reads. The layout gives the shapes of
both, the integer count leaf the snapshot fingerprint reads, and the projection the
trace records. Every Slice 1 history is unchanged: all 112 golden digests match.

## Commits

1. `test(core): pin golden content hashes of every corpus history`, committed
   from unmodified `main` before any other change.
2. `feat(core): observer belief layouts; snapshot_fingerprint takes the layout`
   (types and trace).
3. `feat(core): canonical history, battery and fold counter read the belief layout`
   (comparator and battery).

## Golden-history manifest

`tests/core/test_golden_histories.py` compares every conformance corpus history
with `tests/core/golden_histories.json`: 28 cases, both executors, two worlds
each, 112 entries. Each entry holds two SHA-256 digests
(`spaceodyssey.inputs.digest`, the library's typed content encoding) of the
integer and floating-point arrays of the world's canonical history, as the shared
corpus run store records it, so a change of one bit in any array fails and the
message says which half moved. A one-bit negative control checks that the digest
sees a float bit and an integer. The manifest is rewritten only with
`SPACEODYSSEY_WRITE_GOLDEN=1`, and any moved entry must be named in a change
record. The file runs the whole corpus, so it is listed in `SLOW_FILES` beside
the other corpus files and does not enter the quick tier; in a process that has
already run the corpus (the invariant tests, the gate) it reuses those runs.

The digests are exact bytes. The corpus has been checked on macOS arm64 only; the
first nightly run on the Linux runner shows whether its float bytes agree.

## What changed

- `BeliefLayout` (in `core.snapshot`): `belief` and `summary` shapes, taken from
  example values when given those; `count_leaf`; `project` and an optional
  `project_summary` (defaulting to `project`); comparator names
  `belief_leaves` and `summary_leaves`, which leave out leaves of zero size so a
  zero-capacity leaf changes no history. A layout whose count leaf is missing,
  not an integer, absent from one type or shaped differently in the two is
  refused at construction. `mean_layout` and `project_mean` build the
  `MeanBelief` layout.
- `AbstractObserver` gains `layout(caps)`, `init_summary(caps)` and
  `fold_summary(summary, rec, caps)`. The defaults keep Slice 1 behaviour: the
  summary is the belief (`init_summary` is `init`), `fold_summary` folds and then
  analyzes, as the release handler did for `pbelief`, and the layout is the
  `MeanBelief` layout of `init`. The toy observer declares its layout explicitly:
  `MeanBelief` for both types, count leaf `n`, projection `project_mean`.
- `Study` refuses an observer whose layout does not describe the beliefs it
  initializes (leaf paths, shapes and dtypes of `init` and `init_summary`).
- `init_carry` builds `pbelief` with `init_summary`; the release handler folds it
  with `fold_summary`.
- `snapshot_fingerprint(view, layout)` reads `layout.count(view.belief)` where it
  read `view.belief.n`. The transaction, the host service and the battery's view
  recorder build the layout from the study.
- `advance` and `finish_trace` fill `Trace.belief_mean` and `belief_aux` with the
  layout's projection; no trace field is renamed. `finish_trace` takes the study.
- The canonical history reads belief leaves, `belief_n`, `pbelief_n`,
  `belief_final`, `pbelief_final` and `belief_aux` through the layout
  (`LAYOUT_FIELDS` replaces the two hard-coded belief groups of
  `CANONICAL_FIELDS`). `canonical_history(carry, trace, layout=None)` uses the
  `MeanBelief` layout of the carry when given none; `RunResult` carries the layout
  the executor used, and `history` passes it.
- Battery: `truth_swap` and `latency_shift` name the projection fields through
  `carry.PROJECTION_FIELDS`; `latency_shift` compares the final belief over the
  layout's leaves; the metamorphic relations compare both beliefs over the
  layout's leaves and the relabeling compares the summary's count leaf;
  `compare_views` takes the layout. `executors/replay.py` needed no change: it
  initializes with `init` and folds through the same `analyze(fold(...))` path as
  the release handler, and reads no belief leaf.
- Conformance (added to this task after a review of the spec): the fold counter
  `CountingObserver` wraps whatever full belief and summary the inner layout
  declares, in a generic `CountedBelief(value, ids, counts)`, instead of
  rebuilding `MeanBelief`. Field reads on the wrapper go to the wrapped value, so
  a policy or host service still reads the summary's own fields; this one dynamic
  attribute read is a reviewed allowance in `tests/core/gate_helpers.py`.
  `run_case` takes the canonical history from the unwrapped carry with the inner
  layout, and `one_fold_per_acquisition` counts folds through the count leaf.
  Excluding derived rows from the expected folds is left to a later task.
- The Slice 1 gate's allowance list loses three entries whose lines were
  replaced (two in `metamorphic.py`, one in `replay_latency.py`) and gains the
  wrapper entry above.

## Public API

`AbstractObserver`, `snapshot_fingerprint`, `finish_trace` and `canonical_history`
are exported from `spaceodyssey.core`. Observers that subclass `AbstractObserver`
keep working unchanged through the defaults. `snapshot_fingerprint` now needs the
layout and `finish_trace` the study, which breaks direct callers. The library is
below 1.0, so the commits are `feat(core)` without a breaking marker, and the
first subject names the `snapshot_fingerprint` change.

## Tests

Added:

- `tests/core/test_golden_histories.py`: 58 (one coverage check, 56 per case and
  executor, one negative control); slow tier.
- `tests/core/test_belief_layout.py`: 8 unit tests (refused layouts, four cases;
  the fingerprint reads the declared count leaf; the carry holds the declared
  types and an undeclared summary is refused; the comparator reads layout names
  and drops a zero-size leaf; the fold counter wraps a layout whose summary is not
  a `MeanBelief`).

Changed: the fingerprint call sites in `test_snapshot_fingerprint.py`,
`test_audience_firewall.py` and `test_query_ordering.py` pass the layout;
`test_canonical_fields.py` accepts `LAYOUT_FIELDS` as a disposition and mutates
each toy layout leaf, the same six belief fields as before.

Retired: none. Net: 8 quick, 58 slow.

Deciding tests on the rebased branch: `tests/core/test_corpus_invariants.py`,
`tests/core/test_parity_values.py`, `tests/test_slice1_gate.py` and
`tests/core/test_golden_histories.py` pass, with no history difference.

## Time

Quick tier (`-m "not slow" -n 2 --dist loadfile`, local, four threads, empty
compile cache each time, junit wall time): 91.4 s for 1,686 tests on `9b02fa0`
before T1a; 85.0 s for 1,695 tests on the rebased branch. The difference is
within run-to-run noise on a shared machine; the eight new quick tests take under
1 s together, and the golden manifest runs in the slow tier. Its file takes about
100 s alone with a warm compile cache, about 140 s cold. The full suite on the
rebased branch passes: 2,754 tests in 672 s wall at `-n 2`.

Actual time spent: about 1.3 hours of wall time, of which about 40 minutes was
spent waiting on full-suite and golden runs.
