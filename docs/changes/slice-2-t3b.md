---
orphan: true
---

# Slice 2 T3b: the observer kit extras

T3b adds two checks to the observer kit, each with a negative control, and moves the
kit's own refits onto the refresh protocol's key. No core module changes, and no
golden digest in `tests/core/golden_histories.json` changes.

## What is in

- `check_delayed_correction(study, run, roots)`: after a revision released later than
  the row it supersedes, the final belief of each corrected subject must equal a
  flat-prior refit of the corrected row set. The set is computed in numpy from the
  log (the current revision of each accepted acquisition, derived rows excluded),
  independent of `records.revision_set`. The negative control is an observer whose
  `install` merges the superseded row back in; the check fails it.
- `check_derived_summary(study, run, roots, *, summary_kind)`: a derived summary of
  folded rows must add nothing to the belief. The negative control registers the same
  summary kind as non-derived, so the observer counts it as a datum (mean 10/3 over
  three rows instead of 3 over two); the check fails it. A run in which a summary was
  reissued under a folded acquisition id carries `Capacity.INVALID_CORRECTION` and is
  refused with a `ValueError` naming the cause, because the study, not the observer,
  produced that log.
- Both checks refuse a study without a refresh protocol, an unregistered summary kind
  and a run that did not finish (`COMPLETE` or `IDLE` is accepted, since a scripted
  refresh run of a null-policy study ends `IDLE`).

## Key move

Every kit refit now uses `refit_key(root, subject, revision_set_digest(rows))`. The
digest names the set of rows the refit reads, so in promotion parity the rows are put
in canonical order before hashing, and never in release order. A declared seed still
selects the root: seed 0 is the run's own root, any other seed is `fold_in(root, seed)`.
The earlier `fold_in(fold_in(root, seed), subject)` is gone.

## Moves

- Golden digests: none moved. `test_golden_histories.py` reports 18 failures on this
  machine, all float digests; the same case (`baseline/host/1`) gives identical digests
  on a clean checkout of origin/main (2dc469a), so the cause is this machine's
  floating-point platform, not this change. The integer digests match.
- Kit results: none moved beyond tolerance. All 51 existing kit tests pass after the
  key move, unchanged.

## Tests

| File | Tests | Tier | Time |
|---|---|---|---|
| `tests/core/test_observer_kit.py` | 4 new (23 in the file) | quick | 1.6 s for the file, slowest new test 0.31 s |
| `tests/core/test_observer_kit_corrections.py` | 6 | slow | 30 s, slowest 7.0 s |

The new run-based tests are in the slow tier because one executor compile is near 5 s
cold. The quick tests cover the replacement-rule oracle on a hand log, the kit key
against the protocol's key (it moves with the set and the seed) and the two refusals.

## Full suite

One run, `-n 2 --dist loadfile`, entry-points test deselected: 3,077 tests, 20 failed,
3,057 passed in 78 min. Eighteen failures are the golden float digests described above.
The other two were this change's own (the dispatch lint flagged a `getattr` loop and a
status `.name` in `observer_kit.py`); both are fixed and the two lint files, the slice
gate and the kit files pass (382 tests). The budget report lists six unit tests over 5 s,
all in files this change does not touch.

## Decisions to check

1. The run-based checks compare only `subject_leaves` at the corrected or summarised
   subjects, with integers exact and floats within `GO_TOLERANCE`.
2. The independent oracle is numpy, so it does not share a bug with `revision_set`.
3. A reissued summary is refused rather than failed, since the refusal is the study's.
