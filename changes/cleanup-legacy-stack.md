---
orphan: true
---

# Cleanup: the legacy certified-yield stack is removed

The package carried an older stack for certified yield: plan summaries, profiles and claims,
truth models, predictive distributions, a yield-reporting function and a plotting package.
Nothing in the transaction core, the codec, the adapters or the studies imported it, and the
only outside consumer was an analysis that will not run again. A tag, `legacy-certified-yield`,
marks the last commit that has it.

## Removed

- Modules: `data`, `dist`, `observe`, `universe`, `gaussian`, `metrics`, `reporting`, `plan`,
  `truth`, `profile`, `profiles`, `rng`, `errors`, `policy`, `inputs` and the `viz` package.
- Root exports: everything but `__version__` (`Observation`, `SummaryDataset`,
  `AbstractPredictive`, `GaussianSummary`, `CertificationOutcome`, `CertifiedYield`,
  `PlanSummary`, `Profile`, `Claim`, `Edge`, `Truth`, `AbstractTruthModel`, `PriorSet`,
  `Universe`, `certified_yield`, `stream` and the rest). Import `core`, `codec`, `adapters`,
  `studies` and `plugins` from their modules.
- Tests of those modules and the extras they needed: the `viz` extra (eyepiece, matplotlib)
  and the runtime `scipy` dependency. `scipy` stays in the `test` extra because two tests use it.
- The `save_viz_png` fixture, the budget entries and the validation defaults of the deleted
  test files.

## Kept

- `plugins` (entry-point resolution used by the codec) and the `core`, `codec`, `adapters`
  and `studies` packages, unchanged.
- The typed byte encoding and its SHA-256 digest, which the golden histories pin. They moved
  from `inputs` to `spaceodyssey.core.digest` with the same framing, tags and magic bytes, so
  every golden digest is unchanged. The plan-summary and truth tags, the snapshot copy and the
  tagged JSON form were used only by the removed stack and went with it; the byte vectors,
  refusal paths and normalization cases that apply to plain values moved to
  `tests/core/test_digest.py`.
- `tests/doubles.py` keeps `ToyEpochPhotometry`, used by the realize-context tests.

## Moved and trimmed

- Change records moved from `docs/changes/` to the top-level `changes/`. The docs build no
  longer needs to exclude them, and the fenced-code test run no longer needs an ignore.
- `docs/architecture.md` lost the sections on plan summaries, profiles and yield reports. The
  README now names only libraries the code imports and shows a snippet that runs.
- The stale-reference scan lists the removed module names, and exempts the changelog, which
  records them on purpose.
