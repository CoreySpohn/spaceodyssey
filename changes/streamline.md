---
orphan: true
---

# Streamline: deletions, a smaller conformance kit, tolerance pins, studies moved to tests

## Step 1: deletions

- The legacy-name scanner (`tests/legacy_scan.py`, `tests/test_docs_names.py`) and
  `tests/core/test_reserved_fields.py`.
- Dead symbols with no reference in the libraries or scripts: `ToySolveAnalysis`,
  `installed_missing`, `digest_ints` (a test helper keeps a local copy), `toy_refit_service`,
  `SummaryTruth`, `population_system` (now a helper in `tests/adapters/test_observing_mode.py`).
- The `spaceodyssey.core` entry-point group: core kinds resolve through `CORE_KINDS`.
- `ChargeRule`, which had one member, and `program_charge_rule`.

## Re-baselined pins

Removing `ChargeRule` removes one carry leaf, so the canonical int layout and every history
digest changed. Float statistics and fingerprints did not. Re-baselined: `golden_histories.json`,
`golden_pin.json`, the digests in `test_compile_sharing.py` (equalities between cases are
unchanged) and `N_DEFAULT_CARRY_LEAVES` (134 to 133).

## Step 2: the observer kit is a checklist and a few tests

Removed from `core/conformance/observer_kit.py`: promotion parity, declared statistical
margins, refit calibration and their helpers. Removed tests: promotion parity, margins and
the 2,000-world statistical file (`test_observer_kit_statistics.py`), plus three unused
helpers in `kit_helpers.py`. Kept: single folding, delayed correction, derived summaries,
overlap, zero-weight rows, calibration, replay, and the corpus with its recorded-history
invariants (`studies/conformance.py` stays: the corpus feeds the golden histories and the
invariants). The conformance claims that lost their check are a checklist in
`docs/how-to/observer.md`.
