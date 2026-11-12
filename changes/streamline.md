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
