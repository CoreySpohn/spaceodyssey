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

## Step 3: golden pins are tolerance pins

Each history in `golden_histories.json` and `golden_pin.json` now also carries per-leaf
float statistics (sum, sum of absolute values, non-finite count), compared at rtol 1e-9
under every jaxlib release. The integer digests stay exact everywhere. The float bit
digests, the snapshot fingerprints and the fault float digests stay exact only under the
jaxlib the manifest records as `_jaxlib`; under another release the fingerprints reduce to
their decision counts. Bumping `.github/jax-pin.txt` without regenerating the manifest
(`SPACEODYSSEY_WRITE_GOLDEN=1`) fails `test_the_manifest_was_written_under_the_pinned_jaxlib`:
that is the declared re-baseline. Existing digests, fingerprints and the pin's source
commit are unchanged; only the statistics and `_jaxlib` were added (the pin was captured
again from the same runs, and every pre-existing entry was checked equal).

## Step 4: the study scenarios live in tests

The fifteen `studies/survey_*.py` modules and `studies/allocator.py` (about 3,600 lines) moved
from `src/spaceodyssey/studies/` to `tests/studies/scenarios/`. Nothing in `src/` or the
analysis scripts imported them; only their own tests and each other did, and those imports
now name `tests.studies.scenarios`. They are no longer part of the installed package.
