---
orphan: true
---

# Slice 2b R2: the prior observer, per-audience overlap and the audience-prior boundary

R2 builds on the R1 table contract (`slice-2b-r1.md`): `PedigreeRule` and
`resolve_overlap` (D4), `PriorObserver` and its installation of table priors (D2),
the retained-prior check `D008`, the toy audience-prior refit boundary with its
contract test, findings `D009` and `D010`, the observer kit checks for observers that
read priors (section 5), and the replay check `test_replay_reproduces_prior_run`.

## Commits

1. `feat(core): observer kit checks of overlap weights and zero-weight rows for observers that read priors` (`8273cc7`)
2. `feat(studies): per-audience overlap rule, prior observer and audience-prior refit boundary with findings D008 to D010` (`12629dc`)
3. `test(core): per-audience prior installation, overlap invariance and the kit's prior checks` (`1499433`)
4. `test(core): audience-prior refit contract, D009, D010 and the prior replay check` (`e6b94a2`)
5. `feat(codec): rebuild the run record's refit boundary for the JSON pre-flight` (`914c589`)
6. review round 1 fixes (`bb3fcb4`, `2799006`, `943ca19`; see the section of that name)
7. `chore(pre-commit): the prior test helpers are a support module` (`8ebbae9`)
8. this record (`716c7fa`)
9. review round 2 (independent tier-2 review) fixes: `38ac5bb`, `22e0b01`,
   `aab7092`, and this record's update (see "Review round 2")

Commit 2 is about 900 lines (the new module, the finding codes and the seeded
pre-flight cases that keep `test_every_code_has_a_seeded_mistake` green land
together), above the 600-line guide.

## What changed

- `studies/premission.py` (shared, R1-owned; the only edits): `PedigreeRule`
  (`EXCLUDE_PRIOR`, `EXCLUDE_ROWS`), `overlap_roles(table, audience, *,
  derived_kinds)` and `resolve_overlap(table, rule, audience, *, derived_kinds)`, a
  docstring paragraph on overlap, and their `__all__` entries. Nothing in the
  importer or the table changed.
- `studies/prior_observer.py` (new, R2-owned): `PRIOR_KIND` (`[value, precision]`,
  derived), `prior_registry`, `PriorBelief` (`mean`, `n`, `aux`, `weight`),
  `PriorObserver` (incremental) and `PriorRefitObserver` (incremental and
  refit-capable), the pre-flight functions (`retained_intersections`,
  `boundary_findings`), `AudiencePriorRefit` with its run-document binding
  `audience_prior_binding`, `prior_study`, and the codec kind mapping
  `PRIOR_OBSERVER_KINDS`.
- `core/preflight/findings.py`: `D008`, `D009`, `D010`.
- `core/conformance/observer_kit.py` and its `__init__`: `check_overlap_declared`
  (check 7) and `check_zero_weight_rows`; `check_folded_once` also runs the
  zero-weight check for an observer that declares `reads_priors`.
- `codec/preflight.py`: for a run document checked with `bindings`, the refit
  boundary is rebuilt with the document's record registry (the selected executor's
  side, or both sides merged) and passed in the `PreflightContext`, so `D009` and
  `D010` are reachable through JSON. A study document with a separate run record
  has no registry: nothing is rebuilt and a `D003` warning says so. The docstring
  counts twenty codes (three document codes, seventeen in-memory).
- No core runtime file changed. The core edits are `core/preflight/findings.py` and
  `core/conformance/`, both allowed by K2. `core/battery/` is untouched.

### The prior observer

- A prior is a present row of a derived kind. For audience `a`, its evidence is the
  present raw rows released to `a`; a prior is considered only for the audiences it
  is released to. `EXCLUDE_PRIOR` gives weight 0 to a prior whose pedigree (external
  entries included) names evidence of `a`; `EXCLUDE_ROWS` gives weight 0 to every
  evidence row a considered prior names and keeps the prior. A row `a` never
  receives has weight 0 for `a`. Pure, traceable and order free (it compares
  identities, never positions).
- `init(caps)` installs the observer audience's retained priors and
  `init_summary(caps)` the policy audience's: per subject the precision-weighted mean
  and the total precision (`weight`). The sum runs in the canonical order of the
  source identities, so the installed prior does not depend on the row order bit for
  bit (a three-prior test shows the order would otherwise round differently).
- `fold` uses the observer's weights and `fold_summary` the policy's: an imported raw
  row of weight 0 (matched by channel and local index, which pre-flight keeps unique)
  is not folded. The count leaf `n` counts folded rows only, so the corpus invariant
  that folds do not exceed accepted acquisitions still holds; the prior's precision
  lives in `weight`.
- Prior rows in the log are derived and never folded: a prior counts once, through
  the initial belief. Replay starts from `init` and `init_summary` and reproduces the
  run with no rule change.

### The audience-prior boundary

`AudiencePriorRefit` answers each dirty entry of audience `a` with
`observer.refit_for(a, ...)` started from `observer.prior_belief(caps, a)`. The core's
sharing (`RefitRequest.source`) is honored only for subjects whose audiences resolve
alike (`same_resolution`: equal prior beliefs bit for bit and equal weights on the
public raw rows); elsewhere each audience computes its own entry and `computed` says
so. The boundary reads the priors from the study's observer at call time, so it
cannot disagree with the installed beliefs, and it declares `audience_priors = True`.
It is hashable by class and declaration, so instances share one compiled program.

### Findings

| Code | Raised when | Locator |
|---|---|---|
| D008 | two priors retained for one audience have pedigrees that share a source (external entries count); once per pair | `observer`, `table.pedigree.source_index`, `(later row, entry)` |
| D009 | a retained table prior with `MaskedBatchRefit` of the observer's own kernel (it starts both refits from `observer.init` and shares them) while the audiences resolve differently for some subject | `observer`, `table` |
| D010 | a retained table prior with any boundary other than `AudiencePriorRefit` (A10): `OftiRefit`, a host fit or kernel that never reads the installed prior, the updater boundary when it lands, and the observer-init boundary when the audiences resolve alike; with no boundary supplied and a refresh grid on, a warning that D009 and D010 are unverified | `observer`, `table` |
| D006 | (also) the observer's table differs from the table the process imports | `observer`, `table` |

## Tests

Tests added: 31 functions (46 nodes with parameters), retired: none. Shared builders
are in `tests/core/prior_helpers.py` (a support module; the pre-commit test-naming
exclusion now lists it).

Cold times, each test alone in a fresh process (`SPACEODYSSEY_JAX_CACHE=off`,
`JAX_ENABLE_COMPILATION_CACHE=false`, four threads, wall time including the about
1.5 s import; a parametrized test is timed per node or, where noted, all nodes
together):

| Test | Tier | Cold s | Mutant it kills |
|---|---|---|---|
| `test_premission.py::test_a_disjoint_external_prior_changes_both_beliefs` | unit | 3.3 | prior installed whatever its weight (M5) |
| `::test_exclusion_rules_match_independent_sufficient_statistics` (2) | unit | 3.3, 3.2 | EXCLUDE_PRIOR keeps the prior (M2), EXCLUDE_ROWS keeps rows (M3), dropped rows folded (M6) |
| `::test_a_row_dropped_for_one_audience_is_still_folded_by_the_other` | unit | 4.3 | policy folds with the observer's weights (M7) |
| `::test_overlap_observer_only_row_does_not_move_policy` (K3) | medium | 6.1 | overlap ignores the audience (M1) |
| `::test_overlap_permutation_invariant` (2, K3) | unit | 3.2 | (control on row identity; M1 also fails the policy view) |
| `::test_installed_priors_do_not_depend_on_the_row_order` | unit | 3.1 | priors summed in row order (M8) |
| `::test_retained_priors_disjoint` (K3) | unit | 3.2 | D008 never (M9), D008 ignores external entries (M4) |
| `::test_the_observer_and_the_importer_hold_one_table` | unit | 3.0 | table mismatch unchecked (M23) |
| `test_observer_kit.py::test_the_prior_observer_passes_overlap_and_zero_weight_checks` (2) | unit | 3.6 per node | M1, M2, M3, M5, M6 |
| `::test_overlap_declared_controls` (3) | unit | 3.1 for three | kit audience test off (M16), double-weight test off (M17), disjoint-pedigree test off (M18) |
| `::test_zero_weight_rows_control_an_observer_that_ignores_weights` (2) | unit | 3.8 for two | zero-weight prior part off (M19), fold part off (M20) |
| `::test_the_prior_checks_refuse_an_observer_that_reads_no_priors` | unit | 2.7 | protocol unchecked (N5) |
| `test_premission_refit.py::test_the_audience_prior_boundary_meets_its_contract` | medium | 7.6 | boundary honors the core's sharing (M13), starts from `observer.init` (M14) |
| `::test_d009_refuses_different_priors_on_the_observer_init_boundary` | unit | 3.2 | D009 never (M10), `same_resolution` always true (M15) |
| `::test_d010_refuses_a_table_prior_with_a_boundary_that_ignores_it` (3) | unit | 4.5 (OFTI), 2.9 for the others | D010 never (M11), refresh-off guard removed (N4) |
| `::test_a_policy_only_row_is_judged_by_the_policy_rule` | unit | 3.5 | unreceived rows dropped (N1), resolution compared on public rows only (N2) |
| `::test_without_a_boundary_d009_and_d010_are_reported_unverified` | unit | 2.5 | unverified warning dropped (M12) |
| `::test_a_refit_observer_keeps_its_prior_across_a_refresh` (B3) | medium | 18.2 | M13, M14, folded-once not extended (M21); control: a flat-start refresh |
| `::test_replay_reproduces_prior_run` (named T4c check) | medium | 21.6 | M13, M14, M1 |
| `test_preflight.py::test_seeded_mistake_in_context_is_caught` (2) | unit | 3.4 for two | M10, M11 |
| seeded cases `priors_summarize_one_source` (D008), `prior_refit_boundary_unverified` (D010 warning) in the existing seeded, locator and count tests | unit | under 0.1 each | M9, M12 |
| `test_preflight_json.py` case `D008` (seeded and repaired) | unit | 2.5 | M9, M4 |
| `::test_the_run_record_boundary_reaches_the_prior_checks` (3) | unit | 3.4 for three | JSON boundary not rebuilt (M22) |
| `::test_the_selected_executor_side_is_checked_and_both_when_none_is` (3) | unit | 3.3 | executor side ignored (N3) |

The medium tests compile the refit study once per executor (host about 11 s, compiled
about 13 s cold); the two run tests each compile one executor so each stays under
30 s alone. The contract test is medium because five boundary programs and the eager
import take about 8 s cold. `test_overlap_observer_only_row_does_not_move_policy`
handles two imports event by event (about 6 s). `AudiencePriorRefit` is hashable by
class and declaration, so the host, compiled and replay calls share one boundary
program.

## Checks run

- `tests/core/test_golden_histories.py` with `tests/codec/test_run_document_runs.py`
  (golden pin and corpus document hash pin) at the final tree: 189 passed in 150.9 s.
  No pin file changed (`git diff 7fe3e24 -- tests/core/golden_pin.json
  tests/codec/hash_pin.json` is empty).
- Quick tier (`-m "not slow" -n 2 --dist loadfile`, entry-point test deselected as in
  R0a), at the tree after the review fixes: 2,055 passed in 207.0 s, no budget
  violation.
- Touched files (`test_premission.py`, `test_premission_refit.py`,
  `test_observer_kit.py`, `test_preflight.py`, `test_preflight_json.py`,
  `test_lints.py`): none is in the slow tier, so all of them, medium included, ran
  in the quick tier above; before the review they also ran alone (466 passed).
- `ruff check`, `ruff format --check`, `tools/check_internal_refs.py`, the
  domain-word and core-import lints, and `pre-commit run` over the branch: clean.
  Every touched file is ASCII.
- The full suite and the Sphinx docs build were not run.

## Review round 1

A fresh-eyes review found no P1 and four P2s:

1. The observer-init boundary is accepted with equal resolutions, looser than D2's
   wording. Kept and recorded (silence 1).
2. `same_resolution` compared only public rows: a policy-only raw row was dropped
   from the policy refit of the observer-init boundary (the observer's weight there
   is 0) with no `D009`. Fixed: `dropped(caps, audience)` drops only rows the
   audience receives, and the resolutions are compared on every row the policy
   receives (silence 7). Test: `test_a_policy_only_row_is_judged_by_the_policy_rule`.
3. The JSON executor-side selection was untested. Test added
   (`test_the_selected_executor_side_is_checked_and_both_when_none_is`).
4. The kit's duck protocol was undocumented. Documented and refused when incomplete.

Also fixed from its P3s: a NaN-safe weighted sum in `prior_belief`; no boundary
finding when no checkpoint can run; a NaN-tolerant table comparison. Kept: weights
recomputed per fold, plain int fields (as `ToyRunningMean`), no `lax.cond` in the
boundary lane, D006 for the table mismatch.

The scoped second round confirmed the fixes and found no new problem.

## Review round 2

An independent tier-2 review returned "fix first" with five P2s. Each fix has a test
that failed before it.

1. **The D009/D010 split did not follow the adopted A10.** The observer-init
   boundary with equal resolutions was accepted. Now only `AudiencePriorRefit`
   takes a retained table prior; that case is `D010` (silence 1 records it as a
   candidate spec change). Tests: `test_d009_refuses_different_priors_on_the_observer_init_boundary`
   and `test_a_policy_only_row_is_judged_by_the_policy_rule` now require `D010` for
   equal resolutions.
2. **The zero-weight probe row had audience 0** (from `record_none`), so an observer
   that ignores the weights but folds only rows addressed to the audience passed
   both kit checks, while with the real public audience the excluded row moved the
   posterior from -5 to -0.5. The probe now carries the import's and release's
   metadata. Test: `test_the_zero_weight_probe_carries_the_rows_real_audience`
   (control `GuardsByAudience`).
3. **Non-binary weights passed.** NaN for dropped rows passed both checks (`> 0` in
   one, `== 0` in the other) while the observer double counted (means 1/3 and -0.5
   against -5). Both checks now first require a finite {0, 1} vector of one entry
   per row. Test: `test_the_prior_checks_refuse_weights_outside_zero_and_one`
   (control `NanWeights`, both rules).
4. **Vacuous passes.** An all-padding table passed both checks. Check 7 refuses a
   table with no prior, and the zero-weight check a table with no row of weight
   zero. Test: `test_the_prior_checks_refuse_a_table_with_nothing_to_judge`.
5. **The JSON boundary rebuild called builders with `registry=None`** on the study
   document plus `run=` path, so a builder that reads the registry raised
   `AttributeError`. Boundaries are now rebuilt only with a run document's
   registry; the other path reports a `D003` warning at `/run/refit` and the
   observer's unverified warning. Test:
   `test_a_study_document_with_a_run_record_does_not_rebuild_without_a_registry`
   (the same registry-reading binding rebuilds inside a run document and reaches
   `D010`). The D009 and D010 JSON tests now use run documents
   (`dump_run`), and the executor-side test selects host or compiled on one.

Tests added in this round: 4 functions (5 nodes); 3 existing tests changed to the
adopted contract or to run documents. Cold times, each alone (caches off, four
threads, wall time with imports):

| Test | Tier | Cold s |
|---|---|---|
| `test_observer_kit.py::test_the_zero_weight_probe_carries_the_rows_real_audience` | unit | 4.4 |
| `::test_the_prior_checks_refuse_weights_outside_zero_and_one` (2) | unit | 2.9 per node |
| `::test_the_prior_checks_refuse_a_table_with_nothing_to_judge` | unit | 3.6 |
| `test_preflight_json.py::test_a_study_document_with_a_run_record_does_not_rebuild_without_a_registry` | unit | 3.9 |
| `::test_the_run_record_boundary_reaches_the_prior_checks` (3) with `::test_the_selected_executor_side_is_checked` (2) | unit | 3.4 for all five |
| `test_premission_refit.py::test_d009_refuses_different_priors_on_the_observer_init_boundary` | unit | 3.5 |
| `::test_a_policy_only_row_is_judged_by_the_policy_rule` | unit | 3.6 |
| `::test_a_refit_observer_keeps_its_prior_across_a_refresh` (rerun, uses the changed kit) | medium | 17.9 |

Checks rerun at `aab7092`:

- `test_golden_histories.py` with `test_run_document_runs.py`: 189 passed in
  182.6 s; `git diff 7fe3e24 -- tests/core/golden_pin.json tests/codec/hash_pin.json`
  is empty.
- Quick tier (`-m "not slow" -n 2 --dist loadfile`, entry-point test deselected):
  2,059 passed in 207.5 s. The budget plugin reported one unit over 5 s, the
  existing `test_snapshot_fingerprint.py::test_snapshot_fingerprint_recorded`
  (5.48 s under the parallel load), not touched here.
- ruff, the internal-reference, domain-word and core-import checks, and
  `pre-commit run` over the branch: clean.

## Mutants and controls

Each mutant was applied to a copy of `src/` and the new and touched tests run against
it (`$TMPDIR` script, not committed); the medium tests were included for the mutants
only they reach (M1, M7, M13, M14, M21). All 28 are killed; the killing tests are in
the table above.

| Mutant | Where |
|---|---|
| M1 overlap ignores the audience | `resolve_overlap` |
| M2 EXCLUDE_PRIOR keeps the prior; M3 EXCLUDE_ROWS keeps rows | `resolve_overlap` |
| M4 D008 ignores external entries; M9 D008 never | `retained_intersections` |
| M5 prior installed whatever its weight; M8 priors summed in row order | `prior_belief` |
| M6 dropped rows folded; M7 policy folds with the observer's weights | `fold_for`, `fold_summary` |
| M10 D009 never; M11 D010 never; M12 unverified warning dropped; N4 refresh-off guard removed | `boundary_findings` |
| M13 boundary honors the core's sharing; M14 boundary starts from `observer.init` | `AudiencePriorRefit` |
| M15 `same_resolution` always true; N1 unreceived rows dropped; N2 resolution on public rows only | `same_resolution`, `dropped` |
| M16 to M20 each part of the two kit checks off; M21 `check_folded_once` not extended; N5 protocol unchecked | `observer_kit.py` |
| M22 JSON boundary not rebuilt; N3 executor side ignored | `codec/preflight.py` |
| M23 observer and importer tables not compared | `_table_mismatch` |

Shipped controls: `IgnoresPrior` (fails B2 (a)), `RuleOff` (double counts, fails the
exclusion statistics and check 7), `TableWide` (fails the audience test of check 7
and moves the policy summary), `IgnoresWeights` (passes check 7, fails the zero-weight
check and folded-once), an intersecting-summaries table (fails check 7 and is
`D008`), `SharesAnyway`, `StartsFromInit` and `MaskedBatchRefit()` (fail the
contract), `DropsPrior` (fails the run comparison).

## R1 open points, answered

1. **The policy copy keeps `accepted_tick = -1` (D5).** The prior observer reads its
   priors from its own table, never from the log rows, so neither copy's stamping
   reaches a belief. Replay agrees (it starts from `init` and `init_summary`).
2. **Split prior rows share one identity.** On a run with a split public prior, the
   kit's `check_folded_once` (with its zero-weight extension), `check_derived_summary`
   (`summary_kind="prior"`) and keyed replay all pass, so the pair is accepted. The
   corpus invariants (`check_run`) were not run on a prior study: they need the
   counting-observer corpus harness, which no prior study is part of.
3. **Window columns.** Not written into records here. `resolve_overlap` is traceable,
   so the importer can call it inside `handle` with the event's audience to write the
   "no window" value for an audience that excludes the prior; that is R4b's, with the
   survey 3 window kind.
4. **The JSON refit-boundary context.** Wired (above).

## The T3b derived-summary fixture

`check_derived_summary` runs on the prior refresh run with the prior kind as the
summary: the belief of every subject with a prior row equals a refit from
`observer.init` of the rows, the prior rows left out, so the prior row in the log is
not counted a second time. It does not cover the policy summary (the kit check reads
the observer's belief only), and its reference starts from `observer.init`, so it
trusts the installed prior; the independent prior-plus-rows comparison in
`test_a_refit_observer_keeps_its_prior_across_a_refresh` covers both audiences.

## Spec silences decided here

1. **Only `AudiencePriorRefit` takes a retained table prior (adopted A10).** Every
   other boundary is refused: `D009` when it is the observer-init boundary and the
   audiences resolve differently, `D010` otherwise. Round 1 had accepted the
   observer-init boundary when the audiences resolve alike; round 2 reverted that to
   the adopted contract (see "Review round 2"). That case looks safe for the current
   Gaussian observer (it starts both refits from `observer.init`, which then holds
   the one common prior, and no wrong-posterior counterexample was found) and is a
   candidate for a later, explicit spec change, not a silence here.
2. **`resolve_overlap` takes `derived_kinds` (keyword).** The frozen signature cannot
   tell a prior from a raw row without the registry; a prior is a row of a derived
   kind (R1 silence 4). `overlap_roles` is added beside it for the pre-flight check.
3. **The prior payload is `[value, precision]`**, the precision in units of one
   measurement (a one-column payload means unit precision). `PRIOR_KIND` and
   `prior_registry` register it after the refresh product kind (indices 4 and 3).
4. **The installed precision is its own leaf (`weight`)**; the count leaf `n` keeps
   counting folded rows, so the core invariant that folds do not exceed accepted
   acquisitions holds for prior studies.
5. **Weights are recomputed from the observer's fields** (table and rule) where they
   are used, not stored: there is no second copy to disagree. Cost is O(N^2 W) per
   fold, fine at toy scale; a large exported table would want them precomputed.
6. **The observer holds its own copy of the table**; pre-flight refuses one that
   differs from the importer's, as `D006` (a row the study cannot hold), not a new
   code (the codes are frozen to D010).
7. **A row an audience never receives is not dropped for it.** Weight 0 then means
   "not received"; the fold drops only rows the rule drops for that audience. This
   keeps the observer's own refit correct on policy-only rows.
8. **D008 is reported once per pair of rows**, at the later row's first shared
   entry, even when both audiences retain the pair.
9. **D010 without a boundary is a warning** ("D009 and D010 are unverified"), raised
   only when a table prior is retained and a refresh grid is on, like D004's
   "declared, unverified". With no checkpoint possible, neither code fires.
10. **The boundary reads the priors from the study's observer** at call time rather
    than holding them, so it is stateless like every boundary and cannot disagree
    with the installed beliefs.
11. **The JSON path checks the selected executor's boundary**, or both sides merged
    when none is selected; a side whose binding does not rebuild is skipped (its
    own `D001` or `D003` reports why). Boundaries are rebuilt only with the run
    document's record registry, as `load_run` rebuilds them; a study document with
    a separate run record gets a `D003` warning instead.
12. **The kit's prior checks use a duck protocol** (`reads_priors`, `table`,
    `weights(caps, audience)`, `init`, `init_summary`), documented in
    `_prior_reader` and refused when incomplete; core imports nothing from
    `studies/`. The kit computes each audience's priors and evidence from the table's
    columns itself, never from the observer's resolution. Weights that are not a
    finite {0, 1} vector fail both checks; check 7 refuses a table with no prior and
    the zero-weight check a table with no row of weight zero (inside
    `check_folded_once` such a table adds no failure). The zero-weight probe row
    carries the import's metadata: the row's audience, its span, availability at
    tick 0 and the release stamps of the audiences it is addressed to.

## Spec points not satisfied

- Survey 3 is R4b's: the replay check runs on the toy prior refresh study, not on
  survey 3 as D2's sentence names it. The decision-level part of K3's hidden-row
  check (equal policy decisions) is covered here by the policy snapshot fingerprint
  and released rows at tick 0; the toy plan reads no belief, so decision equality
  across a run is left to survey 3 (d).
- `test_replay_refresh.py` is not extended: the replay check lives in
  `test_premission_refit.py` with the run it replays, so the slow file does not
  compile the prior study.
- The "updater boundary" does not exist on `main` (T7 is unmerged); its refusal is
  tested through a host fit and a masked kernel that never read the installed prior,
  which take the same `D010` branch.
- Windows are not written into records (R1 open point 3, left to R4b).
