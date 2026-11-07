---
orphan: true
---

# Slice 2b R0a: pins and interface freeze

The first task of Slice 2b, done together with Slice 3a S0 (`slice-3a-s0.md`), which
shares the pins. It changes no library code. It pins the golden histories and the
corpus document hashes from a recorded commit, reads the merged replay and the OFTI
adapter, and freezes the pre-mission table contract that R1 to R6 build against.

## Commits

1. `test(core): pin the golden histories, corpus fingerprints and a fault fixture` (`54475fa`)
2. `test(codec): pin the content hash of every corpus run document` (`8a40926`)
3. `test(core): an empty fingerprint leaf moves the digest` (`9666a72`)
4. `chore(tools): step-cost probe for the compiled executor with a calibrated extra-pass mutant` (`ba822c7`)
5. this record and `slice-3a-s0.md` (`9731288`)
6. `ci(pilot): dispatch-only runner pilot for step cost and cold compiles`, with these
   record updates and the codespell word `premission` (see `git log`)

## The pins

Source commit: `96a644728386365177e79f8bca315a746ef2901c` (`main`, "docs(changes): PT3
review fixes"). The pins were captured on a branch whose library code is that commit's,
unchanged; only test files and this record were added. Before capture, the 60 tests of
`tests/core/test_golden_histories.py` passed at that commit on this machine (288.9 s),
so every manifest entry was reproduced, not copied.

Platform: macOS 26 arm64, Python 3.12, JAX 0.12. The float digests of the golden
histories are platform dependent (they differ on the Linux worker), so the history
pin, like the manifest it copies, holds for this platform. The integer digests, the
snapshot fingerprints and the document hashes do not depend on float rounding.

| Pin | File | Count | Test |
|---|---|---|---|
| Golden history digests (`ints`, `floats`) | `tests/core/golden_pin.json` `histories` | 116 (29 cases, two executors, two worlds) | `test_the_manifest_equals_its_pin` |
| Corpus snapshot fingerprints | `golden_pin.json` `fingerprints` | 116 | `test_fingerprints_match_their_pin` |
| Fault fixture history digests | `golden_pin.json` `fault` | 4 (two executors, two worlds) | `test_the_fault_fixture_matches_its_pin` |
| Corpus run document content hashes | `tests/codec/hash_pin.json` `documents` | 58 (29 cases, two executors) | `test_every_corpus_run_reloads_to_the_same_hash_and_history` |
| Schema version | `hash_pin.json` `schema_version` | 1 | `test_the_hash_pin_names_its_commit_and_covers_every_corpus_document` |

Rules:

- The pins extend the existing tests; there is no second harness. The history pin is
  compared with `golden_histories.json`, which the existing tests compare with the
  runs, so a moved history fails twice: against the manifest and against the pin.
- Nothing regenerates a pin. `SPACEODYSSEY_WRITE_GOLDEN=1` rewrites the manifest only,
  and the pin then fails, documented or not. A pin is written once, by running its
  test file with `SPACEODYSSEY_CAPTURE_PIN=<full sha>` while the pin file is absent; the
  capture refuses to replace an existing file. A deliberate re-baseline is therefore
  a deleted and recaptured pin file, visible in the diff.
- Fingerprints, per corpus run: the digest of the snapshot fingerprints the trace
  recorded at its woken steps, their count, and the fingerprints of the final carry's
  policy and observer views (`tests/core/conformance_helpers.py::run_fingerprints`).
- The fault fixture: the corpus never handles a `FAULT` row, so the toy survey-1 study
  gets a process that seeds a live fault on the first lane-0 occupancy (tick 50,
  generation 1) and a stale one (tick 200, same generation). Evidence asserted: two
  handled faults, one failed record, completed records after it. Control: the same
  fixture with a lane `Stop` at each fault (the "lane closure on `FAULT`" mutant) moves
  every pinned integer digest and leaves no completed record.
- Control for the document pin: one changed declaration value (the horizon) moves the
  document hash.

## What R0a read

### Keyed replay (`core/executors/replay.py`, merged with T4c)

- `replay_run` starts from `observer.init(caps)` and `observer.init_summary(caps)`
  (`replay.py:285-286`), so a prior installed in the initial beliefs is reproduced
  with no rule change.
- A derived row is never folded: `replay_belief` skips it (`replay.py:109`) and
  `_release` returns before any fold (`replay.py:382`).
- A release is matched to its log slot by `(slot, subject, acquisition_index,
  channel)` (`_check_release`, `replay.py:55-64`), and every fold is keyed by
  `fold_key` (`keys.py:137-151`), whose coordinates are subject, channel, acquisition
  index, phase, revision and status (`keys.py:115-134`). No queue position enters.
  Keyed replay therefore needs no extra declared index: the process-written record's
  `acquisition_index` is the semantic index, and `local_index` is written there (the
  D3 dependency check, answered).
- T4c and T3b are merged on `main`, so the named interface check
  `test_replay_reproduces_prior_run` (owned by R2) runs when R2 lands and is no longer
  pending. The T3b derived-summary checks that R2 extends are
  `tests/core/test_observer_kit_corrections.py::test_a_derived_summary_adds_nothing_to_the_belief`,
  its control `test_derived_summary_control_the_same_summary_registered_as_non_derived`,
  and `test_a_summary_reissued_under_its_own_id_is_refused_as_a_correction`.

### The OFTI adapter (`adapters/photomancy.py`)

- `OftiRefit.__init__` takes a named eccentricity prior (`ecc_prior`) and builds
  `ofti_model`, whose `prior` is a fixed declaration (`photomancy.py:91-133`).
- `InferenceModel.digest` hashes that declaration's JSON (`photomancy.py:76-81`);
  `OftiRefit.refit_batch` (`photomancy.py:237`) reads `requests` and the study's
  payload audiences, never `observer.init` or any table. Hashing a table into the
  digest would not make OFTI consume it. D010 refuses the combination, as the spec
  states; no OFTI change.
- Refresh starts both audiences' refits from `observer.init`
  (`refresh.py:415`, `refresh.py:592-600`), and `RefitEntry` has no audience field
  (`refresh.py:187-193`); the audience is the leading axis of `RefitRequest`
  (`refresh.py:197-214`), which is what the R2 toy audience-prior boundary selects on.

## The frozen contract

R1 to R6 build against this. A change to any item is a spec change, not an
implementation choice.

### Table schema (`studies/premission.py`)

```python
class SourceId(eqx.Module):
    namespace: I64[N]  # 0 for this table, k for the k-th imported run, -1 pads
    producer: I64[N]  # the source's channel in its own namespace
    subject: I64[N]
    source_index: I64[N]
    revision: I64[N]  # recorded for lineage, not part of overlap


class PreMissionTable(eqx.Module):
    source: SourceId
    local_index: I64[N]
    channel: I64[N]
    kind: I64[N]
    audience: I64[N]
    start_offset: I64[N]
    end_offset: I64[N]
    window_lo_rel: I64[N]
    window_hi_rel: I64[N]
    payload: F64[N, P]
    pedigree: SourceId  # shape [N, W]
    external: B[N, W]
    n_pedigree: I64[N]
    upstream_hash: str


class PedigreeRule(IntEnum):
    EXCLUDE_PRIOR = 0
    EXCLUDE_ROWS = 1


def resolve_overlap(table, rule, audience): ...  # pure; weights in {0, 1} per row
```

Overlap is equality of `(namespace, producer, subject, source_index)`; the revision
never enters it. Offsets are at or below zero and relative to the row's release tick;
windows are relative to the row's `released_tick`.

### Finding codes

New members of `core/preflight/findings.py::FindingCode`, continuing the `D` series of
the codec's `DocumentCode` (`D001` to `D003`, `codec/preflight.py:112-114`) with no
overlap:

| Code | Finding | Needs context |
|---|---|---|
| D004 | declared `upstream_hash` differs from the supplied upstream document's hash | upstream document; "declared, unverified" without it |
| D005 | an in-table pedigree entry (`external` false) names a source absent from the table | no |
| D006 | two rows share a source identity, a `local_index` repeats within a channel, or `n_pedigree` exceeds the width | no |
| D007 | an offset is positive, or a start exceeds its end | no |
| D008 | two retained priors for one audience have intersecting pedigrees | no |
| D009 | a refit-capable observer whose audiences resolve to different priors on a boundary that does not declare audience priors | boundary |
| D010 | a table prior with a refit boundary other than the toy audience-prior boundary (`OftiRefit`, the updater boundary) | boundary |

Slice 3a adds `P001`, `W001` and `W002` to the same enum; no code collides. The
codec docstring that counts "thirteen finding codes" (`codec/preflight.py:3-4`) is
updated by the task that adds the codes.

The pre-flight entry point today is `preflight(study, executor=None)`
(`core/preflight/checks.py:415`). D10's context is added as a keyword after it:
`preflight(study, executor=None, context=None)`, so every existing call keeps its
meaning.

### The `local_index` rule

`local_index` is supplied, never computed from the imported subset: an authored table
gives it explicitly and the exporter sets it to the source's own upstream acquisition
index. It is unique per channel (D006). The importer writes it as the record's
`acquisition_index`, which is what `fold_key` and replay read (above). Inserting or
removing an observer-only source moves no public `acquisition_index`, fold key,
snapshot or fingerprint.

### File ownership

| Path | Owner | Others |
|---|---|---|
| `studies/premission.py` | R1 (integration owner after R1) | R2 and R3 edits serialized through the owner |
| `studies/prior_observer.py` | R2 | |
| `studies/precursor.py` | R3 | |
| `studies/survey_{3,4,6,18}.py` | R4b (3, 6), R5 (4, 18) | |
| `core/battery/metamorphic.py`, `core/battery/views.py` | R4a | |
| `core/conformance/observer_kit.py` | R2 | |
| `core/preflight/findings.py`, `core/preflight/checks.py` | R1 (hook, D004 to D007), then R2 (D008 to D010) | Slice 3a adds `P001`, `W001`, `W002`; one integration owner merges both slices' edits |
| `codec/preflight.py` | R1 (mapping), then R3 (D004) | Slice 3a S2, S3a, S5 add their mappings through the same owner |
| `codec/kinds.py` | R1 (table registration) | |
| `codec/load.py`, `codec/dump.py` | Slice 3a S1 (optional fields with defaults) | 2b needs no edit here (the table is a study field through the existing walk, D6) |
| CI files of Q9 (`pyproject.toml`, `tests/conftest.py`, `.github/workflows/ci.yml`, `.github/scripts/ci_summary.py`) | R6 | |

The pins (`golden_pin.json`, `hash_pin.json`) are shared by both slices and are
never edited by either.

## Compiled signatures and the CI-runner pilot

### Inventory

A compiled signature is one program per executor: a study whose pytree structure or
static fields differ from every study compiled before. Changing array values (an
epoch shift moves `horizon_tick` and `phase_until`, both arrays) reuses it.

| Source | New signatures | Executors | Tier |
|---|---|---|---|
| R1 importer study (survey 1 with the importer process), with and without a zero-information import, stateful policy | 2 (importer process; stateful policy) | both | medium |
| R2 prior observer on the toy | 1 | both | medium |
| R2 toy audience-prior refit boundary (refresh driver) | 1 | both | medium |
| R4a synthetic negative-time fixture | 1 | both | medium |
| Survey 3 | 1 (shares R2's observer type with a window policy) | both | medium |
| Survey 4 (with and without the precursor) | 1 (the table is array data) | both | medium |
| Survey 6 | 1 | both | medium |
| Survey 18 (own truth, channel and observer types) | 1 | both | medium |
| Controls needing an extra signature (truth-reading policy, weight-ignoring observer, absolute-tick value) | 3 | one | slow |

That is 9 medium signatures on two executors (18 cold compiles) and 3 slow ones,
against the spec's estimate of about 7. Local figures with the persistent cache off,
four threads, on a machine shared with other jobs: the compiled executor's cold first
call of the toy study is 6.3 s (`tools/step_cost.py`, `plain.cold_s`); the survey 5
test file, a medium survey test of the kind 2b adds, takes 29.1 s for six tests, its
both-executor round-trip test 8.7 s. Taking 6 to 9 s per signature on both executors,
the medium additions cost about 55 to 80 s locally, or roughly 160 to 240 s on the
runner at its threefold slowdown (E0) when compiled cold.

Risk: that cold figure exceeds K6's 60 s relative limit several times over. K6 as
written compares against the Slice 2 close-out summary "same runner, caches
disabled"; whether the medium survey tests fit depends on the pilot figure, and if they
do not, either some survey tests move to the slow tier or K6 is read with the compile
cache warm. That is a decision for the R6 owner and Corey, made on the pilot figure.

### The CI-runner pilot: `.github/workflows/pilot.yml`

The pilot is measured on the hosted runner by a dispatch-only workflow this branch
adds, `.github/workflows/pilot.yml` (`workflow_dispatch` only; read-only token;
pinned actions; no secrets; `zizmor` clean). It installs like the quick tier of
`ci.yml` with the JAX compile-cache restore removed, `SPACEODYSSEY_JAX_CACHE=off` and
`JAX_ENABLE_COMPILATION_CACHE=false`, so every program compiles cold, then:

1. `python tools/step_cost.py --mutant --repeats 30`: the compiled executor's cold
   first call and the per-step median and minimum, plain and mutant (the K3 baseline
   of Slice 3a S0 and its calibrated control).
2. Each medium study test file in its own fresh process
   (`tests/studies/test_survey_5.py`, `test_survey_2.py`,
   `test_survey_2_calibration.py`, `-m "not slow"`): the cold cost of the compiled
   signatures they need on both executors, the pattern every 2b survey and fixture
   in the inventory above follows. The 2b signatures themselves do not exist until
   R1 to R5 land.
3. `.github/scripts/pilot_summary.py` prints both results as tables to the log and
   the job summary, and fails when an output is missing.

The figures in this record are local. The CI-runner figures come from the first run of
that workflow, dispatched on `main` right after this branch merges, and are recorded
with the run id in R6's change record (the signature cost, against K6's 60 s and K7's
540 s, replacing the local estimate above) and in S5's (the K3 baseline).

## Checks run

- `tests/core/test_golden_histories.py` (pins captured in this run, then rerun):
  124 passed in 224.6 s (60 existing, 64 new: coverage, manifest against pin, 58
  fingerprint cases, fingerprint evidence, two fault cases, the closure control).
  Slow tier, as the whole file already is.
- `tests/codec/test_run_document_runs.py`: 65 passed in 145.5 s (two new tests, and the
  58 reload cases now also assert the pinned hash). Slow tier, as before. The hashes
  were also recomputed in a separate process and matched.
- `tests/core/test_snapshot_fingerprint.py::test_an_empty_leaf_moves_the_fingerprint`:
  2.3 s cold (`SPACEODYSSEY_JAX_CACHE=off`, four threads; 4.3 s with imports), quick
  tier. The only new quick-tier test; every other new test is in a file already in
  the slow tier.
- Quick tier (`-m "not slow" -n 2 --dist loadfile`): 1,949 passed, 6 failed in
  276.6 s. None of the six touches this change: five are strict-xfail anchors in
  `tests/adapters/test_jaxedith_summary.py` that now pass because the jaxedith and
  optixstuff fixes merged upstream (fixed on another branch), and
  `tests/codec/test_entry_points_installed.py` fails because the shared environment's
  installed package metadata predates the `refresh_schedule` entry point (a reinstall
  fixes it; no package metadata changed here). The budget plugin reported four
  existing unit tests over 5 s, none of them new.
- The Sphinx docs build was not run: the shared environment lacks `sphinx-autoapi`.
  The two records are orphan pages like every other change record.
- `ruff check`, `ruff format --check`, `tools/check_internal_refs.py`: clean.

## Spec silences decided here

- "Fingerprints" pinned per corpus run are the decision fingerprints (as one digest
  and a count) and the final policy and observer view fingerprints. Rejected: pinning
  every decision fingerprint as its own entry (the decision fingerprints are also in
  the history's integer digest; the digest form keeps the pin readable).
- The fault fixture is survey 1 with a seeding process, not a new corpus case. Adding
  a corpus case would add golden entries and change `CORPUS`, which the pins freeze.
- The pin is written by an explicit capture mode that refuses to overwrite. Rejected:
  a write mode like `SPACEODYSSEY_WRITE_GOLDEN`, which would let a re-baseline pass
  silently.
- `preflight` keeps its `executor` argument; the context is a new keyword.
