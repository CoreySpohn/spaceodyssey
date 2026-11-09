# Slice 1, T10: conformance corpus

## Behaviour changes

- `spaceodyssey.core.conformance` adds the machinery to run a corpus case and judge it.
  `CorpusCase` (moved from the test tree, with two optional fields: `hand_rows`, the
  per-row `(lane, subject, acquisition_index, n_written, wake)` beside the hand history,
  and `make_service`, which builds a fresh host answerer for each run). `run_case(case,
  executor, ...)` runs a case on `"host"` or `"compiled"` and returns one `CaseRun` per
  world: the final carry, the trace, the canonical history, the policy view of each
  decision (host only) and, with `probe`, the wake and decision counts of the same
  worlds with private events added (`PrivateProbe`, a wrapper process that adds
  observer-only events writing nothing).
- `spaceodyssey.core.conformance.invariants` holds ten invariants. Each takes a
  `CaseRun` and returns a `CheckResult` (name, pass flag, diagnostic, detail):

| Invariant | Fails when |
|---|---|
| `order_keys_monotone` | handled rows come out with a key below the one before, in `(tick, stage, micro, tie, id)` |
| `exactly_once_settlement` | `double_settle` is set, or an acquisition has two revision-0 records |
| `one_fold_per_acquisition` | `dup_fold` is set, or a belief holds more folds than accepted (or released) completed acquisitions |
| `no_violation` | `violation` is set |
| `balance_conserved` | initial balance minus the net charge of every logged record and the full cost of each running acquisition differs from the final balance in any row and resource (1e-9) |
| `unique_acquisition_id` | two records share an acquisition id and a revision |
| `dense_acquisition_index` | per subject, the admitted acquisitions (logged records of a channel kind, one per id, plus completions still queued) repeat an index, exceed the count, or leave a gap |
| `release_not_before_available` | a record was released before its available tick |
| `no_scorer_column_in_snapshot` | a payload column an audience may not read is nonzero in its final snapshot or in a decision view |
| `no_extra_wakes_under_private_events` | the probe run has more wakes or decisions than the run without it |

- The corpus is now `spaceodyssey.studies.conformance` (`CORPUS`, `CORPUS_BY_NAME`,
  `HAND_HISTORIES`, `HAND_ROWS`, `HAND_STUDIES`, `HAND_INVARIANTS`, `frozen`, `stop_grid`).
  `ScriptedRecords` and `scripted_study` moved from the record test helpers to
  `spaceodyssey.studies.scripted`, since the revision case needs them. The test tree
  imports both through `tests/core/corpus.py` and `tests/core/record_helpers.py`.

## New cases

Each has a hand-written history and hand rows, checked on both executors.

| Case | What it pins |
|---|---|
| `coalesced_release` | a bundle at the first decision whose two records (lanes 0 and 2) release at 130: one decision sees both, six decisions in all |
| `bundle_unaffordable` | a bundle of two with 1.5 of the 2 it needs left in the program: both rows refused for funds, nothing kept, the run ends idle |
| `forbidden_effect` | a phase that permits Observe only: a Stop at the second decision and the closing Stop are refused as forbidden, the plan runs between them |
| `catalog_record` | the second record kind: the catalog row is accepted by the observer at 325, never released to the policy, and wakes no decision |
| `host_service` | a question at 130 is a request row, answered by a result at the same tick and stage; the answer wakes the decision that resumes the plan |
| `revision` | two revisions of one acquisition at 100 and 200: the first is marked superseded, the observer folds one value |

## Tests

- `test_corpus_invariants.py`: every invariant over every case (28 cases) on both
  executors, two worlds each; hand history, hand rows and case checks on both executors;
  host and compiled histories equal; a planted violation for each invariant that fails
  it while the unplanted run passes; a wake rule that ignores audience (the existing
  `leaky_wake`) caught by the wake invariant on both executors; an unprobed run cannot
  pass it; an overflow run is judged on what it kept.
- One source for the parity check: `test_parity_corpus.py`, `test_corpus.py`,
  `test_replay_log.py`, `test_preflight.py`, `test_scorer_columns.py` and
  `test_acquisition_index.py` read `CORPUS`, so they run over the new cases too. A case
  with a host service gets a fresh service in each run (`helpers.service_of`).

## Rebaselined cases

None. The 22 earlier cases keep their studies, hand histories, rows, admission codes
and decision counts; only the dataclass gained two defaulted fields.

## Deviations from the spec

- The case catalogue is in `studies/conformance.py`, not `core/conformance/corpus.py`.
  The cases build toy studies, the core may not import `studies/`, and its lints bar
  the vocabulary of the cases. `core/conformance/corpus.py` holds `CorpusCase`, the
  runner and the probe; `invariants.py` is where the spec put it.
- The leak scenarios and draw-invariance perturbations stay in `tests/core/corpus.py`:
  they are test fixtures that perturb the corpus studies, not cases.
- "Keys monotone in `(tick, stage, microstep, tie, stable_id)`" is read off the trace of
  handled rows; the neutral tie rank holds everywhere in the corpus.
- `balance_conserved` and the exact-density half of `dense_acquisition_index` are
  skipped (reported as `skipped` in the detail, or judged on bounds only) when a run
  overflows its log, since dropped records cannot be accounted for. The two overflow
  cases are the only ones.
- The index invariant judges only subjects with a nonzero acquisition count and only
  records of a kind some channel produces, so rows a process or a host service writes
  directly are left out; the older per-index test skips the host-service and revision
  cases for the same reason.
- `test_snapshot_hides_scorer_columns` skips the host-service and revision cases: they
  write no scorer-only value, so its non-vacuity check cannot hold; the snapshot
  invariant covers them.

## Left for later tasks

- The decision views checked for hidden columns exist only on the host executor; the
  compiled executor is judged on its final snapshots.
- The probe adds ten silent events at a fixed period; a study whose own events sit at
  those ticks is still covered, but no probe varies the period.
