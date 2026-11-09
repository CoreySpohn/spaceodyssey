---
orphan: true
---

# Slice 2 T4b: pedigree, revision sets and producer-local identity

T4b records what each refresh installed. Every product row carries a pedigree: the
`(channel, acquisition index, revision)` triples of the rows the entry was computed
from. The revision-set rule, derived kinds, producer-local semantic indices, reserved
internal ids for product rows, the extended pre-flight channel check and the refresh
conformance invariants arrive together. A study without a refresh schedule keeps its
digests: only the host-service and revision corpus histories and the new refresh
entries moved (see Re-baselines).

## What changed

- **Pedigree column.** `RecordLog.product[n_log, product_width]`, with
  `product_width = pedigree_width(rows_per_subject)`: three head words (audience flag,
  refit status, row count) then one triple per counting row, zero padded. The width is
  zero when the protocol is off, so Slice 1 logs carry a zero-size column.
  `commit_refresh` writes it (`pedigree_words`), `pedigree_of(log, slot)` reads it, and
  the canonical history includes it when the study refreshes.
- **Revision sets.** `revision_set(log, caps, audience, subject)` gives the current
  revision of each acquisition among `refit_kinds` rows an audience has received
  (accepted for the observer, released for the policy), dropped when the status is not
  in `refit_statuses`. A later revision replaces earlier ones (`superseded_by`). An
  observer-only correction replaces nothing for the policy. `refit_rows` is built on
  the same set, so the rows a refit sees and the pedigree agree, and
  `revision_set_digest` and `refit_key` from T4a are unchanged.
- **Derived kinds.** A kind with `RecordKind.derived` is never folded (excluded in
  `h_release`, applied only when the study declares derived kinds) and never in a
  revision set or the refit rows.
- **Producer-local indices.** Scripted records take their declared channel and the
  scripted id as index. A `Query` answer takes the service channel and
  `Request.decision`. A refresh product takes its audience's product channel, the grid
  ordinal as index and revision 0. Components state their channels through
  `Declaration.channels`.
- **Reserved internal ids.** Each product row takes its own `acquisition_id` and
  `next_sid`, so a product never shares an identity with a counting row.
- **Pre-flight K002.** Admission, scripted, service and product channels must be
  disjoint. The Finding carries a locator (`subject`, `index`, `member`); the codec
  follows `member` only when the slot has a field of that name.
- **Conformance.** Derived rows are excluded from the one-fold-per-acquisition check.
  New invariants `refit_inputs_equal_revision_set` (each pedigree is the revision set
  its audience held at that checkpoint) and `installs_match_product_rows` (channel,
  pedigree head, ordinal uniqueness, recorded status, and a non-installing observer
  status leaves the belief as it was). New corpus case `refresh`, built from
  `_refresh_study` with a masked-batch boundary through `CorpusCase.make_refit`.

## Re-baselines

Two corpus cases changed history, both because a service answer now carries the
decision as its acquisition index (it was a count of log rows) and a correction's
revision identity is now the semantic one; both executors, both worlds (8 entries):

| Case | Reason |
|---|---|
| `host_service` | the answer's acquisition index is the `Request.decision`, on the service channel |
| `revision` | revision rows take their identity from the declared channel and scripted id, and the log gains the `product` column |

The hand rows of both cases moved with them (`HAND_ROWS`): `host_service` row 5 (the
result) now reads index 1, the decision that asked; `revision` rows 3 and 5 read 7001,
the scripted id. The slice 1 gate's expected invariant tuple lists the two new
invariants, the replay tests skip the refresh case (its belief comes from installs, not
release folds), and the planned-case list names `refresh`.

The `refresh` case is new (4 entries; 116 in all). Every other entry, 104 of them,
is byte-identical to main.

## Tests

`tests/core/test_pedigree_layout.py`, 6 quick tests (1.3 s together): the width
formula and its zero case; `pedigree_of` reads the documented layout; a product row is
never in a revision set or the refit rows (with the measurement row as its control);
K002 is silent on distinct channels and fires, with the locator, for a product
channel on an admission channel or on each other.

`tests/core/test_pedigree.py`, 4 slow tests (the file is in `SLOW_FILES`: each runs a
refresh study on an executor): pedigrees name only logged counting rows in canonical
order; refresh products are indexed by grid position with revision 0; a service
answer carries the decision as its index; an answer indexed by the log length instead
(mutant, with a cache clear) is caught.

`tests/core/test_corpus_invariants.py`: planted-violation tests for the two new
invariants on the `refresh` case, on both executors (two plants each: a wrong or
shortened pedigree; a wrong recorded status or product channel). The completeness
test now names them.

## Results

Full suite: 2859 tests, all passing after the re-baselines above (one earlier pass
showed five host_service failures from the stale digest and hand row, fixed). Quick
tier (`-m "not slow" -n 2 --dist loadfile`): 1691 passed in 55 s; the one over-budget
report, `test_every_hand_study_reproduces_its_history`, was over budget before this
change. Lints: ruff check and format, core imports, domain words, internal references.

## Review fixes

- **Scripted internal ids.** `ScriptedRecords` takes each internal `acquisition_id`
  from the shared allocator (`next_sid`): the first fire of a scripted id takes the next
  one, a correction reuses the id of the record it corrects, and the scripted id stays
  the semantic index. A scripted measurement and a refresh product can no longer share
  an identity. Moved goldens: `revision` and `refresh` (ints only, 8 entries), because
  their internal ids changed; no other entry moved.
- **Replay skips derived rows**, in `replay_belief` and in the observer kit's replay.
- **The snapshot fingerprint hashes the pedigree** when `product_width > 0`; a width-0
  fingerprint is unchanged. Only the `refresh` entries moved for this reason.
- **A derived row never takes part in supersession**: not in revision sets, not in the
  release handler's `superseded_by` metadata, not in snapshot views. A correction
  that reuses a derived acquisition id therefore leaves the source row current.
- **`installs_match_product_rows` ties each product to its checkpoint**: index equals
  the grid ordinal of the checkpoint tick, revision 0, available at that tick. A plant
  indexing a product by log position fails it.
- **Tests.** Quick: exact canonical per-audience triples from a log written against
  the canonical order with observer-only rows; fingerprint covers the pedigree; derived
  rows never supersede. Slow: the scripted collision repro, a scripted correction that
  still finds its original, replay over a derived measurement kind. The run-document
  corpus reload skips the `refresh` case, which needs a bound refit boundary.

## Review fixes, round two

- **Refused corrections.** `write_log` refuses a row that reuses the acquisition id of
  a logged row when a derived kind is on either side (derived to derived, source to
  derived, derived to source). The row is not logged and not released, and the world
  sets `overflow` with the new `Capacity.INVALID_CORRECTION` bit (value 32). It is a
  refusal, not a capacity, and the run reports `OVERFLOW`. No pre-flight code covers
  it: a scripted process names its record kinds at construction, but no existing
  K-code is about record kinds, so the check is at run time only.
- **Scripted pairs.** `ScriptedRecords` raises `ValueError` naming the pair when two
  fires share a (scripted id, revision). It takes an optional per-fire `kinds`.
- **Tests.** `test_derived_corrections.py` (slow): both directions on both executors,
  controls with no derived involvement, the uniqueness check, and direct release tests
  for source then derived and derived then source in both arrival orders (expecting
  `superseded_by` of `[-1, -1]`, with a source-only control). A quick hand-log test runs
  gather, boundary and commit and asserts the exact product column per audience
  (observer four triples, policy two). The width-0 fingerprint is pinned to a value.
  The run-document corpus reload now covers the `refresh` case through its
  `make_refit` boundary and a binding for the observer's own refit kernel.

## Remaining

- Repeated `Query` answers on one channel (a policy asking twice) are covered by the
  decision-index test only for one question; a two-question case belongs with T4c.
- A same-tick correction racing a checkpoint is a T4c case.
