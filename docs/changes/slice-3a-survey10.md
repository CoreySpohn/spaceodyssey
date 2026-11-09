---
orphan: true
---

# Slice 3a survey 10: a time-allocation committee

Survey 10 adds one test file, `tests/studies/test_survey_10.py`, and changes no core file, no
golden and no pin. It reuses `spaceodyssey.studies.allocator` unchanged: the toy committee, the
level-3 allocation policy, the carry-reading control and `replay_allocation`.

The file keeps one module-scoped run store per executor. It holds the default run and four runs of
the same study structure that differ in data only: proposal values and release ticks are traced
numbers, because the file builds its script with a three-line `ProposalRecords` subclass whose
`releases` field is marked traced. The study therefore compiles once per executor, and the other
four runs recompile nothing. Every test reads the store, on both executors, and each count it relies
on is asserted nonzero.

## What it shows

- **(a) sealed proposals do not move the awards.** The observer-only proposal and the proposal that
  stays unreleased through both queries are both raised to 1000 in one run. Their two log rows are
  counted (one observer-only, one unreleased), both queries are answered, and the awards and the
  four `Allocate` effects equal the default run bit for bit. The observer-only request is in no
  policy view; the unreleased one is in no view up to the last query and in later views.
- **(b) releasing a proposal moves the award the expected way.** The unreleased proposal (program 1,
  request 90) is released at tick 180, before the first query. The first award becomes the split of
  the pool over the proposals then released (`split_pool` of 6 and 93), program 1 gains and program 0
  loses against the default award, the effects follow the new awards, all four `Allocate` requests
  are admitted, and the ledger's granted totals equal the awards.
- **(c) a committee that reads the carry fails (a).** With the carry-reading committee the same
  perturbation moves the awards, and its awards differ from the toy committee's even on the
  unperturbed script.
- **(d) action replay.** The default run replays with no failure: 2 awards, 4 `Allocate` effects,
  all 4 `Allocate` and 4 `Observe` requests admitted, and every decision rechecked. The awards are
  `(0, 200, 450, 8, 4)` and `(1, 500, 750, 6, 6)`. Dropping either recorded award fails the replay
  with a query failure; removing a recorded `Allocate`, changing one amount, flipping one admission
  code and replaying with the carry-reading committee each fail it too.
- **Parity.** The host and compiled runs have equal histories, query ticks, awards and `Allocate`
  effects for the default, perturbed and early scripts.

## Core changes

None. `src/spaceodyssey/core` and `src/spaceodyssey/studies/allocator.py` are untouched.

## Evidence

| Item | Asserted nonzero |
|---|---|
| (a) | queries (2), awards (2), perturbed rows (2), observer-only rows (1), unreleased rows (1), decisions, `Allocate` effects (4), later views of the unreleased request |
| (b) | awards (2), moved awards (2), `Allocate` effects (4) |
| (c) | carry-committee awards (2 and 2), queries (2) |
| (d) | decisions, awards (2), allocations (4), admitted `Allocate` (4), admitted `Observe` (4) |

Mutants of the real code, each run as a temporary edit and reverted (the file passes, 9 tests, on
the real code):

| Item | Mutant | Result |
|---|---|---|
| (a) | `ProposalRecords.seed` queues every proposal for the policy audience | (a), (b) and (d) fail on both executors |
| (b) | `ProposalRecords.handle` stamps `available_tick=rec.start_tick`, ignoring the release tick | (a), (b) and (d) fail on both executors |
| (c) | `CarryCommittee.requested` returns `ToyCommittee.requested(self, view)` | (c) and (d) fail on both executors |
| (d) | `award_failures` returns an empty list | (d) fails on both executors |

## Tiers and cost

All five tests are medium with a stated reason. Cold, with caches off, the whole file takes 19 s
(9 passed): the host fixture is about 9.2 s (6.4 s of it the first run), the compiled one about
9.3 s (7.8 s), and the eager replays add 1.7 s and 0.9 s. Alone, the host tests take 10.2 s with
198 programs and the compiled tests 10.6 s with 201 programs; the whole file counts 264 programs
and its budget is 309.

## Notes

- Tests in `test_allocator.py` that survey 10 makes redundant: the slow perturbation test
  (`test_the_committee_reads_only_released_records`) and the slow control
  (`test_a_committee_that_reads_unreleased_records_fails_the_perturbation`), which now compile
  the study four more times between them. The other slow tests stay: they cover the study answer
  detail, the full set of altered-run replays and the capture callback.
