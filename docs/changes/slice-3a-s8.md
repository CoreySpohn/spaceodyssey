---
orphan: true
---

# The allocator path and its replay

This change adds one study module, `spaceodyssey.studies.allocator`, and one test file,
`tests/studies/test_allocator.py`. It changes no core file, no golden and no pin. The
host `Query` service and the `Allocate` effect already exist; this change builds a toy
committee on the first and a policy on the second, and a replay that checks them.

## What is in

- **The committee.** `ToyCommittee` is a host service. At a fixed tick a level-3 array
  policy asks it a `Query`; the service reads the policy's masked snapshot, totals the
  proposal records in it for each of two programs, splits a pool of 12 between them in
  proportion (whole quarters, rounded down), and answers with an `award` record
  `(round, window from, window until, amount per program)`. It receives no truth and no
  carry, only the request, the snapshot and the belief layout.
- **The proposals.** `ProposalRecords` is `ScriptedRecords` with a program, an audience and
  a release tick per record. The default script has four proposals released to the policy,
  one released to the observer only (written at 120), and one written at 150 whose release
  is delayed to 700, so it sits in the log, unreleased, at both queries.
- **The policy.** `AllocationPolicy` keeps two counters in `pstate` (questions asked,
  awards spent). At each query tick it emits a `Query`. When the award record of the round it
  asked is in its view it emits one `Allocate` per funded program (amount, window,
  recipient row, program, remainder rule `FORFEIT`), one `Observe` per program, paid
  from the award, and a `Wait`. After the last award it drains the campaign. It returns
  a uniform draw from its decision key, so a replay can check the key.
- **The study.** `allocator_study()` is the two-program toy study with awards on and no
  credit in the program rows, so every observation is paid from an award. The two new
  record kinds take registry indices 3 and 4.
- **Checks on recorded inputs.** Four functions work on arrays alone, so a small test can
  call them without running a study: `visibility_failures` (the proposals and awards in a
  view are exactly the records the log shows as released to the policy by then),
  `award_failures` (the committee's answer on each query view equals the recorded award
  row and tick, and the counts agree), `allocation_failures` (each recorded `Allocate`
  carries its award's amount, both window bounds, the program's balance row, the program
  and the rule), and `released_rows` and `viewed_rows`, which they share.
- **The replay.** `replay_allocation(study, run, committee=None)` takes an
  `AllocatorRun` (final carry and trace of one world, the `(view, effects)` of every
  executed decision, and the root key) and returns a `ReplayReport`; it never raises on a
  mismatch. For every decision it applies `visibility_failures`, reruns the policy on the
  recorded view with the decision's key, and compares every effect column, the draw, and
  the `Allocate` rows retained in the trace. Each replayed `Allocate` is given the
  admission code the window rule implies (one open window per row) and compared with the
  trace. After the walk it applies `award_failures` and `allocation_failures` to the
  recorded records and effects, and requires the ledger's granted totals to equal the
  admitted awards.
- **The control.** `CarryCommittee` is also handed the carry (`hold`) and totals every
  proposal row in its log by the query tick, released or not and whatever its audience. It is the negative control of the
  perturbation test and of the replay.

## Core changes

None. `src/spaceodyssey/core` is untouched, and no `CORE_KINDS` entry or entry point was added.

| Check | Result |
|---|---|
| Default scan StableHLO | byte-identical to `main` (d4b2c86): `_scan_worlds` on `survey_1`, two worlds, 40 steps, `lower(...).as_text()` compared byte for byte; sha256 `a67917728c4878feaa35f3bccc622be9ff6850ec65b5adf22867e668c9d1d791` on both (1,292,736 bytes) |
| Pins, goldens, fingerprints, corpus hashes | untouched (no core, codec or data file changed) |

## Tests

| Test | Tier | Asserts |
|---|---|---|
| `test_split_pool_is_proportional_in_whole_quanta` | unit | the committee's split rule |
| `test_the_study_has_no_preflight_error` | unit | the study declares what it uses |
| `test_the_committee_reads_only_the_rows_the_view_shows` | unit | the answer ignores invalid rows, rows past `n_rows` and other kinds, and moves with a visible proposal |
| `test_the_policy_turns_an_award_into_its_allocations` | unit | amount, both window bounds, recipient, program and rule follow the award; corrupting any one field is caught; stale, invalid, hidden and unasked awards produce nothing |
| `test_dropping_a_recorded_award_fails_the_check` | unit | either award dropped, or one altered, fails |
| `test_awards_must_be_visible_to_the_policy_as_recorded` | unit | an award recorded for the observer only, unreleased, or released later than the decision fails |
| `test_the_committee_answers_the_policy_on_the_host` | slow | two requests at the query ticks; awards `(0, 200, 450, 8, 4)` and `(1, 500, 750, 6, 6)`; four `Allocate` and four `Observe` effects admitted and following the awards; `balance_conserved`; the pending proposal is in the log before the first query and released after the second; the replay matches |
| `test_replay_fails_when_the_run_is_altered` | slow | each award dropped; each `Allocate` removed; each `Allocate` field corrupted; an award recorded for the observer or unreleased; a flipped admission code; the carry-reading committee |
| `test_the_committee_answers_on_both_executors` | slow | equal histories, requests and awards on both executors, and the replay of the compiled run from the views its callback captured |
| `test_the_committee_reads_only_released_records` | slow | perturbing the observer-only proposal, or the pending one while it is unreleased, leaves both awards bit-identical and the allocations follow them; releasing the pending proposal before the first query moves the awards and the allocations follow the new ones |
| `test_a_committee_that_reads_unreleased_records_fails_the_perturbation` | slow | the same two perturbations move the carry-reading committee's awards |

## Notes

- The replay checks admission codes for `Allocate` only. An observation's code depends on
  the whole ledger, which the replay does not rebuild; the test asserts those separately.
- Compiling the study dominates every test that runs it, so all runs are slow and the
  quick tier calls the committee and the policy directly on small snapshots.
- A later survey can reuse `allocator_study`, `ToyCommittee`, `CarryCommittee`,
  `AllocatorRun` and `replay_allocation` unchanged.
