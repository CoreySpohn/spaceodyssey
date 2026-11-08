---
orphan: true
---

# Slice 3a S8: the allocator path and its replay

S8 adds one study module, `spaceodyssey.studies.allocator`, and one test file,
`tests/studies/test_allocator.py`. It changes no core file, no golden and no pin. The
host `Query` service and the `Allocate` effect already exist; this task builds a toy
committee on the first and a policy on the second, and a replay that checks them.

## What is in

- **The committee.** `ToyCommittee` is a host service. At a fixed tick a level-3 array
  policy asks it a `Query`; the service reads the policy's masked snapshot, totals the
  proposal records in it for each of two programs, splits a pool of 12 between them in
  proportion (whole quarters, rounded down), and answers with an `award` record
  `(round, window from, window until, amount per program)`. It receives no truth and no
  carry, only the request, the snapshot and the belief layout.
- **The proposals.** `ProposalRecords` is `ScriptedRecords` with a program and an audience
  per record. The default script has four proposals released to the policy, one released
  to the observer only (tick 120) and one released after the last query (tick 700).
- **The policy.** `AllocationPolicy` keeps two counters in `pstate` (questions asked,
  awards spent). At each query tick it emits a `Query`. When the award record of the round it
  asked is in its view it emits one `Allocate` per funded program (amount, window,
  recipient row, program, remainder rule `FORFEIT`), one `Observe` per program that the
  award pays for, and a `Wait`. After the last award it drains the campaign. It returns
  a uniform draw from its decision key, so a replay can check the key.
- **The study.** `allocator_study()` is the two-program toy study with awards on and no
  credit in the program rows, so every observation is paid from an award. The two new
  record kinds take registry indices 3 and 4.
- **The replay.** `replay_allocation(study, run, committee=None)` takes an
  `AllocatorRun` (final carry and trace of one world, the `(view, effects)` of every
  executed decision, and the root key). It never raises on a mismatch; it returns a
  `ReplayReport`. For every decision it reruns the policy on the recorded view with the
  decision's key and compares every effect column, the draw, and the `Allocate` rows
  retained in the trace (amount, window, recipient). Each replayed `Allocate` is given the
  admission code the window rule implies (one open window per row) and compared with the
  trace. For every `Query` it checks that the proposals in the view are exactly the
  recorded proposals released to the policy by then, reruns the committee on that view,
  and compares its answer with the award record in the log. The ledger's granted totals
  must equal the admitted awards. The number of recorded awards must equal the number of
  queries.
- **The control.** `OmniscientCommittee` computes its awards from the study's whole
  proposal script, released or not. It is the negative control of the perturbation test and
  of the replay.

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
| `test_the_committee_answers_the_policy_on_the_host` | medium (one host run) | two requests at the query ticks; award rows `(0, 200, 450, 8, 4)` and `(1, 500, 750, 6, 6)`; four `Allocate` effects and four observations all admitted; granted totals; `balance_conserved`; the replay matches (15 decisions, 2 awards, 4 allocations) |
| `test_replay_fails_when_the_run_is_altered` | medium (numpy edits of the host run) | each recorded award dropped in turn fails; each recorded `Allocate` removed fails; a flipped admission code fails; replaying with the omniscient committee fails |
| `test_the_committee_answers_on_both_executors` | slow | equal histories, equal requests and awards on both executors, and the replay of the compiled run from the views its callback captured |
| `test_the_committee_reads_only_released_records` | slow | perturbing the observer-only proposal, or the proposal after the last query, leaves both awards bit-identical; the observer-only request never appears in a policy view; perturbing a released proposal moves the awards |
| `test_a_committee_that_reads_the_script_fails_the_perturbation` | slow | the same perturbations move the omniscient committee's awards |

## Notes for review

- The replay checks admission codes for `Allocate` only. An observation's code depends on
  the whole ledger, which the replay does not rebuild; the test asserts those separately.
- Two executors in one function exceeds the medium budget here (about 40 s cold), because
  the compiled driver and the capture callback each add a program; that test is slow, and
  the medium guard runs the host executor.
- A later survey can reuse `allocator_study`, `ToyCommittee`, `OmniscientCommittee`,
  `AllocatorRun` and `replay_allocation` unchanged.
