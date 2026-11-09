---
orphan: true
---

# Slice 3a survey 10: a time-allocation committee

Survey 10 adds one test file, `tests/studies/test_survey_10.py`, and changes no core file, no
golden and no pin. It reuses the toy committee, the level-3 allocation policy and the
carry-reading control of `spaceodyssey.studies.allocator`. Review of the first version showed that
the study-side replay could not be trusted to re-derive the core's admission rules, so
`replay_allocation` changed design (see Core changes).

The file keeps one module-scoped run store per executor. It runs the production allocator study:
the default run, a run with the observer-only request raised to 1000, a run with the unreleased
request raised to 1000, and the carry-reading committee on the default script and on each
perturbation. Those runs differ in data only (requests are traced numbers) and share one program. One
more run releases the unreleased proposal before the first query; release ticks are static in
`ProposalRecords`, so it compiles a second program per executor. Proposals are identified by their
scripted index in the log, never by value. Every test reads the store, on both executors, and each
count it relies on is asserted nonzero.

## What it shows

- **(a) sealed proposals do not move the awards.** In one run the observer-only request is raised to
  1000, in another the request that stays unreleased through both queries; each row is found by its
  scripted index, and its audience or release tick is asserted. In both runs both queries are
  answered, and the awards and the four `Allocate` effects equal the default run bit for bit. The
  observer-only request is in no policy view. The unreleased one is in no view up to the last query
  and in later views, so a view could have shown it.
- **(b) releasing a proposal moves the award the expected way.** The unreleased proposal (program 1,
  request 90) is released at tick 180, before the first query. The awards are pinned to values worked
  out by hand with the whole-quarter, round-down rule from a pool of 12: the default awards are
  (8, 4) and (6, 6); the early awards are (0.5, 11.25) and (0.75, 11), from requests 6 and 93 and then
  8 and 98. Every award sums to at most the pool, program 1 gains and program 0 loses against the
  default, the effects follow the awards, all four `Allocate` requests are admitted, and the ledger's
  granted totals equal the awards. The test does not call `split_pool`.
- **(c) a committee that reads the carry fails (a).** With the carry-reading committee each of the
  two perturbations moves the awards, and its awards differ from the toy committee's even on the
  unperturbed script.
- **(d) action replay.** The default run replays with no failure: 2 awards, 4 `Allocate` effects,
  all 4 `Allocate` and 4 `Observe` requests admitted, and every decision rechecked. The awards are
  `(0, 200, 450, 8, 4)` and `(1, 500, 750, 6, 6)`. Dropping either recorded award fails the replay
  with a query failure. Each of these fails it too: removing a recorded `Allocate`, changing one
  amount, flipping the admission code of an `Allocate`, `Observe`, `Query`, `Wait` or `Stop`,
  swapping two outcomes, changing the amount, window start, window end or recipient retained in the
  trace for an `Allocate`, changing a recorded proposal, dropping a measurement record, repeating
  one, and replaying with the carry-reading committee.
- **Refused observations.** A run releases the unreleased proposal at tick 180 and raises the
  program-0 request written at tick 400 from 2 to 20. Program 0 then holds 0.5 at the first query
  and 2.5 at the second, so its first `Observe` (cost 1) is refused `NO_FUNDS` and its second is
  admitted; the awards are (0.5, 11.25) and (2.5, 9.25), worked out by hand like the others. That
  run shares the early-release program, so it compiles nothing. Its replay passes, and each of
  these corruptions fails it: swapping two outcomes, changing `NO_FUNDS` to `FORBIDDEN`, marking
  an admitted request refused, and marking the refused one admitted. The early-release run, which
  has two legitimate refusals, replays clean as well.
- **Other admission rules.** Four data variants of the default study replay clean on the host, and
  changing any one `Observe` or `Allocate` outcome in each fails the replay: an award that swamps
  the other, which makes an `Observe` refuse with `OUTSIDE_VALIDITY` (relabelling it `NO_FUNDS`
  fails); a transition time equal to the occupancy; an asset budget of zero, which refuses with
  `NO_FUNDS`; and a permission set that leaves out `Allocate`, which gives four legitimate
  `FORBIDDEN` `Allocate` outcomes (marking one admitted fails). They share the default program, so
  they add no compile. A surcharge study that refunds was not made a test: its refund flag is
  static, so it needs a program of its own, and the funds rule it exercises is the core's, which
  the replay no longer re-derives.
- **Record metadata.** The replay compares every column of every valid measurement and award row
  with the re-execution (the core's own column list), integers and flags exactly and floats within
  the executor go tolerance of 1e-12. Each of these fails it: a measurement's channel, status,
  program, audience, release stamp, acquisition index or acquisition id changed, an award's status,
  program, acquisition index, audience or release stamp changed, and a measurement payload moved by
  1e-9. A one-ULP payload change replays clean.
- **Production process.** All default-structure runs use the production `ProposalRecords`, with
  static release ticks, not a substitute.
- **Parity.** The host and compiled runs have equal histories, query ticks, awards and `Allocate`
  effects for the default, both perturbed and the early scripts.

## Core changes

None. `src/spaceodyssey/core` is untouched. Reviews found that `replay_allocation` re-derived the
core's admission outcomes in study code (funds, occupancy and window rules), and each round found
another rule it got wrong. It no longer does. `src/spaceodyssey/studies/allocator.py` keeps the
checks that are the study's own: view visibility, the policy rerun on each recorded view with its
key (effects, draw, retained `Allocate` rows), the committee rerun per `Query`, reconciliation of
the award records with the answers, each `Allocate` following its award, and for these the run
now carries its root and truth. For outcomes and records it asks the core. `reference_execution`
runs the same study again on the host executor with the run's root and truth and a copy of the
same committee, and `reference_failures` requires that the re-executed trace gives the same
effects and admission codes for every handled decision, that the measurement and award records in
the log are equal, and that the ledger's granted totals are equal. A caller may compute the
reference once and pass it to many replays. The helpers `observation_failures` and
`funds_failures` are deleted.

## Evidence

| Item | Asserted nonzero |
|---|---|
| (a) | queries (2), awards (2), decisions, sealed rows (1 per run), `Allocate` effects (4) |
| (b) | awards (2), moved awards (2), `Allocate` effects (4) |
| (c) | carry-committee awards (2), queries (2), toy awards (2) |
| (d) | decisions, awards (2), allocations (4), admitted `Allocate` (4), admitted `Observe` (4) |

Mutants of the real code, each run as a temporary edit and reverted, against the survey file and
`test_allocator.py` (24 tests, with the slow ones on). Every mutant fails the suite except where
marked:

| Mutant | Killed by |
|---|---|
| committee handed the observer snapshot | (a), (b), (d), edge rules |
| committee totals keyed by the wrong program | (b), (d) |
| `Allocate` keyed to the wrong program | (a), (b), (d), edge rules |
| `Allocate` window bounds swapped | (a), (b), (d), edge rules |
| amount rounding nearest | (b), (d) |
| amount rounding ceiling | (b), (d), edge rules |
| replay skips the award reconciliation | (d) |
| replay skips the retained trace check | (d) |
| replay skips the recorded-proposal check | (d) |
| static release ticks use the write ticks | (a), (b), (d) |
| carry committee reads the masked view | (c), (d) |
| every proposal addressed to the policy | (a), (b), (d), edge rules |
| every proposal released at write time | (a), (b), (d) |
| replay skips the reference comparison | (d), edge rules |
| replay ignores the reference's admission codes | (d), edge rules |
| replay ignores the reference's measurement and award records | (d) |
| replay skips each record column in turn (channel, status, program, audience, release stamp, identity) | (d), the record-metadata corruptions |
| replay compares floats exactly | (d), the one-ULP control |
| replay compares floats within 1e-3 | (d), the 1e-9 payload corruption |
| replay treats every `Allocate` outcome as admitted | (d), edge rules (the refused-`Allocate` run) |
| replay ignores the reference's granted totals | (d), through the granted-total corruption |
| the reference is the run's own trace and log, not a re-execution | (d), through the replay with no precomputed reference |
| the re-execution uses a different root | (d), edge rules |

Mutants in the last four rounds that first survived are killed by additions: the granted-total
check (a ledger corruption), the reference taken from the run itself (a replay with no precomputed
reference), skipping a record column (a corruption of each), exact-float comparison (a one-ULP
control), and treating every `Allocate` as admitted. The last two survived the earlier
design because the run had no refused `Allocate` and no float control; the permission-set variant
now supplies four legitimate refusals, and flipping any of them fails. The earlier mutants that
targeted `observation_failures`, `funds_failures` and the `Allocate`, `Query`, `Wait` and `Stop`
admission derivations no longer apply, since those are deleted.

## Tiers and cost

All seven tests are medium with a stated reason. The runs of each executor are built on first use,
so a test pays only for the runs it reads, and the reference of a run is made only when a test
asks for it. Cold, with caches off, the whole survey file takes about 48 s (11 passed, 311
programs, budget 368, zero violations). Each test alone in a fresh process, with its fixtures
charged, takes at most 24.6 s (the executor parity test, which compiles both structures on both
executors); the next are the compiled replay test at 17.7 s, the edge and early-release tests
at 14 s, and the rest under 13 s. `test_allocator.py` with its slow tests takes 26 s (14 passed,
337 programs, budget 361, zero violations).

## Notes

- The slow tests in `test_allocator.py` stay. The slow early-release perturbation test replays a
  run with legitimate refusals, which no other test did before the refused-observation run of
  survey 10; the survey now covers that too, but both are kept. The slow control and the perturbation
  test also perturb each proposal in its own run, as survey 10 does. The two unit tests of the
  deleted helpers are removed.
