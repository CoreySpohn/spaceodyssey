---
orphan: true
---

# Slice 3a survey 10: a time-allocation committee

Survey 10 adds one test file, `tests/studies/test_survey_10.py`, and changes no core file, no
golden and no pin. It reuses `spaceodyssey.studies.allocator` unchanged: the toy committee, the
level-3 allocation policy, the carry-reading control and `replay_allocation`.

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
  changing the amount,
  window start, window end or recipient retained in the trace for an `Allocate`, changing a
  recorded proposal, and replaying with the carry-reading committee.
- **Refused observations.** A run releases the unreleased proposal at tick 180 and raises the
  program-0 request written at tick 400 from 2 to 20. Program 0 then holds 0.5 at the first query
  and 2.5 at the second, so its first `Observe` (cost 1) is refused `NO_FUNDS` and its second is
  admitted; the awards are (0.5, 11.25) and (2.5, 9.25), worked out by hand like the others. That
  run shares the early-release program, so it compiles nothing. Its replay passes, and each of
  these corruptions fails it: swapping the two outcomes, changing `NO_FUNDS` to `FORBIDDEN`,
  marking an admitted request refused, and marking the refused one admitted. The early-release
  run, which has two legitimate refusals, replays clean as well.
- **Production process.** All default-structure runs use the production `ProposalRecords`, with
  static release ticks, not a substitute.
- **Parity.** The host and compiled runs have equal histories, query ticks, awards and `Allocate`
  effects for the default, both perturbed and the early scripts.

## Core changes

None. `src/spaceodyssey/core` is untouched. Reviews found that `replay_allocation` first checked
admission outcomes only for `Allocate` effects, and then reconciled `Observe` outcomes only as
counts. `src/spaceodyssey/studies/allocator.py` now reconciles each one. An admitted `Observe` must
own exactly one measurement record that starts inside its own occupancy on its channel and subject,
and every record must belong to exactly one admitted request (`observation_failures`), so swapped
outcomes leave a request without a record. An outcome must also agree with the program funds the
replay rebuilds from the awards (window open at the tick, minus earlier charges): an admitted
request needs the funds, and a refusal must be `NO_FUNDS` with the funds short (`funds_failures`;
program rows hold no credit of their own in this study). Each replayed `Query`, `Wait` and `Stop`
must be admitted.

## Evidence

| Item | Asserted nonzero |
|---|---|
| (a) | queries (2), awards (2), decisions, sealed rows (1 per run), `Allocate` effects (4) |
| (b) | awards (2), moved awards (2), `Allocate` effects (4) |
| (c) | carry-committee awards (2), queries (2), toy awards (2) |
| (d) | decisions, awards (2), allocations (4), admitted `Allocate` (4), admitted `Observe` (4) |

Mutants of the real code, each run as a temporary edit and reverted; the file passes (9 tests) on
the real code. Fifteen mutants from an independent review all fail the file except one:

| Mutant | Killed by |
|---|---|
| committee handed the observer snapshot | (a), (b), (d) |
| committee totals keyed by the wrong program | (b), (d) |
| `Allocate` keyed to the wrong program | (a), (b), (d) |
| `Allocate` window bounds swapped | (a), (b), (d) |
| amount rounding nearest | (b) |
| amount rounding ceiling | (b) |
| replay skips the admission comparison | (d) |
| replay assumes every `Allocate` is admitted | survives |
| static release ticks use the write ticks | (a), (b), (d) |
| replay skips the retained trace check | (d) |
| replay skips the recorded-proposal check | (d) |
| replay skips award reconciliation | (d) |
| carry committee reads the masked view | (c), (d) |
| every proposal addressed to the policy | (a), (b), (d) |
| every proposal released at write time | (a), (b), (d) |

The survivor stays alive because no study produces a refused `Allocate`: the two award windows
(200 to 450 and 500 to 750) never overlap, and the window length and query ticks are constants of
the toy committee and policy, not data. Skipping the new observation reconciliation also fails (d).

## Tiers and cost

All five tests are medium with a stated reason. Cold, with caches off, the survey file alone takes
about 31 s (9 passed, 274 programs, budget 326). The host tests alone take about 17 s and the
compiled tests about 18 s; each fixture is about 12.6 s (host) and 13.4 s (compiled), and the eager
replays add about 2 s each.

## Notes

- The slow tests in `test_allocator.py` stay. The slow early-release perturbation test replays a
  run with legitimate refusals, which no other test did before the refused-observation run of
  survey 10; the survey now covers that too, but both are kept. The slow control and the perturbation
  test also perturb each proposal in its own run, as survey 10 does.
