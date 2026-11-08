---
orphan: true
---

# Quick-tier diet after the charging, refund, balance and award tests

The quick tier on main passed every test and failed the job summary on time: 566 s warm
against a 540 s ceiling and a 494.5 s limit over the pinned baseline, after about 270 s
warm before the charging, refund, balance and award tests arrived. Cold, the pilot's
quick job took 851 s against 481 s before them. This change moves the compile-heavy
variants of those tests to the slow tier and keeps a lean guard for each feature in the
quick tier. No test is deleted, no assertion is changed, and no file under `src/`, no
workflow, no baseline file and no summary script changed.

## Where the time went

Tests added since the last cold measurement cost 904 s summed on the runner, of which
`test_awards` was 453 s, `test_snapshot_balances` 153 s, `test_charging` 93 s,
`test_refund` 83 s, `test_conservation` 73 s, and the two new surveys 48 s. The cost is
compilation: a study with its own static capacities (a snapshot of balances, a balance
quantum, a one-row log, refunds on or off, a reshaped policy) compiles the executor once
for each distinct value, about 20 s each on the runner. Where tests differ only in array
values they already share one compile, which is why the cheap award scenarios cost about
1 s each after the first.

## Measurement

Cold means `SPACEODYSSEY_JAX_CACHE=off JAX_ENABLE_COMPILATION_CACHE=0`,
`-m "not slow" -n 2 --dist loadfile`, through `compute-run --threads 4`.

| Quantity | Before | After |
|---|---|---|
| Quick-tier tests | 2363 | 2318 |
| Runner cold, summed test time (junit) | 1636 s | 837 s |
| Runner cold wall, 0.52 times the sum | 851 s | 435 s |
| Warm wall, projected at 0.665 times cold | 566 s | about 290 s |
| Mac cold wall | not measured | 175 s |

The runner after-figure is a projection. Tests outside the touched files keep their
recorded runner times (714 s); the touched files use the Mac time after the change times
3, the runner factor for compile-heavy tests. A factor of 2.6 gives 427 s. The Mac
figure is a loaded shared machine and varied between 175 s and 194 s over three runs.
The first full runner run after merging is the real check.

## What moved, by file

Cold seconds are summed runner times from the junit file of the previous run; the
after-figure is the Mac time times 3.

| File | Before | After | Change |
|---|---|---|---|
| `test_awards.py` | 453 s | 42 s | 27 tests to slow |
| `test_snapshot_balances.py` | 153 s | 22 s | 3 tests to slow |
| `test_charging.py` | 93 s | 22 s | 4 cases to slow |
| `test_refund.py` | 83 s | 35 s | 3 tests to slow |
| `test_conservation.py` | 73 s | 3 s | 5 cases to slow, run store made lazy |
| `test_survey_8.py` | 33 s | 0 s | the merge-time guard to slow |
| `test_survey_9.py` | 15 s | 0 s | the merge-time guard to slow |
| `test_premission.py` | 19 s of 65 s | 19 s less | 1 test to slow |

### Awards

The scenario runs share one compile and stay: the grant and window, the remainder rules,
the refused observation outside a window, the refused award onto an open window, the
expiry ordering, the stale expiry, the conservation check on every scenario, the forged
counters and the paired forged expiry. These give the admitted path, the refusal path and
the conservation check on a run.

To slow, each because it needs a study with capacities of its own:

- the range-bound family (representable range, opposite budgets at three quanta, a
  negative reallocation at three quanta, an allocate after a debit, a large award without
  balances in the snapshot, an allocate mixed with a query);
- the reshaped amount (two shapes);
- the refund-and-award family (a refund before and after the expiry, the forged variants,
  receipts that apply nothing, the newer-generation record, the pending acquisition at
  the expiry);
- the per-payer generation records (two) and the awards-off control.

### Snapshot balances

The three tests that compile one study on both executors (a hosted run answering its
query, a query made stale by a queued refund, a receipt released between a decision and
its request) move to slow. The mixed-effect refusals, the fingerprint rules, the range
and sign checks and the mutants stay.

### Charging

The occupancy case of the halfway cancel stays with the default-charge test and the two
`charge_parts` checks. The statistical, flat and surcharge cases, and the refunds-off
default path, move to slow. The run store now builds a run on first use, so the cases
that stay do not also pay for the ones that moved.

### Refunds

The receipt scenarios share one compile and stay (credit without the surcharge, the
refusals, held and observer-only receipts, the controls, the fingerprint and mask
checks). Three tests with their own channel columns move to slow: the failure kind is not
a receipt, the remainder of a completed acquisition excludes its surcharge, and a flat
price on a failure versus a cancel.

### Conservation

The cases that share the surcharge study stay (every account agrees and every row nets,
the forged balance and the forged record checks). The plain case and the three overflow
studies, which each compile with a one-row log, move to slow. The run store builds a case
on first use, as in charging.

### Surveys 8 and 9 and one older test

The two survey guards each ran a full study for 33 s and 15 s; the core stop, program and
refund tests keep the same mechanisms in the quick tier. `test_premission.py` keeps its
other tests, and its observer-only row test, which grew by 17 s when the ledger gained
columns, moves to slow.

## Product of the shared sources

No shared parametrized source gained a case. The case lists in charging and conservation
only gained a slow mark on some members, so the quick tier runs fewer of the same cases.

## Retired

None. Every moved test still runs under `-m slow`.

## Slow tier

`-m slow` on the eight touched files, with `-n 2 --dist loadfile`, runs 68 tests (the
moved tests and the mutants that were already slow) and all 68 pass in 256 s on the Mac.
The nightly slow job covers the moved tests from here on.

## What remains

1. Merge, then read the junit of the next cold pilot run and compare the cold wall with
   the 435 s projection; the target is at most 450 s.
2. Window-aware refunds, commitments, the allocator path and two more surveys are still to
   come. Each should add one study to the quick tier at most and put its capacity variants
   in slow from the start.
3. If compile cost keeps growing, the next step is a library change so that the balance
   quantum and the snapshot flag are array-valued rather than static, which would let the
   variants share one compile. That change touches `src/` and is not made here.
