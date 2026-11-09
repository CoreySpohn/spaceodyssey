---
orphan: true
---

# Slice 3b U3: bundles

A policy can now run several observations of one lane as one occupancy. Rows of one
admission group that share a bundle ordinal are admitted whole, start together, pay one
transition, share one lane generation and keep the lane busy until the longest member ends.
Cancelling any member settles every pending member. A study without bundles keeps its
program, its histories, its fingerprints and its document hashes.

## What changed

- `core/capacities.py`: `max_bundles` and `max_bundle_size` (static, optional, default 0;
  both zero or both positive); property `bundles`.
- `core/snapshot.py`: `EffectTable.bundle` (optional, -1 for none), `bundle_columns`; the
  observer view carries the bundle id of each visible record to both audiences.
- `core/events.py`, `core/records.py`: `EventTable.bundle`, `RecordLog.bundle` and
  `Record.bundle` (optional leaves, absent when bundles are off). A bundle id is
  `decision * max_bundles + ordinal`, so equal ordinals of different decisions differ.
- `core/transaction.py`: an Observe row with an ordinal bumps the lane generation only for
  the first member of its group and ordinal; the later members take that generation. The
  lane's busy window is `max(stored, done tick)`. A group is refused whole with
  `UNSUPPORTED` when its members differ in lane, group or program, name an ordinal at or past
  `max_bundles`, exceed `max_bundle_size`, or start after the decision tick.
- `core/handlers.py`: Done only writes the bundle id into its record (the lane frees by
  time, so it frees at the last member). `_cancel_lane` cancels the first pending member and
  then loops over the rest of its bundle, bounded by the static `max_bundle_size`; later
  members leave the generation and the busy window alone, so the generation moves once.
  The same path serves Cancel, Fault and Stop.
- `core/executors/base.py`: `bundles_on` predicate and the four `CONDITIONAL_FIELDS`
  entries (`carry.ev`, `carry.log`, `carry.request`, `carry.refresh`); `replay.py` copies
  the column.
- Pre-flight: `Declaration.bundles` as `(ordinal, lane, group, program, start offset)` per
  member and finding code `B001` for a declared bundle admission would refuse.

## Tests

`tests/core/test_bundles.py` (15 tests, one shared compiled structure with programs,
refunds, awards, commitments and bundles on):

| Test | Tier |
| --- | --- |
| members start together on one transition, share a bundle id, ids differ across decisions | medium (7 s, shared fixture) |
| lane frees only at the last member's Done (a decision at 45 is refused, the one at 55 starts) | medium (7 s, shared fixture) |
| shorter member committed second leaves the busy window at the longer end; one generation on both Done rows (both orders) | unit |
| one transition charge for the bundle | unit |
| six refusals (lanes, groups, programs, ordinal, size, future start), carry untouched | unit |
| cancel settles every pending member, with programs, refunds and an award window | medium (7 s, shared fixture) |
| cancel after the short member completed settles only the long one | medium (7 s, shared fixture) |
| host and compiled histories equal on a bundle run | medium (12 s cold) |
| fork battery passes on a bundle run | medium (15 s cold) |

Also: `B001` seeded cases (seven) and one clean case in `preflight_cases.py`, their locators,
the optional-field inventory and a bundle column round trip in `test_optional_fields.py`
(Capacities, effect table, record), the slice-3 guard capacities list, and the
snapshot-fingerprint shim.

Negative controls (run at merge time, not standing tests): removing the member loop fails the
cancel test; taking the busy window as the last member's end fails the three admission tests;
giving every member its own generation bump fails seven tests including the executor and
fork test.

## Receipts

- K4: the only loop added is the member loop, bounded by the static `max_bundle_size` and
  absent from the program when bundles are off. Mutant, run at merge time: an unconditional
  member loop in `_cancel_lane` (a bounded loop run with bundles off) moves the `survey_1`
  `transition` jaxpr hash from `465cc8909391c8c9` to `04fb1eb9ceb3641c` (499196 to 559264
  characters), so the bit-identity check below fails on it. `test_program_size.py` alone does
  not catch it.
- Default program, measured against the commit before this change (not a standing test): the
  lowered jaxpr of one `transition` is identical, hash `465cc8909391c8c9`, 499196
  characters, for the first corpus case (the six cases printed share it); and
  identical for a study with programs, awards, refunds, commitments and snapshot balances on
  but no bundles (`cb1a978939d8846e`, 905596 characters).
- Program budget for `tests/core/test_bundles.py`: 207 programs loaded from the cache
  (warm), budget 250.
- Golden check: `tests/core/test_golden_histories.py -m ""` passes unchanged (124 passed, 847 s
  wall under load); no golden or pin file was touched.
- Quick tier: tests/core and tests/codec, `-n 2 --dist loadfile -m "not slow"`: 1712 of 1713
  pass in 167 s. The one failure, `test_entry_points_installed`, reports stale installed
  package metadata (the jaxedith channel entry point is not in the installed metadata) and is
  unrelated to bundles.
- Existing tests touched: the cancel mutants of `mutants.py`, `battery_helpers.py` and
  `test_charging.py` now target `_cancel_one`, where the realize and credit code moved.
