---
orphan: true
---

# Slice 3a S0: pins, measurements and interface freeze

The first task of Slice 3a, done together with Slice 2b R0a (`slice-2b-r0a.md`). It
changes no library code. It records the pins every later 3a task is held to, the K3
step-cost baseline at eleven handlers, and the answers to the four questions the spec
leaves to S0.

## Commits

Shared with R0a; see `slice-2b-r0a.md` for the list. The S0 parts are the pins (the
first two commits), the empty-leaf fingerprint test (third) and the step-cost probe
`tools/step_cost.py` (fourth).

## Pins

Source commit `96a644728386365177e79f8bca315a746ef2901c`, captured on macOS 26 arm64.
The float digests of the golden histories are platform dependent (they differ on the
Linux worker); the integer digests, fingerprints and document hashes are not. Details,
rules and controls are in `slice-2b-r0a.md`; the S0 items are:

| K1 item | Where | Count |
|---|---|---|
| The history digests | `tests/core/golden_pin.json` `histories` | 116 |
| Every old study's policy and observer fingerprints | `golden_pin.json` `fingerprints` | 116 runs (decision digest and count, final policy view, final observer view) |
| The `FAULT` fixture | `golden_pin.json` `fault` | 4 (host and compiled, two worlds) |
| Every corpus document hash and the schema version | `tests/codec/hash_pin.json` | 58 documents; `schema_version` 1 |

The host and compiled executors give the same fingerprints for every corpus run. The
fault fixture fails one acquisition (`h_fault` today: charged in full, lane not
closed); its control adds a lane `Stop` at each fault, the K1 mutant "lane closure on
`FAULT` by default", and moves every pinned integer digest.

## K3 baseline

Tool: `python tools/step_cost.py --mutant` (from the repository root, package
importable). It times `_scan_worlds`, the compiled executor's scan for studies with no
host service and no refresh schedule, on the toy survey-1 study: the first call alone
(trace, compile and run; persistent compilation cache off), then repeated calls, each
synchronized with `jax.block_until_ready`, the per-step time being the median wall time
over the step count. The handler table is `core/handlers.py:467-479`, eleven entries.
Under `vmap` the handler switch evaluates every branch, so the per-step cost grows with
the table.

Local figures, macOS 26 arm64, four threads through the compute launcher, two worlds,
400 steps, 30 repeats, two independent processes:

| Run | Cold first call | Warm median per step | Warm min per step | Mutant median per step | Mutant ratio |
|---|---|---|---|---|---|
| 1 (machine shared with two other jobs) | 6.30 s | 114.6 us | 104.8 us | 169.5 us | 1.48 |
| 2 | 6.33 s | 100.5 us | 97.7 us | 152.2 us | 1.51 |

Local eleven-handler baseline: about 100 us per step (warm median of the quieter run;
98 to 105 us by the minimum), and a cold first call of 6.3 s. The mutant cold call is
6.9 s.

The two medians differ by 14 percent and the two minima by 7 percent, on a machine
shared with other jobs. That run-to-run spread exceeds K3's 10 percent threshold, so
K3 cannot compare figures from separate jobs. Recommendation for S5 and S10: measure
the eleven-handler baseline and the final table in the same job, interleaved
(checkout of the baseline commit and of the final commit, alternating runs), and
compare the minimum per-step time over at least 30 repeats; the mutant's 48 to 51
percent excess is far outside that noise, so it remains a valid failing control.

The calibrated mutant (`--mutant`, `ExtraPassExecutor`) adds one select-and-handle
pass per step, kept only under a condition that never holds and that the compiler
cannot fold away. Measured: 1.48 and 1.51 times the plain per-step time, well above the 1.10 threshold, so it fails K3 as a control must.

The K3 baseline proper is on the CI runner with caches off; the local figures above
fix the method and calibrate the mutant. The runner figures come from the first run of
the dispatch-only workflow `.github/workflows/pilot.yml` (added with R0a; see
`slice-2b-r0a.md`), dispatched on `main` right after this branch merges. It runs
`python tools/step_cost.py --mutant --repeats 30` with every compile cache off and
writes the per-step minimum and median, plain and mutant, to the job summary. S5's
change record holds that run id and figure as the eleven-handler baseline; S5 and S10
rerun the same workflow at the final table (thirteen handlers), in the interleaved form
recommended above, and compare the per-step minimum.

## Answers to the S0 questions

### Does the loader support default fields?

No. For a document at the current kind version, `_decode_kind` refuses any field not
declared (`_only`, `codec/load.py:214`) and refuses a missing one
(`codec/load.py:215-217`). `_build` makes the object with `object.__new__`
(`codec/load.py:231`), so constructor defaults are never used. The only route to a
value not in the document is a version migration (`_migrated`, `codec/load.py:289`).
The encoder writes every non-version field (`_encode_kind`, `codec/dump.py:134-144`).
So S1's codec change (encoder omits a field at its default, loader fills it) is
required, as D12 states; the corpus document hashes it must keep are now pinned.

### Can `service.py` answers be released as award records?

Yes, with four limits S8 must design around.

- The answer is a record of any declared kind (`service.kind`, `core/service.py:85`)
  with up to `payload_width` float columns (`core/service.py:166-179`), written by
  `commit_result` and released by a `RESULT` row at the request's own tick and stage
  (`core/service.py:116-134`), which `h_result` hands to `h_release`
  (`core/handlers.py:260`). Its `acquisition_index` is the asking decision's index
  (`core/service.py:91`), a producer-local identity that private events do not move.
- Audience: the request is always addressed to the policy (`audience=Audience.POLICY`,
  `core/transaction.py:393-407`) and the answer keeps it (`core/service.py:101`). An
  award record is therefore policy-only; the observer never accepts it. The award
  kind should be declared derived so the policy summary does not fold it as data.
- One question at a time: a `Query` is refused `NO_SLOT` while a request or result is
  outstanding (`core/transaction.py:388-392`), and one answer is one row. A committee
  that grants several awards answers in one row (several awards packed) or over
  several decisions.
- Width: an award needs R + 4 numbers (three amounts, window start and end, recipient,
  program), seven with R = 3, carried as floats (exact integers below 2^53). Declaring
  that kind widens `payload_width` for the study (`KindRegistry.lower`,
  `core/capacities.py:37`); no existing study changes.

The released answer wakes the policy at the same tick (`h_release` sets `woke` for a
public row), so the level-3 policy can emit `Allocate` from the award in the next
microstep. The fingerprint check of `answer_pending` (`core/service.py:164-165`) is
the D7 hazard that `QUERY_MIXED` addresses.

### `COMMIT_START` ordering and the transition-charge timing

Facts today:

- A charge is debited at admission, in full: `_commit_observe` subtracts `cost` from
  the balance and adds it to `debited` (`core/transaction.py:220-227`). There is no
  separate transition charge: the transition is time only. The settlement is queued at
  `t + transition_ticks[lane] + occupancy_ticks` (`core/transaction.py:209-213`).
- The record's `start_tick` excludes the transition (`h_done`, `core/handlers.py:113`),
  and a cancel's spent fraction is measured from that start
  (`core/handlers.py:331-332`), so a cancel during the transition credits the whole
  cost (`core/handlers.py:363`).
- The lane generation increments at admission (`core/transaction.py:216`) and a
  settlement whose generation is not the lane's current one is dropped as stale
  (`core/handlers.py:99`).
- Events sort by `(tick, stage, micro, tie, sid)` (`core/events.py:20-22`); stages are
  SETTLE, RELEASE, ANALYZE, DECIDE, ADMIT (`core/enums.py:18-25`); ties are 0 to 2
  with the neutral rank 1 (`core/enums.py:12-14`), and fault rows take 0 or 2
  (`fault_tie`, `core/ledger.py:70-77`).

Decisions (S6 builds them):

- Charge at commit. A commitment is debited in full at its admission, as an immediate
  admission is; `COMMIT_START` charges nothing. The transition runs from the start
  tick: `COMMIT_START` queues the settlement at `start + transition_ticks[lane] +
  occupancy_ticks` and sets `lane_busy_until` to it, so a commitment that starts now is
  indistinguishable in balance and record from an immediate admission. The
  reservation covers `[start, start + transition + occupancy)`. A cancel before the
  start credits the refundable amount (fraction zero); after it the existing live
  cancel rule applies.
- The lane generation increments at `COMMIT_START`, not at the commitment's
  admission. Incrementing at admission would make the settlement of work admitted
  before the reservation (which D8 allows) stale.
- Order: `COMMIT_START` at its start tick in stage `RELEASE` with the neutral tie.
  Every settlement and fault at that tick (stage SETTLE) runs first, whatever its tie or
  id, so a settlement at the start tick under the previous generation is never made
  stale by the new one. The commitment is live before any refresh (ANALYZE) and any
  decision (DECIDE) at that tick, so a decision at the start tick sees the lane busy
  and a cancel it emits is a live cancel, not a frozen-horizon change. No tie rank is
  added and `N_TIES` is unchanged, so `caps.max_tick` and every history stay as
  pinned. Rejected: an ADMIT-stage row, which runs after the decision at its tick, so
  that decision would see a commitment that is about to start as pending and could
  issue a cancel judged under the freeze rule and executed as a live cancel; and a
  SETTLE-stage row, whose order against a same-tick settlement would depend on the tie
  shared with `fault_tie`'s rank 2 and on queue ids.

### What does an empty leaf do to the fingerprint?

It moves it. `snapshot_fingerprint` tags every listed leaf with its index and size,
even when the size is zero (`core/snapshot.py:288-290`), and salts the digest with the
number of leaves (`core/snapshot.py:291`). Only the `product` column is conditional
today (`core/snapshot.py:283-285`). The log columns hashed are `LOG_INT_FIELDS`, every
integer field of `RecordLog` (`core/records.py:392-396`), so a new `Record` column
enters every fingerprint unless excluded. Test:
`tests/core/test_snapshot_fingerprint.py::test_an_empty_leaf_moves_the_fingerprint`
adds one zero-size integer leaf to an unchanged view and the fingerprint changes. Every
3a leaf must therefore be appended only when its capacity is nonzero (D11), and K1's
"always-on empty leaf" mutant is caught by the fingerprint pin.

## Checks run

See `slice-2b-r0a.md`.

## Spec silences decided here

- The `COMMIT_START` order and charge timing above (the spec leaves both to S0).
- The K3 workload is the toy survey-1 study, two worlds, 400 steps. Rejected: a
  corpus case with a host service or refresh schedule, which runs the chunked driver
  and mixes host time into the per-step figure.
- The mutant is an extra select-and-handle pass, not a dummy handler (the spec notes a
  dummy handler is not guaranteed to cost anything).
