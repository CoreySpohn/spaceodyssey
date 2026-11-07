---
orphan: true
---

# Slice 3a S3a: charge components, basic refunds and conservation

S3a adds the pricing modes and a fixed surcharge to what an admission debits, lets a failure
receipt credit the refundable remainder of the acquisition it names, and extends the balance
conservation invariant to match. It is tier 2 (core handlers, records, snapshot and pre-flight).
With `caps.refunds` off (the default) nothing changes: the compiled program, every history
digest and every pin are those of `main`.

## What is in

- **Charge components.** `ChannelTable.charge` (already present) is the variable debit and is
  read only while `caps.refunds` is on; otherwise the cost is debited. Four optional channel
  columns declare the rest: `pricing` (`ChargePricing`: OCCUPANCY, STATISTICAL, FLAT_FRACTION,
  SURCHARGE), `surcharge`, `failure_kind` and `refund_deadline_ticks`. The fixed part of a charge
  is `surcharge` where the pricing is SURCHARGE and zero elsewhere. `ledger.charge_parts` returns
  `(charge, fixed, pricing)`; with refunds off it returns the cost, zeros and OCCUPANCY.
- **Records.** `Record.charged` is the retained charge and `Record.nonrefundable` the fixed part.
  A cancel keeps `(charge - fixed) * part + fixed`, where `part` is the spent fraction, or 1 for
  FLAT_FRACTION (a flat price returns nothing on a cancel but all of it on a failure).
- **Basic refunds.** A receipt is a process-written record whose kind is a channel's declared
  `failure_kind`. It names its target by semantic identity: `subject`, `source` (the target
  channel id) and `refund_index` (the target acquisition index). The credit is applied in
  `h_release`, on the first release to the policy audience, and only when the target exists, the
  kind is the failure cause its channel declares, the release is within `refund_deadline_ticks`
  of the target's end, and no receipt has credited the acquisition before. The credit is
  `charged - nonrefundable` of the target, goes to the target's lane and program rows, and is
  written on the receipt row as a negative `charged`. A receipt that does not apply counts in
  `ledger.refund_ignored`.
- **Settlement boundary.** Held (never released), not-yet-released and observer-only receipts
  credit nothing and move no policy balance or fingerprint.
- **Masking.** `refund_of` keeps its legacy value of -1 in every view and masked record
  (`records.hide_refund_of`); the policy sees the semantic `refund_index`, never the queue
  identity of the target. A private event before the receipt shifts internal ids and not the
  policy fingerprint.
- **Conditional leaves.** With refunds on: log columns `nonrefundable`, `refund_index` and
  `award_gen`, and `ledger.refund_ignored`. They are `None` otherwise, listed in
  `CANONICAL_FIELDS` and gated by `refunds_on` in `CONDITIONAL_FIELDS`. `refund_index` and
  `award_gen` enter the snapshot fingerprint; `nonrefundable` is a float and does not.
- **Conservation.** `balance_conserved` is program-aware (it reads the balance row of the
  record's program, which S2 left on the lane row) and prices a record from its own charge
  components, written out independently of `charge_parts`. A receipt's negative charge is
  attributed to its target's lane and program.
- **Pre-flight.** X002 now also flags `charge != cost` and the refund declarations while
  `refunds` is off. W002 (ERROR) flags a failure kind that is unknown, or that no policy-readable
  payload column carries, while `refunds` is on.

## Moves

None. `tests/core/golden_histories.json` and every pin file are unchanged.

| Check | Result |
|---|---|
| Default scan StableHLO | byte-identical to `main` (794a715), sha256 2a0a4ec2...; compared after the same lowering |
| Full suite without the slow tier (excluding `tests/adapters` and two OFTI files, which do not import here) | 2180 passed, 2 failed; both fail identically on `main` (an installed photomancy version, an entry point not installed) |
| Golden histories, fingerprint pins, FAULT fixture, 58-document hash pin | pass in that run; float digests differ on this Linux machine as on `main`, so the Mac run is the check |
| Slow tier of `test_slice3_guard.py`, `test_programs.py`, `test_canonical_fields.py` and the five new mutants | pass |

## Sibling sites checked

- Every `charged=` and `cost=` site on a record (`h_done`, `_cancel_lane`, the process rows):
  with refunds off each keeps its original expression, so the lowered module is unchanged.
  The first version computed the charge once ahead of the branch and moved the module text;
  each call now sits inside the `caps.refunds` branch or at its original site.
- The domain-word lint bans the word `target` in core, so the receipt's referent is `origin` in code. The dispatch allowlist (`tests/core/gate_helpers.py`) gained entries for the optional-column walks in `_log_rec`, `_row`, `write_log` and `snapshot_view`; the pre-flight check names its columns instead of using `getattr`.
- `snapshot_view` and `masked_record` apply `hide_refund_of` only while refunds are on.
- Every walk over `Record`, `RecordLog` and `Ledger` fields (`records.py`, `snapshot.py`,
  `handlers.py`, `executors/replay.py`, `executors/base.py`) tolerates the `None` columns.
- `_balance_rows`, `_retained_net_spend` and `_refused_completions` in the invariants.
- Shared-file edit: `.pre-commit-config.yaml` adds `refund` to the `name-tests-test` exclude list
  (support modules), for `tests/core/refund_helpers.py`.

## Tests

| File | Tier | Control or mutant |
|---|---|---|
| `tests/core/test_charging.py` | medium; one slow mutant | pricing matrix, defaults, `charge_parts` units; a cancel that credits the surcharge |
| `tests/core/test_refund.py` | medium; four slow mutants | credit equals the remainder, once-only, deadline, wrong identity, delayed, held, observer-only (W002 and bypassed), views, K6 shift; mutants for whole-charge credit, double credit, no deadline, any-release credit; an unmasked `refund_of` |
| `tests/core/test_conservation.py` | medium | all accounts agree, per-row netting, forged balance, forged consistent balance and counter, lost receipt charge, log overflow |
| `tests/codec/test_optional_fields.py` | unit | every new channel column: omitted at default, round trip, null refused, old document keeps its hash |
| `tests/core/test_preflight.py`, `tests/codec/test_preflight_json.py` | unit | seeded cases and pointers for the new X002 and W002 findings |

## Spec silences decided here

1. A receipt carries its target channel in `Record.source` and the target acquisition index in
   `refund_index`; no new record field names the channel.
2. `refund_ignored` is one scalar, not per resource or per channel.
3. An applied receipt is detected by `refund_of == target acquisition id`, and the receipt row
   carries a negative `charged` so the independent reconciliation still nets.
4. The refund and pricing declarations are optional `ChannelTable` columns, absent at their
   default, rather than new required columns, so no existing document moves.
5. STATISTICAL prices like OCCUPANCY; the fraction is expressed in `charge`.
6. `_balance_rows` became program-aware here because a refund to a program row exposed the S2
   gap in the invariant.
7. A receipt is any record of a declared failure kind written on a channel other than the one
   it names (`channel != source`); an acquisition of the failure kind is not a receipt, and the
   origin may share the receipt's kind. No receipt names its own row (`n != slot`).

## Round 1 review fixes

- Conservation no longer trusts a receipt row's negative `charged`: the allowed credit is
  recomputed from the origin record (`charged` less `nonrefundable`, first receipt only), the
  ledger counters are held to it, and a receipt row that does not carry that credit fails.
  Mutants: whole-charge credit and a missing once-only test, each on program 0 and program 1.
- `Record.nonrefundable`, `refund_index` and `award_gen` are optional document fields (omitted
  at default, old documents load, a null is refused, populated columns round-trip).
- Log overflow: a refused completion is charged to the program the trace names. `Trace.program`
  is a conditional leaf (present only with more than one program, registered in
  `CONDITIONAL_FIELDS` as `trace.program` under `programs_on`), so no one-program history or
  pin moves. A debit moved to the wrong program after an overflow now fails conservation.
- The observer-release mutant matches the handler's multiline source and is asserted before use.
- Default lowered scan hash unchanged (2a0a4ec280c8e0bbeaea...).

## K3 and K4 attribution

`handlers.py` gained branches, all static on `caps.refunds`. With refunds off the lowered
default scan is byte-identical to `main`, so the step cost is that of `main`. `tools/step_cost.py --mutant` was not rerun here; the Mac and CI runner remain the baseline of record.
