---
orphan: true
---

# Slice 3a S5b: window-aware refunds

A late failure receipt no longer credits whatever the paying row holds now. Each debit
keeps the identity of the award it was made under, and a refund is disposed by that
identity. No new `CORE_KINDS`, so no entry-point change. With `caps.awards` off (the
default) nothing changes: the compiled program, every history digest and every pin are
those of `main`.

## What is in

- **Award identity per payer slot.** Alongside `award_gens` (the window generation of
  each paying row at the debit) a debit now keeps the remainder rule and the
  reallocation target of that row's window, `award_rules` and `award_tos`. A debit made
  outside a window has generation 0 and rule NONE, whatever generation the row has
  reached (a closed window leaves the row's counter behind). They are carried from the debit to the settlement or
  cancellation record in the conditional ledger leaves `lane_award_rules` and
  `lane_award_tos`, exactly as `lane_award_gens`, and exist only under `caps.awards`.
- **Disposition.** `_apply_refund` credits a payer row directly only while its window
  generation still equals the one captured at the debit (nothing has opened or closed on
  the row since). Otherwise the credit is disposed under the captured rule, slot by
  slot:
  - FORFEIT lapses it: `refunded` and `lapsed` both rise and the balance is unchanged.
  - NONE and CARRY_OVER credit the row if no window is open on it (it is IDLE or ALWAYS),
    and lapse the credit if a newer award holds it.
  - REALLOCATE credits the captured target row (`refunded` rises there).
  A newer award's balance is never altered by a refund to an older one, and an IDLE row
  does not satisfy the generation test. The eligibility rules of S3a are unchanged.
- **Replay.** `balance_conserved` replays each applied receipt from the origin's captured
  identity against the replayed row state (generation and open window), independently of
  the handler, and requires `balance`, `granted`, `lapsed`, `refunded` and `win_gen` to
  equal the replay. A record's captured rule and target must also match the window its
  paying row held at the debit. The per-row refund account is taken from the replay, so a
  credit that moved to another row or lapsed is accounted where it landed. Cancellation
  credits are replayed into `refunded` as well; the retained-spend reconciliation keeps
  a separate receipt-only account, since a cancel's credit is already in its record.

## Moves

None. `tests/core/golden_histories.json` and every pin file are unchanged. Default
StableHLO is byte-identical to `main`.

## Behaviour change from S5

A refund after the window of a FORFEIT award used to credit the row; it now lapses. The
S5 control that compared an early and a late refund now expects the same lapsed total and
balance for both.

## Tests

| File | Tier | Control or mutant |
|---|---|---|
| `tests/core/test_refund.py` | medium | one table-driven test over scenarios that share one compiled run each: a late refund of a FORFEIT award lapses under a newer award of every rule and, with a refund of exactly 5, when the row ends ALWAYS; an IDLE row with no newer award takes a NONE refund; a NONE refund lapses under a newer award; a REALLOCATE refund goes to its destination; a refund inside the window credits the row. Forged credits to the newer award and to the expired row fail (slow) |
| `tests/core/test_conservation.py` | medium | a cancellation under an award conserves (the ledger refunds 5); a debit after its window expired names generation 0 and rule NONE; a refund across an expiry and a re-award conserves; a handler mutant that always credits the row fails (slow) |
| `tests/codec/test_optional_fields.py` | unit | the two new record columns: omitted at default, null refused, populated round trip |

## Core paths changed (K4)

The exact core paths that S5b changed after the S0 close-out, which the Slice 3a gate matches against this record:

- `src/spaceodyssey/core/carry.py`
- `src/spaceodyssey/core/conformance/invariants.py`
- `src/spaceodyssey/core/executors/base.py`
- `src/spaceodyssey/core/handlers.py`
- `src/spaceodyssey/core/ledger.py`
- `src/spaceodyssey/core/records.py`
- `src/spaceodyssey/core/snapshot.py`
- `src/spaceodyssey/core/transaction.py`

Core tests that cover this change: `tests/core/test_conservation.py`, `tests/core/test_refund.py`.
