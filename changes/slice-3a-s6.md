---
orphan: true
---

# Slice 3a S6: commitments, the freeze rule and disruptive activations

S6 adds commitments (a request that names a future start), reservations on a lane, the
insertion and freeze rule, disruptive activations and their charge, and lifts the X002
refusal of `freeze_ticks` and `disruptive`. Every new leaf is conditional: a study with
`n_commitments == 0` lowers the program it always did.

## Moves

None. The default lowered program is byte-identical to the base (same HLO text hash), and no
pinned digest, fingerprint, FAULT fixture or corpus hash changes.

## What is in

- `COMMIT_START` event kind and the `CommitState` states (none, pending, started, cancelled).
- Ledger leaves `cm_state`, `cm_lane`, `cm_program`, `cm_start`, `cm_end`, `cm_index`,
  `cm_count`; `EffectTable.start_tick`. All are omitted at the default, load from old
  documents, refuse an explicit null and round trip exactly.
- Admission: a start-based completion time, room counted over free commitment slots, overlap
  with a pending reservation refused as `LANE_BUSY`, and `FROZEN_HORIZON` for an insertion
  that starts less than `freeze_ticks` ahead.
- Disruptive activations: an insertion inside the horizon that is flagged disruptive is
  admitted and debited one activation.
- `h_commit_start` occupies the lane at the start tick and queues the settlement.
- Cancel can name a pending commitment; a cancel outside the horizon credits in full.

## Decided ambiguities

- The freeze boundary is half open: an insertion is frozen when `start - t < freeze_ticks`.
- A disruptive activation is debited from `balance[asset row, disruptive_resource column]`,
  counted in `debited` and `disruptive_used`; `balance_conserved` adds it back.
- A Cancel addresses a commitment with `start_tick > now` naming the pending commitment's
  start on its lane; no match is `UNSUPPORTED`.
- Awards and commitments together are refused by an X001 preflight finding (path `caps`,
  member `n_commitments`). See "Deviations from the spec".
- The charge components of a commitment are not stored: a cancel derives them from the
  channel's price tables when it runs.
- A pending commitment whose lane (or program) is closed or drained is refused when its
  start comes due, and an abort of its scope cancels it when the abort is ordered. Either
  way it is cancelled as a Cancel would: the record is the partial one at fraction 0 and
  the credit is what its pricing allows.
- A campaign DRAIN waits for a pending commitment: `COMMIT_START` counts as drainable work,
  so the run does not end while one is pending. When the start comes due the campaign is
  draining, which admits no new running work, so the start is refused and credited back as
  above. A campaign ABORT cancels pending commitments at once.
- A group is admitted on the summed demand per resource row, activations included, in the same
  left-to-right sum as the charges. Two grouped rows that each cost 1 plus one activation are
  refused whole against a balance of 3, and no balance goes below zero.
- A second Cancel of a commitment that already has a Cancel pending is `UNSUPPORTED` and spends
  no activation. So is a Cancel behind an ABORT, already in the queue, whose scope reaches
  the commitment (lane, asset, program or campaign; `stop_reaches` in `handlers.py` is the
  one reach rule, shared with `h_stop`); a Cancel ahead of the abort, or behind one that does
  not reach it, is unaffected. The activation of a Cancel is debited only when the Cancel is
  enqueued; a full queue, or event ids spent, refuses it as `NO_SLOT`.
- A disruptive resource index outside `-1 .. n_resources - 1` is the pre-flight finding A001
  (path `assets.disruptive_resource`), covering the negative bound too.

## Deviations from the spec

Awards together with commitments stay refused (X001). Award capture today is per lane
(`lane_award_gens`, `lane_award_rules`, `lane_award_tos` are set by lane at admission), so a
later commitment on a lane would overwrite what earlier work on that lane still runs under.
Supporting both needs per-commitment capture leaves, a debit at the window current at the
commit, the windowed-refund cancel path, expiry between commit and cancel, and tests: about
two cards. The decision to build it is Corey's.

## What a cancel of a pending commitment credits

The elapsed fraction is 0, so each resource column credits as follows. OCCUPANCY and
STATISTICAL: the whole charge. SURCHARGE: the charge less the surcharge, which is fixed
and stays with the record (`nonrefundable`). FLAT_FRACTION: nothing, by the literal rule
that a flat price returns nothing on a cancel (D8 credits the refundable amount, and under
a flat price that amount is 0), even though the commitment never started. The disruptive
activation of an insertion is debited at admission and is never credited back by a cancel.

## Tests

`tests/core/test_frozen_horizon.py` runs the A4 controls on one module-scoped compile:
the plan executes; an immediate frozen insertion is refused; a disruptive one is admitted and
debits one; an uncovered activation is refused; a zero horizon is unaffected; a cancel inside
the horizon is refused; a disruptive cancel succeeds; a cancel outside credits fully; a
cancel of a started commitment is not frozen; work before a reservation is admitted and
overlapping work is refused. A Stop of the lane (close, drain, abort) or of the program that
holds a pending commitment is covered, one test per case, with compiled-versus-host
agreement in a slow parametrized test, and one priced pending cancel per pricing mode
asserts the credit, `charged`, `nonrefundable`, the record fraction and conservation.
A campaign drain and abort, the group and Cancel overdraw cases, the slot capacity (a table of
one row), the horizon boundary (a start exactly one horizon away is not an insertion) and a
request during a running commitment are covered the same way. Each asserts conservation and
that no balance is negative. The mutants of `h_commit_start` (lane busy until the end),
`_frozen` (`<` to `<=`) and `_group_code` (room over free slots) each fail one of these tests.

## Changed core paths and their tests (K4)

| Core path | Test |
|---|---|
| `core/enums.py` | `tests/core/test_frozen_horizon.py` |
| `core/capacities.py` | `tests/core/test_frozen_horizon.py`, `tests/core/test_preflight.py` |
| `core/ledger.py`, `core/carry.py` | `tests/codec/test_optional_fields.py`, `tests/core/test_frozen_horizon.py` |
| `core/snapshot.py` | `tests/codec/test_optional_fields.py`, `tests/core/test_frozen_horizon.py` |
| `core/executors/base.py` | `tests/codec/test_optional_fields.py`, `tests/core/test_canonical_fields.py`, `tests/core/mutants.py` |
| `core/transaction.py` (group demand, cancel dedupe, slots) | `tests/core/test_frozen_horizon.py` |
| `core/transaction.py`, `core/handlers.py` (`advance` drain, `h_commit_start`, `h_stop`) | `tests/core/test_frozen_horizon.py` |
| `core/handlers.py` | `tests/core/test_frozen_horizon.py`, `tests/core/test_program_size.py` |
| `core/conformance/invariants.py` | `tests/core/test_frozen_horizon.py` |
| `core/preflight/checks.py`, `core/preflight/findings.py` (A001) | `tests/core/test_preflight.py`, `tests/codec/test_preflight_json.py` |

## Core paths changed (K4)

The exact core paths that S6 changed after the S0 close-out, which the Slice 3a gate matches against this record:

- `src/spaceodyssey/core/capacities.py`
- `src/spaceodyssey/core/carry.py`
- `src/spaceodyssey/core/conformance/invariants.py`
- `src/spaceodyssey/core/enums.py`
- `src/spaceodyssey/core/executors/base.py`
- `src/spaceodyssey/core/handlers.py`
- `src/spaceodyssey/core/ledger.py`
- `src/spaceodyssey/core/preflight/checks.py`
- `src/spaceodyssey/core/preflight/findings.py`
- `src/spaceodyssey/core/snapshot.py`
- `src/spaceodyssey/core/transaction.py`

Core tests that cover this change: `tests/core/test_canonical_fields.py`, `tests/core/test_frozen_horizon.py`, `tests/core/test_preflight.py`, `tests/core/test_program_size.py`.
