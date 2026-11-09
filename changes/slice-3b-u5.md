---
orphan: true
---

# Slice 3b U5: asset ops state and ops-dependent cost

An asset now has an ops state that a process can change, and an admission is priced from
the cost row of the state it finds. The state is stamped on the occupancy at admission and
read back when the occupancy completes or is cancelled, so the charge kept on its record
and the refund of a cancel are exactly the debit, even when the asset has changed state
in between. A commitment keeps the state it was admitted under. A version-2 channel model
reads the stamp as `context.ops_state`. A study without ops states keeps one state, carries
none of the new leaves and lowers the same program.

## What changed

- `core/capacities.py`: `n_ops` (static, optional, default 1) is the number of ops
  states; property `ops` (true above 1). It is the static switch for the feature.
- `core/ledger.py`: `Ledger.ops_state` (`I64[A]`) and `Ledger.cm_ops` (`I64[K]`, size zero
  while commitments are off), present with `n_ops > 1`; `ChannelTable.cost_by_ops`
  (optional, `F64[n_ops, C, R]`); `charge_parts(study, channel_index, ops=0)` reads
  `cost_by_ops[ops]` when declared, in place of `cost` (no refunds) or `charge` (refunds);
  `set_ops_state(carry, asset, state, caps, where=True)`, the call a process makes from
  its `handle` to move an asset (the state is clamped to 0 .. `n_ops - 1`).
- `core/events.py`, `core/records.py`: `EventTable.ops` (the request and refresh rows share
  the type) and `Record.ops` / `RecordLog.ops`, present with `n_ops > 1`; `enqueue` takes
  `ops`. The log column is not an integer column of the generic lists (`OPS_COLUMNS`,
  beside the bundle column), and `write_log`, the replay row and the log-row reader carry
  it.
- `core/transaction.py`: admission reads `ops_state[lane_asset[lane]]`, debits
  `charge_parts(..., ops)` (the group check of `_group_code` prices each row the same
  way), and puts the state on the Done row, and on the start row and `cm_ops` of a
  commitment. Bundle members share the admission tick and the ledger, so they share it.
- `core/handlers.py`: Done and `_cancel_one` price the record, the retained charge and
  the refund from the stored state (a cancelled commitment reads it from its start row);
  a commitment's start hands `cm_ops` to its Done row; `Record.ops` carries the state.
- `core/kinds.py`: `RealizeContext.ops_state`, from the record. The fork battery reads the
  log row, which holds `ops`, so it needed no change.
- `core/snapshot.py`: the policy view's log has no ops column; `snapshot_view` otherwise
  unchanged.
- `core/executors/base.py`: `ops_on` (the event table has the column) gates `carry.ev.ops`,
  `carry.log.ops`, `carry.request.ops`, `carry.refresh.ops`, `carry.ledger.cm_ops` and
  `carry.ledger.ops_state`.
- `core/study.py`: the balance-range check includes `cost_by_ops`.
- `core/conformance/invariants.py`: the conservation check prices queued settlements and
  award-replay debits from the ops state stored on the event or the record (the
  declaration is read independently of the ledger helper). A completion the log refused
  is not priced from the trace, which holds no state: with a per-state table declared the
  check raises rather than guess.
- Pre-flight: finding code `O001` for `cost_by_ops` declared while `n_ops` is 1, a wrong
  shape, and an entry that is negative or not finite. An ops index out of range is not a
  declaration the study can see (a process writes it at run time); `set_ops_state`
  clamps it.

## Tests

`tests/core/test_ops_state.py` (9 tests; one compiled structure with commitments and two
states on, scenarios are script data and process leaves; the failure process and the version-2
model are doubles in `tests/doubles.py`):

| Test | Tier |
| --- | --- |
| three admissions before the failure debit 2, the two after it 5; only asset 0 changes state | medium (shared host run) |
| a completion after the failure and a cancel of an occupancy admitted before it keep price 2, one admitted after keeps 5; the refund total is exact | medium (shared host run) |
| a commitment admitted at 0 and started at 260 completes charged 2, state 0 | medium (shared host run) |
| a version-2 model realizes with the stored state, including after the failure | medium (shared host run) |
| the failure notice is released at the failure tick plus the delay; the policy view has no ops column | medium (shared host run) |
| host and compiled histories equal; a later failure and a longer delay on the same program move the release and add no trace | medium (shared compiled run) |
| fork battery passes on a run whose records straddle the failure | medium |
| the cost table and `n_ops` round trip through the codec; a study without them writes neither | unit |
| a study without ops carries no ops leaf | unit |

`tests/core/test_conservation.py` gains two tests on a host run of the same study: the
debits equal the retained charges plus the refunds in every account, and a completion
repriced at the current state is caught.

Also: `O001` seeded cases (three) and one clean case in `preflight_cases.py` with their
locators, `n_ops`, `cost_by_ops` and `Record.ops` in the optional-field inventories of
`test_optional_fields.py` and `test_slice3_guard.py` (the non-default round trip carries
`n_ops`), and two malformed `n_ops` cases. `tests/adapters/test_summary_channel.py` builds
its `RealizeContext` with the two fields the context gained.

Negative controls (run at merge time, not standing tests): recomputing the charge at Done
from the current state fails two tests; recomputing it at cancel fails two; admission
always reading state 0 fails two; a commitment's Done row reading the current state in
place of `cm_ops` fails the commitment test and the conservation test; a context that
ignores the stored state fails the version-2 test; a process that does not set the state
fails two; the fork battery dropping the log's ops column fails the fork test; a refund
credit that ignores the stored state (using the declared cost) fails two.

## Receipts

- Default program, measured against the commit before this change (not a standing test):
  the lowered jaxpr of one `step` is identical for the default capacities (`810cd70791ee7703`,
  269213 characters) and for a study with programs, awards, refunds, commitments, bundles
  and a configuration timeline on (`3840e24ef74e1f38`, 647041 characters).
- Program budgets: `tests/core/test_ops_state.py` 216 programs loaded from the cache
  (warm), budget 260; `tests/core/test_conservation.py` 167, budget raised from 170 to 205
  for the added structure. Wall: about 6 s warm for the new file.
