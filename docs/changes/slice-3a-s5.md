---
orphan: true
---

# Slice 3a S5: awards, windows and remainder rules

S5 lets a policy grant a balance row an amount for a window with an Allocate effect, closes
the window with a `WINDOW_END` event, and applies the remainder rule the award declared. It is
tier 2 (core handlers, ledger, admission, snapshot, pre-flight and conformance). With
`caps.awards` off (the default) nothing changes: the compiled program, every history digest
and every pin are those of `main`. The window-aware late-refund disposition (S5b) is not in
this change.

## What is in

- **Window states.** `WindowState` is ALWAYS (no window, the default), WINDOWED (an award is
  open) and IDLE (the window ended and nothing reopened it). `RemainderRule` (NONE, FORFEIT,
  CARRY_OVER, REALLOCATE) is no longer reserved.
- **Allocate.** An admitted Allocate adds its amount, one entry per resource, to the recipient
  balance row, opens the window `[window_from, window_until)` (the existing validity columns
  of the ledger), records the rule and the REALLOCATE destination, bumps the row's window
  generation `win_gen` and queues one `WINDOW_END` event. The effect table gains the optional
  columns `amount`, `window_from`, `window_until`, `recipient`, `remainder` and `realloc_to`.
- **Admission.** A malformed Allocate (recipient or destination out of range, a negative or
  non-finite amount, an empty or finished window, a window past the last tick, an unknown
  rule, a REALLOCATE onto its own row) is UNSUPPORTED. An award onto a row that already holds
  an open window is FORBIDDEN, so one row holds one window and the refused award changes
  nothing. A full queue is NO_SLOT. A row used outside its window is OUTSIDE_VALIDITY, before
  `from` and after the expiry.
- **WINDOW_END.** The twelfth handler. It fires at tick `until`, in stage ANALYZE with
  `TIE_NEUTRAL` (so it sorts after a same-tick settlement and release, ahead of any decision
  in that tick, and after a same-tick refresh, whose tie `REFRESH_TIE` is 0). No new tie rank
  was added and `N_TIES` is unchanged. The event names its row and the generation it closes;
  an event whose generation is no longer the row's changes nothing. A live one bumps the
  generation, then applies the rule: NONE keeps the balance (IDLE), FORFEIT lapses it (IDLE),
  CARRY_OVER keeps it and reopens the row (ALWAYS), REALLOCATE moves it to the declared row
  (granted there, lapsed here; IDLE). An acquisition that completes after the expiry is
  neither cancelled nor re-validated.
- **Adjacent windows.** A new Allocate in the DECIDE stage of the expiry tick sees the row IDLE
  and is admitted.
- **Counters.** `ledger.granted` and `ledger.lapsed` (per row and resource) record every grant
  and lapse; `ledger.win_state`, `win_gen` and `realloc_row` hold the window. `balance_conserved`
  adds granted less lapsed to its three-way account and checks the counters against the trace:
  a row's granted is at least what the trace allocated to it, any excess is a reallocation
  that lapses the same amount at its source, and nothing lapses that was never given.
- **Trace.** The Allocate rows of a decision keep their amount, window and recipient in
  `trace.alloc_amount`, `alloc_from`, `alloc_until` and `alloc_recipient`.
- **Conditional leaves.** With awards on: `ledger.win_state`, `win_gen`, `realloc_row`,
  `granted`, `lapsed`, the four `trace.alloc_*` leaves and the six effect columns. They are
  `None` otherwise, listed in `CANONICAL_FIELDS` and gated by `awards_on` in
  `CONDITIONAL_FIELDS`. `step` slices the handler table (`HANDLERS[:-1]`) when awards are off,
  so a study without awards lowers the table it always had.
- **Pre-flight.** X001 (effect not supported by the executor) is lifted for Allocate while
  `caps.awards` is on, through `GATED_EFFECTS`. W001 (ERROR) flags a declared award whose
  balance row is out of range, whose window is empty or outside the run, or that overlaps
  another award on the same row. `Declaration.awards` is its input: a tuple of
  `(row, start, until)`.
- **Late refunds.** Under `caps.refunds` and `caps.awards`, a DONE or cancel row records the
  generation of the window it was charged in (`award_gen`, carried through `disposition`), so
  S5b can decide a late refund by window. S5 itself does not use it.

## Moves

None. `tests/core/golden_histories.json` and every pin file are unchanged.

| Check | Result |
|---|---|
| Default scan StableHLO | byte-identical to `main`; both lowered here (Linux x86_64, JAX 0.10.2) to sha256 83c60f89...; the figure in earlier change records (2a0a4ec2...) came from another machine |
| Golden histories | integer digests match; float digests differ on this Linux machine exactly as on `main` (the Mac run is the check) |

## Sibling sites checked

- `h_done` and `_cancel_lane` set `award_gen` only under `caps.refunds` and `caps.awards`;
  `_commit_observe` reads the window generation only under both. The `_group_code`
  admission chain keeps its original conditions and gains the OUTSIDE_VALIDITY test as a
  static insertion under `caps.awards`.
- Every walk over `Ledger`, `Trace` and `EffectTable` fields tolerates the `None` leaves: the
  battery walkers (truth swap, latency shift, fork, metamorphic epoch) skip absent trace leaves
  through `carry.absent`, which spells each name out; `mutants._step_with_table` leaves the
  window handler out of a mutant table when awards are off, as `step` does.
- The dispatch lint bans `getattr` walks: the pre-flight check and `_with_award_columns` name
  their columns.
- Shared-file edit: `.pre-commit-config.yaml` adds `award` to the `name-tests-test` exclude list
  (support modules), for `tests/core/award_helpers.py`.
- The refresh mutation pair enables awards, so the award leaves are held and mutated on a run
  that carries them.

## Tests

| File | Tier | Control or mutant |
|---|---|---|
| `tests/core/test_awards.py` | medium; one slow parity run | first allocation, each remainder rule's balance, OUTSIDE_VALIDITY before and after, an overlap refused with the carry unchanged, adjacent windows, completion after expiry, same-tick ordering, a stale expiry (with a live control), trace retention, conservation per scenario, forged granted, lapsed and balance counters, awards off holds no award leaf or event, host and compiled parity |
| `tests/codec/test_optional_fields.py` | unit | the six effect columns: omitted at default, old document loads, null refused, populated round trip |
| `tests/core/test_preflight.py`, `tests/codec/test_preflight_json.py` | unit | W001 seeded cases and pointers, X001 with and without awards |
| `tests/core/test_refresh_runs.py` | medium | the mutation pair holds and mutates every award leaf |

## Spec silences decided here

1. FORBIDDEN is the code for an award onto a row with an open window; the spec names no code.
2. UNSUPPORTED is the code for a malformed Allocate row.
3. `award_gen` takes the generation of the first payer row of the charge.
4. `disposition` carries that generation on the DONE row (the column was otherwise unused
   there), so no new event field was needed.
5. No X002 window field existed to lift: X002 flags channel refund declarations, which S3a
   already gated, so only X001 changed.
6. `Declaration.awards` is the W001 input; a policy that allocates without declaring is not
   checked by W001 (the run itself still refuses a malformed or overlapping award).
7. A CARRY_OVER row returns to ALWAYS and keeps its `valid_until_tick`; the next Allocate
   resets both.

## K3 step cost

`tools/step_cost.py` measures only the default study. With awards off `step` slices the table
to the eleven handlers `main` has and the lowered module is byte-identical, so the default step
cost is that of `main` by construction. The awards-on cost (twelve branches, plus the extra
carry leaves) was measured with the same method (`measure`, two worlds, 400 steps, 30
repeats, caches off), interleaved in one job as the S0 record recommends: `main` default,
S5 default and S5 with `caps.awards`, three rounds, on the Linux desktop under the compute
launcher (a shared WSL machine, so the spread between rounds is about 8 percent).

| Round | `main` default min per step | S5 default | S5 awards on | Awards on / `main` |
|---|---|---|---|---|
| 1 | 205.2 us | 203.2 us | 237.0 us | 1.155 |
| 2 | 216.8 us | 191.2 us | 235.3 us | 1.085 |
| 3 | 221.8 us | 198.4 us | 231.5 us | 1.044 |

Cold first call: 13.6 to 15.1 s on `main`, 13.7 to 15.8 s for S5 default, 15.6 to 16.8 s with
awards on. The default stays at or below `main` in every round. Awards on costs 4 to 16
percent more per step than `main` (mean 9.5 percent, median 8.5 percent), which brackets the
10 percent threshold of K3, so the awards-on figure is not a clear pass. It is a figure for the
twelve-handler table without `COMMIT_START`; K3 is judged at the final table on the CI runner,
and this run does not replace it. Raw output is in the session scratchpad, not in the repository.

## K4 attribution

K4 is 3b.
