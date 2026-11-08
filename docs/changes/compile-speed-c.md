# Compile speed: rolled admission, shared handler bodies, one policy view

Three refactors shrink the compiled transition. Every simulation history, fingerprint,
golden pin, the FAULT fixture and every corpus document hash is bit-identical to before.
The compiled program itself (StableHLO) changes by design, so its hash changes.

## What changed

- **Admission loops rolled** (`core/transaction.py`). `_admit_rows` ran its per-row body
  once per effect row in Python, and again once per member row for the observation
  commit, so the program grew with `n_effects` squared. The row loop and the commit loop
  are now `lax.fori_loop`s, and `_group_code` judges all rows in one vectorized pass.
  The order is unchanged: rows are admitted in table order, a refused group takes no
  acquisition index, stable ids are allocated in the same order, the first refusal in
  row order wins, and the demand on a resource row is still added one member at a time
  from left to right (a `fori_loop`, never a parallel reduction).
- **Shared handler bodies** (`core/handlers.py`). `step` held one branch per event kind,
  so RESULT traced the RELEASE body a second time, and FAULT traced a second copy of the
  CANCEL body. `step` now builds the switch from the distinct bodies and maps each kind
  to its body with an index table: 9 branches without awards (was 11) and 10 with
  awards (was 12). A fault runs the cancel body with its status and refund taken from
  the event kind (`_cancel_lane` takes them as values, not static arguments, so the
  cancel, fault and stop callers share one traced function). `h_result` and `h_fault`
  stay as functions with the same behavior.
- **One policy view** (`core/transaction.py`). `transition` builds the policy snapshot
  view once and hands it to both `decide` and `decision_fingerprint` (both take an
  optional `view`). The saving is small because XLA already merged the duplicate work.

A decision whose effect table has fewer rows than `Capacities.n_effects` was
silently read with clamped indices before. It now fails at trace time with a shape
error. A study must build effect tables of `n_effects` rows.

## Measured effect

Mac, cold (`JAX_ENABLE_COMPILATION_CACHE=0`), lowering `_scan_worlds` for 2 worlds and
40 steps. The `n_effects=4` study is the toy survey with a plan policy padded to four
effect rows.

| study | metric | before | after |
|---|---|---|---|
| default | cold trace, lower and compile | 5.1 s | 3.5 s |
| default | jaxpr equations | 13,873 | 8,891 |
| default | StableHLO lines | 14,671 | 10,962 |
| n_effects=4 | cold trace, lower and compile | 9.3 s | 3.5 s |
| n_effects=4 | jaxpr equations | 24,463 | 8,909 |
| n_effects=4 | StableHLO lines | 19,661 | 10,995 |

The default program grows by 18 equations from two to four effect rows (it grew by
10,600 before).

Cold quick tier on the Mac (`-m "not slow" -n 2 --dist loadfile`, compilation cache
off): about 250 to 290 s before, 220 s after (the before runs shared the machine with
other jobs, so treat the gap as indicative).

Compiled-program hash, default study (`survey_1`, 2 worlds, 40 steps,
`lower().as_text()`):

- before: `a67917728c4878feaa35f3bccc622be9ff6850ec65b5adf22867e668c9d1d791`, 14,671 lines
- after: `8b8ee715bb59413b179fa7c79f086ac18556ed7ae56d0b639c617fb291fd00a0`, 10,962 lines

Per refactor, default study (equations, lines, cold time): rolled admission 10,803,
12,746, 4.3 s; plus shared handler bodies 9,206, 10,969, 3.5 s; plus one policy view
8,891, 10,962, 3.5 s.

## Evidence that results are unchanged

- Golden pins, corpus invariants, the corpus and all of `tests/codec` (slow tests
  included) pass unchanged.
- `admit` was compared before and after on randomized effect tables (all effect kinds,
  groups, refusals of every code, Query, Allocate, mixed Query, narrow masks, random
  balances, ticks, busy lanes and closed lanes), carry and codes, dtype, shape and raw
  bytes: no difference.
- Every carry and trace leaf of every executor run (734 runs, 121,594 leaves) in the
  award, refund, program, balance, charge, conservation, stop, corpus and survey tests
  was compared between the two versions on both executors, by dtype, shape and raw bytes: no difference.

## Tests

`tests/core/test_program_size.py` holds the admission program flat in `n_effects`,
checks the switch holds only distinct bodies, and checks the shared branches equal the
handlers of their own kinds.
