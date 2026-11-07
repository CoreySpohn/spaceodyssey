---
orphan: true
---

# Slice 3a S4: balances in the snapshot and the QUERY_MIXED refusal

S4 lets a policy read the ledger. With the capacity flag `snapshot_balances` on, the policy
view carries the balance and an integer count of quanta, the integer joins the policy
fingerprint, and a decision that combines a Query with any other effect is refused whole with
a new admission code. With the flag off (the default) nothing exists and the compiled program,
every history digest and every pin are those of `main`. It is tier 2 (core snapshot,
transaction and pre-flight).

## What is in

- **Capacities.** `snapshot_balances` (added by S1) now has a reader. A new optional field
  `balance_quantum` (float, default 1.0, finite and positive, validated at construction) is
  the unit of the integer view. Both are optional document fields.
- **View.** `Snapshot.balance` (the ledger balance, rows assets then programs then the shared
  row, one column per resource) and `Snapshot.balance_q` (`round(balance / balance_quantum)`
  as int64) are `None` unless the flag is on and the audience is the policy. A `None` field is
  no pytree leaf, so every other view keeps its structure. Program `p` is row `n_assets + p`,
  so the policy sees every program row in the order its program ids use.
- **Fingerprint.** `balance_q` is appended to the policy fingerprint through
  `snapshot.conditional_view_leaves`, only when present. The float is a convenience and is not
  hashed: a change below half a quantum leaves the digest alone, a change that crosses one
  moves it. The observer fingerprint never changes.
- **`QUERY_MIXED`.** `AdmissionCode.QUERY_MIXED = 11`, appended to the enum. While the flag is
  on, `transaction.admit` selects the permitted, non-empty rows of a decision that holds a
  Query and at least one other such row, refuses every one of them with `QUERY_MIXED` and runs
  admission as if the rows were empty. The carry is unchanged (the test compares it). The
  check is behind a static branch, `_refuses_mixed(caps)`, so with the flag off the lowered
  program is the one of `main`.
- **Pre-flight.** `balance_quantum` set to a non-default value while `snapshot_balances` is
  off is X002 at `/study/caps/balance_quantum` (an unread field, two documents for one
  behavior). Invalid values (zero, negative, nonfinite, an integer, a bool) cannot be
  constructed.
- **Codec.** Both fields are omitted at their defaults, a document from before them loads at
  the defaults, a field written at its default is refused, an explicit `null` is refused, and
  a populated pair round-trips (`tests/codec/test_optional_fields.py`).

## Moves

None. `tests/core/golden_histories.json` and every pin file are unchanged.

| Check | Result |
|---|---|
| Default scan StableHLO | byte-identical to `main` (423d74c): `_scan_worlds` on `survey_1`, two worlds, 40 steps, `lower(...).as_text()` compared byte for byte; sha256 `a67917728c4878feaa35f3bccc622be9ff6850ec65b5adf22867e668c9d1d791` on both (1,292,736 bytes); `_init_worlds` `09226cf96404994bdcd4034eb93ca3487b9bd792d502d5dd55d31251afb99f71` on both |
| Golden histories, fingerprint pins, FAULT fixture | pass |
| `tests/codec/test_run_document_runs.py` (hash pin, 58 documents) | passes |
| Canonical fields, guard, program, refresh and lint tests | pass |

The first draft of `admit` moved the `phase_at` and mask lines above the `supported`
expression; the lowered text differed from `main` in 94 lines (operation order only). The
mixed check now follows the original statements and edits `supported` only inside the static
branch.

## Sibling sites checked

- `snapshot_view` callers: `service._snapshot_for` (answers rebuild the view and compare
  fingerprints; a None field is a no-op in `tree_map`), `conformance/invariants.py`, the
  executors. The only place a `Snapshot` is built is `snapshot_view`; tests that build a
  shim (`test_snapshot_fingerprint.py`, `test_epoch_relation.py`) carry or tolerate the
  new field.
- `admit` callers: `transaction.transition` only. Both executors reach it through
  `transition`, so the host and compiled paths share the refusal; the hosted parity test runs
  both.
- Fingerprint consumers: `snapshot_fingerprint` is the only hasher; `decision_fingerprint`,
  `_snapshot_for` and the `answer_pending` rebuild all call it on a `snapshot_view`.
- Observer audience: `balance_view` returns all `None` for any audience but the policy;
  the observer fingerprint test covers on, bumped and off carries.
- Codec: `Capacities` is the only declaration that gained fields; `dump` and `load` use the
  S1 optional rule unchanged.
- Pre-flight and load agree: a document with a non-default quantum and the flag off loads
  and is flagged X002 at the same pointer the in-memory study reports.

## Tests

| File | Tests | Tier | Cold time (alone, cache off, 4 threads) |
|---|---|---|---|
| `tests/core/test_snapshot_balances.py` | 25 | unit (20, the four view mutants among them), medium (2), slow (3) | whole file 23 s; slowest alone: hosted run 12.8 s (medium), debit test 4.9 s (medium), query-alone 2.2 s |
| `tests/codec/test_optional_fields.py` | +4 cases (24 in the file) | unit | 1.3 s the file |
| `tests/core/test_preflight.py`, `tests/codec/test_preflight_json.py`, `tests/core/preflight_cases.py` | 1 seeded mistake, 1 clean case, 1 JSON pointer variant | unit | no change |

Added 25 new tests plus 4 codec cases and 3 pre-flight cases (32 in all); retired 0, net +32. Existing
tests edited: the capacities guard (`test_the_slice_3a_capacities_default_to_off` lists the
new marker) and the two `Snapshot` shims that feed `snapshot_fingerprint`
(`test_an_empty_leaf_moves_the_fingerprint` and `test_pedigree_layout.py`, which pass the new
field).

Controls and mutants (A7):

- Query alone passes, and the request names the fingerprint the rebuilt view gives.
- Query with Observe, Allocate or Wait, in both orders where an order exists (five cases), is
  refused whole and leaves the carry unchanged. `Allocate` has no admission yet (S5), so its
  row uses the declared effect kind; the refusal does not depend on whether it could run.
- With balances off, Query plus Observe admits both and Query plus Allocate gives OK and
  `UNSUPPORTED`, as on `main`.
- A forbidden row is not an effect of the decision (it keeps `FORBIDDEN`) and a second Query is
  refused as `NO_SLOT`, not as a mix.
- Mutants, each failing its checker: a refusal that selects no rows (it also shows the rebuilt
  fingerprint then differs from the request's), a refusal live without the flag, a refusal
  that counts forbidden rows, a hashed float, a dropped integer, a view shown to the observer
  and a view shown with the flag off. The three admission mutants compile a copy of
  `admit` and are in the slow tier; the four view mutants run eagerly.

## Spec silences decided here

1. **Where the quantum lives.** An optional `Capacities.balance_quantum` field, default 1.0.
   Rejected: a field on `Study` (it would need `STUDY_SLOTS` and a key version) and a ledger
   array (the value is static and sets a shape-free rounding only).
2. **The flag.** `Capacities.snapshot_balances` as S1 declared it. No new study field.
3. **Balance rows and ids.** The view carries the whole ledger plus a `balance_rows` column
   naming each row (kind 0 asset, 1 program, 2 shared; index within the kind), hashed with the
   integer balance. Review round 1 replaced the first choice, which relied on row order alone and
   let a (1 asset, 3 programs) and a (2 assets, 2 programs) layout share one digest.
4. **What counts as another effect.** Any permitted, non-empty row (Observe, Wait, Stop, Cancel,
   Allocate or a second Query row alongside a Query that is not alone). A row the phase mask
   forbids keeps `FORBIDDEN`. Two Queries and nothing else are not a mix; the second gets
   `NO_SLOT` from the existing outstanding-request rule. Rejected: counting forbidden rows
   (a fault in the policy would turn into a different code) and refusing `Wait` as harmless
   (the spec says any other effect).
5. **Whole refusal.** Every refused row gets `QUERY_MIXED`, including the Query. Rejected:
   admitting the Query and refusing the siblings (the spec says whole).
6. **Observer view.** No balance at all, not a masked zero. The observer fingerprint never
   changes.
7. **Rounding.** `jnp.round` (half to even) over the quantum, int64. A study whose
   budget over the quantum exceeds 2**53 is refused at construction (review round 1).
8. **Pre-flight.** One finding: X002 for a quantum with the flag off. No new code, since the
   spec names only `QUERY_MIXED` (runtime) for S4. A mixed decision is a runtime property of a
   policy and cannot be found from its declaration.
9. **The check is a static branch** `_refuses_mixed(caps)` rather than inline, so a mutant can
   force it.

## Gate results

Final tree, `compute-run --threads 4 --mem 14336`, `-n 2 --dist loadfile`, `-m "not slow"`,
`tests/codec/test_entry_points_installed.py` deselected (its installed metadata is stale and it
fails the same way on `main`): 2222 passed, 1 failed in 254 s, peak 3.7 GB resident. The one
failure was `test_reserved_fields.py::test_reserved_table_field_has_no_behaviour[ChannelTable.charge]`
over the unit budget (9.5 s) as the first compiled test of its worker under load; the file passes
alone (15 tests, 10 s, no violations) and S4 does not touch it. The three slow admission
mutants pass in the file run (25 passed in 23 s).

## K3 and K4 attribution

`transaction.admit` gained a static branch, so the default program is unchanged (byte
comparison above); with the flag on the added work is two reductions over the effect rows and
a `where`. K3 was not rerun: the compiled default scan is the same program as `main`. Core
edits (`enums.py`, `capacities.py`, `snapshot.py`, `transaction.py`, `preflight/checks.py`)
belong to S4, each with a test under `tests/core/`.

## Merge notes

S3a (pricing and refunds) edits `handlers.py`, `transaction.py` (`_commit_observe`,
`_group_code`), `records.py`, `snapshot.py`, `ledger.py` and `invariants.py`. S4 touches
`transaction.py` only in `admit` (new `_refuses_mixed` and `_query_mixed_rows`, a few lines in
`admit`), and `snapshot.py` in `Snapshot`, the end of `snapshot_view`, a new `balance_view` and
`conditional_view_leaves`. If S3a adds refund columns to the view, they extend the same
`conditional_view_leaves` tuple. Balance timing: the balance is the carry's ledger at the
decision, so a refund S3a credits at settlement is visible at the next decision after it is
credited, not at its release; S3a's private-event invariance (K6) needs its own check.

## Review round 1

Three findings, all accepted. The default scan and init StableHLO are unchanged
(`_scan_worlds` sha256 `a67917728c4878feaa35f3bccc622be9ff6850ec65b5adf22867e668c9d1d791`,
1,292,736 bytes; `_init_worlds` `09226cf96404994bdcd4034eb93ca3487b9bd792d502d5dd55d31251afb99f71`),
and no pin moved.

### 1. A refund queued ahead of a lone Query

A Cancel queued by an earlier decision returns balance when it runs at its tick, which can be
ahead of the request a later decision queues at the same tick. The request carried the
fingerprint of the decision's snapshot, so `answer_pending` rebuilt a view with the refunded
balance and raised "the snapshot changed between the request and its answer" on both executors.
The repro is the script Observe and Wait, Wait and Cancel, Query, Stop on the toy study.

Choice: `h_request`, the handler that holds the run for the host, now stamps the request again
with the fingerprint of the policy snapshot as it stands when the run is held, at the
decision that asked. It does so only while `caps.snapshot_balances` is on (a static branch), so
the default program is untouched. Why this one:

- The host answers on the state the run is held at, and `answer_pending` rebuilds exactly that,
  so the stamp and the rebuild agree for any ledger event ordered ahead of the request, whatever
  produced it (a refund, a settlement, a completion), not only the one in the repro. Nothing
  runs between holding the run and answering it.
- Reordering the request ahead of same-tick ledger events would move ties and `N_TIES`, and
  would change every history that holds a Query.
- Answering from a snapshot kept on the request needs a second copy of the view in the carry.
- Turning the mismatch into an admission outcome cannot work: the mismatch arises after
  admission, at a later event.

What it costs: with balances on, the request stamp is no longer the decision's fingerprint, so
the stamp stops checking that nothing visible moved between the decision and the hold. The
mixed-decision refusal and the event order already keep every source of that move out except
ledger events ahead of the request, which are exactly the case the stamp now follows. The trace
still records the decision's own fingerprint. The change is in `handlers.py` (one branch in
`h_request` and one import), which S3a also edits.

Tests: `test_a_lone_query_survives_a_refund_queued_ahead_of_it_on_both_executors` runs the repro
on both executors with balances on (it failed before the change; the same script with balances off passes on
`main`, run once by hand and not kept in the suite); `test_a_held_request_is_stamped_with_the_snapshot_it_is_answered_on` checks the stamp
directly, including that an off study leaves it alone; the mutant that keeps the decision's
stamp fails it.

### 2. Row identity

The policy view now carries `balance_rows` (`(rows, 2)` int64: kind and index within the kind),
and the digest hashes it before the integer balance. `conditional_view_leaves` lists it with
`balance_q`, so it exists only with balances on. Test: the (1, 3) and (2, 2) layouts on one
`arange(15).reshape(5, 3)` budget now give different fingerprints, and the row names are
asserted; a mutant that drops the names from the leaf list fails it.

### 3. The integer range

Choice: validate at construction, in `Study.__check_init__`, not at view time. With the flag on,
`max |budget| / balance_quantum` must be finite and at most 2**53. A debit lowers a balance and
a refund restores at most what was debited, so the budget bounds every balance and one check
covers the run. Past 2**53 a float no longer separates neighbouring counts, and the int64
conversion saturates at 2**63 (the review's quantum of 1e-12 on a budget of 10,000,000 gave
INT64_MAX for 10,000,000 and 9,999,999). A view-time check would need `checkify` inside every
compiled scan and would still fail late. A traced budget is not judged, and a study with the
flag off keeps any budget. Test: six budget and quantum pairs (the saturating pair, 2**53
accepted, 2**54, infinity and NaN refused, a pair that fits), the off study, and a mutant that
removes the check.

### Review round 1 tests and gates

Added 13 tests to `tests/core/test_snapshot_balances.py` (38 in the file): 1 refund repro run
(medium), 1 stamp check, 1 stamp mutant, 2 layout (check and mutant), 6 range cases, 1 off
study, 1 range mutant. The `Snapshot` shims in `test_snapshot_fingerprint.py` and
`test_pedigree_layout.py` carry the new field. Gate results are in the section below.

### Review round 1 gate results

- Full quick tier (`-m "not slow"`, `-n 2 --dist loadfile`, entry-points file deselected) on the
  fixed tree before the last two edits: 2232 passed, 6 failed in 350 s, peak 7.7 GB under queue
  load. Five of the six were not caused by S4: the entry-points test (stale installed metadata,
  as on `main`) and four unit or medium budget overruns under load in files S4 does not touch or
  in the balances-off control run, which was then dropped from the suite (cold alone 12 s, 39 s
  under load). The sixth, `test_allowance_is_scoped_to_rule_function_and_count`, was this
  change: the range check tested the argument with `isinstance`, which the study-dispatch lint
  forbids; it now catches the tracer conversion error instead.
- After those two edits, re-run of the lint, balances, fingerprint, pedigree, guard and codec
  files including the slow mutants (entry-points file deselected): 678 passed in 121 s.
- `ruff check` and `ruff format --check` clean. Default `_scan_worlds` and `_init_worlds`
  StableHLO hashes unchanged (above); no pin moved.
