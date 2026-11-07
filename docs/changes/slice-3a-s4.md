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
- Observer audience: `balance_view` returns `(None, None)` for any audience but the policy;
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
3. **Balance rows and ids.** The view carries the whole ledger with no extra id column: program
   `p` is row `n_assets + p`. Rejected: a separate id column (one more leaf and a second
   fingerprint question for no information the row order does not hold).
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
7. **Rounding.** `jnp.round` (half to even) over the quantum, int64. Values past int64 are not
   guarded; a balance that large is outside the ledger's float range of use.
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
