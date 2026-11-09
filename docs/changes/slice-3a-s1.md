---
orphan: true
---

# Slice 3a S1: capacities, conditional leaves, optional codec fields and key versions

S1 adds the machinery every later 3a task builds on, and changes no history. It is tier 2
(core and codec loading). The pins from S0 are the K1 evidence: no history digest,
fingerprint or document hash moves.

## What is in

- **Capacities.** `Capacities` gains `awards`, `refunds` (bools, default off),
  `n_commitments` (nonnegative int, default 0) and `snapshot_balances` (bool, default off),
  each validated at construction. They carry the plain metadata key `optional`
  (`capacities.OPTIONAL`), which the codec reads; core does not import the codec.
- **Conditional leaves.** A leaf exists in the canonical history and in the snapshot
  fingerprint only where its feature does. `executors/base.py` holds two tables,
  `CONDITIONAL_GROUPS` and `CONDITIONAL_FIELDS`, each entry a path with the predicate that
  says whether the carry holds the feature (today `refresh_on`, the refresh status leaf has
  size). `snapshot.conditional_view_leaves(view)` lists the view leaves of the same kind and
  `snapshot_fingerprint` hashes only those with size. `REFRESH_FIELDS` and `REFRESH_GROUPS`
  stay as views of the tables. Behaviour today is identical: only the refresh fields and
  the `product` column are listed.
- **Codec optional-field rule (D12).** The encoder omits a field marked optional when it is
  at its default; the loader fills a missing optional field with that default and refuses a
  present one that is at its default ("omit it"), so one study has one document. Defaults
  must be plain (bool, int, float, str, or a tuple of them); a factory or an array default is
  a registration error. Comparison is by exact type and content, element by element for a tuple (`True` is not
  `1`, `-0.0` is not `0.0`, `nan` equals `nan`, also inside a tuple). Migrated documents are unchanged.
- **`key_version` (D9).** A static optional `Study` field, 1 or 2, validated at
  construction. Version 1 keeps the asset coordinate at the Python integer 0 for every
  channel, so keys and compiled programs are as before. Version 2 looks the coordinate up,
  channel to lane to asset (`keys.asset_coordinate`), a gather on arrays fixed at
  construction. Asset 0 folds nothing, so draws on asset 0 keep their keys under version 2.

## Moves

None. `tests/core/golden_histories.json` is unchanged and no pin file is edited.

| Check | Result |
|---|---|
| Golden integer digests, all 116 histories | Equal to the pins (every one of the 18 reported mismatches on this machine is the float digest only; the integer digest in each equals the pin) |
| Golden float digests | 18 differ on this Linux machine, as on `main` and as recorded in S0; the spaceodyssey Mac run is the check |
| `tests/codec/test_run_document_runs.py` (hash pin, 58 documents) | passes |
| Fingerprint pins, gate files, canonical-fields, document tests | pass |
| Fault fixture pin | the two reported failures are the float digest on this machine; the integer digests equal the pin |

## Sibling sites checked

- `draw_key(` callers: `handlers._realize_rec` and `battery/fork._keyed_draw` take the new
  coordinate; `keys.fold_key` is an inference key with no asset and stays at 0.
- Conditional gates: `base._canonical_leaves` is the only consumer of the tables;
  `tests/core/test_canonical_fields.py` reads `REFRESH_FIELDS` and `REFRESH_GROUPS`, kept.
- Fingerprint: `snapshot.snapshot_fingerprint` is the only hasher of view leaves.
- Future conditional record columns (S2 to S5) must touch: `records.py` (`empty_log`,
  `write_log`, `LOG_INT_FIELDS`, `RECORD_INT_FIELDS`), `snapshot.py` (view masking and
  `conditional_view_leaves`), `executors/replay.py`, `executors/base.py` (the tables) and
  `tests/core/gate_helpers.py`.
- The K1a dispatch lint rejects `type(...)` and `getattr` on study or caps objects in
  core, so the new validations use `isinstance` and explicit attributes.

## Tests

| File | Tests | Tier | Cold time (alone, cache off, 4 threads) |
|---|---|---|---|
| `tests/codec/test_optional_fields.py` | 15 | unit | 2.5 s, slowest 0.7 s |
| `tests/core/test_key_version.py` | 12 | unit | 1.6 s, slowest 0.45 s |
| `tests/core/test_slice3_guard.py` | 18 (6 medium) | unit and medium | 33 s, medium tests about 10 s each |
| `tests/core/test_metamorphic.py` | 2 new (10 in the file) | slow (existing file) | the facility-insertion guard below; each run-level test one cold compile pair |

Three slowest new tests: the asset-zero key test, the zero-lookup control and the
canonical-history gate test, each one cold executor compile. The medium tier of the guard
file takes about 35 s in total; the only new slow-tier tests are the two run-level ones in the existing slow file `test_metamorphic.py`. Added 48, retired 0, net +48 (the idle-facility coordinate test is in the guard file, unit tier).

Each check has a control: an encoder that keeps defaults, a loader that requires every field,
an open conditional gate, a nonempty extra view leaf, a zeroed `refund_of`, a changed layout
for the lookup, a lookup pinned to zero and a lookup that keys by `asset * n_assets`.

## Review round 1

- Tuple defaults are compared element by element with the scalar rules, so `(True,)` against
  `(1,)`, `(-0.0,)` against `(0.0,)` and a `nan` tuple are decided as scalars are. The test
  failed first on `(True,)`.
- The facility-insertion guard extends the existing `zero_time_asset` test: asset-1 draws
  under key version 2 must be unchanged after an idle asset is added, a `asset * n_assets`
  lookup must fail the relation, and inserting the asset must keep every channel's
  coordinate.
- `.gitignore` now ignores `/src/spaceodyssey/_version.py` only, so `tests/core/test_key_version.py`
  is no longer hidden.

## Spec silences decided here

1. `refund_of` stays an unconditional column with sentinel -1 (a populated internal id is
   hidden in views only). Rejected: masking it to zero or omitting it, which moves a legacy
   fingerprint; the guard pins this.
2. Optional marker is a plain field-metadata key defined in core. Rejected: importing a
   codec marker, since core may not import the codec.
3. Writing a present optional field at its default is refused. Rejected: accepting it, which
   would give two documents (two hashes) for one study.
4. Boolean values are not accepted for `key_version`, and floats such as `2.0` are refused.
5. Version 1 returns a Python integer, not a traced zero, so the compiled program is
   unchanged. Rejected: always gathering and relying on the value being zero.
6. The guard's run-level tests are in the medium tier, with the unit-tier checks in the same
   file.

## K4 attribution

Core edits (`capacities.py`, `keys.py`, `study.py`, `handlers.py`, `battery/fork.py`,
`executors/base.py`, `snapshot.py`) belong to S1. Event order, handlers and the K3 baseline
are untouched.

## Core paths changed (K4)

The exact core paths that S1 changed after the S0 close-out, which the Slice 3a gate matches against this record:

- `src/spaceodyssey/core/battery/fork.py`
- `src/spaceodyssey/core/capacities.py`
- `src/spaceodyssey/core/executors/base.py`
- `src/spaceodyssey/core/handlers.py`
- `src/spaceodyssey/core/keys.py`
- `src/spaceodyssey/core/preflight/checks.py`
- `src/spaceodyssey/core/snapshot.py`
- `src/spaceodyssey/core/study.py`

Core tests that cover this change: `tests/core/test_canonical_fields.py`, `tests/core/test_key_version.py`, `tests/core/test_metamorphic.py`, `tests/core/test_slice3_guard.py`.
