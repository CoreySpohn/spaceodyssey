---
orphan: true
---

# Slice 3a S7: completeness audit of pre-flight, codec and the re-baselines

S7 adds tests only, plus one docstring correction. It changes no source behavior, no golden
and no pin. The audit compared the finding registry before the slice (21 in-memory codes now,
against ten before) and every field the slice added, and asked of each: does an in-memory
pre-flight test trigger it, does a JSON pre-flight test locate it at its pointer, does the
codec round-trip it exactly, and is it absent when its feature is off.

## What was missing, and the test that now covers it

- **X002 on refund-only channel columns.** A nonzero surcharge and a nonzero refund deadline
  with refunds off had no seeded case in either layer. Both are now seeded in memory and in
  JSON, each with a clean control that keeps the value and turns refunds on.
- **X002 lifts.** Unequal charge, non-occupancy pricing, a declared failure kind and a balance
  quantum each had the refusal tested but not the lift. Each now has a control that changes
  only the feature flag and requires a clean pre-flight.
- **W001, P001, W002 branches.** An out-of-range recipient, overlapping windows (with
  adjacency as the clean control), negative start, reversed bounds, an end beyond the last
  tick, a negative program and an unregistered failure kind, in memory and in JSON, each
  located at the offending award, program or channel entry.
- **D006 and D007.** The log-budget branch of D006 in JSON, and the two remaining offset
  predicates of D007 in both layers.
- **Malformed feature metadata in JSON.** Ill-typed awards, refunds, commitment and balance
  capacities, quantum and key version each report a load finding whose pointer resolves.
- **Codec round trips.** A refunds-on study whose charge differs from cost; a commitments
  recipe with freeze ticks, a disruptive resource and a policy that declares disruptive
  effects; a policy declaring several programs and award windows; `Record.program`,
  `Record.award_gen` and the award columns of an effect at nonzero values; every
  `ChargePricing` member; the explicit default of `refunds` and the null cases of the
  optional fields, each refused at its pointer.
- **Run documents.** The allocator recipe (process, policy, committee binding) and its
  record registry round-trip with kind order, columns, widths and audiences intact, and one
  run recipe per feature (programs, refunds, awards, commitments, balances, key version 2)
  is dumped, loaded and re-dumped to identical JSON. The loaded study equals the original on
  every leaf and static field, and the original and the loaded recipe execute to identical
  host histories (the key-version recipe draws on asset 1, so its history depends on the key
  version).
- **Conditional leaves.** Each feature alone (refunds, awards, commitments, balances) adds
  exactly its own leaves and no other feature's, and with balances off the snapshot holds no
  balance, quantum, balance-row or query-refusal field, in the policy view or the observer
  view.
- **Re-baselines named by the slice spec.** The three tests that asserted `Allocate` answers
  `UNSUPPORTED` keep that assertion for a study without awards, as the control, and now also
  admit a valid `Allocate` with awards on and check the credited row; the agenda-level test
  asserts level 1 stays forbidden and level 3 admits it.

## Corrections

- `spaceodyssey.codec.preflight` stated seventeen in-memory and twenty total codes. The
  registry holds twenty-one in-memory codes, so the document checks report twenty-four, with
  `L001` separate.
- The slice spec lists the window and remainder fields among the X002 flags lifted in S5. As
  the S5 record says, no X002 window field existed: X002 flags channel refund declarations,
  which S3a gated, and an award study was refused by X001, which S5 lifted. There was nothing
  further to lift.

## Cost

The ten touched test files (`tests/codec/test_optional_fields.py`,
`test_preflight_json.py`, `test_run_document.py`, `test_run_document_runs.py` and
`tests/core/test_admission_rules.py`, `test_agenda_levels.py`, `test_core.py`,
`test_preflight.py`, `test_slice3_guard.py`, `test_snapshot_balances.py`) run cold with
caches off in 128 s at two workers: 721 tests, 2271 programs, peak 4.2 GB, no budget
violation. The command, from the repository root:

```
SPACEODYSSEY_JAX_CACHE=off compute-run --threads 4 --mem 14336 -- uv run python -m pytest \
  <the ten files> -n 2 --dist loadfile -p no:cacheprovider --budget-json s7.json
```

The run-document file gained the executed-history comparison: six features, each a small
plan study run twice on the host (3 to 5 s cold each), so its program budget rose from 575 to
702 (601 programs measured, `ceil(601 * 1.15) + 10`). The three re-baselines each compile one
admission step of a small awards study, inside the unit budget.
