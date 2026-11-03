# fix-parity

## Item 1: complete canonical comparisons

Canonical parity now has an explicit inventory of every output leaf in the
record log, trace, final carry, both beliefs, ledger, queued events and retained
request. The whole-field exclusion map is empty. The inventory regression
recursively enumerates the output dataclasses, requires exact coverage without
overlap, and requires `Record` and `RecordLog` to have identical field names.
A newly added field cannot silently escape comparison.

The general property is that changing any compared output field in the compiled
result alone must reject parity with the host result. The regression changes one
element at a time, preserving dtype and shape, over all 135 current output leaves.
It uses a real truncated run with valid records and pending events, rather than
an empty fixture. Separate mutations check history counts, lane tuples, status
and overflow metadata.

## History changes

Existing hand event and row tuples, draws and simulation behavior are unchanged.
Existing short history keys remain. Dotted output paths extend the integer and
floating maps to include every inventoried field. Newly covered data includes:

- Record source, configuration id, lane and channel index, and the validity mask.
- Trace channel, audience, log slot, policy random draw, cumulative log count and
  compressed handled mask.
- Final policy belief auxiliary state, policy state, scheduling and decision
  coordinates, wake/drain state, duplicate-fold flag and lane occupancy state.
- All valid pending event columns and their validity mask, and all fields of the
  retained request row, including its snapshot and decision identity.
- The history object's own counts, lane tuple, status and overflow metadata.

Both beliefs' means, sample counts and auxiliary state compare; the existing
observer mean and auxiliary paths still compare. Every compared integer or
boolean is exact. Floating values use the existing go and kill tolerances, and
shape and dtype checks are unchanged.

## Storage rules and limits

No whole output field is excluded. Invalid queue/log row contents are unused
storage; their full validity masks and every valid row compare. Unhandled trace
rows are executor padding; every field compares on handled rows. A positive
regression verifies that changes confined to these storage regions preserve
parity. The scalar retained request compares in full, even after being answered.

The check proves coverage of final output state and fields already retained in
the trace. It does not create an intermediate path for fields absent from the
trace, or establish correctness when both executors share the same bug.

## Validation

- `tests/core/test_canonical_fields.py`: 142 passed. Against the previous `base.py`: 72 failed (every
  field the comparator did not read, plus the inventory test).
- `tests/core/test_parity_corpus.py` 29 passed and `tests/core/test_parity_values.py` 15 passed with
  the extended history: no canonical field differs between the executors on the corpus.
- The reviewer's K4 mutants (record source, configuration id, lane, channel index, policy belief
  auxiliary state, trace channel, audience, log slot, policy draw) now each fail the gate.

Adding a new carry field is one entry in `CANONICAL_FIELDS` (or a reasoned entry in
`CANONICAL_EXCLUSIONS`); the inventory test fails until one of the two is made.
