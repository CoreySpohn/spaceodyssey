---
orphan: true
---

# Slice 3b U4: configuration timeline

An asset can now declare the ticks at which its configuration switches. Every admission is
stamped with the asset's configuration at the admission tick, and the stamp follows the
occupancy into its record, whether it completes, is cancelled live or is refused when a
commitment comes to start. A commitment keeps the configuration it was admitted under even
when it starts after a switch. A version-2 channel model reads the stamp as
`context.config_id`. A study without a timeline keeps `config_id = 0`, carries none of the
new leaves and lowers the same program.

## What changed

- `core/capacities.py`: `max_config_switches` (static, optional, default 0) is the padded
  width of each asset's timeline; property `configs`. It is the static switch for the
  feature, so the carry shape and the program depend only on it (a declared tick leaf alone
  does not turn the feature on; pre-flight refuses that).
- `core/ledger.py`: `AssetTable.config_switch_ticks` (optional, `I64[A, S]`, sorted per
  asset and padded on the right with `CONFIG_PAD`), a traced leaf like every other array of
  the table; `Ledger.cm_config` (`I64[K]`, present with `max_config_switches`, size zero
  while commitments are off).
- `core/events.py`: `EventTable.config` and an `enqueue` argument of the same name; the
  request and refresh rows share the table type and gain the column.
- `core/transaction.py`: at admission the configuration is the number of the asset's switch
  ticks at or before the decision tick. It goes on the Done row of an immediate admission,
  and on the start row and `cm_config` of a commitment.
- `core/handlers.py`: Done and `_cancel_one` write it to `Record.config_id` in place of the
  constant 0 (a cancelled commitment reads it from its start row); a commitment's start
  hands `cm_config` to its Done row. Bundle members share the bundle's admission tick, so
  they share a configuration.
- `core/kinds.py`: `RealizeContext.config_id`, from the record. The fork battery builds the
  context from the log row, which already holds `config_id`, so it needed no change.
- `core/executors/base.py`: `config_on` (true when the event table has the column, that is
  when `max_config_switches` is positive) gates `carry.ev.config`, `carry.request.config`,
  `carry.refresh.config` and `carry.ledger.cm_config`.
- `core/battery/metamorphic.py`: the extra asset of `zero_time_asset` gets a padded row.
- Pre-flight: finding code `S001` for a timeline set while `max_config_switches` is 0, a
  missing or wrongly shaped timeline while it is positive, a tick outside 0 to the horizon,
  and a row that is not ascending with its padding last.
- `studies/toy.py`: `survey_1` accepts `config_switch_ticks`.

## Tests

`tests/core/test_config_timeline.py` (6 tests, one compiled structure with commitments and a
two-slot timeline on; scenarios are script data and the timeline leaf):

| Test | Tier |
| --- | --- |
| records admitted before the switch carry 0, after it 1 (a live cancel after the switch keeps 0; a cancel of an occupancy admitted after it keeps 1; an asset without switches stays 0); the stamp is the first payload column a version-2 model writes | medium (shared host run) |
| a commitment admitted at 0 and started at 150 completes with 0; the same commitment refused at its start by a drain is a partial record with 0 | medium (shared host run) |
| host and compiled histories equal, compiled configs `[0, 1, 1, 1, 1]` | medium (shared compiled run) |
| a new timeline on the compiled program adds no scan trace and changes the stamps | medium (shared compiled run) |
| fork battery passes on a run whose records straddle the switch | medium |
| the timeline round trips through the codec with its dtype, and a study without one writes no field | unit |

Also: `S001` seeded cases (four) and one clean case in `preflight_cases.py` with their
locators, `max_config_switches` in the optional-capacity inventories of
`test_optional_fields.py` and `test_slice3_guard.py` (the non-default round trip carries
it).

Negative controls (run at merge time, not standing tests): a constant 0 in `h_done` fails
three tests; a constant 0 in `_cancel_one` fails the first; stamping with the start tick in
place of the admission tick fails the commitment test; reading the commitment's
configuration at its start (a recomputation from the start tick) fails the commitment test;
indexing the timeline by lane 0's asset in place of the lane's asset fails the first.

## Receipts

- Default program, measured against the commit before this change (not a standing test):
  the lowered jaxpr of one `step` is identical for the default capacities (`b61115d92ab22009`,
  276720 characters) and for a study with programs, awards, refunds, commitments and bundles
  on (`6d841acd46ec02b9`, 645200 characters).
- Program budget for `tests/core/test_config_timeline.py`: 126 programs loaded from the
  cache (warm), budget 155. Wall: about 7 s warm, about 10 s cold for the file.
- Golden check: `tests/core/test_golden_histories.py -m ""` passes unchanged (124 passed,
  200 s wall); no golden or pin file was touched.
- Quick tier: tests/core and tests/codec, `-n 2 --dist loadfile -m "not slow"`: 1729 passed in
  232 s (peak footprint 3.8 GB). `test_entry_points_installed.py` reports 92 to 94 programs
  against its budget of 91 from run to run (cache state); it is unrelated to the timeline.
- Existing tests touched: the optional-capacity lists and the seeded-code table named above.
