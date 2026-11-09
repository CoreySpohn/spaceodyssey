---
orphan: true
---

# Slice 3b U6: world-drawn trajectories

A world source may return a trajectory table inside its truth, drawn under its own purpose
key. A version-2 channel model samples it at `context.start_tick` inside `realize`; nothing
else reads it.

## What changed

- `core/enums.py`: `Purpose.TRAJECTORY = 5`, appended; codes 0 to 4 are unchanged, so no
  existing key moves.
- `core/trajectory.py`: `trajectory_key(world_key)` (the stream a world source draws tables
  from), `TrajectoryTable` (`ticks`, `values`, `at(tick)` interpolating linearly and holding
  the end values) and `draw_walk(key, ticks, scale)`. All three are exported from `core`.
- `core/battery/fork.py`: `RECORD_PURPOSES` gains the purpose, so `key_report` lists it.
- Truth stays study-defined; the core truth type, the codec and the study declaration are
  unchanged. No new declaration field, carry leaf or trace leaf, so no `OPTIONAL` or
  `CONDITIONAL_FIELDS` entry is needed.
- Handlers and the fork battery already pass the truth to `realize_row`, so both executors
  and the battery sample the same table from the same record.

## Tests

| Test | Tier |
| --- | --- |
| table interpolates between ticks and holds its ends | unit |
| table moves with its own purpose key and with no other draw | unit |
| existing purpose codes unchanged | unit |
| sample equals the table at each record's start tick; a delay moves later rows; host and compiled histories equal | medium |
| fork battery of the delayed table study | medium |
| table study passes pre-flight and round-trips with a stable content hash | unit |

## Receipts

- Program budget for `tests/core/test_world_tables.py`: 158 programs measured cold, budget 192.
  File time about 13 s cold, 6 s warm.
- Studies that use no table build the same program: no handler, executor or policy-path
  source changed. Golden check: `tests/core/test_golden_histories.py -m ""` passes
  unchanged (124 passed, 168 s); no golden or pin file was touched.
