# Fewer compiled programs

A study compiles a new program whenever a number it holds as a Python value differs, or
the step count differs. Most of a study's wall time is that compile, about 4 s on a
laptop and about 20 s on a CI runner. This change removes three causes. No history,
fingerprint, golden pin, FAULT fixture or document hash moves, and the default compiled
scan is byte for byte the program it was.

## What changed

### Numbers held as arrays

These fields were Python floats, ints or tuples, so each value was a new static key.
They are now int64 or float64 arrays, so studies that differ only in these values share
one program:

- `ThresholdStopPolicy.stop_snr`, `StochasticPolicy.n_obs` and `wait_ticks`;
- `ScriptedRecords.ticks`, `values`, `ids`, `revisions`, `statuses` and `kinds`: the
  number of fires is the array length and stays part of the program;
- `QueryPolicy.query_at`, `subject` and `service`;
- `CatalogObserver.catalog_kind`.

The dtype is the one the arithmetic used before, so every history is bit for bit the
same. Each array is a `PlainArray` (`spaceodyssey.plain`), a NumPy array that also keeps
the value it was built from. A document writes the field as that value (`3`, not `3.0`;
an enum member, not its integer), so every earlier document re-dumps byte for byte and
keeps its content hash. The kept value plays no part in a compiled program, which sees
only dtype and shape. An array made any other way, such as one put in with `tree_at`,
is written as the plain numbers it holds.

The loader makes the array again through the field's converter, also for an omitted
optional field. It refuses a stored form that is not numbers or tuples of numbers
(including a nested array tag), a fraction for an integer field, a number out of range,
and a shape the converter would change, each with a pointer to the field.

Left as they are:

- `ToyRunningMean.measurement_kind` and `auxiliary_kind`. The default study holds them as
  literals in its program, and turning them into inputs changes that program.
- `ScriptedRecords.audience` and the prior observer's kinds and rule. A document writes
  the audience as an enum member, and the rule selects a Python branch.
- Every field that sets a shape or selects a Python branch.

### Bucketed run

`CompiledExecutor.run(..., bucket=N)` rounds `n_steps` up to a multiple of `N` for the
compiled scan and passes the step count as an array. Steps past `n_steps` take no step,
and the trace is cut to `n_steps` rows. Every step count in one bucket then shares one
program. The carry and trace equal the default path's bit for bit, including a step
count that is not a multiple of `N`. The default (`bucket=None`) is the unchanged scan.

A `bucket` must be a positive int. Rows past `n_steps` still evaluate the step before it
is masked, so a padded row costs run time, but the returned carry and trace show no
extra consumption.

The initial carry is built inside the program, as the default scan does. A program that
takes the carry as an input, such as the chunk program of the host-service path, compiles
the same arithmetic to different last bits (a 1 ulp difference in the running mean), so
the bucketed run does not continue from a carry.

### Names out of the compile key

`Capacities.kind_names` is read only while a study is built or checked, never under
trace. It is now excluded from the equality and hash of `Capacities`, so studies that
differ only in the registered kind names share a program. The check that a registry
lowers to the study's capacities compares the names explicitly, so it refuses a mismatch
as before. `PreMissionTable.upstream_hash` is unchanged: it is a static field the codec
and the lineage hash read by name, and no test varies it.

## Measured

Mac, single thread, compile plus run of two worlds, seconds:

| Variation | Before | After |
| --- | --- | --- |
| `stop_snr` 3.0 then 4.0 | 5.8 | 0.006 |
| `StochasticPolicy` `n_obs` and `wait_ticks` changed | 5.7 | 0.007 |
| scripted values and ticks changed | 4.7 | 0.004 |
| kind names renamed | 5.2 | 0.01 |
| 40 steps then 50, `bucket=32` | 3.9 | 0.10 |
| 40 steps then 70, `bucket=32` | 3.9 | 4.5 (next bucket) |

The default scan StableHLO (survey 1, 2 worlds, 40 steps) has sha256
`a67917728c4878feaa35f3bccc622be9ff6850ec65b5adf22867e668c9d1d791` before and after.

## Tests

`tests/core/test_compile_sharing.py` compares the bucketed run with the default scan leaf
by leaf on two hand studies and a scripted study at 12, 40 and 45 steps, checks that
varied values, scripts, step counts within a bucket and kind names add no program, that
the lowering check still refuses mismatched kind names, and the plain document form.
