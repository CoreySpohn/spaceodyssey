# Fewer compiled programs

A study compiles a new program whenever a number it holds as a Python value differs, or
the step count differs. Most of a study's wall time is that compile, about 4 s on a
laptop and about 20 s on a CI runner. This change removes three causes. No history,
fingerprint, golden pin, FAULT fixture or document hash moves, and the default compiled
scan is byte for byte the program it was.

## What changed

### Numbers lifted at the call

These fields were Python floats, ints or tuples, so each value was a new static key.
They stay plain Python numbers and tuples on the objects, exactly as before, so every
document, content hash and constructor is unchanged. Each is marked with
`spaceodyssey.core.lift.traced(dtype)`, and the executors lift a marked field to an array
of that dtype for the duration of a call (`run`, `advance` and `decide`). The study you
pass in is not modified. Studies that differ only in these values then share one program:

- `ThresholdStopPolicy.stop_snr` (float64), `StochasticPolicy.n_obs` and `wait_ticks`
  (int64);
- `ScriptedRecords.ticks`, `values`, `ids`, `revisions`, `statuses` and `kinds`: the
  number of fires is the tuple length and stays part of the program;
- `QueryPolicy.query_at`, `subject` and `service` (int64);
- `CatalogObserver.catalog_kind` (int64).

The dtypes are the ones the arithmetic used before. The histories are compared lifted
against unlifted, leaf by leaf (dtype, shape, weak type and raw bytes), on both
executors, for every study listed above in its default and varied forms: no leaf
differs.

A lift never changes a value. Lifting converts only what it can prove exact: a Python
int, bool or float, or a flat tuple or list of them, each number checked on its own.
Everything else passes through unchanged, as it did before lifting existed: arrays,
NumPy scalars, Fractions, Decimals, nested sequences, a float in an integer field (`1.5`)
and an integer outside int64. NumPy arrays in the `ScriptedRecords` tuple fields are not
supported: the fields are typed as tuples, and the previous commit accepted such arrays only
when its executor had not yet run a tuple script.
Every compiled entry point that takes a policy, process or observer
(the executors, the replay and conformance replays, the masked refit and the service
commit) lifts through `lifting_jit`, so a value variant reuses the program there too.

Left as they are, because lifting changes a program or a history, or the field selects a
Python branch or a shape:

- `ToyRunningMean.measurement_kind` and `auxiliary_kind`. The default study holds them as
  literals in its program, and turning them into inputs changes that program.
- `ScriptedRecords.audience`, the prior observer's kinds and rule, and
  `HeartbeatProcess.writes`.
- Every other field that sets a shape or selects a Python branch.

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

The default scan StableHLO (survey 1, 2 worlds, 40 steps) is the same with and without
this change: sha256 `8b8ee715bb59413b179fa7c79f086ac18556ed7ae56d0b639c617fb291fd00a0`
on the branch that carries the program-size reduction, and
`a67917728c4878feaa35f3bccc622be9ff6850ec65b5adf22867e668c9d1d791` on the one before it.

## Tests

`tests/core/test_compile_sharing.py` compares the bucketed run with the default scan leaf
by leaf, checks that varied values, scripts, step counts within a bucket and kind names
add no program, that the lowering check still refuses mismatched kind names, that a bad
`bucket` is refused, that each lifted class of input (bool, int, large exact int,
integral float, float subclass, sequences) gives the history recorded on the base commit,
and that each compiled entry point keeps one program across value variants.
