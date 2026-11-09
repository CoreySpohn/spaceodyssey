---
orphan: true
---

# Slice 2 T8b: planit plan-table adapter

`spaceodyssey.adapters.planit` lowers a planit-py `PlanTable` into a
`PlanTablePolicy`. The adapter is outside the transaction core. No core file
changes, and import lint negative controls now explicitly refuse adapters,
codec and plugins.

`plan_table_to_ticks(table, clock)` computes float64 ticks with NumPy on the
host, then rounds to int64 once. A start `x` is accepted exactly when
`abs(x - round(x)) <= min(1e-3, max(1e-9, 4 * numpy.spacing(x)))`.
A fractional-tick error reports its row, day and offset. Negative, non-finite
and overflowing starts are refused; padding is ignored. The policy loader
checks the study horizon, subject and channel identifiers, asset membership
and effect capacity. The runtime dependency floor is `planit-py>=0.2.0`;
`hwoutils>=1.10.0` supplies the day-to-second conversion constant.

The policy keeps subject, channel mask and index, start tick, validity and
group in fixed-shape JAX arrays, with its own version default of 1. Its cursor
is `pstate[0]`. At each decision it emits the next valid row if due and advances
the cursor, even if admission refuses the row. Before a row is due or after
exhaustion it emits nothing. The study supplies decision wakes and channel
durations. The NumPy decision form applies the same rule.

Each row emits its channel set as one atomic group. Only default transition
kind 0 is supported, using the study asset transition durations. Groups spanning
multiple rows are refused because replaying one row per decision cannot honor
atomic admission across rows.

## Verification

Six new quick-tier test functions cover the two B7 sequences on both executors,
fixture utilities of 1.3100000000000003 and 2.21 within absolute 1e-12, exact
subject/start-tick acquisitions, complete record parity including the NumPy
policy form, half-tick and 7.8-tick refusal, absolute tolerance acceptance and
refusal, invalid identifiers and starts, channel column remapping, a multi-channel
row, padding and exhaustion. Fixture evidence describes a deterministic toy
campaign.

Both B7 plans share one study shape. Counting wake traces checks one new
`PlanTablePolicy` study signature per executor and no additional trace when
switching to the second plan. This counts study signatures, rather than the
individual helper kernels compiled internally by an executor.

| Measured run | Result |
|---|---|
| New file, quick tier (`-m 'not slow'`), warm cache | 6 passed; 7.44 s pytest, 10.28 s end-to-end wall; shared setup 4.42 s; slowest test call 0.86 s |
| Adapter, core lints and internal-reference guard tests | 192 passed in 11.14 s; zero test-budget violations |
| New study signatures for the two B7 plans | 1 `PlanTablePolicy` study per executor; 2 total; switching plans adds 0 traces |
| First run after reducing replay capacities, cold for this shape | 6 passed; 13.84 s pytest, 16.20 s end-to-end wall; shared setup 10.81 s; slowest test call 1.00 s |

All six test call phases are below 5 s, with zero test-budget violations. Cold
shared fixture compilation exceeds 5 s; warm shared setup is below 5 s. No
additional compiled study shape is introduced for the second plan.

`uv lock` could not resolve the new floor: the configured package index offers
planit-py only through 0.1.0. The lockfile was left unchanged. Validation uses
the supplied local planit-py 0.2.0 source in the worktree virtual environment,
with `uv run --no-env-file --no-sync` to avoid unrelated parent environment
settings and dependency resolution.
