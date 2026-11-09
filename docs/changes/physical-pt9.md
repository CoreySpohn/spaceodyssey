---
orphan: true
---

# Physical track PT9: the summary channel

PT9 turns an executed occupancy into a summary record. `JaxedithSummaryChannel` is a
version-2 channel model (the realize context of slice 3b U1): it reads the record's own
start and end ticks, derives the science interval, and writes one full summary row with
PT4's `sample_report`. No file under `src/spaceodyssey/core/` or `codec/` changed and
`tests/core/golden_histories.json` is unchanged. The code is in
`src/spaceodyssey/adapters/jaxedith.py`; the tests are
`tests/adapters/test_summary_channel.py`.

## The contract

- One channel per planned visit: channel `i` of the study is `visits[i]`, and
  `from_plan(visits)` certifies every visit (`require_certified`, with an optional
  response-support predicate) and refuses an uncertified plan at construction.
- At realization the occupancy is `[context.start_tick, context.end_tick]`, so a delayed
  admission and a cancellation move it. The science interval is `[start + settle, end]`
  (PT2), rechecked against the visit's certificate: `valid_science` is 1 only for a
  certified window, a nonempty science interval and an occupancy inside the window.
- The row is `sample_report(key, law, expected, limit, ...)` with `expected` the true
  planet's budget over that interval, `limit` the planet-free limit of the nominal
  planet over the same interval, `time_d` the midpoint of the science interval (the
  epoch, from `PhysicalClock.jd`), `occupied_s` the executed occupancy, and the
  calibration id of the build. The key is the core's measurement key, so a partial
  occupancy shares its draw with its full twin.
- Nothing scales counts by the occupancy fraction (the `fraction` argument is unused):
  the realized science duration is `max(0, end - start - settle)` ticks, and a
  cancellation inside settling is a no-science row (`valid_science = 0`, zero science
  time, zero limit, zero counts), with the occupancy still reported.
- A no-science row draws from a finite budget at one detector frame (the row's flag and
  zero science time clear every science column), so it is never a NaN row.
- The model holds plain declarations only (profile label, target, detector spec
  values, node count, exozodi level, law, one certified window per channel, the
  expected calibration id), so a study document stores it. The report, the observing
  context and the nominal planet are built on first use, once per declaration set
  (`functools.lru_cache`), and a build whose calibration id differs from the declared
  one raises instead of writing rows under another calibration.
- The build runs on a fresh thread. Tracing state belongs to the thread, so the first
  call can come from inside a trace (a reloaded study builds at its first realization)
  and still be the ordinary eager build: the response tables of a build made inside
  `jax.ensure_compile_time_eval` differ from the eager build in the last bit (an
  element-wise difference up to 7e-13 in yippy's tabulated curves), which moved the
  row by one unit in the last place against an offline report.
- The truth is a `PhysicalTruth`; `subject` picks the entry. The true scene is the
  nominal one (the truth side differs from the model in the planet only), as in PT4's
  survey.
- Registered as `channel_model` `jaxedith.summary_channel@2` in `pyproject.toml`.

## Tests (tests/adapters/test_summary_channel.py)

| Test | Tier | What it catches |
|---|---|---|
| declarations refused before anything is built | unit | a malformed window, calibration, profile, detector, law, target or node count; empty and mixed-target plans |
| the build runs under the caller's JAX settings | unit | a build thread that drops `jax.enable_x64`, the default device or matmul precision (a float32 build cached for every later caller); one setting served another's build; a repeat that rebuilds |
| an edited profile is not served a stale build | unit | a build cache that ignores the profile text |
| realize equals the offline report at the executed interval | medium | any bit of difference from `sample_report` for a full exposure, a delayed one (epoch moves by the delay), an admission across the keepout edge and a cancellation inside settling (no-science rows, bit-anchored too), a cancellation 400 s in; a calibration id other than the PT4 survey's; an epoch away from the science midpoint (KP9) |
| the core moves, charges and refuses around the edge | medium | a delayed admission that does not move the record or reports science; a charge to the wrong channel or more than once; a core row that is not the offline row at the record's ticks and key; an uncertified plan or stale calibration accepted |
| both executors and a new process | slow | a compiled history or row that differs from the host run; a document that drops a declaration; a reloaded channel that builds another calibration; a new process that writes a different row |

The channel's own `realize` runs under one `eqx.filter_jit` with array context ticks and
keys, so one compile covers every scenario; the key is the core's measurement key of the
one host run, which is a module fixture shared by the medium and slow tests. The offline
reference uses the channel's report build (one build per process) but makes its scene and
interval itself and draws with `sample_report`. Both functions close over the same build,
because a closed-over constant and a jit argument fuse differently in the last bit, so
the bit-for-bit anchor compares like with like.

## Optional extra

The entry point is declared with extras syntax (`... [physical]`). In an install without
the extra, `kinds_from_entry_points` skips that entry (a declared requirement of the
extra is not installed, or the load fails with `ModuleNotFoundError` for a module outside
spaceodyssey) and `installed_missing()` names it; `installed_resolver()` returns an
`InstalledResolver` that refuses the kind with an "install `spaceodyssey[physical]`"
error only when a document asks for it. An entry without extras that fails to import
still raises `RegistrationError`. `tests/codec/test_entry_points_installed.py` is back to
the core assertions plus the plain-install path and the physical kind's round trip when
the extra is present.

## Build settings and cache key

The asset build runs on a fresh thread so that it is the ordinary eager build even when
the first call is inside a trace. The thread starts from the process defaults, so the
caller's thread-local settings (`jax.enable_x64`, default device, matmul precision) are
read on the calling thread and entered again on the worker, and they join a digest of
the profile and pinned response in the cache key.

## Measured cost and mutants

The machine was heavily loaded for the final measurements (load average 25 to 47, both
compute slots queued behind other jobs), so the times below are upper bounds: the
medium tier measured 40 s of shared setup plus 13 s and 4 s of calls, the slow test 82 s,
the full file 142 s wall. An earlier run under a lighter load (load average 8) measured
the first design at 33 s and 43 s. The file compiles 353 programs, so its entry in
`module_compile_budget_by_file` is ceil(353 x 1.15) + 10 = 416.

Mutants of the first design (each applied alone, each failed a quick test): the context
ticks ignored, no settling subtracted, no certificate check, a fixed key, and counts
scaled by the occupancy fraction. The realize-level test fails each of them through the
same bit anchor.

Tests: 6 in the file (3 unit, 2 medium, 1 slow), plus 2 in the entry-point file.

## Where the spec was silent or the code differs

- The spec's true scene is not specified; the channel declares it equal to the
  nominal scene, as PT4's survey does.
- The spec's `fraction` argument is unused: the realized duration comes from the
  context ticks, so a partial occupancy needs no scaling.
- `from_plan` certifies keepout only unless a response-support predicate is passed.
- The lane's transition adds 10 ticks, so a record's start is the planned start plus
  that offset; the tests assert the settle-corrected science time, not the plan.
- `SceneSpec` and `PriorSpec` cannot be registered through entry points, so a run
  document of a physical study needs an explicit kinds mapping at load.
- The metadata in a shared environment stays stale until a reinstall, so the
  installed-metadata comparison in `test_entry_points_installed.py` fails there until
  the package is reinstalled.
- The bit-for-bit anchor needs a jit-compiled offline reference: eager and fused
  arithmetic differ by one unit in the last place.
