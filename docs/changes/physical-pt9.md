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
| executed record equals the offline report | medium | any bit of difference from `sample_report` at the executed interval and key, a calibration id that is not the independent report's, an epoch away from the science midpoint (KP9, with a shifted-epoch control) |
| delay, edge and cancellation | medium | a delay that does not move the epoch by the delay; science reported after an admission shifted across the keepout edge; a second charge; science after a cancellation inside settling; a mid-exposure cancellation not scored over the observed time; an uncertified plan or a stale calibration accepted |
| both executors and a new process | slow | executors that write different histories or rows, a document that drops a declaration, a reloaded channel that builds another calibration, a new process that writes a different row |

The offline reference builds its own `SummaryReport` from the profile and calls
`sample_report` itself; the epoch and interval come from `PhysicalClock`, the keepout
edge is located with the context's visibility test, and KP9 uses plain tick arithmetic.
The offline function is compiled like the channel's: eager and fused arithmetic differ
in the last bit (two columns, relative 1e-15), so the bit-for-bit anchor compares like
with like.

All scenarios share one study structure: the policy's start tick, cancellation tick
and channel are array leaves, so the five host runs share one compiled program.

## Measured cost and mutants

Cold (`SPACEODYSSEY_JAX_CACHE=off`, 4 threads) on a loaded laptop (load average 7 to 9):
unit 0.4 s; offline-report medium 16 s call plus 11 s shared setup (the independent
report build); delay, edge and cancellation medium 8 s (it reuses the setup and the
host program); both executors and a new process (slow) 43 s. The slow test holds the
only compiled-executor program of the file, which is why the executor comparison moved
out of the quick tier: with it, the first medium took 32 s, over its 30 s budget. The
slow tier for this file is therefore about 43 s on this laptop, above the 20 s laptop
share implied by 60 s of runner time; it is the one thing in this change over budget.
The file compiles 350 programs, so its entry in
`module_compile_budget_by_file` is ceil(350 x 1.15) + 10 = 413.

Mutants (each applied alone, each fails a quick test): the context ticks ignored (the
planned interval used), no settling subtracted, no certificate check, a fixed key, and
counts scaled by the occupancy fraction.

Checklist notes: the offline reference and the channel share no code beyond the
library functions under test; the truth side differs from the model in the planet
only; no controlled data is read; the new entry point has an installed-kinds test and
`tests/codec/test_entry_points_installed.py` exempts the physical kind.

Tests: 4 added, none retired, net +4.


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
- The new entry point requires the `physical` extra: `installed_kinds()` raises
  `RegistrationError` in an install without it. The metadata in a shared environment
  stays stale until a reinstall.
- The bit-for-bit anchor needs a jit-compiled offline reference: eager and fused
  arithmetic differ by one unit in the last place.
