---
orphan: true
---

# Physical PT2: observing context and certification

PT2 ties ticks to absolute time, derives science intervals from occupancy and
certifies planned visits against keepout before acquisition. Everything is in
`src/spaceodyssey/adapters/physical_context.py`. No file under `core/` or `codec/`
changed and no golden history moved; the core stays unaware of visibility.

## What is in

- `PhysicalClock(clock, origin_ticks)`: the core `Clock` plus the tick-0 origin as an
  int64 count of ticks after J2000.0 (JD 2451545.0 TDB). `jd(tick, offset_ticks)` forms
  the day count and remainder in integers before one division, so an opposite shift of
  origin and ticks gives identical bits; `mjd` is exactly `jd - 2400000.5`; `tick(jd)`
  rounds back; `from_jd(jd0, tick_length_s)`. A tick that does not divide a day is
  refused.
- `ObservingContext`: an orbix `Observatory` (halo orbit, settling time, Sun keepout
  limits, set explicitly to 45 to 180 deg), Earth limits (0 to 180, inactive), target
  coordinates, the certification grid step and the angle-rate bound.
  - `science_interval(start, end)`: ``[start + settle, end]``, empty when the visit
    ended during settling; `science_midpoint_jd` is the record epoch.
  - `observable(target, jd, margin_deg)`: orbix `is_observable` with every limit passed
    explicitly; active limits tightened by the margin. One compiled kernel, epochs
    padded to a power of two.
  - `certify(PlannedVisit)`: samples ``[start, start + max_delay + occupancy]`` every
    `step_d` (0.25 d) with the limits tightened by `angle_rate_bound * step / 2` (3 deg
    per day, 0.375 deg), and returns a `Certificate` with a `Certification` code:
    `CERTIFIED`, `NO_VISIBILITY`, `KEEPOUT_EDGE`, `OUTSIDE_SUPPORT` or `NO_SCIENCE`.
  - `realized_science(certificate, start, end)`: a `ScienceInterval` whose
    `valid_science` is 1 only for a certified window, a nonempty science interval and
    an occupancy inside the window; otherwise science is 0 and the occupied time is
    still reported.
- `certify_plan`, `require_certified` (raises `UncertifiedPlan` with one finding per
  failed visit) and `context_from_profile(profile)`.

## Anchors (tests/adapters/test_physical_context.py, 11 unit tests)

| Anchor | Reference | Mutant killed |
|---|---|---|
| One million random ticks round-trip exactly; opposite relabelling is bit-identical; tick 0 at JD 2462502.5 is MJD 62502 | integers | origin summed in floating point |
| Science interval is the last exposure of the occupancy; epoch at start + settle + exposure / 2 within half a tick; partial science is end - start - settle | integers | science interval without settling |
| Keepout days per year at 45 deg: 70.75 for 47 UMa against 69.73 from arccos(cos 45 / cos beta), 90.5 for the twin against 91.3 | spherical trigonometry, astropy ecliptic latitude | Sun keepout left at orbix's 0 deg default |
| Inside certified; across the edge `KEEPOUT_EDGE`; start inside but slack across the edge `KEEPOUT_EDGE`; in keepout `NO_VISIBILITY`; zero exposure `NO_SCIENCE`; failing support `OUTSIDE_SUPPORT` | located keepout edge | certifying the planned start only |
| An occupancy ending an hour before the edge is refused, two days before is certified; the rate bound exceeds the fastest Sun and Earth angle change over two years (0.86 and 2.27 deg per day) | orbix angles | no keepout margin |
| An uncertified plan is refused at load, naming the visit and `NO_VISIBILITY` (B5 on the context alone) | | |
| A delay beyond the slack across the edge gives `valid_science = 0` with the occupancy charged; a cancellation in settling gives no science | | runtime check ignoring the window |
| Malformed visits are refused by name; an empty epoch list gives an empty answer | | |

All six named mutants were applied in memory and each failed its test.

## Survey (tests/adapters/test_physical_surveys.py, slow)

Two-year calendar from 2030-01-01: 47 UMa observable 80.7 percent of days, the
ecliptic twin 75.2 percent. Over one 1078-day orbit of 47 UMa b, 80.3 percent of days
are outside keepout, 53.2 percent have the nominal planet inside the working angles,
and 42.9 percent have both.

## Cold budget

Each test alone, `SPACEODYSSEY_JAX_CACHE=off`, 4 threads: process wall 1.2 to 4.5 s.
Slowest three: conservative edge (4.5 s), certification outcomes (3.6 s), realized
science (2.9 s). The file takes 5.2 s.

## Where the spec was silent

- The certificate's window is the union of every admissible occupancy; the runtime
  check requires the executed occupancy (not only the science interval) inside it.
- The science interval is anchored at `start + settle`; with cancellation its end is
  the record's end tick, which the core sets to the cancellation tick, so the realized
  duration is `end - start - settle`.
- The sampled certification is made sound by a declared angle-rate bound (3 deg per
  day) rather than by root finding; the test checks the bound against orbix over two
  years for both targets.
- The response support is an injected predicate; `SeparationSupport` builds it from
  the nominal scene and explicit working angles. The table-driven support of the
  response arrives with the radiometry task.
- hwoutils has no MJD origin; the module states the IAU value once.
- orbix's `Observatory` carries no Earth keepout limits, so the context holds them.
