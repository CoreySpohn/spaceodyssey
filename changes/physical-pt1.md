---
orphan: true
---

# Physical PT1: the hand-built scene

PT1 builds the first physical scene from the profile table and draws truth worlds from
photomancy's orbit prior. Everything is in `src/spaceodyssey/adapters/skyscapes.py`.
No file under `core/` or `codec/` changed and no golden history moved.

## What is in

- `SceneSpec`, plain data with units in the names: host (mass, distance, coordinates,
  luminosity, effective temperature, radius, angular diameter) and planet (period,
  eccentricity, the radial-velocity argument of periastron, periastron epoch,
  inclination, node, mass, radius, albedo), plus the band. `SceneSpec.from_profile`
  reads a target of the profile table; for 47 UMa b the mass is M sin i over sin(60
  deg). `omega_planet_deg` is the stellar value plus 180 deg.
- `build_star(spec)`: a skyscapes `Star` whose single-epoch flux table is skyscapes'
  `BlackbodyStar` spectrum (catalog temperature and radius) over the band, converted
  with hwoutils, carrying the coordinates, luminosity and angular diameter.
- `build_scene(spec)`: a skyscapes `System` with that star and one `Planet`
  (`KeplerianOrbit.from_period` with the star plus planet mass, grey
  `LambertianPhysicalModel`), solved with orbix's Markley Kepler solver.
- `geometry(system, epoch_jd, wavelength_nm)`: east and north offsets, separation,
  phase angle, star-planet distance and contrast, all from skyscapes and orbix calls.
- `PriorSpec` and `PhysicalWorldSource` (an `AbstractWorldSource`): each subject draws
  from photomancy's `build_model(sample_planet_scale=True, ...)` through NumPyro's
  seed and trace handlers, with `fold_in(key, subject)`; photomancy's periastron time is
  placed on the TDB axis at `epoch_jd`. `PhysicalTruth` holds period, eccentricity,
  orientation, periastron epoch, mass, radius and albedo per subject. The draw is
  traceable, so `make_worlds`-style `vmap` over keys works.
- `truth_scene(source, truth, subject)`, `SeparationSupport` (the nominal planet's
  separation inside the working angles, the support predicate the context's
  certification takes) and `diagnose(truth, record, source=, context=)`, which reads
  only the record's subject and ticks and returns the truth-side geometry at the
  science midpoint (`TruthDiagnosis`).

## Anchors (tests/adapters/test_skyscapes_scene.py, 14 unit tests)

Expected values come from `tests/anchors/reference_geometry.py`, which imports no
library under test (mpmath at 30 digits, astropy constants, Thiele-Innes constants
written out), or are written by hand.

| Anchor | Allocation | Mutant killed |
|---|---|---|
| Edge-on circular orbit: node receding, superior conjunction behind (phase 0, Lambert 1), inferior in front (phase pi, contrast 0) | exact signs, phase 1e-9 rad | phase supplement |
| Signs of (east, north) at four phases for i = 60 and 120 deg, position angle increasing for i below 90 | exact signs | east-north swap |
| Sky offsets at e = 0, 0.5, 0.9 over 64 epochs against the 30-digit Kepler solution | 0.01 mas; phase 1e-9 rad | (sensitive to node, periastron, scale) |
| Contrast at superior conjunction, quadrature and near inferior conjunction | 0.1 percent | (radius unit, distance power) |
| East offset at dec 60 deg is Delta alpha cos(delta) (astropy spherical offsets), unchanged by declination | 1e-6 relative | cos(delta) dropped; cos(delta) twice |
| The scene's stellar reflex velocity equals the published K [cos(f + w) + e cos w] with K = 48.4 m/s | 1.5 m/s | planet periastron set to the stellar value |
| World draws keyed by subject; capacity mismatch refused | exact | (subject count changes draws) |
| The truth scene of a draw equals photomancy's forward model of the same sites | 1e-9 arcsec | photomancy time origin dropped |
| `diagnose` gives the truth world's geometry at the science midpoint, not the nominal scene's or the occupancy midpoint; no science when the visit ends in settling | 1e-12 relative | occupancy midpoint |

All six named mutants were applied in memory and each failed its test.

## Survey (tests/adapters/test_physical_surveys.py, slow)

With the pinned response (IWA 4.06 lambda/D = 0.116 arcsec, OWA 32 lambda/D = 0.917
arcsec at 1000 nm):

- 47 UMa b over one period: separation 0.075 to 0.156 arcsec, outside the IWA 53
  percent of the orbit, always inside the OWA; contrast 3.7e-10 to 2.3e-8 (7.8e-9 at
  quadrature with the declared radius and albedo).
- 64 worlds from the prior: largest separation 0.228 arcsec (all inside the field);
  every world clears the IWA for part of its orbit, on average 62 percent of epochs.

## Cold budget

Each test alone, `SPACEODYSSEY_JAX_CACHE=off`, 4 threads: every test under 2.5 s
inside pytest (process wall 3.4 to 5.4 s including the skyscapes and ExoJAX import).
Slowest three by process wall: `diagnose` (5.4 s), sky signs retrograde (5.0 s),
photomancy match (5.0 s). The whole file takes 6.5 s.

## Where the spec was silent

- The truth draw uses photomancy's prior for every element, as Q10 says; the measured
  eccentricity, periastron, epoch and M sin i of 47 UMa b do not constrain the worlds
  (photomancy's prior has no such terms without radial-velocity data). The period
  range is the published period plus or minus three standard deviations.
- `diagnose` returns geometry, not rates: count rates belong to the summary radiometry
  task, which will extend it. Geometry is at the science midpoint; the flux-weighted
  integral over the science interval also arrives with radiometry.
- `truth_scene` and `diagnose` take a static subject index (the stars differ per
  subject). A compiled channel will need per-subject arrays.
- The star is a tabulated `Star` with a blackbody table, because `FlatStar` carries no
  angular diameter on skyscapes main; it can be swapped once that lands.
- hwoutils has no Jupiter radius; the profile states it in metres (IAU 2015 nominal).

## Review fixes

- `SeparationSupport` was checked only at the sampled epochs. It now takes `margin_d`
  and narrows the working-angle band at each edge by `margin_d` times an upper bound on
  the separation rate (the periastron sky-plane speed `n a sqrt((1 + e) / (1 - e))`
  over the distance), so a pass at every sample with `margin_d` half the spacing holds
  for the whole interval.
- New anchors for a nonzero node: the 30-digit comparison adds a 40 degree node, and a
  hand test puts an edge-on orbit with the node at 90 degrees due east of the star at
  the node epoch (270 degrees: due west). A node measured north through west now fails
  both.
