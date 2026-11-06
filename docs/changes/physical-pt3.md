---
orphan: true
---

# Physical track PT3: summary radiometry

PT3 adds the summary-radiometry adapter: expected counts and count variances of one
companion over a science interval, with every component separate, from jaxedith's rates
on the pinned public coronagraph response. The code is
`src/spaceodyssey/adapters/jaxedith.py`; the tests are `tests/adapters/test_jaxedith_summary.py`
and `tests/adapters/test_refinement.py`, with shared physical fixtures in
`tests/adapters/conftest.py`. No file under `src/spaceodyssey/core/` or `codec/` changed and
`tests/core/golden_histories.json` is unchanged. `pyproject.toml` gains the `physical` extra
(Q7 default A) and two warning filters for imports that jaxedith pulls in.

## Commits

Replayed onto main `45bb1c2` (PT0 to PT2 merged), with the physical extra merged into
PT1's list:

1. `feat(adapters): summary radiometry through jaxedith with time-integrated counts and variance components` (`581dc7e`)
2. `fix(adapters): zero summary SNR for a zero-length interval` (`e0ef50d`)
3. `fix(adapters): finite summary centroid for a zero-length interval` (`0b3983d`)
4. `docs(changes): physical PT3 summary radiometry record` (`032cf93`)
5. `build: pin unreleased yippy for the summary adapter's response metadata` (`c176aea`)
6. `fix(adapters): rates through jaxedith's QE-aware intermediates and finite-frame variance when present` (`e8d651e`)
7. `test(adapters): hold the B4 observation fixed across the time ladder` (`36bbf55`)
8. this record's review-fix update

## What is in

- `DetectorSpec`: a frozen, declared detector (name, family, quantum efficiency, dQE and its
  meaning, dark current, read noise, clock-induced charge, frame time, pixel scale, cadence).
  Construction refuses an empty name or family, a dQE meaning other than
  `survival_fraction`, an unknown cadence and any non-finite or out-of-range number (KP10).
  `fixture()` is jaxedith's test detector (QE 0.9, dark 3e-5 e/pix/s, read noise 0, CIC
  1.3e-3 e/pix/frame, 1000 s frames, 0.01 arcsec pixels) in the `photon_counting_legacy`
  cadence. The read time is the frame time. `cadence_code` is the integer a record column
  carries (`CADENCE_CODES`: legacy 0, fixed frame 1).
- `SummaryScene` (stellar flux density at the band center, distance, mass, luminosity,
  coordinates, angular diameter, exozodi level in zodis) and `SummaryTruth(scene, planet)`,
  with the planet in photomancy's physical names (`a`, `e`, `cos_i`, `W`, `cos_w`, `sin_w`,
  `tp`, `Lambda`, optional `Mp`). `nominal_planet` builds that dict from plain elements.
- `ScienceInterval(midpoint_jd, duration_s)` with `from_start`: the realized science interval.
- `SummaryReport.from_yip(detector_spec, nominal_scene, orbit_epoch_jd, observatory, ...)`
  fetches the pinned package through `fetch_pinned_yip` (yippy's hash-checked loader; refuses
  when yippy's digest or data release differ from `md5:1f4892faff18e55cbec9781a055bea4d`,
  `data-v2`), takes D = 7.2 m and the 1000 nm center from the package, builds
  `SimplePrimary(D, sqrt(OBSCURED))`, optics throughput 0.5, `bg_multiplier = 1`,
  `ppfact = inf`, `eps_warm_T_cold = 0`, 8 time nodes, band 100 nm.
- `rates(prepared, planet, times_jd)` returns `RateComponents` per epoch (planet, leakage,
  zodi, exozodi, thermal, detector, floor rates; contrast, east and north offsets,
  separation, in-field flag). `budget(prepared, planet, interval)` integrates them with
  Gauss-Legendre nodes into a `CountBudget`: counts `planet`, `leakage`, `zodi`, `exozodi`,
  `thermal`; variances `var_planet`, `var_background`, `var_detector`, `var_systematic`;
  `variance`, `snr`; flux-weighted `ra_arcsec`, `dec_arcsec`; `science_s`, `in_field`,
  `cadence`.
- `expected(truth, interval)` is `budget(prepare(truth.scene), truth.planet, interval)`;
  `predict(theta, interval)` is `budget(nominal, theta, interval)`. One function serves both;
  `predict` has no access to a truth scene.

## How the evaluation is cached

Geometry-invariant factors live in `PreparedScene` and on the report: the flux zero point,
the stellar ratio, the exozodi surface brightness at 1 AU
(`n_exozodi * zodi_fn_ayo(lambda) * L/Lsun`, the J2 formula), distance, mass, the stellar
diameter in lambda/D and the band.
The stellar leakage at the host's diameter is tabulated once on the response's own radial
profile nodes (`yippy.performance.PROFILE_STEP_PIX` times the stellar pixel scale, 0.0625
lambda/D) and read with PCHIP, which reproduces the library's curve exactly; evaluating it
directly at a finite diameter re-averages a whole stellar map per call. The table is served
through `TabulatedLeakageCoronagraph`, which delegates throughput, core area and occulter
transmission to the response, and each `PreparedScene` carries an optical path with it. At
every node of every proposal the planet position and contrast come from photomancy's
`predict_relative_astrometry` and `predict_photometry`, and every rate from jaxedith's
intermediates (`planet_signal`, `stellar_leakage`, `zodi_background`,
`exozodi_background`, `thermal_background`, `detector_noise`, `stellar_noise_floor`) on
that optical path, so detector QE and dQE are applied wherever jaxedith applies them. Local zodi is
`zodi_fn_leinert` at each node's epoch.

## Library state and blocked anchors

The adapter builds against merged library mains. The legacy photon-counting cadence is the
fixture's (`cadence = 0` in every budget). A `fixed_frame` detector uses jaxedith's
`count_variance_components_ayo` (which applies `detector_variance_counts`) for the whole
interval when jaxedith provides it, and is refused with `NotImplementedError` naming that
API when it does not. Verified against archives of the two branches: with
`codex-detector-qe` (`a0db634`) both QE anchors pass; with `codex-fixed-frame` (`9f1c13b`)
both QE and both fixed-frame anchors pass, so on merge each strict expected failure turns
into an unexpected pass. Each blocked anchor is a strict expected failure that names its library item
and the exception it must fail with, so a merge turns it into an unexpected pass.

| Test | Library item | Fails on main with |
|---|---|---|
| `test_blocked_quantum_efficiency_applies_exactly_once` | J0, jaxedith `codex-detector-qe` | `AssertionError` (main applies no QE to photon rates) |
| `test_blocked_dqe_is_a_survival_fraction` | J0, jaxedith `codex-detector-qe` | `AssertionError` (main ignores dQE outside thermal) |
| `test_blocked_fixed_frame_detector_variance` | J1, jaxedith `codex-fixed-frame` | `ImportError` (`detector_variance_counts`) |
| `test_blocked_fixed_frame_floor_agrees_with_jaxedith` | J1, jaxedith `codex-fixed-frame` | `ImportError` (`count_variance_components_ayo`) |
| `test_blocked_primary_from_yip_matches_the_adapter_area` | O1, optixstuff `codex-primary-from-yip` | `AttributeError` (`SimplePrimary.from_yip`) |

Consequence of J0 for current numbers: on jaxedith main the planet and photon background
rates carry no detector QE, so with the fixture's QE 0.9 every photon count is 1/0.9 of the
declared detector's. The adapter does not compensate (no copy of the branch).

## Tests

| Test | File | Tier | Cold s (setup + call) |
|---|---|---|---|
| detector and report refusals (KP10, fixed-frame refusal) | summary | unit | 0.1 |
| hand count and variance budget (k 1 and 2, floor off and on, zero-length interval, raw-area control) | summary | unit | 3.4 |
| exozodi radius, luminosity, level, aperture and distance scalings | summary | unit | 3.1 |
| expected and predict share one function; components split | summary | unit | 3.0 |
| moving planet counts and centroid against a fine grid | summary | unit | 3.2 |
| five blocked anchors | summary | unit (xfail) | 0.1 to 3.0 |
| pinned digest equals yippy's catalog | summary | unit | 0.0 |
| cached rates against jaxedith's full call near the IWA, single-coefficient mutant | summary | medium | 6.7 |
| SNR against `snr_ayo` and `exptime_ayo` (floor off and on); rates against `count_rates_from_system_ayo`; collecting area from the header | summary | medium | 12.5 |
| leakage operator against a circle average of the cube | summary | slow | 4.5 |
| detectability survey over one orbit of 47 UMa b | summary | slow | 17.4 |
| B4 time refinement of counts, log p(f, pos), P(null) | refinement | slow | 16.4 |

Added 16 (11 pass, 5 strict expected failures), retired 0, net +16. The quick tier gains
seven unit tests, five expected failures and two medium tests (the A.4 allowance).

## Anchor and mutant results

- Cached against full call (24 proposals, seed 0, 8 nodes over two days, ppfact 30): the
  largest relative difference is 0.0 on the planet rate, 4.4e-16 on the total background,
  0.0 on the floor and 3.3e-16 on the integrated counts (tolerance 1e-9). Ten proposals lie
  within 1 lambda/D of the IWA (4.06 lambda/D). The single-coefficient mutant misses by
  5.0 to 59 percent at the nine other near-IWA proposals (threshold 1 percent).
- System wrapper: rates equal `count_rates_from_system_ayo` to 7e-7 (skyscapes' grid Kepler
  solver places the planet within 1e-6 of a / d; test tolerance 1e-5). SNR equals
  `snr_ayo` to 1e-6 with the floor off and on; the SNR at `exptime_ayo(10)` is 10.
- Leakage operator (P3): the cube's circle mean about 127.5, divided by 0.25^2, agrees with
  the adapter to 0.53 percent from 3 to 20 lambda/D (test 1 percent, allocation 5 percent).
  Origin 128 misses by up to 4.1 percent (5 lambda/D); annular bins by 3.4 percent at
  3 lambda/D (17.5 percent at 5); the omitted division by 16x. The 0.5 lambda/D rebinned
  cube divided by 0.5^2 agrees within 0.5 percent from 8 to 20 lambda/D. A 0.03 lambda/D
  host leaks 0.68 percent more than a point star at 5 lambda/D (1.8 percent at 2.5 to
  3 lambda/D).
- Collecting area: pi (D/2)^2 (1 - OBSCURED) to 1e-12; the raw fraction as obscuration
  overstates it by 17 percent.
- B4 (fast mover: 0.6 AU edge-on at 4 pc, 4.7 to 4.3 lambda/D in five days, N about
  5 s0), every level scored against one fixed observation (counts, position and
  `limit_counts = 5 s0` from the 64-node reference): errors against 64 nodes at 1, 4, 8,
  16 nodes are, for counts (relative) 3.6e-3, 3.7e-12, 8e-14, 2e-13 (roundoff from 4 nodes:
  the throughput is one smooth table segment over the arc); for log p 1.3e-2, 4.8e-6,
  1.6e-6, 1.4e-7; for P(null) 7.2e-3, 4.9e-9, 1.6e-9, 1.4e-10. Observed order over the 4 to
  16 ladder 2.5 for both likelihood quantities (the leakage table is piecewise cubic, so no
  spectral order is claimed). The 8-node default sits inside the allocation (1e-4 counts,
  0.01 log p, 1e-3 P(null)); the midpoint misses it.
- Survey, 47 UMa b on general-knowledge elements (2.1 AU, e 0.03, i 60 deg, Rp 11.2
  Rearth, Ag 0.3; host 42 Jy at 1 micron, 13.8 pc, 0.79 mas), jaxedith's `exptime_ayo` for
  SNR 10 at the start, then the integrated summary SNR over that exposure:

| Phase | Separation (lambda/D) | Contrast | t(SNR 10), h | Summary SNR |
|---|---|---|---|---|
| 0.000 | 4.98 | 7.5e-9 | 0.02 | 10.000 |
| 0.083 | 3.94 | 1.2e-8 | 0.02 | 10.000 |
| 0.167 | 2.75 | 1.4e-8 | 0.03 | 10.000 |
| 0.250 | 3.06 | 1.3e-8 | 0.03 | 10.000 |
| 0.333 | 4.41 | 1.0e-8 | 0.02 | 10.000 |
| 0.417 | 5.34 | 6.3e-9 | 0.03 | 10.000 |
| 0.500 | 5.34 | 3.2e-9 | 0.06 | 10.000 |
| 0.583 | 4.43 | 1.2e-9 | 0.41 | 9.999 |
| 0.667 | 3.12 | 3.6e-10 | 4.04 | 9.986 |
| 0.750 | 2.77 | 2.6e-10 | 10.27 | 10.027 |
| 0.833 | 3.90 | 9.2e-10 | 0.45 | 10.001 |
| 0.917 | 4.95 | 3.0e-9 | 0.09 | 10.000 |

  The planned exposures are minutes to hours at 1 micron, many shorter than one 1000 s
  frame; under a fixed-frame cadence each would be at least one frame.

## Builder self-review

- Cold budget: every new test alone with `SPACEODYSSEY_JAX_CACHE=off` under the four-thread
  launcher (table above). Slowest: survey 17.4 s, B4 16.4 s, system-wrapper medium 12.5 s.
  All units under 5 s, both mediums under 30 s.
- Mutants, each run as an in-memory patch against its test, each failing: refusal checks
  removed (`DID NOT RAISE`); leakage table at twice the node step (parity misses by 1.1e-3);
  exozodi without the luminosity factor (scaling and hand tests); a time-mean centroid
  (moving test); the detector variance without the background multiplier (hand test); the
  leakage curve read a half pixel off (operator anchor); midpoint-only integration (moving
  test). The single-coefficient, origin-128, annular, omitted-division and raw-area mutants
  are inside the tests. B4's control is the midpoint evaluation, which misses the
  allocation; a NaN budget fails every comparison.
- Audience isolation: no record kind or payload column is added (PT4). `expected` is the
  truth path; `predict` takes only a parameter dict and the report's nominal scene, and the
  sharing test shows a different truth scene does not reach it.
- Ordering and metadata: nothing stamps, digests or releases rows.
- Refusal paths: undeclared detector fields, wrong types (`None`, `True`), NaN and infinity,
  out-of-range QE and dQE, zero frame time, unknown cadence, the fixed-frame cadence,
  `n_time_nodes` 0, `bg_multiplier` 0, negative or NaN `ppfact`; a zero-length interval
  gives zero counts, SNR 0 and a finite centroid (the centroid assertion failed before the
  third commit: the time-mean fallback divided by the zero total weight).
- Sibling sites: detector validation runs at `DetectorSpec` construction and the cadence and
  option checks in `SummaryReport.__init__`, which `from_yip` calls; `eqx.tree_at` bypasses
  both by design (tests use it to swap node sets and `ppfact`). The pinned digest is checked
  in `fetch_pinned_yip`, the only loader path.
- Goldens: unchanged; the golden test and both gate files pass in the full run.
- Full suite (worktree, serial under the four-thread launcher with a 6 GiB ceiling, entry-point
  install test deselected, at the second commit): 3,193 passed, 5 expected failures (the
  blocked anchors), 0 failed, in 3,735 s, peak footprint 4.9 GiB. A first attempt at
  `-n 2 --dist loadfile` exceeded the 6 GiB ceiling and was killed by the launcher. The third
  commit touches only the zero-length centroid; the adapter files and the quick tier were
  rerun after it.
- Quick tier at the third commit (`-m "not slow"`, serial; `-n 2` exceeded the 6 GiB ceiling):
  1,909 passed, 5 expected failures, 0 failed, in 310 s, peak footprint 3.8 GiB. The slow
  adapter tests (leakage anchor, survey, B4) pass at the same tree.
- Quick tier after the review fixes (`36bbf55`, serial): 1,948 passed, 5 expected failures, 0
  failed, in 364 s, peak footprint 3.3 GiB; the slow adapter tests (leakage anchor, survey,
  B4) pass at the same tree.
- Docs: `sphinx -W` builds (autoapi documents the new module).

## Spec silence

1. The row of `expected(truth, row)` and `predict(theta, row)` is a `ScienceInterval`
   (midpoint JD, duration), not a payload row: the realize context (PC1) is not merged and
   the record kind is PT4's. Alternative rejected: reading payload columns by index here.
2. Truth and prediction share photomancy's forward model for geometry (PL1: the same
   function). PT1's skyscapes scene maps onto `SummaryTruth`; geometry conventions are its
   anchors. Alternative rejected: a skyscapes `System` as the truth type before PT1.
3. The leakage cache is tabulated on the response's profile nodes (exact parity) rather than
   a dense grid with an interpolation tolerance (1e-4 to 7e-4 measured at 2 to 8 times the
   node density). The other response curves are not re-tabulated.
4. Local zodi is evaluated at each node's epoch; jaxedith's system wrapper uses the exposure
   start. The parity tests pass the same per-node value to both sides.
5. In the legacy cadence the background multiplier also scales the detector term, as
   jaxedith's exposure-time equation does; the spec's form applies it to photon backgrounds
   only. The two agree at the declared k = 1.
6. QE is not compensated while jaxedith main ignores it (see Library state).
7. The survey compares with the scalar `exptime_ayo`, which the system wrapper vmaps, since
   the skyscapes scene is PT1's.
8. The leakage anchor's test tolerance is 1 percent, inside the 5 percent allocation, so the
   origin-128 mutant (4.1 percent) fails as P3 requires.
9. B4 requires an observed order of at least 2 over the 4 to 16 ladder and monotone errors,
   not at each step; quantities at roundoff from the first level are reported as converged.
   The default of 8 nodes is set from this ladder.
10. `in_field` is the separation inside the 32 lambda/D field; PT3 flags and does not refuse
    (refusal is the context's and the channel's). The leakage table reaches the field corner
    and is clipped beyond it.
11. 47 UMa values are general knowledge (flux 42 Jy at 1 micron from 2MASS J about 3.96,
    13.8 pc, 1.03 Msun, 1.54 Lsun, 0.79 mas), pending PT0's pinned sources.
12. The planet-free reference `s0` is not provided: in the legacy cadence the detector term
    depends on the planet rate through the photon-counting time. PT4 builds it.
13. `pyproject.toml`: the `physical` extra (jaxedith, optixstuff, yippy, orbix, skyscapes,
    interpax at their current release floors) and the test extra now installs it plus
    coronalyze; two warning filters copied from skyscapes for ExoJAX's unclosed `/dev/null`
    and HAPI, ExoJAX and Vaex escape sequences, which jaxedith's skyscapes import triggers.

## Review fixes

1. Unreleased yippy API (P1): `from_yip` reads `EqxCoronagraph.stellar_frame` and
   `stellar_pixel_scale_lod`, which yippy 2.11.0 lacks; both arrived in yippy `471826b`.
   `.github/sibling-pins.txt` now pins that commit beside photomancy, and `pyproject.toml`
   keeps `yippy>=2.11` with a comment: the floor moves to the first release that carries
   them.
2. QE through the library (P2): the rates now come from jaxedith's intermediates on an
   optical path whose coronagraph serves the leakage table (`TabulatedLeakageCoronagraph`),
   instead of calling the primitives with the bare optics throughput. With
   `codex-detector-qe` the half-QE and 0.8 dQE anchors pass (they reproduced 467.58 for
   233.79 before). Main's QE gap is still recorded and not compensated. The moving-planet
   test now uses QE 1 and dQE 1, like the hand budget, so it holds on both jaxedith states.
3. Fixed-frame cadence (P2): refused only when jaxedith lacks `count_variance_components_ayo`;
   otherwise the detector variance of the interval is jaxedith's finite-frame value (no
   background multiplier), averaged over the nodes' aperture pixel counts. The refusal test
   checks whichever applies. With `codex-fixed-frame` both fixed-frame anchors pass.
4. B4 observation (P2): the record now carries a fixed `limit_counts` (5 s0 of the
   reference), so the detection threshold and the astrometric covariance are the same at
   every level; only the predicted counts, centroid and variance are refined. Numbers above.

## Not done

- B4 on the band axis (the tilted synthetic spectrum): the adapter evaluates the band at its
  center times its width, as jaxedith does; band quadrature needs a spectral star model and
  belongs with the scene. The Done-when line requires the time axis only.
