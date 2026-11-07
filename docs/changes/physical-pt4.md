---
orphan: true
---

# Physical track PT4: reporting law and record kind

PT4 turns summary count budgets into records: one normalized joint law of measured
counts and position, with detection and null as complementary events, a sampler and a
likelihood that share it, the record kind with its payload columns and audiences read
from the profile table, and a digest of the declarations each row was made under. The
code is in `src/spaceodyssey/adapters/jaxedith.py`; the tests are
`tests/adapters/test_reporting_law.py`, with the B4 refinement in
`tests/adapters/test_refinement.py` now scored by the law. No file under
`src/spaceodyssey/core/` or `codec/` changed and `tests/core/golden_histories.json` is
unchanged. Pure functions: nothing here touches a carry.

## Commits

On main `96a6447` (PT0 to PT3 merged):

1. `fix(adapters): cadence codes as the profile table states them` (`a625907`)
2. `feat(adapters): summary reporting law, record kind and calibration id` (`b31cf73`)
3. `test(adapters): score the B4 refinement with the reporting law` (`533ca85`)
4. `docs(changes): physical PT4 reporting law record` (`9449fe2`)
5. `fix(adapters): calibration id covers every budget declaration, per-law survey ids, Monte Carlo panels slow` (`88a042c`)
6. `docs(adapters): literal separation-rate bound in the support docstring` (`64003ee`)
7. `build(adapters): merged library pins, yippy 2.12 floor, fixed-frame default cadence, blocked anchors become ordinary` (`6ec9e9a`)
8. this record's update

## What is in

- `ReportingLaw(threshold=5, kappa=1, corr=0)`: static declarations, refused when the
  threshold or kappa is not positive and finite or `corr` is outside (-1, 1).
  - `limit_counts(reference)`: the planet-free noise standard deviation
    `s0 = sqrt(var_background + var_detector + var_systematic)` of a reference budget,
    which the caller builds from the model's nominal scene and the planned planet with
    no reflected light (`planet_free(planet)` zeroes `Lambda`) over the realized science
    interval. Never the truth scene.
  - `log_null_probability(limit, budget)`: `log Phi((threshold * limit - N) / s)` with
    `s = sqrt(budget.variance)`, the noise the budget predicts including the planet's
    own shot noise.
  - `log_counts_density(counts, limit, budget)`: `Normal(f; N, s^2)` on
    `f >= threshold * limit`, minus infinity below it.
  - `log_position_density(row, budget, valid)`: photomancy's
    `loglike_relative_astrom` with the row's own `ra_err`, `dec_err` and `corr`.
  - `position_error(snr, lod)`: `kappa * lod / snr`.
- `sample_report(key, law, expected, limit_counts, *, lod_arcsec, time_d, occupied_s,
  calibration_id, valid_science=True)`: three standard normals; counts
  `f = N + s eps1`; detection when `f >= threshold * limit`; on a detection the row
  carries `f`, `snr = observed_snr(f, limit) = f / sqrt(limit^2 + max(f, 0))` and a
  position `mu + L(f) (eps2, eps3)` with per-axis error `kappa * lod / snr` and
  correlation `corr`. A null carries epoch, durations, limit, `corr`, cadence and
  calibration id with zero counts, snr, position and errors. A row with zero science
  time, zero limit or `valid_science` false has no science (zero limit). Non-finite
  expected counts, variance or limit give NaN `detected`, `counts` and `limit_counts`,
  never a silent null.
- `loglike_report(law, row, predicted)`: detection rows add the counts and position
  densities, nulls the null log probability at the row's limit, rows without science
  zero; a NaN flag gives NaN.
- `SummaryRow`: one field per payload column in record order, `payload()` and
  `from_payload()` (refuses a width other than 15).
- `summary_kind(profile=None)`: `RecordKind("summary_measurement", ...)` with each
  column's `PayloadColumn` audience from the profile table through `audience_flags`;
  refuses a profile that fails `check_profile` (KP5) or whose columns are not
  `SUMMARY_COLUMNS` in order.
- `calibration_id(report, law)`: a six-byte BLAKE2b digest, as an integer below 2^48
  (exact in a float64 column, never zero), of every declaration a predicted budget
  depends on: the detector spec and cadence; every field of the optical path the nominal
  scene is evaluated on (primary, attenuating elements, detector, disperser,
  `n_channels`, `npix_multiplier`, speckle) except the coronagraph, which enters through
  its working angles, pixel scale and its throughput, core area, occulter transmission
  and leakage curves at the nominal host; the observatory (its orbit sets the local zodi,
  its temperature the thermal term); the band; `bg_multiplier`, `ppfact`,
  `eps_warm_T_cold`, the time nodes and weights and the orbit epoch; the field radius and
  leakage step; the nominal scene; and the law. Floats are digested to 12 significant
  digits and curves to 9, so last-bit platform differences do not move it.
- `CADENCE_CODES` is now `fixed_frame: 0, photon_counting_legacy: 1`, as the profile
  table's `cadence` column states (PT3 had them swapped; nothing had written a record
  yet). `DetectorSpec.fixture()` reads in fixed frames (code 0); the legacy cadence is
  used only when a detector declares it.

## Anchors and results

| Check | Result |
|---|---|
| Detection frequency against `1 - P(null)` at N = 3, 5, 7 s0 (s/s0 = 1.14, 1.22, 1.30), 20,000 draws, corr 0 and 0.5 | within 3 Monte Carlo standard errors at every level |
| Retained counts against the normal truncated at 5 s0 | Kolmogorov-Smirnov p above 0.01 at every level |
| Position Mahalanobis distance read off the likelihood against chi-square(2) | p above 0.01, corr 0 and 0.5; whitened means and cross moment within 4 standard errors |
| Joint normalization (counts and position quadrature plus the null row) | one to 1e-9 at three levels, corr 0 and 0.5 |
| Observables only | a second truth with twice the noise and the counts moved to the same draw gives the identical row in all 15 columns for 64 keys; `snr`, `ra_err`, `dec_err` recomputed from `counts` and `limit_counts` |
| KP5 | kind columns, order and audiences equal the profile table; `check_profile` empty; the cadence column's stated codes equal `CADENCE_CODES`; an incomplete or reordered profile is refused |
| photomancy null with s(theta) = s0 | equals `loglike_null` and the null branch of `loglike_imaging` to 1e-12 at five levels |
| Always-null scene | a null against planet-free proposals with background within 10 percent of the reference: log likelihood within 1e-5 of zero |

Controls inside the tests: a model-side threshold of 5.5, the fixed-denominator null
form and a doubled variance each miss the detection frequency by more than 3 standard
errors (except at the threshold, where every form gives one half); the
fixed-denominator null misses normalization by more than 1e-3 below the threshold; a
row flagged detected below the threshold scores minus infinity; positions scored with
the opposite correlation fail the chi-square check (p below 1e-6).

B4 rescored with the law (the record is now a detection at 1.1 N, about 5.5 s0, with
the limit from the 64-node reference): errors against 64 nodes at 1, 4, 8 and 16 nodes
are 3.6e-3, 3.7e-12, 8e-14, 2e-13 (relative counts), 3.3e-3, 4.8e-6, 1.6e-6, 1.4e-7
(log p) and 7.2e-3, 4.9e-9, 1.6e-9, 1.4e-10 (P(null)); observed order over the 4 to 16
ladder 2.5 for log p and P(null). The 8-node default stays inside the allocation.

## Survey: 200 worlds over the planned visits (slow tier)

Numbers at `6ec9e9a` (merged libraries, fixed-frame cadence). Plan from nominal
quantities only (profile target 0, declared i = 60 deg, node 0, Jupiter radius, albedo
0.5; the fixture detector in fixed 1000 s frames; true scene equal to the nominal one,
n_ez = 3): the first day of each 120-day step with the whole occupancy outside keepout
and the nominal planet between IWA + 0.5 and the field edge less 1 lambda/D, within four
years of the study epoch. Six visits qualify (science starts JD 2462681.5, 2462801.5,
2463193.5, 2463313.5, 2463759.5, 2463879.5); `exptime_ayo` for SNR 10 in whole fixed
frames gives one 1000 s frame at every visit (the legacy plan was 40 to 346 s); limits
11.8, 10.7, 10.3, 9.3, 10.9 and 11.9 e-. Worlds come from photomancy's prior through
`PhysicalWorldSource.from_profile` (seed 2026); draws fold in world and visit. Each
kappa's rows carry that law's own calibration id (47548537668345 at kappa 1,
246545890451345 at kappa 2).

- 1,200 rows, 423 detections against 427.6 expected (standard error 3.9); per visit
  70, 72, 68, 72, 70, 71. Detections per world: 0 for 112 worlds, then 5, 7, 3, 19, 5
  and 49 worlds with 1 to 6. Every row's log likelihood at the truth is finite.
- s(theta)/s0 at the truth runs from 0.77 to 27.9 (median 1.11): planet shot noise
  above the threshold, and the background at the true position differing from the
  planned position's below it. For the 605 rows below 1 s0, s/s0 is 0.91 to 1.17 (5 to
  95 percent) with four rows above 1.5 (largest 23.7, planets near the inner working
  angle where the leakage is far above the planned reference); one row below 1 s0 is a
  detection.
- photomancy null difference (law minus fixed-denominator form, log P(null) at the
  truth, every row), by N/s0:

| N/s0 | Rows | Median difference | Largest magnitude |
|---|---|---|---|
| 0 to 1 | 605 | -3.5e-7 | 0.54 |
| 1 to 3 | 121 | -7.5e-4 | 0.021 |
| 3 to 5 | 48 | -0.034 | 0.10 |
| 5 to 7 | 36 | +0.21 | 0.83 |
| above 7 | 390 | +3.5e3 | 2.6e7 |

  Below the threshold the law penalizes more; above it photomancy's fixed denominator
  ignores the planet's shot noise and over-penalizes a missed bright planet by orders of
  magnitude.
- photomancy's magnitude-space detection term (`dmag_err = 1.0857 / snr`, Jacobian
  added) against the law's counts term: median difference 0.013 nats (95th percentile
  0.13). Where the observable noise matches the truth (s / s_hat below 1.1, 421 of 423
  detections) the median is 0.013 and the largest 0.46; the remainder, up to 3.3e5, are
  detections whose true background exceeds the planned reference (s / s_hat up to
  10.5), where `snr` from observables overstates the signal-to-noise.
- kappa sensitivity: Laplace widths at the truth (detections' photometric, noise and
  astrometric Fisher information plus a Gaussian stand-in for each prior) for the 83
  worlds with two or more detections. kappa 2 over kappa 1, median (5 to 95 percent):
  a 1.37 (1.00 to 1.91), e 1.71 (1.16 to 1.96), cos i 1.74 (1.25 to 1.99), Omega 1.56
  (1.14 to 1.97), omega 1.53 (1.03 to 1.94), tp 1.60 (1.04 to 1.93), log10 Lambda 1.09
  (1.02 to 1.56). Every ratio lies in [1, 2]: the orbital angles and epoch widen nearly
  in proportion to kappa, the brightness scale barely.
- The 61 worlds whose planet never reaches 0.5 s0 gain at most 1.5e-4 nats between the
  truth and the same orbit with no reflected light.
- The survey runs in about 51 s cold.

## Builder self-review

- Cold budget (each test alone, `SPACEODYSSEY_JAX_CACHE=off`, four threads, setup plus
  call): the 10 unit tests under 1.7 s (slowest: no-science rows 1.7 s, photomancy
  cross-check 1.0 s, calibration mutations 1.3 s); the digested-field medium test 1.6 s.
  Slow: survey 50.7 s, B4 15.7 s, and the 20,000-draw panels (1.2 s and 0.6 s) and the
  correlated-error control (0.2 s), which spec section E assigns to the slow tier.
- Mutants (in-memory patches, each run against the quick tests of the file, each
  failing): snr from the expected budget (observables-only test); the fixed-denominator
  null, a doubled variance, a detection threshold on the true noise, a sign-cancelling
  correlation and a centroid shifted by 0.1 sigma (sampling test); correlation dropped
  from sampling or flipped in the row (sampling test); no support restriction
  (normalization test); kappa ignored (kappa test); NaN guard removed and the zero-limit
  validity check removed (no-science test); the law left out of the calibration digest
  (calibration test); the optical path, the occulter transmission curve or the
  observatory left out of the digest (calibration test, naming the mutations that
  collide); swapped cadence codes (KP5 test). The 0.1-sigma centroid shift
  passed the chi-square check alone and was caught only after the whitened-mean
  check was added.
- Audience isolation: every column is readable by observer, policy and scorer, as the
  profile states; the kind lowers to fifteen readable columns. No scorer-only value
  exists in this row: the noise variates are never stored, and the observables-only
  test shows no expected-truth quantity reaches a column.
- Ordering and metadata: the calibration id depends on declarations only, not on
  capacities or padding.
- Refusal paths: law declarations (zero, negative, NaN, infinite, bool, None);
  payload width; incomplete and reordered profiles; zero-length interval, zero limit and
  invalid science; NaN and infinite truth.
- Sibling sites: the cadence code is read only through `DetectorSpec.cadence_code`
  (one site); the profile's columns are read by `summary_kind` and the PT0 page test.
- Goldens unchanged; the golden test and both gate files run in the quick tier.
- Quick tier at `6ec9e9a` (serial, `-m "not slow"`, entry-point install test
  deselected): 1,964 passed, 0 expected failures, 0 failed, in 221 s, peak footprint
  2.4 GiB. Every slow and medium test of the three touched adapter files (survey, B4,
  Monte Carlo panels, PT3's leakage anchor, cached-rate parity, jaxedith agreement and
  detectability survey) passes at the same tree. The full suite was not rerun.
- Docs: `sphinx -W` builds at `64003ee` (the separation-rate docstring in
  `adapters/skyscapes.py` now quotes `|d separation / dt|` as a literal).
- Tests added 15 (10 unit, 1 medium, 4 slow including the survey; the B4 test is
  rescored, not added), retired 0, net +15.

## Review fixes

1. The calibration id now digests every field of the optical path except the
   coronagraph (so `n_channels`, `npix_multiplier`, the primary, the attenuating
   elements and the detector), the coronagraph's occulter transmission curve beside its
   other curves, the observatory and the time weights. `test_calibration_id_digests_every_declaration`
   applies 21 single-field mutations and requires 22 distinct ids;
   `test_every_digested_report_field_moves_the_predicted_budget` (medium) shows each
   report mutation changes a predicted budget.
2. The survey computes the id once per law; the kappa-2 rows carry the kappa-2 law's
   id, and the survey test asserts each kappa's rows carry exactly their law's id.
3. The 20,000-draw panels and the correlated-error control are marked slow with reasons.
4. The `sphinx -W` failure in `adapters/skyscapes.py` is fixed.

## Library merges

jaxedith `9f1c13b` (QE times dQE on photon rates, frame-time read noise, the
fixed-frame variance and planners), optixstuff `735a651` (`SimplePrimary.from_yip`,
dQE default 1.0) and skyscapes `bf22955` (`FlatStar.diameter_arcsec`) are on main but
unreleased: `.github/sibling-pins.txt` pins them beside photomancy, and the fixture
manifest records those revisions with development floors (2.0.1.dev0, 4.0.2.dev0,
2.1.1.dev0) that the manifest test checks against the pins. yippy 2.12.0 carries the
response metadata, so the yippy pin is dropped and both yippy floors are `>=2.12.0`
(manifest revision `ac0c24f`, the release merge). The manifest test now covers every
manifest library, jaxedith and optixstuff included.

- PT3's five strict expected failures are ordinary anchors, renamed without
  `blocked_`: QE once, dQE as a survival fraction, the fixed-frame detector variance
  (sub-frame, one, three and three-plus-partial frames), the fixed-frame SNR against
  jaxedith with the floor off and on, and the primary from the package header. All pass.
- The adapter imports `count_variance_components_ayo` directly; the `NotImplementedError`
  refusal of the fixed-frame cadence is gone.
- `DetectorSpec.fixture()` reads in fixed frames (Q8 A, the profile's cadence),
  recording code 0; the legacy cadence stays available by declaration. The hand budget
  test declares the legacy cadence (it writes out the legacy detector term); the
  jaxedith agreement test checks `snr_ayo` in both cadences and, for fixed frames,
  `exptime_ayo` with minimum-duration planning (SNR 10) and whole-frame planning (SNR at
  least 10, one frame less below it); both surveys plan in fixed frames.
- The profile's `cadence` convention now states fixed frames with the legacy cadence by
  declaration, and the `detector_variance` boundary is `pinned`; the page matches.
- In the workspace, the editable installs of jaxedith, optixstuff, skyscapes and yippy
  were reinstalled without dependencies so their recorded versions match the checked-out
  commits, and the yippy clone was fast-forwarded to `origin/main` (the 2.12.0 release
  commit, a changelog only).
- FlatStar does not replace the PT1 `Star`: it has no coordinates or luminosity, which
  the local zodi and the exozodi scaling read, and its flat spectrum differs from the
  blackbody table across the band. The profile page says so.

## Spec silence

1. The limit's reference position: `s0` is built at the planned (nominal) planet's
   position over the realized interval, so it is a model quantity and the same for
   every proposal. Where the true planet sits much closer to the star, the true
   background exceeds the reference and the observable `snr` overstates the true one
   (survey above). Alternative rejected: a limit as a function of separation (a
   contrast curve), which the scalar `limit_counts` column (P8) cannot carry.
2. `s0` includes the systematic-floor variance and the background multiplier, so it
   equals the law's `s(theta)` at zero planet counts at the same position; the spec
   writes `sqrt(B_nominal + D(t))`, which is the same in the base case (k = 1, floor
   off).
3. `limit_counts` is `s0` itself (the profile's measure), and detection is
   `counts >= 5 limit_counts`, as the profile's `detection` convention states. PT3's
   B4 record carried `5 s0`; B4 now uses the law.
4. A detection row's position error uses `kappa * (lambda/D) / snr` on both axes with
   the law's `corr`; `corr` is written on every row, including nulls, because the
   profile states it as "in (-1, 1)" without a detection condition.
5. The cadence codes followed the profile table rather than PT3's constant.
6. The law's SNR agreement with `snr_from_system_ayo` (floor off and on) is PT3's
   `test_variance_agrees_with_jaxedith_and_the_system_wrapper`: the law's `s` is
   `sqrt(budget.variance)`, the quantity that test checks.
7. The pre-flight refusal of analytic or integrate requests belongs to the channel
   (PT9); only `sample_report` and `loglike_report` exist here.
8. The survey's Laplace widths use the detections only (nulls do not depend on kappa)
   and a Gaussian stand-in for each prior so unconstrained directions stay finite.
