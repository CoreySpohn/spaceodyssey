---
orphan: true
---

# Profile first-summary-v1 (draft)

A physical profile fixes every quantity, sign and time choice that a physical
study makes, so that no choice is implicit. `first-summary-v1` covers one
companion, one facility, one band and scalar summary measurements. This page is
the readable form of the table that ships with the package
(`spaceodyssey/adapters/first_summary_v1.json`); a test reads both and fails if
they disagree.

Status: draft. Every entry is a proposal until it is adopted.

## Conventions

- **sky basis.** Right-handed, star-centred (east, north, away) frame in AU; row 2 is positive away from the observer.
- **east offset.** The east offset is Delta alpha cos(delta): the declination factor is already applied and is never applied again.
- **node.** The longitude of the ascending node is measured from north through east; the ascending node is the sky-plane crossing where the planet moves away from the observer.
- **position angle.** Position angle increases with time for inclinations below 90 degrees.
- **inclination.** Inclination lies in [0, pi], isotropic in the prior (cos i uniform in [-1, 1]).
- **phase angle.** Standard star-planet-observer phase angle with cos(beta) = z / r and z away: 0 at superior conjunction (full phase), pi at inferior conjunction. No supplement conversion is applied anywhere.
- **argument of periastron.** The planet's argument of periastron is the radial-velocity (stellar) argument of periastron plus 180 degrees.
- **time scale.** Absolute epochs are Julian Dates in TDB as float64; MJD = JD - 2400000.5.
- **tick.** One tick is 1 ms; tick 0 of a study is the context's origin epoch, held as an integer number of ticks from J2000.0 (JD 2451545.0 TDB).
- **intervals.** A record's start and end ticks delimit occupancy (settling, then science). The science interval is [start + settle, end]; for a complete visit it is the last exposure_ticks of the occupancy.
- **record epoch.** The record epoch is the midpoint of the science interval.
- **certification.** Before acquisition, every admissible executed start in [planned_start, planned_start + max_delay] must keep the whole occupied interval inside the keepout window and the response support; at run time a science interval outside the certified window gives valid_science = 0.
- **keepout.** Sun keepout 45 to 180 degrees, Earth keepout 0 to 180 degrees (inactive), both passed explicitly to orbix.
- **response origin.** Header centre N/2 is zero-based pixel-centre index N/2 - 0.5 (127.5 on a 256 grid) for every azimuthal operation.
- **leakage operator.** Stellar leakage in the aperture is the azimuthal-mean intensity per pixel divided by 0.25^2 and multiplied by the core area in (lambda/D)^2.
- **dqe.** dQE is an explicit survival fraction, neutral 1.0, always passed; QE is applied exactly once.
- **background multiplier.** Background variance multiplier k = 1 (one roll, no differencing, known mean subtracted), passed explicitly.
- **systematic floor.** The systematic noise floor is off in the planner and the law (ppfact infinite).
- **thermal.** Thermal emission is off (eps_warm_T_cold = 0); the telescope temperature is inert.
- **exozodi.** Fexozodi at 1 AU = n_ez x one-zodi surface brightness x (L / Lsun), with n_ez = 3 passed explicitly.
- **detection.** Detection is the event counts >= 5 limit_counts; a null is the complementary event at a realized science duration t > 0.
- **cadence.** Fixed frames with integer frame counts: read noise and clock-induced charge per frame (a partial last frame counts as one), dark current per second; planned exposures are whole frames. The legacy photon-counting cadence stays available when a detector declares it, and every row records its cadence.

## Time

Epochs are Julian Dates in TDB, held as float64. MJD is JD - 2400000.5. One tick
is 1 ms, and tick 0 of a study is held as an integer number of ticks after
J2000.0 (JD 2451545.0 TDB), so moving the origin one way and every tick the other
way changes no epoch. The first campaign starts at JD 2462502.5 (2030-01-01
00:00 TDB), a declared fixture choice. The difference between TDB and UTC (under
70 s) moves 47 UMa b by less than 0.001 mas and is declared outside the budget.

## Targets

### 47 UMa (HIP 53721)

Host values come from the NASA ExEP Mission Star List for HWO (Mamajek and
Stapelfeldt 2024, arXiv:2402.12414), row HIP 53721, as served by the NASA
Exoplanet Archive table `di_stars_exep` (retrieved 2026-11-06).

| Quantity | Value |
|---|---|
| Mass | 1.07 Msun |
| Distance | 13.8875 pc (Gaia DR3 parallax 72.0070 mas) |
| Right ascension, declination | 164.8665531 deg, +40.4302557 deg |
| Luminosity | log10(L / Lsun) = 0.1978 |
| Effective temperature | 5880 K |
| Radius | 1.21 Rsun |
| Angular diameter | 0.81 mas |
| V magnitude | 5.037 |

The flux table of the star is a blackbody at the catalog effective temperature
and radius, tabulated across the band at one epoch.

### 47 UMa b

Elements come from Gregory and Fischer (2010), MNRAS 403, 731
(arXiv:1003.5549), Table 6: the final three-planet model fitted to the combined
Lick, HET and HJS velocities (medians, planet 1).

| Element | Value |
|---|---|
| Period | 1078 +- 2 d |
| Eccentricity | 0.032 +- 0.014 |
| Stellar argument of periastron | 334 +- 23 deg |
| Planet argument of periastron | 154 deg (stellar value plus 180 deg) |
| Periastron passage | JD 2451917 (listed as JD - 2440000 = 11917) |
| M sin i | 2.53 Jupiter masses |
| Semimajor axis (paper, for a 1.063 Msun host) | 2.10 AU |

Not measured, so declared: inclination 60 deg (the median of an isotropic
inclination prior), node 0 deg (a convention choice), radius 71492 km (the IAU
2015 nominal Jovian equatorial radius) and geometric albedo 0.5 (a planning
value). The planet mass is M sin i divided by sin(60 deg). The semimajor axis
follows from the period through Kepler's third law with the catalog host mass.
These nominal values enter the plan only. The world source draws each world from
photomancy's orbit prior: the period uniform in log10 between 1072 and 1084 d
(the published period plus or minus three standard deviations), the `kipping13`
eccentricity prior, isotropic orientation, uniform phase, and photomancy's
default log-uniform ranges for mass, radius and albedo. The measured
eccentricity, argument of periastron, periastron epoch and M sin i do not
constrain the drawn worlds.

### Second target: a solar twin at 10 pc

A synthetic, unrelated target: a solar twin (1 Msun, 1 Lsun, 5772 K, 1 Rsun) at
10 pc on the celestial equator at right ascension 0 (ecliptic latitude 0), with
an Earth twin on a circular one-year orbit (inclination 60 deg, radius 6378.1 km,
albedo 0.2, one Earth mass).

## Response

The coronagraph response is the public yield input package `eac1_aavc_2d`,
release `data-v2`, digest `md5:1f4892faff18e55cbec9781a055bea4d`, fetched and
hash-verified by `yippy.datasets.fetch_yip`. Its header gives an aperture of
7.2 m, an obscured area fraction of 0.17481, a band of 950 to 1050 nm (centre
1000 nm), pixels of 0.25 lambda/D and a 256-pixel grid with header centre 128.

## Payload columns

| Column | Unit | Sign | Measure | Audience |
|---|---|---|---|---|
| `time_d` | d | increasing with time | Julian Date (TDB) of the science-interval midpoint | observer, policy, scorer |
| `science_s` | s | nonnegative | realized science duration | observer, policy, scorer |
| `occupied_s` | s | nonnegative | executed occupancy (settling plus science) | observer, policy, scorer |
| `detected` | 1 | flag: 1 detection, 0 null or no science | whether measured counts reached 5 limit_counts | observer, policy, scorer |
| `counts` | e- | signed (noise can make it negative); 0 and unread unless detected = 1 | measured planet counts integrated over the band and the science interval, known background mean subtracted | observer, policy, scorer |
| `limit_counts` | e- | positive when valid_science = 1, else 0 | planet-free noise standard deviation from the nominal backgrounds and the detector variance at the realized science duration | observer, policy, scorer |
| `snr` | 1 | signed; 0 and unread unless detected = 1 | counts divided by the observable noise estimate sqrt(limit_counts^2 + max(counts, 0)) | observer, policy, scorer |
| `ra` | arcsec | positive east | east offset Delta alpha cos(delta) of the planet from the star, flux-weighted over the science interval; 0 and unread unless detected = 1 | observer, policy, scorer |
| `dec` | arcsec | positive north | north offset Delta delta of the planet from the star, flux-weighted over the science interval; 0 and unread unless detected = 1 | observer, policy, scorer |
| `ra_err` | arcsec | positive when detected = 1, else 0 | standard deviation of ra, kappa (lambda/D) / snr | observer, policy, scorer |
| `dec_err` | arcsec | positive when detected = 1, else 0 | standard deviation of dec, kappa (lambda/D) / snr | observer, policy, scorer |
| `corr` | 1 | in (-1, 1) | correlation of the ra and dec errors | observer, policy, scorer |
| `cadence` | code | 0 fixed_frame, 1 photon_counting_legacy | detector cadence mode the row was computed with | observer, policy, scorer |
| `calibration_id` | code | positive integer below 2^48 | six-byte digest of the detector, cadence, optics, response, nominal scene and law declarations | observer, policy, scorer |
| `valid_science` | 1 | flag: 1 science interval inside the certified window, 0 otherwise | whether the visit produced a certified science interval | observer, policy, scorer |

A row with `valid_science = 0` carries no counts and no limit; it is not a null
observation, and its occupied time is still charged.

## Error budgets

Reference noise at the plan (signal-to-noise 10, kappa 1) is about 2.9 mas per
axis and 10 percent in flux. The systematic error of each boundary is at most 10
percent of the statistical error it feeds; a sign or definition error is a
failure, never a budget line. Status `pinned` means the allocation is fixed
before its test exists; `blocked` names the library change the anchor waits on.

| Boundary | Observable | Unit | Sign | Measure | Allocation | Status |
|---|---|---|---|---|---|---|
| `geometry_position` | sky position (east, north) | arcsec | east and north positive | maximum difference over three eccentricities by 64 epochs; signs at four phases; east offset with and without cos(delta) | 0.29 mas total; Kepler solve 0.01 mas; phase angle 1e-9 rad | pinned |
| `geometry_contrast` | contrast Ag Phi(beta) (R / r)^2 | 1 | nonnegative | superior conjunction, quadrature, near inferior conjunction | 0.1 percent | pinned |
| `time_epoch` | record epoch and science midpoint | s | increasing with time | exact tick to JD round trip; opposite relabelling; JD to MJD; midpoint formula | 1 s | pinned |
| `time_keepout` | keepout window length | d per yr | nonnegative | orbix is_observable sampled every 6 hours over two years | 70 d per yr, plus or minus 2 d, for a 45 degree Sun keepout at the target's ecliptic latitude | pinned |
| `radiometry_planet_counts` | planet counts N | e- | nonnegative | constant planet; two bandwidths; time-averaged value on a moving planet | 1 percent | pinned |
| `radiometry_zodi` | local zodi count rate | e- s^-1 | nonnegative | dependence on ecliptic latitude and solar longitude across a year | 2 percent | pinned |
| `radiometry_exozodi` | exozodi count rate | e- s^-1 | nonnegative | separation sweep in AU; aperture sweep; two hosts of different luminosity | 2 percent | pinned |
| `radiometry_leakage` | stellar leakage in the aperture | e- s^-1 | nonnegative | native 0.25 and rebinned 0.5 sampling; diameter 0 versus 0.03 lambda/D; annular-bin control fails | 5 percent | pinned |
| `detector_variance` | noise variance and frame counts | e-^2 | nonnegative | zero-noise and each-source-alone cases; sub-frame, exact-frame, n frames and n frames plus a partial last frame | 2 percent on variance | pinned |
| `reporting_law` | detection frequency, retained counts, astrometric covariance | 1 | frequencies in [0, 1] | 20000 draws at three signal levels with s(theta) != s0; corr = 0 and corr = 0.5; wrong-threshold control fails | 3 Monte Carlo standard errors; Kolmogorov-Smirnov p above 0.01; whitened residual chi-square | pinned |
| `inference_calibration` | posterior calibration | 1 | probabilities in [0, 1] | calibration panel of 100 worlds | probability-integral-transform CDF inside the DKW band at 100 worlds (half-width 0.136); coverage inside a Bonferroni band | pinned |

## Exozodi level

`n_ez = 3`: the best-fit median habitable-zone dust level of Sun-like stars in
the LBTI HOSTS survey (Ertel et al. 2020, AJ 159, 177, abstract).

## Provisional entries

- The response origin rule (header centre N/2 is pixel-centre index N/2 - 0.5)
  is taken from the package headers and yippy's implementation; the text of the
  public yield standard has not yet been checked.
- The astrometric scale `kappa = 1` is uncalibrated for this response; the
  first campaign runs `kappa` in 1 and 2.
- The host star is a tabulated `Star`. skyscapes' `FlatStar` now carries an
  angular diameter, but it has no coordinates or luminosity (which the local zodi
  and exozodi read) and a flat spectrum in place of the blackbody table, so it does
  not replace the tabulated star.
