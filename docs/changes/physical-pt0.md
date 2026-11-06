---
orphan: true
---

# Physical PT0 (code half): profile table, pins and the profile-table test

PT0 fixes the first physical profile, `first-summary-v1`, as data. The table ships
with the package as `src/spaceodyssey/adapters/first_summary_v1.json` and is read by
`src/spaceodyssey/adapters/physical_profile.py`; the readable page is the draft
`docs/profiles/first-summary-v1.md`; the fixture manifest is
`tests/anchors/fixture_manifest.json`. No file under `core/` or `codec/` changed and no
golden history moved.

## What is in

- The profile table: the conventions (sky basis, east offset, node, position angle,
  inclination, phase angle, the radial-velocity argument of periastron, time scale,
  tick, intervals, record epoch, certification, keepout, response origin, leakage
  operator, dQE, background multiplier, systematic floor, thermal, exozodi, detection,
  cadence), the time block, the keepout block, the response pin, the two targets with
  sources, the fifteen payload columns and the eleven error-budget boundaries.
- `load_profile(label)`, `check_profile(profile)` (the KP5 check: every payload column
  and boundary has a unit, a sign and a measure, every column a nonempty audience of
  known names, every boundary a status of `pinned` or `blocked: <library item>`, no
  repeated names, no empty lists), `audience_flags(names)` (names to the core
  `Audience` flags) and `payload_column_names(profile)`.
- The fixture manifest: the 40-hex revision and a version floor of each consumed
  library (orbix, photomancy, skyscapes, hwoutils, yippy, and jaxedith and optixstuff
  for the later tasks), the response pin, and the catalog query the host values came
  from.
- `pyproject.toml`: a `physical` extra (`skyscapes>=2.1`, `orbix>=3.0`,
  `numpyro>=0.21.0`); the `test` extra now pulls `spaceodyssey[physical,viz]`, `yippy`,
  `astropy` and `mpmath`; two warning filters for the ExoJAX import that skyscapes
  scenes trigger (the unclosed `/dev/null` stream and invalid escapes on a cold import),
  the same filters skyscapes carries.

## Pinned inputs and their sources

| Input | Value | Source |
|---|---|---|
| Response package | `eac1_aavc_2d`, release `data-v2`, `md5:1f4892faff18e55cbec9781a055bea4d` | yippy's catalog; the fetched archive hashes to it and its header gives D 7.2 m, OBSCURED 0.17481001240599225, band 0.95 to 1.05 micron, PIXSCALE 0.25, 256 pixels, centre 128 |
| 47 UMa host | 1.07 Msun, 13.8875 pc, RA 164.8665531 deg, Dec 40.4302557 deg, log L 0.1978, 5880 K, 1.21 Rsun, 0.81 mas | ExEP Mission Star List for HWO (Mamajek and Stapelfeldt 2024), row HIP 53721 from the NASA Exoplanet Archive table `di_stars_exep`, retrieved 2026-11-06 |
| 47 UMa b | P 1078 +- 2 d, e 0.032, stellar omega 334 deg, periastron JD 2451917, M sin i 2.53 MJup, K 48.4 m/s | Gregory and Fischer (2010), MNRAS 403, 731, Table 6 (arXiv:1003.5549 source read and checked) |
| n_ez | 3 | Ertel et al. (2020), AJ 159, 177, abstract: best-fit median 3 zodis for Sun-like stars |
| Study epoch | JD 2462502.5 (2030-01-01 TDB) | declared fixture choice |

Declared, not measured: inclination 60 deg (the median of an isotropic prior), node 0
deg, radius 71492 km, albedo 0.5; the second target is a synthetic solar twin at 10 pc
on the ecliptic with an Earth twin.

## Tests

`tests/anchors/test_profile_table.py`, 11 tests (10 unit, 1 medium):

- the shipped table passes `check_profile`, lists the record columns in record order
  and gives each a scorer-readable audience;
- six mutated tables (missing unit, blank sign, unknown audience, blocked without an
  item, repeated name, empty column list) are each named by `check_profile`;
- the page's payload-column and error-budget tables equal the shipped table cell by
  cell;
- the page has no em or en dashes and no contractions;
- the manifest pins every consumed library with a full revision and a floor the
  installed version meets, and its response pin equals the profile's and yippy's
  catalog entry;
- medium: the archive fetched through yippy's loader hashes to the pinned digest and
  its header equals the profile's response block (it downloads the 78 MB archive on a
  cold cache, hence medium).

Mutants: a `check_profile` that stops checking the column sign, and a page cell that
drifts from the table, are both killed.

## Where the spec was silent

- The machine table lives in the package (the record kind of the reporting-law task
  reads its audiences), not under `tests/anchors/`; the manifest lives under
  `tests/anchors/`.
- Revisions in the manifest are provenance, not enforced: installed metadata can be
  stale for editable installs, and CI installs releases. The floors are enforced.
- photomancy's floor is `0.4.1.dev0`: the prior-only model with `sample_planet_scale`
  is on photomancy main after 0.4.0 and not yet released.
- Absent payload values (counts, position and errors on a null) are 0 and unread, not
  NaN, so the payload stays finite.
- The response origin rule is marked provisional on the page: the public yield
  standard's text was not checked.
