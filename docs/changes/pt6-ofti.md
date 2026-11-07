---
orphan: true
---

# Physical track PT6: OFTI stand-in refit

PT6 refits physical summary records with photomancy's OFTI rejection sampler through
the merged `OftiRefit`, as a first-demonstration stand-in. The code is
`src/spaceodyssey/studies/physical.py`; the tests are
`tests/studies/test_physical_ofti.py`, and `tests/adapters/test_photomancy_ofti.py` now
uses the promoted particle observer. No file under `src/spaceodyssey/core/` or `codec/`
changed, no `pyproject.toml` entry changed, and no golden pin moved. No core change was
needed.

## Commits

On main `97753d5` (PT0 to PT4 merged), branch `pt6-ofti`:

1. `feat(studies): OFTI stand-in refit of physical summary rows with particle observer, fixed plan and scripted rows` (`c161a4f`)
2. `test(studies): OFTI stand-in anchors, one real fit, contraction survey and both executors` (`e71495e`)
3. `docs(changes): physical PT6 OFTI stand-in refit record` (`cfa6764`)
4. `fix(studies): readable stand-in scope in saved run documents and a medium executor-parity guard`
5. this record's review-fix update

## Scope

The stand-in is an approximation with a declared domain. `STANDIN_SCOPE` states it as
plain JSON, and it is carried in the refit's model declaration (`support.scope`), so in
its model hash, and readable in the refit's configuration (`scope`), so in words in every
run document that binds it, with the scripted rows' reporting law (threshold 5, kappa 7,
correlation 0, planned signal-to-noise 10, limit 10 e-). The module and class docstrings
state the same.

- Conditional on detection: only rows with `detected = 1` and `valid_science = 1` count
  as astrometry.
- Conditional on OFTI's induced prior: the semimajor axis and node follow from
  scale-and-rotate through the reference measurement. This is not the prior
  `PhysicalWorldSource` draws truth from.
- Instantaneous astrometry at the science-interval midpoint (the record epoch), not
  integrated over the exposure.
- Informative nulls are ignored: an entry with no detection row is `NO_SUPPORT` and
  leaves the belief unchanged.
- Domain: loose astrometry. Measured on the nominal 47 UMa b orbit at the fixed plan's
  epochs with the profile's period support (1072 to 1084 d) and the strict bound, 64
  particles wanted: at 2.9 mas per axis (kappa 1 at signal-to-noise 10) OFTI accepted
  39 of 2,000,000 proposals with two epochs and 2 of 2,000,000 with three, so it cannot
  fill a particle set; at 10 mas it filled three epochs and stopped at five (rate
  1.8e-5); at 20 mas it filled all five in about 4 s. The scripted rows therefore use
  `STANDIN_KAPPA = 7` (20.0 mas at signal-to-noise 10). PT7 carries the real noise level.

## What is in

- `ParticleObserver(refit_kind, n_particles)`: the test's `OftiObserver`, promoted. It
  holds `[n_subjects, n_particles, 7]` particles in `PARAM_NAMES` order and an epoch
  count `n`, changes only by refresh and refits `summary_measurement` rows by default.
  `test_photomancy_ofti.py` subclasses it with `refit_kind="astrometry"`.
- `StandInOftiRefit(time_origin_jd, **OftiRefit keywords)`: an `OftiRefit` whose columns
  are the summary record's (`time_d`, `ra`, `dec`, `ra_err`, `dec_err`, `corr`). Before
  the fit, rows that are not detections with science are marked not counting
  (`astrometric_rows`) and the Julian Date epochs become days from `time_origin_jd`, the
  origin of the particles' `tp` (the study epoch, as for `PhysicalWorldSource` and
  `SummaryReport`). The audience check also covers the two flag columns. `count`
  defaults to `n`; passing `columns` is a `TypeError`. `from_profile(profile, target)`
  takes the host mass and distance, the eccentricity prior and the period support from
  the profile and the time origin from the study epoch. `standin_model` restates OFTI's
  model with the scope and the time origin; the model hash differs from plain OFTI's.
  `standin_binding()` names it `spaceodyssey.ofti_standin` in run documents, built from
  `configuration()` (the time origin and the readable `scope` included, `columns` left
  out). Rebuilding refuses a document whose scope differs from `STANDIN_SCOPE`.
- `fixed_plan(context)`: five `PlannedVisit`s of 47 UMa b with occupancy starts 178, 298,
  690, 810 and 1256 days after the study epoch, one day of settling and one 1000 s frame.
  These are the first five visits the PT4 planner chose from the nominal orbit
  (outside keepout for the whole occupancy, nominal planet between IWA + 0.5 lambda/D
  and the field edge less 1 lambda/D, 120-day steps). The plan certifies against keepout
  (`require_certified`).
- `nominal_orbit(profile)`: the target's elements in OFTI's fixed-mass parameterization
  (the semimajor axis from the period and the stellar mass alone).
- `scripted_rows(context, visits, orbit, key=None, ...)`: one summary payload per visit,
  through PT4's `SummaryRow`. The epoch is `science_midpoint_jd(start, end)`, the
  position photomancy's instantaneous relative astrometry there, the expected counts
  give signal-to-noise 10 against a planet-free noise of 10 e-. Without a key the rows
  are noiseless detections; with a key each row is `sample_report` under
  `ReportingLaw(kappa=7)` from a scripted `CountBudget`. The calibration id is a six-byte
  digest of the scripted declarations.
- `PhysicalScript` and `physical_study(rows, context)`: scripted records written at the
  end of each occupancy with the occupancy start as `start_tick`, on a 1 ms clock from the
  study epoch, a silent policy and a one-day refresh grid. `physical_registry(profile)` is
  the toy kinds, the refresh product kind and the summary kind (index 4).

## Tests

`tests/studies/test_physical_ofti.py`, six unit tests (eight cases), two medium, two
slow. Cold times are each test alone with the compile cache off (setup plus call; for
the review-fix tests, the whole process with imports in brackets).

| Test | Tier | Cold |
|---|---|---|
| noiseless physical rows hand OFTI the midpoints and the true orbit | unit | 1.10 s |
| null and no-science rows never count as astrometry | unit | 0.72 s |
| an audience blind to a column the stand-in reads gets no rows (3 cases) | unit | 0.01 s |
| the scope is declared in the model hash and the document configuration | unit | 0.02 s |
| a saved run document states the scope in words | unit | 0.86 s (5.6 s) |
| one real fit reproduces the orbit and equals OFTI on the T4d table | medium | 4.54 s |
| executor parity on one certified visit | medium | 18.6 s (24.6 s) |
| survey: posterior contraction over five scripted visits | slow | 96.6 s (call) |
| five refreshes install particles on both executors | slow | 90.6 s (call) |

Anchors:

- Noiseless rows of the T4d orbit (a 1.2 AU, e 0.1, cos i 0.6, node 0.8, omega 0.5,
  tp 3 d) around the profile's solar-mass host at 10 pc, with visits placed so the
  science midpoints fall on the T4d epochs 0, 12 and 30 days in integer ticks. The table
  OFTI receives has exactly those epochs, the forward-model positions to 1e-15 arcsec,
  and errors `kappa (lambda/D) / snr` with lambda/D written out by hand.
- Two extra rows copy a detection's position, errors and epoch: a null and a no-science
  detection. Neither reaches OFTI and the epoch count is three; an entry with only those
  two is `NO_SUPPORT` with no OFTI call.
- The real fit (medium) of the three detections plus the two extra rows is bit-identical
  to plain `OftiRefit` on the three-row table in days under the same key, and every
  particle predicts every row within 5 sigma.

Mutants (applied to a copy of the tree, each run against its anchor, each failing):

| Mutant | Caught by |
|---|---|
| record epoch at the science start | midpoint anchor |
| record epoch at the occupancy midpoint | midpoint anchor |
| time origin not subtracted | midpoint anchor |
| position computed half a day off the epoch | midpoint anchor |
| null rows counted | row anchor; the medium real fit alone |
| no-science rows counted | row anchor; the medium real fit alone |
| flag columns left out of the audience check | blind-audience test |
| scope left out of the model declaration | scope test |
| scope left out of the saved configuration | saved-document test |
| scripted kappa dropped from the saved scope | saved-document test |
| refit keyed by a call counter, so the host and compiled fits differ | medium parity test |

## Survey: contraction over the five scripted visits (slow)

This survey is a deterministic single-seed regression (one row draw, one refit key per
visit count), not a statistical campaign: it shows contraction on one realization and
pins it against regressions. It stays in the slow tier for its cost (about 100 s, most
of it the five-epoch fit).

Rows: the certified fixed plan, drawn with key 2026 under `ReportingLaw(kappa=7)` at
signal-to-noise 10; all five visits are detections (per-axis errors 20.1 to 21.1
mas, observed signal-to-noise 9.5 to 10.0).
After visit k the stand-in refits the first k rows (64 particles, one key, the
profile's period support). Spread is the root mean square, over 32 epochs spanning one
period after the last visit, of the particles' predicted (east, north) scatter.

| Visits | Spread (mas) | std e | std cos i |
|---|---|---|---|
| 1 | 97.2 | 0.202 | 0.650 |
| 2 | 72.1 | 0.202 | 0.307 |
| 3 | 37.2 | 0.092 | 0.178 |
| 4 | 27.8 | 0.055 | 0.151 |
| 5 | 26.0 | 0.038 | 0.157 |

The spread after five visits is 0.27 of the spread after one (asserted below 0.4), and
no visit widens it by more than 15 percent. The true track stays within 4 predictive
standard deviations at every epoch after every visit. The semimajor axis is pinned by
the narrow period support from the first visit; the inclination stops contracting after
four visits at this noise. The fifth five-epoch fit is the costly one: the acceptance
falls with each epoch, and a 128-particle trial took 109 s for that fit alone.

## Done criterion (slow)

The fixed plan certifies against keepout; its scripted rows (the survey's) run in
`physical_study` with the stand-in for 47 UMa b, 16 particles, 25 steps.

- Host executor: ten product rows, all `OK`, with row counts 1, 1, 2, 2, 3, 3, 4, 4, 5,
  5 (observer and policy at each of the five checkpoints); the final belief holds five
  epochs.
- Every summary record's epoch equals the science midpoint of its logged start and end
  ticks within half a tick (KP9 on the scripted rows).
- Compiled executor (chunked driver, host boundary): five epochs and particles
  bit-identical to the host run.

Five refreshes install particles on both executors, the PT6 done criterion.

## Spec silence and deviations

1. `PlanTablePolicy` is not in the PT6 study. With no channel there is no acquisition
   for it to admit, and the policy only acts when a plan row is due at a wake, which the
   scripted records do not provide. The fixed plan is a tuple of `PlannedVisit`s,
   certified by the context; PT10 lowers it into a `PlanTablePolicy` over the summary
   channel.
2. The scripted rows use kappa 7, not the profile's kappa 1, for the domain reason
   above. The stand-in itself does not depend on kappa.
3. OFTI's mass model is fixed at the stellar mass, so `nominal_orbit` takes the
   semimajor axis from the period and the stellar mass alone; the skyscapes scene adds
   the planet mass. The difference is inside the stand-in's approximation and does not
   enter any anchor.
4. The contraction survey refits each prefix of the rows directly through the stand-in
   (one fit per visit count), not from inside an executor run: a run installs only its
   latest particles. The executors are exercised by the done-criterion test.
5. The particle observer is parameterized by refit kind so the T4d test keeps its
   astrometry kind; its run documents name the observer class as before.

## Review fixes

1. Saved outputs carried only the constructor settings and the model hash. The
   configuration now holds the readable `STANDIN_SCOPE`, which also states the domain
   (about 20 mas, kappa 7, not the profile's kappa 1) and the scripted reporting law.
   `test_a_saved_run_document_states_the_scope_in_words` dumps a physical study bound to
   the stand-in, checks the serialized JSON for the induced prior, the midpoint
   astrometry, the ignored nulls, the loose domain and kappa 7, checks every `scope`
   object in it equals `STANDIN_SCOPE`, and reloads it to the same model hash.
2. Executor parity had no merge-time guard. `check_executor_parity` (host and compiled
   runs: `OK` products with counts for both audiences, record epochs at the science
   midpoints on both executors, bit-identical particles, epoch counts and policy
   summaries) is shared by the slow five-visit demonstration and the new medium
   `test_executor_parity_on_one_certified_visit` (the first visit of the plan,
   certified, four particles, batches of 4,096, eight steps). A seeded parity mutant (a
   refit keyed by a call counter) fails it at the particle comparison.

After the fixes, one launcher run (`-n 2 --dist loadfile`) of the whole PT6 file slow
tests included, the T4d tests and the golden suite (golden histories, run-document
runs, corpus): 246 passed in 494 s, peak footprint 8.7 GiB; `golden_pin.json`,
`golden_histories.json` and `hash_pin.json` unchanged. The quick-tier physical addition
is now two medium tests (4.5 s and 18.6 s cold) and six unit tests (under 3 s
together), about 26 s inside the track's 30 s share.

## Builder self-review

- Golden suite (`tests/core/test_golden_histories.py`, `tests/codec/test_run_document_runs.py`,
  `tests/core/test_corpus.py`, `-n 2 --dist loadfile`, compute launcher): 223 passed in
  421 s; `golden_pin.json`, `golden_histories.json` and `hash_pin.json` unchanged.
- Quick tier (`-m "not slow"`, `-n 2 --dist loadfile`, entry-point install test
  deselected): 1,972 passed in 260 s, peak footprint 6.4 GiB. The budget report names
  three existing unit tests over 5 s under the shared load (snapshot fingerprint 9.6 s,
  wake coalescing 5.6 s, detector-spec refusals 6.0 s); none is touched here.
- Budget: the PT6 quick-tier addition is the one medium test (4.5 s cold) and five unit
  tests (under 2 s together), inside the physical track's 30 s share.
- The T4d tests (`test_photomancy_ofti.py`, `test_survey_2_ofti.py`) pass with the
  promoted observer.
- ruff, the internal-reference check and the pre-commit hooks are clean; the Python is
  ASCII.
- Tests added 11 (6 unit, 8 cases with the parametrized one; 2 medium; 2 slow), retired 0.
  The T4d test lost its local observer and belief classes.
