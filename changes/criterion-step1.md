# Declared characterization criteria and package planning

This record describes the completed interfaces and behavior. Compatibility and
coverage notes below retain the removals and serialized changes made during the
migration.

## Declared populations, bands and truth (breaking)

- World-source constructors (`ExoVistaWorldSource.from_files`, `from_draws`,
  `from_population`) and `star_spec_from_model` require `bands=`. A band is
  `Band(center_nm, width_nm)`; no wavelength is a library default. Stored spectra
  must cover every declared band. `PopulationTruth.albedo` has slot and band
  axes, with one geometric albedo for every planet and band.
- Host flux densities are band means, rather than center-wavelength samples. For
  `Band(1000, 100)` this is the 950-1050 nm mean.
- Every world-source constructor requires `generator`, a `PopulationModel`.
  Its occurrence grid, types and config supply draws; `declared_population()`
  returns that declaration. `PopulationModel.exovista()` explicitly constructs
  the ExoVista population with `SupportRange()`.
- Truth keeps every planet up to `n_slots`, ordered by descending periastron peak
  flux ratio, without a belief-support filter. `dropped` counts only planets past
  capacity. Extra slots use finite inert fillers. Separate scorer-only overflow
  parameters retain dropped planets for claim-specific loss accounting.
- `support`, `dropped_class1` and `occupancy` leave the world source. Truth has no
  stored class code; claims evaluate its named physical parameters. The former
  `population_ln_lam_range` helper and `ReportRefit(ln_lam_range=...)` are removed.
- Belief support comes from the declared population: `period_support_d` and
  `scale_range_au2(band)`, the padded range of log(albedo * radius_AU squared).
  Radius bounds intersect type bounds with the mass-radius relation's range on
  the occurrence grid. Unsupported truth is still observed and fit; decision-side
  slot status depends on the fit, never on a comparison with truth.

Over five drawn ExoVista worlds, 8240 planets were present. The old source dropped
5506: 5399 outside belief support and 107 past capacity. Of those dropped, 5044
lay beyond the 0.3404 arcsec outer working angle of a 500 nm mode and 11 were class
1. The maximum per-star count was 21. Capacity bounds truth storage; support
bounds inference.

## Named parameters, modes and measurement rows (breaking)

- `PlanetParameters` replaces positional orbit matrices above the photomancy
  conversion boundary. Period, time, masses, radius and all band albedos travel
  together, including through resampling and cached resummary. Mass and radius
  conversions use named physical constants; inverse mass conversion caps at
  131.6 Earth masses.
- Observing sources require mode specs, science durations and a profile. The
  immutable mode table owns channel order, bands, catalog channel and named
  resources. Reports use the declared throughput, frame time and resolving power.
  Imaging modes share a band and reporting threshold; incompatible modes or an
  explicit reporting law with a different threshold are refused.
- The stored fixture observes 1000/100 and 940/6.7 nm bands. Drawn-host tests cover
  the unmodified survey's 500/100 and 910/6.5 nm modes. No stored spectrum is
  extrapolated or relabeled.
- `MeasurementRow` replaces visit and block record kinds. Its codec declares
  exposure metadata, assignment, astrometry, counts and variance on detection and
  band axes. Imaging exposes detected counts in its band; spectral rows expose
  uncensored noisy counts. Measured variance uses reported information. Unexposed
  bands are zero. Generating counts, variance and identity are scorer-only unless
  an explicit grant opens them.
- Imaging entries contribute identity evidence only when detected. Spectral
  entries require finite counts and positive finite variance. Null padding cannot
  create a join or association error. Truth association is declared explicitly;
  inferred association remains deferred.
- Completed measurements and partial measurements with valid finite science use
  the same eligibility rule in incremental folds, host refolds and the JAX
  objective. Unusable partials still contribute occupied time.
- Measurement noise uses an observation identity: subject, scorer identity,
  band and that band's ordinal, excluding duration. Completed logged rows advance
  optional key-group counters; cancellations, duplicate or refused completions
  do not. Undeclared key groups preserve the previous core layout.

## Decision views, grants and inference

- Decision views withhold the world source, substitute the planning model and
  belief population, and retain the observer's declarations. A private opaque
  withheld-source marker identifies an existing view; a user source's public
  `withheld` attribute cannot bypass the boundary.
- Observer `Grants` replace coded arms and the separate policy-refit switch.
  Presets are `replan`, `oracle`, `prior_planned` and `shrinkage`. Grants, kernel
  and declared inputs participate in documents and their hashes. Result exports
  include a grants line and `grants.txt`.
- Every observer entry point receives its own audience-masked record. Refresh
  eligibility is evaluated separately for observer and policy, in execution and
  replay. Scorer-only columns cannot schedule decision-side refits.
- Detected-track fits draw the declared population's albedo mixture. Occurrence
  cells and eligible type weights define the mixture; one type supplies all
  bands. Host validation rejects selectable nonpositive or non-finite albedos.
  Shrinkage holds radius and masses fixed and uses the declared log-albedo widths.
- Dark-track first tiers weight whole population rows at their own physical
  scales by the null-report likelihood, then resample whole rows. Evidence is the
  mean likelihood over valid rows. Padding has zero weight, including in ESS and
  fit status. Empty support has zero existence and finite inert leaves.
- `draw_planets` appends fresh keyed pools of `n_pool_systems` hosts until the
  requested distinct size is reached, a pool is empty, or the bounded pool count
  is reached: 4 * max(1, ceil(requested / n_pool_systems)). Retry keys and larger
  requests produce fresh larger clouds. Solar-host tests require 4096 distinct
  first-tier rows and 16384 distinct retry rows with no reused rows between them.
- The expected supported count uses a reproducible estimator, keyed by host index
  and a fixed tag at initialization and all refits. Changing reports or retry
  keys changes conditional clouds while preserving that count estimate.
- Supported counts follow a Poisson model with that mean, conditioned on at least
  the detected-track count. Each dark slot receives
  E[min(N - detected, dark slots) | N >= detected] / dark slots before the existing
  null-evidence Bayes update. The prior remains below one, including numerical
  saturation handled by nextafter(1, 0). Zero mean and zero dark slots are guarded.
- Failed or unsupported fits retain prior slot leaves with the new status. A
  successful slot installs independently. Refits replace inference leaves while
  preserving accumulated criterion state.

## Criteria, forecasts and packages

- The dependency-light `criterion` family declares named state, updates, claims,
  labels, values and package options. One storage codec packs named measurement,
  catalog, criterion and policy layouts. A summary stores criterion state per
  star; released measurements fold it incrementally in each audience.
- Transition tables declare acyclic steps, allowed modes, requirements, margins,
  repeats, successors and achieved labels. Rows coadd measured counts and
  variance, with at most one pass per row. Active steps retain intermediate
  credit. Failure edges are validated; this family does not infer failure.
- The water/oxygen example explicitly assumes continuum SNR 5 for both steps
  pending emulated spectra. Its modes declare 940/760 nm, resolving power 140,
  throughput 0.29165, 300 s frames and widths center/resolving power. Its initial
  node has no achieved label; zero-valued labels remain legal answers.
- Refit forecasts carry paired count/variance quantiles by epoch and non-catalog
  channel, plus claim probabilities and availability. Quantiles and dimensions
  are configurable. Availability is all true with the current source model.
- Package gain is incremental label credit weighted by reach probabilities and
  target claims. Costs use charged resource vectors including overheads. The
  first exposure is shared by the star; later slot costs are weighted by reach.
- Imaging requirements advance on the exposure that satisfies them. Candidate
  images are required before spectroscopy, with declared detection-count,
  status and position-scatter guards; passing imaging steps reduce that gate.
  A completing image is charged once. Its progress probability is detection
  probability times the conditional requirement-pass probability; nondetections
  deliver no counts or variance and earn no progress or continuation credit.
- Packages receive remaining imaging/spectrum row allowances at both star and
  track interfaces. Tracks compose in their existing order, consuming one shared
  per-star allowance. The shared first exposure consumes one row for all tracks;
  every continuation consumes its kind's remaining allowance. Unreachable final
  credit is excluded while reachable intermediate credit is retained.
- Threshold pricing keeps strictly positive finite gain and net value, with none
  winning zero-net ties. Fixed resource-price sweeps end with deterministic joint
  feasibility pruning. Dispatch ranks by the named telescope-days column and
  starts only the current node. Capacity is the lesser of visible ledger balance
  and remaining days.

## Policy state and diagnostics

- `PackagePolicy` replaces rule-order selection and the cheapest-visit fallback.
  It retains catalog kickoff, declared row caps and horizon drain. Both audiences
  receive population forecasts at zero time, including without measurement grants.
- Named policy state stores row-request counts, requested completion ticks,
  kickoff, stop, bounded trace and final per-track selection evidence. The final
  evidence includes its tick and survives stale waits and horizon stop.
- Stale-summary waiting compares current released policy-visible imaging rows
  with the summary's folded count. A refused start with no row cannot stall the
  policy. An independent completion-tick guard prevents overlapping exposures.
- Each fresh decision writes six trace columns: price, ledger net charged days,
  selected package days, open-opportunity proxy, acted, and `cum_value`.
  Charged days are initial budget minus the visible ledger at snapshot clock
  precision, before the new request is admitted; refunds reduce net spending.
- `cum_value` sums the declared values of the current policy-visible track labels.
  Failed or unfinished actions add no forecast credit; completed labels count
  once. The name and six-column export format remain unchanged.
- The opportunity proxy is the largest finite positive-gain package time, not a
  measured duration of continuous visibility windows. `stranded_d` uses final
  unspent ledger balance when any written refresh recorded opportunity. A richer
  window-duration definition remains unresolved.
- `early_overspend` counts every measurement-package NO_FUNDS refusal, excluding
  kickoff, waits and padding. Trace extraction returns padded rows with written
  lengths; scalar calculations and CSV exports consume only valid prefixes.
- Capacity accounts for daily waits, action refreshes and forecast renewal;
  exhausted trace capacity raises an explicit error. Named row-capacity defaults
  retain MAX_VISITS=24 and MAX_SPECTRA=12, with log/count/ID capacity floors.

## Verdicts, activity and compatibility

- The scorer joins actual participating observations to catalog identity, credits
  each identity once, and reports mixed associations, duplicates, missed planets,
  mission/scorer labels, yield and wrong answers. Correctness is independent of a
  label's utility, so a correct zero-value negative is not wrong.
- Open-track reasons are `no_package` or `below_price` from the retained final
  selection; scorer-only `unsupported` overrides them for true period,
  eccentricity or detection-band scale outside belief support.
- `activity_table` and `run_grants` replace `outcome_table` and `run_arms`.
  Activity includes occupied days by observing kind, visits, dropped planets,
  refusals, ledger charges and degraded slots. Dropped claim counts are exported
  as `dropped_claim_<name>`. Source-order catalogs are validated before attributing
  dropped counts; unsupported source subsets or permutations are refused.
- Per-effect `trace.eff_subject` attributes refusals to the requested subject.
  Adding that provenance leaf changed checkpoint trace schemas and the 116
  integer history digests, four fault histories and 14 compile-sharing raw hashes.
  Excluding the leaf reproduces the earlier hashes. Float histories, snapshot
  fingerprints and `tests/codec/hash_pin.json` remain unchanged.
- Study capacities omit derived payload paths from JSON. `load_study(registry=)`
  and run loading restore and validate those paths through the record registry.
  Document content and pinned document hashes remain unchanged.
- External criteria register through `BRACKET_KINDS`. Step-1 ladders require a
  common concrete family, packed width and ordered claim schema; each entry
  initializes its own state. Requirements and transition values remain traced.
  Cross-criterion yield uses chronological scorer refolds; foreign criteria
  require an observed allowed mode for every step, while native empty flights
  retain their defined verdict.
- With packed state width 36, summary memory at six slots, sixteen nodes,
  thirty-six variants, three quantiles, one claim and three bands is 169600 bytes
  per star per audience,
  below the 256 KiB ceiling. For 150 stars, both audiences use 50880000 bytes,
  excluding logs, executable memory and refit workspaces.
- Removed APIs include the shadow characterization state, block-SNR channel
  requirements and reveal/class plumbing, `n_char`, retirement/reserves, coded
  arms, positional planet vectors, fixed class-conditioned first tiers and the
  unused fixed-threshold `candidate_slots`. Their operational checks now exercise
  criterion folds, packages, audience grants, verdicts and slot installation.
## Retired coverage and replacement guards

Twenty tests pinning removed block rules are retired; two forecast-renewal tests
move without losing their guards, and one lightweight summary/channel API guard
is added (net nineteen fewer test functions). No parameter matrix is expanded.

- Retry, retirement and spectrum-count tests removed from
  `test_bracket_characterization.py`:
  `test_a_block_that_fell_short_is_retried_at_a_longer_mode`,
  `test_three_failures_in_a_row_retire_the_slot`,
  `test_n_char_two_retires_after_failures_that_follow_a_success`,
  `test_n_char_two_needs_two_delivered_spectra`, and
  `test_a_slot_is_confirmed_at_the_configured_n_char`.
  Package repeat/pass-probability and criterion-verdict properties replace the
  old spectrum-count and confirmation guards; consecutive-failure retirement is
  intentionally removed.
- Selection/reserve tests removed from that file:
  `test_the_block_and_follow_up_thresholds_are_knobs`,
  `test_the_reserve_blocks_a_search_that_would_spend_what_an_open_track_needs`,
  `test_a_slot_no_longer_mode_serves_is_retired_and_holds_no_reserve`,
  `test_the_block_log_maps_every_block_row_to_its_slot`,
  `test_a_block_the_budget_cannot_cover_is_not_run_as_a_visit`, and
  `test_a_mode_of_infinite_time_is_never_chosen`.
  Package selection restores unreachable/infinite-time rejection and ordering of
  finishing packages; ledger/day capacity and admission properties restore the
  no-overspend guard. The heuristic reserve, block log and block threshold knob
  are intentionally removed. The combined test's follow-up threshold guard returns
  with the replacement policy's decision harness.
- `test_bracket_block_claim.py` is deleted after moving
  `test_a_star_past_its_forecast_window_is_refit_without_new_rows` and
  `test_the_observer_expires_a_summary_at_its_last_node` to
  `test_bracket_renewal.py`. Its six removed tests are
  `test_a_blocks_class_column_is_the_only_one_an_observer_may_read`,
  `test_a_confirmed_slot_is_class_one_and_only_a_delivered_block_reveals`,
  `test_a_reveal_overrides_the_class_probability_even_on_a_kept_slot`,
  `test_a_candidate_is_blocked_and_a_revealed_slot_is_not`,
  `test_a_revealed_slot_is_not_followed_up`, and
  `test_a_candidate_takes_the_shortest_block_mode_that_can_succeed`.
  Truth reveals and their class-probability override are intentionally removed;
  criterion verdicts and package feasibility/ordering replace their scientific
  and scheduling guards.
- Also retired: `test_only_a_revealed_class_one_slot_is_confirmed` from
  `test_bracket_arms.py`, `test_a_block_that_does_not_fit_is_not_started` from
  `test_bracket_outcomes.py`, and `test_a_star_that_got_a_block_is_visited_again`
  from `test_bracket_stale.py`. Criterion verdicts replace confirmation, package
  admission covers funds, and imaging refit-wait tests retain the liveness guard.
  The duplicated bracket `_decide` harnesses and block-only helpers are gone.
