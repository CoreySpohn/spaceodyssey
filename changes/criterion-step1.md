# Declared bands, a declared population and a truth that keeps every planet

## Declared bands

- Every world-source constructor (`ExoVistaWorldSource.from_files`, `from_draws`,
  `from_population`) and `star_spec_from_model` take a required `bands=`, a tuple of
  `Band(center_nm, width_nm)`. No wavelength is a default. `PopulationTruth.albedo`
  carries one geometric albedo per declared band (`[..., n_slots, n_bands]`), and
  `from_files` refuses a band its stored spectra do not cover.
- A channel built with the default profile now reads `flux_density_jy` as the mean over
  the band (950 to 1050 nm for a `Band(1000, 100)`), where it was a single-wavelength
  sample at the band center.

## The truth keeps every planet up to its capacity (breaking)

- `ExoVistaWorldSource` no longer filters planets by a support range. Each star keeps
  its planets in descending peak flux ratio up to `n_slots`, whatever their period or
  eccentricity. `dropped` counts only the planets past `n_slots`. A capacity above the
  number of drawn planets fills the extra slots with the inert filler.
- `support`, `dropped_class1` and `occupancy` are no longer attributes of the world
  source. The class-1 planets in slots are `population.klass.sum(-1)`.
- Every constructor takes a required keyword `generator`, a `PopulationModel` (no
  default). `from_draws` draws from its occurrence grid, types and config, which replace
  the `config` argument; `from_population` reads the albedo from its types, which replace
  the `types` argument. `declared_population()` returns it.
- `PopulationModel(occurrence, types, config, support)` is the declared population.
  `PopulationModel.exovista()` is the ExoVista generator with `SupportRange()`.
  `scale_range_au2(band)` is the range of `ln(albedo * radius^2)` in AU^2 the population
  can draw in a band, padded by `LN_LAM_PAD`, and `period_support_d` is the period range
  of its `SupportRange`.
- `ReportRefit` reads its scale prior and period range from the belief population of the
  study's decision view (`scale_range_au2(detection_band)` and `period_support_d`) when it
  refits. Cached fits are keyed by those bounds. A study without a declared belief
  population uses its world source's
  `generator`. `photomancy.population_ln_lam_range` and the `ln_lam_range` argument of
  `ReportRefit` are removed, and `ln_lam_range` leaves `ReportRefit.configuration()`.
  `bracket_boundary` no longer reads the world source, and `bracket_study` takes
  `belief_population=` and `planning_model=`.

### Why the capacity holds every planet

Over five drawn ExoVista worlds there were 8240 planets. The old source dropped 5506 of
them: 5399 for lying outside the support range and 107 past `n_slots`. Of the dropped
planets, 5044 lay beyond the 0.3404 arcsec outer working angle of a 500 nm mode, and 11
were class 1. The largest number of planets on one star was 21. Filtering by the belief's
support removed real planets from the truth, so a belief was never scored against a
planet it cannot represent. The capacity alone now bounds the truth, and the belief's
support bounds only the refit's prior.

## Imaging-only bracket policy and summary (breaking)

- `DecisionSummary` and its slot leaves no longer carry `p_block`, `revealed`,
  `n_spectra` or `n_blocks`. Summary construction no longer takes a block-mode
  count. Refits no longer read block rows or override class probabilities from
  revealed truth. Forecast renewal and replay remain covered in
  `test_bracket_renewal.py`.
- `ObservingModeChannel` removes `block_snr_required`, `block_snr`, `block_time`
  and `reveal_columns`; `characterize_time(subject, snr_required)` requires the
  caller's SNR requirement. Block rows temporarily retain accumulated SNR and
  observable fraction, with every column scorer-only and no class reveal.
- `BracketPolicy` keeps catalog kickoff, imaging follow-up and search. Block
  selection, retries, retirement, characterization counts, reserves and the block
  log are removed. `pstate_size(n_subjects)` now holds three values per star
  (`last_tick`, `n_vis`, `n_search`) followed by `charged`, `kick`, `stopped`.
  Capacity bounds hold one catalog row plus `max_visits` counting rows per star,
  with refit and forecast-renewal products counted separately.
- `outcome_table` removes the slot keys `revealed`, `claimed`, `claim_error_lod`,
  `block_attempted`, `block_attempts`, `n_spectra`, `block_delivered`, `block_snr`
  and `block_days`; the star keys `n_block` and `t_block`; and the world keys
  `confirmed`, `false_claims`, `wasted_blocks`, `wasted_days` and `delivered`.
  Their aggregation code and the `n_char` table/paired-run argument are removed.
  Imaging endpoints retain their meanings, including observer-summary `candidate`.

### Retired coverage and replacement guards

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
- The observer-versus-policy outcome test remains: it distinguishes the two
  summaries through imaging candidacy, which reads the observer's `n_det`, class
  probability, status and scatter, independently of the raw row-log detections.
- The shrinkage arm's paired gap now uses `charged`, the telescope days its imaging
  decisions spend: the fixture oracle spent 1.11574074 days per world while the
  shrinkage arm spent zero. This preserves a contrast that can differ between the
  arms without manufacturing a confirmation endpoint.
- Cold timing of the affected non-slow rule files before removal: 18 passing tests
  in 38.55 s, sampled peak footprint 1937.3 MiB. Their capacity/renewal replacements
  after removal: four passing tests in 5.24 s, sampled peak footprint 806.9 MiB.
  The standalone lightweight summary/expiration checks pass in 3.31 s. These
  timings describe the explicitly changed coverage, not a speedup at equal coverage.
- Verification: 1766 quick-tier tests pass in 132.79 s with zero test-budget
  violations; the focused slow renewal/replay, refit-cache and paired-arm checks
  also pass. Per-file program budgets are regenerated from measured compilations
  and cache loads with the standard headroom. No golden file changes.

## Declared row capacities (breaking)

- `default_capacities` requires `horizon_d` and accepts `max_spectra`, defaulting
  to `MAX_SPECTRA = 12`. Each star's counting rows include one catalog row, up to
  `MAX_VISITS = 24` imaging rows and the declared spectrum rows. Imaging rows
  and forecast renewals each add `REFIT_ROWS` log products; spectra add none.
  The 1024 log-row, 32 counting-row and 4096 ID values remain floors. Clock
  horizons that cannot fit still fail at construction.
- `bracket_study` sizes rows from the policy's imaging and spectrum limits, with
  the declared limits as fallbacks. The survey channel-table test supplies its
  horizon explicitly.
- `test_bracket_capacities.py` replaces `test_bracket_characterization.py`.
  Three tests (five collected cases) replace the two capacity tests: hand-pinned
  small and survey bounds, sensitivity to spectrum limits and the required
  horizon, and clock refusal with recovery on a longer tick. Net: one test
  function and three collected cases added. No shared parameter matrix expands.


## Named physical planet parameters and catalog rows (breaking)

- `PlanetParameters` replaces unnamed orbit matrices above the photomancy boundary.
  Periastron time is days from the study epoch; stellar and planet masses, physical
  radius, and all declared-band albedos travel together through a cloud. The
  photomancy conversion preserves its AU-squared scale and caps the inverse mass
  relation at 131.6 Earth masses.
- `TruthCatalogRow` is packed from named fields. The catalog kind and audience
  groups come from its template; lowering preserves each kind's own column paths.
  Refit validation uses those paths, excluding padding from wider kinds. Run
  documents omit these derived paths and rebuild them from the stored registry;
  pinned documents and goldens do not change.
- `ObservingModeChannel.from_stars` requires `bands=` and exposes `n_bands`.
  Study construction refuses channel/source band disagreement. Detected-slot fit
  and shrinkage clouds draw albedo spectra with a fixed folded slot key from the
  decision view's belief population, retaining the association during cached
  resummary. Cache identity includes the population and band declarations.
- The albedo mixture weights positive-rate occurrence cells, draws log-uniform
  mass and scaled semimajor axis within each cell, then normalizes eligible type
  weights including the configured candidate probability. It rejects locations
  without selectable types. This is the occurrence prior before stability and
  host/support/observation conditioning; one type supplies every band of a draw.
  Source and refit host validation reject nonpositive or non-finite albedos of
  selectable types in any declared band and require an explicit population repair.
- The first tier retains its interim class-1 prior, scale window, fixed albedo,
  semimajor-axis convention, pool, filter, evidence and search-occupancy law.
  Shrinkage holds radius and masses fixed, with log-period width 1.5 times the
  declared log-albedo width.
- Fourteen test functions are added, including seventeen cases in the parameter
  file, one lowering guard and one codec round-trip guard. Two observing-mode
  functions retire into the parameter tests: catalog layout and stored positions.
  The independent raw-file/exact-solver and tabulated-solver position guards
  remain. Net: twelve functions and seventeen collected cases added; no shared
  scenario matrix expands. Cache renewal pins the albedos bitwise and permits at
  most four ULP in recomputed positions across the two compiled summary programs.

## Configured catalog permission validation

- `ReportRefit.check` reads column paths and audience permissions from the
  configured `catalog_kind`. A renamed catalog works without the standard kind;
  when both are registered, only the selected catalog's permissions determine
  whether construction is allowed. Padding from wider kinds remains excluded.
- One regression function adds three cases: a renamed catalog alone, a hidden
  selected catalog beside an open standard catalog, and an open selected catalog
  beside a hidden standard catalog. No tests retire or shared matrices expand.

## Population first tiers and physical claims (breaking)

- `AbstractClaim` and `EarthCandidateClaim` evaluate named physical parameters.
  Weighted claim means become `DecisionSummary.p_claim [n, S, C]`; an empty
  claim tuple has an empty last axis and values any planet. The scorer evaluates
  claims on truth with `claims_on_truth`, masking absent slots. Population truth
  carries no stored class code; outcome tables expose the evaluated `real` flag.
- `PopulationModel.draw_planets` draws a bounded pool of 256 identical hosts by
  default (`n_pool_systems`), keeps supported present planets in system order,
  and returns parameters plus an original-entry validity mask. Exhaustion cycles
  available supported draws; an empty pool uses finite inert parameters and zero
  existence. Support includes both period and eccentricity. The expected supported
  count uses a separate keyed pool of 256 systems (`n_estimate_systems`).
- `ReportRefit` reads the decision view's population and the observer's claims.
  The first tier weights population draws from its slot key at their own scales,
  resamples whole physical rows, and stores the expected count for resummary. Its
  cache identity includes the population. `occupancy`, the fixed cloud seed, the
  interim class window, `p_class` and `p_new` are removed. Dark slots carry
  conditional detection probabilities and period quantiles, and per-slot
  `p_exist` from a clipped expected-count prior updated by null evidence. Detected
  slots have existence one. Failed dark slots retain their prior summary leaves.
- Imaging candidacy requires detections, supported status and position precision.
  The policy reads claim probabilities separately and uses existence in both
  search and follow-up probabilities, so an impossible slot cannot be followed
  merely because the criterion has no claims.
- Six regression functions are added for declared-population first tiers, empty
  and all-false claims, scorer/catalog agreement, weighted claim probabilities,
  bounded exhaustion and full support, and zero-existence policy decisions. The
  exhaustion check also covers zero dark slots and saturated priors. The previous
  interim semimajor-axis test becomes a physical claim boundary test. No tests
  retire and no shared parameter matrix expands; net six functions added.
- Program budgets include measured compiles and cache loads with the standard
  headroom; existing larger allowances are retained for unmeasured slow cases.

## Population padding has zero prior weight

- Exhausted first-tier pools exclude padded rows before resampling and normalize
  null evidence over valid draws. ESS, maximum weight and fit status use the same
  mask. Empty supported pools retain zero existence and finite inert parameters.
- One regression function checks three distinct survivors padded to four, both
  without reports and after a miss. It pins equal prior mass, likelihood-weighted
  resampling, evidence and ESS. No tests retire or shared matrices expand; net one
  function added. First-tier likelihoods use each population row's own physical
  scale, preserving its radius, masses and albedo spectrum when resampled.

## Skyscapes public-name migration

- Follow the upstream capitalized unit suffixes for population fields and
  `Star.luminosity_Lsun`; observation times already pass positionally through the
  renamed `time_jd` APIs. Read stellar coverage through `Star.wavelength_range_nm`.
  Spaceodyssey parameter names and nonpositive selectable-albedo validation stay
  unchanged; the formerly negative 1842 nm spectrum is now clipped to zero upstream
  and still rejected. No golden or pinned values change.

## First-tier inference preserves joint population draws

- Evaluate each row's report likelihood with photomancy's public photometric
  forward model and fixed-scale reporting law. Resample whole physical rows and
  compute null evidence as the mean likelihood over valid rows. Both full and
  exhausted pools preserve radius, mass and albedo correlations; padding has zero
  weight. No replacement scale prior enters the first tier.
- Strengthen the population test to check same-key radius and scale distributions.
  Extend the padding regression to a full pool, with equal periods and distinct
  radii and spectra, checking every physical field and preferential weight on
  faint rows after a miss against an analytic face-on circular-orbit likelihood.
  One parameter case is added; no test functions retire or shared matrices expand.

## Named observing modes and a single spectral-channel model

- Require mode specs, science durations and a profile when constructing an observing
  source. The immutable mode table owns channel order, distinct bands, the catalog
  channel and the named telescope-days resource. Build one report per band using
  its declared throughput, frame time and resolving power.
- Require imaging modes to share one band and reporting threshold, since the refit
  fits one photometric scale and one reporting law per track. Refuse an explicit
  law with a different threshold; an omitted law uses the declared threshold.
- Remove the first-summary block model, its photon-rate divisor and implicit
  first-band stellar flux. Spectral outcomes use the channel's own count budget;
  reference-twin characterization times take a mode name and an explicit requirement.
  Imaging summaries and observation costs use the table's channel indices.
- Keep the stored ExoVista worlds and their covered 1000/100 and 940/6.7 nm bands.
  Fixture mode specs change only the survey modes' bands. A PHOENIX-host drawn
  survey test separately pins the unmodified 500/100 and 910/6.5 nm declarations.
- Add eight mode-table tests and a survey-band assertion; retire the two tests for
  the first-summary profile split and merge the survey grid test into named channel
  lookup (net six test functions). The outer-angle likelihood probe keeps two
  radius cases and the same tolerance, with brighter probes above the survey's
  reporting threshold. Migrate dependent test helpers without changing populations.
- Regenerate program budgets from measured runs with standard headroom, preserving
  larger existing allowances where only part of a slow file was measured. No golden
  or pinned hash changes.

## One named measurement row (breaking)

- Replace positional visit reports and block outcomes with `MeasurementRow` and
  the `measurement` record kind. Its template defines scalar exposure metadata,
  detection leaves and counts/variance on detection and band axes. Every writer
  constructs named leaves and packs them through the shared codec.
- Imaging exposes only detected counts in its band, with measured variance from
  the reported limit and counts. Scorer-only true counts and variance use the
  generating budget under the same detection mask. Spectrographs expose
  uncensored noisy counts and observable variance in their band, alongside
  scorer-only generating counts and model variance. Unexposed bands are zero.
- Declare `association="truth"` on observing channels: assignment and private
  identity follow the slot axis; other association settings fail explicitly.
  The bracket registry is none, product, measurement, catalog. Summaries read
  imaging measurements only, and posterior permissions check their required
  named measured columns without requiring scorer-only truth grants.
- Add five row-contract tests and move the null-row firewall test to the new
  module (six functions there, net five added). The firewall compares the same
  slot under absent and present-undetected controls after audience masking.
  Strengthen the existing refit permission regression to require a computed
  fit with ordinary measured grants. Migrate dependent test rows and helpers to
  named leaves and audience masking. No shared parameter matrix expands and no
  golden or pinned hash changes.
- Keep cached and freshly fitted summaries at the cloud's stored precision.
  Reject non-finite summary leaves from both cache admission and fitted output;
  unsupported or failed slots keep their status and emit finite placeholders.
- Center the detection-frequency control on the reporting threshold instead of
  depending on a fixture planet landing near it. Increase the giant-planet
  control's proposal pool to retain its ESS guard. Check refusal totals against
  the actual trace and a controlled two-refusal trace rather than assuming that
  an affordable-mode policy must request an unaffordable observation.
- Extend the shrinkage registry guard to unpack measurement audience grants by
  named leaf, checking that measured data remain open and generating identities,
  counts and variances remain scorer-only. Strengthen one existing test; no test
  functions or shared parameter cases are added or retired.

## Optional observation identity counters

- Declare optional per-subject key-group counters and per-channel group membership.
  A realized record carries the subject's current ordinals; only a completion
  successfully written to the log advances its channel's groups. Cancellations,
  stale or duplicate completions, and full-log refusals consume no ordinal.
- Preserve ordinals through snapshots, their fingerprints, canonical comparison,
  replay row readers and checkpoint templates. The default carries no new array
  leaves; existing golden histories and pinned hashes remain unchanged.
- Document the optional key-group structure in checkpoint restoration and fork
  row reconstruction. Extend the existing core tests to reject missing enabled
  checkpoint leaves and preserve absent counters and ordinals when disabled.
- Add four core tests for ordinal timing, cancellation and duplicate handling,
  subject isolation, default layout, full logs, executor parity and checkpoint
  resume. No tests are retired.
- Extend the codec's explicit optional-field inventory and round trips for the
  new capacity, channel membership and record ordinals; update the default-off
  guard and the snapshot fingerprint test double. The two generic null-field
  matrices each gain the corresponding declaration field.

## Measurement draws follow observation identity

- The bracket shares measurement keys across channel and acquisition index,
  declares one key group per band and advances only the groups exposed by each
  channel. The catalog channel belongs to no group.
- Imaging and spectral measurement keys fold in the true slot, exposed band and
  that band's ordinal on the subject, without duration. Channels without key
  groups retain their earlier draw path.
- Add three adapter tests for paired noise across different durations and
  sequences, changes to the exposed ordinal, imaging band isolation, distinct
  slot draws and the bracket declarations. Compare normalized draws only where
  the shared mask has finite positive generating variance. No tests are retired.

## Row-level refresh eligibility

- Observers may decline a row's refresh mark with `refits_on`, which defaults to
  true. Both the compiled handler and keyed replay apply the hook after checking
  the refit kind. A declined row is still accepted, released and logged, and
  remains in the subject's revision set for later refits.
- Add one core test with two executor cases, sharing one compiled run, for
  declined marks and retained rows in both audiences. No tests are retired.

## Bracket refit marks follow observing modes

- Bind the summary observer's static mode table from the study's channel model.
  Spectrograph rows do not mark a refit; imaging rows, including nulls, and the
  catalog row do. Spectra remain available to later refits.
- Add one study test covering the declared channels and the same rule under JIT
  and batching. No tests are retired.

## Named per-star refit lanes

- Gather imaging reports into a private per-star lane and retain track execution
  groups by engine and row bucket. Detected, dark, shrinkage and stored-cloud
  paths return named result modules, including physical draws and raw updater
  status. Retry from the raw updater status and scatter summary leaves by name.
- Add three tests: distinct-field scatter with failed and unsupported controls,
  the raw-status retry mask, and bitwise parity with an executable positional
  reference on both posterior engines using reduced proposal counts. Adapt
  existing private-interface tests; no tests are retired, shared parameter
  matrices expand, or serialized goldens and pinned hashes change.
- Reduce the executable positional reference to direct fixture row gathering and
  slice assembly, sharing numerical kernels and the lane runner. Remove copied
  fit kernels, engine/bucket orchestration, cache, shrinkage and forwarding paths.
  Require both engines independently in the reference and boundary runs; retain
  every slice, status, diagnostic and computed-mask comparison. No tests or
  parameter cases are added or retired, and no serialized golden changes.

## Criterion state in the summary observer

- Register the dependency-light criterion family and its abstract state, value,
  and package interfaces. A summary stores one packed state vector per star;
  changing the criterion changes its width without changing the summary tree.
- Fold measured rows incrementally for each audience through the record codec.
  Refits install only the fitted summary leaves and preserve criterion state.
  Refit and outcome claims now come from the observer's declared criterion.
- Add three contract tests for duplicate releases and track assignments, state
  layouts and refit preservation, and imports without physical dependencies.
  Existing imaging fixtures supply an explicit empty-state criterion. No tests
  are retired, parameter matrices expanded, or goldens and pinned hashes changed.

## Forecast and availability leaves

- Summaries carry criterion-independent count quantiles and the model variance
  of each selected draw for every future node and non-catalog channel. The
  quantiles default to (1/6, 1/2, 5/6) and are configurable in the report refit.
  Truth, posterior, shrinkage, dark-slot and stored-cloud paths carry these leaves.
- Forecasts select the channel's band, albedo and science duration against its
  nominal catalog-mass scene. Node epochs start the science exposure. Availability
  is a Boolean per-star [G, V] leaf and is all true, including without a fit.
- At S=6, G=16, V=36, Q=3, C=1, B=3, with three imaging durations and packed
  state width W=2*S*B, the complete summary uses 169600 bytes per star per
  audience (165.625 KiB), or 339200 bytes for both audiences (331.25 KiB).
  The closed form is 8*(2 + S*(10+C+G*M) + 2*S*G*V*Q + W) + G*V bytes,
  where M=3. The admission test enforces a 256 KiB ceiling per star per audience.
- Add five tests for weighted forecasts, paired variance, configurable quantiles,
  dark clouds, memory admission and one-point real-channel integration. Extend the
  named scatter checks to forecasts and availability; preserve infinite model
  variance and reject NaNs. No tests are retired or parameter matrices expanded.
  Goldens and pinned hashes are unchanged.
