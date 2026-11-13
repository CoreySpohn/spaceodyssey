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
