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
