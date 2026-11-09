---
orphan: true
---

# Slice 3b surveys 4 (full), 7, 13 and 19

Four surveys that use the new bundle, configuration, key-version and commitment features
through study code only: no file under `core/` changed.

## What each survey shows

- **Survey 4 (full).** The downstream study gains an RV channel (`RV_CHANNEL`, 61) on lane 2,
  which is asset 1's lane with its own 5 tick transition. `RVFollowUpPolicy` asks for one
  observation on it with its first observation of the campaign, of the target the summary holds
  the fewest measurements of. The RV request is admitted in decision 0's group on both lanes,
  the record starts at lane 2's own transition (tick 5), asset 1 is debited its cost (1.0), and
  the follow-up goes to the target the precursor does not cover (subject 1) while it goes to the
  first target without the precursor (subject 0); the value with the precursor is 3 against 2.
  Counted: one RV record per run, two imported rows in the run with the precursor only. The
  export, import, lineage and paired-difference tests are unchanged.
- **Survey 7.** Channels 12 and 13 share lane 2 in one bundle (`PairPolicy`).
  (a) With the balance of asset 1 short of both members, both are refused (`NO_FUNDS`, 2 of 2,
  no record, no debit) and the same balance admits one member alone; with a full balance both
  are admitted (2 records). (b) The lane frees at tick 55 = decision tick 0 + one 5 tick
  transition + the longest occupancy (50); the serializing control (the same channels requested
  one after the other) ends at 100, later, and breaks the rule. (c) `BundleBackgroundObserver`
  groups records by `Record.bundle`: the two bundled records give information 1.6, equal to the
  one-background reference `1 / (v / 2 + b)`, and differ from the two-background value 2.667;
  the same two rows through the observer with `grouped=False` reach 2.667, as does the
  serialized run whose records are not bundled.
- **Survey 13.** Asset 0 switches at tick 100. The observation admitted at 0 carries
  configuration 0 and the one admitted at 130 carries 1 (both counted); the commitment admitted
  at 0 and started at 210 (after the switch) carries 0 (one record). `ConfigPhotometry`
  (version 2) takes its gain from `context.config_id` and writes the stamp into the row; each
  value equals the gain of the stamped configuration times the signal plus the noise draw, and
  differs by more than 0.1 from the other configuration's value.
- **Survey 19.** `JointPolicy` plans two facilities (channels 10 and 12, assets 0 and 1) as one
  admission group each round: two rounds, four members, all admitted whole. A third facility with
  a zero balance and no channel (`with_idle_facility`) changes no row: keys (the measurement draw
  key of each record, key version 2), payloads, belief leaves, every field of the action trace
  and each balance row of the first two facilities (and of the program and shared rows after the
  third) are equal, compared record by record and row by row. The asset-key mutant
  (`asset * n_assets`, patched into the handler and into the key helper) moves the key and the
  payload of both asset-1 records and none of asset 0's.

## Controls and how they failed

| Survey | Control | Result |
| --- | --- | --- |
| 4 | no precursor (padded table) | RV follow-up goes to subject 0, value 2 against 3 |
| 7 | serialized request of the pair | lane frees at 100, not 55; the pointing rule fails |
| 7 | balance short of both members | pair refused whole, one member alone admitted |
| 7 | observer with `grouped=False` | information 2.667, not the 1.6 reference |
| 13 | value under the other configuration's gain | differs by more than 0.1 for every record |
| 19 | `asset * n_assets` lookup | key and payload of both asset-1 records differ; asset 0 rows equal |

## Tests

| File | Tests | Tier | Programs (warm), budget |
| --- | --- | --- | --- |
| `tests/studies/test_survey_4.py` | 1 added (5 total) | slow | 332, 408 (unchanged) |
| `tests/studies/test_survey_7.py` | 3 | slow | 103, 128 |
| `tests/studies/test_survey_13.py` | 2 | slow | 56, 75 |
| `tests/studies/test_survey_19.py` | 3 | slow | 203, 244 |

Each file shares one module-scoped set of host runs (survey 19's mutant test recompiles after
patching the asset lookup and clears the caches again afterward). No test was retired; no pinned
golden, fingerprint, FAULT fixture or corpus document hash was touched, and none of the studies
is in a pinned corpus. `RVFollowUpPolicy` is a subclass registered as its own kind, so
`LeastKnownPolicy` documents are unchanged.

## Receipts

- Test time and the checks run at the final commit are in the section below.

- The four survey files together: 13 passed in 27 s (37 s wall under `compute-run`, warm
  compile cache; about 45 s cold from the per-file timings), peak footprint 1.5 GB.
- `tests/studies -m "not slow"` at `-n 2 --dist loadfile`: 88 passed in 50 s (83 s wall).
- `tests/core/test_golden_histories.py -m ""`: 123 passed and one failure,
  `test_history_matches_its_golden_digest[refresh-compiled]`, in the full-file run; the same
  test passes when run alone (2 passed, 7 s). The diff of `src/spaceodyssey/core` from the base
  is empty, so the failure is not from these commits; it looks order or cache dependent and
  is left for the close-out.
- No file under `core/` changed; no golden, pin or corpus file changed.
