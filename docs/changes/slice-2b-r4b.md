---
orphan: true
---

# Slice 2b R4b: surveys 3 and 6

R4b builds the two Slice 2b surveys that follow the prior and epoch mechanisms:
survey 3 (known planets: priors with a declared pedigree and window-valued
scheduling, after R2) and survey 6 (aging ephemerides: a value that reads the age of
the accepted prior record, after R1 and R4a). It also closes the two points R1 and R2
left here: the window columns are now written into the imported prior rows, and
replay and decision equality under a hidden row run on survey 3 (spec survey 3 (d)).
Both surveys run the epoch relation (B6).

## Commits

1. `feat(studies): survey 3, priors with a declared pedigree and window-valued scheduling`
2. `feat(studies): survey 6, a value that reads the age of the accepted prior record`
3. this record

## What changed

- `studies/survey_3.py` (new): `WINDOW_PRIOR_KIND` (`value`, `precision`,
  `window_lo_rel`, `window_hi_rel`, all public, derived), `survey_3_registry`,
  `WindowImporter`, `WindowPolicy`, `survey_3_table` and `survey_3`.
- `studies/survey_6.py` (new): `survey_6_registry` (the toy kinds and R2's
  `PRIOR_KIND`), `AgingValuePolicy`, `survey_6_table` and `survey_6`.
- `tests/studies/test_survey_3.py`, `tests/studies/test_survey_6.py` (new).
- No shared file changed: `studies/premission.py`, `studies/prior_observer.py`,
  `core/`, `codec/` and `tests/conftest.py` are untouched. The golden and document
  hash pins are untouched.

### Window columns in records (D5)

`WindowImporter` is the R1 importer with one addition in `handle`: a row of the
windowed prior kind gets the table's `window_lo_rel` and `window_hi_rel` in its
payload columns 2 and 3 when the observer's overlap resolution retains the prior for
the audience of the event that writes it, and `NO_WINDOW` (`(0, -1)`, an empty
window) when it drops it. A public prior is written once per audience (R1), so each
copy carries its own audience's resolution, decided by the evidence that audience
receives. The row's metadata (audience, release and acceptance stamps, count) are the
R1 importer's and never depend on the exclusion. The resolution is read from the
study's observer at trace time (`study.observer.weights(caps, audience)`, the duck
protocol of R2), so there is no second copy of the rule to disagree with; an observer
that does not read priors is refused with a `TypeError` when the handler is traced.

### Survey 3

The table has five rows of one shape, varied by padding, audience and order, so every
variant shares one compiled program per executor:

| Row | Kind | Subject | Audience | Pedigree |
|---|---|---|---|---|
| 0 | measurement | 0 | public | |
| 1 | measurement | 1 | public | |
| 2 | measurement | 0 | observer only, public, or not imported | |
| 3 | windowed prior (4, precision 2, window 40 to 80) | 0 | public | an external source and row 2 (external when row 2 is not imported) |
| 4 | windowed prior (-3, precision 1) | 1 | public | row 1 |

`WindowPolicy`'s first decision waits until the target prior's window opens
(`released_tick + window_lo_rel`), or, with no window released to it, 150 ticks from
the decision; the next four decisions observe the subject with the largest posterior
mean among subjects 0 and 1 (so decisions read the posterior); then the campaign
drains. Every tick it reads is relative.

### Survey 6

The table holds a public prior of subject 0 spanning `[-2 age, -age]` and a public raw
row of subject 1 at `[-5, -1]`. `AgingValuePolicy` takes the prior's age at a decision
as the decision tick less the prior row's `end_tick`, values it as `base * decay **
age` (array fields, so every decay shares one program), waits the ticks the value
would take to reach `threshold` (within `[1, 400]`), observes the planet twice and
drains. The value of the first decision is kept in `pstate[0]`. The control flag
`absolute` (an array) takes the age as minus the end tick, as if the run started at
tick 0.

## Tests

Tests added: 12 (9 in `test_survey_3.py`, 3 in `test_survey_6.py`); retired: none.

Cold times, each test alone in a fresh process, serially
(`SPACEODYSSEY_JAX_CACHE=off JAX_ENABLE_COMPILATION_CACHE=false`, four threads, wall
time including the about 1.5 s import, on a machine shared with other jobs):

| Test | Spec | Tier | Cold s |
|---|---|---|---|
| `test_survey_3.py::test_a_disjoint_external_prior_changes_both_beliefs` | 3 (a) | medium | 18.3 |
| `::test_exclude_prior_with_an_imported_source_equals_the_raw_only_run` | 3 (b) | medium | 18.3 |
| `::test_exclude_rows_matches_independent_sufficient_statistics` | 3 (c) | medium | 15.4 |
| `::test_an_observer_only_source_moves_no_policy_view_or_decision` | 3 (d), D5, K3 | medium | 20.1 |
| `::test_intersecting_priors_are_refused_and_one_datum_survives_once` | 3 (e) | unit | 3.0 |
| `::test_the_policy_starts_in_the_window_or_at_the_fallback` | 3 (f) | medium | 18.5 |
| `::test_replay_reproduces_the_survey_with_its_priors` | D2 replay, B7 executors | medium | 21.5 |
| `::test_the_epoch_relation_holds_with_windows_and_hidden_rows` | B6 | medium | 21.7 |
| `::test_the_controls_fail_their_checks` | 3 (g) | slow | 23.1 |
| `test_survey_6.py::test_the_epoch_relation_holds_for_three_shifts_on_both_executors` | 6 (a), B6 | medium | 15.0 |
| `::test_a_value_reading_an_absolute_end_tick_fails_the_epoch_relation` | 6 (c) | medium | 8.7 |
| `::test_doubling_the_prior_age_moves_the_value_monotonically` | 6 (b) | medium | 22.3 |

The six survey 3 tests marked "the run store" share one module fixture: the survey on
both executors for six table variants of one shape (no prior, external prior, hidden
row, raw only, public row 2, permuted), so the quick tier compiles survey 3 once per
executor for all of them; each test alone pays that fixture (about 15 s). Test (c)
compiles its own program per executor (the rule is a static field of the observer);
the epoch tests compile the process-wrapped study per executor. The whole of
`test_survey_3.py` takes 72 s in one process (52 s without the slow control) and
`test_survey_6.py` 25 s.

Compiled signatures added, per executor: survey 3 (with its views captured), survey 3
under `EXCLUDE_ROWS`, survey 3 under the epoch wrapper, survey 6, survey 6 under the
epoch wrapper; and, slow and compiled only, the two control observers. R0a's
inventory counted one signature for each survey; the overlap rule being static, the
view capture of the run store and the epoch wrapper make it five medium signatures
for the two surveys (ten cold compiles), about 60 to 70 s of medium time locally.

## Mutants and controls

Each mutant was applied to a copy of `src/` and the survey's test file run against it
(`$TMPDIR` script, not committed). All 17 are killed.

| Mutant | Where | Failing tests |
|---|---|---|
| m1 window written whatever the resolution | `WindowImporter` | 3 (d), 3 (f) |
| m2 both copies take the observer's resolution | `WindowImporter` | 3 (d) |
| m3 window start read as an absolute tick | `WindowPolicy` | B6 survey 3 |
| m4 fallback start an absolute tick | `WindowPolicy` | B6 survey 3 (fallback case) |
| m5 window ignored (always the fallback) | `WindowPolicy` | 3 (f) |
| m6 overlap ignores the audience | `resolve_overlap` | 3 (d), replay |
| m7 `EXCLUDE_PRIOR` keeps the prior | `resolve_overlap` | 3 (a), (b), (d), (e), (f) |
| m8 `EXCLUDE_ROWS` keeps the rows | `resolve_overlap` | 3 (c) |
| m9 prior installed whatever its weight | `PriorObserver.prior_belief` | 3 (a), (b), (d), (e), (f) |
| m10 prior never installed | `PriorObserver.prior_belief` | 3 (a), (c), (d), (e), replay, (g) |
| m11 `D008` never | `retained_intersections` | 3 (e) |
| m12 replay starts the policy summary from `init` | `replay_run` | replay |
| n1 age read against tick 0 (absolute) | `AgingValuePolicy.value` | 6 (a) |
| n2 value grows with age | `AgingValuePolicy.value` | 6 (b) |
| n3 age from the release stamp, not the end tick | `AgingValuePolicy.value` | 6 (b) |
| n4 wait counted from tick 0 | `AgingValuePolicy.decide` | 6 (a), 6 (c) |
| n5 the control's flag ignored | `AgingValuePolicy.value` | 6 (c) |

Shipped controls: `RuleOff` (installs prior 4 and folds the raw row it summarizes:
fails (b), matches the double-counted statistics), `IgnoresPrior` (passes nothing in
(a)); survey 6's `absolute` flag (fails (a) in the ticks only).

## Checks run

- `tests/core/test_golden_histories.py` with `tests/codec/test_run_document_runs.py`
  (golden pin and corpus document hash pin, `-n 2 --dist loadfile`): 189 passed in
  258.4 s. No pin file changed.
- Quick tier (`-m "not slow" -n 2 --dist loadfile`, deselecting
  `tests/codec/test_entry_points_installed.py` for the stale environment metadata
  noted in R0a): 2,085 passed in 269.1 s, no budget violation. Both new files are in
  it, medium tests included; the slow survey 3 control ran alone (above).
- The touched files: `test_survey_3.py` 9 passed (72.4 s), `test_survey_6.py` 3
  passed (25.3 s).
- `ruff check`, `ruff format --check`, `tools/check_internal_refs.py`,
  `tests/core/test_lints.py` (187 passed) and `pre-commit` on the touched files:
  clean. Every touched file is ASCII.
- The full suite and the Sphinx docs build were not run.

## Spec silences decided here

1. **Where the window is written.** In a survey-3 importer subclass
   (`WindowImporter`), not in `studies/premission.py`. The window is a property of a
   kind (which payload columns hold it) and of the observer's rule, neither of which
   the R1 importer knows, and survey 3 is its only consumer in this slice. Rejected:
   new fields on `PreMissionImporter` (a shared file, and its document form is R3's
   to round-trip).
2. **The rule is read from the observer**, not duplicated on the importer, so the
   written window always follows the installed resolution (and a control observer's
   rule, for instance `RuleOff`, writes its own windows).
3. **"Not released to it" in (f) reads as "no window released to it".** Spec row
   (f) still has the round-2 wording ("the row is not released to it"); D5 (round 4)
   fixes the row's release and metadata and puts the exclusion only inside the
   audience's own row. The test checks that the policy's copy is released and carries
   `NO_WINDOW`, and that the policy takes the fallback start.
4. **The fallback start is relative** (150 ticks from the first decision), so the
   epoch relation holds on the fallback path too; an absolute fallback tick would
   break B6 by design.
5. **"Every logged field of the policy's rows" in (d)** is compared as the policy
   view (snapshot views at every decision and at the end, their fingerprints, the
   effects and the policy summary). The internal `acquisition_id` of a row written
   after the hidden row shifts with it; the core hides it from every policy view
   ("settlement ids shift when private events are queued"), so it is not
   policy-visible metadata.
6. **Permutation (d)** compares decisions bit for bit, the sorted written rows
   (identity, audience and payload) and the beliefs in the go band: the observer
   folds the imported raw rows in release order, which a permutation changes, so the
   running mean may differ in the last bits.
7. **The (b) "raw-only" run** is the same table with prior 4 padded. **The (a)
   "no-prior" run** pads prior 3 only; prior 4 is present and excluded in both.
8. **Survey 6's "value"** is a policy quantity that drives an action (the wait), not
   the policy's third return value (a uniform draw in the core contract). It is read
   from `pstate[0]` for (b); it drives the wait so that (a) and (c) see it through
   the trace and the views.
9. **Survey 6's age** is measured from the prior row's `end_tick`, not its
   `accepted_tick`: the policy's copy of a public prior keeps `accepted_tick = -1`
   (R1 silence 5), and spec (c) names the end tick as the absolute reading.
10. **Tiers.** Each survey test that runs the study is medium (one compiled program
    per executor, under 30 s cold alone). The survey 3 controls replace the observer
    type (two more programs) and are slow, as D11 lists them. Survey 6's control
    shares the relation's program (its flag is an array) and is medium.
11. **(e) is a unit test** on the survey's own builder: pre-flight on three
    configurations whose retained priors intersect, and the eager import beliefs on
    the one where `EXCLUDE_PRIOR` drops both summaries.

## Spec points not satisfied

- B6 "on the four studies": surveys 3 and 6 run the relation here; surveys 4 and 18
  are R5's.
- B7 "on both executors": every survey test that runs a study runs both, except the
  slow survey 3 controls and the survey 6 control (compiled executor only), as R0a's
  inventory and D11 list the controls.
- K6: the five medium signatures exceed R0a's one-per-survey estimate (above); the
  quick-tier cost against the K6 baseline is R6's to read.
- The window columns are written only by the survey 3 importer. A general importer
  option in `studies/premission.py` waits for a second consumer.
