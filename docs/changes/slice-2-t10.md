---
orphan: true
---

# Slice 2 T10: surveys 2, 5 and 14

Three studies in their Slice 2 variants, each with a pass test and a run-document round
trip (`dump_run`, then `load_run`, gives an equal history and the same content hash).
The change adds only `studies/` modules and `tests/studies/` files, plus the survey test
tier declarations in `tests/conftest.py` and the individual survey files. No file
under `src/spaceodyssey/core/` is touched, and `tests/core/golden_histories.json` is
unchanged.

## Commits

1. `feat(studies): survey 14, a prescribed phased mission with a no-switch reduction` (`92f3202`)
2. `feat(studies): survey 5, astrometric candidates on a delayed catalog roster` (`203a214`)
3. `feat(studies): survey 2, an adaptive revisit search with calibration and an OFTI variant` (`79cc898`)
4. this record

## Survey 14, phased mission

`survey_14(switch_tick=None)` is survey 1 with two plan policies and `phase_until`. A
threshold of `caps.max_tick` (written `{"no_switch": true}` in a document) never switches.

- Pass test: with no switch the full canonical history equals survey 1's on the host and
  the compiled executor. A switch at tick 300 hands the second phase the decision at that
  tick; the run document reproduces the history and the hash.
- Negative controls: a switch at tick 300 and one at tick 1 both give histories that
  differ from survey 1's; a different boundary gives a different content hash.

## Survey 5, astrometric candidates

An admission-channel catalog (channel 13, public audience) released at tick 325, read by an
observer that keeps a candidate roster. Candidate ids use the negative namespace
`-1 - roster ordinal`, kept in belief
aux slots. Subject and acquisition ids are nonnegative, so neither range can collide
with a candidate, even when capacities exceed 1,000. The policy follows a
candidate up on channel 11 once the roster has one.

- Pass tests: decisions at ticks 0 and 130 read an empty roster and the decision at 325 is
  the first to read a candidate; a release 100 ticks later changes the belief updates and
  nothing before them; candidate ids are disjoint from subject ids and the catalog value
  stays private (policy, observer and scorer probes); host and compiled agree and the
  document round-trips on both; `battery.latency_shift` passes for an absolute-tick plan.
- Negative controls: an early release with latency 0 is caught by `premature_reads`;
  `latency_shift` fails for the reactive follow-up policy; pre-flight C001 flags a decision
  at tick 130 and passes at 325, and flags an observer-only catalog. The conservative
  cancellation bound is independently derived as admission tick 130 plus latency 150,
  or 280: C001 flags 279 and passes 280 and 281. This differs from the actual
  uncanceled release at 325 (start 135 plus occupancy 40 plus latency 150).

## Survey 2, adaptive search

`RevisitPolicy` revisits the undetected subject with the highest signal-to-noise ratio
(`mean * sqrt(n) / noise` against the objective threshold), an unseen subject first, and stops
when all are detected. Declared `ignore_belief` it takes plan entries and equals survey 1.
`survey_2(every_ticks)` adds the refresh grid and the refit-capable `SearchMean`.

- Pass tests (host and compiled where stated):
  - belief-ignoring with no refresh: full canonical history equals survey 1's, both executors;
  - belief-ignoring with refresh on both executors: action projection equals survey
    1's, and the full history does not (the refresh rows differ);
  - adaptive refresh on equals adaptive refresh off in the action projection;
  - an eager NumPy recomputation of the rule matches every recorded decision;
  - compiled equals host with a refresh grid;
  - round trip with the masked refit binding, with and without refresh;
  - calibration on a Gaussian prior (`GaussianSignals`, `GaussianMean`, 2,000 worlds on the
    compiled executor) through the kit's `check_calibration`, margin 0.075, both pooled
    and separately for each subject;
  - OFTI: `OrbitSearchObserver` folds measurement means and counts into a small
    `MeanBelief` policy summary while holding orbit particles and fitted epoch counts
    in the full belief. Scripted astrometry with `OftiRefit(count="epochs")` installs
    particles (`epochs == 3`) and leaves adaptive actions unchanged. The adaptive
    choices also match an eager rule derived from the captured summary. The ignoring
    policy reduction remains a separate parameter case.
- Negative controls: the adaptive policy differs from survey 1 (so the reductions are not
  vacuous); a different plan changes the OFTI action projection; calibration fails for
  doubled noise, halved noise, a narrow prior, a wide prior and a shifted centre; a NaN true
  signal fails; a NaN, infinite, negative or zero margin is refused. Opposite PIT
  errors (CDFs x^2 and 2x - x^2) pass the pooled check but fail both subject checks.
  The action projection uses native host decision/effect logs and compiled
  `DecisionViews`, and compares every requested effect field, admission codes, and
  trace disruptive flags. Nine requested-field mutations and a trace disruptive mutation
  of a rejected request, with no acquisition rows, each fail the comparison.

## Waiting on other tasks

- Survey 2 replay assertions (replay of the recorded adaptive run with `OftiRefit` or the
  masked refit, equal beliefs and products) wait on T4c: `replay_run(run, roots, carry,
  trace, refit=...)` is not on this branch's base. The test
  `test_replay_of_the_adaptive_run_waits_on_keyed_replay` skips with that reason and fails
  loudly (it raises) once `replay_run` appears, so it cannot be forgotten. No T4c code was copied.
- Nothing waits on T3b.

## Builder self-review

- Cold timings (`SPACEODYSSEY_JAX_CACHE=off`, 4 threads), per file: survey 14 slow 16.8 s and
  static 1.5 s; survey 5 slow 27.9 s and static 1.5 s; survey 2 45.9 s, calibration 9.4 s,
  OFTI 15.0 s, static 1.2 s. Three slowest tests including fixture setup: survey 14
  baseline fixture 14.5 s, survey 2 executor agreement 9.9 s, survey 2 belief-ignoring
  reduction 9.2 s. These original file totals do not establish individual test tiers.
  A cold test under 5 s is unit, under 30 s is medium (included in quick CI), and over 30 s is
  slow. The blanket survey entries in `SLOW_FILES` have been removed; markers belong
  to individual tests or homogeneous files, using setup plus call time. Revised
  measurements appear below.
- Mutants and controls: listed under each survey above. The statistical checks (calibration)
  have spread, shifted-center, NaN-input and invalid-margin controls. The earlier
  dismissal of cancellation was incorrect: pooling opposite subject CDF errors can
  yield uniform PITs even when both subjects are severely miscalibrated. Each subject
  is now checked separately, and the canceling-CDF control demonstrates the failure
  of the pooled-only guard.
- Audience isolation: no new column, record kind or product; survey 5 uses the existing
  public catalog column and probes policy, observer and scorer audiences (above). Survey 2
  reads only the policy's belief summary; the OFTI variant uses the existing public
  astrometry columns.
- Ordering and metadata: nothing new that stamps or digests rows; survey 2 refresh uses the
  merged protocol unchanged.
- Goldens: `tests/core/golden_histories.json` unchanged; the golden test and both gate files
  pass (see the report).
- Sibling sites for the rules added: causality (pre-flight C001) and latency shift for the
  catalog channel; host and compiled executors for each reduction; codec dump, load and
  re-dump for each study; refresh on and off for survey 2.

## Spec silence

1. Survey 5 uses an admission-channel catalog with a public release, not a scripted
   process, because `latency_shift` and C001 act on admission channels. Alternative rejected:
   a scripted process (neither check would see it).
2. Candidate ids use a separate negative namespace. Large-capacity probes cover
   1,001 subjects and roster ordinals 0, 1,000 and 100,000; all are disjoint from
   subject and acquisition ranges. Reusing truth ids would expose their mapping.
3. Survey 2's detection-count belief is the `MeanBelief` mean and count through the
   objective's threshold, so the survey 1 reduction holds exactly; alternative rejected: a
   new belief class with an explicit count (the reduction would need a second observer).
4. Survey 2 calibration uses a Gaussian prior with a conjugate observer so it is exact under
   an adaptive design; alternative rejected: the toy uniform prior (no closed-form posterior).
5. C001 permits cancellation during the transition and therefore has a conservative
   bound of 280 for this fixture. The bisection must equal that independently derived
   bound, with explicit checks at 279, 280 and 281. The actual release remains 325.
6. No entry points or `pyproject.toml` kind registrations were added; the tests map kinds
   explicitly, as survey 25 does.
7. The OFTI scripted astrometry wakes both policies at ticks 10, 40 and 70. Some
   observes are rejected. Complete requested effects are captured before admission,
   so the equality includes those requests; acquisition rows alone are insufficient.

## Tests

Original implementation: 46 added (survey 14: 9, survey 5: 9, survey 2: 28 including
one skip), none retired, net +46. Review fixes add 17 cases (one executor case, one
cancellation control, two OFTI policy cases, three large-capacity cases, and ten
complete-effect mutations); none retired, net +17, total 63 including the unchanged
replay skip.


## Review fixes

| Finding | Change | Regression evidence |
|---|---|---|
| 2 | Detection summary alongside orbit particles; adaptive and ignoring OFTI cases | OFTI installation, unchanged complete actions, eager adaptive choices, and a different-plan control |
| 3 | Pooled and per-subject PIT checks | Opposite CDF errors pass pooled but fail both subject checks |
| 4 | Refresh-enabled ignoring reduction on host and compiled | Both executor cases compare against survey 1 |
| 5 | Capture complete effects before admission with native host logs and compiled `DecisionViews`; the ignoring branch delegates to `FixedPlanPolicy` | Every effect field and the trace disruptive flag are mutated with an empty acquisition log; start and stop tables must match survey 1 including inactive columns |
| 6 | Independent cancellation bound 280, actual release 325 | Bisection equals 280; C001 at 279, clean at 280 and 281 |
| 7 | Negative candidate namespace | 1,001 subjects and up to 100,001 acquisitions cannot overlap candidate ids |
| 8 | Remove blanket survey slow marking; assign tiers from cold setup plus call | Quick-tier collection and cold timing probes, recorded below |

Finding 1 is out of scope. Replay activation and its existing skip remain unchanged.

## Verification after review fixes

Runs use `PYTHONPATH=src`, the shared environment's Python,
`-m pytest -p no:cacheprovider`, and exclude the installed-entry-point test. Every
run-based survey file and cold probe runs through `compute-run` with four threads
and a 4,096 MiB ceiling, one survey file per pytest process. Persistent JAX caching
is disabled for the cold measurements. Queue wait is excluded from pytest times.

| Cold individual survey 2 check | Setup + call (s) | Tier |
|---|---:|---|
| No-refresh ignoring reduction, both executors | 24.01 | medium |
| Refresh ignoring reduction, compiled | 28.08 | medium |
| Adaptive actions, refresh on/off | 22.12 in this probe; about 41 observed in review | slow |
| Adaptive executor parity with refresh | 28.31 in this probe; about 32 observed in review | slow |
| Eager adaptive rule | 19.96 | medium |
| Run document without refresh | 26.63 | medium |
| Run document with refresh | 40.93 | slow |

The shared adaptive comparisons retain slow classification because their cold
observations exceed 30 s, even though the latest isolated samples were faster.
The survey 14 no-switch fixture took 52.67 s cold. Isolated finite-switch and
early-switch checks took 42.88 s and 41.69 s including setup, so every survey 14
run case remains slow, with explicit per-test markers. Calibration and OFTI remain medium; survey 5's complete
file takes more than 30 s, but its individual checks qualify as medium. No
run-based survey file is blanket-marked slow in `tests/conftest.py`. The replay
guard retains its prior slow-tier skip.

The complete requested-effect comparison initially failed on both executors: the
ignoring policy populated inactive fields differently from survey 1. Delegating to
`FixedPlanPolicy` fixes the mismatch. The fast start/stop table control catches it
without compiling a full run. Native host captures reuse the executor's own logs;
compiled captures use `DecisionViews`.

| Validation | Result | Pytest wall (s) | Peak RSS (GB) |
|---|---|---:|---:|
| Static surveys and complete-effect controls | 23 passed | 2.44 | 0.32 |
| Repository gate, core lints, internal-reference guard | 221 passed | 3.75 | 0.15 |
| Survey 2 reductions and documents | 9 passed, 1 expected replay skip | 64.85 | 2.67 |
| Survey 2 calibration | 14 passed | 10.50 | 1.10 |
| Survey 2 OFTI | 4 passed | 41.56 | 1.48 |
| Survey 5 | 6 passed | 46.18 | 1.56 |
| Survey 14 | 6 passed | 59.81 | 1.40 |
| Compatibility gate and golden histories | 226 passed | 539.98 | 3.51 |

Quick collection over all 63 survey and helper cases retains 53: 23 unit controls
and 30 medium cases. The ten excluded cases are the six survey 14 runs, two shared
adaptive comparisons, the refresh-enabled document round trip, and the unchanged
replay guard. The collection probe uses `--collect-only -m "not slow"` and confirms
that medium cases are selected.

Final distinct test results: 509 passed and one expected replay skip (62 survey
and helper passes, 221 repository/lint/guard passes, 226 compatibility/golden
passes). Isolated timing probes also passed. The three slowest final cold survey
checks including setup were the survey 14 no-switch reduction (53.83 s), its finite
switch (42.88 s), and its early switch (41.69 s). The compatibility process sampled
a peak footprint of 3,835 MiB, below the 4,096 MiB ceiling. The original change
record did not capture baseline peak memory; the table reports the measured final
values rather than inventing a before/after memory comparison.

`ruff check .`, changed-file formatting, the internal-reference scanner, ASCII
validation of changed Python, and `git diff --check` pass. A repository-wide
formatter check also reports an existing fenced-stub formatting issue in the
unrelated `docs/changes/slice-2-t1b.md`; that file is left untouched. Core sources
and golden histories have no diff. No commit or push was made.
