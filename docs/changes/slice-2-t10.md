---
orphan: true
---

# Slice 2 T10: surveys 2, 5 and 14

Three studies in their Slice 2 variants, each with a pass test and a run-document round
trip (`dump_run`, then `load_run`, gives an equal history and the same content hash).
The change adds only `studies/` modules and `tests/studies/` files, plus the survey test
files in the slow list of `tests/conftest.py`. No file under `src/spaceodyssey/core/` is
touched, and `tests/core/golden_histories.json` is unchanged.

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
observer that keeps a candidate roster. Candidate ids are `1000 + roster ordinal`, kept in
belief aux slots, so they never collide with truth subject ids. The policy follows a
candidate up on channel 11 once the roster has one.

- Pass tests: decisions at ticks 0 and 130 read an empty roster and the decision at 325 is
  the first to read a candidate; a release 100 ticks later changes the belief updates and
  nothing before them; candidate ids are disjoint from subject ids and the catalog value
  stays private (policy, observer and scorer probes); host and compiled agree and the
  document round-trips on both; `battery.latency_shift` passes for an absolute-tick plan.
- Negative controls: an early release with latency 0 is caught by `premature_reads`;
  `latency_shift` fails for the reactive follow-up policy; pre-flight C001 flags a decision
  at tick 130 and passes at 325, and flags an observer-only catalog.

## Survey 2, adaptive search

`RevisitPolicy` revisits the undetected subject with the highest signal-to-noise ratio
(`mean * sqrt(n) / noise` against the objective threshold), an unseen subject first, and stops
when all are detected. Declared `ignore_belief` it takes plan entries and equals survey 1.
`survey_2(every_ticks)` adds the refresh grid and the refit-capable `SearchMean`.

- Pass tests (host and compiled where stated):
  - belief-ignoring with no refresh: full canonical history equals survey 1's, both executors;
  - belief-ignoring with refresh: action projection equals survey 1's, and the full history
    does not (the refresh rows differ);
  - adaptive refresh on equals adaptive refresh off in the action projection;
  - an eager NumPy recomputation of the rule matches every recorded decision;
  - compiled equals host with a refresh grid;
  - round trip with the masked refit binding, with and without refresh;
  - calibration on a Gaussian prior (`GaussianSignals`, `GaussianMean`, 2,000 worlds on the
    compiled executor) through the kit's `check_calibration`, margin 0.075;
  - OFTI: scripted astrometry with `OftiRefit` on the host installs particles (`n == 3`) and
    leaves the policy's actions unchanged.
- Negative controls: the adaptive policy differs from survey 1 (so the reductions are not
  vacuous); a different plan changes the OFTI action projection; calibration fails for
  doubled noise, halved noise, a narrow prior, a wide prior and a shifted centre; a NaN true
  signal fails; a NaN, infinite, negative or zero margin is refused.

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
  reduction 9.2 s. All three are over the 5 s unit budget, so every run-based file is in
  `SLOW_FILES` (as survey 25 is); the only unit-tier files are the `*_static.py` ones, all
  under 1.6 s per file.
- Mutants and controls: listed under each survey above. The statistical checks (calibration)
  each have a doubled-variance, a shifted-centre, a NaN-input and a NaN-margin control; the
  "effect that cancels" control does not apply because the PIT test is one-sided in distance
  from uniform, not a mean.
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
2. Candidate ids are `1000 + roster ordinal`; alternative rejected: reuse truth ids (would
   leak the truth-to-candidate map).
3. Survey 2's detection-count belief is the `MeanBelief` mean and count through the
   objective's threshold, so the survey 1 reduction holds exactly; alternative rejected: a
   new belief class with an explicit count (the reduction would need a second observer).
4. Survey 2 calibration uses a Gaussian prior with a conjugate observer so it is exact under
   an adaptive design; alternative rejected: the toy uniform prior (no closed-form posterior).
5. C001 is conservative: it flags the catalog channel until tick 325 plus the declared need;
   the survey 5 test finds the first clean tick by bisection rather than asserting 280.
6. No entry points or `pyproject.toml` kind registrations were added; the tests map kinds
   explicitly, as survey 25 does.
7. The OFTI scripted astrometry wakes the policy at ticks 10, 40 and 70, so the belief-ignoring
   plan spends entries on rejected observes; the test asserts the equality of actions, not the
   number of acquisitions beyond two.

## Tests

46 added (survey 14: 9, survey 5: 9, survey 2: 28 including one skip), none retired, net +46.
