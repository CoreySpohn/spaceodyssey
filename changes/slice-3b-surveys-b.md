---
orphan: true
---

# Slice 3b surveys 12, 22 and 24

Three survey studies on the Slice 3b features, with no change under `core/`. Each runs on
the host executor, one compiled structure per survey, and is marked slow.

## Survey 12: a failure sets the ops state and a released record reports it

`studies/survey_12.py`: asset 0 fails at tick 100 (`OpsFailure` sets ops state 1 and
writes a notice released 25 ticks later; the double moved here from `tests/doubles.py`,
which re-exports it). A version-2 model realizes with the noise floor of the stored
admission state. `FailureWatch` is a level-1 policy with leaf switches: `react` (act on
the notice), `read_balance` (act on a debit that differs from the first seen) and
`cancel_tick`.

- (a) Fixed plan, unequal costs (2 and 5): of three admissions, the two after the failure
  are in state 1, debited 5 each and realized with floor 0.5 (counted: 2 after, 1 before;
  every residual equals floor times variate). The admission cancelled at tick 150 keeps
  `HIGH * 35/150` and the ledger refund is the rest of the 5 (conservation).
- (b) The notice's release tick equals failure plus delay (125); the fixed plan never
  changes its action.
- (c) Equal costs, record reader: the first changed action is at tick 125, the release.
  Positive control: delay 50 moves it to 150 (later by exactly 25). Negative control:
  a balance-reading policy at unequal costs changes its action at tick 110, before the
  release, so the check "first change not before the release" fails for it. A record
  reader at unequal costs still acts at 125.

## Survey 22: a table swap moves the target and nothing the policy read

`studies/survey_22.py`: the world source draws one stream per purpose and a trajectory
table under its own purpose; a version-2 channel writes the table position at the start
tick. Two truths that differ only in the table: the streams are equal, all four realized
target positions differ and equal the table at their start ticks; every policy view
before the first release of a target record is identical (both the fingerprints and the
full views; 2 decisions compared), and the full views differ afterwards.

Negative control: the leaky variant adds finder channel 12 (own lane, transition 0,
occupancy 0, latency 0), realized and released at tick 0 with the table position as its
payload. Its payload differs across the truths, and the views before the target's first
release differ, so the invariance check fails as designed.

Finding: the snapshot fingerprint hashes integer content only, so the leak is not visible
to a fingerprint comparison (it is equal across the two truths in the leaky variant). The
survey therefore compares the full views too, and the test asserts the fingerprints agree
in the control to document this.

## Survey 24: a latent is drawn under its own purpose and shown only by a record

`studies/survey_24.py` reuses the world source and channel of survey 22.

- (a) Changing the salt of any of the five other purposes leaves the table unchanged and
  moves only that purpose's stream; changing the table's own salt moves the table and no
  stream.
- (b) Across two truths that differ only in the latent, no policy view (fingerprint and
  full) and no observer belief step before the first release differs; the first Done
  precedes that release.
- (c) Positive control: the belief first differs exactly at the release step.

## Tests and cost

| File | Tests | Cold programs | Budget | Warm wall |
| --- | --- | --- | --- | --- |
| `test_survey_12.py` | 3 (5 host runs) | 73 | 94 | 3 s |
| `test_survey_22.py` | 2 (4 host runs) | 64 | 84 | 2 s |
| `test_survey_24.py` | 2 (2 host runs) | 68 | 88 | 2 s |

About 10 s of cold wall for the three on a laptop. The survey helpers for the two-truth
runs went into `tests/studies/survey_helpers.py`.
