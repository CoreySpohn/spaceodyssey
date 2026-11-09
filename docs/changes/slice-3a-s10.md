# Slice 3a S10: the gate, the pruning pass and the budget check

`tests/test_slice3a_gate.py` runs the kill rules K1 to K8 and the acceptance map A1 to A10
as one command. It follows `tests/test_slice2b_gate.py`: evidence that already lives in a
test is invoked by alias or through a thin wrapper that supplies the shared run, and only
what no test held is built in the gate, next to its mutant. Every rule has checks whose
names start with its number, and each check fails when its evidence is empty (a pin file
with no entries, a commit range with no commits, a loop that ran zero times).

The whole file is marked slow in `tests/conftest.py` (`SLOW_FILES`), like the Slice 2b
gate, because it reruns the golden histories and four mutant runs. It stays out of the
quick tier. No test in it takes more than 12 s cold; the longest are the golden digests of
`refresh` on both executors.

## Cold run

One process, `SPACEODYSSEY_JAX_CACHE=off`, four threads, through `compute-run`.

| Measure | Value |
|---|---|
| Tests | 231 |
| Wall time | about 225 s |
| Programs (per-file count) | 1588 |
| Peak footprint | 2.6 GiB |
| Budget violations | 0 |
| Budget entry | `tests/test_slice3a_gate.py = 1837` (ceil(1588 * 1.15) + 10) |

The gate is slow, so the quick tier never collects it and the quick count is zero; the
full count is the larger of the two and sets the entry.

## Rule to evidence

Each mutant row names the check that must fail on it. A check is a test that passes only
when the mutant is caught, so a passing gate means the mutant failed the rule.

| Rule | Evidence invoked | Built here | Mutants and controls (kill result) |
|---|---|---|---|
| K1 earlier slices stay true | the golden digests of all 116 histories, the fingerprints and their evidence, the fault fixture, the corpus hashes and schema versions, the pin commit | a check that no commit after the pins touched a pin file | an asset-key change moves `phase_switch` floats (killed); an always-on empty leaf moves the fingerprint and the history ints (killed); a changed `N_TIES` moves `baseline` (killed); lane closure on a fault breaks the fault pin (killed); a one-bit digest change and a changed study miss their pins (killed) |
| K2 conservation | every account on every case, the forged balances, every award scenario and remainder rule, the refund across an expiry and a re-award, the paired forged expiry | none | a refund of the whole charge, a second receipt credit, an old award's refund credited to the newer award, and the paired expiry omission each fail conservation (killed) |
| K3 step cost | none (measured on the CI runner by `pilot.yml`) | the wiring check: thirteen handlers, the probe and its calibrated mutant, the workflow step, the measured excess in the S0 record | not measurable locally (see below) |
| K4 a survey needs no core patch | none | every core commit after S0 is listed under a task, the task record cites a core test, and no survey commit touches the core | an unlisted commit and a record that cites no test are reported; a core edit in a survey commit is flagged (killed) |
| K5 the Q22 frame | the domain-word and import lints over the core and their fixtures | none | each denied token and each import fixture is caught (killed) |
| K6 truth and release discipline | refund views invariant under a private event, survey 8 policy stop and its control, survey 10 awards ignore unreleased and observer-only proposals, a release moves its award | none | a visible internal `refund_of` and a truth-reading committee are caught (killed) |
| K7 quick-tier time | the summary script over its limits, the step's budget flags | the wiring check: no advisory baseline, the 540 s ceiling, medium tests counted, the budget plugin enforcing | measured on the runner (see below) |
| K8 action replay | the survey 10 replay of awards, `Allocate` effects and admission outcomes on both executors | none | dropping one award from the log fails the replay and the allocator check (killed) |

Mutants new in this change live in `tests/core/mutants.py`: an asset-keyed host executor,
a listing of empty conditional leaves, an opened conditional gate, and a changed `N_TIES`
that clears the compile caches. The gate's last test fails if a rule has no check or a
spec mutant is not tied to a check that exists.

## Acceptance map

| Item | Evidence |
|---|---|
| A1 charge components | `tests/core/test_charging.py` |
| A2 refunds | `tests/core/test_refund.py`, `tests/core/test_awards.py` |
| A3 awards | `tests/core/test_awards.py` |
| A4 frozen horizon | `tests/core/test_frozen_horizon.py` |
| A5 programs and Stop | `tests/core/test_programs.py`, and the twelve `stop_*` golden pins, checked in the gate |
| A6 conservation | `tests/core/test_conservation.py`, K2 |
| A7 snapshot balances | `tests/core/test_snapshot_balances.py` |
| A8 pre-flight | `tests/core/test_preflight.py`, `tests/codec/test_preflight_json.py` |
| A9 surveys | `tests/studies/test_survey_8.py` to `test_survey_11.py` |
| A10 earlier slices unchanged | K1 |

The gate checks that each file exists and still defines tests. Those files run in the quick
tier, so the gate does not rerun them.

## K4 attribution

The range is the last S0 commit to HEAD: 53 non-merge commits touch `core/`. Each is listed
under the task whose record describes it: S1, S2, S3a, S4, S5, S5b, S6, the Slice 2b
records, or the compile-speed records. The gate fails on a commit no list names, a stale
entry, or a record that cites no existing `tests/core` file. No survey commit (surveys 8 to
11, records and tests) touches `core/`.

What the gate does not prove, and prints instead of failing: the records describe behavior
and name a module only where it helps, so 34 touched paths are not named in their task's
record (module name or `package/` directory not found). They are listed by the gate on
every run. They fall in S3a (`transaction`, `enums`, `preflight/checks`, `core/__init__`,
the `truth_swap` and `replay_latency` batteries), S5 and S5b (`transaction`, `invariants`,
`executors/base`, `carry`, `handlers`, `records`, `snapshot`, `enums`, `preflight/findings`,
two batteries) and the compile-speed work (`observer_kit`, `refresh`). Two further
weaknesses: the compile-speed commits are attributed to their own records, not to a 3a
feature task, and their evidence is K1 (every history and pin bit-identical); and no record
carries a machine-readable list of core paths, so the check cannot be made stricter without
adding one.

## K3 and K7

K3 cannot be measured from inside pytest. The gate checks the wiring and prints a manual
read: dispatch `pilot.yml` on the final commit and compare the per-step minimum with the
eleven-handler baseline (limit 1.10). The S0 record holds the mutant's measured excess
(1.48 and 1.51), which shows the probe can fail. There is no K3 figure for the thirteen
handler table in this change.

K7 is enforced by the quick-tier step in `ci.yml`: the summary runs without
`--advisory-baseline`, with the 540 s ceiling, with medium tests counted
(`--require-tier medium`), and the budget plugin in enforce mode. The last warm figure is
CI run 42576344367 for commit ad113d3, job "Quick tier (py3.12)", success: 2681 tests in
391.2 s against the 540 s ceiling, 0 budget violations under enforce with
`SPACEODYSSEY_BUDGET_SCALE=3` and 229 compiles. The gate prints this line on every run and
fails if this record does not carry the run id and the time.

## Pruning pass

Method: for each family of 3a tests, compare what each member kills, not what it is named,
and delete only a test whose assertions a stronger test over the same object already makes.

Removed:

- `test_each_row_nets_debits_and_refunds` (5 cases, `tests/core/test_conservation.py`).
  It asserts that the final balance equals the budget less debits plus refunds in every row.
  `balance_conserved`, called on the same five runs by
  `test_every_account_agrees_after_pricing_cancels_and_refunds`, checks the same equation
  with the same 1e-9 tolerance (no awards, so the granted term is zero) and two further
  accounts. Every mutant the removed test killed fails the stronger one. Net 5 cases; no
  compiled program is removed because the runs are shared.

Reviewed and kept:

| Candidate | Evidence | Verdict |
|---|---|---|
| `test_mutant_a_refund_of_the_whole_charge_is_caught` and `..._applied_twice_is_caught` in `test_refund.py` against the two `test_mutant_..._fails_conservation` in `test_conservation.py` | the same two mutants on the same carry; the refund tests assert the credited amount, the conservation tests assert that the invariant fails, and only the refund file also kills the late-receipt and any-release mutants | kept: different object under test, and the conservation pair is the K2 negative control |
| `test_mutant_a_cancel_that_credits_the_surcharge_is_caught` in `test_charging.py` | the mutant on `_cancel_lane` was applied and the conservation runs and tests all passed on it (the books stay internally consistent); no conservation test kills it | kept: it is the only kill of that mutant |
| `test_dropping_a_recorded_award_fails_the_check` in `test_allocator.py` against the replay test in `test_survey_10.py` | the first calls `award_failures` on hand-built views and also kills an altered award amount; the second kills the drop through the full replay | kept: different function under test |
| the slow tests in `test_allocator.py` that survey 10 overlaps (the committee on the host, edits of the host run, the replay on both executors) | survey 10 replays its own study through `replay_allocation`; none of its tests is shown to kill every edit and perturbation these make on the allocator study | kept, as required: no strictly dominating test is shown |
| the `test_mutant_...` families in `test_snapshot_balances.py` | each targets a different changed function or flag (admission never refuses, refuses without the flag, counts every row, the view floats, the hidden refusal, the stale checks) | kept: no two share a kill set |

The removal is a separate commit so it can be reverted if a reviewer disagrees.
