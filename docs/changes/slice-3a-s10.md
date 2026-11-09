# Slice 3a S10: the gate, the pruning pass and the budget check

`tests/test_slice3a_gate.py` runs the kill rules K1 to K8 and the acceptance items A1 to
A10 as one command. It does not alias other tests. It holds an explicit inventory: for each
rule and each acceptance item, the exact canonical test ids that carry the evidence,
mutant checks included. One test runs the whole inventory once in a child pytest process
(`python -m pytest <ids> -m "" -n 0 -q -p no:cacheprovider --budget-json ... --junitxml ...`).
The child gets the parent environment without any `PYTEST_*` variable, so a marker or
keyword filter, a disabled plugin or xdist settings in `PYTEST_ADDOPTS` cannot reach it,
while `SPACEODYSSEY_JAX_CACHE` and the JAX settings still do. A second test reads the
child's budget file: the budget plugin must have loaded in enforce mode, with no
violation and a record for every module of the inventory. The junit parser counts each
node id once, so a duplicated entry adds nothing. Every listed id must be collected and
pass with the listed number of cases. A missing id,
a skip, an xfail, a failure or a different case count fails the gate and names the rule.

What no test held is built in the gate file next to its mutant (pin bytes, K3 ratio
check, K4, the K7 wiring, three K1 mutants). A recorder registered for the module checks
that each of those ran and passed in the parent process. The last test checks that every
rule and item has a listed check, that every own check passed, and that `MUTANT_CHECKS`
holds exactly the twelve mutant labels the spec names (`SPEC_MUTANTS`), each mapped to an
inventory id that ran and passed in the child run or to an own check that passed.

The whole file is marked slow in `tests/conftest.py` (`SLOW_FILES`), like the Slice 2b
gate, because it reruns the golden histories, four mutant runs and the acceptance files.
The quick tier deselects it.

## Cold run

One process, `SPACEODYSSEY_JAX_CACHE=off`, four threads, through `compute-run`.

| Measure | Value |
|---|---|
| Wall time, gate alone | 388.5 s (18 tests); the one child run of the inventory took 364.5 s for 819 cases |
| Peak footprint | 2.2 GiB sampled by `compute-run` (child run: 1.0 GiB peak resident, 0 budget violations) |
| Gate file module programs | 237 measured |
| Budget entry | `tests/test_slice3a_gate.py = 283` in `[tool.testbudget.module_compile_budget_by_file]`, ceil(237 * 1.15) + 10 |

The budget plugin counts programs in the process that compiles them. The child pytest
process compiles the inventory tests and applies each source file's own budget (enforce
mode, caches cleared between files); the gate file's entry covers only the programs the
parent compiles (the child's 3962 programs are counted in the child process) for its own checks and the import of the package.

## Rule to evidence

Each mutant row names the check that must fail on it. A check is a test that passes only
when the mutant is caught, so a passing gate means the mutant failed the rule.

| Rule | Inventory evidence (child run) | Own checks (parent) | Mutants and controls |
|---|---|---|---|
| K1 earlier slices stay true | golden digests of all 116 histories, fingerprints and their evidence, fault fixture, corpus hashes and schema versions, pin commit; live runs must yield every world key | each pin file is byte-identical to the blob of its capture commit; no commit after the pins touched a pin file | asset-key change moves `phase_switch` floats; always-on empty leaf moves fingerprint and history ints; changed `N_TIES` moves `baseline`; lane closure on a fault breaks the fault pin; one-bit digest change and a changed study miss their pins |
| K2 conservation | every account on every case, forged balances, every award scenario and remainder rule, refund across expiry and re-award, paired forged expiry | none | whole-charge refund, second receipt credit, old award's refund credited to the newer award, paired expiry omission |
| K3 step cost | none (measured on the CI runner by `pilot.yml`) | wiring (13 handlers, the probe's extra pass still runs, workflow input); the recorded CI figures and their ratio; the limit check | the extra handler pass per step costs 1.439 and 1.531 times the plain step inside its own job, above the 1.10 limit; a probe whose extra pass is removed fails the wiring check |
| K4 a survey needs no core patch | none | every core commit listed under a task; each task record names the exact path and cites a core test; no survey commit touches the core | a synthetic history with a survey commit that edits the core is flagged; an unlisted commit and loose naming are caught |
| K5 the Q22 frame | domain-word and import lints over the core and their fixtures | none | each denied token and import fixture |
| K6 truth and release discipline | refund view under a private event, survey 8 policy stop, survey 10 awards ignore unreleased and observer-only, a release moves the award | none | visible internal `refund_of`, truth-reading committee, removed policy stop |
| K7 quick-tier time | the summary script over its limits, the step's budget flags | wiring: no advisory baseline, 540 s ceiling, medium tests counted, plugin enforcing; the child run's budget file shows the plugin loaded and enforcing | measured on the runner (below) |
| K8 action replay | survey 10 replay of awards, `Allocate` effects and admission outcomes on both executors; the allocator check | none | dropping one award from the log |

New mutants live in `tests/core/mutants.py`: an asset-keyed host executor, a listing of
empty conditional leaves, an opened conditional gate, and a changed `N_TIES` that clears
the compile caches.

## Acceptance items

Each item runs canonical ids in the child run, so a skipped, missing or short file fails.

| Item | Evidence run by the gate |
|---|---|
| A1 charge components | the six charging tests by exact id (nine cases) |
| A2 refunds | all of `tests/core/test_refund.py` (deadline, cause, identity and release rules, and the deadline and release mutants); the window-refund matrix in `tests/core/test_awards.py` (late refunds disposed by the charging window, wrong-row and wrong-acquisition refunds, receipts that apply nothing, refunds before and after expiry, refund across an expiry and re-award) |
| A3 awards | every award scenario, every remainder rule, survey 10 release moves the award; window admission (grant opens the window, refusals outside and at the end tick, no balances, award onto an open window, re-award at the expiry tick); boundary ordering (window end event, completion before window end, expiry tie after a refresh); stale expiry after a re-award, and a pending acquisition at the expiry |
| A4 frozen horizon | all of `tests/core/test_frozen_horizon.py` |
| A5 programs and Stop | all of `tests/core/test_programs.py`; the twelve `stop_*` golden pins (own check) |
| A6 conservation | every account on every case |
| A7 snapshot balances | all of `tests/core/test_snapshot_balances.py` |
| A8 pre-flight and codec | all of `tests/core/test_preflight.py` and `tests/codec/test_preflight_json.py`; the optional-field round trips with populated refund, award, commitment and program columns (`tests/codec/test_optional_fields.py`, `tests/codec/test_run_document_features.py`) |
| A9 surveys | survey 8 policy stop and its host-versus-compiled parity test, all of survey 9 and survey 11, survey 10 executors agree |
| A10 earlier slices unchanged | the K1 inventory |

Survey 9 is entirely slow-marked: the quick tier selects none of it, so the gate is where
it runs on a pull request-sized change.

## K4 attribution

The range is the last S0 commit to HEAD: 53 non-merge commits touch `core/`. Each is
listed under the task whose record describes it. The spec allows feature tasks only. Two
other tasks changed the core inside the range: Slice 2b (its records, judged by the 2b
gate) and the compile-speed work (bit-identical, so the K1 pins are its evidence). Their
commits are attributed by their own task records, with K1 as the evidence; that is a
stated scope note, not an exception the gate hides.

The gate fails on a commit no list names, a stale entry, a record that cites no existing
`tests/core/test_*.py` file, and any touched core path that the record of its task does
not name exactly (the full repository path, not a module name, a directory, or a path
with a different prefix or suffix such as `.bak`). To meet
that, each responsible record now carries a "Core paths changed (K4)" section with the
exact paths and the core tests that cover them: `slice-3a-s1`, `slice-3a-s2`,
`slice-3a-s3a`, `slice-3a-s4`, `slice-3a-s5`, `slice-3a-s5b`, `slice-3a-s6`,
`slice-2b-r6` (for the Slice 2b commits R1 to R6) and `compile-speed-c` (for the
compile-speed commits). No survey commit (surveys 8 to 11, records and tests) touches
`core/`; the negative control runs the same discovery on a synthetic history that holds a
survey commit that edits the core.

Limit: the commit-to-task table is hand-listed in the gate, so a new core commit needs an
entry and a record update, by design.

## K3 and K7

K3 is measured on GitHub runners by `pilot.yml` (report-only). Two runs were dispatched
together, so the runners were comparable in time:

| Run | Commit | Handlers | Plain per-step minimum | Extra-pass mutant | Mutant ratio (same job) |
|---|---|---|---|---|---|
| 42584020950 | 97753d5 (S0 end) | 11 | 244.11 us | 355.99 us | 1.439 |
| 42584023997 | b851654 (s10-gate) | 13 | 181.28 us | 277.31 us | 1.531 |

K3 ratio: 181.28 / 244.11 = 0.743 against the 1.10 limit, so K3 passes. Caveat: the two
plain figures come from separate jobs on separate runners, which differ by more than the
10 percent limit from run to run (the S0 record measured that spread), so the ratio
carries runner noise. It is far enough below the limit that the noise does not change the
verdict. The evidence that the probe can see a regression is the mutant ratio inside
each job: the calibrated extra-pass mutant costs 1.439 and 1.531 times the plain step
measured next to it, above 1.10 in both jobs.

The mutant ratio is the one the probe printed. The probe measures its own plain step in
the mutant invocation, so dividing the two table figures gives 1.458 and 1.529, close to
the printed ratios and above 1.10 either way; the gate accepts a printed ratio within
0.05 of the quotient and requires both to exceed the limit.

The gate holds both runs as data (`K3_RUNS`) and fails if a figure is missing from this
record, if a recorded mutant ratio is far from its two figures or does not exceed the
limit, if the thirteen-handler step costs more than 1.10 times the eleven-handler step,
or if the probe's `ExtraPassExecutor.transition` no longer runs the extra
select-and-handle pass (an ast check, so a neutralized mutant fails the wiring test).
The probe `tools/step_cost.py` takes `--baseline-per-step-us` and exits 1 when the plain
per-step minimum exceeds the baseline by more than 1.10; `pilot.yml` takes the input
`baseline_per_step_min_us`. The gate cannot re-measure on a runner; a re-measurement
means dispatching the workflow again and updating `K3_RUNS` and this table.

K7 is enforced by the quick-tier step in `ci.yml`: the summary runs without
`--advisory-baseline`, with the 540 s ceiling, with medium tests counted
(`--require-tier medium`), and the budget plugin in enforce mode. The last warm figure is
CI run 42576344367 for commit ad113d3, job "Quick tier (py3.12)", success: 2681 tests in
391.2 s against the 540 s ceiling, 0 budget violations under enforce with
`SPACEODYSSEY_BUDGET_SCALE=3` and 229 compiles. It is a measurement of that commit, not of
the final one. The gate prints this line on every run and fails if this record does not
carry the run id and the time.

## Slow nightly tier

Section 7 asks for the status of the slow tier. The scheduled `CI` runs on main, newest
first: 2026-11-08 (run 42489079500) cancelled during the slow tier after about 13
minutes; 2026-11-07 (42379525028) cancelled; 2026-11-06 (42239283717) cancelled;
2026-11-05 (42055076555) failed. No scheduled run in that window completed green, so there
is no successful slow-tier run to cite. The gate does not depend on the nightly: it runs
its own slow evidence.

## Known limits of a node id inventory

Three weakenings stay open by design. A gate that lists node ids and counts passed cases
cannot see them; each is a code-review concern.

- Count-matched placeholder: a test file whose body is replaced by a parametrized
  placeholder with the same number of cases passes under the same node ids.
- Trivial inventory entry: an entry that names a test that asserts nothing (a listed
  function whose body was emptied) passes.
- Pin regeneration to identical bytes: a loader that rewrites a pin file with
  byte-identical content leaves nothing to detect. Regeneration that changes the bytes is
  caught by the byte comparison with the capture blob.

## Pin immutability

Git history shows which commits edited the pin files, and the gate checks that only the
two capture commits did. It also compares the bytes of each pin file with the blob at its
capture commit, after the child run, so a loader that rewrites a pin during the run is
seen. A test-loader change that regenerates pins into a different file the loader then reads
is also a code-review concern, not a gate concern.

## Pruning pass

Method: for each family of 3a tests, compare what each member kills, not what it is named,
and delete only a test whose assertions a stronger test over the same object already makes.

Removed:

- `test_each_row_nets_debits_and_refunds` (5 cases, `tests/core/test_conservation.py`).
  It asserted that the final balance equals the budget less debits plus refunds in every
  row, with `assert_allclose` at its default relative tolerance of 1e-7 and an absolute
  1e-9. `balance_conserved`, called on the same five runs by
  `test_every_account_agrees_after_pricing_cancels_and_refunds`, checks the same equation
  against an absolute 1e-9 bound (no awards, so the granted term is zero) and two further
  accounts. The survivor is stricter: an error of 1e-6 passed the removed test and fails
  the survivor. Every mutant the removed test killed fails the stronger one. Net 5 cases;
  no compiled program is removed because the runs are shared.

Reviewed and kept:

| Candidate | Evidence | Verdict |
|---|---|---|
| `test_mutant_a_refund_of_the_whole_charge_is_caught` and `..._applied_twice_is_caught` in `test_refund.py` against the two `test_mutant_..._fails_conservation` in `test_conservation.py` | the same two mutants on the same carry; the refund tests assert the credited amount, the conservation tests assert that the invariant fails, and only the refund file also kills the late-receipt and any-release mutants | kept: different object under test, and the conservation pair is the K2 negative control |
| `test_mutant_a_cancel_that_credits_the_surcharge_is_caught` in `test_charging.py` | the mutant on `_cancel_lane` was applied and the conservation runs and tests all passed on it (the books stay internally consistent); no conservation test kills it | kept: it is the only kill of that mutant |
| `test_dropping_a_recorded_award_fails_the_check` in `test_allocator.py` against the replay test in `test_survey_10.py` | the first calls `award_failures` on hand-built views and also kills an altered award amount; the second kills the drop through the full replay | kept: different function under test |
| the slow tests in `test_allocator.py` that survey 10 overlaps (the committee on the host, edits of the host run, the replay on both executors) | survey 10 replays its own study through `replay_allocation`; none of its tests is shown to kill every edit and perturbation these make on the allocator study | kept, as required: no strictly dominating test is shown |
| the `test_mutant_...` families in `test_snapshot_balances.py` | each targets a different changed function or flag (admission never refuses, refuses without the flag, counts every row, the view floats, the hidden refusal, the stale checks) | kept: no two share a kill set |

The removal is a separate commit so it can be reverted if a reviewer disagrees.
