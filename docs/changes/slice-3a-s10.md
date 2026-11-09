# Slice 3a S10: the gate, the pruning pass and the budget check

`tests/test_slice3a_gate.py` runs the kill rules K1 to K8 and the acceptance items A1 to
A10 as one command. It does not alias other tests. It holds an explicit inventory: for each
rule and each acceptance item, the exact canonical test ids that carry the evidence,
mutant checks included. One test runs the whole inventory once in a child pytest process
(`python -m pytest <ids> -m "" -n 0 -q -p no:cacheprovider --budget-json ... --junitxml ... -o xfail_strict=true --timeout 900`).
The child gets the parent environment without any `PYTEST_*` variable, so a marker or
keyword filter, a disabled plugin or xdist settings in `PYTEST_ADDOPTS` cannot reach it,
while `SPACEODYSSEY_JAX_CACHE` and the JAX settings still do. A completeness check
at the end of the module fails the run if any check of the gate did not run and pass
(unless the run was narrowed with `-k`, `--deselect` or a node id), so deleting or
disabling a gate test is seen. A second test reads the
child's budget file: the budget plugin must have loaded in enforce mode, with no
violation and a record for every module of the inventory. The junit parser counts each
node id once, so a duplicated entry adds nothing. Every listed id must be collected and
pass with the listed number of cases. A missing id,
a skip, an xfail, a test that passes under a non-strict xfail (the child runs with `xfail_strict=true`, so it fails), a failure or a different case count fails the gate and names the rule.

What no test held is built in the gate file next to its mutant (pin bytes, K3 ratio
check, K4, the K7 wiring, three K1 mutants). A recorder registered for the module checks
that each of those ran and passed in the parent process. The last test checks that every
rule and item has a listed check, that every own check passed, and that `MUTANT_CHECKS`
holds exactly the twelve mutant labels of the slice's acceptance list (`SPEC_MUTANTS`), each mapped to an
inventory id that ran and passed in the child run or to an own check that passed.

The whole file is marked slow in `tests/conftest.py` (`SLOW_FILES`), like the earlier
premission-slice gate, because it reruns the golden histories, four mutant runs and the acceptance files.
The quick tier deselects it.

## Cold run

One process, `SPACEODYSSEY_JAX_CACHE=off`, four threads, through `compute-run`.

| Measure | Value |
|---|---|
| Wall time, gate alone | 434.3 s (18 tests); the one child run of the inventory took about 404 s for 826 cases |
| Peak footprint | 2.7 GiB sampled by `compute-run` (0 budget violations in the gate and in the child run) |
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
| K3 step cost | none (measured by `pilot.yml` in one job for both trees) | wiring (13 handlers, the probe's extra pass still runs, the pair step and its exit status in the workflow); the same-job record or its pending state; the limit check | the extra handler pass per step costs 1.439 and 1.531 times the plain step inside its own job in earlier runs, above the 1.10 limit; a probe whose extra pass is removed fails the wiring check |
| K4 a survey needs no core patch | none | every core commit listed under a task; each task record names the exact path and cites a core test; no survey commit touches the core | a synthetic history with a survey commit that edits the core is flagged; an unlisted commit and loose naming are caught |
| K5 the domain-word frame | domain-word and import lints over the core and their fixtures | none | each denied token and import fixture |
| K6 truth and release discipline | refund view under a private event, the policy view unchanged under an observer-only release and a private event's identity hidden in the policy fold, survey 8 policy stop, survey 10 awards ignore unreleased and observer-only, a release moves the award | none | visible internal `refund_of`, truth-reading committee, removed policy stop |
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
| A3 awards | every award scenario, every remainder rule, survey 10 release moves the award; window admission (grant opens the window, refusals outside and at the end tick, no balances, award onto an open window, re-award at the expiry tick); boundary ordering (window end event, completion before window end, expiry tie after a refresh); stale expiry after a re-award, and a pending acquisition at the expiry; the host and compiled executors making the same history of an award, a completion after the expiry still settling, and the three allocator-study slow tests that carry `Allocate` admission (the committee on the host, on both executors, and the altered-run replay) |
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

The range runs from the last S0 commit to `K4_END`, `ad113d3`, the last commit before this task, so a later commit cannot widen or break it: 53 non-merge commits touch `core/`. Each is
listed under the task whose record describes it. The slice allows feature tasks only. Two
other tasks changed the core inside the range: the earlier premission work (its records, judged by its own gate) and the compile-speed work (bit-identical, so the K1 pins are its evidence). Their
commits are attributed by their own task records, with K1 as the evidence; that is a
stated scope note, not an exception the gate hides.

The gate fails on a commit no list names, a stale entry, a record that cites no existing
`tests/core/test_*.py` file that some commit of the range changed, and any touched core path that the record of its task does
not name exactly (the full repository path, not a module name, a directory, or a path
with a different prefix or suffix such as `.bak`). To meet
that, each responsible record now carries a "Core paths changed (K4)" section with the
exact paths and the core tests that cover them: `slice-3a-s1`, `slice-3a-s2`,
`slice-3a-s3a`, `slice-3a-s4`, `slice-3a-s5`, `slice-3a-s5b`, `slice-3a-s6`,
`slice-2b-r6` (for the earlier premission commits R1 to R6) and `compile-speed-c` (for the
compile-speed commits). No survey commit (surveys 8 to 11, records and tests) touches
`core/`; the negative control runs the same discovery on a synthetic history that holds a
survey commit that edits the core.

Limits. The commit-to-task table is hand-listed in the gate, so a new core commit needs an
entry and a record update, by design. The cited-test check does not tie the cited file to
the task's own commits: the first task's tests landed in other commits, so a stronger
tie would fail it. A record could therefore cite a core test that another task changed;
that part is a code-review concern.

## K3 and K7

### K3: measured in one job, ratio 0.805

K3 passes. Pilot run 42601136720 (commit a560033, job "Step cost and cold compiles",
success) measured the S0 end commit 97753d5 (eleven handlers) and this commit (thirteen
handlers) in one job, interleaved over five rounds of 30 repeats:

| Tree | Handlers | Plain per-step minimum | Extra-pass mutant | Mutant ratio | Cold first call, plain | Cold first call, mutant |
|---|---|---|---|---|---|---|
| base, 97753d5 | 11 | 99.11 us | 150.67 us | 1.526 | 8.60 to 8.75 s | 10.56 to 10.67 s |
| final, a560033 | 13 | 79.76 us | 126.83 us | 1.618 | 5.42 to 5.51 s | 6.62 to 6.70 s |

The ratio of the final minimum to the base minimum is 0.805, against the 1.10 limit; the
per-round ratios run from 0.787 to 0.807, so the round-to-round spread is about 2
percent, well inside the limit. The extra-pass mutant costs 1.526 and 1.618 times the
plain step of its own tree, above 1.10, so the probe resolves a regression of the size
K3 guards against. The thirteen-handler table is faster per step than the eleven-handler
one because of the compile and step refactors that landed between S0 and S10; every pin
stayed bit-identical through them (K1).

The earlier measurement below does not decide K3. Two runs were dispatched together, but
each is a separate job on a separate runner. Runner speed differs from job to job by more
than the 10 percent limit, so the quotient of their plain figures (181.28 / 244.11 =
0.743) is dominated by that difference, and K3 has no statistical power from it. An
earlier version of this record called the 0.743 a large margin; that framing was wrong
and is withdrawn. The two runs are kept as context only:

| Run | Commit | Handlers | Plain per-step minimum | Extra-pass mutant | Mutant ratio (same job) | Cold first call, plain | Cold first call, mutant |
|---|---|---|---|---|---|---|---|
| 42584020950 | 97753d5 (S0 end) | 11 | 244.11 us | 355.99 us | 1.439 | 11.59 s | 14.279 s |
| 42584023997 | b851654 (s10-gate) | 13 | 181.28 us | 277.31 us | 1.531 | 9.632 s | 10.592 s |

What those runs still show is the sensitivity of the probe: inside each job the
calibrated extra-pass mutant costs 1.439 and 1.531 times the plain step measured next to
it, above 1.10. The printed mutant ratio comes from the probe's own plain step in the
mutant invocation, so dividing the two table figures gives 1.458 and 1.529; the gate
accepts a printed ratio within 0.05 of the quotient (for the recorded same-job figures)
and requires both to exceed the limit.

The same-job measurement is `tools/step_cost_pair.py`, run by `pilot.yml` after the
cold medium tests. The workflow checks out the S0 end commit (97753d5, the input
`k3_base_ref`) into `base-tree` and runs, for five rounds, the base tree's own
`tools/step_cost.py --mutant` against the base tree's package and then this tree's
`tools/step_cost.py --mutant` against this tree's package, each in a fresh process with
the same repeats (30). Each tree is therefore measured with its own copy of the probe,
not the final tool pointed at the base package; the two copies differ only in the
removed typed-baseline option. The ratio is computed in the job: the final tree's lowest
per-step minimum over the base tree's, with the per-round ratios printed as well; the
step exits 1 above 1.10, and its output is a JSON file shown in the job summary. Cold
compile of the step is reported separately from the per-step cost (the cold first calls
in the table above, per-round in the JSON).

The gate takes the figures from `K3_SAME_JOB` (run id, base and final commits, rounds,
both minima, the ratio, both mutant ratios): it requires the ratio to follow from the two
minima, to be at most 1.10, and both mutant ratios to exceed 1.10, and it requires the
figures in this record. The gate also fails if the
thirteen-handler step table is not the one K3 is judged at, if the probe's
`ExtraPassExecutor.transition` no longer runs the extra select-and-handle pass (an ast
check), if the pair tool stops alternating the trees, or if the workflow loses the pair
step, the base checkout or the exit status of a measurement (`set -o pipefail` before a
pipe into `tee`, which would otherwise hide a failed measurement). The gate cannot
re-measure on a runner.

K7 is enforced by the quick-tier step in `ci.yml`: the summary runs without
`--advisory-baseline`, with the 540 s ceiling, with medium tests counted
(`--require-tier medium`), and the budget plugin in enforce mode. The last warm figure is
CI run 42576344367 for commit ad113d3, job "Quick tier (py3.12)", success: 2681 tests in
391.2 s against the 540 s ceiling, 0 budget violations under enforce with
`SPACEODYSSEY_BUDGET_SCALE=3` and 229 compiles. It is a measurement of that commit, not of
the final one. The gate prints this line on every run and fails if this record does not
carry the run id and the time.

## Slow nightly tier

Scheduled `CI` runs on main, newest first: 2026-11-08 (run 42489079500) cancelled during
the slow tier after about 13 minutes; 2026-11-07 (42379525028) cancelled; 2026-11-06
(42239283717) cancelled; 2026-11-05 (42055076555) failed. No scheduled run in that window
completed green, so there is no successful slow-tier run to cite. The gate does not
depend on the nightly: it runs its own slow evidence.

Whether the gate fits the nightly. The slow job runs `pytest tests/ -m slow --timeout=900`
with a 90-minute job limit. The gate's child run is about 370 s on a laptop and about
1100 s at the runner's three-fold slowdown, and pytest-timeout charges a fixture's setup
to the first test that uses it, so the default 900 s would kill the gate in its child
fixture on a runner. The gate file therefore carries (when `pytest-timeout` is installed, as CI does through the `test` extra) a `timeout` mark of 2400 s on every
test (above three times the child's cold time with margin; a check asserts the relation),
and the child gets its own `--timeout 900` per case, as the nightly gives each test. The
nightly also runs every inventory test a second time on its own, in the job's normal
pass over the slow tests; that duplicate cost is accepted and not engineered around.

Manual dispatch of the slow tier: the `CI` workflow (`ci.yml`) has a `workflow_dispatch`
trigger with one input, `tier`, a choice of `not slow` (the default), `slow` or `all`.
To run the slow tier on this branch, dispatch `CI` on ref `s10-gate` with `tier=slow`.

That run (42601139487, commit a560033) was cancelled at the 90-minute job limit while it
was still in `tests/test_slice1_gate.py`, about 89 percent through the slow selection, one
error already reported there; it never reached this gate. The slow tier on main has the
same problem: it does not finish inside 90 minutes, which is why the scheduled runs above
were cancelled. That is a slow-tier budget problem older than this slice, and it is
reported here, not fixed. The gate's runner time is therefore unmeasured; its local cold
time is 434 s, inside the 2400 s timeout at the runner's slowdown.

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
