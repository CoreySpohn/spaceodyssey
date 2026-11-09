# Slice 3a S10: the gate, the pruning pass and the budget check

`tests/test_slice3a_gate.py` runs the kill rules K1 to K8 and the acceptance items A1 to
A10 as one command. It does not alias other tests. It holds an explicit inventory: for each
rule and each acceptance item, the exact canonical test ids that carry the evidence,
mutant checks included. One test runs the whole inventory once in a child pytest process
(`python -m pytest <ids> -m "" -p no:cacheprovider --junitxml ...`) with the parent
environment, so `SPACEODYSSEY_JAX_CACHE` and the budget plugin apply to it. It requires
every listed id to be collected and to pass with the listed number of cases. A missing id,
a skip, an xfail, a failure or a different case count fails the gate and names the rule.

What no test held is built in the gate file next to its mutant (pin bytes, K3 ratio
check, K4, the K7 wiring, three K1 mutants). A recorder registered for the module checks
that each of those ran and passed in the parent process. The last test checks that every
rule and item has a listed check, that every own check passed, and that each spec mutant
maps to an inventory id that ran and passed in the child run or to an own check that
passed.

The whole file is marked slow in `tests/conftest.py` (`SLOW_FILES`), like the Slice 2b
gate, because it reruns the golden histories, four mutant runs and the acceptance files.
The quick tier deselects it.

## Cold run

One process, `SPACEODYSSEY_JAX_CACHE=off`, four threads, through `compute-run`.

| Measure | Value |
|---|---|
| Wall time, gate alone | 361.7 s (16 tests, of which 335.9 s is the one child run of the inventory; 697 inventory cases passed in the child, 334.7 s) |
| Peak footprint | 2.2 GiB sampled by `compute-run` |
| Gate file module programs | 237 measured |
| Budget entry | `tests/test_slice3a_gate.py = 283` in `[tool.testbudget.module_compile_budget_by_file]`, ceil(237 * 1.15) + 10 |

The budget plugin counts programs in the process that compiles them. The child pytest
process compiles the inventory tests and applies each source file's own budget (enforce
mode, caches cleared between files); the gate file's entry covers only the programs the
parent compiles (the child's 3718 programs are counted in the child process) for its own checks and the import of the package.

## Rule to evidence

Each mutant row names the check that must fail on it. A check is a test that passes only
when the mutant is caught, so a passing gate means the mutant failed the rule.

| Rule | Inventory evidence (child run) | Own checks (parent) | Mutants and controls |
|---|---|---|---|
| K1 earlier slices stay true | golden digests of all 116 histories, fingerprints and their evidence, fault fixture, corpus hashes and schema versions, pin commit; live runs must yield every world key | each pin file is byte-identical to the blob of its capture commit; no commit after the pins touched a pin file | asset-key change moves `phase_switch` floats; always-on empty leaf moves fingerprint and history ints; changed `N_TIES` moves `baseline`; lane closure on a fault breaks the fault pin; one-bit digest change and a changed study miss their pins |
| K2 conservation | every account on every case, forged balances, every award scenario and remainder rule, refund across expiry and re-award, paired forged expiry | none | whole-charge refund, second receipt credit, old award's refund credited to the newer award, paired expiry omission |
| K3 step cost | none (measured on the CI runner by `pilot.yml`) | wiring (13 handlers, probe, workflow input, S0 excess); ratio check | an extra handler pass per step (1.48 and 1.51 times the plain step) fails the ratio check |
| K4 a survey needs no core patch | none | every core commit listed under a task; each task record names the exact path and cites a core test; no survey commit touches the core | a synthetic history with a survey commit that edits the core is flagged; an unlisted commit and loose naming are caught |
| K5 the Q22 frame | domain-word and import lints over the core and their fixtures | none | each denied token and import fixture |
| K6 truth and release discipline | refund view under a private event, survey 8 policy stop, survey 10 awards ignore unreleased and observer-only, a release moves the award | none | visible internal `refund_of`, truth-reading committee, removed policy stop |
| K7 quick-tier time | the summary script over its limits, the step's budget flags | wiring: no advisory baseline, 540 s ceiling, medium tests counted, plugin enforcing | measured on the runner (below) |
| K8 action replay | survey 10 replay of awards, `Allocate` effects and admission outcomes on both executors; the allocator check | none | dropping one award from the log |

New mutants live in `tests/core/mutants.py`: an asset-keyed host executor, a listing of
empty conditional leaves, an opened conditional gate, and a changed `N_TIES` that clears
the compile caches.

## Acceptance items

Each item runs canonical ids in the child run, so a placeholder or skipped file fails.

| Item | Evidence run by the gate |
|---|---|
| A1 charge components | all of `tests/core/test_charging.py` |
| A2 refunds | the refund remainder tests, the refund across an expiry and re-award |
| A3 awards | every award scenario, every remainder rule, survey 10 release moves the award |
| A4 frozen horizon | all of `tests/core/test_frozen_horizon.py` |
| A5 programs and Stop | all of `tests/core/test_programs.py`; the twelve `stop_*` golden pins (own check) |
| A6 conservation | every account on every case |
| A7 snapshot balances | all of `tests/core/test_snapshot_balances.py` |
| A8 pre-flight | all of `tests/core/test_preflight.py` and `tests/codec/test_preflight_json.py` |
| A9 surveys | survey 8 policy stop, all of survey 9 and survey 11, survey 10 executors agree |
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
not name exactly (the full repository path, not a module name or a directory). To meet
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

K3 is pending CI measurement. It cannot be measured from inside pytest. The probe
`tools/step_cost.py` now takes `--baseline-per-step-us` and exits 1 when the plain
per-step minimum exceeds the baseline by more than 1.10; `pilot.yml` takes the input
`baseline_per_step_min_us` and prints the ratio and the mutant ratio in the job summary.
The exact read: dispatch `pilot.yml` on the final commit with `baseline_per_step_min_us`
set to the eleven-handler per-step minimum, measured in the same job (the S0 record
recommends interleaving the two tables, because separate jobs differ by more than 10
percent). The gate checks the wiring and that the ratio check fails the calibrated mutant
(the S0 record measured 1.48 and 1.51 locally). It does not claim a K3 result.

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

## Pin immutability

Git history shows which commits edited the pin files, and the gate checks that only the
two capture commits did. It also compares the bytes of each pin file with the blob at its
capture commit, after the child run, so a loader that rewrites a pin during the run is
seen. A test-loader change that regenerates pins in some other way (for example into a
different file the loader then reads) is a code-review concern, not a gate concern.

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
