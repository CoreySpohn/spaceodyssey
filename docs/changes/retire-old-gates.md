# Retire the Slice 1, 2 and 2b gates

A slice gate retires when the next slice merges, and its lasting checks move to the regular
suite. The merge-time process checks (commit ranges, path allowlists, pin capture history,
inventories, measured ratios) ran once at merge, and their results are recorded in the
earlier change records. The three gate files are deleted: `tests/test_slice1_gate.py`,
`tests/test_slice2_gate.py` and `tests/test_slice2b_gate.py`. `tests/test_slice3a_gate.py`
stays as the current gate.

## Deleted as duplicates

Each of these reran, or aliased, a check that a regular test file already holds.

- Slice 1 gate: the study-dispatch scan of the core (`test_lints.py` scans a copy of the core
  and checks the allowlist for stale entries), typed composition (`test_contracts.py`), the
  core boundary lints (`test_lints.py`), the replay acceptance runs (`test_replay_log.py`), the fork
  and latency checks (`test_fork.py`, `test_latency_shift.py`), executor parity (`test_parity_corpus.py`
  holds the stricter 1e-12 band), the overflow status check (`test_run_status.py`), the leaky
  wake invariant (`test_corpus_invariants.py`), the current-executor leaky wake counts
  (`test_leak.py`), the draw count-key controls (`test_draws.py` asserts the same outcome without
  pinning the number of changed draws), the decrement mutant (`test_acquisition_index.py`),
  the domain-word and import lint fixtures (`test_lints.py`) and the ordinary-comparison
  case of the dispatch scan.
- Slice 2b gate: the K1, K3, K4 and K5 aliases and the two named interface aliases, which
  reran the golden, premission, codec, run-document and lint tests under other names.
- Three meta tests in `tests/core/test_contracts.py` that only checked the wiring of a gate
  file: the replay mutation of the gate, and the presence of its fork and latency tests.

## Deleted as merge-time process checks

- Slice 2b K2: the commit-range path allowlist, the other-task commit list, the merge-commit
  check and the allowlist control. Its range is open-ended, so the allowlist test fails on the
  current history; it ran at the merge.
- The Slice 2b pin capture-history check, which Slice 3a repeats from its own range.
- The check that no named interface check is pending.
- The CI step text check of the quick tier. Slice 3a K7 reads the same step.

## Moved to the regular suite

| From | To | What |
| --- | --- | --- |
| Slice 1 `test_corpus_invariants` | `tests/core/test_corpus_invariants.py` | The existing corpus test now judges every run against an independent invariant list, an independent snapshot layout and the world numbers 0 and 1. It replaces the version that compared the invariant names with `INVARIANTS` itself. |
| Slice 1 leaking payload and layout leak | same file | One test, on both executors, that a layout showing the raw variate fails the snapshot invariant against the independent layout. The end-to-end variant that reran the case with a leaking study is folded into it (see below). |
| contracts meta test of the world check | same file | The corpus test rejects missing, repeated and renumbered worlds. |
| Slice 1 replay negative controls | `tests/core/test_replay_log.py` | Three tests: a replay that folds nothing, one that returns the live belief, and one that folds in log order each fail. |
| Slice 1 historical leaky wake counts | `tests/core/test_leak.py` | The per-event executor with the historical leaky wake rule adds 80 and 160 wakes. |
| Slice 1 dispatch scan cases | `tests/core/test_lints.py` | Eight cases that pin the rule and the line of a reported branch, including the import alias and the local alias. |
| Slice 1 parity tolerance pin | `tests/core/test_parity_corpus.py` | The two tolerance bands are pinned to their values. |
| Slice 2 repository scanner | `tests/legacy_scan.py` (helper), `tests/test_docs_names.py` (tests) | The scan for references to removed modules over the whole repository, its negative controls and its coverage checks. |
| Slice 2b summary script checks | `tests/test_ci_summary.py` | The CI summary script fails over its budget, baseline and tier limits, and its advisory mode. Slice 3a K7 lists the new ids. |
| contracts NaN control | `tests/core/test_contracts.py` | Retargeted to the regular draw tests. |

`tests/pin_helpers.py` holds the pin constants and git helpers that the Slice 3a gate read from
the Slice 2b gate. In the Slice 3a gate, the two Slice 2b ids of K7 are replaced by the moved
summary script test (4 cases); the step text check is deleted because the gate's own K7 check
reads the step. K7 still has a listed check and the gate's mutant and completeness checks hold.

## Judgment calls

- The end-to-end layout leak test (a second run of the baseline case with a leaking study on
  each executor) was folded into the invariant-level test. It checked the same defect and cost
  about 11 s and a second program set.
- The dispatch scan cases and the historical leaky wake counts are kept because no regular test
  pins the rule name and line, or the per-event historical control.

## Time and budgets

Measured once each on a laptop under the compute guard, whole file, `-m ""`, before the change:
the Slice 1 gate took 317 s (166 tests), the Slice 2 gate 1.2 s (30 tests) and the Slice 2b
gate 145 s (153 tests, one failing: the K2 range test). The Slice 1 and Slice 2b files were in
the slow tier, so the slow tier loses about 460 s of wall time on this machine. Added back, measured on the moved tests: about 14 s
for the two historical leaky wake cases, 4 s for the replay controls and under 1 s for the
rest in the slow files, and about 3 s in the unit tier (the repository scan and the lint cases).
The net slow-tier change is about 440 s less.

Program budgets (`[tool.testbudget.module_compile_budget_by_file]`, rule ceil(n x 1.15) + 10 from
the measured count): the three gate entries (3768, 10 and 3477) are removed, and
`tests/test_ci_summary.py` gets 10. Only `tests/core/test_corpus_invariants.py` grew past its
entry: it now loads 1434 programs, so its budget moves from 1425 to 1657. Every other receiving
file stays under its current entry: replay log 279 of 334, leak 87 of 98, parity 320 of 382,
contracts 117 of 150. Each receiving file ran alone with the budget plugin in enforce mode and
no violation.

`tests/conftest.py` drops the two gate entries from `SLOW_FILES` and the
executor-family xdist group that only the Slice 1 gate used. `.pre-commit-config.yaml` adds
the two new support modules to the test naming exclusion.

## Slow tier on the runner

With these changes and the slow tier on two workers (`-n 2 --dist loadfile`), the dispatched
run 42641435989 finished the slow tier for the first time since its job limit started cutting
it off: 1206 passed in 3822 s (64 min), where earlier runs were cancelled at 90 min. One budget
failed: `tests/core/test_corpus_invariants.py` loads 2341 programs under the full slow selection
on the runner, above the 1657 measured on the quick selection alone, so its budget is now
ceil(2341 x 1.15) + 10 = 2703.

The largest single cost was the current slice gate: 1292 s of setup in
`tests/test_slice3a_gate.py`, where it reruns its inventory of checks that the same tier
already runs in their own files. The scheduled run now skips that file; a manual slow or full
run still includes it. Without it the two workers carry about 6350 s of work, an estimated
53 min of wall time, still above the 45 min target; the next largest costs are the physical
OFTI survey (312 s over three tests) and the gradient checks (about 250 s).
