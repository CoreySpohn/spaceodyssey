---
orphan: true
---

# Slice 1 acceptance gate

`pytest tests/test_slice1_gate.py` runs the complete gate, including negative
controls. Add `-rP` to display the separately captured `1e-12` go-band reports.
Nothing is deselected by the module. It uses the existing corpus,
conformance runner, parity comparator, lint tools, wake experiments, draw mutants,
payload layout and admission-index driver. Shared test helpers now live in
`tests/core/`; the original focused tests import them too. No simulation behavior
or stored hand history changes in this close-out.

## Acceptance checklist

Results below are for branch `slice-1-t12`. Gate names refer to
`tests/test_slice1_gate.py`; other paths identify the focused acceptance tests.

| Check | Evidence | Result |
| --- | --- | --- |
| A1: histories, records, balances and beliefs agree across executors | `tests/core/test_parity_corpus.py::test_ints_exact_and_floats_in_the_go_band`; gate `test_executor_parity[K4-parity]`, every corpus case, two worlds and each supported host policy form | Pass: all 28 cases in the go band |
| A2: host-log replay reproduces belief bit for bit | `tests/core/test_replay_log.py::test_replay_reproduces_belief_bitwise`, every non-overflow case | Pass: bitwise replay |
| A3: a fork preserves unrelated draws and follows declared coupling | `tests/core/test_fork.py::test_fork` and `test_fork_shares_later_draws_only_where_declared`, five acquisitions forked at index 2 | Pass: default and declared coupling |
| A4: a two-day release delay moves belief updates only on an absolute fixed plan | `tests/core/test_latency_shift.py::test_latency_shift` | Pass: two-day shift |
| K1: no study dispatch or untyped composition escape | `test_no_study_dispatch[K1a]`, `test_typed_composition[K1b]`, `test_negative_ast`; `test_study_has_no_untyped_fields` discovers every Abstract definition in the core, including unexported classes | Pass: clean core and all seeded AST/annotation controls |
| K2: core dependencies and vocabulary stay inside their boundaries; toys contain no owned physics | `test_core_boundaries[K2a]`, `test_negative_domain_lint`, `test_negative_import_lint`; channel and service review below | Pass: both lints, all location/import fixtures and ownership review |
| K3: corpus invariants hold | `test_corpus_invariants[K3]`, all 28 cases, both executors, two worlds each; the returned names must equal the independent ten-name list | Pass: all ten checks over all cases and both executors, with the log-overflow limits below |
| K4: executor parity and explicit overflow status | `test_executor_parity[K4-parity]` requires exact integers and floats within `1e-9`, reports the `1e-12` band separately; `test_overflow_status[K4-overflow]` covers queue, stable-id and log exhaustion on both executors | Pass: exact integers, all cases in the go band and all three overflow bits/statuses checked |

The ten required invariant names are `order_keys_monotone`,
`exactly_once_settlement`, `one_fold_per_acquisition`, `no_violation`,
`balance_conserved`, `unique_acquisition_id`, `dense_acquisition_index`,
`release_not_before_available`, `no_scorer_column_in_snapshot` and
`no_extra_wakes_under_private_events`. Event ordering includes the declared tie
rank before stable id. Acquisition uniqueness is by identity and revision, so
later revisions of the same acquisition remain legal.

Negative controls require the specific property to fail: the leaky wake rule
fails the named isolation invariant on both current executors. Across eight
worlds, the current host adds 80 wakes for either private scenario. The existing
`PerWakeHostExecutor` restores historical scheduling for a separate control,
paired with `historical_leaky_wake`, which also wakes on releases whose audience
the release handler filtered out. Proper audience filtering still adds zero
wakes, while this deliberately audience-blind rule adds 80
heartbeat wakes and 160 private-writer wakes. Both variants use the same original
scenarios and the same executor for baseline and experiment. Coalescing combines
a writer's same-tick process and release on the current host; the historical
control deliberately retains one wake per event and ignores the release-side
audience filter. Neither production protection is disabled in the current-executor
controls. Decision-count mutants
change shared measurement or policy draws on both executors; the leaking layout
fails the hidden-column invariant; the decrement mutant produces an observed
counter decrement and a non-dense admission sequence; each denied-word location
and both forbidden-import fixtures produce their own lint diagnostic. The AST
control plants forbidden comparisons and kind dispatch in temporary Python files.

## Channel and service ownership review

The inventory covers every channel-model implementation and host service defined
in `src/spaceodyssey/studies/`. The catalog table is listed separately because it
uses the same model for a different record kind. Ownership was checked against
the suite library ownership map; no other library implementation was inspected.

| Implementation and location | Toy prefix | Physics or service represented | Owning library | Why this toy is used; real physics present? |
| --- | --- | --- | --- | --- |
| `toy.ToyPhotometry`, measurement channels 10, 11 and 12 (and the unused default fourth row) | `ToyPhotometry`: yes | A noisy scalar measurement and its occupancy fraction | `jaxedith` for count rates and exposure times; `skyscapes` for source spectra; `optixstuff` for detector response | Prescribed dimensionless signals, gain and Gaussian noise isolate keys, release and accounting. There is no photon budget, spectral model, detector model or calibrated exposure calculation. No real physics. |
| `catalog.catalog_study`, channel 13, realized by `toy.ToyPhotometry` | `ToyPhotometry`: yes; there is no separate catalog model class | A delayed catalog product to an observer audience | `skyscapes` for catalog access (`skyscapes.io.from_exovista`) and for a JAX scene built from catalog data | A registered record kind, fixed delay and auxiliary sum exercise the open record protocol. No catalog is queried and no astrophysical property is calculated. No real physics. |
| `oracle.ToyOracleService` | `ToyOracleService`: yes | A host inference answer from released information | `photomancy` for beliefs, inference and evidence | Copies the snapshot mean, sample count and release version. It isolates Query/Result sequencing and audience checks without a posterior fit or scientific likelihood. No real physics. |

All channel-model and service implementation names start with `Toy`. There is
no separate model in `conformance.py`, `scripted.py`, `survey_25.py` or
`__init__.py`. `ScriptedRecords` supplies literal process records, while policies,
world sources and observers compose these fixtures; they introduce no additional
channel or host service. No channel is flagged as carrying real physics.

## Recorded-history changes across Slice 1

This inventory distinguishes a changed baseline from added canonical columns and
new reference cases. Each source is an existing change record, which identifies
the change without requiring an external development document.

| Change record | History, reference or expectation affected | Change and reason |
| --- | --- | --- |
| [Keys](slice-1-t3.md) | Recorded policy variates and draw-comparison expectations | Keys moved from decision count to phase, tick and ordinal. The old shifted-policy-draw expectation became zero changed draws, matched by tick and ordinal. Measurement variates and all 22 hand histories stayed unchanged. |
| [Transaction semantics](slice-1-t4.md) | `bundle` study, hand event history and hand rows | Channel 12 occupancy changed from 50 to 125 ticks: releases formerly at 185 and 260 now coincide at 260. The new 14-row, six-record history has no refusal. This intentionally retimed case exercises coalescing; the old timing remains in `test_wakes_at_distinct_ticks_stay_separate`. |
| [Transaction semantics](slice-1-t4.md) | Process-wake expectation, admission example and ordering reference | The process coincident with release at 130 now adds zero wakes, rather than one per world. The cumulative-admission example now uses separate groups because a same-group pair is atomic. `order_key_np` gained the coincidence tie rank. Reserved fields were added to canonical histories with default round trips; other hand histories stayed unchanged. |
| [Executors](slice-1-t5.md) | Canonical trace and final-state columns | Added handled tie, pending wake, capacity mask and run status, including the final pending wake. Explicit status distinguishes overflow, stall and truncation. No hand events, rows or flags were re-baselined. |
| [Records](slice-1-t6.md) | Record canonical columns and snapshot content | Added audience, revision and supersession to canonical histories. Raw noise became scorer-only, changing snapshots while keeping its logged payload position and values. No hand history changed. |
| [Host protocol](slice-1-t7.md) | Canonical fingerprints and release versions | Added `snapshot_fp`, `event_fp`, `release_version` and `release_version_final`. Existing cases issued no Query, so their hand event and row histories stayed unchanged. |
| [Pre-flight](slice-1-t8.md) | Existing histories | No handler, executor or table changed; no re-baseline. |
| [Generated battery](slice-1-t9.md) | Existing draws and histories | Default branch and coupling preserve existing keys; no re-baseline. |
| [Conformance corpus](slice-1-t10.md) | Six new hand event and row histories | Added `coalesced_release`, `bundle_unaffordable`, `forbidden_effect`, `catalog_record`, `host_service` and `revision`. These are new references, not replacements for the earlier 22 cases. |
| [Audience and sequencing corrections](fix-firewall.md) | Fingerprint integers, audience-specific beliefs, acceptance ticks and mixed-Query event order | Digest chaining changed recorded fingerprints. Observer views now use observer folds; policy-only rows retain observer acceptance tick -1. Mixed Queries handle Request and Result before siblings, sometimes moving sibling microsteps. Pending answers prevent drain completion. No stored hand history was re-baselined. |
| [Executor corrections](fix-executors.md) | Canonical integer identity map and replayed revision choice | Added handled microsteps, handled stable ids and record acquisition ids so parity detects identity corruption. Replay now uses exact log slots in release order, including reordered revisions. Existing hand event and row tuples keep their values. Default policy coupling changes no draw. |
| [Validation corrections](fix-checks.md) | Canonical `kind` and final `next_sid`; case assertions | Added fields make wrong record kinds and consumed ids on refused bundles detectable. The unaffordable bundle requires `next_sid == 3`; the catalog case checks its registered kind and auxiliary contribution. No hand history changed. The impossible-need fixture moved its declared decision from tick 50 to 19 because cancellation can produce a partial record before 50; this changes a validation fixture, not a run baseline. |
| [Study conformance and guide](slice-1-t11.md) | Existing histories, channel tables and world roots | No re-baseline. The hand-counted deterministic search and network reference actions are new tests. A second gradient seed supplements the original seed without changing its baseline. |

The supplied change-record directory has no separate records for the initial port
or the lint addition. The inventory makes no claim about unrecorded changes there.
The only replacement hand history recorded by these sources is the retimed bundle.

## Validation and limits

The final gate run reported `132 passed` (126 gate cases and the six original
leak tests); the 395 focused core tests also passed. Every corpus case was in the
`1e-12` go band, with a maximum absolute float difference of
`2.220446049250313e-16` and exact integers. The longest single case took about
27 seconds, so no slow marker is used. Ruff, the domain-word, core-import and
internal-reference checks pass; both core lints scan all 37 Python files.

The gate uses the existing conformance checks with their stated limits. If the
record log overflowed, exact balance reconstruction and the exact-density half of
the index check cannot account for discarded records; they report that condition
and still check retained bounds and finite balances. Stable-id and queue overflow
do not waive accounting. Compiled decision snapshots are not retained by this
runner, so its hidden-column invariant checks final snapshots; host runs also
check every captured decision snapshot. Private-event probes use ten fixed-period
events, rather than all possible private schedules. These limits were not expanded
into new runtime behavior in the close-out.
