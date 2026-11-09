# Slice 1, T9: generated test battery

## Behaviour changes

- `spaceodyssey.core.battery` adds checks that take a study and return a `CheckResult`
  (`name`, `passed`, `diagnostic`, `detail`; truthy when the property holds). Each check
  runs the study through an executor (`executor=None` means the compiled one) and fails
  when it would pass vacuously.

| Check | Property | Assumption stated in its docstring |
|---|---|---|
| `null_reference` | a world with nothing in it earns exactly the closed-form credit, or a mean credit inside the false-alarm interval (`false_alarm_interval`) | the null worlds hold nothing the objective counts; the study observes in every world |
| `truth_swap` | with two independent truths, every trace field is bitwise equal up to the first released record that differs, and the observer belief up to the first such release to anyone | the swap changes some released record; truth reaches the policy only through releases |
| `latency_shift` | a release delay moves belief updates and nothing else on a fixed plan | acquisitions start at ticks that do not depend on releases; the delay is shorter than the plan spacing |
| `fork_check` | a fork at an acquisition index keeps the prefix bitwise, leaves every trunk draw equal to the draw of its coordinates, and shares later draws only where the coupling says | one subject, complete acquisitions, the channel model draws only from its key |
| `relabel_ids` | relabeling channel ids changes no event, admission, record or ledger value | every reference to an id is relabeled too; nothing orders ids by value; variates are keyed by id, so payloads and beliefs are not compared |
| `shift_epoch` | starting the whole run later moves every tick by the shift and changes nothing else | no component reads an absolute tick; the policy draw is keyed by its decision tick and is not compared |
| `zero_time_asset` | adding an asset with no balance and an idle lane changes nothing | nothing counts assets or lanes |
| `declared_matches_run` | every stop a policy asks for is declared for its phase, and every declared need is met (an acquisition starts at its start tick, a decision runs at its decision tick, the record is released by then) | a declared stop that a run never reaches is reported, not failed |

- Policy draws after a fork are checked on the keys (`key_report`), because the policy
  key has a lineage and no branch coordinate and the executors give a decision no lineage.
  Measurement draws are checked in the run.
- Core edit, needed because the handlers never passed a branch or a coupling to
  `draw_key`: `keys.Fork` (a start index and a branch; the default forks nothing),
  `Study.fork` and `Study.coupling` (defaults: no fork, `Coupling()`), and the measurement
  draw in `handlers._realize_rec` reads them. A zero branch leaves every key unchanged, so
  no logged draw moved.

## Tests

Each check has its tests and its negative controls (broken executors are the real
handlers or transaction cloned with one name rebound, in `tests/core/battery_helpers.py`).

| Check | Negative control | Caught |
|---|---|---|
| `null_reference` (`test_null_reference.py`) | an observer that adds a prior to every record, exact and interval modes | yes, both |
| `truth_swap` (`test_truth_swap.py`) | a snapshot that shows written but unreleased records, with a policy that reads values | yes, in `eff_kind` before the first release |
| `latency_shift` (`test_latency_shift.py`) | measurement draws also keyed by the release tick; a plan that reacts to releases | yes, both |
| `fork_check` (`test_fork.py`) | draws that ignore the branch; a branch applied before the fork; key functions that never separate a fork | yes, all three |
| `relabel_ids`, `shift_epoch`, `zero_time_asset` (`test_metamorphic.py`) | a policy that picks the smallest id; one that reads the tick; one that reads the asset count | yes, each |
| `declared_matches_run` (`test_declarations_match_run.py`) | a policy that drains but declares no stop; a stop declared for another scope; a need on a channel never acquired; a need at a tick with no decision | yes, each |

The fork test forks one subject's five acquisitions at index 2 (indices 0 to 4): 0 and 1
are shared, 2 to 4 are drawn on the branch unless the coupling shares it, and the trunk's
0, 1, 3 and 4 equal their keyed draws. The latency test uses a clock of one day per tick
so a two-day delay is two ticks.

## Rebaselined cases

None. The default fork and coupling leave every key as it was.

## Deviations from the spec

- Two modules beyond the layout: `battery/result.py` (the result type and shared readers)
  and `battery/declarations.py` (the declaration check). The metamorphic checks share
  `battery/metamorphic.py`.
- A fork is a coordinate fork (branch from an acquisition index), not a changed action.
  "Indices 0, 1, 3 and 4 on the side that did not fork" is read as: the trunk's draws at
  those indices equal the draws recomputed from their coordinates, and the forked side's
  draws at 3 and 4 follow the coupling.
- "Relabel ids" covers channel ids. Subjects are indices with no id table in the core.
- "Shift the epoch origin" starts the run `delta` ticks later and moves the study's
  absolute horizon and phase boundaries, since the core has no epoch field.
- `latency_shift` expresses the delay in seconds and converts it on the study clock; the
  fixed plan is a policy that observes at absolute ticks, because the spec's fixed plan
  (`FixedPlanPolicy`) starts each acquisition at a release and so moves with the delay
  (`test_latency_shift_rejects_a_plan_that_reacts_to_releases`).
- No existing draw or leak test was rewired onto the battery: their helpers compare
  perturbed studies of the corpus, a different question from the battery's checks, and
  pointing them at it would have weakened them.

## Left for later tasks

- Policy draws after a fork run in no executor: a decision has no lineage, so only the
  keys are checked.
- The truth swap sees the policy through its snapshot and its trace; a component that
  closes over truth outside the study (a global) is invisible to it.
- The declaration check compares declared stops per phase against requested stops; it
  does not check `effects`, `reads` or `key_coords`.
- Adaptive latency fixtures ("every changed decision traces to a record whose release
  moved") are not built.
