---
orphan: true
---

# Study conformance and the core guide

## Behaviour changes

- `studies.survey_25` adds `NetworkPolicy` and `survey_25`. A frozen
  `eqx.nn.MLP` reads policy-snapshot means, sample counts, normalized tick and
  normalized decision count. Subject-major, channel-index-minor logits select
  one observation, with the first maximum breaking ties. After five requests by
  default, the policy drains the campaign. Other effect rows stay empty. The
  policy declares its effects and stop, preserves policy state and uses no
  policy randomness. Its network initialization key is independent of world roots.
- No core transaction, record, executor or key behavior changes.
- `docs/core.md` describes the implemented layers, study composition, completion
  checks, pre-flight, the battery and conformance, with a worked search example
  on both executors. The index includes it and the architecture page points to it.

## Tests

- `test_survey_1.py` narrows the existing fixed-plan study to one asset, lane and
  channel. A deterministic channel returns fixed unit signals for two subjects,
  retaining the declared uncertainty of 0.5 for scoring. The plan observes them
  three and two times, respectively. All five acquisitions fit the budgets and
  release by tick 650, before the horizon of 1000. Their signal-to-noise ratios
  are `2*sqrt(3)` and `2*sqrt(2)`; exactly one exceeds 3. The expected credit is
  written as 1 before execution, with this derivation in the test. Both runs
  must complete, and their histories pass the existing comparator's `go` band:
  exact integers and absolute float differences no greater than `1e-12`.
- `test_survey_25.py` exercises two network seeds on both executors. It captures
  actual compiled snapshots and effects using `DecisionViews`, then evaluates
  the same MLP eagerly on each captured snapshot. A separate Python decoder
  constructs every reference effect column. Host, compiled and reference action
  tables agree exactly; histories also agree. A controlled-network test changes
  only a released belief mean and requires the selected subject to change.
- The existing d2 statistic check now runs on roots 0 and 1. The shared world
  helper accepts a keyword seed, retaining root 0 as its default. Both roots use
  the existing 1,024 worlds and the unchanged finite-statistic, nonzero-flip,
  positive-standard-error and completion assertions. Other estimator tests and
  their thresholds are unchanged.

## Validation

- Focused survey and gradient tests: `18 passed in 175.53s (0:02:55)`.
  The longest individual case was 62.60 seconds. No slow marker was added.
- Ruff check and format, domain-word, core-import and internal-reference checks
  passed. Changed Python files are ASCII.
- A read-only code review found no blocking issues.
- HTML documentation built successfully. Both the updated pages and an unchanged
  source copy produced the same 496 warning messages; none were introduced.
  The active environment lacked `myst_nb`, AutoAPI and the book theme, so the
  build used `uv run --active --no-sync` with temporary `--with` dependencies
  and a cache inside the worktree. The shared environment and lockfile were not
  changed. The change record is marked orphan to avoid a new toctree warning.

## Rebaselined cases

None. Existing histories, channel tables and default world keys are unchanged.

## Implementation choices and limits

- `survey_1` remains in `studies.toy`, its established import location.
- The known-signal channel is local to the conformance test and documentation;
  ordinary `survey_1` runs retain Gaussian measurement noise.
- The existing d2 test is a 1,024-world structural check, not a 32,768-world
  statistical acceptance test. This change preserves its world count and
  assertions rather than claiming a new bound on estimator bias.
- The network has no training loop or external agent integration. Its limit counts
  requests, including refusals; arbitrary table overrides can therefore produce
  fewer admitted acquisitions. An executor's completion status must still be checked.
