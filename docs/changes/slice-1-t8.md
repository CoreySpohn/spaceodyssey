# Slice 1, T8: pre-flight

## Behaviour changes

- `spaceodyssey.core.preflight` adds `preflight(study, executor=None) -> list[Finding]`,
  `Finding` (code, severity, path, message, fix hint), `FindingCode`, `Severity` and
  `StudyGraph`. Pre-flight is structural: it reads declarations and the tables of a
  `Study` and never runs it (`test_preflight_runs_nothing` patches both executors and the
  step function to fail). A clean result says the declarations are consistent, not that
  the science is right.
- `StudyGraph.from_study` compiles a study to typed nodes (`source`, `measure`, `update`,
  `decide`, `score`), each on a side (`truth`, `observer`, `scorer`), with edges from
  reader to source: the world and process sources, one measure per channel, the
  observer, one decide per agenda phase and the objective. A measure and the process are
  the crossings; a channel feeds the observer or the policy phases per its audience.
- `spaceodyssey.core.declarations` adds `Declaration` and `Need`. A component states
  itself by defining `declare()`; one without it declares nothing unusual, so the
  abstract kinds are unchanged. A declaration holds `reads` (`truth` or `scorer`, beyond
  released records), `key_coords`, `stops`, `effects`, `needs`, `requires` (executor
  capabilities), `executors` and `disruptive`. Unknown names fail at construction.
  `EXECUTOR_CAPABILITIES` lists what each executor provides (`batched_worlds` and
  `gradients` on the compiled executor, `python_policy` on the host executor) and
  `SUPPORTED_EFFECTS` the effects both run.
- `FixedPlanPolicy.declare` states its stops (the final campaign drain and any special
  stop), the only edit outside the new modules and the exports.

## Codes

| Code | Severity | Raised when |
|---|---|---|
| F001 | error | an observer-side node (observer, policy phase) reads truth outside a crossing |
| F002 | error | an observer-side node reads the objective |
| C001 | error | a policy phase `Need` is released later than its decision tick, to no policy audience, or on a channel the table lacks |
| K001 | error | a measure or source draws without its stable-id coordinates (`subject`, `channel`, `acquisition_index` for a measure; `world` for a source) |
| K002 | error | two measures share key coordinates, for example duplicate channel ids |
| X001 | error | a component asks for `Allocate`, requires a capability the selected executor lacks, or runs on other executors only |
| X002 | error | `charge` differs from `cost`, `freeze_ticks` is nonzero, `disruptive_resource` is set, or a policy declares a disruptive effect |
| T001 | warning | a policy asks DRAIN below campaign scope, where it equals CLOSE |
| T002 | error | `coincident_order` is missing, has the wrong shape or holds a value outside `CoincidentOrder` |
| R001 | error | a channel declares a record kind index the lowered registry lacks |

T001, T002 and R001 are new names for the items earlier tasks deferred. With no
executor named, X001 fails a requirement only when no executor provides it.

## Tests

- `test_preflight.py`: one seeded study per code and per variant (19 in all), each raising
  exactly its own code (`test_seeded_mistake_is_caught`); 26 correct studies (`survey_1`,
  the catalog study, `d2_study`, a study with stated components and every corpus case)
  raising only the warnings the fixture expects, which are DRAIN at lane, asset and
  program scope in the stop grid (`test_correct_study_is_clean`); every code has a
  seeded study; executor selection changes X001; the graph crosses from truth only at
  measures; `test_preflight_reports_counts` prints errors, warnings, codes exercised,
  false positives and missed mistakes.
- Seeded studies and the declaring component stand-ins live in
  `tests/core/preflight_cases.py`.

Report line from the suite: `19 seeded and 26 correct studies; 20 errors (0 on correct
studies), 4 warnings; codes exercised: C001, F001, F002, K001, K002, R001, T001, T002,
X001, X002; false positives: 0; missed: 0`.

## Rebaselined cases

None. No handler, executor or table changed.

## Deviations from the spec

- Pre-flight lives in a package whose checks are one module (`checks.py`) rather than a
  module per check, and the declarations sit beside it in `core/declarations.py`.
- The checks rest on declarations. A component that reads truth without declaring it
  passes F001; the check is only as honest as the declaration. A generated test battery
  is the guard against a false declaration.
- C001 is a lower bound: a product is released no earlier than start plus occupancy plus
  latency; a lane transition only delays it, so the check never raises on a decision
  that could be fed.
- Of the "claimed on a path the study does not request" half of X001, only the missing
  side exists; there is nothing in Slice 1 to claim.
- Ledger reserved fields live in the carry, which is initialized to the defaults, so X002
  checks the study tables and the declared disruptive effect only. `Query` was flagged
  with `Allocate` until the host protocol made it a supported effect.
- The spec's X002 sentence names a later slice; the message says no rule reads the field.

## Left for later tasks

- Units (U), identifiability (I), scoring (S) and lineage (D001) codes are not built.
- A document codec would give each finding a JSON path; the path today is a graph path.
- Policies state their stops and needs by hand. The generated test battery should
  confirm that the stated stops match the effects a run emits.
