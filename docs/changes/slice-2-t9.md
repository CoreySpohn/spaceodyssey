---
orphan: true
---

# Slice 2 T9: legacy deletion

The mission loop built on `MissionSpec`, the implementation registry and the
wiring checks is gone, and with it the certificate stack that registered into
that registry. Two commits did the work, and each one imports cleanly and passes
the suite on its own. The core and the codec are unchanged apart from two
docstrings; no run behavior changed and the golden manifest is untouched.

This is a breaking change to the top-level package, so both removal commits carry
`!` (`feat(core)!:`). The release-please configuration sets `bump-minor-pre-major`,
so on 0.1.0 a breaking commit produces 0.2.0 and a breaking-changes entry in the
changelog; it does not graduate the library to 1.0.

## Commits

1. `docs(core): fix two docstrings that break the warnings-as-errors docs build` (`84cec69`)
2. `feat(core)!: remove the spec, registry and wiring mission loop in favor of the core and codec` (`83a1ad5`)
3. `feat(core)!: remove the certificate stack that registered into the removed registry` (`0f3c2b2`)
4. this record

## Removed modules

First commit, the mission-loop half (16 files, 2,329 lines; the `__init__` rewrite is separate):

| Module | Lines | What it was |
|---|---|---|
| `spec.py` | 332 | `MissionSpec`, `FidelityConfig`, `SeamRef`, `FileRef`, JSON form |
| `registry.py` | 250 | `Registry`, `REGISTRY`, `SeamInfo`, `register` |
| `wiring.py` | 257 | configuration validation, `WiringReport` |
| `seams.py` | 42 | seam names, data-plane edges, operator vocabulary |
| `contracts.py` | 91 | contract-type resolution and edge compatibility |
| `aliases.py` | 20 | qualified-name aliases |
| `testing.py` | 329 | seam conformance suites and the reference truth model |
| `account.py` | 130 | `Ledger`, `ConstantOverheads` |
| `belief.py` | 105 | `MissionBelief`, `TargetBelief` |
| `build.py` | 122 | `build`, `Mission`, `ModelStack`, `TruthStack` |
| `context.py` | 78 | observing-context providers |
| `loop.py` | 182 | `run_mission`, `MissionState`, `step`, `observe` |
| `io.py` | 168 | run manifests and run directories |
| `engines/__init__.py`, `engines/analytic.py`, `engines/mc.py` | 98, 71, 54 | engine registry, integrate and sample engines |

Second commit, the certificate half (2 files): `certify.py` (439 lines) and
`gaussian_profile.py` (584 lines). `certify.py` imported the registry, so the first commit
removed the `@register` decorators and registry imports from both modules (38 lines) and
left the classes working; the second commit removed the modules (985 lines).

The package `__init__.py` went from 155 to 63 lines. The two commits delete 18 files
and 3,352 lines (the figure above, counted before the decorator edits) and rewrite the
`__init__`, which matches E0's inventory (3,507 lines with the old `__init__`). Also removed:
`RegistryError` and `WiringError` from `errors.py`, `tools/legacy_inventory.py` and its
test (the scan did its job), and three diagrams of the removed design
(`mc-loop`, `two-axes`, `wiring-contracts`).

## What replaces them

| Removed | Replacement |
|---|---|
| `MissionSpec`, `FidelityConfig`, `SeamRef`, `FileRef`, `to_json`, `spec_from_json` | `spaceodyssey.codec`: `dump_study`, `load_study`, `KindResolver`, and the codec-owned array file reference |
| `Registry`, `REGISTRY`, `register`, `SeamInfo` | kinds registered under entry points in `spaceodyssey.<family>` groups, read by `spaceodyssey.plugins` (`installed_resolver`, `kinds_from_entry_points`, `RegistrationError`, `FAMILIES`) |
| qualified-name aliases | none; a document names a kind id and an integer version |
| `run_mission`, `build`, the engines | `spaceodyssey.core` executors |

The top-level package keeps the plan, truth, profile, claim, outcome, yield-report,
distribution, data and universe types. It no longer exports any name of the removed
API; `tests/test_public_api.py` checks fifteen legacy names and the six certificate
names, and that the codec and plugin entry points import.

## Gate checks added

- `tests/test_slice2_gate.py::test_no_legacy_imports` (B5): an AST scan of every `.py`
  and notebook in the repository, and a text scan of markdown, TOML and YAML. It
  resolves relative imports against the importing file, catches removed root names
  (`from spaceodyssey import Ledger, Mission, run`), and parses `pyproject.toml`
  entry-point targets, flagging a target in a deleted module or in a module that does
  not exist. Only the prose of `docs/changes` is exempt; Python and notebook code is
  always scanned. The scan asserts that its sentinel files and a minimum number of
  Python files were reached, so an empty scan fails.
- Negative controls run the whole scanner on a temporary tree with one injected
  violation per case: plain, from-module, from-package, root name, `importlib` string,
  three relative forms, markdown code, a broken file under `docs/changes`, a notebook,
  and entry points to a deleted and a missing module. Valid imports and history prose
  must pass.
- `tests/test_docs_names.py` (B5): README, `docs/` and diagrams, plus the module, class
  and function docstrings that AutoAPI publishes, must name no deleted module path or
  public name. It asserts coverage and has injected-violation controls for each kind.
- `tests/test_public_api.py`: checks the legacy and certificate names are gone and that
  the codec and plugin entry points import.
- Docs build: `sphinx-build -W --keep-going -b html docs` passes. It failed on main with
  four warnings from two docstrings, fixed in the first commit. The `dist.py` and
  `observe.py` module docstrings no longer describe the removed engines or registry.

## Tests

Test functions and methods (`def test_` count; parametrized cases are not counted):
967 on `8c4d307`, 838 after the first removal commit, 774 after the second.

Tests of a removed module were deleted with it (the registry, spec, wiring, loop, build,
engines, io, context, contracts, seams, belief, account and alias suites, the legacy
inventory scan, and the certify, toy certifier, profile certificate and Gaussian profile
suites). The reporting cases of `test_analytic`, `test_replicates`, `test_claim_outcome`,
`test_dist`, `test_truth` and `test_plan` were kept with imports rewritten; the helpers
`check_predictive`, `check_outcome` and `LinearGaussianTruthModel` moved to
`tests/doubles.py`. `test_differentiable_in_snr` drove the removed toy certifier, so
`test_differentiable_in_probabilities` checks that `weighted_yields` is differentiable in
the outcome probabilities with a hand-built outcome.

The yield summary lost its coverage when `test_engines` was deleted, so `tests/test_metrics.py`
holds it again: `TestYieldSummary` is copied unchanged from the old suite, and
`TestSummarize` tests `summarize` on small state doubles (run accounting, false
certifications, wrong claims apart from absent sources, claim scoring off by default,
a custom `is_real`).

## Timing

Local runs on the four-thread compute launcher, `-n 2 --dist loadfile`, with other
sessions' jobs holding the machine in every row.

| Run | Result |
|---|---|
| Full suite after the first commit | 2,674 passed, 1,305 s wall, peak RSS 4.4 GB, peak footprint 9.6 GB |
| Full suite after the second commit | 2,566 passed, 744 s wall, peak RSS 4.1 GB, peak footprint 8.6 GB |
| Quick tier (`-m "not slow"`) after the second commit | 1,519 passed, 44.7 s wall |
| Docs build with `-W --keep-going` | passes |

The first-commit run was the most contended. In report mode the test-budget plugin flags
`test_every_hand_study_reproduces_its_history` (5.2 s here, unchanged by this work) after
the second commit; the first-commit run also flagged `test_gain_enters_measurement` and
`test_statuses_agree_between_executors_and_replay` under load. The CI quick tier is read
from the job summary after the merge to main: the CI workflow does not run on a branch push.

## Post-merge follow-ups

- The import gate tracks package aliases and checks removed root attributes and
  entry-point object paths. Negative controls cover direct and aliased attributes,
  removed root names and modules after the colon, and script targets.
- Only `.md`, `.rst` and `.txt` change records are exempt. Negative controls cover
  a `docs/changes/pyproject.toml` entry point to `spaceodyssey.registry:Registry`,
  YAML and other configuration files; Python and notebooks remain scanned.
- The docs gate shares the removed root-name inventory with the import gate,
  including `Mission` and `Ledger`, and checks qualified forms. A method-docstring
  control covers `Construct a spaceodyssey.Mission with a Ledger.`
- The yield-summary JSON round trip fills every list field, including nonempty
  wrong-certified targets, so replacing a serialized list with `[]` fails.
