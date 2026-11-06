---
orphan: true
---

# Slice 2 T9: legacy deletion

The mission loop built on `MissionSpec`, the implementation registry and the
wiring checks is gone, and with it the certificate stack that registered into
that registry. Two commits did the work, and each one imports cleanly and passes
the suite on its own. The core and the codec are unchanged apart from two
docstrings; no run behavior changed and the golden manifest is untouched.

This is a breaking change to the top-level package. Library versions are below
1.0, so the commits are marked `feat` without a major-version marker.

## Commits

1. `docs(core): fix two docstrings that break the warnings-as-errors docs build`
2. `feat(core): remove the spec, registry and wiring mission loop in favor of the core and codec`
3. `feat(core): remove the certificate stack that registered into the removed registry`
4. `docs(changes): record Slice 2 T9`

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
  and notebook in the repository, and a text scan of every markdown, TOML and YAML
  file, for an import of, or a dotted string naming, a deleted module. History under
  `docs/changes` is exempt. A second test is the negative control: it feeds the scan
  each import form (plain, from-module, from-package, public name, `importlib` string).
- `tests/test_docs_names.py` (B5): one case per doc file (README, `docs/`, diagrams)
  that fails on any deleted module path or public name, with a negative control.
- `tests/test_public_api.py`: rewritten as above.
- Docs build: `sphinx-build -W --keep-going -b html docs` passes. It failed on
  main with four warnings from two docstrings (a markdown table in
  `codec/document.py` and a backtick-plus-letter in `core/snapshot.py`); the first
  commit fixes them.

## Tests retired, ported and added

Counts are test functions and methods (parametrized cases are not counted):
962 before, 762 after.

Retired with the first commit (142):

| Module | Retired | Reason |
|---|---|---|
| `test_account`, `test_build`, `test_context`, `test_contracts`, `test_engines`, `test_io`, `test_loop`, `test_registry`, `test_seams`, `test_spec`, `test_wiring`, `test_belief`, `test_qualname_aliases` | 106 | each tests a removed module |
| `test_legacy_inventory` | 1 | tests the deleted scan tool |
| `test_certify`, `test_toy_certifier`, `test_profile_certificate`, `test_gaussian_profile`, `test_analytic` | 28 | the cases that went through `build`, the registry, the engines or the seam conformance suites; the certificate-only cases stayed until the second commit |
| `test_public_api` | 5 | registry, engine and `observe` entry points |
| `test_plan` | 1 | contract-namespace registration |
| `test_truth` | 1 | the deprecated alias in `testing.py` |

Retired with the second commit (65): `test_certify` (4), `test_toy_certifier` (15),
`test_profile_certificate` (10) and `test_gaussian_profile` (34) with their modules, and
two `test_analytic` cases that drove the removed toy certifier (the matched flat-prior
bound belongs to that certifier).

Ported (kept, imports rewritten, logic unchanged): `test_replicates` (19),
`test_claim_outcome` (11), `test_dist` (7), `test_truth` (13), `test_plan` (27) and the
reporting cases of `test_analytic` now import from `spaceodyssey.reporting` and
`tests.doubles`. `check_predictive`, `check_outcome` and `LinearGaussianTruthModel` moved
from the deleted `testing.py` to `tests/doubles.py`. One case was rewritten:
`test_differentiable_in_snr` drove the removed toy, so
`test_differentiable_in_probabilities` checks that `weighted_yields` is
differentiable in the outcome probabilities with a hand-built outcome and a known gradient.

Added: `test_slice2_gate.py` (2), `test_docs_names.py` (2 functions, 10 cases),
`test_public_api.py` (2 functions replacing 5), and the rewritten differentiability case.

The E0 inventory counted 410 functions in 26 modules as affected, of which 167 sit in
five modules that test retained code and only import a moved helper. This work retired
207 functions in total (142 and 65); the remaining affected functions were ported.

## Timing

Local runs on the four-thread compute launcher, `-n 2 --dist loadfile`, the machine shared
with other sessions' jobs in every row (the compute queue held two to four jobs).

| Run | Result |
|---|---|
| Quick tier on `8c4d307` (before) | 1,791 passed, 120.4 s wall, peak footprint 5.1 GB |
| Quick tier after the first commit | 1,599 passed, 68.9 s wall |
| Quick tier after the second commit | 1,497 passed, 68.5 s wall |
| Full suite after the first commit | 2,652 passed, 1,269.5 s wall, peak footprint 9.1 GB |
| Full suite after the second commit | 2,544 passed, 805.6 s wall, peak RSS 3.8 GB, peak footprint 7.9 GB |

The before row ran while two other jobs held the slots, so it is the high end; E0
measured the 26 legacy files at 52 s wall and 77 s summed on a cold cache, and the
quick-tier figure after is consistent with removing that. Every unit test is under 5 s
in all but the `test_every_hand_study_reproduces_its_history` outlier (6.6 s under load, in
report mode, unchanged by this work). The CI quick tier for K7 is read from
the job summary after the merge to main: the CI workflow does not run on a branch push.
