---
orphan: true
---

# Slice 2b R6: close-out

R6 closes Slice 2b. It adds the gate that runs kill rules K1 to K5 as one command,
turns on the test-budget enforcement of decision Q9 (option A), measures the cold
quick-tier cost the slice added and moves tests to the slow tier until it fits (K6,
Corey's 2026-11-07 decision), and records the status of B1 to B8 and K1 to K7.

## Commits

TBD

## What changed

- `tests/test_slice2b_gate.py` (new, slow tier): K1 to K5, the K6 and K7 wiring, and
  the named interface checks. One command: `pytest tests/test_slice2b_gate.py`.
- `tests/conftest.py`: the budget plugin is loaded unconditionally; every test
  records its tier (`unit`, `medium`, `slow`) as a junit property; with
  `SPACEODYSSEY_BUDGET_SCALE` set, every tier budget is multiplied by it; the gate
  file joins `SLOW_FILES`.
- `pyproject.toml`: `[tool.testbudget] mode = "enforce"`; the `test` extra requires
  `hwoutils>=1.11.0`, the first release with the budget plugin.
- `.github/scripts/ci_summary.py`: `--baseline <summary.json>` with
  `--max-increase-seconds`, `--require-tier`, `--write-summary` and `--commit`; the
  summary reports tests and summed time per tier.
- `.github/budget/quick_baseline.json` (new): the pinned, provisional baseline.
- `.github/workflows/ci.yml`: the quick tier's summary step enforces 540 s, the
  baseline with a 60 s limit and the presence of medium tests;
  `SPACEODYSSEY_BUDGET_SCALE: "3"`; the slow job checks out the full history (the
  gate's K2 reads it).
- `.github/workflows/pilot.yml`: a `quick_ref` input runs the whole quick tier cold
  at that ref and prints its summary in the baseline format.
- `.github/workflows/upstream.yml`: the canary runs the budget plugin in report mode.
- Six Slice 2b tests move from medium to slow (K6, below).

TBD
