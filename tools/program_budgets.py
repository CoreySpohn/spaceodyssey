"""Print per-file program budgets for ``[tool.testbudget]`` from a session JSON.

Usage: python tools/program_budgets.py budget.json [budget.json ...]
       [--scale 1.15] [--pad 10]

Each JSON is written by ``pytest --budget-json PATH`` on a run with
``clear_caches = true``, where a file's count is the programs it compiles or
loads from the persistent cache with nothing carried over from earlier files.
With several runs the largest count of each file is used. The budget is
``ceil(count * scale) + pad``: the headroom absorbs a session-scoped fixture
built by whichever file uses it first, and differences between machines.
Paste the output over the ``[tool.testbudget.module_compile_budget_by_file]``
table in pyproject.toml.
"""

import argparse
import json
import math


def main():
    """Read the session files and print the budget table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--scale", type=float, default=1.15)
    parser.add_argument("--pad", type=int, default=10)
    args = parser.parse_args()
    counts = {}
    for path in args.runs:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        loads = data.get("module_cache_loads", {})
        for name, compiled in data["module_compiles"].items():
            programs = compiled + loads.get(name, 0)
            counts[name] = max(counts.get(name, 0), programs)
    print("[tool.testbudget.module_compile_budget_by_file]")
    for name in sorted(counts):
        budget = math.ceil(counts[name] * args.scale) + args.pad
        print(f'"{name}" = {budget}')


if __name__ == "__main__":
    main()
