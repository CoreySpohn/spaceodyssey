#!/usr/bin/env python3
"""Per-step cost of two source trees, measured in one job and interleaved.

Each round runs `tools/step_cost.py --mutant` of the base tree and then of the final
tree, one fresh process each, with the same repeats and the same process setup. A tree
is measured by its own copy of the probe against its own package (`PYTHONPATH` is that
tree's `src`), so each figure is the cost of the code in that tree. The probe differs
between the trees only in the baseline option, which this driver does not use.

The cost ratio is the final tree's lowest per-step minimum over the base tree's, taken
over all rounds, so that runner noise (a slow moment of the shared machine) cannot
inflate either figure alone. The per-round ratios are printed as well. The extra-pass
mutant ratio of each tree (its mutant over its plain step, measured in the same
process) shows that the probe can resolve a regression.

Exit status 1 when the cost ratio exceeds `--limit` (1.10). Prints one JSON object.

Usage: python tools/step_cost_pair.py --base-tree base-tree --final-tree . --rounds 5
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

LIMIT = 1.10  # the final table may cost at most 10 percent more per step


def run_probe(tree, repeats):
    """One fresh process: the tree's own probe against the tree's own package."""
    tree = Path(tree).resolve()
    env = dict(os.environ, PYTHONPATH=str(tree / "src"))
    done = subprocess.run(
        [
            sys.executable,
            str(tree / "tools/step_cost.py"),
            "--mutant",
            "--repeats",
            str(repeats),
        ],
        cwd=tree,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout)


def summarize(runs):
    """Lowest plain minimum, cold times and mutant ratios of one tree's rounds."""
    return {
        "handlers": runs[0]["handlers"],
        "per_step_min_us": min(r["plain"]["per_step_min_us"] for r in runs),
        "mutant_per_step_min_us": min(r["mutant"]["per_step_min_us"] for r in runs),
        "mutant_ratio": round(statistics.median(r["mutant_ratio"] for r in runs), 3),
        "cold_s": [r["plain"]["cold_s"] for r in runs],
        "mutant_cold_s": [r["mutant"]["cold_s"] for r in runs],
        "per_round_min_us": [r["plain"]["per_step_min_us"] for r in runs],
    }


def compare(base_runs, final_runs, limit=LIMIT):
    """The paired result from the runs of each tree, one run per round."""
    base, final = summarize(base_runs), summarize(final_runs)
    ratio = final["per_step_min_us"] / base["per_step_min_us"]
    per_round = [
        round(f / b, 3)
        for b, f in zip(
            base["per_round_min_us"], final["per_round_min_us"], strict=True
        )
    ]
    return {
        "rounds": len(base_runs),
        "base": base,
        "final": final,
        "per_round_ratio": per_round,
        "ratio": round(ratio, 3),
        "limit": limit,
        "within_limit": ratio <= limit,
    }


def main(argv=None):
    """Measure both trees, interleaved, and print the comparison as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-tree", required=True)
    parser.add_argument("--final-tree", default=".")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--limit", type=float, default=LIMIT)
    args = parser.parse_args(argv)
    base_runs, final_runs = [], []
    for _ in range(args.rounds):
        base_runs.append(run_probe(args.base_tree, args.repeats))
        final_runs.append(run_probe(args.final_tree, args.repeats))
    out = compare(base_runs, final_runs, args.limit)
    out["repeats"] = args.repeats
    print(json.dumps(out, indent=1))
    if not out["within_limit"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
