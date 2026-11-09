"""Write the pilot job summary: step cost and cold medium-test times.

Usage: python pilot_summary.py step_cost.json [step_cost_pair.json] junit-pilot-*.xml

Reads the JSON that `tools/step_cost.py --mutant` prints, optionally the JSON of
`tools/step_cost_pair.py` (the same-job comparison of two trees), and one pytest junit
file per test file run cold, and prints markdown (the workflow appends it to
$GITHUB_STEP_SUMMARY). Exits non-zero when the step-cost JSON or a junit file is
missing, so a pilot that did not measure cannot pass.
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def step_cost_table(path):
    """Markdown rows for the plain and mutant step costs."""
    data = json.loads(Path(path).read_text())
    lines = [
        f"## Step cost ({data['handlers']} handlers, {data['study']}, "
        f"{data['n_worlds']} worlds, {data['n_steps']} steps, "
        f"{data['repeats']} repeats, JAX {data['jax']})",
        "",
        "| Executor | Cold first call (s) | Per step, median (us) "
        "| Per step, min (us) |",
        "|---|---|---|---|",
    ]
    for name in ("plain", "mutant"):
        if name in data:
            row = data[name]
            lines.append(
                f"| {name} | {row['cold_s']} | {row['per_step_us']} | "
                f"{row['per_step_min_us']} |"
            )
    if "mutant_ratio" in data:
        lines += ["", f"Mutant ratio (median per step): {data['mutant_ratio']}"]
    return lines


def pair_table(path):
    """Markdown rows for the same-job comparison of the base and final trees."""
    data = json.loads(Path(path).read_text())
    base, final = data["base"], data["final"]
    lines = [
        f"## Same-job step cost ({data['rounds']} interleaved rounds, "
        f"{data['repeats']} repeats each)",
        "",
        "| Tree | Handlers | Plain per step, min (us) | Mutant per step, min (us) "
        "| Mutant ratio | Cold first call, plain (s) | Cold first call, mutant (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, row in (("base", base), ("final", final)):
        lines.append(
            f"| {name} | {row['handlers']} | {row['per_step_min_us']} | "
            f"{row['mutant_per_step_min_us']} | {row['mutant_ratio']} | "
            f"{min(row['cold_s'])} | {min(row['mutant_cold_s'])} |"
        )
    lines += [
        "",
        f"K3 cost ratio (final over base, lowest minimum of each): {data['ratio']} "
        f"against the limit {data['limit']} "
        f"({'pass' if data['within_limit'] else 'FAIL'}); per round: "
        f"{data['per_round_ratio']}",
    ]
    return lines


def _failed(case):
    """Whether a junit test case holds a failure or an error element."""
    return any(case.find(tag) is not None for tag in ("failure", "error"))


def junit_table(paths):
    """Markdown rows: one per test file run cold, with its slowest test."""
    lines = [
        "## Cold medium tests (one process per file, compile caches off)",
        "",
        "| File | Tests | Failed | Wall (s) | Slowest test | Its time (s) |",
        "|---|---|---|---|---|---|",
    ]
    for path in paths:
        root = ET.parse(path).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        cases = suite.findall("testcase")
        bad = sum(1 for c in cases if _failed(c))
        slow = max(cases, key=lambda c: float(c.get("time", 0)), default=None)
        name = Path(path).stem.removeprefix("junit-pilot-")
        top = "" if slow is None else slow.get("name")
        top_s = "" if slow is None else f"{float(slow.get('time', 0)):.2f}"
        lines.append(
            f"| {name} | {len(cases)} | {bad} | {float(suite.get('time', 0)):.1f} | "
            f"{top} | {top_s} |"
        )
    return lines


def main(argv):
    """Print the summary; non-zero when an input is missing."""
    if len(argv) < 2:
        print("usage: pilot_summary.py step_cost.json [pair.json] junit-pilot-*.xml")
        return 2
    missing = [p for p in argv if not Path(p).exists()]
    if missing:
        print(f"missing pilot output: {', '.join(missing)}")
        return 1
    pair = [p for p in argv if Path(p).name == "step_cost_pair.json"]
    junit = [p for p in argv[1:] if p not in pair]
    lines = ["# CI-runner pilot", ""]
    lines += [*step_cost_table(argv[0]), ""]
    if pair:
        lines += [*pair_table(pair[0]), ""]
    lines += junit_table(junit)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
