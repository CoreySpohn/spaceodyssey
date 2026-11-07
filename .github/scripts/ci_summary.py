"""Write a job summary from a pytest junit file and enforce suite time budgets.

Usage: python ci_summary.py junit.xml [--title TITLE] [--budget-seconds N]
       [--baseline SUMMARY.json --max-increase-seconds N] [--require-tier TIER]
       [--write-summary OUT.json [--commit SHA]] [--allow-skips]

Appends markdown to $GITHUB_STEP_SUMMARY (stdout when unset). Exits non-zero when:

- the suite's wall time exceeds ``--budget-seconds`` (the absolute ceiling);
- the wall time exceeds the baseline summary's wall time by more than
  ``--max-increase-seconds`` (the relative limit; both options go together);
- a tier named by ``--require-tier`` has no test in the file (so a selection that
  drops the medium tests cannot pass the budget unnoticed);
- any test was skipped (a skip hides a missing dependency), unless allowed.

The wall time is the junit testsuite time, which pytest measures over the whole
session (collection included) and which stays correct under pytest-xdist; the summed
per-test time is reported alongside it. The tier of each test (unit, medium, slow)
comes from the ``tier`` property that tests/conftest.py records for every test.
``--write-summary`` saves this run's figures as JSON in the baseline format, so a
later run can be compared with it.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

TIERS = ("unit", "medium", "slow")


def read_junit(path):
    """Wall time, (seconds, name, tier) per test, and the skipped test names."""
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    wall = sum(float(suite.get("time", 0.0)) for suite in suites)
    cases, skipped = [], []
    for case in root.iter("testcase"):
        name = f"{case.get('classname', '')}::{case.get('name', '')}"
        seconds = float(case.get("time", 0.0))
        tier = ""
        for prop in case.iter("property"):
            if prop.get("name") == "tier":
                tier = prop.get("value", "")
        cases.append((seconds, name, tier))
        # Pytest records expected failures as skipped; pytest enforces strict xfails.
        skip = case.find("skipped")
        if skip is not None and skip.get("type") != "pytest.xfail":
            skipped.append(name)
    return wall, cases, skipped


def tier_totals(cases):
    """Test count and summed time per tier; untagged tests are reported as such."""
    totals = {}
    for seconds, _, tier in cases:
        count, total = totals.get(tier or "untagged", (0, 0.0))
        totals[tier or "untagged"] = (count + 1, total + seconds)
    return totals


def load_baseline(path):
    """The baseline summary: its wall time and the notes printed beside it."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    wall = float(data["wall_s"])
    if not wall > 0:
        raise ValueError(f"{path}: wall_s must be positive, got {wall}")
    return data


def main():
    """Summarize collected tests and check the requested budgets and skip policy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("junit")
    parser.add_argument("--title", default="Tests")
    parser.add_argument("--budget-seconds", type=float, default=0.0)
    parser.add_argument("--baseline", default="")
    parser.add_argument("--max-increase-seconds", type=float, default=None)
    parser.add_argument("--require-tier", action="append", default=[], choices=TIERS)
    parser.add_argument("--write-summary", default="")
    parser.add_argument("--commit", default=os.environ.get("GITHUB_SHA", ""))
    parser.add_argument("--allow-skips", action="store_true")
    args = parser.parse_args()
    if bool(args.baseline) != (args.max_increase_seconds is not None):
        parser.error("--baseline and --max-increase-seconds go together")

    if not os.path.exists(args.junit):
        print(f"{args.junit} not found: the test step did not run")
        return 1
    wall, cases, skipped = read_junit(args.junit)
    total = sum(s for s, _, _ in cases)
    totals = tier_totals(cases)
    baseline = load_baseline(args.baseline) if args.baseline else None
    ranked = sorted(cases, reverse=True)

    lines = [
        f"### {args.title}",
        "",
        f"- tests: {len(cases)}, skipped: {len(skipped)}",
        f"- wall time: {wall:.1f} s"
        + (f" (budget {args.budget_seconds:.0f} s)" if args.budget_seconds else ""),
        f"- summed test time: {total:.1f} s",
    ]
    lines += [
        f"- {tier} tests: {count}, summed {seconds:.1f} s"
        for tier, (count, seconds) in sorted(totals.items())
    ]
    if baseline is not None:
        increase = wall - float(baseline["wall_s"])
        lines += [
            f"- baseline: {float(baseline['wall_s']):.1f} s "
            f"({baseline.get('status', 'unstated')}; "
            f"cache {baseline.get('cache', 'unstated')}), "
            f"increase {increase:+.1f} s (limit {args.max_increase_seconds:.0f} s)",
            f"- baseline source: {baseline.get('source', 'unstated')}",
        ]
    lines += ["", "| slowest tests | tier | seconds |", "|---|---|---|"]
    lines += [
        f"| `{name}` | {tier} | {seconds:.2f} |" for seconds, name, tier in ranked[:10]
    ]
    if skipped:
        policy = "allowed" if args.allow_skips else "treated as a failure"
        lines += ["", f"Skipped ({policy}):"]
        lines += [f"- `{name}`" for name in skipped[:20]]

    text = "\n".join(lines) + "\n"
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text)

    if args.write_summary:
        summary = {
            "wall_s": round(wall, 2),
            "tests": len(cases),
            "tiers": {
                t: {"tests": c, "summed_s": round(s, 2)} for t, (c, s) in totals.items()
            },
            "commit": args.commit,
            "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        }
        with open(args.write_summary, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=1, sort_keys=True)
            handle.write("\n")
        print(json.dumps(summary, sort_keys=True))

    failed = False
    if args.budget_seconds and wall > args.budget_seconds:
        print(f"wall time {wall:.1f} s exceeds budget {args.budget_seconds:.0f} s")
        failed = True
    if (
        baseline is not None
        and wall - float(baseline["wall_s"]) > args.max_increase_seconds
    ):
        print(
            f"wall time {wall:.1f} s is {wall - float(baseline['wall_s']):.1f} s above "
            f"the baseline {float(baseline['wall_s']):.1f} s; the limit is "
            f"{args.max_increase_seconds:.0f} s"
        )
        failed = True
    for tier in args.require_tier:
        if tier not in totals:
            print(f"no {tier} test in {args.junit}: the selection dropped that tier")
            failed = True
    if skipped and not args.allow_skips:
        print(f"{len(skipped)} tests were skipped")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
