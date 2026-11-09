"""The CI summary script fails a run over its budget, baseline or tier limits."""

import json
import os
import subprocess
import sys

import pytest

from tests.pin_helpers import REPO

CI_SUMMARY = REPO / ".github/scripts/ci_summary.py"

JUNIT = """<?xml version="1.0"?>
<testsuites><testsuite name="pytest" time="{wall}">
<testcase classname="t" name="a" time="1.0">
<properties><property name="tier" value="unit"/></properties></testcase>
<testcase classname="t" name="b" time="2.0">
<properties><property name="tier" value="{tier}"/></properties></testcase>
</testsuite></testsuites>
"""


@pytest.mark.parametrize(
    ("wall", "tier", "ok"),
    [
        (159.0, "medium", True),  # 59 s above the baseline, under the ceiling
        (161.0, "medium", False),  # 61 s above the baseline
        (150.0, "unit", False),  # no medium test: the selection dropped the tier
        (541.0, "medium", False),  # over the 540 s ceiling (baseline raised below)
    ],
)
def test_the_summary_script_fails_over_its_limits(tmp_path, wall, tier, ok):
    """Fails if ci_summary.py passes a run 61 s above its baseline, over 540 s, or
    with no medium test, or fails a run within every limit.
    """
    junit = tmp_path / "junit.xml"
    junit.write_text(JUNIT.format(wall=wall, tier=tier))
    baseline = tmp_path / "baseline.json"
    base = 500.0 if wall > 540 else 100.0
    baseline.write_text(json.dumps({"wall_s": base, "source": "test", "status": "x"}))
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_STEP_SUMMARY"}
    result = subprocess.run(
        [
            sys.executable,
            str(CI_SUMMARY),
            str(junit),
            "--budget-seconds",
            "540",
            "--require-tier",
            "medium",
            "--baseline",
            str(baseline),
            "--max-increase-seconds",
            "60",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert (result.returncode == 0) == ok, result.stdout + result.stderr


def test_an_advisory_baseline_prints_the_increase_without_failing(tmp_path):
    """Fails if --advisory-baseline still fails a run 61 s above its baseline, or
    stops printing the increase, or is accepted without a baseline.
    """
    junit = tmp_path / "junit.xml"
    junit.write_text(JUNIT.format(wall=161.0, tier="medium"))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"wall_s": 100.0, "source": "t", "status": "x"}))
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_STEP_SUMMARY"}
    command = [sys.executable, str(CI_SUMMARY), str(junit), "--advisory-baseline"]
    alone = subprocess.run(command, capture_output=True, text=True, env=env)
    assert alone.returncode != 0
    command += ["--baseline", str(baseline), "--max-increase-seconds", "60"]
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "advisory: not a failure" in result.stdout
