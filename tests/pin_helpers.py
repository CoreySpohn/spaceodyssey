"""Pin constants and git helpers shared by the current slice gate."""

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDEN_PIN = REPO / "tests/core/golden_pin.json"
HASH_PIN = REPO / "tests/codec/hash_pin.json"
CI = REPO / ".github/workflows/ci.yml"

# The pins' source commit and the commits that captured each pin file.
PIN_SOURCE = "96a644728386365177e79f8bca315a746ef2901c"
PIN_CAPTURES = {
    "5b9fbd5f35d2d6be6286f178445fea7c2381d7f3",  # golden_pin.json
    "fc0e2e4ae257c3fef7282747927dac7a03d5a928",  # hash_pin.json
}


def git(*args):
    """Output of a git command in the repository; fails the test without history."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.fail(
            f"git {' '.join(args)} failed ({error}); the gate needs the repository "
            "with its full history (checkout with fetch-depth 0)"
        )
    return result.stdout


def resolve(short):
    """The full sha of a commit given by any unambiguous prefix."""
    return git("rev-parse", "--verify", f"{short}^{{commit}}").strip()


def quick_summary_step():
    """The CI workflow text and its quick-tier summary step."""
    text = CI.read_text()
    step = text[text.index('--title "Quick tier"') :].split("\n\n")[0]
    return text, step
