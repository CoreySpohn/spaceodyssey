"""Shared lint invocations and isolated fixture trees."""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parents[2]
CORE = REPO / "src/spaceodyssey/core"
FIXTURES = Path(__file__).with_name("lint_fixtures")
DENIED = (
    "planet star exoplanet coronagraph telescope exposure slew keepout band "
    "wavelength flux magnitude orbit ephemeris exozodi tdb visit mission target "
    "facility nondetection completeness photometry detection"
).split()
PHYSICS = (
    "orbix jaxedith photomancy skyscapes coronagraphoto coronachrome coronalyze "
    "yippy optixstuff physicaloptix exojax planit_py"
).split()


def run_lint(tool, *args):
    return subprocess.run(
        [sys.executable, str(REPO / "tools" / tool), *map(str, args)],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def fixture_core(tmp_path, text, name="sample.py"):
    root = tmp_path / "core"
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return root, path
