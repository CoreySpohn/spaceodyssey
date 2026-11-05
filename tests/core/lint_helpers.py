"""Shared lint invocations and isolated fixture trees."""

import contextlib
import functools
import importlib
import io
import subprocess
import sys
from pathlib import Path

import pytest

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


def run_lint_subprocess(tool, *args):
    return subprocess.run(
        [sys.executable, str(REPO / "tools" / tool), *map(str, args)],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


@functools.cache
def _lint_main(tool):
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(REPO / "tools"))
        return importlib.import_module(Path(tool).stem).main


@functools.cache
def _repository_lint(tool):
    return run_lint_subprocess(tool)


def run_lint(tool, *args):
    """Reuse loaded lint programs; retain subprocess checks of the repository."""
    if not args:
        return _repository_lint(tool)
    main = _lint_main(tool)
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(list(map(str, args)))
    return subprocess.CompletedProcess(
        [tool, *map(str, args)], code, stdout.getvalue(), stderr.getvalue()
    )


def fixture_core(tmp_path, text, name="sample.py"):
    root = tmp_path / "core"
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return root, path
