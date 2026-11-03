"""Negative controls and repository checks for the core boundaries."""

import re
import tempfile
import tomllib
from pathlib import Path

import pytest

from tests.core.lint_helpers import (
    CORE,
    DENIED,
    FIXTURES,
    PHYSICS,
    REPO,
    fixture_core,
    run_lint,
)


@pytest.mark.parametrize(
    ("fixture", "token"),
    [
        ("identifier", "planet"),
        ("docstring", "telescope"),
        ("comment", "orbit"),
        ("error_message", "exposure"),
        ("preflight_sentence", "target"),
        ("keyword", "band"),
        ("plural", "facilities"),
    ],
)
def test_lint_catches_each_location(tmp_path, fixture, token):
    root, path = fixture_core(tmp_path, (FIXTURES / f"{fixture}.txt").read_text())
    result = run_lint("check_domain_words.py", "--core-dir", root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert re.search(rf"{re.escape(str(path))}:\d+:{token}\b", result.stdout)


@pytest.mark.parametrize("word", DENIED)
def test_lint_catches_each_denied_token(tmp_path, word):
    root, _ = fixture_core(tmp_path, f"{word} = 0\n")
    result = run_lint("check_domain_words.py", "--core-dir", root)
    assert result.returncode == 1
    assert f":1:{word}" in result.stdout


@pytest.mark.parametrize(
    ("source", "token", "line"),
    [
        ("def process(planet):\n    pass\n", "planet", 1),
        ("row.star = 1\n", "star", 1),
        ('class Row:\n    """A telescope."""\n', "telescope", 2),
        ('def process():\n    """A mission."""\n', "mission", 2),
        ('"""Clean first line.\nAn orbit.\n"""\n', "orbit", 2),
        ("nPlanet2 = 1\n", "planet", 1),
        ("TDBClock = 1\n", "tdb", 1),
        ("nTDBs = 1\n", "tdbs", 1),
        ('text = "fluxes ephemerides photometries"\n', "ephemerides", 1),
        ('text = "tar" "get"\n', "target", 1),
        ('text = ("clean "\n        "planet")\n', "planet", 2),
        ('text = "\\x70lanet"\n', "planet", 1),
        ('text = "clean\\nplanet"\n', "planet", 1),
        ('text = """clean\\nline\nplanet"""\n', "planet", 2),
        ('text = f"orbit {row}"\n', "orbit", 1),
        ('text = f"{planet}"\n', "planet", 1),
        ('text = f"{row.star}"\n', "star", 1),
        ('text = f"{row:{band}}"\n', "band", 1),
        ('text = f"{configure(target=0)}"\n', "target", 1),
        ('text = b"planet"\n', "planet", 1),
        ("planet = 1  # domain-word-ok\n", "planet", 1),
        ("planet = 1  # noqa\n", "planet", 1),
    ],
)
def test_lint_token_boundaries_and_strings(tmp_path, source, token, line):
    root, path = fixture_core(tmp_path, source)
    result = run_lint("check_domain_words.py", "--core-dir", root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"{path}:{line}:{token}" in result.stdout


def test_lint_passes_clean_fixture(tmp_path):
    root, _ = fixture_core(tmp_path, (FIXTURES / "clean.txt").read_text())
    result = run_lint("check_domain_words.py", "--core-dir", root)
    assert result.returncode == 0, result.stdout + result.stderr


def test_lint_scans_the_core():
    result = run_lint("check_domain_words.py")
    count = re.search(r"Scanned (\d+) Python files", result.stdout)
    assert count, result.stdout + result.stderr
    assert int(count[1]) >= len(list(CORE.rglob("*.py"))) > 0


def test_core_is_clean():
    result = run_lint("check_domain_words.py")
    assert result.returncode == 0, result.stdout + result.stderr


def write_allowlist(tmp_path, entries):
    path = tmp_path / "allowlist.toml"
    path.write_text(
        "".join(
            "[[entries]]\n"
            + "".join(f'{key} = "{value}"\n' for key, value in entry.items())
            for entry in entries
        )
    )
    return path


def test_allowlist_limits():
    data = tomllib.loads((REPO / "tools/domain_words_allowlist.toml").read_text())
    entries = data.get("entries", [])
    assert len(entries) <= 3
    for entry in entries:
        assert len(entry["reason"].strip()) >= 20
        assert entry["file"] and entry["token"]
        assert bool(entry.get("symbol")) != bool(entry.get("line"))
    result = run_lint("check_domain_words.py")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("scope", ["line", "symbol"])
def test_allowlist_is_narrow_and_checked_for_staleness(tmp_path, scope):
    root, _ = fixture_core(tmp_path, "def planet():\n    pass\n")
    entry = {
        "file": "sample.py",
        "token": "planet",
        scope: "def planet():" if scope == "line" else "planet",
        "reason": "A deliberately scoped test exception.",
    }
    allowlist = write_allowlist(tmp_path, [entry])
    args = ("--core-dir", root, "--allowlist", allowlist)
    result = run_lint("check_domain_words.py", *args)
    assert result.returncode == 0, result.stdout + result.stderr
    (root / "other.py").write_text("planet = 0\n")
    result = run_lint("check_domain_words.py", *args)
    assert result.returncode == 1
    assert "other.py:1:planet" in result.stdout
    (root / "sample.py").write_text("def subject():\n    pass\n")
    result = run_lint("check_domain_words.py", *args)
    assert result.returncode == 1
    assert "stale" in result.stdout.lower()


def test_allowlist_line_handles_decoded_newlines(tmp_path):
    source = 'text = "clean\\nplanet"'
    root, _ = fixture_core(tmp_path, source + "\n")
    entry = {
        "file": "sample.py",
        "token": "planet",
        "line": source,
        "reason": "A deliberately scoped test exception.",
    }
    # This source needs a literal backslash in its TOML line value.
    allowlist = tmp_path / "allowlist.toml"
    allowlist.write_text(
        "[[entries]]\n"
        + "".join(f"{key} = '{value}'\n" for key, value in entry.items())
    )
    result = run_lint(
        "check_domain_words.py", "--core-dir", root, "--allowlist", allowlist
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_allowlist_cannot_cover_an_unrelated_line(tmp_path):
    root, path = fixture_core(tmp_path, 'text = "clean\\nplanet"\nclean = 1\n')
    entry = {
        "file": "sample.py",
        "token": "planet",
        "line": "clean = 1",
        "reason": "A deliberately scoped test exception.",
    }
    allowlist = write_allowlist(tmp_path, [entry])
    result = run_lint(
        "check_domain_words.py", "--core-dir", root, "--allowlist", allowlist
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"{path}:1:planet" in result.stdout
    assert "stale" in result.stdout.lower()


@pytest.mark.parametrize(
    "fault", ["too_many", "short_reason", "missing_reason", "scope"]
)
def test_invalid_allowlist_fails(tmp_path, fault):
    root, _ = fixture_core(tmp_path, "planet = 0\n")
    entry = {
        "file": "sample.py",
        "token": "planet",
        "line": "planet = 0",
        "reason": "A deliberately scoped test exception.",
    }
    entries = [entry]
    if fault == "too_many":
        entries = [entry] * 4
    elif fault == "short_reason":
        entry["reason"] = "short"
    elif fault == "missing_reason":
        del entry["reason"]
    else:
        del entry["line"]
    allowlist = write_allowlist(tmp_path, entries)
    result = run_lint(
        "check_domain_words.py", "--core-dir", root, "--allowlist", allowlist
    )
    assert result.returncode == 1
    assert "allowlist" in result.stdout.lower()


@pytest.mark.parametrize("fixture", ["import_direct", "import_from"])
def test_core_imports_catches_fixtures(tmp_path, fixture):
    root, path = fixture_core(tmp_path, (FIXTURES / f"{fixture}.txt").read_text())
    result = run_lint("check_core_imports.py", "--core-dir", root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"{path}:1:" in result.stdout
    for library in PHYSICS:
        assert library in result.stdout
    assert "hwoutils is not allowed in the core in Slice 1" in result.stdout


@pytest.mark.parametrize(
    "source",
    [
        *(f"import {name}\n" for name in [*PHYSICS, "hwoutils", "scipy"]),
        "import jax, orbix\n",
        "from spaceodyssey.studies import toy\n",
        "from spaceodyssey import studies\n",
        "import spaceodyssey.core_extra\n",
        "from .. import studies\n",
        "from ... import core\n",
        "if False:\n    import orbix\n",
    ],
)
def test_core_imports_rejects_outside_boundary(tmp_path, source):
    root, _ = fixture_core(tmp_path, source)
    result = run_lint("check_core_imports.py", "--core-dir", root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "not allowed" in result.stdout


def test_core_imports_passes_clean_fixture(tmp_path):
    root, _ = fixture_core(tmp_path, (FIXTURES / "imports_clean.txt").read_text())
    (root / "executors").mkdir()
    (root / "executors/nested.py").write_text(
        "from .. import records\nfrom .host import HostExecutor\n"
    )
    result = run_lint("check_core_imports.py", "--core-dir", root)
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_imports_clean():
    result = run_lint("check_core_imports.py")
    assert result.returncode == 0, result.stdout + result.stderr
    count = re.search(r"Scanned (\d+) Python files", result.stdout)
    assert count and int(count[1]) >= len(list(CORE.rglob("*.py"))) > 0


@pytest.mark.parametrize("tool", ["check_domain_words.py", "check_core_imports.py"])
def test_lints_exclude_fixture_path(tmp_path, tool):
    with tempfile.TemporaryDirectory(dir=FIXTURES) as directory:
        path = Path(directory) / "bad.py"
        path.write_text("import orbix\nplanet = 0\n")
        result = run_lint(tool, "--core-dir", FIXTURES)
    assert "Scanned 0 Python files" in result.stdout
    assert f"{path}:" not in result.stdout


@pytest.mark.parametrize("tool", ["check_domain_words.py", "check_core_imports.py"])
def test_lints_scan_core_paths_resembling_fixtures(tmp_path, tool):
    root, path = fixture_core(
        tmp_path, "import orbix\nplanet = 0\n", "tests/core/lint_fixtures/bad.py"
    )
    result = run_lint(tool, "--core-dir", root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"{path}:" in result.stdout
    assert "Scanned 1 Python files" in result.stdout


@pytest.mark.parametrize("tool", ["check_domain_words.py", "check_core_imports.py"])
@pytest.mark.parametrize("source", [None, "def broken(\n"])
def test_lints_fail_closed(tmp_path, tool, source):
    root = tmp_path / "core"
    root.mkdir()
    if source is not None:
        (root / "broken.py").write_text(source)
    result = run_lint(tool, "--core-dir", root)
    assert result.returncode == 1, result.stdout + result.stderr


def test_lint_runs_in_pre_commit():
    config = (REPO / ".pre-commit-config.yaml").read_text()
    for name in ("check-domain-words", "check-core-imports"):
        block = re.search(rf"(?m)^( *)- id: {name}\n((?:\1  .*\n)*)", config)
        assert block, f"missing hook: {name}"
        hook = dict(line.strip().split(": ", 1) for line in block[2].splitlines())
        assert hook["language"] == "system"
        assert hook["pass_filenames"] == "false"
        assert hook["entry"] == f"python3 tools/{name.replace('-', '_')}.py"
        assert re.search(hook["files"], "src/spaceodyssey/core/events.py")
        assert not re.search(hook["files"], "src/spaceodyssey/studies/toy.py")
        assert not re.search(hook["files"], "tests/core/lint_fixtures/bad.py")
