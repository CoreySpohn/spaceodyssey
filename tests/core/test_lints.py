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


@pytest.mark.parametrize(
    "fixture",
    [
        "dynamic_importlib",
        "dynamic_dunder",
        "dynamic_reexport",
        "dynamic_spec",
        "dynamic_loader",
        "dynamic_alias",
        "dynamic_unknown",
    ],
)
def test_core_imports_rejects_dynamic_loading(tmp_path, fixture):
    root, path = fixture_core(tmp_path, (FIXTURES / f"{fixture}.txt").read_text())
    result = run_lint("check_core_imports.py", "--core-dir", root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"{path}:" in result.stdout and "dynamic" in result.stdout


@pytest.mark.parametrize(
    "fixture",
    ["getattr", "typename", "match", "class-dict", "format", "helper"],
)
def test_dispatch_forms_are_rejected_everywhere(tmp_path, fixture):
    from tests.core.gate_helpers import study_branches

    source = (FIXTURES / f"dispatch_{fixture}.txt").read_text()
    root, _ = fixture_core(tmp_path, source, name="nested/dispatch_helper.py")
    assert study_branches(root), source


def test_dispatch_allowlist_has_no_stale_entries():
    from tests.core.gate_helpers import ALLOWED, stale_allowances

    assert ALLOWED
    assert stale_allowances(CORE) == []


@pytest.mark.parametrize(
    "source",
    [
        'text = "survey-782"\n',
        'text = f"survey_{number}"\n',
        "label = study.name\n",
        "label = campaign.study.name\n",
        "label = study.channel_model.__class__\n",
        "label = model.__qualname__\n",
        "label = getattr(model, attribute)\n",
        'kind = getattr(model, "kind")\nif kind == selected:\n    pass\n',
        "match model:\n    case Concrete(): pass\n",
        "if issubclass(model, Concrete):\n    pass\n",
        "if isinstance(model, (int, Concrete)):\n    pass\n",
        "dispatch = {Concrete: handle}\nresult = dispatch[cls]\n",
        "result = dispatch[Concrete]\n",
        "result = dispatch.get(Concrete)\n",
        "from builtins import type as kind_of\nresult = kind_of(model)\n",
        (
            "from builtins import isinstance as check\n"
            "if check(model, Concrete):\n"
            "    pass\n"
        ),
    ],
)
def test_dispatch_is_forbidden_without_an_explicit_exception(tmp_path, source):
    from tests.core.gate_helpers import study_branches

    root, _ = fixture_core(tmp_path, source)
    assert study_branches(root), source


def test_dispatch_allows_only_declared_type_contracts(tmp_path):
    from tests.core.gate_helpers import study_branches

    source = (
        "from spaceodyssey.core.enums import Stage as Phase\n"
        "from spaceodyssey.core.kinds import AbstractPolicy as Policy\n"
        "if isinstance(value, (int, str, Phase, Policy)):\n    pass\n"
        'if event.name == "release":\n    pass\n'
    )
    root, _ = fixture_core(tmp_path, source)
    assert study_branches(root) == []


@pytest.mark.parametrize(
    "source",
    [
        "import runpy\nrunpy.run_path(where)\n",
        "result = eval(text)\n",
        "exec(compile(text, 'x', 'exec'))\n",
        "from builtins import __import__ as load\nload(name)\n",
        "import pkgutil\n",
        "module = loader.load_module(name)\n",
        "f = __import__\nowned = f('orbix')\n",
        "import builtins\nowned = getattr(builtins, '__import__')('orbix')\n",
        "import builtins\nowned = vars(builtins)['__import__']('orbix')\n",
        "import builtins\nbuiltins.exec('import orbix')\n",
        "import pydoc\nowned = pydoc.locate('orbix')\n",
        "import pickle\nowned = pickle.loads(data)\n",
        "import unittest.mock as m\nowned = m.patch('orbix.x')\n",
        "from unittest import mock\n",
        "import marshal\n",
        "run = tool.exec\n",
        "make = obj.compile\n",
        "owned = globals()['__import__']('orbix')\n",
        "owned = getattr(module, name)(arg)\n",
        "owned = sys_modules.__dict__['x'](arg)\n",
    ],
    ids=lambda s: s.replace("\n", " ; ")[:60],
)
def test_core_imports_rejects_other_dynamic_loading_spellings(tmp_path, source):
    root, path = fixture_core(tmp_path, source)
    result = run_lint("check_core_imports.py", "--core-dir", root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"{path}:" in result.stdout and "dynamic" in result.stdout


@pytest.mark.parametrize(
    "source",
    [
        "import jax\nout = jax.vmap(one)(xs)\n",
        "out = branches[0](a, b)\n",
        "new = type(value)(items)\n",
        "import re\nout = re.match('a', text)\n",
    ],
)
def test_core_imports_accepts_ordinary_calls_through_calls_and_tables(tmp_path, source):
    root, _ = fixture_core(tmp_path, source)
    result = run_lint("check_core_imports.py", "--core-dir", root)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("source", "rule"),
    [
        ("ok = study.label == 'survey_' + str(1)\n", "survey-name"),
        ("ok = study.label == 'survey_' + str(1)\n", "survey-id"),
        ("name = 'survey_%d' % number\n", "survey-name"),
        ("name = 'survey_{}'.format(number)\n", "survey-name"),
        ("ok = 'ToyPhotometry' in repr(study.channel_model)\n", "study-text"),
        ("ok = 'ToyPhotometry' in label\n", "kind-name"),
        ("ok = str(jax.tree_util.tree_structure(study)) == text\n", "study-text"),
        ("ok = study.channel_model.__module__.endswith('toy')\n", "introspection"),
        ("ok = vars(study)['name'] == other\n", "study-vars"),
        ("ok = study.channel_model.__dict__ == other\n", "study-vars"),
        ("ok = hasattr(study.channel_model, 'gain')\n", "study-hasattr"),
        ("ok = study.survey_id == 25\n", "survey-id"),
        ("ok = 25 == model.survey_id\n", "survey-id"),
    ],
)
def test_dispatch_neighbouring_spellings_are_rejected(tmp_path, source, rule):
    from tests.core.gate_helpers import study_branches

    root, _ = fixture_core(tmp_path, source)
    assert rule in {hit[2] for hit in study_branches(root)}, source


def test_allowance_is_scoped_to_rule_function_and_count(tmp_path):
    import shutil

    from tests.core.gate_helpers import study_branches

    line = "bool(getattr(carry, n))"
    copy = tmp_path / "core"
    shutil.copytree(CORE, copy)
    base = copy / "executors/base.py"
    assert study_branches(copy) == []

    # the identical line in another function is not exempt
    original = base.read_text()
    base.write_text(
        original + f"\n\ndef other(carry, names):\n    return [{line} for n in names]\n"
    )
    assert [hit[2] for hit in study_branches(copy)] == ["dynamic-getattr"]


def test_allowance_count_and_rule_are_enforced(tmp_path):
    from tests.core.gate_helpers import Allowance, stale_allowances, study_branches

    source = (
        "def walk(carry, n):\n"
        "    a = getattr(carry, n)\n"
        "    b = getattr(carry, n)\n"
        "    return a, b\n"
    )
    root, _ = fixture_core(tmp_path, source)
    key = ("sample.py", "a = getattr(carry, n)")
    other = ("sample.py", "b = getattr(carry, n)")
    once = {key: Allowance("test", ("dynamic-getattr",), "walk", 1)}
    assert [hit[1] for hit in study_branches(root, once)] == [3]
    both = {**once, other: Allowance("test", ("dynamic-getattr",), "walk", 1)}
    assert study_branches(root, both) == [] and stale_allowances(root, both) == []
    wrong_rule = {key: Allowance("test", ("study-kind",), "walk", 1)}
    assert len(study_branches(root, wrong_rule)) == 2
    assert stale_allowances(root, wrong_rule) == [key]
    twice = {key: Allowance("test", ("dynamic-getattr",), "walk", 2)}
    assert stale_allowances(root, twice) == [key]
