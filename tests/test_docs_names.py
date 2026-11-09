"""No file, document or published docstring names a deleted module or interface."""

import ast
import json
import re
from pathlib import Path

import pytest

from tests.legacy_scan import (
    DELETED_MODULES,
    DELETED_NAMES,
    DELETED_ROOT_NAMES,
    MIN_PYTHON_FILES,
    PACKAGE,
    REPO,
    SELF,
    SKIP_DIRS,
    scan_repo,
    walk_repo,
)
from tests.legacy_scan import SENTINELS as SCAN_SENTINELS

DOC_SUFFIXES = {".md", ".rst", ".txt", ".d2", ".svg", ".ipynb"}
TOP_LEVEL_DOCS = ("README.md", "SECURITY.md")
# Dotted module paths that have no deleted-module component of their own.
EXTRA_PATTERNS = (r"spaceodyssey\.(?:analytic|expected|mc)\b",)
# What a scan of this repository must reach.
SENTINELS = ("README.md", "docs/architecture.md", "docs/core.md")
SOURCE_SENTINELS = ("src/spaceodyssey/__init__.py", "src/spaceodyssey/core/study.py")


def doc_files(root: Path = REPO) -> list[Path]:
    """The README and every doc under docs/."""
    paths = [root / name for name in TOP_LEVEL_DOCS if (root / name).exists()]
    for path in sorted((root / "docs").rglob("*")):
        if not path.is_file() or path.suffix not in DOC_SUFFIXES:
            continue
        parts = path.relative_to(root).parts
        if SKIP_DIRS & set(parts):
            continue
        paths.append(path)
    return [p for p in paths if p.name not in SELF]


def source_files(root: Path = REPO) -> list[Path]:
    """The package modules whose docstrings AutoAPI publishes."""
    return sorted((root / "src" / PACKAGE).rglob("*.py"))


def docstrings(path: Path) -> str:
    """Every module, class and function docstring of a source file, joined."""
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            text = ast.get_docstring(node, clean=False)
            if text:
                found.append(text)
    return "\n".join(found)


def deleted_name_pattern():
    qualified = "|".join(sorted(DELETED_MODULES | DELETED_ROOT_NAMES))
    names = "|".join(sorted(DELETED_NAMES))
    # Root-only type names can also be ordinary prose or surviving core types.
    root_types = "|".join(
        sorted(n for n in DELETED_ROOT_NAMES - DELETED_NAMES if n[0].isupper())
    )
    return re.compile(
        rf"\bspaceodyssey\.(?:{qualified})\b|\b(?:{names})\b|"
        rf"\b(?:a|an)\s+(?:{root_types})\b|`{{1,2}}(?:{root_types})`{{1,2}}|"
        + "|".join(EXTRA_PATTERNS)
    )


def scan_docs(
    root: Path = REPO, *, sentinels=SENTINELS, source_sentinels=SOURCE_SENTINELS
):
    """Map each doc or source path to the deleted names it contains.

    Raises AssertionError if the scan did not reach its sentinel files, so a
    broken root cannot pass with nothing scanned.
    """
    pattern = deleted_name_pattern()
    docs, sources = doc_files(root), source_files(root)
    reached = {p.relative_to(root).as_posix() for p in docs if p.exists()}
    reached_sources = {p.relative_to(root).as_posix() for p in sources}
    assert set(sentinels) <= reached, f"docs scan missed {set(sentinels) - reached}"
    assert set(source_sentinels) <= reached_sources, "source scan missed its sentinels"
    found = {}
    for path in docs:
        hits = sorted(set(pattern.findall(path.read_text())))
        if hits:
            found[path.relative_to(root).as_posix()] = hits
    for path in sources:
        hits = sorted(set(pattern.findall(docstrings(path))))
        if hits:
            found[path.relative_to(root).as_posix()] = hits
    return found


def test_docs_and_docstrings_name_no_deleted_interface():
    """Fails if a doc or a published docstring names a deleted module or public name."""
    assert scan_docs() == {}


def test_doc_set_is_not_empty():
    """Fails if the doc scan covers fewer files than the repository's docs."""
    names = {p.relative_to(REPO).as_posix() for p in doc_files()}
    assert set(SENTINELS) <= names
    assert any(n.endswith(".svg") for n in names)
    assert len(source_files()) >= 40


def _make_root(tmp_path, extra):
    files = {
        "README.md": "readme\n",
        "docs/architecture.md": "arch\n",
        "docs/core.md": "core\n",
        "src/spaceodyssey/__init__.py": '"""Package."""\n',
        "src/spaceodyssey/core/study.py": '"""Distributions."""\n',
        **extra,
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("docs/architecture.md", "see spaceodyssey.registry for details"),
        ("docs/core.md", "a `MissionSpec` document"),
        ("README.md", "call run_mission(...)"),
        ("docs/guide.md", "the spaceodyssey.mc engine"),
        ("docs/diagram.d2", "label: 'MissionSpec, seam registries'"),
        ("src/spaceodyssey/core/study.py", '"""Engines use the Registry."""\n'),
        ("src/spaceodyssey/extra.py", 'def f():\n    """Calls run_mission."""\n'),
        (
            "src/spaceodyssey/method.py",
            "class Example:\n    def make(self):\n"
            '        """Construct a spaceodyssey.Mission with a Ledger."""\n',
        ),
        ("docs/mission.md", "Construct a Mission."),
        ("docs/ledger.md", "Construct a spaceodyssey.Ledger."),
        ("docs/bare-ledger.md", "Construct a Ledger."),
        ("docs/root-type.md", "Use `Mission`."),
        ("docs/run.md", "Call spaceodyssey.run."),
    ],
)
def test_scan_finds_each_kind(tmp_path, name, text):
    """Fails if the full scan misses a module path, a name, or a docstring."""
    root = _make_root(tmp_path, {name: text})
    assert name in scan_docs(root), name


def test_scan_passes_clean_docs_and_requires_coverage(tmp_path):
    """Fails if clean docs are flagged or a root without the docs passes."""
    root = _make_root(
        tmp_path / "ok",
        {
            "docs/guide.md": (
                "Mission planning: build the core and run it.\n"
                "Use spaceodyssey.core.ledger.Ledger for accounting.\n"
            )
        },
    )
    assert scan_docs(root) == {}
    (tmp_path / "empty" / "src" / PACKAGE).mkdir(parents=True)
    with pytest.raises(AssertionError):
        scan_docs(tmp_path / "empty")


def test_no_legacy_imports():
    """Fails if any source, test, doc, notebook or config refers to a deleted module.

    Change records under ``changes`` name deleted modules on purpose and
    their prose is exempt; Python and notebook code there is still scanned.
    """
    assert scan_repo(REPO) == {}


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


SENTINEL_FILES = {
    "pyproject.toml": "[project]\nname = 'x'\n",
    "src/spaceodyssey/__init__.py": '__all__ = ["core"]\n',
    "src/spaceodyssey/core/study.py": "",
    "docs/architecture.md": "",
    "tests/conftest.py": "",
}

VIOLATIONS = {
    "tests/test_plain.py": "import spaceodyssey.registry\n",
    "tests/test_from.py": "from spaceodyssey.spec import MissionSpec\n",
    "tests/test_package.py": "from spaceodyssey import registry\n",
    "tests/test_root_name.py": "from spaceodyssey import Ledger, Mission, run\n",
    "tests/test_root_attribute.py": "import spaceodyssey\nspaceodyssey.Mission\n",
    "tests/test_alias_attribute.py": "import spaceodyssey as so; so.Mission\n",
    "tests/test_module_attribute.py": (
        "import spaceodyssey\nspaceodyssey.registry.REGISTRY\n"
    ),
    "tests/test_alias_module_attribute.py": (
        "import spaceodyssey as so\nso.registry.REGISTRY\n"
    ),
    "tests/test_alias_function_attribute.py": "import spaceodyssey as so\nso.run\n",
    "tests/test_string.py": (
        "import importlib\nimportlib.import_module('spaceodyssey.wiring')\n"
    ),
    "src/spaceodyssey/relative.py": "from .registry import REGISTRY\n",
    "src/spaceodyssey/core/relative_up.py": "from ..loop import step\n",
    "src/spaceodyssey/relative_pkg.py": "from . import build\n",
    "docs/guide.md": "```python\nfrom spaceodyssey.engines import analytic\n```\n",
    "docs/root.md": "```python\nfrom spaceodyssey import build, core\n```\n",
    "changes/broken.py": "import spaceodyssey.registry\n",
    "changes/pyproject.toml": (
        "[project.entry-points.'spaceodyssey.core']\n"
        "'reg@1' = 'spaceodyssey.registry:Registry'\n"
    ),
    "changes/broken.yaml": "target: spaceodyssey.registry:Registry\n",
    "changes/broken.yml": "target: spaceodyssey.registry:Registry\n",
    "changes/broken.cfg": "target = spaceodyssey.registry:Registry\n",
    "changes/broken.ipynb": json.dumps(
        {
            "cells": [
                {"cell_type": "code", "source": ["import spaceodyssey.loop\n"]},
                {"cell_type": "markdown", "source": ["spaceodyssey.registry"]},
            ]
        }
    ),
    "pyproject.toml": (
        "[project.entry-points.'spaceodyssey.core']\n"
        "'reg@1' = 'spaceodyssey.registry:Registry'\n"
    ),
    "missing/pyproject.toml": (
        "[project.entry-points.'spaceodyssey.core']\n"
        "'gone@1' = 'spaceodyssey.core.missing:Thing'\n"
    ),
    "root-name/pyproject.toml": (
        "[project.entry-points.'spaceodyssey.core']\n"
        "'mission@1' = 'spaceodyssey:Mission'\n"
    ),
    "root-module/pyproject.toml": (
        "[project.entry-points.'spaceodyssey.core']\n"
        "'reg@1' = 'spaceodyssey:registry.REGISTRY'\n"
    ),
    "root-script/pyproject.toml": "[project.scripts]\nrun = 'spaceodyssey:run'\n",
}


@pytest.mark.parametrize("name", sorted(VIOLATIONS))
def test_scan_catches_each_form(tmp_path, name):
    """Fails if the full scan misses one form of reference (negative control)."""
    _write(tmp_path, {**SENTINEL_FILES, name: VIOLATIONS[name]})
    found = scan_repo(tmp_path, min_python=1)
    assert name in found, found


def test_scan_ignores_history_prose_and_valid_imports(tmp_path):
    """Fails if the scan flags change-record prose or imports that still exist."""
    _write(
        tmp_path,
        {
            **SENTINEL_FILES,
            "changes/history.md": "`spaceodyssey.registry` was removed.\n",
            "changes/history.rst": "spaceodyssey.registry was removed.\n",
            "changes/history.txt": "spaceodyssey.registry was removed.\n",
            "tests/test_ok.py": (
                "from spaceodyssey import core\nfrom spaceodyssey.core import study\n"
                "import spaceodyssey\nspaceodyssey.core\n"
                "import spaceodyssey as so\nso.core\nso.__version__\n"
                "import spaceodyssey.core as core\ncore.run\n"
                "other.Mission\n"
            ),
            "src/spaceodyssey/core/sibling.py": "from . import study\n",
            "valid/pyproject.toml": (
                "[project.entry-points.'spaceodyssey.core']\n"
                "'core@1' = 'spaceodyssey:core.study'\n"
                "'study@1' = 'spaceodyssey.core.study:Study'\n"
                "'external@1' = 'other:Mission'\n"
            ),
        },
    )
    assert scan_repo(tmp_path, min_python=1) == {}


def test_scan_requires_coverage(tmp_path):
    """Fails if an absent root or a thin scan can pass."""
    with pytest.raises(AssertionError, match="did not reach"):
        scan_repo(_empty_root(tmp_path), min_python=1)
    _write(tmp_path / "thin", SENTINEL_FILES)
    with pytest.raises(AssertionError, match="Python files"):
        scan_repo(tmp_path / "thin", min_python=50)


def _empty_root(tmp_path):
    root = tmp_path / "empty"
    (root / "src" / PACKAGE).mkdir(parents=True)
    return root


def test_repo_scan_reaches_this_repository():
    """Fails if the real scan stops reading the repository's own files."""
    scanned = {rel.as_posix() for _, rel in walk_repo(REPO)}
    assert set(SCAN_SENTINELS) <= scanned
    assert sum(r.endswith(".py") for r in scanned) >= MIN_PYTHON_FILES
