"""No document or published docstring names a deleted module or interface (gate B5)."""

import ast
import re
from pathlib import Path

import pytest

from tests.test_slice2_gate import (
    DELETED_MODULES,
    DELETED_NAMES,
    PACKAGE,
    REPO,
    SELF,
    SKIP_DIRS,
)

DOC_SUFFIXES = {".md", ".rst", ".txt", ".d2", ".svg", ".ipynb"}
TOP_LEVEL_DOCS = ("README.md", "SECURITY.md")
# Dotted module paths that have no deleted-module component of their own.
EXTRA_PATTERNS = (r"spaceodyssey\.(?:analytic|expected|mc)\b",)
# What a scan of this repository must reach.
SENTINELS = ("README.md", "docs/architecture.md", "docs/core.md")
SOURCE_SENTINELS = ("src/spaceodyssey/__init__.py", "src/spaceodyssey/dist.py")


def doc_files(root: Path = REPO) -> list[Path]:
    """The README and every doc under docs/, except the change records."""
    paths = [root / name for name in TOP_LEVEL_DOCS if (root / name).exists()]
    for path in sorted((root / "docs").rglob("*")):
        if not path.is_file() or path.suffix not in DOC_SUFFIXES:
            continue
        parts = path.relative_to(root).parts
        if SKIP_DIRS & set(parts) or parts[:2] == ("docs", "changes"):
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
    modules = "|".join(sorted(DELETED_MODULES))
    names = "|".join(sorted(DELETED_NAMES))
    return re.compile(
        rf"spaceodyssey\.(?:{modules})\b|\b(?:{names})\b|" + "|".join(EXTRA_PATTERNS)
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
        "src/spaceodyssey/dist.py": '"""Distributions."""\n',
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
        ("src/spaceodyssey/dist.py", '"""Engines use the Registry."""\n'),
        ("src/spaceodyssey/extra.py", 'def f():\n    """Calls run_mission."""\n'),
    ],
)
def test_scan_finds_each_kind(tmp_path, name, text):
    """Fails if the full scan misses a module path, a name, or a docstring."""
    root = _make_root(tmp_path, {name: text})
    assert name in scan_docs(root), name


def test_scan_passes_clean_docs_and_requires_coverage(tmp_path):
    """Fails if clean docs are flagged or a root without the docs passes."""
    root = _make_root(tmp_path / "ok", {"docs/guide.md": "build the core\n"})
    assert scan_docs(root) == {}
    (tmp_path / "empty" / "src" / PACKAGE).mkdir(parents=True)
    with pytest.raises(AssertionError):
        scan_docs(tmp_path / "empty")
