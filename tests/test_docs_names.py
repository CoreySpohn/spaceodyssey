"""No document names a deleted module or interface (Slice 2 gate B5)."""

import re

import pytest

from tests.test_slice2_gate import DELETED_MODULES, DELETED_NAMES, REPO, SELF, SKIP_DIRS

DOC_SUFFIXES = {".md", ".rst", ".txt", ".d2", ".svg", ".ipynb"}
# Dotted module paths that have no deleted-module component of their own.
EXTRA_PATTERNS = (r"spaceodyssey\.(?:analytic|expected|mc)\b",)


def doc_files():
    """The README and every doc under docs/, except the change records."""
    paths = [REPO / "README.md", REPO / "SECURITY.md"]
    for path in (REPO / "docs").rglob("*"):
        if not path.is_file() or path.suffix not in DOC_SUFFIXES:
            continue
        parts = path.relative_to(REPO).parts
        if SKIP_DIRS & set(parts) or parts[:2] == ("docs", "changes"):
            continue
        paths.append(path)
    return [p for p in paths if p.exists() and p.name not in SELF]


def deleted_name_pattern():
    modules = "|".join(sorted(DELETED_MODULES))
    names = "|".join(sorted(DELETED_NAMES))
    return re.compile(
        rf"spaceodyssey\.(?:{modules})\b|\b(?:{names})\b|" + "|".join(EXTRA_PATTERNS)
    )


@pytest.mark.parametrize("path", doc_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_doc_names_no_deleted_interface(path):
    """Fails if a doc names a deleted module or public name of the legacy API."""
    found = sorted(set(deleted_name_pattern().findall(path.read_text())))
    assert not found, found


def test_name_scan_matches_each_kind():
    """Fails if the pattern misses a module path, a class or a function name."""
    pattern = deleted_name_pattern()
    for text in (
        "see spaceodyssey.registry for details",
        "a `MissionSpec` document",
        "call run_mission(...)",
        "the spaceodyssey.mc engine",
    ):
        assert pattern.search(text), text
    assert not pattern.search("build the core with spaceodyssey.core and the codec")
