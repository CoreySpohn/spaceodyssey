"""Slice 2 gate checks that read the repository itself.

B5: no source, test, doc or script imports a module that Slice 2 deleted.
"""

import ast
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE = "spaceodyssey"

# Modules removed with the legacy spec, registry and wiring closure. A submodule
# of a deleted package counts as deleted too.
DELETED_MODULES = frozenset(
    {
        "account",
        "aliases",
        "belief",
        "build",
        "certify",
        "context",
        "contracts",
        "engines",
        "gaussian_profile",
        "io",
        "loop",
        "registry",
        "seams",
        "spec",
        "testing",
        "wiring",
    }
)

# Public names of the deleted modules that no surviving module defines. A
# generic word such as "build" is left out so that prose is not flagged.
DELETED_NAMES = frozenset(
    {
        "AbstractCertificate",
        "AbstractContextProvider",
        "AbstractProfileCertificate",
        "AlwaysObservable",
        "CertificateState",
        "ConstantOverheads",
        "DATA_PLANE_EDGES",
        "DETECTION_CERTIFICATE",
        "FidelityConfig",
        "FileRef",
        "GaussianProfileCertifier",
        "KNOWN_OPERATORS",
        "MissionBelief",
        "MissionSpec",
        "MissionState",
        "ModelStack",
        "QUALNAME_ALIASES",
        "REGISTRY",
        "RegisteredImpl",
        "Registry",
        "RegistryError",
        "RunManifest",
        "SEAMS",
        "SeamChoice",
        "SeamInfo",
        "SeamRef",
        "SnrThreshold",
        "TargetBelief",
        "ToyGaussianCertifier",
        "TruthStack",
        "WiringError",
        "WiringFailure",
        "WiringReport",
        "config_from_json",
        "get_engine",
        "read_manifest",
        "register_engine",
        "resolve_contract",
        "resolve_qualname_alias",
        "run_integrate",
        "run_mission",
        "run_sample",
        "spec_from_json",
        "validate_config",
        "write_run",
    }
)

SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        ".ruff_cache",
        ".pytest_cache",
        ".mypy_cache",
        "__pycache__",
        "_build",
        "autoapi",
        "build",
        "dist",
        "node_modules",
    }
)
# These two files list the deleted names on purpose.
SELF = frozenset({"test_slice2_gate.py", "test_docs_names.py"})
TEXT_SUFFIXES = frozenset({".md", ".rst", ".txt", ".toml", ".yml", ".yaml", ".cfg"})

_FROM_RE = re.compile(
    r"^\s*from\s+spaceodyssey(?:\.(\w+))?[\w.]*\s+import\s+(.+)$", re.M
)
_IMPORT_RE = re.compile(r"^\s*import\s+spaceodyssey\.(\w+)", re.M)


def _hits_module(top: str | None, imported: list[str]) -> list[str]:
    """Return the deleted modules named by ``from spaceodyssey[.top] import ...``."""
    if top is not None:
        return [top] if top in DELETED_MODULES else []
    return [name for name in imported if name in DELETED_MODULES]


def python_imports_of_deleted(source: str) -> list[str]:
    """Deleted modules or names that a Python source imports or names in a string."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if (
                    parts[0] == PACKAGE
                    and len(parts) > 1
                    and parts[1] in DELETED_MODULES
                ):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            parts = node.module.split(".")
            if parts[0] != PACKAGE:
                continue
            if len(parts) > 1:
                if parts[1] in DELETED_MODULES:
                    hits.append(node.module)
            else:
                for alias in node.names:
                    if alias.name in DELETED_MODULES | DELETED_NAMES:
                        hits.append(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            parts = node.value.split(".")
            if (
                len(parts) >= 2
                and parts[0] == PACKAGE
                and parts[1] in DELETED_MODULES
                and all(p.isidentifier() for p in parts)
            ):
                hits.append(node.value)
    return hits


def text_imports_of_deleted(text: str) -> list[str]:
    """Deleted modules that code in a text file imports."""
    hits = [
        f"spaceodyssey.{m}" for m in _IMPORT_RE.findall(text) if m in DELETED_MODULES
    ]
    for top, tail in _FROM_RE.findall(text):
        names = [n.strip(" ()") for n in re.split(r"[,\s]+", tail) if n.strip(" ()")]
        hits += [f"spaceodyssey.{m}" for m in _hits_module(top or None, names)]
    return hits


def _notebook_code(path: Path) -> str:
    cells = json.loads(path.read_text())["cells"]
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def repo_files():
    """Every file under the repository root that can hold an import."""
    for path in REPO.rglob("*"):
        if not path.is_file() or path.name in SELF:
            continue
        if SKIP_DIRS & set(path.relative_to(REPO).parts):
            continue
        if path.name.startswith(".codex-"):
            continue
        if path.suffix in {".py", ".ipynb"} | TEXT_SUFFIXES:
            yield path


def test_no_legacy_imports():
    """Fails if any source, test, doc, notebook or script imports a deleted module.

    Docs that record history under ``docs/changes`` name deleted modules on
    purpose and are exempt; everything else is scanned.
    """
    offenders = {}
    for path in repo_files():
        rel = path.relative_to(REPO)
        if rel.parts[:2] == ("docs", "changes"):
            continue
        if path.suffix == ".py":
            hits = python_imports_of_deleted(path.read_text())
        elif path.suffix == ".ipynb":
            hits = python_imports_of_deleted(_notebook_code(path))
        else:
            hits = text_imports_of_deleted(path.read_text())
        if hits:
            offenders[str(rel)] = sorted(set(hits))
    assert not offenders, offenders


def test_import_scan_catches_each_form():
    """Fails if the scan misses a form of import (negative control for the gate)."""
    cases = [
        "import spaceodyssey.registry",
        "from spaceodyssey.spec import MissionSpec",
        "from spaceodyssey import registry",
        "from spaceodyssey import REGISTRY",
        "import importlib\nimportlib.import_module('spaceodyssey.wiring')",
    ]
    for case in cases:
        assert python_imports_of_deleted(case), case
    assert not python_imports_of_deleted(
        "from spaceodyssey import core, codec\n"
        "from spaceodyssey.plugins import FAMILIES"
    )
    assert text_imports_of_deleted("    from spaceodyssey.engines import analytic")
    assert text_imports_of_deleted("from spaceodyssey import build, core")
    assert not text_imports_of_deleted("from spaceodyssey import core")
