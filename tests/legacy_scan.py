"""Scanner for references to modules and names the package no longer has.

``scan_repo`` takes a root so that the negative controls in ``test_docs_names.py``
can run the whole scanner, traversal included, over a small repository with injected
violations. The lists of removed modules and names are also read by that file.
"""

import ast
import json
import re
import tomllib
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
# generic word such as "build" is left out so that prose is not flagged. Python
# imports from the package root are checked against the real export set instead,
# which also catches the generic names.
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

# Generic removed root exports are checked when qualified, not in bare prose.
DELETED_ROOT_NAMES = DELETED_NAMES | {"Ledger", "Mission", "build", "run"}

SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        ".worktrees",
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
# These files list the deleted names on purpose.
SELF = frozenset({"legacy_scan.py", "test_docs_names.py"})
PROSE_SUFFIXES = frozenset({".md", ".rst", ".txt"})
TEXT_SUFFIXES = PROSE_SUFFIXES | {".toml", ".yml", ".yaml", ".cfg"}
CODE_SUFFIXES = frozenset({".py", ".ipynb"})

# Files that a scan of this repository must reach, and how many Python files it
# must read, so that a broken root or traversal cannot pass by scanning nothing.
SENTINELS = (
    "pyproject.toml",
    "src/spaceodyssey/__init__.py",
    "src/spaceodyssey/core/study.py",
    "docs/architecture.md",
    "tests/conftest.py",
)
MIN_PYTHON_FILES = 50

_FROM_RE = re.compile(r"^\s*from\s+spaceodyssey\s+import\s+(.+)$", re.M)
_MODULE_RE = re.compile(r"\bspaceodyssey\.(\w+)")


def walk_repo(root: Path):
    """Every file under ``root`` that can hold a reference, with its relative path."""
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in SELF:
            continue
        rel = path.relative_to(root)
        if SKIP_DIRS & set(rel.parts) or path.name.startswith(".codex-"):
            continue
        yield path, rel


def valid_root_names(root: Path) -> set[str]:
    """Names that ``from spaceodyssey import name`` may use: modules and ``__all__``."""
    package = root / "src" / PACKAGE
    names = {p.stem for p in package.glob("*.py")}
    names |= {p.name for p in package.iterdir() if p.is_dir()}
    init = package / "__init__.py"
    if init.exists():
        for node in ast.parse(init.read_text()).body:
            if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "__all__" for t in node.targets
            ):
                names |= set(ast.literal_eval(node.value))
    return names


def _resolve(node: ast.ImportFrom, rel: Path) -> list[str] | None:
    """The absolute module parts of a ``from`` import; None if not resolvable."""
    parts = node.module.split(".") if node.module else []
    if node.level == 0:
        return parts
    if rel.parts[:2] != ("src", PACKAGE):
        return None
    package = list(rel.parts[1:-1])
    keep = len(package) - (node.level - 1)
    if keep < 1:
        return None
    return package[:keep] + parts


def python_hits(source: str, rel: Path, valid: set[str]) -> list[str]:
    """Deleted imports, package attributes and module strings in Python source."""
    hits = []
    tree = ast.parse(source)
    package_aliases = {
        alias.asname or PACKAGE
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == PACKAGE
        or (alias.name.startswith(f"{PACKAGE}.") and alias.asname is None)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if (
                    parts[0] == PACKAGE
                    and len(parts) > 1
                    and parts[1] in DELETED_MODULES
                ):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            parts = _resolve(node, rel)
            if not parts or parts[0] != PACKAGE:
                continue
            if len(parts) > 1:
                if parts[1] in DELETED_MODULES:
                    hits.append(".".join(parts))
            else:
                for alias in node.names:
                    if alias.name != "*" and alias.name not in valid:
                        hits.append(f"{PACKAGE}.{alias.name}")
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in package_aliases
            and node.attr in DELETED_ROOT_NAMES | DELETED_MODULES
        ):
            hits.append(f"{PACKAGE}.{node.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            parts = node.value.split(":")[0].split(".")
            if (
                len(parts) >= 2
                and parts[0] == PACKAGE
                and parts[1] in DELETED_MODULES
                and all(p.isidentifier() for p in parts)
            ):
                hits.append(node.value)
    return hits


def text_hits(text: str, valid: set[str]) -> list[str]:
    """Deleted modules that a text file names or imports."""
    hits = [f"{PACKAGE}.{m}" for m in _MODULE_RE.findall(text) if m in DELETED_MODULES]
    for tail in _FROM_RE.findall(text):
        for name in re.split(r"[,\s()]+", tail):
            if name and name != "*" and name.isidentifier() and name not in valid:
                hits.append(f"{PACKAGE}.{name}")
    return hits


def entry_point_hits(text: str, root: Path) -> list[str]:
    """Entry-point and script targets naming missing modules or removed root names."""
    hits = []
    project = tomllib.loads(text).get("project", {})
    groups = list(project.get("entry-points", {}).values())
    groups.append(project.get("scripts", {}))
    for group in groups:
        for target in group.values():
            module_name, _, attr = str(target).partition(":")
            module = module_name.split(".")
            if module[0] != PACKAGE:
                continue
            base = root / "src" / Path(*module)
            if module[1:2] and module[1] in DELETED_MODULES:
                hits.append(str(target))
            elif module == [PACKAGE] and attr.split(".")[0] in (
                DELETED_ROOT_NAMES | DELETED_MODULES
            ):
                hits.append(str(target))
            elif not (base.with_suffix(".py").exists() or base.is_dir()):
                hits.append(str(target))
    return hits


def _notebook_code(text: str) -> str:
    cells = json.loads(text)["cells"]
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def is_history(rel: Path) -> bool:
    """Historical prose that names deleted modules on purpose: change records."""
    return rel.parts[:2] == ("docs", "changes") and rel.suffix in PROSE_SUFFIXES


def scan_repo(root: Path, *, min_python: int = MIN_PYTHON_FILES, sentinels=SENTINELS):
    """Map each offending relative path to its deleted-module references.

    Raises AssertionError when the scan did not reach the sentinel files or read
    fewer than ``min_python`` Python files, so an empty scan cannot pass.
    """
    valid = valid_root_names(root)
    scanned, python_files, offenders = set(), 0, {}
    for path, rel in walk_repo(root):
        if rel.suffix not in CODE_SUFFIXES | TEXT_SUFFIXES:
            continue
        scanned.add(rel.as_posix())
        text = path.read_text()
        if rel.suffix == ".py":
            python_files += 1
            hits = python_hits(text, rel, valid)
        elif rel.suffix == ".ipynb":
            hits = python_hits(_notebook_code(text), rel, valid)
        elif is_history(rel):
            continue
        else:
            hits = text_hits(text, valid)
            if rel.name == "pyproject.toml":
                hits += entry_point_hits(text, root)
        if hits:
            offenders[rel.as_posix()] = sorted(set(hits))
    missing = [s for s in sentinels if s not in scanned]
    assert not missing, f"scan did not reach {missing}"
    assert python_files >= min_python, f"scan read only {python_files} Python files"
    return offenders
