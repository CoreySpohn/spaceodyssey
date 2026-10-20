"""Keep runtime dependency declarations aligned with source imports."""

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "spaceodyssey"
VIZ = PACKAGE / "viz"


def _dependency_names(requirements):
    return {
        re.split(r"[\s\[<>=!~;@]", requirement, maxsplit=1)[0].lower().replace("-", "_")
        for requirement in requirements
    }


def _imports(tree, *, module_level_only=False):
    names = set()
    pending = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
        if not module_level_only or not isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
        ):
            pending.extend(ast.iter_child_nodes(node))
    return names - sys.stdlib_module_names - {"__future__", "spaceodyssey"}


@pytest.fixture(scope="module")
def project():
    with (ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)["project"]


@pytest.fixture(scope="module")
def source_trees():
    return {
        path: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in sorted((ROOT / "src").rglob("*.py"))
    }


@pytest.fixture(scope="module")
def core_imports(source_trees):
    names = set()
    for path, tree in source_trees.items():
        if path.is_relative_to(PACKAGE) and not path.is_relative_to(VIZ):
            names.update(_imports(tree))
    return names


def test_core_imports_are_declared(project, core_imports):
    missing = core_imports - _dependency_names(project["dependencies"])
    assert not missing, f"Undeclared core imports: {sorted(missing)}"


def test_core_dependencies_are_imported(project, core_imports):
    # jaxlib supplies the backend for jax without a direct core import.
    allowlist = {"jaxlib"}
    unused = _dependency_names(project["dependencies"]) - core_imports - allowlist
    assert not unused, f"Unused core dependencies: {sorted(unused)}"


def test_viz_imports_are_declared(project, source_trees):
    declared = _dependency_names(project["dependencies"])
    declared.update(_dependency_names(project["optional-dependencies"]["viz"]))
    names = set()
    for path, tree in source_trees.items():
        if path.is_relative_to(VIZ):
            names.update(_imports(tree))
    missing = names - declared
    assert not missing, f"Undeclared viz imports: {sorted(missing)}"


def test_physics_dependencies_are_not_imported_at_module_level(project, source_trees):
    physics = set()
    for extra, requirements in project["optional-dependencies"].items():
        if extra.startswith("physics-"):
            physics.update(_dependency_names(requirements))
    eager_imports = {}
    for path, tree in source_trees.items():
        offending = _imports(tree, module_level_only=True) & physics
        if offending:
            eager_imports[str(path.relative_to(ROOT))] = sorted(offending)
    assert not eager_imports, f"Module-level physics imports: {eager_imports}"
