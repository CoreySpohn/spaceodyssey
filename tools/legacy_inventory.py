#!/usr/bin/env python3
"""Inventory the legacy spec/registry/wiring closure with a static AST scan.

The scan reads every module under ``src/spaceodyssey`` and every Python file
under ``tests``, records the ``spaceodyssey`` and ``tests`` modules each one
imports (top-level and function-level imports alike), and then computes the
reverse-dependency closure of two seed sets:

* ``named``: the three legacy modules ``spaceodyssey.spec``,
  ``spaceodyssey.registry`` and ``spaceodyssey.wiring``;
* ``expanded``: ``named`` plus ``spaceodyssey.seams``,
  ``spaceodyssey.contracts`` and ``spaceodyssey.aliases``.

Edges are explicit imports only. ``from spaceodyssey import name`` and
``package.name`` attribute access resolve a re-exported name to the module
that defines it, so the package ``__init__`` (which imports every public
module) does not make every importer depend on everything. The package
``__init__`` is reported separately because it is rewritten, not deleted.
String-based dynamic imports (``importlib.import_module``) are not followed.

A test module is affected when it imports a closure module directly or
through test helpers (``tests.*`` imports are followed transitively). Test
functions are counted as definitions, not parametrized cases.

Run ``python tools/legacy_inventory.py`` from the repository root; add
``--list`` to print every file.
"""

import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE = "spaceodyssey"
NAMED = ("spaceodyssey.spec", "spaceodyssey.registry", "spaceodyssey.wiring")
SUPPORT = ("spaceodyssey.seams", "spaceodyssey.contracts", "spaceodyssey.aliases")


@dataclass
class Module:
    """One scanned file: its dotted name, path, line count and import edges."""

    name: str
    path: Path
    lines: int
    is_package: bool
    tree: ast.Module
    imports: set[str] = field(default_factory=set)
    exports: dict[str, str] = field(default_factory=dict)


def module_name(path, root):
    """Return the dotted module name of ``path`` relative to ``root``."""
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def load_modules(src_root, tests_root):
    """Parse every Python file of the package and the test tree."""
    modules = {}
    for root in (src_root, tests_root):
        base = root.parent
        for path in sorted(root.rglob("*.py")):
            text = path.read_text()
            modules[module_name(path, base)] = Module(
                name=module_name(path, base),
                path=path,
                lines=len(text.splitlines()),
                is_package=path.name == "__init__.py",
                tree=ast.parse(text, filename=str(path)),
            )
    return modules


def _absolute(module, node):
    """Resolve the module of a ``from ... import`` node to a dotted name."""
    if not node.level:
        return node.module or ""
    parts = module.name.split(".")
    if not module.is_package:
        parts = parts[:-1]
    if node.level > 1:
        parts = parts[: len(parts) - (node.level - 1)]
    return ".".join([*parts, node.module] if node.module else parts)


def _collect_exports(modules):
    """Map each package's re-exported names to the module they come from."""
    for module in modules.values():
        if not module.is_package:
            continue
        for node in module.tree.body:
            if isinstance(node, ast.ImportFrom):
                source = _absolute(module, node)
                for alias in node.names:
                    module.exports[alias.asname or alias.name] = source


def resolve(modules, owner, name, seen=None):
    """Return the module that defines ``owner.name``, or ``owner`` itself."""
    candidate = f"{owner}.{name}"
    if candidate in modules:
        return candidate
    package = modules.get(owner)
    if package is None or name not in package.exports:
        return owner
    seen = seen or set()
    if (owner, name) in seen:
        return owner
    seen.add((owner, name))
    return resolve(modules, package.exports[name], name, seen)


def _tracked(name):
    return name == PACKAGE or name.startswith((PACKAGE + ".", "tests"))


def _collect_imports(modules):
    """Fill each module's set of imported scanned modules."""
    for module in modules.values():
        bound = {}
        for node in ast.walk(module.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not _tracked(alias.name):
                        continue
                    if alias.asname:
                        bound[alias.asname] = alias.name
                        module.imports.add(alias.name)
                    else:
                        head = alias.name.split(".")[0]
                        bound.setdefault(head, head)
                        if alias.name != head:
                            module.imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                source = _absolute(module, node)
                if not _tracked(source):
                    continue
                for alias in node.names:
                    target = resolve(modules, source, alias.name)
                    module.imports.add(target)
                    if f"{source}.{alias.name}" in modules:
                        bound[alias.asname or alias.name] = target
        for node in ast.walk(module.tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in bound
            ):
                module.imports.add(resolve(modules, bound[node.value.id], node.attr))
        module.imports &= modules.keys()
        module.imports.discard(module.name)


def reverse_closure(modules, seeds, exclude=()):
    """Return ``seeds`` plus every module that imports one of them, transitively."""
    found = set(seeds)
    changed = True
    while changed:
        changed = False
        for module in modules.values():
            if module.name in found or module.name in exclude:
                continue
            if module.imports & found:
                found.add(module.name)
                changed = True
    return found


def count_tests(module):
    """Count ``test_*`` functions and ``Test*`` class methods in a test module."""
    count = 0
    for node in module.tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            count += node.name.startswith("test")
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            count += sum(
                isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                and item.name.startswith("test")
                for item in node.body
            )
    return count


@dataclass
class Inventory:
    """Source and test footprint of one seed set."""

    seeds: tuple[str, ...]
    source: list[Module]
    init: Module | None
    direct_tests: list[Module]
    tests: list[Module]

    @property
    def source_lines(self):
        """Lines in the closure, excluding the package ``__init__``."""
        return sum(module.lines for module in self.source)


def inventory(modules, seeds, include_init):
    """Compute the inventory of ``seeds`` over the scanned modules."""
    source_names = {name for name in modules if name.startswith(PACKAGE)}
    test_names = set(modules) - source_names
    closure = reverse_closure(
        {name: modules[name] for name in source_names}, seeds, exclude={PACKAGE}
    )
    tainted = reverse_closure(modules, closure, exclude={PACKAGE})
    is_test = [
        modules[name]
        for name in sorted(test_names)
        if modules[name].path.name.startswith("test_")
    ]
    return Inventory(
        seeds=tuple(seeds),
        source=[modules[name] for name in sorted(closure)],
        init=modules[PACKAGE] if include_init else None,
        direct_tests=[module for module in is_test if module.imports & closure],
        tests=[module for module in is_test if module.name in tainted],
    )


def scan(repo=REPO):
    """Scan ``repo`` and return the named and expanded inventories."""
    modules = load_modules(repo / "src" / PACKAGE, repo / "tests")
    _collect_exports(modules)
    _collect_imports(modules)
    named = inventory(modules, NAMED, include_init=False)
    expanded = inventory(modules, NAMED + SUPPORT, include_init=True)
    return named, expanded


def report(label, result, repo, show_files):
    """Print counts (and optionally files) for one inventory."""
    init_lines = result.init.lines if result.init else 0
    print(f"[{label}] seeds: {', '.join(result.seeds)}")
    print(
        f"  source modules: {len(result.source)} files, {result.source_lines} lines"
        + (
            f"; with package __init__ ({init_lines} lines): "
            f"{len(result.source) + 1} files, {result.source_lines + init_lines} lines"
            if result.init
            else ""
        )
    )
    for kind, tests in (("direct", result.direct_tests), ("transitive", result.tests)):
        print(
            f"  test modules ({kind}): {len(tests)} files, "
            f"{sum(count_tests(module) for module in tests)} test functions, "
            f"{sum(module.lines for module in tests)} lines"
        )
    if show_files:
        for module in result.source:
            print(f"    src  {module.lines:5d}  {module.path.relative_to(repo)}")
        for module in result.tests:
            mark = " " if module in result.direct_tests else "*"
            path = module.path.relative_to(repo)
            print(f"    test{mark}{count_tests(module):5d}  {path}")


def main(argv=None):
    """Print the legacy inventory; return 0."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--list", action="store_true", help="print every file")
    args = parser.parse_args(argv)
    named, expanded = scan(args.repo)
    report("named", named, args.repo, args.list)
    report("expanded", expanded, args.repo, args.list)
    if args.list:
        print("  (* = affected only through test helpers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
