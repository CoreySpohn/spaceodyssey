#!/usr/bin/env python3
"""Restrict the transaction core to its declared numerical dependencies.

Run without arguments to scan the whole core, or use --core-dir for a negative
control. Relative imports must resolve inside spaceodyssey.core.
"""

import argparse
import ast
import importlib.util
import sys
import tokenize
from pathlib import Path

from check_domain_words import CORE, core_files

ALLOWED = sys.stdlib_module_names | {"jax", "jaxlib", "numpy", "equinox", "jaxtyping"}
LOADER_MODULES = {
    "importlib",
    "runpy",
    "pkgutil",
    "imp",
    "zipimport",
    "builtins",
    "pydoc",
    "pickle",
    "marshal",
}
LOADER_PREFIXES = ("unittest.mock",)
LOADER_CALLS = {
    "import_module",
    "__import__",
    "spec_from_file_location",
    "spec_from_loader",
    "module_from_spec",
    "exec_module",
    "load_module",
    "find_spec",
    "run_path",
    "run_module",
    "load_source",
    "SourceFileLoader",
    "SourcelessFileLoader",
    "ExtensionFileLoader",
    "get_loader",
    "find_loader",
}
BARE_CODE_CALLS = {"eval", "exec", "compile"}
NAME_LOOKUPS = {"getattr", "vars", "globals", "locals", "__getattribute__"}
LOADER_NAMES = LOADER_CALLS | BARE_CODE_CALLS | {"__builtins__", "builtins"}
EXPLANATION = (
    "Only the standard library, jax, jaxlib, numpy, equinox, jaxtyping and "
    "spaceodyssey.core are allowed. Physics libraries belong outside the core: "
    "orbix, jaxedith, photomancy, skyscapes, coronagraphoto, coronachrome, "
    "coronalyze, yippy, optixstuff, physicaloptix, exojax, planit_py. "
    "hwoutils is not allowed in the core in Slice 1 because it needs no constant. "
    "Dynamic loading (importlib, __import__, builtins, pydoc, pickle, marshal, "
    "unittest.mock, runpy, loader objects, eval, exec, compile, or a callee looked up with "
    "getattr, vars, globals or locals) is never allowed in the core, whatever its argument."
)


def allowed(module):
    """Compare package boundaries, never arbitrary string prefixes."""
    return (
        module.split(".")[0] in ALLOWED
        or module == "spaceodyssey.core"
        or module.startswith("spaceodyssey.core.")
    )


def import_names(node, package):
    """Resolve static imports, including relative parent-package traversal."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    module = node.module or ""
    if node.level:
        module = importlib.util.resolve_name("." * node.level + module, package)
    if not node.module or module == "spaceodyssey":
        return [f"{module}.{alias.name}" for alias in node.names]
    return [module]


def _lookup(node):
    """A name-based lookup: getattr, vars, globals or locals, or a `__dict__` read."""
    if isinstance(node, ast.Call):
        func = node.func
        return (isinstance(func, ast.Name) and func.id in NAME_LOOKUPS) or (
            isinstance(func, ast.Attribute) and func.attr in NAME_LOOKUPS
        )
    return isinstance(node, ast.Attribute) and node.attr == "__dict__"


def _loader_import(module):
    return module.split(".")[0] in LOADER_MODULES or any(
        module == prefix or module.startswith(prefix + ".")
        for prefix in LOADER_PREFIXES
    )


def dynamic_loading(node):
    """Name the dynamic loader a node uses, or None; the argument is never inspected.

    Any reference counts, not only a call: a bare name, an attribute, a string naming
    one (as in `getattr(builtins, "exec")`) or a call whose callee is looked up by name
    (`getattr(...)(...)`, `vars(...)[...](...)`). Calls through a transformed function
    (`jax.vmap(f)(x)`) or a table (`branches[0](x)`) are ordinary and pass.
    """
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        else:
            base = node.module or ""
            modules = [base, *(f"{base}.{alias.name}" for alias in node.names)]
        for module in modules:
            if _loader_import(module):
                return module
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in LOADER_NAMES:
                    return alias.name
    if isinstance(node, ast.Name) and node.id in LOADER_NAMES:
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in LOADER_NAMES:
        return node.attr
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in LOADER_NAMES
    ):
        return node.value
    if isinstance(node, ast.Call):
        func = node.func
        if _lookup(func) or (isinstance(func, ast.Subscript) and _lookup(func.value)):
            return "a callee looked up by name"
    return None


def main(argv):
    """Report all forbidden imports and fail closed on unreadable sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, default=CORE)
    args = parser.parse_args(argv)
    files = core_files(args.core_dir)
    failed = not files
    if not files:
        print(f"{args.core_dir}:0:no Python sources found")
    for path in files:
        try:
            with tokenize.open(path) as stream:
                tree = ast.parse(stream.read(), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as error:
            print(f"{path}:{getattr(error, 'lineno', 0)}:cannot scan: {error}")
            failed = True
            continue
        parts = path.relative_to(args.core_dir).parent.parts
        package = ".".join(("spaceodyssey", "core", *parts))
        for node in ast.walk(tree):
            loader = dynamic_loading(node)
            if loader:
                print(
                    f"{path}:{node.lineno}:dynamic loading via {loader} is not allowed"
                )
                failed = True
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            try:
                names = import_names(node, package)
            except ImportError:
                names = ["." * node.level + (node.module or "")]
            for name in names:
                if not allowed(name):
                    print(f"{path}:{node.lineno}:{name} is not allowed in the core")
                    failed = True
    print(f"Scanned {len(files)} Python files for core imports.")
    if failed:
        print(EXPLANATION)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
