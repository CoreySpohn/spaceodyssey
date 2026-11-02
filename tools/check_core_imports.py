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
EXPLANATION = (
    "Only the standard library, jax, jaxlib, numpy, equinox, jaxtyping and "
    "spaceodyssey.core are allowed. Physics libraries belong outside the core: "
    "orbix, jaxedith, photomancy, skyscapes, coronagraphoto, coronachrome, "
    "coronalyze, yippy, optixstuff, physicaloptix, exojax, planit_py. "
    "hwoutils is not allowed in the core in Slice 1 because it needs no constant."
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
