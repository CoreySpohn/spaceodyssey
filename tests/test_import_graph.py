"""The layer import rule: no module imports upward, and the core imports nothing it may
not.
"""

import ast
import sys
from pathlib import Path

import pytest

PACKAGE = "spaceodyssey"
SRC = Path(__file__).parent.parent / "src" / PACKAGE

# Rank order of the core; a module may import only modules earlier in this order.
# Extended each layer as modules arrive; a source file whose module is missing here
# fails the test. `_version` is the generated version leaf. `_standin` holds local
# stand-ins for the library functions rank 0 may import (photomancy's diagnostics): each
# mirrors an external function, so it comes first and imports nothing in the package,
# and it is deleted when its replacement lands. `_numerics` holds the guarded numerics
# every module shares and imports nothing in the package either. The package
# `__init__` comes last: it may re-export any core module.
ORDER = [
    "_version", "_standin", "_numerics",
    "types", "claim", "roles", "design", "belief", "information", "value",  # rank 0
    "price", "policy",  # rank 1
    "world", "log", "toy",  # rank 2
    "study", "score",  # rank 3
    "__init__",
]  # fmt: skip
RANK = {name: i for i, name in enumerate(ORDER)}
RANK_0 = ORDER.index("price")  # everything before `price` is rank 0
FORBIDDEN = {"hwo", "hwopaths", "viz"}
# Layer 1 may import JAX, Equinox, jaxtyping and NumPy only (plus the standard library).
EXTERNAL = {"jax", "jaxlib", "equinox", "jaxtyping", "numpy"}


def _package(path: str) -> list[str]:
    """The dotted package of a source path, as a list: ``spaceodyssey`` and any
    subpackage directories (an ``__init__.py`` is in the package it defines).
    """
    return [PACKAGE, *path.split("/")[:-1]]


def _imports(source: str, path: str):
    """Yield (target, node) for every import in ``source``, wherever it sits.

    ``ast.walk`` also reaches imports under ``if TYPE_CHECKING`` and inside functions,
    so annotation-only and deferred imports count. A target is a dotted name; a relative
    one is resolved against the package of ``path``, one level up per extra dot.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = _package(path)
                up = len(package) - (node.level - 1)
                base = package[: max(up, 0)] + (
                    node.module.split(".") if node.module else []
                )
            else:
                base = (node.module or "").split(".")
            for alias in node.names:
                # `from spaceodyssey import x` imports the submodule x.
                yield ".".join(base if len(base) > 1 else [*base, alias.name]), node


def _module(path: str) -> str:
    """The rank-table name of a source path: its top-level module or subpackage."""
    return path.split("/")[0].removesuffix(".py")


def violations(sources: dict[str, str]) -> list[str]:
    """Return one message per import that breaks the layer rule.

    ``sources`` maps a path relative to the package (or a bare module name) to the
    source; a file in a subpackage ranks as the subpackage.
    """
    found = []
    for path, source in sources.items():
        module = _module(path)
        if module not in RANK:
            found.append(f"{path} is an unranked module (add it to ORDER)")
            continue
        for target, node in _imports(source, path):
            top, _, rest = target.partition(".")
            where = f"{path}:{node.lineno} imports {target}"
            if top == PACKAGE:
                name = rest.split(".")[0]
                if name in FORBIDDEN:
                    found.append(f"{where} (the core never imports it)")
                elif name not in RANK:
                    found.append(f"{where} (unknown module)")
                elif RANK[name] >= RANK[module]:
                    found.append(f"{where} (not earlier in the order)")
                elif module == "world" and RANK[name] >= RANK_0:
                    found.append(f"{where} (world imports rank 0 only)")
                elif module == "policy" and name == "world":
                    found.append(f"{where} (policy never imports world)")
            elif top in FORBIDDEN:
                found.append(f"{where} (the core never imports it)")
            elif top not in EXTERNAL and top not in sys.stdlib_module_names:
                found.append(f"{where} (not an allowed dependency of the core)")
    return found


def test_the_core_imports_nothing_upward():
    """Every source file of the package imports only earlier modules and allowed
    packages.

    Every `.py` file under the package is scanned, the package and subpackage
    `__init__.py` files included.

    Fails if a module imports a later module, `world` imports above rank 0, a module
    imports `hwo`, `hwopaths`, `viz`, or a package outside JAX, Equinox, jaxtyping and
    NumPy, including under `if TYPE_CHECKING` or inside a function; if a stand-in
    imports from the package; if a source file is missing from the rank table; or if
    the scan misses the package `__init__`, `roles`, `world`, `toy` or the stand-ins.
    """
    sources = {
        path.relative_to(SRC).as_posix(): path.read_text() for path in SRC.rglob("*.py")
    }
    assert {
        "__init__.py",
        "_numerics.py",
        "roles.py",
        "world.py",
        "toy.py",
        "_standin/__init__.py",
        "_standin/diagnostics.py",
    } <= set(sources)
    assert violations(sources) == []


@pytest.mark.parametrize(
    ("module", "source", "message"),
    [
        ("claim", "from spaceodyssey.belief import Belief", "not earlier in the order"),
        ("claim", "import spaceodyssey.belief", "not earlier in the order"),
        ("claim", "from . import belief", "not earlier in the order"),
        ("claim", "from .belief import Belief", "not earlier in the order"),
        (
            "claim",
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from spaceodyssey.belief import Belief",
            "not earlier in the order",
        ),
        (
            "claim",
            "def f():\n    from spaceodyssey import belief",
            "not earlier in the order",
        ),
        ("belief", "from spaceodyssey.hwo import preset", "never imports it"),
        ("belief", "import hwopaths", "never imports it"),
        ("belief", "import scipy", "not an allowed dependency"),
        (
            "policy",
            "from spaceodyssey.world import SealedWorld",
            "not earlier in the order",
        ),
        ("score", "from spaceodyssey import nowhere", "unknown module"),
        (
            "_standin/diagnostics.py",
            "from spaceodyssey.types import Atoms",
            "not earlier in the order",
        ),
        ("_standin/diagnostics.py", "from ..types import Atoms", "not earlier"),
        ("_standin/__init__.py", "from .. import belief", "not earlier"),
        ("_numerics.py", "from spaceodyssey.types import Atoms", "not earlier"),
        ("__init__.py", "from spaceodyssey.hwo import preset", "never imports it"),
        ("cells.py", "import jax", "unranked module"),
        ("viz/plots.py", "import jax", "unranked module"),
        ("world.py", "from spaceodyssey.toy import GaussianGasModel", "not earlier"),
        ("world.py", "from .toy import GaussianGasModel", "not earlier"),
        ("world.py", "from spaceodyssey.policy import X", "world imports rank 0 only"),
        ("world.py", "from spaceodyssey import price", "world imports rank 0 only"),
        ("world.py", "from . import price", "world imports rank 0 only"),
        ("toy.py", "from spaceodyssey.score import X", "not earlier"),
    ],
)
def test_the_checker_catches_each_kind_of_bad_import(module, source, message):
    """The checker itself flags upward, deferred, relative and forbidden imports, and
    modules missing from the rank table.

    Each case is a deliberately bad fixture module. A relative import in a subpackage
    resolves against that subpackage, one level up per extra dot.

    Fails if the checker misses an import form or an unranked module, so that the test
    above would pass over a cycle.
    """
    found = violations({module: source})
    assert len(found) == 1
    assert message in found[0]
