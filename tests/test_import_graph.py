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
# Extended each layer as modules arrive.
ORDER = [
    "types", "claim", "roles", "design", "belief", "information", "value",  # rank 0
    "price", "policy",  # rank 1
    "world", "log", "toy",  # rank 2
    "score", "study",  # rank 3
]  # fmt: skip
RANK = {name: i for i, name in enumerate(ORDER)}
RANK_0 = 7  # the first seven modules of ORDER are rank 0
FORBIDDEN = {"hwo", "hwopaths", "viz"}
# Layer 1 may import JAX, Equinox, jaxtyping and NumPy only (plus the standard library).
EXTERNAL = {"jax", "jaxlib", "equinox", "jaxtyping", "numpy"}


def _imports(source: str):
    """Yield (target, node) for every import in ``source``, wherever it sits.

    ``ast.walk`` also reaches imports under ``if TYPE_CHECKING`` and inside functions,
    so annotation-only and deferred imports count. A target is a dotted name, or a
    package-relative one resolved to ``spaceodyssey.<name>``.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: resolve against this module's package
                base = [PACKAGE] + ([node.module] if node.module else [])
            else:
                base = (node.module or "").split(".")
            for alias in node.names:
                # `from spaceodyssey import x` imports the submodule x.
                yield ".".join(base if len(base) > 1 else [*base, alias.name]), node


def violations(sources: dict[str, str]) -> list[str]:
    """Return one message per import that breaks the layer rule."""
    found = []
    for module, source in sources.items():
        for target, node in _imports(source):
            top, _, rest = target.partition(".")
            where = f"{module}:{node.lineno} imports {target}"
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
    """Every core module imports only earlier modules and allowed packages.

    Fails if a module imports a later module, `hwo`, `hwopaths`, `viz`, or a package
    outside JAX, Equinox, jaxtyping and NumPy, including under `if TYPE_CHECKING` or
    inside a function.
    """
    sources = {
        path.stem: path.read_text() for path in SRC.glob("*.py") if path.stem in RANK
    }
    assert sources, "no core modules found"
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
    ],
)
def test_the_checker_catches_each_kind_of_bad_import(module, source, message):
    """The checker itself flags upward, deferred, relative and forbidden imports.

    Each case is a deliberately bad fixture module.

    Fails if the checker misses an import form, so that the test above would pass over a
    cycle.
    """
    found = violations({module: source})
    assert len(found) == 1
    assert message in found[0]
