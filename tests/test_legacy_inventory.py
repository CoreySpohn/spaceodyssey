"""Tests for the legacy deletion inventory scan (tools/legacy_inventory.py)."""

import importlib.util
from pathlib import Path

TOOL = Path(__file__).parents[1] / "tools" / "legacy_inventory.py"
_spec = importlib.util.spec_from_file_location("legacy_inventory", TOOL)
legacy_inventory = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy_inventory)

FILES = {
    "src/spaceodyssey/__init__.py": (
        "from spaceodyssey.spec import Spec\nfrom spaceodyssey.kept import Kept\n"
    ),
    "src/spaceodyssey/spec.py": "from .wiring import check\nclass Spec: ...\n",
    "src/spaceodyssey/registry.py": "REGISTRY = {}\n",
    "src/spaceodyssey/wiring.py": "def check(): ...\n",
    "src/spaceodyssey/seams.py": "SEAMS = ()\n",
    "src/spaceodyssey/contracts.py": "from spaceodyssey.seams import SEAMS\n",
    "src/spaceodyssey/aliases.py": "ALIASES = {}\n",
    "src/spaceodyssey/kept.py": "class Kept: ...\n",
    "src/spaceodyssey/user.py": (
        "def lazy():\n    from spaceodyssey.registry import REGISTRY\n"
    ),
    "src/spaceodyssey/helper.py": "from spaceodyssey.contracts import SEAMS\n",
    "tests/__init__.py": "",
    "tests/helpers.py": "import spaceodyssey\nVALUE = spaceodyssey.Spec\n",
    "tests/test_kept.py": (
        "from spaceodyssey import Kept\ndef test_a(): ...\ndef test_b(): ...\n"
    ),
    "tests/test_direct.py": (
        "from spaceodyssey import Spec\n"
        "def test_a(): ...\n"
        "class TestGroup:\n    def test_b(self): ...\n    def helper(self): ...\n"
    ),
    "tests/test_via_helper.py": "from tests.helpers import VALUE\ndef test_a(): ...\n",
    "tests/test_support.py": (
        "from spaceodyssey.helper import SEAMS\ndef test_a(): ...\n"
    ),
}


def test_closure_follows_reexports_lazy_imports_and_helpers(tmp_path):
    """Fails if the scan misses a re-exported name, a function-level import or a
    test helper, or if the package __init__ drags every importer into the closure.
    """
    for name, text in FILES.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    named, expanded = legacy_inventory.scan(tmp_path)

    def stems(modules):
        return sorted(module.path.stem for module in modules)

    assert stems(named.source) == ["registry", "spec", "user", "wiring"]
    assert stems(named.direct_tests) == ["test_direct"]
    assert stems(named.tests) == ["test_direct", "test_via_helper"]
    assert sum(map(legacy_inventory.count_tests, named.tests)) == 3
    assert "helper" in stems(expanded.source)
    assert stems(expanded.tests) == ["test_direct", "test_support", "test_via_helper"]
    assert expanded.init is not None
