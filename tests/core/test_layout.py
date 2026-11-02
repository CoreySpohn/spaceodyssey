"""The transaction core has no physics-library imports."""

import ast
from pathlib import Path

from spaceodyssey import core


def test_no_physics_imports():
    here = Path(core.__file__).parent
    forbidden = {
        "orbix",
        "jaxedith",
        "photomancy",
        "coronalyze",
        "skyscapes",
        "yippy",
        "optixstuff",
        "coronagraphoto",
    }
    files = sorted(here.rglob("*.py"))
    assert files
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                names = []
            assert not {name.split(".")[0] for name in names} & forbidden, path.name
