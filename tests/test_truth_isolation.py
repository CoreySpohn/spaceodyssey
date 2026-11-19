"""Truth isolation, read off the source: who may name, call or flatten the sealed truth.

Rules over every source file of the package:

- `_truth` is named only in `world.py`: in the class body's field line and in
  `__init__`, `observe`, `reveal` and `unseal` (and `draw_worlds`, which may build one);
- `unseal` is called only in `score.py`; `observe` and `reveal` only in `log.py` and in
  `world.py` itself;
- no module outside `world.py` flattens a world: no tree or partition function
  (`jax.tree.leaves`, `jax.tree_util.tree_flatten`, `eqx.partition`, `eqx.filter` and
  the like) is applied to a name annotated `SealedWorld` or named for a world;
- no rank-1 module (`policy`, `price`) imports `world`.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src" / "spaceodyssey"
TRUTH_READERS = {"__init__", "observe", "reveal", "unseal", "draw_worlds"}
CALLERS = {"unseal": {"score.py"}, "observe": {"log.py", "world.py"}}
CALLERS["reveal"] = CALLERS["observe"]
FLATTENERS = {
    "tree_leaves", "tree_flatten", "tree_flatten_with_path", "tree_leaves_with_path",
    "tree_map", "tree_map_with_path", "tree_structure", "tree_unflatten", "leaves",
    "flatten", "flatten_with_path", "leaves_with_path", "map", "map_with_path",
    "structure", "partition", "filter", "combine", "ravel_pytree", "tree_at",
}  # fmt: skip
RANK_1 = {"policy.py", "price.py"}


def _world_names(function: ast.AST) -> set[str]:
    """Names in a function that hold a world: parameters annotated `SealedWorld`, and
    any name that says `world`.
    """
    names = set()
    for node in ast.walk(function):
        if isinstance(node, ast.arg) and node.annotation is not None:
            if "SealedWorld" in ast.unparse(node.annotation):
                names.add(node.arg)
        if isinstance(node, ast.Name) and "world" in node.id.lower():
            names.add(node.id)
    return names


def _holds_world(node: ast.AST, names: set[str]) -> bool:
    """Whether an argument is a world: a world name or an attribute named for one."""
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Attribute):
        return "world" in node.attr.lower() or _holds_world(node.value, names)
    return False


def _callee(call: ast.Call) -> str:
    """The last name of a call's function: `jax.tree.leaves(x)` gives `leaves`."""
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def violations(sources: dict[str, str]) -> list[str]:
    """One message per breach of the truth-isolation rules."""
    found = []
    for path, source in sources.items():
        tree = ast.parse(source)
        # Every node's enclosing function, to place each `_truth`.
        owner = {}
        for function in ast.walk(tree):
            if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
                for node in ast.walk(function):
                    owner.setdefault(node, function.name)
        for node in ast.walk(tree):
            where = f"{path}:{getattr(node, 'lineno', 0)}"
            name = node.attr if isinstance(node, ast.Attribute) else None
            name = node.id if isinstance(node, ast.Name) else name
            if name == "_truth":
                in_function = owner.get(node)
                if path != "world.py" or (
                    in_function is not None and in_function not in TRUTH_READERS
                ):
                    found.append(f"{where} names _truth")
            if isinstance(node, ast.Call):
                callee = _callee(node)
                if callee in CALLERS and isinstance(node.func, ast.Attribute):
                    if path not in CALLERS[callee]:
                        found.append(f"{where} calls {callee}")
            if isinstance(node, ast.Import | ast.ImportFrom) and path in RANK_1:
                text = ast.unparse(node)
                if "world" in text.split("import")[-1] or ".world" in text:
                    found.append(f"{where} imports world")
        if path != "world.py":
            for function in ast.walk(tree):
                if not isinstance(function, ast.FunctionDef | ast.Module):
                    continue
                names = _world_names(function)
                for call in ast.walk(function):
                    if (
                        isinstance(call, ast.Call)
                        and _callee(call) in FLATTENERS
                        and any(
                            _holds_world(arg, names)
                            for arg in [*call.args, *(k.value for k in call.keywords)]
                        )
                    ):
                        found.append(f"{path}:{call.lineno} flattens a world")
    return sorted(set(found))


def test_only_the_world_holds_the_truth_and_only_the_scorer_unseals_it():
    """Every source file obeys the truth-isolation rules.

    Fails if a module outside `world.py` names `_truth`, a function other than the
    world's own reads it, anything but `score` unseals a world, anything but `log`
    draws rows from one, a module flattens a world to reach its leaves, or a rank-1
    module imports `world`.
    """
    sources = {
        path.relative_to(SRC).as_posix(): path.read_text() for path in SRC.rglob("*.py")
    }
    assert {"world.py", "score.py", "log.py", "policy.py"} <= set(sources)
    assert violations(sources) == []


@pytest.mark.parametrize(
    ("path", "source", "message"),
    [
        ("policy.py", "def f(world):\n    return world._truth", "names _truth"),
        (
            "world.py",
            "class SealedWorld:\n    def peek(self):\n        return self._truth",
            "names _truth",
        ),
        ("log.py", "def f(world):\n    return world.unseal()", "calls unseal"),
        ("score.py", "def f(world, key, m, a):\n    return world.observe(key, m, a)",
         "calls observe"),
        ("study.py", "def f(world, k, g, m, d):\n    return world.reveal(k, g, m, d)",
         "calls reveal"),
        ("policy.py", "from spaceodyssey.world import SealedWorld", "imports world"),
        ("price.py", "from spaceodyssey import world", "imports world"),
        (
            "log.py",
            "import jax\ndef f(w: SealedWorld):\n    return jax.tree.leaves(w)",
            "flattens a world",
        ),
        (
            "score.py",
            "import jax\ndef f(world):\n    return jax.tree_util.tree_flatten(world)",
            "flattens a world",
        ),
        (
            "policy.py",
            "import equinox as eqx\ndef f(state):\n"
            "    return eqx.partition(state.world, eqx.is_array)",
            "flattens a world",
        ),
        (
            "study.py",
            "import equinox as eqx\ndef f(world):\n"
            "    return eqx.filter(world, eqx.is_array)",
            "flattens a world",
        ),
    ],
)  # fmt: skip
def test_the_checker_catches_each_breach(path, source, message):
    """The checker flags a deliberately bad fixture module for each rule.

    Fails if the checker misses a way to name, unseal, observe, import or flatten the
    truth, so that the test above would pass over a leak.
    """
    found = violations({path: source})
    assert len(found) == 1, found
    assert message in found[0]
