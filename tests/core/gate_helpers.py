"""Static study-dispatch scan used by the close-out gate."""

import ast
import re
from pathlib import Path

SURVEY_NAME = re.compile(r"\bsurvey[ _-]*\d+\b", re.IGNORECASE)
KIND_NAMES = {
    "Study",
    "AbstractHostService",
    "AbstractChannelModel",
    "AbstractDesign",
    "AbstractObjective",
    "AbstractObserver",
    "AbstractPolicy",
    "AbstractProcess",
    "AbstractWorldSource",
}


def _symbol(node):
    return node.id if isinstance(node, ast.Name) else getattr(node, "attr", None)


def _study_kind_names():
    """Find concrete kind descendants without importing domain implementations."""
    source = Path(__file__).parents[2] / "src/spaceodyssey"
    classes = [
        node
        for directory in (source / "core", source / "studies")
        for path in directory.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef)
    ]
    names = set(KIND_NAMES)
    while True:
        extended = names | {
            node.name
            for node in classes
            if any(_symbol(base) in names for base in node.bases)
        }
        if extended == names:
            return names
        names = extended


def _study_reference(node, study_names):
    return any(
        (isinstance(part, ast.Name) and part.id in study_names)
        or (isinstance(part, ast.Attribute) and part.attr == "study")
        for part in ast.walk(node)
    )


def _study_aliases(tree):
    """Follow simple aliases of the study and its components conservatively."""
    names = {"study"}
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
        and node.value is not None
    ]
    while True:
        extended = set(names)
        for node in assignments:
            if _study_reference(node.value, names):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                extended.update(
                    part.id
                    for target in targets
                    for part in ast.walk(target)
                    if isinstance(part, ast.Name)
                )
        if extended == names:
            return names
        names = extended


def study_branches(root):
    """Return (path, line, rule) for forbidden study dispatch, failing closed."""
    files = sorted(root.rglob("*.py"))
    assert files, f"no Python sources under {root}"
    hits = []
    kind_names = _study_kind_names()
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        study_names = _study_aliases(tree)
        aliases = kind_names | {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if alias.name in kind_names
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                operands = tuple(ast.walk(node))
                if any(
                    isinstance(part, ast.Constant)
                    and isinstance(part.value, str)
                    and SURVEY_NAME.search(part.value)
                    for part in operands
                ):
                    hits.append((path, node.lineno, "survey-name"))
                if any(
                    isinstance(part, ast.Attribute)
                    and part.attr == "name"
                    and _study_reference(part.value, study_names)
                    for part in operands
                ):
                    hits.append((path, node.lineno, "study.name"))
            if (
                path.name in {"handlers.py", "transaction.py"}
                and isinstance(node, ast.Call)
                and _symbol(node.func) == "isinstance"
                and len(node.args) == 2
            ):
                names = {
                    part.id if isinstance(part, ast.Name) else part.attr
                    for part in ast.walk(node.args[1])
                    if isinstance(part, (ast.Name, ast.Attribute))
                }
                if _study_reference(node.args[0], study_names) or names & aliases:
                    hits.append((path, node.lineno, "study-kind"))
    return hits
