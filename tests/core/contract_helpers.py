"""Typed-field checks over Study and every Abstract class defined in the core."""

import ast
import dataclasses
import importlib
import typing
from collections.abc import Callable
from pathlib import Path

from spaceodyssey import core

CORE = Path(core.__file__).parent
FOREIGN = {"jax", "jaxlib", "numpy", "equinox", "jaxtyping", "builtins"}


def is_untyped(annotation):
    """Find escape annotations even within unions, containers and call signatures."""
    if isinstance(annotation, (list, tuple)):
        return any(is_untyped(arg) for arg in annotation)
    if annotation in (object, typing.Any, Callable, typing.Callable):
        return True
    return any(is_untyped(arg) for arg in typing.get_args(annotation))


def abstract_bases():
    """Discover definitions in all core modules, including non-exported bases."""
    bases = set()
    for path in sorted(CORE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name.startswith("Abstract")
        ]
        if names:
            parts = path.relative_to(CORE).with_suffix("").parts
            module = importlib.import_module(".".join(("spaceodyssey", "core", *parts)))
            bases.update(getattr(module, name) for name in names)
    assert bases, "no Abstract bases were checked"
    return bases


def _own_classes(annotation):
    """Every dataclass or package class named anywhere inside an annotation."""
    if isinstance(annotation, (list, tuple)):
        return {cls for arg in annotation for cls in _own_classes(arg)}
    found = {cls for arg in typing.get_args(annotation) for cls in _own_classes(arg)}
    if isinstance(annotation, type):
        root = annotation.__module__.split(".")[0]
        if root not in FOREIGN and (
            root == "spaceodyssey" or dataclasses.is_dataclass(annotation)
        ):
            found.add(annotation)
    return found


def _field_names(cls):
    if dataclasses.is_dataclass(cls):
        return {field.name for field in dataclasses.fields(cls)}
    return {
        name
        for base in cls.__mro__[:-1]
        for name, value in vars(base).items()
        if not name.startswith("__")
        and not callable(value)
        and not isinstance(value, (property, classmethod, staticmethod))
    }


def assert_typed_fields(classes):
    """Assert every field of each class, and of every class composed into it, is typed.

    The walk follows every field annotation (concrete classes, static fields,
    containers, unions and callable signatures) and visits each package class once.
    """
    pending, seen = list(classes), set()
    while pending:
        cls = pending.pop()
        if cls in seen:
            continue
        seen.add(cls)
        annotations = typing.get_type_hints(cls)
        for name in _field_names(cls) | annotations.keys():
            assert name in annotations, (cls, name, "missing annotation")
            assert not is_untyped(annotations[name]), (cls, name, "untyped field")
            pending.extend(_own_classes(annotations[name]) - seen)


def assert_study_typed():
    """The complete typed composition contract, also used by the gate."""
    assert_typed_fields(abstract_bases() | {core.Study})
    assert {"key_hook", "handler_table", "n_extra_kinds"}.isdisjoint(
        field.name for field in dataclasses.fields(core.Study)
    )
