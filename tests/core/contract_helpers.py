"""Typed-field checks over Study and every Abstract class defined in the core."""

import ast
import dataclasses
import importlib
import typing
from collections.abc import Callable
from pathlib import Path

from spaceodyssey import core

CORE = Path(core.__file__).parent


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


def assert_typed_fields(classes):
    """Assert every dataclass or plain class data field has a concrete annotation."""
    for cls in classes:
        annotations = typing.get_type_hints(cls)
        if dataclasses.is_dataclass(cls):
            fields = {field.name for field in dataclasses.fields(cls)}
        else:
            fields = {
                name
                for base in cls.__mro__[:-1]
                for name, value in vars(base).items()
                if not name.startswith("__")
                and not callable(value)
                and not isinstance(value, (property, classmethod, staticmethod))
            }
        for name in fields | annotations.keys():
            assert name in annotations, (cls, name, "missing annotation")
            assert not is_untyped(annotations[name]), (cls, name, "untyped field")


def assert_study_typed():
    """The complete typed composition contract, also used by the gate."""
    assert_typed_fields(abstract_bases() | {core.Study})
    assert {"key_hook", "handler_table", "n_extra_kinds"}.isdisjoint(
        field.name for field in dataclasses.fields(core.Study)
    )
