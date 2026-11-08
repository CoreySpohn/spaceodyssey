"""Lift marked Python numbers to arrays where a study enters a compiled call.

A policy, process or observer holds a number as a plain Python value, so it is built,
written to a document, copied and pickled exactly as a Python number is. Handed to
`eqx.filter_jit` unchanged, that value is static, and a study that differs only in it
compiles a program of its own. A field marked `traced(dtype)` is replaced, for the
call only, by an array of that dtype, so such studies share one program. The study the
caller holds is never changed.
"""

import copy
import dataclasses

import jax.numpy as jnp


def traced(dtype):
    """Field metadata marking a number, or a tuple of numbers, to lift to `dtype`."""
    return {"traced": dtype}


def lift_traced(node):
    """`node` with every field marked `traced` replaced by an array of its dtype.

    Walks dataclasses (Equinox modules included) and tuples. Nothing is rebuilt that
    holds no marked field, and the construction hooks of a module are not run again.
    """
    if isinstance(node, tuple):
        items = tuple(lift_traced(x) for x in node)
        same = all(a is b for a, b in zip(items, node, strict=True))
        return node if same else items
    if not dataclasses.is_dataclass(node):
        return node
    changes = {}
    for field in dataclasses.fields(node):
        value = getattr(node, field.name)
        dtype = field.metadata.get("traced")
        new = lift_traced(value) if dtype is None else jnp.asarray(value, dtype)
        if new is not value:
            changes[field.name] = new
    if not changes:
        return node
    out = copy.copy(node)
    for name, value in changes.items():
        object.__setattr__(out, name, value)
    return out
