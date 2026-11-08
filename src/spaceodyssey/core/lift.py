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
import functools

import equinox as eqx
import jax.numpy as jnp
import numpy as np


def traced(dtype):
    """Field metadata marking a number, or a tuple of numbers, to lift to `dtype`."""
    return {"traced": dtype}


def _exact(field, value, dtype):
    """`value` as an array of `dtype`, refusing any conversion that changes it."""
    if getattr(value, "weak_type", None) is not None and value.dtype == dtype:
        return value  # already a JAX array of the dtype, or a tracer of one
    try:
        wanted = np.asarray(value)
        got = wanted.astype(dtype)
        same = wanted.dtype != object and np.array_equal(wanted, got, equal_nan=True)
    except (TypeError, ValueError, OverflowError):
        same = False
    if not same:
        raise ValueError(
            f"field {field!r} is {value!r}, which {np.dtype(dtype).name} "
            "cannot hold exactly"
        )
    return jnp.asarray(got)


def lifting_jit(fn):
    """`eqx.filter_jit(fn)` that lifts marked fields in every argument first."""
    jitted = eqx.filter_jit(fn)

    @functools.wraps(fn)
    def call(*args, **kwargs):
        args = tuple(lift_traced(a) for a in args)
        kwargs = {k: lift_traced(v) for k, v in kwargs.items()}
        return jitted(*args, **kwargs)

    return call


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
        new = lift_traced(value) if dtype is None else _exact(field.name, value, dtype)
        if new is not value:
            changes[field.name] = new
    if not changes:
        return node
    out = copy.copy(node)
    for name, value in changes.items():
        object.__setattr__(out, name, value)
    return out
