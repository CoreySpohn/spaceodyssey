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


def _refuse(field, value, dtype):
    return ValueError(
        f"field {field!r} is {value!r}, which {np.dtype(dtype).name} "
        "cannot hold exactly"
    )


def _items(value):
    """Nested lists of Python scalars from a tuple, list, array or scalar."""
    if isinstance(value, tuple | list):
        return [_items(v) for v in value]
    return value.tolist() if hasattr(value, "tolist") else value


def _element(v, want):
    """One Python number as `want` holds it, or None when that would change it."""
    if isinstance(v, list):
        out = [_element(x, want) for x in v]
        return None if any(x is None for x in out) else out
    if isinstance(v, bool | int):
        n = int(v)
    elif isinstance(v, float):
        if np.issubdtype(want, np.integer):
            if not v.is_integer():
                return None
            n = int(v)
        else:
            return v
    else:
        return None
    if np.issubdtype(want, np.integer):
        info = np.iinfo(want)
        return n if info.min <= n <= info.max else None
    try:
        return float(n) if int(float(n)) == n else None
    except OverflowError:
        return None


def _exact(field, value, dtype):
    """`value` as an array of `dtype`, refusing any conversion that changes it.

    Each element is checked as a Python number before NumPy sees it, so a large
    integer is never rounded through a float and an out-of-range one is never clamped.
    A JAX array or tracer is converted only by a safe cast.
    """
    want = np.dtype(dtype)
    if getattr(value, "weak_type", None) is not None:
        if value.dtype == want:
            return value  # already a JAX array of the dtype, or a tracer of one
        if jnp.can_cast(value.dtype, want, casting="safe"):
            return value.astype(want)
        raise _refuse(field, value, dtype)
    out = _element(_items(value), want)
    if out is None:
        raise _refuse(field, value, dtype)
    return jnp.asarray(np.array(out, dtype=want))


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
