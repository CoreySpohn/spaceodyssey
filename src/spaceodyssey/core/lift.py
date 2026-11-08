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


def traced(dtype, *, sequence=False):
    """Field metadata marking a number, or with `sequence` a tuple of them, to lift."""
    return {"traced": dtype, "sequence": sequence}


def _number(v, want):
    """One Python number as `want` holds it exactly, or None when it cannot be told."""
    if hasattr(v, "dtype") or not isinstance(v, bool | int | float):
        return None  # arrays, NumPy scalars, Fraction, Decimal and the like
    if isinstance(v, float) and np.issubdtype(want, np.integer):
        return None  # a float stays what it is, as it was before lifting
    if not isinstance(v, float) and not (
        np.iinfo(np.int64).min <= int(v) <= np.iinfo(np.int64).max
    ):
        return None  # a Python int past int64 is not a number the program ever held
    if np.issubdtype(want, np.integer) or isinstance(v, float):
        return int(v) if np.issubdtype(want, np.integer) else v
    return float(v) if int(float(v)) == int(v) else None


def _lifted(value, dtype, sequence=False):
    """`value` as an array of `dtype` when that is provably exact, else `value` itself.

    Lifted: a Python int, bool or float for a number field, or a flat tuple or list of
    them for a `sequence` field, each number checked on its own. Everything else is
    handed on untouched, exactly as it was before lifting existed: arrays and tracers
    (which are already arrays), other dtypes, NumPy scalars, object arrays, Fractions,
    Decimals, nested or ragged sequences. Such a value only costs a compile of its own.
    """
    want = np.dtype(dtype)
    if sequence:
        if not isinstance(value, tuple | list):
            return value
        out = [_number(v, want) for v in value]
        if any(v is None for v in out):
            return value
    else:
        out = _number(value, want)
        if out is None:
            return value
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
        new = (
            lift_traced(value)
            if dtype is None
            else _lifted(value, dtype, field.metadata.get("sequence", False))
        )
        if new is not value:
            changes[field.name] = new
    if not changes:
        return node
    out = copy.copy(node)
    for name, value in changes.items():
        object.__setattr__(out, name, value)
    return out
