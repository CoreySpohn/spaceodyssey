"""Rebuild a study from a `StudyDocument` by the inverse field walk.

Each module is rebuilt from its stored field values as they are, the way a pytree
is unflattened: neither `__init__` nor a field converter runs, so array dtypes,
weak types and Python types come back exactly as written. Validation does run:
each class's `__check_init__` (an `eqx.Module`) or `__post_init__` (a plain
dataclass), and an error it raises is reported at the module's pointer.
"""

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.codec.document import (
    DocumentError,
    KindTable,
    StudyDocument,
    check_schema_version,
    is_version_field,
    pointer,
)
from spaceodyssey.codec.dump import ARRAY_DTYPES
from spaceodyssey.core.capacities import require_x64
from spaceodyssey.core.study import Study

_FLOAT_TAGS = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}
_ARRAY_KEYS = frozenset({"$array", "shape", "data", "weak", "type"})
_WEAK_ZEROS = {"int64": 0, "float64": 0.0}


def load_study(document, kinds=None):
    """Rebuild the `Study` a `StudyDocument` (or its JSON text) describes.

    `kinds` is the mapping `dump_study` took. An unknown kind reference, a field
    missing or unknown, an array whose data does not match its declared shape or
    dtype, and a module whose validation fails each raise `DocumentError` carrying
    the JSON pointer of the offending value.
    """
    require_x64()
    if isinstance(document, str):
        document = StudyDocument.from_json(document)
    check_schema_version(document.schema_version)
    study = decode(document.study, ("study",), KindTable(kinds))
    if not isinstance(study, Study):
        raise DocumentError("the document's study is not a Study", "/study")
    return study


def decode(node, path, table):
    """The Python value of one JSON node; `path` locates it for error pointers."""
    if node is None or type(node) in (bool, int, float, str):
        return node
    if type(node) is list:
        return tuple(decode(v, (*path, i), table) for i, v in enumerate(node))
    if type(node) is not dict:
        raise DocumentError(f"no value for a {type(node).__name__}", pointer(path))
    tags = [k for k in node if k.startswith("$")]
    if len(tags) != 1:
        raise DocumentError("an object needs exactly one $ tag", pointer(path))
    tag = tags[0]
    if tag == "$kind":
        return _decode_kind(node, path, table)
    if tag == "$array":
        return _decode_array(node, path)
    _only(node, {tag} | ({"value"} if tag == "$enum" else set()), path)
    body = node[tag]
    if tag == "$float":
        return _decode_float(body, (*path, tag))
    if tag == "$list":
        if type(body) is not list:
            raise DocumentError("$list holds an array", pointer((*path, tag)))
        return [decode(v, (*path, tag, i), table) for i, v in enumerate(body)]
    if tag == "$dict":
        return _decode_dict(body, (*path, tag), table)
    if tag == "$enum":
        cls = table.resolve(_ref(body, (*path, tag)), (*path, tag))
        value = decode(node.get("value"), (*path, "value"), table)
        try:
            return cls(value)
        except (ValueError, TypeError) as err:
            raise DocumentError(str(err), pointer((*path, "value"))) from err
    raise DocumentError(f"unknown tag {tag}", pointer(path))


def _decode_dict(body, path, table):
    """A dict from its `[key, value]` pairs, in their stored order."""
    if type(body) is not list:
        raise DocumentError("$dict holds a list of [key, value] pairs", pointer(path))
    out = {}
    for i, pair in enumerate(body):
        if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str:
            raise DocumentError("a $dict entry is [key, value]", pointer((*path, i)))
        if pair[0] in out:
            raise DocumentError(f"key {pair[0]!r} repeats", pointer((*path, i, 0)))
        out[pair[0]] = decode(pair[1], (*path, i, 1), table)
    return out


def _only(node, allowed, path):
    """Refuse keys outside `allowed`."""
    extra = sorted(set(node) - allowed)
    if extra:
        raise DocumentError(f"unknown key {extra[0]!r}", pointer((*path, extra[0])))


def _ref(body, path):
    """A `[group, id, version]` reference as a tuple."""
    ok = (
        type(body) is list
        and len(body) == 3
        and type(body[0]) is str
        and type(body[1]) is str
        and type(body[2]) is int
    )
    if not ok:
        raise DocumentError("a kind reference is [group, id, version]", pointer(path))
    return tuple(body)


def _decode_float(body, path):
    """The float a non-finite tag names."""
    if type(body) is not str or body not in _FLOAT_TAGS:
        raise DocumentError("$float is 'nan', 'inf' or '-inf'", pointer(path))
    return _FLOAT_TAGS[body]


def _decode_kind(node, path, table):
    """Rebuild a module or dataclass from its fields, then validate it."""
    ref = _ref(node["$kind"], (*path, "$kind"))
    cls = table.resolve(ref, (*path, "$kind"))
    fields = [f.name for f in dataclasses.fields(cls)]
    stored = [n for n in fields if not is_version_field(cls, n)]
    _only(node, {"$kind", *stored}, path)
    missing = [n for n in stored if n not in node]
    if missing:
        raise DocumentError(f"field {missing[0]!r} is missing", pointer(path))
    obj = object.__new__(cls)
    for name in fields:
        if is_version_field(cls, name):
            value = ref[2]
        else:
            value = decode(node[name], (*path, name), table)
        object.__setattr__(obj, name, value)
    try:
        _validate(obj, cls)
    except DocumentError:
        raise
    except (ValueError, TypeError) as err:
        raise DocumentError(f"{cls.__qualname__}: {err}", pointer(path)) from err
    return obj


def _validate(obj, cls):
    """Run the checks construction would have run."""
    if isinstance(obj, eqx.Module):
        for kls in cls.__mro__:
            check = kls.__dict__.get("__check_init__")
            if check is not None:
                check(obj)
    elif hasattr(cls, "__post_init__"):
        obj.__post_init__()


def _decode_array(node, path):
    """Rebuild an array with its dtype, shape, weak type and container."""
    _only(node, _ARRAY_KEYS, path)
    name = node["$array"]
    if type(name) is not str or name not in ARRAY_DTYPES:
        raise DocumentError(f"unknown dtype {name!r}", pointer((*path, "$array")))
    dtype = jnp.dtype(name)
    shape = node.get("shape")
    if type(shape) is not list or not all(type(n) is int and n >= 0 for n in shape):
        raise DocumentError("shape is a list of sizes", pointer((*path, "shape")))
    if "data" not in node:
        raise DocumentError("the array has no data", pointer(path))
    data = _elements(node["data"], dtype, (*path, "data"))
    try:
        arr = np.array(data, dtype=dtype)
    except (ValueError, OverflowError) as err:
        raise DocumentError(f"bad array data: {err}", pointer((*path, "data"))) from err
    if (
        arr.size == 0
        and math.prod(shape) == 0
        and arr.shape == tuple(shape[: arr.ndim])
    ):
        arr = arr.reshape(shape)
    if arr.shape != tuple(shape):
        raise DocumentError(
            f"data has shape {list(arr.shape)} but shape says {shape}",
            pointer((*path, "data")),
        )
    weak = node.get("weak", False)
    container = node.get("type")
    if type(weak) is not bool or (weak and (container or name not in _WEAK_ZEROS)):
        raise DocumentError(
            "only an int64 or float64 JAX array can be weak", pointer((*path, "weak"))
        )
    if container == "numpy":
        return arr
    if container == "numpy_scalar":
        if arr.shape != ():
            raise DocumentError(
                "a NumPy scalar has shape []", pointer((*path, "shape"))
            )
        return arr[()]
    if container is not None:
        raise DocumentError(
            f"unknown array type {container!r}", pointer((*path, "type"))
        )
    if weak:
        zero = jnp.asarray(_WEAK_ZEROS[name])
        return jax.lax.full_like(zero, 0, shape=arr.shape).at[...].set(arr)
    return jnp.asarray(arr)


def _elements(data, dtype, path):
    """Check element types against the dtype and resolve float tags."""
    if type(data) is list:
        return [_elements(x, dtype, (*path, i)) for i, x in enumerate(data)]
    if dtype == np.bool_:
        ok = type(data) is bool
    elif np.issubdtype(dtype, np.integer):
        ok = type(data) is int
    else:
        if type(data) is dict and set(data) == {"$float"}:
            return _decode_float(data["$float"], (*path, "$float"))
        ok = type(data) in (int, float)
    if not ok:
        raise DocumentError(f"{data!r} is not a {dtype.name} element", pointer(path))
    return data
