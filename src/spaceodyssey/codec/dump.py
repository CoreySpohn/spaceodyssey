"""Encode a study as a `StudyDocument` by a generic walk over its fields."""

import dataclasses
import enum
import math

import jax
import numpy as np

from spaceodyssey.codec.document import DocumentError, StudyDocument, pointer
from spaceodyssey.codec.kinds import as_resolver, is_version_field
from spaceodyssey.core.capacities import require_x64
from spaceodyssey.core.study import Study

# Element types the document stores; complex and object arrays are refused.
ARRAY_DTYPES = frozenset(
    {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    }
)


def dump_study(study, kinds=None):
    """Encode `study` as a `StudyDocument`.

    `kinds` is a `KindResolver`, or a mapping of `(group, id, version)` to each
    class the study holds beyond the core's own types (`CORE_KINDS`). Every field
    of every module is written, static and dynamic alike, with exact Python types
    and array dtypes, so `load_study` rebuilds an equal pytree. A value of a type
    the document has no form for (a function, a set, a complex array) raises
    `DocumentError` naming its pointer.
    """
    require_x64()
    if not isinstance(study, Study):
        raise TypeError(f"dump_study needs a Study, not {type(study).__name__}")
    return StudyDocument(study=encode(study, ("study",), as_resolver(kinds)))


def encode(value, path, table):
    """The JSON form of one value; `path` locates it for error pointers."""
    if isinstance(value, enum.Enum):
        ref = table.ref_of(value, path)
        return {"$enum": list(ref), "value": encode(value.value, path, table)}
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return encode_float(value)
    if isinstance(value, jax.Array):
        return _encode_array(np.asarray(value), path, weak=value.weak_type)
    if isinstance(value, np.ndarray):
        return _encode_array(value, path, container="numpy")
    if isinstance(value, np.generic):
        return _encode_array(np.asarray(value), path, container="numpy_scalar")
    if type(value) is tuple:
        return [encode(v, (*path, i), table) for i, v in enumerate(value)]
    if type(value) is list:
        items = [encode(v, (*path, "$list", i), table) for i, v in enumerate(value)]
        return {"$list": items}
    if type(value) is dict:
        if not all(type(k) is str for k in value):
            raise DocumentError("a dict needs string keys", pointer(path))
        return {
            "$dict": [
                [k, encode(v, (*path, "$dict", i, 1), table)]
                for i, (k, v) in enumerate(value.items())
            ]
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _encode_kind(value, path, table)
    name = f"{type(value).__module__}.{type(value).__qualname__}"
    raise DocumentError(f"no document form for a {name}", pointer(path))


def encode_float(x):
    """A float, or its tag when it is not finite."""
    if math.isfinite(x):
        return x
    if math.isnan(x):
        return {"$float": "nan"}
    return {"$float": "inf" if x > 0 else "-inf"}


def _encode_kind(value, path, table):
    """A module or dataclass as its kind reference and its fields."""
    ref = table.ref_of(value, path)
    cls = type(value)
    out = {"$kind": list(ref)}
    for field in dataclasses.fields(value):
        if is_version_field(cls, field.name):
            continue
        item = getattr(value, field.name)
        if field.type is jax.Array and not isinstance(item, (jax.Array, np.ndarray)):
            raise DocumentError(
                f"field {field.name!r} is annotated as an array but holds a "
                f"{type(item).__name__}",
                pointer((*path, field.name)),
            )
        out[field.name] = encode(item, (*path, field.name), table)
    return out


def _encode_array(arr, path, *, weak=False, container=None):
    """An array as its dtype, shape and nested element lists."""
    name = arr.dtype.name
    if name not in ARRAY_DTYPES:
        raise DocumentError(f"no document form for dtype {name}", pointer(path))
    if not arr.dtype.isnative:
        raise DocumentError(
            f"dtype {arr.dtype.str} is not in native byte order; convert it first",
            pointer(path),
        )
    if np.issubdtype(arr.dtype, np.floating) or name == "bfloat16":
        data = _tag_floats(arr.astype(np.float64).tolist())
    else:
        data = arr.tolist()
    out = {"$array": name, "shape": list(arr.shape), "data": data}
    if weak:
        out["weak"] = True
    if container is not None:
        out["type"] = container
    return out


def _tag_floats(data):
    """Replace non-finite floats in nested lists with their tags."""
    if isinstance(data, list):
        return [_tag_floats(x) for x in data]
    return encode_float(data)
