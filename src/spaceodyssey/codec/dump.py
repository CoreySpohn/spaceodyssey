"""Encode a study as a `StudyDocument` by a generic walk over its fields."""

import dataclasses
import enum
import math

import jax
import numpy as np

from spaceodyssey.codec.document import (
    CallableRefused,
    DocumentError,
    StudyDocument,
    current_max_tick,
    pointer,
    with_max_tick,
)
from spaceodyssey.codec.files import (
    DEFAULT_MAX_INLINE,
    current_store,
    make_store,
    using_store,
)
from spaceodyssey.codec.kinds import (
    as_resolver,
    at_default,
    is_optional_field,
    is_version_field,
)
from spaceodyssey.core.capacities import require_x64
from spaceodyssey.core.study import Agenda, Study

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


def dump_study(study, kinds=None, *, files=None, max_inline=DEFAULT_MAX_INLINE):
    """Encode `study` as a `StudyDocument`.

    `kinds` is a `KindResolver`, or a mapping of `(group, id, version)` to each
    class the study holds beyond the core's own types (`CORE_KINDS`). Every field
    of every module is written, static and dynamic alike, with exact Python types
    and array dtypes, so `load_study` rebuilds an equal pytree. A value of a type
    the document has no form for (a function, a set, a complex array) raises
    `DocumentError` naming its pointer, and a callable (an activation function, a
    hook) raises `CallableRefused`.

    An array of more than `max_inline` elements is written to a file in the directory
    `files`, when one is given, and the document keeps a reference with its sha256
    (see `spaceodyssey.codec.files`). A `phase_until` entry equal to `caps.max_tick`
    is written as `{"no_switch": true}`; a study whose `horizon_tick` reaches
    `caps.max_tick` is refused, as the loader would refuse it.
    """
    require_x64()
    if not isinstance(study, Study):
        raise TypeError(f"dump_study needs a Study, not {type(study).__name__}")
    store = make_store(files, max_inline)
    table = as_resolver(kinds)
    with using_store(store), with_max_tick(study.caps.max_tick):
        if int(study.horizon_tick) >= study.caps.max_tick:
            raise DocumentError(
                f"horizon_tick {int(study.horizon_tick)} reaches caps.max_tick "
                f"{study.caps.max_tick}, the tick a no-switch boundary stands for",
                "/study/horizon_tick",
            )
        return StudyDocument(study=encode(study, ("study",), table))


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
    if callable(value):
        label = getattr(value, "__qualname__", None) or type(value).__qualname__
        raise CallableRefused(
            f"{getattr(value, '__module__', None) or ''}.{label}", pointer(path)
        )
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
        if is_optional_field(field) and at_default(field, item, cls):
            continue
        if field.type is jax.Array and not isinstance(item, (jax.Array, np.ndarray)):
            raise DocumentError(
                f"field {field.name!r} is annotated as an array but holds a "
                f"{type(item).__name__}",
                pointer((*path, field.name)),
            )
        out[field.name] = encode(item, (*path, field.name), table)
        if cls is Agenda and field.name == "phase_until":
            out[field.name] = _tag_no_switch(out[field.name])
    return out


def _tag_no_switch(node):
    """Write each boundary equal to `caps.max_tick` as the no-switch tag."""
    top = current_max_tick()
    if "data" not in node:
        return node
    node["data"] = [{"no_switch": True} if t == top else t for t in node["data"]]
    return node


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
    out = {"$array": name, "shape": list(arr.shape)}
    store = current_store()
    if store is not None and store.wants(arr.size):
        out["file"] = store.put(np.ascontiguousarray(arr).tobytes())
    elif np.issubdtype(arr.dtype, np.floating) or name == "bfloat16":
        out["data"] = _tag_floats(arr.astype(np.float64).tolist())
    else:
        out["data"] = arr.tolist()
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
