"""Rebuild a study from a `StudyDocument` by the inverse field walk.

Each module is rebuilt from its stored field values as they are, the way a pytree
is unflattened, so array dtypes, weak types and Python types come back exactly as
written; `__init__` does not run. The guarantees construction gives are checked
instead, never applied, so a valid stored value is not coerced and an invalid one
is refused:

- a field with a converter must hold a value the converter returns unchanged
  (same type, dtype, shape, weak type and bytes): an `int64` tick stored as
  `10.75` is refused, not truncated;
- a field annotated `jax.Array` must hold an array;
- a class's `__post_init__`, where it defines one, runs and must leave every field
  as stored; then each `__check_init__` runs.

Every failure is a `DocumentError` at the pointer of the field or module.
"""

import contextlib
import contextvars
import dataclasses
import enum
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.codec.document import (
    CallableRefused,
    DocumentError,
    StudyDocument,
    check_integers,
    check_schema_version,
    current_max_tick,
    pointer,
    with_max_tick,
)
from spaceodyssey.codec.dump import ARRAY_DTYPES
from spaceodyssey.codec.files import (
    DEFAULT_MAX_BYTES,
    current_store,
    make_store,
    using_store,
)
from spaceodyssey.codec.kinds import (
    as_resolver,
    at_default,
    is_kind_class,
    is_optional_field,
    is_version_field,
    optional_default,
)
from spaceodyssey.core.capacities import Capacities, require_x64
from spaceodyssey.core.study import Agenda, Study

_CALLABLE_TAGS = frozenset({"$callable", "$function", "$lambda", "$fn"})
_FLOAT_TAGS = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}
_ARRAY_KEYS = frozenset({"$array", "shape", "data", "file", "weak", "type"})
_WEAK_ZEROS = {"int64": 0, "float64": 0.0}


_NULL_OK = contextvars.ContextVar("null_ok", default=frozenset())


@contextlib.contextmanager
def tolerating_null_arrays(fields):
    """Let `(class, field name)` pairs hold `None` where an array is annotated.

    Pre-flight uses it so a missing value reaches the structural checks, which
    report it with a code, instead of stopping the load. Only the named fields of
    exactly the named classes are exempt.
    """
    token = _NULL_OK.set(frozenset(fields))
    try:
        yield
    finally:
        _NULL_OK.reset(token)


def load_study(document, kinds=None, *, files=None, max_file_bytes=DEFAULT_MAX_BYTES):
    """Rebuild the `Study` a `StudyDocument` (or its JSON text) describes.

    `kinds` is a `KindResolver` or the mapping `dump_study` took. A module stored
    at an older kind version is rebuilt at the registered version through its
    class's `migrate`. An unknown kind id or version, a field missing or unknown,
    an array whose data does not match its declared shape or dtype, and a module
    whose validation fails each raise `DocumentError` carrying the JSON pointer of
    the offending value.

    `files` is the directory of array files a dump with large arrays wrote; a
    reference to a file that is missing, or whose bytes do not match the stored
    sha256, is refused at its pointer, and a document with references and no `files`
    directory is refused. A `{"no_switch": true}` phase boundary becomes
    `caps.max_tick`, and a study whose `horizon_tick` reaches `caps.max_tick` is
    refused. A file is read only when its size is the one the dtype and shape imply
    and at most `max_file_bytes`; it must be a regular file, never a symlink.
    """
    require_x64()
    if isinstance(document, str):
        document = StudyDocument.from_json(document)
    check_schema_version(document.schema_version)
    check_integers(document.study, ("study",))
    with using_store(make_store(files, 0, max_file_bytes)):
        study = decode(document.study, ("study",), as_resolver(kinds))
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
    if tag in _CALLABLE_TAGS:
        raise CallableRefused(tag, pointer(path))
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
        return _decode_enum(node, path, table)
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


def _decode_enum(node, path, table):
    """An enum member from its class reference and its value."""
    cls = table.resolve(_ref(node["$enum"], (*path, "$enum")), (*path, "$enum"))
    if not (isinstance(cls, type) and issubclass(cls, enum.Enum)):
        raise DocumentError(
            f"{cls.__qualname__} is not an enum", pointer((*path, "$enum"))
        )
    if "value" not in node:
        raise DocumentError("an $enum needs a value", pointer(path))
    value = decode(node["value"], (*path, "value"), table)
    try:
        member = cls(value)
    except (ValueError, TypeError) as err:
        raise DocumentError(str(err), pointer((*path, "value"))) from err
    if type(member.value) is not type(value):
        raise DocumentError(
            f"{cls.__qualname__} values are {type(member.value).__name__}",
            pointer((*path, "value")),
        )
    return member


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
    if not is_kind_class(cls):
        raise DocumentError(
            f"{cls.__qualname__} is not a module or dataclass",
            pointer((*path, "$kind")),
        )
    fields = dataclasses.fields(cls)
    stored = [f.name for f in fields if not is_version_field(cls, f.name)]
    version = table.current_version(ref[0], ref[1])
    migrated = None
    if ref[2] == version:
        _only(node, {"$kind", *stored}, path)
        optional = {f.name for f in fields if is_optional_field(f)}
        missing = [n for n in stored if n not in node and n not in optional]
        if missing:
            raise DocumentError(f"field {missing[0]!r} is missing", pointer(path))
    else:
        migrated = _migrated(node, ref, stored, path, table)
    if cls is Study and migrated is None:
        caps = decode(node["caps"], (*path, "caps"), table)
        if not isinstance(caps, Capacities):
            raise DocumentError("caps is not a Capacities", pointer((*path, "caps")))
        with with_max_tick(caps.max_tick):
            return _build(cls, node, path, table, version, migrated, fields)
    return _build(cls, node, path, table, version, migrated, fields)


def _build(cls, node, path, table, version, migrated, fields):
    """Decode each field of `node`, build the object and validate it."""
    obj = object.__new__(cls)
    values = {}
    for field in fields:
        where = (*path, field.name)
        if is_version_field(cls, field.name):
            value = version
        elif migrated is None and field.name not in node:
            # an optional field the encoder omitted at its default
            value = optional_default(field, cls)
        elif migrated is None:
            item = node[field.name]
            if cls is Agenda and field.name == "phase_until":
                item = _lowered_no_switch(item, where)
            value = decode(item, where, table)
            _check_field(cls, field, value, where)
            if is_optional_field(field) and at_default(field, value, cls):
                raise DocumentError(
                    f"field {field.name!r} is at its default; omit it", pointer(where)
                )
        else:
            value = migrated[field.name]
            _check_field(cls, field, value, where if field.name in node else path)
        values[field.name] = value
        object.__setattr__(obj, field.name, value)
    if cls is Study:
        _check_horizon(values, path)
    try:
        _validate(obj, cls, values, path)
    except DocumentError:
        raise
    except _CHECK_ERRORS as err:
        raise DocumentError(f"{cls.__qualname__}: {err}", pointer(path)) from err
    return obj


def _lowered_no_switch(node, path):
    """The `phase_until` node with each `{"no_switch": true}` as `caps.max_tick`."""
    top = current_max_tick()
    if top is None or type(node) is not dict or type(node.get("data")) is not list:
        return node
    data = []
    for i, item in enumerate(node["data"]):
        if type(item) is dict:
            if set(item) != {"no_switch"} or item["no_switch"] is not True:
                raise DocumentError(
                    'a phase boundary tag is {"no_switch": true}',
                    pointer((*path, "data", i)),
                )
            item = top
        data.append(item)
    return {**node, "data": data}


def _check_horizon(values, path):
    """Refuse a horizon at or beyond the tick a no-switch boundary lowers to."""
    horizon = values["horizon_tick"]
    top = current_max_tick()
    if top is not None and int(horizon) >= top:
        raise DocumentError(
            f"horizon_tick {int(horizon)} reaches caps.max_tick {top}, the tick a "
            "no-switch phase boundary stands for",
            pointer((*path, "horizon_tick")),
        )


def _migrated(node, ref, stored, path, table):
    """The current version's fields, from a module stored at an older version."""
    params = {
        key: decode(value, (*path, key), table)
        for key, value in node.items()
        if key != "$kind"
    }
    out = table.migrate(ref, params, path)
    if sorted(out) != sorted(stored):
        raise DocumentError(
            f"migrating kind {list(ref)} gives fields {sorted(out)}, "
            f"not {sorted(stored)}",
            pointer(path),
        )
    return out


# What a validation hook raises for a bad value; anything else propagates.
_CHECK_ERRORS = (ValueError, TypeError, AttributeError, IndexError, KeyError)


def _check_field(cls, field, value, path):
    """Refuse a value its converter would change or its annotation does not fit."""
    if value is None and (cls, field.name) in _NULL_OK.get():
        return
    if cls is Capacities and value is None:
        # no capacity is nullable; refuse at the field, not at the caps table
        raise DocumentError(f"field {field.name!r} cannot be null", pointer(path))
    if field.type is jax.Array and not isinstance(value, (jax.Array, np.ndarray)):
        raise DocumentError(
            f"field {field.name!r} holds an array, not a {type(value).__name__}",
            pointer(path),
        )
    converter = field.metadata.get("converter")
    if converter is None:
        return
    try:
        converted = converter(value)
    except _CHECK_ERRORS as err:
        raise DocumentError(f"field {field.name!r}: {err}", pointer(path)) from err
    if not same_value(converted, value):
        raise DocumentError(
            f"field {field.name!r} is not stored as its converter makes it "
            "(type, dtype, shape or value differs)",
            pointer(path),
        )


def _validate(obj, cls, values, path):
    """Run the hooks construction runs, requiring them to leave the fields alone."""
    if hasattr(cls, "__post_init__"):
        obj.__post_init__()
        for name, value in values.items():
            if not same_value(getattr(obj, name), value):
                raise DocumentError(
                    f"{cls.__qualname__}.__post_init__ changes field {name!r}",
                    pointer((*path, name)),
                )
    if isinstance(obj, eqx.Module):
        for kls in cls.__mro__:
            check = kls.__dict__.get("__check_init__")
            if check is not None:
                check(obj)


def same_value(a, b):
    """Equal in type and content: dtype, shape, weak type and bytes for arrays."""
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, jax.Array):
        if a.weak_type != b.weak_type:
            return False
        a, b = np.asarray(a), np.asarray(b)
    if isinstance(a, (np.ndarray, np.generic)):
        return (a.dtype, a.shape) == (b.dtype, b.shape) and a.tobytes() == b.tobytes()
    if type(a) in (tuple, list):
        return len(a) == len(b) and all(map(same_value, a, b))
    if type(a) is dict:
        return list(a) == list(b) and all(same_value(a[k], b[k]) for k in a)
    if type(a) is float:
        both_nan = math.isnan(a) and math.isnan(b)
        return both_nan or (a == b and math.copysign(1, a) == math.copysign(1, b))
    if is_kind_class(type(a)):
        return all(
            same_value(getattr(a, f.name), getattr(b, f.name))
            for f in dataclasses.fields(a)
        )
    return a == b


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
    largest = np.iinfo(np.intp).max
    if any(n > largest for n in shape) or math.prod(shape) > largest // dtype.itemsize:
        raise DocumentError(
            "the shape holds more elements than an array can", pointer((*path, "shape"))
        )
    if ("data" in node) == ("file" in node):
        raise DocumentError(
            "an array has either data or a file reference", pointer(path)
        )
    if "file" in node:
        arr = _read_file(node["file"], dtype, shape, (*path, "file"))
    else:
        data = _elements(node["data"], dtype, (*path, "data"))
        try:
            arr = np.array(data, dtype=dtype)
        except (ValueError, OverflowError) as err:
            raise DocumentError(
                f"bad array data: {err}", pointer((*path, "data"))
            ) from err
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


def _read_file(ref, dtype, shape, path):
    """The array a file reference holds, checked against its dtype and shape."""
    store = current_store()
    if store is None:
        raise DocumentError(
            "the document holds an array file reference; pass the files directory",
            pointer(path),
        )
    expected = math.prod(shape) * dtype.itemsize
    raw = store.get(ref, path, expected)
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


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
