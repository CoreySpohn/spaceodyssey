"""Deterministic typed encoding and snapshots of execution inputs."""

import dataclasses as _dataclasses
import hashlib as _hashlib
import math as _math
import struct as _struct
from typing import Any as _Any

import jax as _jax
import numpy as _np

from spaceodyssey.plan import PlanSummary as _PlanSummary
from spaceodyssey.truth import Truth as _Truth

__all__ = [
    "CODEC",
    "MAGIC",
    "digest",
    "encode",
    "from_json_value",
    "snapshot",
    "to_json_value",
]

CODEC = "spaceodyssey.inputs/1"
MAGIC = b"spaceodyssey.inputs/1\x00"
_ARRAY_CODES = {("b", 1): 1, ("i", 4): 2, ("i", 8): 3, ("f", 4): 4, ("f", 8): 5}


def _utf8(value: str, path: str) -> bytes:
    try:
        return value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError(f"{path}: invalid UTF-8 string") from error


def _kind(value: _Any, path: str) -> tuple[int, _Any]:
    if value is None:
        return 0, None
    if isinstance(value, (bool, _np.bool_)):
        return 1, bool(value)
    if isinstance(value, (int, _np.integer)):
        return 2, int(value)
    if isinstance(value, (float, _np.float16, _np.float32, _np.float64)) or (
        isinstance(value, _np.longdouble) and value.dtype == _np.dtype("f8")
    ):
        value = float(value) + 0.0
        if _math.isnan(value):
            raise ValueError(f"{path}: NaN scalar is unsupported")
        return 3, value
    if isinstance(value, str):
        value = str(value)
        _utf8(value, path)
        return 4, value
    if type(value) is _PlanSummary:
        return 16, value
    if type(value) is _Truth:
        return 17, value
    if isinstance(value, (_np.ndarray, _jax.Array)):
        return 7, value
    if type(value) in (tuple, list):
        return 5, value
    if type(value) is dict:
        return 6, value
    raise TypeError(f"{path}: unsupported type {type(value).__name__}")


def _entries(value: dict, path: str) -> list[tuple[bytes, str, _Any]]:
    entries = []
    for key, child in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{path}: map keys must be strings")
        key = str(key)
        key_path = f"{path}[{key!r}]"
        entries.append((_utf8(key, key_path), key, child))
    return sorted(entries, key=lambda entry: entry[0])


def _u64(value: int) -> bytes:
    return _struct.pack("<Q", value)


def _blob(value: bytes) -> bytes:
    return _u64(len(value)) + value


def _decimal(value: int) -> str:
    # Small chunks avoid Python's configurable decimal conversion limit.
    sign = "-" if value < 0 else ""
    value = abs(value)
    chunks = []
    while value >= 10**9:
        value, remainder = divmod(value, 10**9)
        chunks.append(f"{remainder:09d}")
    return sign + str(value) + "".join(reversed(chunks))


def _parse_integer(value: str, path: str) -> int:
    digits = value.removeprefix("-")
    if (
        not digits
        or not digits.isascii()
        or not digits.isdecimal()
        or (len(digits) > 1 and digits[0] == "0")
        or value == "-0"
    ):
        raise TypeError(f"{path}: invalid decimal integer")
    parsed = 0
    for start in range(0, len(digits), 9):
        chunk = digits[start : start + 9]
        parsed = parsed * 10 ** len(chunk) + int(chunk)
    return -parsed if value.startswith("-") else parsed


def _array(value: _Any, path: str) -> bytes:
    if isinstance(value, _jax.Array) and _jax.dtypes.issubdtype(
        value.dtype, _jax.dtypes.prng_key
    ):
        raise TypeError(f"{path}: PRNG-key arrays are unsupported")
    host = _np.asarray(value)
    code = _ARRAY_CODES.get((host.dtype.kind, host.dtype.itemsize))
    if code is None:
        raise TypeError(f"{path}: unsupported array dtype {host.dtype}")
    host = host.astype(host.dtype.newbyteorder("<"), copy=False)
    if code == 1:
        # NumPy can retain noncanonical true bytes from buffer views.
        host = host.astype(_np.uint8)
    return (
        bytes([code])
        + _u64(host.ndim)
        + b"".join(_u64(dim) for dim in host.shape)
        + _blob(host.tobytes(order="C"))
    )


def _map(value: dict, path: str, attributes: bool = False) -> bytes:
    entries = _entries(value, path)
    return (
        b"\x06"
        + _u64(len(entries))
        + b"".join(
            _blob(key_bytes)
            + _node(child, f"{path}.{key}" if attributes else f"{path}[{key!r}]")
            for key_bytes, key, child in entries
        )
    )


def _node(value: _Any, path: str) -> bytes:
    tag, value = _kind(value, path)
    if tag == 0:
        body = b""
    elif tag == 1:
        body = bytes([value])
    elif tag == 2:
        body = _blob(_decimal(value).encode("ascii"))
    elif tag == 3:
        body = _struct.pack("<d", value)
    elif tag == 4:
        body = _blob(_utf8(value, path))
    elif tag == 5:
        body = _u64(len(value)) + b"".join(
            _node(child, f"{path}[{i}]") for i, child in enumerate(value)
        )
    elif tag == 6:
        return _map(value, path)
    elif tag == 7:
        body = _array(value, path)
    elif tag == 16:
        body = _node(_PlanSummary.CONTRACT_VERSION, f"{path}.CONTRACT_VERSION")
        body += _map(
            {
                field.name: getattr(value, field.name)
                for field in _dataclasses.fields(_PlanSummary)
            },
            path,
            attributes=True,
        )
    else:
        for field, expected in (("model_id", 4), ("theta", 7), ("context", 6)):
            field_path = f"{path}.{field}"
            if _kind(getattr(value, field), field_path)[0] != expected:
                raise TypeError(f"{field_path}: invalid Truth field type")
        for _, key, child in _entries(value.context, f"{path}.context"):
            child_path = f"{path}.context[{key!r}]"
            if _kind(child, child_path)[0] != 7:
                raise TypeError(f"{child_path}: Truth context values must be arrays")
        body = (
            _node(value.model_id, f"{path}.model_id")
            + _node(value.theta, f"{path}.theta")
            + _map(value.context, f"{path}.context")
        )
    return bytes([tag]) + body


def encode(value: _Any, name: str = "value") -> bytes:
    """Return the framed typed byte stream, with paths in refusal errors."""
    return MAGIC + _node(value, name)


def digest(value: _Any, name: str = "value") -> str:
    """Return the lowercase SHA-256 digest of the framed input."""
    return _hashlib.sha256(encode(value, name)).hexdigest()


def _snapshot(value: _Any) -> _Any:
    if isinstance(value, _np.ndarray):
        copied = value.copy()
        copied.flags.writeable = False
        return copied
    if type(value) in (tuple, list):
        return tuple(_snapshot(child) for child in value)
    if type(value) is dict:
        return {key: _snapshot(child) for key, child in value.items()}
    if type(value) is _Truth:
        return _jax.tree_util.tree_map(_snapshot, value)
    return value


def snapshot(value: _Any, name: str = "value") -> _Any:
    """Validate and copy mutable inputs while retaining immutable JAX arrays."""
    _node(value, name)
    return _snapshot(value)


def to_json_value(value: _Any, name: str = "value") -> object:
    """Return a tagged JSON value for an array-free input."""
    tag, value = _kind(value, name)
    if tag == 0:
        return {"t": "none"}
    if tag in (1, 2, 3, 4):
        label = {1: "bool", 2: "int", 3: "float", 4: "str"}[tag]
        if tag == 2:
            value = _decimal(value)
        elif tag == 3:
            value = value.hex()
        return {"t": label, "v": value}
    if tag == 5:
        return {
            "t": "seq",
            "v": [
                to_json_value(child, f"{name}[{i}]") for i, child in enumerate(value)
            ],
        }
    if tag == 6:
        return {
            "t": "map",
            "v": [
                [key, to_json_value(child, f"{name}[{key!r}]")]
                for _, key, child in _entries(value, name)
            ],
        }
    raise TypeError(f"{name}: arrays and science containers cannot be stored inline")


def _from_json(obj: object, path: str) -> _Any:
    if type(obj) is not dict or type(obj.get("t")) is not str:
        raise TypeError(f"{path}: expected a tagged JSON value")
    tag = obj["t"]
    if set(obj) != ({"t"} if tag == "none" else {"t", "v"}):
        raise TypeError(f"{path}: invalid tagged JSON fields")
    if tag == "none":
        return None
    value = obj["v"]
    if tag == "bool" and type(value) is bool:
        return value
    if tag == "str" and type(value) is str:
        _utf8(value, path)
        return value
    if tag == "int" and type(value) is str:
        return _parse_integer(value, path)
    if tag == "float" and type(value) is str:
        try:
            parsed = float.fromhex(value)
        except (ValueError, OverflowError) as error:
            raise TypeError(f"{path}: invalid {tag} string") from error
        return _kind(parsed, path)[1]
    if tag == "seq" and type(value) is list:
        return tuple(_from_json(child, f"{path}[{i}]") for i, child in enumerate(value))
    if tag == "map" and type(value) is list:
        result = {}
        for i, entry in enumerate(value):
            if type(entry) is not list or len(entry) != 2 or type(entry[0]) is not str:
                raise TypeError(f"{path}[{i}]: expected a string-keyed map entry")
            key, child = entry
            key_path = f"{path}[{key!r}]"
            _utf8(key, key_path)
            if key in result:
                raise TypeError(f"{key_path}: duplicate map key")
            result[key] = _from_json(child, key_path)
        return result
    raise TypeError(f"{path}: invalid JSON tag or value for {tag!r}")


def from_json_value(obj: object, codec: str, name: str = "value") -> _Any:
    """Decode a tagged JSON value using the declared codec identifier."""
    if codec != CODEC:
        raise TypeError(f"{name}.codec: unknown input codec {codec!r}")
    return _from_json(obj, name)
