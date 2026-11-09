"""Deterministic typed encoding and SHA-256 digest of plain inputs.

Handles None, bool, int, float, str, NumPy and JAX arrays, and lists, tuples and
string-keyed dicts of those. The golden histories pin digests made with this
encoding, so the framing and tags must not change.
"""

import functools as _functools
import hashlib as _hashlib
import math as _math
import struct as _struct
from typing import Any as _Any

import jax as _jax
import numpy as _np

__all__ = ["CODEC", "MAGIC", "digest", "encode"]

CODEC = "spaceodyssey.inputs/1"
MAGIC = b"spaceodyssey.inputs/1\x00"
_ARRAY_CODES = {("b", 1): 1, ("i", 4): 2, ("i", 8): 3, ("f", 4): 4, ("f", 8): 5}


def _utf8(value: str, path: str) -> bytes:
    try:
        return str.encode(value, "utf-8", "strict")
    except UnicodeEncodeError as error:
        raise ValueError(f"{path}: invalid UTF-8 string") from error


def _kind(value: _Any, path: str) -> tuple[int, _Any]:
    if value is None:
        return 0, None
    if isinstance(value, (bool, _np.bool_)):
        return 1, bool(value)
    if isinstance(value, int):
        # Unbound base methods bypass subclass conversions and preserve huge ints.
        return 2, int.__int__(value)
    if isinstance(value, _np.integer):
        return 2, _np.integer.__int__(value)
    if isinstance(value, (float, _np.float16, _np.float32, _np.float64)) or (
        isinstance(value, _np.longdouble) and value.dtype == _np.dtype("f8")
    ):
        value = (
            float.__float__(value)
            if isinstance(value, float)
            else _np.floating.__float__(value)
        ) + 0.0
        if _math.isnan(value):
            raise ValueError(f"{path}: NaN scalar is unsupported")
        return 3, value
    if isinstance(value, str):
        encoded = _utf8(value, path)
        if type(value) is not str:
            value = encoded.decode("utf-8")
        return 4, value
    if isinstance(value, (_np.ndarray, _jax.Array)):
        return 7, value
    if type(value) in (tuple, list):
        return 5, value
    if type(value) is dict:
        return 6, value
    raise TypeError(f"{path}: unsupported type {type(value).__name__}")


def _entries(value: dict, path: str) -> list[tuple[bytes, str, _Any]]:
    entries = []
    keys = set()
    for key, child in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{path}: map keys must be strings")
        key = _kind(key, path)[1]
        key_path = f"{path}[{key!r}]"
        if key in keys:
            raise TypeError(f"{key_path}: duplicate canonical map key")
        keys.add(key)
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


def _traversal(operation: _Any) -> _Any:
    @_functools.wraps(operation)
    def guarded(value: _Any, path: str, active: set[int] | None = None) -> _Any:
        if active is None:
            active = set()
        tracked = type(value) in (list, tuple, dict)
        identity = id(value)
        if tracked:
            if identity in active:
                raise TypeError(f"{path}: cyclic input")
            active.add(identity)
        try:
            return operation(value, path, active)
        except RecursionError as error:
            raise TypeError(f"{path}: input nesting is too deep") from error
        finally:
            if tracked:
                active.remove(identity)

    return guarded


def _map(value: dict, path: str, active: set[int], attributes: bool = False) -> bytes:
    entries = _entries(value, path)
    return (
        b"\x06"
        + _u64(len(entries))
        + b"".join(
            _blob(key_bytes)
            + _node(
                child, f"{path}.{key}" if attributes else f"{path}[{key!r}]", active
            )
            for key_bytes, key, child in entries
        )
    )


@_traversal
def _node(value: _Any, path: str, active: set[int]) -> bytes:
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
            _node(child, f"{path}[{i}]", active) for i, child in enumerate(value)
        )
    elif tag == 6:
        return _map(value, path, active)
    elif tag == 7:
        body = _array(value, path)
    else:
        body = _array(value, path)
    return bytes([tag]) + body


def encode(value: _Any, name: str = "value") -> bytes:
    """Return the framed typed byte stream, with paths in refusal errors."""
    return MAGIC + _node(value, name)


def digest(value: _Any, name: str = "value") -> str:
    """Return the lowercase SHA-256 digest of the framed input."""
    return _hashlib.sha256(encode(value, name)).hexdigest()
