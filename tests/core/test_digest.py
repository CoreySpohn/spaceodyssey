"""Tests for the deterministic typed encoding and digest."""

import re
from collections import namedtuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core.digest import CODEC, MAGIC, digest, encode

VECTORS = [
    (
        {"b": 1, "a": True},
        "0602000000000000000100000000000000610101010000000000000062"
        "02010000000000000031",
        "b3bd9997616e5a9a62193db3a894518491162bd59397c2534ae5f46ea677e653",
    ),
    (
        ("\u00e9", 0.95, -0.0),
        "050300000000000000040200000000000000c3a903666666666666ee3f030000000000000000",
        "60f4a8cc9765e91ac698587349e1599aaeadd5034de254ae2a2d14ab00bdaa39",
    ),
    (
        np.array([1.5, -2.0], dtype="<f8"),
        "0705010000000000000002000000000000001000000000000000"
        "000000000000f83f00000000000000c0",
        "07f27894c5afde4175920563b14a0e04f28fe50422df0afae2265745922a037b",
    ),
]


@pytest.mark.parametrize(("value", "node_hex", "expected"), VECTORS)
def test_byte_vectors(value, node_hex, expected):
    assert CODEC == "spaceodyssey.inputs/1"
    assert MAGIC == b"spaceodyssey.inputs/1\x00"
    assert len(MAGIC) == 22
    assert encode(value) == MAGIC + bytes.fromhex(node_hex)
    assert digest(value) == expected


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ({"a": 1}, (("a", 1),)),
        (1, 1.0),
        (np.array([1.5], dtype="f4"), np.array([1.5], dtype="f8")),
        ((), {}),
        ("\u00e9", "e\u0301"),
        ({"a": None}, {}),
        (np.array(1, dtype="i4"), np.array(1, dtype="i8")),
        (np.array(1, dtype="i4"), 1),
        (np.array([], dtype="f4"), {}),
        (np.empty((0, 2), dtype="f4"), np.empty((2, 0), dtype="f4")),
        (np.array([-0.0]), np.array([0.0])),
    ],
)
def test_distinct_values(left, right):
    assert digest(left) != digest(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ({"b": 1, "a": True}, {"a": True, "b": 1}),
        (["\u00e9", 0.95, -0.0], ("\u00e9", 0.95, 0.0)),
        (-0.0, 0.0),
        (np.array([1.5, -2.0], dtype=">f8"), VECTORS[2][0]),
        (np.array([1, -2], dtype=">i4"), np.array([1, -2], dtype="<i4")),
        (np.float64(0.95), 0.95),
        (np.float32(0.95), float(np.float32(0.95))),
        (np.float16(0.95), float(np.float16(0.95))),
        (np.int64(1), 1),
        (np.uint64(2**64 - 1), 2**64 - 1),
        (np.bool_(True), True),
        (np.str_("x"), "x"),
        (jnp.array([1.5, -2.0], dtype=jnp.float32), np.array([1.5, -2.0], "f4")),
        (np.array([[1, 2], [3, 4]], "i4").T, np.array([[1, 3], [2, 4]], "i4")),
    ],
)
def test_normalized_values(left, right):
    assert encode(left) == encode(right)
    assert digest(left) == digest(right)


@pytest.mark.parametrize("dtype", ["?", "i4", "i8", "f4", "f8"])
def test_byte_order_and_zero_dimensional_arrays(dtype):
    little = np.array([0, 1], dtype=np.dtype(dtype).newbyteorder("<"))
    big = little.astype(little.dtype.newbyteorder(">"))
    assert encode(little) == encode(big)
    scalar_array = np.array(1, dtype=dtype)
    assert encode(scalar_array) != encode(scalar_array.item())
    assert encode(scalar_array)[len(MAGIC) + 2 : len(MAGIC) + 10] == bytes(8)


def test_bool_array_bytes():
    assert encode(np.array([True, False])) == MAGIC + bytes.fromhex(
        "07010100000000000000020000000000000002000000000000000100"
    )
    unusual_true_bytes = np.array([255, 0], dtype="u1").view("?")
    assert encode(unusual_true_bytes) == encode(np.array([True, False]))


@pytest.mark.parametrize("dtype", ["<u4", ">u4", "<u8", ">u8"])
def test_array_nan_payloads(dtype):
    bits = (
        [0x7FC00001, 0x7FC00002]
        if "4" in dtype
        else [
            0x7FF8000000000001,
            0x7FF8000000000002,
        ]
    )
    floats = np.array(bits, dtype=dtype).view(dtype.replace("u", "f"))
    assert np.isnan(floats).all()
    assert digest(floats[:1]) != digest(floats[1:])
    assert encode(floats).endswith(
        np.array(bits, dtype=dtype.replace(">", "<")).tobytes()
    )


@pytest.mark.parametrize("value", [2**200, -(2**200), float("inf"), -float("inf")])
def test_large_integers_and_infinities(value):
    assert digest(value) == digest(value)
    assert encode(value).startswith(MAGIC)


def test_integers_beyond_decimal_conversion_limit():
    for value, sign in [(10**5000, ""), (-(10**5000), "-")]:
        decimal = sign + "1" + "0" * 5000
        assert encode(value).endswith(decimal.encode("ascii"))


class OtherModule(eqx.Module):
    value: int


@pytest.mark.parametrize(
    "value",
    [
        np.array([object()], dtype=object),
        np.array(["a"]),
        np.array([b"a"]),
        np.array([1j]),
        np.array([1], dtype="u4"),
        np.array([1], dtype="u8"),
        np.array([1], dtype="i1"),
        np.array([1], dtype="i2"),
        np.array([1], dtype="f2"),
        np.array([(1,)], dtype=[("x", "i4")]),
        jax.random.key(0),
        jax.random.PRNGKey(0),
        namedtuple("Point", "x")(1),
        OtherModule(1),
        lambda: None,
        1j,
        {1: "a"},
        {"a"},
    ],
)
@pytest.mark.parametrize("operation", [encode, digest])
def test_refusal_paths(value, operation):
    with pytest.raises(TypeError, match=re.escape("weights[3]")):
        operation([None, None, None, value], name="weights")


@pytest.mark.parametrize("operation", [encode, digest])
def test_large_non_string_map_key_refused(operation):
    with pytest.raises(TypeError, match=re.escape("data[0]")):
        operation([{10**5000: None}], name="data")


@pytest.mark.parametrize(
    "value", [float("nan"), np.float16("nan"), np.float32("nan"), np.float64("nan")]
)
@pytest.mark.parametrize("operation", [encode, digest])
def test_scalar_nan_refused(value, operation):
    with pytest.raises(ValueError, match=re.escape("data['x']")):
        operation({"x": value}, name="data")


@pytest.mark.parametrize("value", ["\ud800", {"\ud800": 1}])
@pytest.mark.parametrize("operation", [encode, digest])
def test_surrogates_refused(value, operation):
    with pytest.raises(ValueError, match="data"):
        operation(value, name="data")


def test_extended_float_refused_when_wider():
    value = np.longdouble(1)
    if np.dtype(np.longdouble) == np.dtype(np.float64):
        assert digest(value) == digest(1.0)
    else:
        with pytest.raises(TypeError, match="data"):
            encode(value, name="data")


class LyingString(str):
    replacement = "b"

    def __str__(self):
        return self.replacement

    def encode(self, *args, **kwargs):
        return self.replacement.encode(*args, **kwargs)

    def __getitem__(self, key):
        return self.replacement[key]


class LyingInteger(int):
    replacement = 99

    def __int__(self):
        return self.replacement

    def __index__(self):
        return self.replacement

    def __repr__(self):
        return str(self.replacement)


class LyingFloat(float):
    replacement = 99.0

    def __float__(self):
        return self.replacement


@pytest.mark.parametrize(
    ("value", "expected"),
    [(LyingString("a"), "a"), (LyingInteger(7), 7), (LyingFloat(0.5), 0.5)],
)
def test_scalar_subclasses_use_underlying_values(value, expected):
    assert encode(value) == encode(expected)
    original_digest = digest(expected)
    value.replacement = "changed" if isinstance(value, str) else 42
    assert digest(value) == original_digest


def test_string_subclass_map_keys_normalized():
    key = LyingString("a")
    value = {key: 1, "b": 2}
    expected = {"a": 1, "b": 2}
    assert encode(value) == encode(expected)
    assert digest(dict(reversed(list(value.items())))) == digest(expected)
    key.replacement = "changed"
    assert digest(value) == digest(expected)


@pytest.mark.parametrize("operation", [encode, digest])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("plain_key", [False, True])
def test_canonical_map_key_collisions_refused(operation, reverse, plain_key):
    class IdentityString(LyingString):
        __eq__ = object.__eq__
        __hash__ = object.__hash__

    entries = [(IdentityString("a"), 1), ("a" if plain_key else IdentityString("a"), 2)]
    value = dict(reversed(entries) if reverse else entries)
    assert len(value) == 2
    with pytest.raises(
        TypeError, match=re.escape("data[0]['a']: duplicate canonical map key")
    ):
        operation([value], name="data")


def test_integer_subclass_beyond_decimal_conversion_limit():
    value = LyingInteger(10**5000)
    assert encode(value) == encode(10**5000)


@pytest.mark.parametrize("operation", [encode, digest])
@pytest.mark.parametrize("container_type", [list, dict])
def test_container_cycles_refused_with_path(operation, container_type):
    value = container_type()
    if container_type is list:
        value.append(value)
        path = "data[0]"
    else:
        value["self"] = value
        path = "data['self']"
    with pytest.raises(TypeError, match=re.escape(path)):
        operation(value, name="data")


@pytest.mark.parametrize("operation", [encode, digest])
def test_shared_containers_are_not_cycles(operation):
    child = {"x": [1]}
    assert operation([child, child]) == operation([{"x": [1]}, {"x": [1]}])


@pytest.mark.parametrize("operation", [encode, digest])
def test_deep_nesting_refused_as_type_error(operation):
    value = None
    for _ in range(3000):
        value = [value]
    with pytest.raises(TypeError, match="data"):
        operation(value, name="data")


@pytest.mark.parametrize(
    ("value", "stream"),
    [
        (None, b"spaceodyssey.inputs/1\x00\x00"),
        (
            np.array([1, -2], dtype="i8"),
            b"spaceodyssey.inputs/1\x00"
            b"\x07\x03\x01\x00\x00\x00\x00\x00\x00\x00"
            b"\x02\x00\x00\x00\x00\x00\x00\x00"
            b"\x10\x00\x00\x00\x00\x00\x00\x00"
            b"\x01\x00\x00\x00\x00\x00\x00\x00"
            b"\xfe\xff\xff\xff\xff\xff\xff\xff",
        ),
        (
            np.array([1, -2], dtype="i4"),
            b"spaceodyssey.inputs/1\x00"
            b"\x07\x02\x01\x00\x00\x00\x00\x00\x00\x00"
            b"\x02\x00\x00\x00\x00\x00\x00\x00"
            b"\x08\x00\x00\x00\x00\x00\x00\x00"
            b"\x01\x00\x00\x00\xfe\xff\xff\xff",
        ),
        (
            np.array([True, False]),
            b"spaceodyssey.inputs/1\x00"
            b"\x07\x01\x01\x00\x00\x00\x00\x00\x00\x00"
            b"\x02\x00\x00\x00\x00\x00\x00\x00"
            b"\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00",
        ),
        (
            np.array([1.5, -2.0], dtype="f4"),
            b"spaceodyssey.inputs/1\x00"
            b"\x07\x04\x01\x00\x00\x00\x00\x00\x00\x00"
            b"\x02\x00\x00\x00\x00\x00\x00\x00"
            b"\x08\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\xc0\x3f\x00\x00\x00\xc0",
        ),
    ],
)
def test_complete_stream_bytes(value, stream):
    assert encode(value) == stream


def test_scalar_bool_bytes():
    assert encode(True) == b"spaceodyssey.inputs/1\x00\x01\x01"
    assert encode(False) == b"spaceodyssey.inputs/1\x00\x01\x00"
