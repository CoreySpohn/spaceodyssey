"""Tests for deterministic typed inputs and private snapshots."""

import dataclasses
import json
import os
import re
import subprocess
import sys
from collections import namedtuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.inputs import (
    CODEC,
    MAGIC,
    digest,
    encode,
    from_json_value,
    snapshot,
    to_json_value,
)
from spaceodyssey.plan import PlanSummary
from spaceodyssey.truth import Truth

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

# Explicit dtypes keep these containers independent of JAX precision settings.
CONTAINER_CODE = """
import jax.numpy as jnp
from spaceodyssey.inputs import digest
from spaceodyssey.plan import PlanSummary
from spaceodyssey.truth import Truth
plan = PlanSummary(
    bin_wavelength_nm=jnp.array([500., 700.], dtype=jnp.float32),
    noise_sigma=jnp.array([[0.25, 0.5]], dtype=jnp.float32),
    noise_known=jnp.array([True]),
    spectral_resolution=jnp.array([70.], dtype=jnp.float32),
    visit_epochs_d=jnp.array([[0., 90.]], dtype=jnp.float32),
    visit_success=jnp.array([[True, False]]),
    visit_photometric_snr=jnp.array([[5., 0.]], dtype=jnp.float32),
    spectrum_visit=jnp.array([0], dtype=jnp.int32),
    astrometric_precision_mas=jnp.array([[2., 3.]], dtype=jnp.float32),
    k_precision_ms=jnp.array([1.], dtype=jnp.float32),
    noise_corr_factors=jnp.array([[[0.125], [0.25]]], dtype=jnp.float32),
    flux_reference="unit-flux",
    k_precision_assumed_ms=jnp.array([2.], dtype=jnp.float32),
)
truth = Truth(
    theta=jnp.array([[1., 2.]], dtype=jnp.float32),
    context={key: jnp.array([value], dtype=jnp.float32)
             for key, value in {("offset", 0.5), ("scale", 2.)}},
    model_id="toy/1",
)
"""


@pytest.fixture
def containers():
    namespace = {}
    exec(CONTAINER_CODE, namespace)
    return namespace["plan"], namespace["truth"]


@pytest.mark.parametrize(("value", "node_hex", "expected"), VECTORS)
def test_byte_vectors(value, node_hex, expected):
    assert CODEC == "spaceodyssey.inputs/1"
    assert MAGIC == b"spaceodyssey.inputs/1\x00"
    assert len(MAGIC) == 22
    assert encode(value) == MAGIC + bytes.fromhex(node_hex)
    assert digest(value) == expected


@pytest.mark.parametrize("seed", ["1", "98765"])
def test_container_digests_across_processes(seed, containers):
    expected = [
        "3958cf1b00ad1c966aad1e0bfe64ac0f3396af02a82966d8300cc36b4ebc93bf",
        "a72f3d90df553a2eb115008c843f6376f6efd463191f7c592a386d085eab8733",
    ]
    assert [digest(value) for value in containers] == expected
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            CONTAINER_CODE + "\nprint(digest(plan))\nprint(digest(truth))",
        ],
        env={**os.environ, "PYTHONHASHSEED": seed},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == expected


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
    restored = from_json_value(to_json_value(value), CODEC)
    assert digest(restored) == digest(value)
    assert encode(value).startswith(MAGIC)


def test_integers_beyond_decimal_conversion_limit():
    for value, sign in [(10**5000, ""), (-(10**5000), "-")]:
        decimal = sign + "1" + "0" * 5000
        assert to_json_value(value) == {"t": "int", "v": decimal}
        assert encode(value).endswith(decimal.encode("ascii"))
        assert from_json_value(to_json_value(value), CODEC) == value


def test_plan_fields_affect_digest(containers, monkeypatch):
    plan, _ = containers
    original = digest(plan)
    for field in dataclasses.fields(PlanSummary):
        value = getattr(plan, field.name)
        if field.name == "flux_reference":
            altered = dataclasses.replace(plan, flux_reference="other-flux")
            assert digest(altered) != original
            continue
        elif value.dtype == jnp.bool_:
            changed = ~value
        else:
            changed = value + 1
        altered = eqx.tree_at(lambda p, key=field.name: getattr(p, key), plan, changed)
        assert digest(altered) != original, field.name
    for name in ("noise_corr_factors", "k_precision_assumed_ms"):
        altered = eqx.tree_at(lambda p, key=name: getattr(p, key), plan, None)
        assert digest(altered) != original
    monkeypatch.setattr(
        PlanSummary, "CONTRACT_VERSION", PlanSummary.CONTRACT_VERSION + 1
    )
    assert digest(plan) != original


def test_truth_fields_affect_digest(containers):
    _, truth = containers
    for changed in (
        Truth(truth.theta, truth.context, "toy/2"),
        Truth(truth.theta + 1, truth.context, truth.model_id),
        Truth(truth.theta, {"offset": jnp.array([1.0])}, truth.model_id),
        Truth(truth.theta, {}, truth.model_id),
    ):
        assert digest(changed) != digest(truth)
    assert encode(Truth(jnp.empty((0, 2)), {}, "empty")).startswith(MAGIC + b"\x11")


class OtherModule(eqx.Module):
    value: int


class ChildPlan(PlanSummary):
    pass


class ChildTruth(Truth):
    pass


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
@pytest.mark.parametrize("operation", [encode, digest, snapshot, to_json_value])
def test_refusal_paths(value, operation):
    with pytest.raises(TypeError, match=re.escape("weights[3]")):
        operation([None, None, None, value], name="weights")


@pytest.mark.parametrize("operation", [encode, digest, snapshot, to_json_value])
def test_large_non_string_map_key_refused(operation):
    with pytest.raises(TypeError, match=re.escape("data[0]")):
        operation([{10**5000: None}], name="data")


def test_science_subclasses_refused(containers):
    plan, truth = containers
    child_plan = ChildPlan(
        **{f.name: getattr(plan, f.name) for f in dataclasses.fields(plan)}
    )
    child_truth = ChildTruth(truth.theta, truth.context, truth.model_id)
    for value in (child_plan, child_truth):
        with pytest.raises(TypeError, match="predictive"):
            encode(value, name="predictive")


def test_truth_context_refusal_path():
    truth = Truth(
        jnp.ones((1, 1)), {"offset": jnp.array([1], dtype=jnp.uint32)}, "toy/1"
    )
    with pytest.raises(
        TypeError, match=re.escape("predictive.truth.context['offset']")
    ):
        encode(truth, name="predictive.truth")


@pytest.mark.parametrize(
    ("field", "value", "path"),
    [
        ("model_id", 7, ".model_id"),
        ("theta", 7, ".theta"),
        ("context", {"offset": 7}, ".context['offset']"),
        ("context", [], ".context"),
    ],
)
def test_truth_requires_string_and_array_nodes(containers, field, value, path):
    _, truth = containers
    if field == "model_id":
        altered = Truth(truth.theta, truth.context, value)
    else:
        altered = eqx.tree_at(lambda t: getattr(t, field), truth, value)
    with pytest.raises(TypeError, match=re.escape("data" + path)):
        encode(altered, name="data")


@pytest.mark.parametrize(
    "value", [float("nan"), np.float16("nan"), np.float32("nan"), np.float64("nan")]
)
@pytest.mark.parametrize("operation", [encode, digest, snapshot, to_json_value])
def test_scalar_nan_refused(value, operation):
    with pytest.raises(ValueError, match=re.escape("data['x']")):
        operation({"x": value}, name="data")


@pytest.mark.parametrize("value", ["\ud800", {"\ud800": 1}])
@pytest.mark.parametrize("operation", [encode, digest, snapshot, to_json_value])
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


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        -12,
        2**200,
        0.95,
        -0.0,
        float("inf"),
        -float("inf"),
        "\u00e9",
        "e\u0301",
        "",
        (),
        [],
        {},
        {"b": 1, "a": True},
        ("\u00e9", 0.95, -0.0),
        [None, {"a": [np.int64(1), np.float32(0.5)]}],
    ],
)
def test_json_round_trip(value):
    tagged = to_json_value(value)
    assert "codec" not in tagged
    loaded = json.loads(json.dumps(tagged, allow_nan=False))
    assert digest(from_json_value(loaded, CODEC)) == digest(value)


def test_json_tags_and_return_types():
    assert to_json_value(None) == {"t": "none"}
    assert to_json_value(True) == {"t": "bool", "v": True}
    assert to_json_value(-12) == {"t": "int", "v": "-12"}
    assert to_json_value(0.95) == {"t": "float", "v": "0x1.e666666666666p-1"}
    assert to_json_value("x") == {"t": "str", "v": "x"}
    tagged = to_json_value({"\u00e9": [], "a": ()})
    assert tagged == {
        "t": "map",
        "v": [["a", {"t": "seq", "v": []}], ["\u00e9", {"t": "seq", "v": []}]],
    }
    assert from_json_value(tagged, CODEC) == {"a": (), "\u00e9": ()}
    for value in (None, True, 12, 0.5, "x", (), {}):
        assert type(from_json_value(to_json_value(value), CODEC)) is type(value)


def test_json_string_scalars_round_trip_directly():
    for value in (np.str_("x"), {np.str_("x"): np.str_("y")}):
        tagged = to_json_value(value)
        assert digest(from_json_value(tagged, CODEC)) == digest(value)


def test_unknown_codec_refused():
    with pytest.raises(TypeError, match="record"):
        from_json_value({"t": "none"}, "spaceodyssey.inputs/2", name="record")


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"t": "other"},
        {"t": "bool", "v": 1},
        {"t": "none", "v": None},
        {"t": "int", "v": "01"},
        {"t": "int", "v": "1.0"},
        {"t": "float", "v": "invalid"},
        {"t": "map", "v": [[1, {"t": "none"}]]},
        {"t": "map", "v": [["a", {"t": "none"}], ["a", {"t": "none"}]]},
    ],
)
def test_invalid_json_refused(value):
    with pytest.raises(TypeError, match="record"):
        from_json_value(value, CODEC, name="record")


def test_json_nested_refusal_paths():
    with pytest.raises(TypeError, match=re.escape("record['x'][0]")):
        from_json_value(
            {"t": "map", "v": [["x", {"t": "seq", "v": [None]}]]},
            CODEC,
            name="record",
        )
    with pytest.raises(ValueError, match="record"):
        from_json_value({"t": "float", "v": "nan"}, CODEC, name="record")
    with pytest.raises(ValueError, match="record"):
        from_json_value({"t": "str", "v": "\ud800"}, CODEC, name="record")


@pytest.mark.parametrize("value", [np.array(1), jnp.array([1])])
def test_json_arrays_refused(value):
    with pytest.raises(TypeError, match=re.escape("data['x'][0]")):
        to_json_value({"x": [value]}, name="data")


def test_json_science_containers_refused(containers):
    for value in containers:
        with pytest.raises(TypeError, match="data"):
            to_json_value(value, name="data")


def test_snapshot_private_arrays_and_containers():
    array = np.array([1.5, -2.0], dtype=">f8")
    original = {"x": [array]}
    copied = snapshot(original)
    expected = digest(copied)
    assert copied is not original
    assert type(copied["x"]) is tuple
    assert copied["x"][0] is not array
    assert not np.shares_memory(copied["x"][0], array)
    assert not copied["x"][0].flags.writeable
    array[0] = 100
    original["x"].append(None)
    np.testing.assert_array_equal(copied["x"][0], [1.5, -2.0])
    assert digest(copied) == expected
    with pytest.raises(ValueError):
        copied["x"][0][0] = 1


def test_snapshot_science_containers(containers):
    plan, truth = containers
    assert snapshot(plan) is plan
    assert snapshot(plan.noise_sigma) is plan.noise_sigma
    copied = snapshot(truth)
    assert copied is not truth
    assert copied.context is not truth.context
    assert copied.theta is truth.theta
    assert copied.context["offset"] is truth.context["offset"]
    expected = digest(copied)
    truth.context["offset"] = truth.context["offset"] + 1
    assert digest(copied) == expected
    assert digest(truth) != expected


def test_snapshot_truth_numpy_context_preserves_dtype(containers):
    _, truth = containers
    for dtype in ("f8", "i8"):
        array = np.array([1], dtype=dtype)
        truth.context["offset"] = array
        copied = snapshot(truth)
        assert type(copied.context["offset"]) is np.ndarray
        assert copied.context["offset"].dtype == array.dtype
        assert not copied.context["offset"].flags.writeable
        assert not np.shares_memory(copied.context["offset"], array)
        assert digest(copied) == digest(truth)
        expected = digest(copied)
        array[0] = 2
        assert digest(copied) == expected
    empty = Truth(truth.theta, {}, truth.model_id)
    copied = snapshot(empty)
    assert copied is not empty
    assert copied.context is not empty.context
    assert digest(copied) == digest(empty)
