"""The storage codec packs named module leaves into payload columns."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from spaceodyssey.core.enums import EVERYONE, F64, I64, Audience
from spaceodyssey.core.records import (
    RecordKind,
    column_names,
    layout_size,
    pack,
    unpack,
)


class _Row(eqx.Module):
    time_d: jax.Array  # []
    flag: jax.Array  # I64[D]
    counts: jax.Array  # F64[D, B]


class _Outer(eqx.Module):
    present: jax.Array  # F64[2]
    inner: _Row


def _template(d, b):
    s = jax.ShapeDtypeStruct
    return _Row(s((), F64), s((d,), I64), s((d, b), F64))


@pytest.mark.parametrize("d, b", [(1, 1), (3, 2), (4, 5)])
def test_unpack_inverts_pack_for_any_axis_sizes(d, b):
    """Fails if a leaf lands at another leaf's offset or loses its dtype."""
    row = _Row(
        jnp.asarray(2.5),
        jnp.arange(d, dtype=I64),
        jnp.arange(d * b, dtype=F64).reshape(d, b),
    )
    flat = pack(row)
    assert flat.shape == (layout_size(_template(d, b)),) == (1 + d + d * b,)
    back = unpack(_template(d, b), flat)
    for x, y in zip(jax.tree.leaves(back), jax.tree.leaves(row), strict=True):
        assert x.dtype == y.dtype and bool(jnp.all(x == y))


def test_unpack_keeps_leading_batch_axes():
    """Fails if a batch of payloads is unpacked as one row."""
    flat = jnp.stack(
        [pack(_Row(jnp.asarray(1.0), jnp.zeros(2, I64), jnp.ones((2, 3))))] * 4
    )
    assert unpack(_template(2, 3), flat).counts.shape == (4, 2, 3)


def test_column_names_follow_field_order_then_index():
    """Fails if a generated name drops an index or reorders fields."""
    assert column_names(_template(2, 2)) == (
        "time_d",
        "flag_0",
        "flag_1",
        "counts_0_0",
        "counts_0_1",
        "counts_1_0",
        "counts_1_1",
    )


def test_a_nested_module_is_walked_in_place_with_dotted_paths():
    """Fails if a nested field is skipped, reordered or gets an indexed key."""
    t = _Outer(jax.ShapeDtypeStruct((2,), F64), _template(1, 1))
    assert column_names(t) == (
        "present_0",
        "present_1",
        "inner.time_d",
        "inner.flag_0",
        "inner.counts_0_0",
    )
    seen = []
    kind = RecordKind.from_template("o", t, lambda p: seen.append(p) or EVERYONE)
    assert sorted(set(seen)) == [
        "inner.counts",
        "inner.flag",
        "inner.time_d",
        "present",
    ]
    assert kind.width == layout_size(t) == 5


def test_a_kind_from_a_template_gives_each_field_its_audience():
    """Fails if an audience attaches to the wrong column."""
    kind = RecordKind.from_template(
        "row",
        _template(2, 1),
        lambda f: Audience.SCORER if f == "flag" else EVERYONE,
    )
    by_name = {c.name: c.audience for c in kind.columns}
    assert by_name["flag_1"] == Audience.SCORER and by_name["counts_1_0"] == EVERYONE
    assert kind.width == layout_size(_template(2, 1))


def test_unpack_refuses_a_payload_narrower_than_the_template():
    """Fails if a short payload is read with silent zero padding."""
    with pytest.raises(ValueError, match="columns"):
        unpack(_template(2, 2), jnp.zeros(3))
