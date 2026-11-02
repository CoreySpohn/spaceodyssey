"""Draw keys depend on coordinates rather than step or batch position."""

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from spaceodyssey import core as sch
from spaceodyssey.core.enums import Purpose
from spaceodyssey.core.keys import COORDS, MASK32, draw_key, fold64, key_for, policy_key

ROOT = jr.PRNGKey(0)
BASE = {name: 1 for name in COORDS}


def same(a, b):
    return bool(jnp.array_equal(a, b))


def test_mask_is_32_bits():
    assert MASK32 == (1 << 32) - 1


def test_fold64_separates_high_word():
    f = jax.jit(fold64)
    assert not same(f(ROOT, 7), f(ROOT, 7 + 2**32))
    # plain fold_in drops the high word, which is why fold64 exists
    high = jnp.asarray(7 + 2**32, dtype=jnp.int64)
    assert same(jr.fold_in(ROOT, 7), jr.fold_in(ROOT, high))


def test_fold64_is_deterministic_and_value_sensitive():
    f = jax.jit(fold64)
    assert same(f(ROOT, 12345), f(ROOT, 12345))
    assert not same(f(ROOT, 12345), f(ROOT, 12346))
    assert not same(f(ROOT, -1), f(ROOT, 1))


def test_fold64_folds_low_word_before_high_word():
    expected = jr.fold_in(jr.fold_in(ROOT, jnp.uint32(7)), jnp.uint32(3))
    assert same(fold64(ROOT, 7 + 3 * 2**32), expected)


def test_fold64_jit_vmap_preserves_signed_64_bit_words():
    values = jnp.asarray([0, 7 + 2**32, -1, -(2**63)], dtype=jnp.int64)
    expected = jnp.stack(
        [
            jr.fold_in(jr.fold_in(ROOT, jnp.uint32(lo)), jnp.uint32(hi))
            for lo, hi in [(0, 0), (7, 1), (MASK32, MASK32), (0, 2**31)]
        ]
    )
    actual = jax.jit(jax.vmap(lambda x: fold64(ROOT, x)))(values)
    assert same(actual, expected)


@pytest.mark.parametrize("name", COORDS)
def test_each_coordinate_changes_key(name):
    base = key_for(ROOT, **BASE)
    bumped = key_for(ROOT, **{**BASE, name: BASE[name] + 1})
    assert not same(base, bumped)


def test_coordinates_default_to_zero():
    assert same(
        key_for(ROOT, subject=3),
        key_for(ROOT, **{**{n: 0 for n in COORDS}, "subject": 3}),
    )


def test_unknown_coordinate_is_rejected():
    with pytest.raises((TypeError, ValueError)):
        key_for(ROOT, step=3)


def test_batch_position_never_enters():
    ids = jnp.asarray([7, 3, 9, 1])
    fwd = jax.vmap(lambda i: key_for(ROOT, world=i))(ids)
    revision = jax.vmap(lambda i: key_for(ROOT, world=i))(ids[::-1])
    assert same(fwd[::-1], revision)
    single = key_for(ROOT, world=3)
    assert same(fwd[1], single)


def _rec(**kw):
    caps = sch.Capacities()
    r = sch.record_none(caps)
    fields = {k: jnp.asarray(v, dtype=getattr(r, k).dtype) for k, v in kw.items()}
    return dataclasses.replace(r, **fields)


def test_draw_key_ignores_rec_value():
    a = _rec(
        subject=1,
        channel=10,
        acquisition_index=2,
        payload=jnp.asarray([0.1, 0.2, 0.3, 0.4]),
    )
    b = _rec(
        subject=1,
        channel=10,
        acquisition_index=2,
        payload=jnp.asarray([9.0, 8.0, 7.0, 6.0]),
    )
    assert same(
        draw_key(ROOT, a, Purpose.MEASUREMENT), draw_key(ROOT, b, Purpose.MEASUREMENT)
    )


def test_draw_key_reads_the_record_coordinates():
    base = _rec(
        subject=1, channel=10, acquisition_index=2, phase=0, revision=0, status=0
    )
    k0 = draw_key(ROOT, base, Purpose.MEASUREMENT)
    for name in (
        "subject",
        "channel",
        "acquisition_index",
        "phase",
        "revision",
        "status",
    ):
        bumped = dataclasses.replace(base, **{name: getattr(base, name) + 1})
        assert not same(k0, draw_key(ROOT, bumped, Purpose.MEASUREMENT)), name
    assert not same(k0, draw_key(ROOT, base, Purpose.INFERENCE))


def test_draw_key_matches_key_for():
    r = _rec(subject=2, channel=11, acquisition_index=3, phase=1, revision=0, status=2)
    expected = key_for(
        ROOT,
        world=0,
        subject=2,
        channel=11,
        acquisition_index=3,
        phase=1,
        lineage=0,
        purpose=Purpose.MEASUREMENT,
        revision=0,
        status=2,
    )
    assert same(draw_key(ROOT, r, Purpose.MEASUREMENT), expected)


def test_policy_key_by_n_dec():
    assert not same(policy_key(ROOT, 3), policy_key(ROOT, 4))
    assert same(policy_key(ROOT, 3), fold64(fold64(ROOT, Purpose.POLICY), 3))
