"""Draw keys depend on coordinates rather than step or batch position."""

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from spaceodyssey import core as sch
from spaceodyssey.core.enums import Purpose
from spaceodyssey.core.keys import (
    COORDS,
    MASK32,
    OPTIONAL_COORDS,
    draw_key,
    fold64,
    key_for,
    policy_key,
)

ROOT = jr.PRNGKey(0)
BASE = {name: 1 for name in COORDS}
OPT = OPTIONAL_COORDS
BASE_WITHOUT_OPTIONAL = {n: 1 for n in COORDS if n not in OPT}


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


def test_policy_key_follows_meaning():
    base = policy_key(ROOT, 0, 130, 0)
    assert same(
        base, fold64(fold64(fold64(fold64(fold64(ROOT, Purpose.POLICY), 0), 0), 130), 0)
    )
    assert not same(base, policy_key(ROOT, 0, 131, 0))
    assert not same(base, policy_key(ROOT, 0, 130, 1))
    assert not same(base, policy_key(ROOT, 1, 130, 0))
    assert not same(base, policy_key(ROOT, 0, 130, 0, lineage=1))


def test_coupling_shares_declared_coordinates():
    shared = sch.Coupling(shared=((Purpose.POLICY, ("lineage",)),))
    a = policy_key(ROOT, 0, 130, 0, lineage=1, coupling=shared)
    b = policy_key(ROOT, 0, 130, 0, lineage=2, coupling=shared)
    assert same(a, b)
    assert not same(
        policy_key(ROOT, 0, 130, 0, lineage=1), policy_key(ROOT, 0, 130, 0, lineage=2)
    )
    # the declaration is per purpose: measurement draws still separate lineages
    rec = _rec(subject=1, channel=10, acquisition_index=2)
    m1 = draw_key(ROOT, rec, Purpose.MEASUREMENT, lineage=1, coupling=shared)
    m2 = draw_key(ROOT, rec, Purpose.MEASUREMENT, lineage=2, coupling=shared)
    assert not same(m1, m2)
    both = sch.Coupling(shared=((Purpose.MEASUREMENT, ("lineage", "branch")),))
    assert same(
        draw_key(ROOT, rec, Purpose.MEASUREMENT, lineage=1, branch=3, coupling=both),
        draw_key(ROOT, rec, Purpose.MEASUREMENT, lineage=2, branch=4, coupling=both),
    )


def test_coupling_rejects_unknown_names():
    with pytest.raises(ValueError):
        sch.Coupling(shared=((Purpose.POLICY, ("step",)),))
    with pytest.raises(ValueError):
        sch.Coupling(shared=((Purpose.POLICY, ("tick",)),))


def test_asset_and_branch_separate_draws_only_when_nonzero():
    base = key_for(ROOT, **BASE_WITHOUT_OPTIONAL)
    assert same(base, key_for(ROOT, **BASE_WITHOUT_OPTIONAL, asset=0, branch=0))
    assert not same(base, key_for(ROOT, **BASE_WITHOUT_OPTIONAL, asset=1))
    assert not same(base, key_for(ROOT, **BASE_WITHOUT_OPTIONAL, branch=1))
    assert not same(
        key_for(ROOT, **BASE_WITHOUT_OPTIONAL, asset=1),
        key_for(ROOT, **BASE_WITHOUT_OPTIONAL, asset=2),
    )


def test_keys_trace_under_jit():
    traced = jax.jit(lambda c: key_for(ROOT, **c))
    coords = {n: jnp.asarray(i + 1, dtype=jnp.int64) for i, n in enumerate(COORDS)}
    eager = key_for(ROOT, **{n: i + 1 for i, n in enumerate(COORDS)})
    assert same(traced(coords), eager)
    zero_optional = {**coords, "asset": jnp.asarray(0), "branch": jnp.asarray(0)}
    assert same(
        traced(zero_optional),
        key_for(
            ROOT,
            **{n: v for n, v in zip(COORDS, range(1, 12), strict=True) if n not in OPT},
        ),
    )
    pk = jax.jit(policy_key)
    assert same(
        pk(ROOT, jnp.asarray(1), jnp.asarray(130), jnp.asarray(2)),
        policy_key(ROOT, 1, 130, 2),
    )
    rec = _rec(subject=2, channel=11, acquisition_index=3)
    dk = jax.jit(lambda r: draw_key(ROOT, r, Purpose.MEASUREMENT))
    assert same(dk(rec), draw_key(ROOT, rec, Purpose.MEASUREMENT))


def test_inserting_acquisition_changes_no_other_draw():
    """Draws of (subject, k) do not move when another subject gains an acquisition."""

    def keys(counts):
        return {
            (s, k): key_for(ROOT, subject=s, channel=10, acquisition_index=k)
            for s, n in counts.items()
            for k in range(n)
        }

    before, after = keys({0: 3, 1: 2}), keys({0: 3, 1: 3, 2: 1})
    assert all(same(before[k], after[k]) for k in before)
    assert len({tuple(map(int, jr.key_data(v))) for v in after.values()}) == len(after)
