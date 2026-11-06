"""Refresh protocol: pure checks (schedule, keys, slices, Study refusals)."""

import types

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from spaceodyssey.core import RefreshSchedule, init_carry, next_grid_tick, refit_key
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import Capacity, EventKind, RecordStatus, Stage
from spaceodyssey.core.refresh import (
    _grid_position,
    put_subject,
    revision_set_digest,
    schedule_refresh,
    slice_spec,
    take_subject,
)
from spaceodyssey.studies import toy
from tests.core.helpers import one_world


def test_next_grid_tick_before_offset_is_ordinal_zero():
    s = RefreshSchedule(100, 50)
    assert [int(x) for x in next_grid_tick(s, 10, Stage.RELEASE)] == [50, 0]


def test_next_grid_tick_on_grid_stays_until_after_analyze():
    s = RefreshSchedule(100, 0)
    assert [int(x) for x in next_grid_tick(s, 200, Stage.RELEASE)] == [200, 2]
    assert [int(x) for x in next_grid_tick(s, 200, Stage.ANALYZE)] == [200, 2]
    # a result accepted after ANALYZE of a grid tick falls in the next checkpoint
    assert [int(x) for x in next_grid_tick(s, 200, Stage.ADMIT)] == [300, 3]


def test_next_grid_tick_between_ticks_rounds_up():
    s = RefreshSchedule(100, 0)
    assert [int(x) for x in next_grid_tick(s, 201, Stage.SETTLE)] == [300, 3]


def test_refit_key_has_no_audience_input_and_is_stable():
    root = jr.PRNGKey(3)
    digest = jnp.asarray(11, jnp.int64)
    a = refit_key(root, 1, digest)
    assert (a == refit_key(root, 1, digest)).all()
    # control: the key does move with the subject and with the revision set
    assert not (a == refit_key(root, 2, digest)).all()
    assert not (a == refit_key(root, 1, digest + 1)).all()


def test_take_put_subject_roundtrip_and_slice_size():
    study = toy.survey_refresh(100)
    names = study.observer.subject_leaves
    belief = study.observer.init(study.caps)
    spec = slice_spec(study)
    assert [n for n, _, _ in spec] == list(names)
    got = take_subject(belief, names, 1)
    for value, (_, tail, _) in zip(got, spec, strict=True):
        assert value.shape == tail  # a per-subject slice, not a full belief
    shifted = tuple(v + 5 for v in got)
    out = put_subject(belief, names, shifted, 1)
    again = take_subject(out, names, 1)
    assert all(bool((a == b).all()) for a, b in zip(again, shifted, strict=True))
    other = take_subject(out, names, 0)
    assert all(
        bool((a == b).all())
        for a, b in zip(other, take_subject(belief, names, 0), strict=True)
    )


def _build(schedule, **kw):
    observer = toy.ToyRefitMean()
    caps = toy.refresh_registry().lower(
        Capacities(rows_per_subject=8, refit_lanes=2), observer
    )
    return toy.survey_1(caps, observer=observer, refresh=schedule, **kw)


def test_study_without_refresh_has_no_refresh_leaves():
    study = toy.survey_1()
    truth = toy.draw_truth(jr.PRNGKey(0), study.caps)
    carry = eqx.filter_eval_shape(init_carry, study, truth)
    assert carry.refresh_status.size == 0
    with_refresh = toy.survey_refresh(100)
    truth = toy.draw_truth(jr.PRNGKey(0), with_refresh.caps)
    assert (
        eqx.filter_eval_shape(init_carry, with_refresh, truth).refresh_status.size > 0
    )


def test_refresh_requires_structural_capacity():
    ok = _build(RefreshSchedule(100, 0, 90, 91, toy.PRODUCT_KIND))
    assert ok.refresh.every_ticks == 100  # control: the builder is valid
    with pytest.raises(ValueError):
        toy.survey_1(refresh=RefreshSchedule(100, 0, 90, 91, toy.PRODUCT_KIND))


def test_refresh_refuses_equal_product_channels():
    with pytest.raises(ValueError):
        _build(RefreshSchedule(100, 0, 90, 90, toy.PRODUCT_KIND))


def test_refresh_refuses_unregistered_product_kind():
    with pytest.raises(ValueError):
        _build(RefreshSchedule(100, 0, 90, 91, 1))  # the measurement kind: not derived


def test_refresh_refuses_negative_ticks():
    with pytest.raises(ValueError):
        _build(RefreshSchedule(-1, 0, 90, 91, toy.PRODUCT_KIND))


def test_incremental_observer_without_subject_leaves_is_refused():
    with pytest.raises(ValueError):
        toy.survey_refresh(100, observer=toy.ToyRunningMean())


def _rows(capacity, triples):
    """Refit-row columns padded to `capacity` with sentinels that do not count."""
    n = len(triples)
    pad = capacity - n
    col = lambda k, fill: jnp.asarray(  # noqa: E731
        [t[k] for t in triples] + [fill] * pad, jnp.int64
    )
    ns = types.SimpleNamespace
    return ns(
        channel=col(0, 7),
        acquisition_index=col(1, 9),
        revision=col(2, 3),
        valid=jnp.arange(capacity) < n,
    )


def test_revision_set_digest_ignores_padding_and_names_the_set():
    triples = [(10, 0, 0)]
    digests = {int(revision_set_digest(_rows(c, triples))) for c in (4, 8, 16)}
    assert len(digests) == 1
    two = [(10, 0, 0), (11, 1, 2)]
    assert int(revision_set_digest(_rows(8, two))) == int(
        revision_set_digest(_rows(16, two))
    )
    # control: the digest still moves with the set, its order and its count
    base = int(revision_set_digest(_rows(8, two)))
    assert base != int(revision_set_digest(_rows(8, triples)))
    assert base != int(revision_set_digest(_rows(8, two[::-1])))
    assert base != int(revision_set_digest(_rows(8, [(10, 0, 0), (11, 1, 3)])))


def test_next_grid_tick_never_wraps_near_the_int64_limit():
    big = np.iinfo(np.int64).max
    s = RefreshSchedule(big, 0)
    tick, k, fits = _grid_position(s, 130, Stage.RELEASE)
    assert bool(fits) and int(tick) == big and int(k) == 1
    # the next grid point is not representable: saturate and say so
    tick, k, fits = _grid_position(s, big, Stage.ADMIT)
    assert not bool(fits) and int(tick) == big
    half = RefreshSchedule(2**62, 0)
    assert bool(_grid_position(half, 2**62, Stage.ANALYZE)[2])
    assert not bool(_grid_position(half, 2**62, Stage.ADMIT)[2])
    assert int(next_grid_tick(half, 2**62 + 1, Stage.SETTLE)[0]) == big
    # the ordinal increment past ANALYZE at the last tick must not wrap
    tick, k, fits = _grid_position(RefreshSchedule(1, 0), big, Stage.ADMIT)
    assert not bool(fits) and int(tick) == big and int(k) >= 0
    tick, k, fits = _grid_position(RefreshSchedule(1, 0), big, Stage.ANALYZE)
    assert bool(fits) and int(tick) == big and int(k) == big


def test_overflowing_checkpoint_sets_the_grid_tick_overflow():
    big = np.iinfo(np.int64).max
    study = toy.survey_refresh(big)
    truth, _ = one_world()
    carry = init_carry(study, truth)
    carry = eqx.tree_at(lambda c: c.tick, carry, jnp.asarray(130, jnp.int64))
    out = schedule_refresh(carry, jnp.asarray(True), study)
    assert bool(out.overflow)
    assert int(out.overflow_mask) & int(Capacity.GRID_TICK)
    assert not bool((out.ev.valid & (out.ev.kind == EventKind.REFRESH)).any())


def test_toy_kernel_and_host_fit_agree_bit_for_bit_on_cancelling_rows():
    """Fails if the two toy adapters sum in different orders."""
    study = toy.survey_refresh(100)
    kind = study.observer.measurement_kind
    rng = np.random.default_rng(4)
    big = rng.uniform(0.5e15, 1.5e15, 16)
    values = np.concatenate([big, -big + rng.uniform(0.0, 1.0, 16)])
    values = rng.permutation(values)
    n = values.size
    payload = np.zeros((n, 3))
    payload[:, 0] = values
    rows = types.SimpleNamespace(
        valid=np.ones(n, bool),
        kind=np.full(n, kind),
        status=np.full(n, int(RecordStatus.COMPLETED)),
        payload=payload,
    )
    as_jax = types.SimpleNamespace(**{k: jnp.asarray(v) for k, v in vars(rows).items()})
    kernel_mean = float(
        toy.toy_mean_kernel(study, types.SimpleNamespace(rows=as_jax))[0][0]
    )
    entry = types.SimpleNamespace(rows=rows)
    fit_mean = float(toy.toy_refit_fit(study, entry)[0][0])
    assert kernel_mean == fit_mean
    # control: a different reduction order moves the answer
    assert float(np.sum(values)) / n != fit_mean
