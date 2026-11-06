"""Refresh protocol: pure checks (schedule, keys, slices, Study refusals)."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

from spaceodyssey.core import RefreshSchedule, init_carry, next_grid_tick, refit_key
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import Stage
from spaceodyssey.core.refresh import put_subject, slice_spec, take_subject
from spaceodyssey.studies import toy


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
