"""The slice 3a guards: what a capacity that is off must leave alone.

The pins in `golden_pin.json` and `hash_pin.json` hold every shipped history,
fingerprint and document fixed. These tests hold the machinery behind them: a leaf of a
feature that is off is absent from the canonical history and from the snapshot
fingerprint, a feature that is on adds it, the unconditional `refund_of` column keeps
its legacy sentinel, and `key_version` 2 moves only draws on a later asset. Each has a
negative control that breaks the thing it guards.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import spaceodyssey.core.snapshot as snapshot
from spaceodyssey.core import (
    Audience,
    HostExecutor,
    handlers,
    init_carry,
    snapshot_fingerprint,
    snapshot_view,
)
from spaceodyssey.core.battery.metamorphic import with_zero_time_asset
from spaceodyssey.core.capacities import OPTIONAL, Capacities
from spaceodyssey.core.executors import base
from spaceodyssey.core.keys import asset_coordinate
from spaceodyssey.studies import toy
from tests.core.corpus import frozen
from tests.core.helpers import baseline_host_run, one_world

N_STEPS = 40
ON_ASSET_0 = ((0, 10), (1, 11), (0, 10))
ON_ASSET_1 = ((0, 12), (1, 12), (0, 12))


def _by_path(carry, path):
    obj = carry
    for part in path.split(".")[1:]:
        obj = getattr(obj, part)
    return obj


def _leaves_of(carry, trace):
    layout = toy.survey_1().observer.layout(toy.survey_1().caps)
    handled = np.asarray(trace.handled)
    return base._canonical_leaves(carry, trace, handled, layout)


def _fingerprint(study, carry):
    """Eager, so a patched leaf list is read afresh (a jitted call is cached)."""
    view = snapshot_view(carry, Audience.POLICY, study.caps)
    return int(snapshot_fingerprint(view, study.observer.layout(study.caps))), view


# ---- the conditional-leaf tables --------------------------------------------------


def test_every_conditional_entry_names_an_inventoried_field():
    """Fails if a table entry drifts from the inventory (it would gate nothing)."""
    fields = {
        f"{group}.{name}"
        for group, names in base.CANONICAL_FIELDS.items()
        for name in names
    }
    assert set(base.CONDITIONAL_FIELDS) <= fields
    assert set(base.CONDITIONAL_GROUPS) <= set(base.CANONICAL_FIELDS)
    assert base.REFRESH_FIELDS == tuple(base.CONDITIONAL_FIELDS)
    assert base.REFRESH_GROUPS == tuple(base.CONDITIONAL_GROUPS)


def test_a_default_study_holds_every_gated_field_empty():
    """Fails if a gated field has size in a study whose feature is off."""
    study = toy.survey_1()
    truth, _ = one_world()
    carry = eqx.filter_jit(init_carry)(study, truth)
    for path, gate in base.CONDITIONAL_FIELDS.items():
        assert not gate(carry), path
        leaf = _by_path(carry, path)  # empty, or absent (None) for a column
        assert leaf is None or np.asarray(leaf).size == 0, path


@pytest.mark.medium  # A cold host run compiles the executor, over the 5 s unit budget.
def test_the_canonical_history_omits_the_gated_leaves():
    """Fails if a gated leaf enters the history of a study whose feature is off."""
    _, run, _ = baseline_host_run()
    leaves = _leaves_of(run.carry, run.trace)
    for path in base.CONDITIONAL_FIELDS:
        assert path not in leaves
    assert not any(k.startswith("carry.refresh.") for k in leaves)
    assert "carry.log.refund_of" in leaves  # an unconditional column stays


@pytest.mark.medium  # A cold host run compiles the executor, over the 5 s unit budget.
def test_control_an_open_gate_puts_the_empty_leaf_in_the_history(monkeypatch):
    """Control: the gates are what keep the leaves out, so opening them adds them."""
    _, run, _ = baseline_host_run()
    quiet = set(_leaves_of(run.carry, run.trace))
    # an absent (None) column has no empty leaf to put in the history
    held = {p for p in base.CONDITIONAL_FIELDS if _by_path(run.carry, p) is not None}
    opened = {
        p: (lambda carry: True) if p in held else gate
        for p, gate in base.CONDITIONAL_FIELDS.items()
    }
    monkeypatch.setattr(base, "CONDITIONAL_FIELDS", opened)
    loud = _leaves_of(run.carry, run.trace)
    assert held <= set(loud) - quiet


# ---- the snapshot fingerprint -----------------------------------------------------


def test_an_empty_conditional_view_leaf_is_not_hashed(monkeypatch):
    """Fails if a leaf of size zero is listed: its tag and count move the digest."""
    study = toy.survey_1()
    truth, _ = one_world()
    carry = eqx.filter_jit(init_carry)(study, truth)
    base_fp, _ = _fingerprint(study, carry)
    empty = jnp.zeros((0,), jnp.int64)
    monkeypatch.setattr(
        snapshot, "conditional_view_leaves", lambda v: (v.log.product, empty)
    )
    assert _fingerprint(study, carry)[0] == base_fp


def test_control_a_nonempty_conditional_view_leaf_moves_the_digest(monkeypatch):
    """Control: a leaf with content, by contrast, is hashed and moves the digest."""
    study = toy.survey_1()
    truth, _ = one_world()
    carry = eqx.filter_jit(init_carry)(study, truth)
    base_fp, _ = _fingerprint(study, carry)
    full = jnp.ones((3,), jnp.int64)
    monkeypatch.setattr(snapshot, "conditional_view_leaves", lambda v: (full,))
    assert _fingerprint(study, carry)[0] != base_fp


@pytest.mark.medium  # A cold host run compiles the executor, over the 5 s unit budget.
def test_refund_of_stays_unconditional_with_its_legacy_sentinel():
    """Fails if the column is masked, zeroed or dropped: a legacy fingerprint moves."""
    study, run, _ = baseline_host_run()
    valid = np.asarray(run.carry.log.valid)
    assert valid.any()
    assert np.all(np.asarray(run.carry.log.refund_of) == -1)
    base_fp, view = _fingerprint(study, run.carry)
    shown = np.asarray(view.log.refund_of)[np.asarray(view.log.valid)]
    assert shown.size and np.all(shown == -1)
    assert "refund_of" in snapshot.LOG_INT_FIELDS
    zeroed = eqx.tree_at(
        lambda v: v.log.refund_of, view, jnp.zeros_like(view.log.refund_of)
    )
    layout = study.observer.layout(study.caps)
    assert int(snapshot_fingerprint(zeroed, layout)) != base_fp  # the control


def test_the_added_asset_keeps_every_existing_asset_coordinate():
    """Fails if inserting the idle facility renumbers a channel's asset."""
    study = toy.survey_1(key_version=2)
    more = with_zero_time_asset(study)
    assert more.key_version == 2
    for index in range(study.caps.n_channels):
        assert int(asset_coordinate(more, index)) == int(asset_coordinate(study, index))


# ---- capacities -------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"awards": 1},
        {"refunds": "yes"},
        {"snapshot_balances": 0},
        {"n_commitments": -1},
        {"n_commitments": True},
        {"n_commitments": 1.0},
    ],
)
def test_a_malformed_slice_3a_capacity_is_refused(bad):
    """Fails if a capacity that is not a bool or a nonnegative int builds."""
    with pytest.raises(ValueError):
        Capacities(**bad)


def test_the_slice_3a_capacities_default_to_off():
    """Fails if a default capacity turns a feature on (pinned studies would move)."""
    caps = Capacities()
    assert (caps.awards, caps.refunds, caps.n_commitments, caps.snapshot_balances) == (
        False,
        False,
        0,
        False,
    )
    marked = [f.name for f in dataclasses.fields(Capacities) if f.metadata == OPTIONAL]
    assert marked == ["awards", "refunds", "n_commitments", "snapshot_balances"]


# ---- key_version: runs ------------------------------------------------------------


def _run(plan, key_version):
    study = toy.survey_1(policies=(frozen(plan=plan),), key_version=key_version)
    truth, root = one_world()
    return HostExecutor().run(study, truth, root, N_STEPS)


def _state(run):
    return [np.asarray(x) for x in jax.tree.leaves((run.carry.belief, run.carry.log))]


def _same(a, b):
    return len(a) == len(b) and all(
        x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b, strict=True)
    )


@pytest.fixture(scope="module")
def key_runs():
    """The four runs the key-version tests compare, taken once: two compiles, no more.

    The plans differ in data only, so the runs of one key version share a compiled
    program; the version is static, so the two versions compile apart.
    """
    return {
        (plan, version): _state(_run(plan, version))
        for plan in (ON_ASSET_0, ON_ASSET_1)
        for version in (1, 2)
    }


@pytest.mark.medium  # The shared runs compile the executor twice, over the unit budget.
def test_a_draw_on_asset_zero_is_unchanged_by_key_version_2(key_runs):
    """Fails if version 2 moves a draw on asset 0: those keys stay the old ones."""
    assert _same(key_runs[ON_ASSET_0, 1], key_runs[ON_ASSET_0, 2])


@pytest.mark.medium  # The shared runs compile the executor twice, over the unit budget.
def test_a_draw_on_a_later_asset_moves_under_key_version_2(key_runs):
    """Fails if version 2 keys asset 1 as before (the version would do nothing)."""
    assert not _same(key_runs[ON_ASSET_1, 1], key_runs[ON_ASSET_1, 2])


@pytest.mark.medium  # The shared runs compile the executor twice, over the unit budget.
def test_control_a_lookup_that_returns_zero_leaves_the_draws_alone(
    key_runs, monkeypatch
):
    """Control: with the lookup pinned to zero, version 2 reproduces version 1."""
    monkeypatch.setattr(handlers, "asset_coordinate", lambda study, index: 0)
    jax.clear_caches()  # the compiled program closed over the real lookup
    try:
        assert _same(key_runs[ON_ASSET_1, 1], _state(_run(ON_ASSET_1, 2)))
    finally:
        jax.clear_caches()
