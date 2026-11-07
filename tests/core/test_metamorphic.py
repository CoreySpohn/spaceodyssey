"""Edits that must change nothing, and the components that make them matter."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from spaceodyssey.core import battery, handlers
from spaceodyssey.core.battery.metamorphic import with_zero_time_asset
from spaceodyssey.core.enums import I64
from spaceodyssey.core.keys import asset_coordinate
from tests.core.battery_helpers import PLAN, SelectingPolicy, survey
from tests.core.corpus import frozen
from tests.core.helpers import worlds

N_STEPS = 40
ID_MAP = {10: 30, 11: 20, 12: 10}
EPOCH_SHIFT = 1000


@pytest.fixture(scope="module")
def inputs():
    return worlds(2)


def relabeled(study, policy):
    ids = jnp.asarray([ID_MAP[int(i)] for i in study.channels.id], I64)
    out = eqx.tree_at(lambda s: s.channels.id, study, ids)
    return dataclasses.replace(
        out, agenda=dataclasses.replace(out.agenda, policies=(policy,))
    )


def test_relabel_ids(inputs):
    truths, roots = inputs
    plan = tuple((subject, ID_MAP[cid]) for subject, cid in PLAN)
    result = battery.relabel_ids(
        survey(), relabeled(survey(), frozen(plan=plan)), ID_MAP, truths, roots, N_STEPS
    )
    assert result, result.diagnostic
    assert result.detail["n_records"] >= 10


def test_relabel_ids_catches_a_policy_that_orders_ids(inputs):
    """Negative control: choosing the channel with the smallest id."""
    truths, roots = inputs
    policy = SelectingPolicy(rule="smallest_id")
    result = battery.relabel_ids(
        survey(policy), relabeled(survey(), policy), ID_MAP, truths, roots, N_STEPS
    )
    assert not result
    assert "changed under relabeling" in result.diagnostic


def test_relabel_ids_refuses_an_identity_map(inputs):
    truths, roots = inputs
    with pytest.raises(ValueError):
        battery.relabel_ids(survey(), survey(), {10: 10}, truths, roots, N_STEPS)


def test_shift_epoch(inputs):
    truths, roots = inputs
    result = battery.shift_epoch(survey(), truths, roots, N_STEPS, EPOCH_SHIFT)
    assert result, result.diagnostic
    assert result.detail["n_records"] >= 10


def test_shift_epoch_catches_a_component_that_reads_absolute_time(inputs):
    """Negative control: a policy that acts on the tick itself."""
    truths, roots = inputs
    study = survey(SelectingPolicy(rule="late_tick"))
    result = battery.shift_epoch(study, truths, roots, N_STEPS, EPOCH_SHIFT)
    assert not result
    assert "changed under the shift" in result.diagnostic


def test_zero_time_asset(inputs):
    truths, roots = inputs
    result = battery.zero_time_asset(survey(), truths, roots, N_STEPS)
    assert result, result.diagnostic
    assert result.detail["n_records"] >= 10


def test_zero_time_asset_catches_a_policy_that_counts_assets(inputs):
    """Negative control: the subject observed depends on the number of assets."""
    truths, roots = inputs
    study = survey(SelectingPolicy(rule="asset_parity"))
    result = battery.zero_time_asset(study, truths, roots, N_STEPS)
    assert not result
    assert "trace or the records changed" in result.diagnostic


# channel 12 sits on lane 2, which is on asset 1: these draws carry a nonzero asset
# coordinate under key version 2
ASSET_1_PLAN = ((0, 12),) * 5


def test_zero_time_asset_with_asset_1_draws_under_key_version_2(inputs):
    """Adding an idle facility moves no asset-1 key, payload or belief."""
    truths, roots = inputs
    study = survey(frozen(plan=ASSET_1_PLAN), key_version=2)
    result = battery.zero_time_asset(study, truths, roots, N_STEPS)
    assert result, result.diagnostic
    assert result.detail["n_records"] >= 10


def test_zero_time_asset_catches_a_key_that_counts_assets(inputs, monkeypatch):
    """Negative control: a lookup of `asset * n_assets` moves asset-1 keys when an
    asset is added, which the relation must report.
    """

    def counting(study, channel_index):
        lane = study.channels.lane[channel_index]
        return study.assets.lane_asset[lane].astype(I64) * study.caps.n_assets

    truths, roots = inputs
    study = survey(frozen(plan=ASSET_1_PLAN), key_version=2)
    monkeypatch.setattr(handlers, "asset_coordinate", counting)
    jax.clear_caches()  # a compiled program closed over the real lookup
    result = battery.zero_time_asset(study, truths, roots, N_STEPS)
    assert not result
    assert "trace or the records changed" in result.diagnostic


def test_the_added_asset_keeps_every_existing_asset_coordinate():
    """Fails if inserting the idle facility renumbers a channel's asset."""
    study = survey(key_version=2)
    more = with_zero_time_asset(study)
    assert more.key_version == 2
    for index in range(study.caps.n_channels):
        assert int(asset_coordinate(more, index)) == int(asset_coordinate(study, index))


def test_the_added_asset_holds_nothing():
    more = with_zero_time_asset(survey())
    assert more.caps.n_assets == survey().caps.n_assets + 1
    assert float(more.budget[survey().caps.n_assets].sum()) == 0.0
