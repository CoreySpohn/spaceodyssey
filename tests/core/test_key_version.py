"""`key_version`: which asset coordinate a measurement draw is keyed by.

Version 1 keeps the asset coordinate at zero for every channel, so every key (and every
pinned history) is the one built before assets were a coordinate. Version 2 looks the
coordinate up from the layout (channel to lane to asset): the toy layout puts channel
12 on asset 1 and channels 10 and 11 on asset 0.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core.keys import KEY_VERSIONS, asset_coordinate
from spaceodyssey.studies import toy


def test_the_known_versions():
    assert KEY_VERSIONS == (1, 2)
    assert toy.survey_1().key_version == 1


@pytest.mark.parametrize("bad", [0, 3, -1, True, 2.0, "2", None])
def test_an_unknown_key_version_is_refused_at_construction(bad):
    """Fails if a version the lookup does not define builds a study."""
    with pytest.raises(ValueError, match="key_version must be one of"):
        toy.survey_1(key_version=bad)


def test_version_1_keys_every_channel_by_asset_zero():
    """Fails if version 1 moves any channel off the old key, or returns an array."""
    study = toy.survey_1()
    for index in range(study.caps.n_channels):
        coordinate = asset_coordinate(study, index)
        assert coordinate == 0
        assert type(coordinate) is int  # no traced op joins the compiled program


def test_version_2_looks_the_asset_up_through_the_lane():
    """Fails if the lookup skips a hop (channel to lane, lane to asset)."""
    study = toy.survey_1(key_version=2)
    got = [int(asset_coordinate(study, i)) for i in range(study.caps.n_channels)]
    assert got == [0, 0, 1]
    lane = np.asarray(study.channels.lane)
    lane_asset = np.asarray(study.assets.lane_asset)
    assert got == [int(lane_asset[lane[i]]) for i in range(len(got))]


def test_version_2_follows_a_changed_layout():
    """Control: the lookup reads the layout, not a constant table."""
    study = toy.survey_1(key_version=2)
    lanes = jnp.asarray([1, 1, 0])
    moved = dataclasses.replace(
        study, assets=dataclasses.replace(study.assets, lane_asset=lanes)
    )
    got = [int(asset_coordinate(moved, i)) for i in range(moved.caps.n_channels)]
    assert got == [1, 1, 0]


def test_version_2_works_under_a_traced_channel_index():
    """Fails if the lookup needs a concrete index (the handlers pass a traced one)."""
    study = toy.survey_1(key_version=2)
    got = jax.jit(lambda i: asset_coordinate(study, i))(jnp.asarray(2))
    assert int(got) == 1
