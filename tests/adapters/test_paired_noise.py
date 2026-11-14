"""Observation identities couple draws across divergent acquisition sequences."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core.enums import I64, Purpose, RecordStatus
from spaceodyssey.core.keys import asset_coordinate, draw_key
from spaceodyssey.core.kinds import realize_row
from spaceodyssey.core.records import record_none, unpack
from tests.adapters.test_observing_mode import _truth
from tests.studies.bracket_helpers import build


@pytest.fixture(scope="module")
def study():
    return build(run=False)[1]


@pytest.fixture(scope="module")
def truth(study):
    return _truth(study.world_source, 0)


def realize_measurement(
    study, truth, subject, channel_index, acquisition_index, group_ordinal
):
    """Realize with the handler's record key, including the study's coupling."""
    start = round(100 * study.channel_model.assets().context.clock.ticks_per_day)
    rec = dataclasses.replace(
        record_none(study.caps),
        subject=jnp.asarray(subject, I64),
        channel_index=jnp.asarray(channel_index, I64),
        channel=study.channels.id[channel_index],
        acquisition_index=jnp.asarray(acquisition_index, I64),
        group_ordinal=jnp.asarray(group_ordinal, I64),
        start_tick=jnp.asarray(start, I64),
        end_tick=jnp.asarray(start, I64)
        + study.channels.occupancy_ticks[channel_index],
        status=jnp.asarray(RecordStatus.COMPLETED, I64),
    )
    key = draw_key(
        jax.random.PRNGKey(0),
        rec,
        Purpose.MEASUREMENT,
        asset=asset_coordinate(study, rec.channel_index),
        branch=study.fork.branch_of(rec.acquisition_index),
        coupling=study.coupling,
    )
    return unpack(
        study.channel_model.measurement_template,
        realize_row(study, truth, rec, jnp.asarray(1.0), key),
    )


@pytest.mark.medium(reason="builds the fixture and realizes coupled spectral rows")
def test_entries_that_diverge_in_sequence_share_the_draw_of_an_observation(
    study, truth
):
    """Fails if a draw depends on acquisition count or exposure length."""
    t = study.channel_model.mode_table
    a = realize_measurement(
        study, truth, 0, t.channel("characterization", 0), 3, [1, 0]
    )
    b = realize_measurement(
        study, truth, 0, t.channel("characterization", 2), 7, [1, 0]
    )
    c = realize_measurement(
        study, truth, 0, t.channel("characterization", 0), 3, [1, 1]
    )
    mask = np.ones(a.counts.shape, bool)
    for row in (a, b, c):
        mask &= np.isfinite(row.true_variance) & (np.asarray(row.true_variance) > 0)
        mask &= np.isfinite(row.counts) & np.isfinite(row.true_counts)
    assert mask.any(), "the control must compare exposed, finite variates"

    def variates(row):
        return (
            np.asarray(row.counts)[mask] - np.asarray(row.true_counts)[mask]
        ) / np.sqrt(np.asarray(row.true_variance)[mask])

    za, zb, zc = map(variates, (a, b, c))
    np.testing.assert_allclose(za, zb, rtol=1e-12)
    assert not np.allclose(za, zc)
    assert len(np.unique(za)) == za.size, "true slots must have distinct draws"
    d = realize_measurement(
        study, truth, 0, t.channel("characterization", 0), 9, [9, 0]
    )
    np.testing.assert_allclose(za, variates(d), rtol=1e-12)


@pytest.mark.medium(reason="realizes coupled imaging rows at two durations")
def test_imaging_draws_use_the_exposed_band_ordinal(study, truth):
    """Fails if imaging omits ordinals or keys noise by duration or another band."""
    t = study.channel_model.mode_table
    rows = [
        realize_measurement(study, truth, 0, t.channel("detection", d), a, g)
        for d, a, g in ((0, 3, [1, 0]), (2, 7, [1, 9]), (0, 3, [2, 0]))
    ]
    mask = np.ones(rows[0].counts.shape, bool)
    for row in rows:
        mask &= np.isfinite(row.true_variance) & (np.asarray(row.true_variance) > 0)
        mask &= np.isfinite(row.counts) & np.isfinite(row.true_counts)
    assert mask.any(), "the control needs commonly detected slots"
    za, zb, zc = [
        (np.asarray(row.counts)[mask] - np.asarray(row.true_counts)[mask])
        / np.sqrt(np.asarray(row.true_variance)[mask])
        for row in rows
    ]
    np.testing.assert_allclose(za, zb, rtol=1e-12)
    assert not np.allclose(za, zc)


def test_bracket_declares_exposed_groups_and_measurement_coupling(study):
    """Fails if catalog rows consume ordinals or measurements retain sequence."""
    t = study.channel_model.mode_table
    assert study.caps.n_key_groups == t.n_bands
    assert study.coupling.shared == (
        (Purpose.MEASUREMENT, ("channel", "acquisition_index")),
    )
    for c in range(t.catalog_channel):
        expected = np.zeros(t.n_bands, bool)
        expected[t.band_index(t.modes[t.mode_of(c)].name)] = True
        np.testing.assert_array_equal(study.channels.key_groups[c], expected)
    np.testing.assert_array_equal(study.channels.key_groups[t.catalog_channel], False)
