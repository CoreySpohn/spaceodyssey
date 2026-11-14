"""Only rows with astrometric information trigger a bracket refit."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from spaceodyssey.core import record_none
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import I64
from tests.studies.bracket_helpers import build


@pytest.fixture(scope="module")
def study():
    return build(run=False)[1]


def record_at(channel_index, *, detected=True):
    """A row with a channel index and a detected or null measurement payload."""
    return dataclasses.replace(
        record_none(Capacities()),
        channel_index=jnp.asarray(channel_index, I64),
        payload=jnp.asarray([detected, 0.0, 0.0, 0.0]),
    )


@eqx.filter_jit
def _refits_on_rows(observer, rows):
    return jax.vmap(observer.refits_on)(rows)


@pytest.mark.medium(reason="builds a bracket study and its declared channel assets")
def test_a_spectrograph_row_marks_no_refit_and_an_imaging_null_and_the_catalog_do(
    study,
):
    """Fails if a row without astrometry queues a refit, or a null or the catalog
    row skips one.
    """
    t, obs = study.channel_model.mode_table, study.observer
    recs = [
        record_at(t.channel("detection", 0), detected=False),
        record_at(t.catalog_channel),
        record_at(t.channel("characterization", 0)),
    ]
    assert [bool(obs.refits_on(r)) for r in recs] == [True, True, False]
    rows = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *recs)
    assert _refits_on_rows(obs, rows).tolist() == [True, True, False]
    assert [bool(obs.refits_on(record_at(c))) for c in range(t.n_channels)] == [
        True
    ] * len(t.imaging_channels) + [False] * len(t.spectrograph_channels) + [True]
