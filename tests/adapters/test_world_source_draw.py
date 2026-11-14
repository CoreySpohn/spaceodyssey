"""Draw-based world source: the slot and class rules of the file-based one.

Worlds come from `skyscapes.population.draw_population` on two hand-written G dwarfs
(no model-atmosphere download). The layout test also loads the stored fixture source
to compare tree structure and dtypes with it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.adapters.skyscapes import (
    ExoVistaWorldSource,
    PopulationModel,
    StarSpec,
)
from tests.studies.bracket_helpers import FIXTURE_BANDS, fixture_source

N_SLOTS = 4
STARS = tuple(
    StarSpec(f"HIP {k}", 1.0, 5.0 + k, 10.0, 20.0, 1.0 + 0.2 * k, 1.0) for k in range(2)
)
KEYS = jax.random.split(jax.random.PRNGKey(0), 6)
GEN = PopulationModel.exovista()


@pytest.fixture(scope="module")
def source():
    return ExoVistaWorldSource.from_draws(
        KEYS, STARS, n_slots=N_SLOTS, bands=FIXTURE_BANDS, generator=GEN
    )


@pytest.mark.medium(reason="six drawn worlds plus a stored fixture for layout parity")
def test_arrays_follow_the_stored_source_layout(source):
    """Fails if a leaf loses the [world, star, slot] layout or a slot count drifts.

    Absent slots must hold the inert filler (zero radius and albedo, one-year
    circular orbit) and the present ones must come first, as in `from_files`, whose
    structure and dtypes the arrays must match; `albedo` adds one entry per band.
    """
    shape = (len(KEYS), len(STARS), N_SLOTS)
    pop = source.population
    assert pop.period_d.shape == pop.present.shape == shape
    assert pop.albedo.shape == (*shape, len(FIXTURE_BANDS))
    assert (np.asarray(pop.albedo)[~np.asarray(pop.present)] == 0.0).all()
    present = np.asarray(pop.present)
    assert (np.diff(present.astype(int), axis=-1) <= 0).all()
    assert (np.asarray(pop.Rp_AU)[~present] == 0.0).all()
    assert (np.asarray(pop.period_d)[~present] == 365.25).all()
    assert source.epoch_jd == 2451545.0
    stored = fixture_source(1)
    assert jax.tree.structure(stored.population) == jax.tree.structure(pop)
    assert jax.tree.map(lambda x: x.dtype, stored.population) == jax.tree.map(
        lambda x: x.dtype, pop
    )
    assert stored.dropped.dtype == source.dropped.dtype


@pytest.mark.medium(reason="draws and fills three worlds of the fixture stars")
def test_worlds_are_keyed(source):
    """Fails if a world depends on another world's key.

    World `w` must not change when later keys are removed.
    """
    first = ExoVistaWorldSource.from_draws(
        KEYS[:3], STARS, n_slots=N_SLOTS, bands=FIXTURE_BANDS, generator=GEN
    )
    np.testing.assert_array_equal(
        np.asarray(first.population.period_d),
        np.asarray(source.population.period_d)[:3],
    )
    assert jnp.isfinite(source.population.tp_jd).all()
