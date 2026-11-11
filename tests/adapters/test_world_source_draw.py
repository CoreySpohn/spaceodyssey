"""Draw-based world source: the slot, class and drop rules of the file-based one.

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
    StarSpec,
    SupportRange,
)
from tests.studies.bracket_helpers import fixture_source

N_SLOTS = 4
STARS = tuple(
    StarSpec(f"HIP {k}", 1.0, 5.0 + k, 10.0, 20.0, 1.0 + 0.2 * k, 1.0, 1.0e3)
    for k in range(2)
)
KEYS = jax.random.split(jax.random.PRNGKey(0), 6)


@pytest.fixture(scope="module")
def source():
    return ExoVistaWorldSource.from_draws(KEYS, STARS, n_slots=N_SLOTS)


def test_arrays_follow_the_stored_source_layout(source):
    """Fails if a leaf loses the [world, star, slot] layout or a slot count drifts.

    Absent slots must hold the inert filler (zero radius, one-year circular orbit)
    and the present ones must come first, as in `from_files`, whose structure and
    dtypes the arrays must match.
    """
    shape = (len(KEYS), len(STARS), N_SLOTS)
    pop = source.population
    assert pop.period_d.shape == pop.present.shape == pop.klass.shape == shape
    present = np.asarray(pop.present)
    assert (np.diff(present.astype(int), axis=-1) <= 0).all()
    assert (np.asarray(pop.Rp_AU)[~present] == 0.0).all()
    assert (np.asarray(pop.period_d)[~present] == 365.25).all()
    assert (np.asarray(pop.klass)[~present] == 0).all()
    assert source.epoch_jd == 2451545.0
    stored = fixture_source(1)
    assert jax.tree.structure(stored.population) == jax.tree.structure(pop)
    assert jax.tree.map(lambda x: x.dtype, stored.population) == jax.tree.map(
        lambda x: x.dtype, pop
    )
    assert stored.occupancy.dtype == source.occupancy.dtype
    assert stored.dropped.dtype == source.dropped.dtype


def test_counts_account_for_every_drawn_class_one_planet(source):
    """Fails if a class-1 planet is lost between the draw and the slots.

    Class-1 planets in slots plus `dropped_class1` must equal the drawn class-1
    planets, and `occupancy` must equal the class flags of the slots.
    """
    occurrence = source.population.klass.sum(-1)
    np.testing.assert_array_equal(source.occupancy, occurrence)
    drawn = ExoVistaWorldSource.from_draws(
        KEYS, STARS, n_slots=30, support=SupportRange((0.0, 1e9), 1.1)
    )
    total_class1 = np.asarray(drawn.occupancy)
    np.testing.assert_array_equal(
        np.asarray(source.occupancy + source.dropped_class1), total_class1
    )
    total = np.asarray(drawn.population.present.sum(-1))
    np.testing.assert_array_equal(
        np.asarray(source.population.present.sum(-1) + source.dropped), total
    )


@pytest.mark.medium(reason="draws and fills two worlds of the fixture stars")
def test_support_range_drops_planets_and_worlds_are_keyed(source):
    """Fails if `support` is ignored or a world depends on another world's key.

    A narrow period range must leave only planets inside it, count the rest as
    dropped, and world `w` must not change when later keys are removed.
    """
    narrow = SupportRange((100.0, 400.0), 0.9)
    tight = ExoVistaWorldSource.from_draws(
        KEYS[:3], STARS, n_slots=N_SLOTS, support=narrow
    )
    period = np.asarray(tight.population.period_d)
    present = np.asarray(tight.population.present)
    assert ((period[present] >= 100.0) & (period[present] <= 400.0)).all()
    assert (np.asarray(tight.dropped) >= np.asarray(source.dropped)[:3]).all()
    wide = ExoVistaWorldSource.from_draws(
        KEYS[:3], STARS, n_slots=N_SLOTS, support=source.support
    )
    np.testing.assert_array_equal(
        np.asarray(wide.population.period_d), np.asarray(source.population.period_d)[:3]
    )
    assert jnp.isfinite(source.population.tp_jd).all()
