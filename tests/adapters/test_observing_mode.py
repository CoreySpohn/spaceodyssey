"""Stored-universe truth and the observing-mode channel, on the checked-in fixture.

The fixture (`tests/fixtures/exovista`) is one ExoVista universe of two stars.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from orbix.kepler import markley_solve_trig
from skyscapes.io import from_exovista

from spaceodyssey.adapters.jaxedith import column_index, visit_columns
from spaceodyssey.adapters.photomancy import CATALOG_FIELDS
from spaceodyssey.adapters.skyscapes import geometry, population_system
from spaceodyssey.core.kinds import RealizeContext
from tests.studies.bracket_helpers import (
    N_SLOTS,
    fixture_channel,
    fixture_files,
    fixture_source,
)

pytestmark = pytest.mark.medium(reason="loads the fixture and compiles the channel")

DETECT_MODE = 4  # 160000 s


@pytest.fixture(scope="module")
def source():
    return fixture_source()


@pytest.fixture(scope="module")
def channel(source):
    return fixture_channel(source)


def _context(channel, start_d, length_s):
    clock = channel.assets().context.clock
    start = int(start_d * clock.ticks_per_day)
    settle = int(channel.assets().context.settle_ticks)
    end = start + settle + int(length_s * 1000)
    zero = jnp.asarray(0)
    return RealizeContext(
        jnp.asarray(start), jnp.asarray(end), zero, jnp.asarray(1.0), zero, zero
    )


def _truth(source, world):
    return jax.tree_util.tree_map(lambda x: x[world], source.population)


def _one(system, k):
    return eqx.tree_at(lambda s: s.planets, system, (system.planets[k],))


def test_stored_positions_survive_the_slot_conversion(source):
    """Slot sky separations equal those of the stored planets to 1e-9 arcsec.

    The stored system carries a tabulated float32 Kepler solver; both sides use the
    exact one here, so the comparison isolates the element conversion. Over the
    same epochs the tabulated solver stays within 1e-6 arcsec.
    """
    stored = from_exovista(str(fixture_files()[0]), load_disk=False)
    exact = dataclasses.replace(stored, trig_solver=markley_solve_trig)
    epochs = source.epoch_jd + jnp.asarray([0.0, 91.3, 400.7, 20000.0])
    truth = _truth(source, 0)
    system = population_system(source, truth, 0)
    ours = [
        np.asarray(geometry(_one(system, k), epochs, 1000.0).separation_arcsec)
        for k in range(N_SLOTS)
        if bool(truth.present[0, k])
    ]

    def sep(system_, j):
        return np.asarray(geometry(_one(system_, j), epochs, 1000.0).separation_arcsec)

    assert ours, "the first star holds no planet in its slots"
    for s in ours:
        assert (
            min(np.max(np.abs(s - sep(exact, j))) for j in range(len(stored.planets)))
            < 1e-9
        )
        assert (
            min(np.max(np.abs(s - sep(stored, j))) for j in range(len(stored.planets)))
            < 1e-6
        )


def test_absent_slot_never_detects(source, channel):
    """A visit never reports a detection in a slot that holds no planet."""
    truth = _truth(source, 0)
    context = _context(channel, 100.0, 160000.0)
    keys = jax.random.split(jax.random.key(3), 24)
    rows = jax.vmap(
        lambda k: channel.realize(
            truth, 0, DETECT_MODE, 1.0, k, None, None, context=context
        )
    )(keys)
    names = visit_columns(N_SLOTS)
    absent = [s for s in range(N_SLOTS) if not bool(truth.present[0, s])]
    assert absent, "the first star fills every slot"
    for slot in absent:
        assert float(rows[:, column_index(names, f"detected_{slot}")].max()) == 0.0


def test_a_star_row_ignores_the_other_stars_truth(source, channel):
    """Replacing star 1's planets leaves star 0's visit row unchanged."""
    truth = _truth(source, 0)
    other = _truth(source, 1)
    mixed = jax.tree_util.tree_map(
        lambda a, b: a.at[1].set(b[1]) if a.ndim >= 2 else a, truth, other
    )
    context = _context(channel, 100.0, 160000.0)
    key = jax.random.key(5)

    def row(t):
        return channel.realize(t, 0, DETECT_MODE, 1.0, key, None, None, context=context)

    np.testing.assert_array_equal(np.asarray(row(truth)), np.asarray(row(mixed)))


def test_the_catalog_row_follows_the_declared_layout(source, channel):
    """The catalog row has one block per slot in `CATALOG_FIELDS` order."""
    truth = _truth(source, 0)
    row = channel.realize(
        truth,
        0,
        channel.catalog_mode,
        1.0,
        jax.random.key(0),
        None,
        None,
        context=_context(channel, 0.0, 1.0),
    )
    cells = np.asarray(row)[: N_SLOTS * len(CATALOG_FIELDS)].reshape(N_SLOTS, -1)
    names = [f for f, _ in CATALOG_FIELDS]
    present = np.asarray(truth.present[0])
    np.testing.assert_array_equal(cells[:, names.index("present")], present)
    np.testing.assert_allclose(
        np.exp(cells[present, names.index("ln_period")]),
        np.asarray(truth.period_d[0])[present],
        rtol=1e-12,
    )
    np.testing.assert_array_equal(
        cells[:, names.index("class1")], np.asarray(truth.klass[0])
    )


def test_realized_detection_matches_the_stated_probability(source, channel):
    """Detection frequency over 2000 keys is within 3 standard errors of the model."""
    context = _context(channel, 100.0, 160000.0)
    interval = channel._interval(context, settle=True)[0]
    time_d = float(interval.midpoint_jd) - float(source.epoch_jd)
    names = visit_columns(N_SLOTS)
    checked = 0
    for world in range(2):
        truth = _truth(source, world)
        for star in range(len(source.stars)):
            cells = np.asarray(
                channel.realize(
                    truth,
                    star,
                    channel.catalog_mode,
                    1.0,
                    jax.random.key(0),
                    None,
                    None,
                    context=_context(channel, 0.0, 1.0),
                )
            )[: N_SLOTS * len(CATALOG_FIELDS)].reshape(N_SLOTS, -1)
            for slot in range(N_SLOTS):
                if not bool(truth.present[star, slot]):
                    continue
                elements = jnp.asarray([*cells[slot, 1:8].tolist(), cells[slot, 9]])
                p = float(
                    channel.detect_probability(
                        star, DETECT_MODE, time_d, elements[None]
                    )[0]
                )
                if not 0.02 < p < 0.98:
                    continue
                keys = jax.random.split(jax.random.key(11 + checked), 2000)
                rows = jax.vmap(
                    lambda k, t=truth, s=star: channel.realize(
                        t, s, DETECT_MODE, 1.0, k, None, None, context=context
                    )
                )(keys)
                got = float(rows[:, column_index(names, f"detected_{slot}")].mean())
                sigma = np.sqrt(p * (1 - p) / 2000)
                assert abs(got - p) < 3 * sigma + 1e-3
                checked += 1
                if checked == 3:
                    return
    assert checked > 0, "no slot had a detection probability inside (0.02, 0.98)"
