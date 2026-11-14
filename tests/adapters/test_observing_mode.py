"""Stored-universe truth and the observing-mode channel, on the checked-in fixture.

The fixture (`tests/fixtures/exovista`) is one ExoVista universe of two stars.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.adapters.jaxedith import planet_free
from spaceodyssey.core.kinds import RealizeContext
from spaceodyssey.core.records import unpack
from tests.studies.bracket_helpers import (
    N_SLOTS,
    fixture_channel,
    fixture_source,
)

pytestmark = pytest.mark.medium(reason="loads the fixture and compiles the channel")

DETECT_DURATION = 1


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


def test_absent_slot_never_detects(source, channel):
    """Fails if a visit reports a detection in a slot that holds no planet."""
    truth = _truth(source, 0)
    truth = eqx.tree_at(lambda t: t.present, truth, truth.present.at[0, -1].set(False))
    mode = channel.mode_table.channel("detection", DETECT_DURATION)
    context = _context(channel, 100.0, channel.mode_table.duration_s(mode))
    keys = jax.random.split(jax.random.key(3), 24)
    rows = jax.vmap(
        lambda k: channel.realize(truth, 0, mode, 1.0, k, None, None, context=context)
    )(keys)
    absent = [s for s in range(N_SLOTS) if not bool(truth.present[0, s])]
    assert absent, "the first star fills every slot"
    for slot in absent:
        assert (
            float(unpack(channel.measurement_template, rows).detected[:, slot].max())
            == 0.0
        )


def test_a_star_row_ignores_the_other_stars_truth(source, channel):
    """Replacing star 1's planets leaves star 0's visit row unchanged."""
    truth = _truth(source, 0)
    other = _truth(source, 1)
    mixed = jax.tree_util.tree_map(
        lambda a, b: a.at[1].set(b[1]) if a.ndim >= 2 else a, truth, other
    )
    mode = channel.mode_table.channel("detection", DETECT_DURATION)
    context = _context(channel, 100.0, channel.mode_table.duration_s(mode))
    key = jax.random.key(5)

    def row(t):
        return channel.realize(t, 0, mode, 1.0, key, None, None, context=context)

    np.testing.assert_array_equal(np.asarray(row(truth)), np.asarray(row(mixed)))


def test_realized_detection_matches_the_stated_probability(source, channel):
    """Fails if a threshold-centered control detects at the wrong frequency.

    N = 2000 draws resolves a probability error of 0.045 at four sigma near 0.5.
    """
    mode = channel.mode_table.channel("detection", DETECT_DURATION)
    context = _context(channel, 100.0, channel.mode_table.duration_s(mode))
    interval = channel._interval(context, settle=True)[0]
    time_d = float(interval.midpoint_jd) - float(source.epoch_jd)
    truth = _truth(source, 0)
    band = channel.imaging_band_index
    report, law = channel.assets().reports[band], channel.assets().law
    prepared = channel._prepared(0, band, nominal=False)
    planets = channel._slot_planets(
        truth, 0, truth.band_index(channel.detection_band.center_nm)
    )
    budgets = jax.vmap(lambda p: report.budget(prepared, p, interval))(planets)
    slot = int(jnp.argmax(jnp.where(truth.present[0], budgets.planet, 0.0)))
    planet = jax.tree.map(lambda x: x[slot], planets)
    limit = law.limit_counts(report.budget(prepared, planet_free(planet), interval))
    assert float(budgets.planet[slot]) > 0
    radius = truth.Rp_AU[0, slot] * jnp.sqrt(
        law.threshold * limit / budgets.planet[slot]
    )
    truth = eqx.tree_at(lambda t: t.Rp_AU, truth, truth.Rp_AU.at[0, slot].set(radius))
    elements = jax.tree.map(lambda x: x[0, slot][None], channel.truth_parameters(truth))
    p = float(channel.detect_probability(0, mode, time_d, elements)[0])
    np.testing.assert_allclose(p, 0.5, atol=1e-8)
    keys = jax.random.split(jax.random.key(0), 2000)
    rows = jax.vmap(
        lambda k: channel.realize(truth, 0, mode, 1.0, k, None, None, context=context)
    )(keys)
    got = float(unpack(channel.measurement_template, rows).detected[:, slot].mean())
    sigma = np.sqrt(p * (1 - p) / 2000)
    assert abs(got - p) < 3 * sigma + 1e-3
