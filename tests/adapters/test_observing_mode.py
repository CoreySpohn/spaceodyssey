"""Stored-universe truth and the observing-mode channel, on the checked-in fixture.

The fixture (`tests/fixtures/exovista`) is one ExoVista universe of two stars.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.adapters.jaxedith import column_index, visit_columns
from spaceodyssey.core.kinds import RealizeContext
from tests.studies.bracket_helpers import (
    N_SLOTS,
    fixture_channel,
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


def test_absent_slot_never_detects(source, channel):
    """Fails if a visit reports a detection in a slot that holds no planet."""
    truth = _truth(source, 0)
    truth = eqx.tree_at(lambda t: t.present, truth, truth.present.at[0, -1].set(False))
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
            params = channel.truth_parameters(truth)
            for slot in range(N_SLOTS):
                if not bool(truth.present[star, slot]):
                    continue
                elements = jax.tree.map(
                    lambda x, star=star, slot=slot: x[star, slot][None], params
                )
                p = float(
                    channel.detect_probability(star, DETECT_MODE, time_d, elements)[0]
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
