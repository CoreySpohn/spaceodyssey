"""A visit row never shows whether a slot holds a planet.

A slot with no planet and a slot with a planet that was not detected must give
identical report cells, bit for bit, in every column. The checked-in fixture
(`tests/fixtures/exovista`) is one ExoVista universe of two stars.
"""

import jax
import numpy as np
import pytest

from spaceodyssey.adapters.jaxedith import VISIT_REST
from tests.adapters.test_observing_mode import DETECT_MODE, _context, _truth
from tests.studies.bracket_helpers import N_SLOTS, fixture_channel, fixture_source

pytestmark = pytest.mark.medium(reason="loads the fixture and compiles one visit")


def test_absent_slot_row_equals_a_present_undetected_slot_row():
    """Every report column of an absent slot matches a present, undetected slot."""
    source = fixture_source(1)
    channel = fixture_channel(source)
    truth = _truth(source, 0)
    present = np.asarray(truth.present[0])
    context = _context(channel, 100.0, 160000.0)
    keys = jax.random.split(jax.random.key(3), 8)
    rows = np.asarray(
        jax.vmap(
            lambda k: channel.realize(
                truth, 0, DETECT_MODE, 1.0, k, None, None, context=context
            )
        )(keys)
    )
    detected = rows[:, :N_SLOTS] > 0.5
    n_rest = len(VISIT_REST)
    cells = rows[:, -N_SLOTS * n_rest :].reshape(len(keys), N_SLOTS, n_rest)
    full = np.concatenate([rows[:, :N_SLOTS, None], cells], axis=-1)
    absent = np.flatnonzero(~present)
    assert absent.size, "the fixture star fills every slot"
    compared = 0
    for key in range(len(keys)):
        quiet = np.flatnonzero(present & ~detected[key])
        for a in absent:
            for q in quiet:
                assert full[key, a].tobytes() == full[key, q].tobytes()
                compared += 1
    assert compared > 0, "no present slot went undetected"
