"""A visit row never shows whether a slot holds a planet.

A slot with no planet and a slot with a planet that was not detected must give
identical report cells, bit for bit, in every column. The checked-in fixture
(`tests/fixtures/exovista`) is one ExoVista universe of two stars, each with more
planets than `N_SLOTS`; the test reads it into more slots and keeps the brightest
`N_SLOTS - 1` planets and one absent slot.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.adapters.jaxedith import VISIT_REST
from spaceodyssey.adapters.skyscapes import ExoVistaWorldSource, PopulationModel
from tests.adapters.test_observing_mode import DETECT_MODE, _context, _truth
from tests.studies.bracket_helpers import (
    FIXTURE_BANDS,
    N_SLOTS,
    fixture_channel,
    fixture_files,
)

# More slots than either star's planets, so the last is absent.
WIDE_SLOTS = 12

pytestmark = pytest.mark.medium(reason="loads the fixture and compiles one visit")


def test_absent_slot_row_equals_a_present_undetected_slot_row():
    """Fails if a null report exposes existence in any declared report column."""
    source = ExoVistaWorldSource.from_files(
        [fixture_files()],
        n_slots=WIDE_SLOTS,
        bands=FIXTURE_BANDS,
        generator=PopulationModel.exovista(),
    )
    channel = fixture_channel(source)
    kept = jnp.asarray([*range(N_SLOTS - 1), WIDE_SLOTS - 1])
    truth = jax.tree_util.tree_map(lambda x: x[:, kept], _truth(source, 0))
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
    rows = rows[:, : channel.widths()[0]]
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
