"""Visit pacing and paired grants on the fixture run with a real refit."""

import itertools

import jax
import numpy as np
import pytest

from spaceodyssey.studies import bracket as br

from .bracket_helpers import N_STEPS, N_WORLDS, build

# Enough telescope days for a second visit to one star in each world, and few
# enough to keep the admission trace bounded.
BUDGET_D = 200.0
pytestmark = pytest.mark.slow(reason="compiles the chunked refresh driver once")


@pytest.fixture(scope="module")
def built():
    _, study, executor, truths, roots, run = build(budget_d=BUDGET_D)
    table = br.activity_table(run, study)
    return study, executor, truths, roots, run, table


def test_a_visit_waits_for_a_refit_that_read_the_previous_one(built):
    """Fails if a star is revisited before a refit read its last released row."""
    _, _, _, _, run, _ = built
    log = run.carry.log
    kind, valid = np.asarray(log.kind), np.asarray(log.valid)
    subject, start = np.asarray(log.subject), np.asarray(log.start_tick)
    released = np.asarray(log.released_tick)
    checked = 0
    for w in range(N_WORLDS):
        rows = np.flatnonzero((kind[w] == br.KIND_MEASUREMENT) & valid[w])
        for s in set(subject[w][rows]):
            mine = rows[subject[w][rows] == s]
            mine = mine[np.argsort(start[w][mine])]
            for a, b in itertools.pairwise(mine):
                day = br.TICKS_PER_DAY
                assert start[w, b] // day > released[w, a] // day
                checked += 1
    assert checked > 0, "no star was visited twice"


def test_run_grants_is_deterministic_and_a_self_pair_is_zero(built):
    """Fails if shared truths and roots change a table or a self contrast is nonzero."""
    study, executor, truths, roots, _, table = built
    arms = {"a": (study, executor.boundary), "b": (study, executor.boundary)}
    out = br.run_grants(arms, truths, roots, N_STEPS)
    for name in ("star", "world"):
        for key, value in getattr(out.activity["a"], name).items():
            reference = getattr(table, name)[key]
            np.testing.assert_array_equal(value, reference)
            np.testing.assert_array_equal(
                getattr(out.activity["b"], name)[key], reference
            )
    for name in ("a", "b"):
        scored = br.verdict(built[4], study)
        np.testing.assert_array_equal(out.verdict[name].yield_, scored.yield_)
        for key, value in out.verdict[name].counts.items():
            np.testing.assert_array_equal(value, scored.counts[key])
    assert out.gap("a", "b", "yield_")[:2] == (0.0, 0.0)
    mean, se, _, n = out.gap("a", "b", "charged")
    assert (mean, se, n) == (0.0, 0.0, N_WORLDS)
    jax.clear_caches()
