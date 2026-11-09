"""Survey 22: a trajectory-table swap moves the target and nothing the policy read.

Two truths that differ only in the trajectory table (`spaceodyssey.studies.survey_22`)
run on the host executor, once on the clean study and once on the leaky one (a finder
channel realized and released at tick 0 whose payload is the table's position). The
invariance check compares every policy view handed out before the first release of a
target record: the fingerprints, which hold only integer content, and the views in
full, whose payload and belief columns carry the floats the table moves.
"""

import jax.random as jr
import numpy as np
import pytest

from spaceodyssey.core.enums import Purpose, RecordStatus
from spaceodyssey.studies.survey_22 import (
    FINDER,
    N_PURPOSES,
    TARGET,
    TICKS,
    TrajectoryWorldSource,
    survey_22,
)
from tests.studies.trajectory_helpers import (
    first_release,
    run_one,
    same_view,
    views_before,
)

pytestmark = pytest.mark.slow(
    reason="one compiled structure shared by four host runs, about 10 s cold"
)

N_STEPS = 60
TABLE_SALT = tuple(1 if p == Purpose.TRAJECTORY else 0 for p in range(N_PURPOSES))


def _truths(study):
    """Two truths from one world key that differ only in the trajectory table."""
    root = jr.PRNGKey(0)
    key = jr.fold_in(root, Purpose.WORLD)
    a = study.world_source.draw(key, study.caps)
    b = TrajectoryWorldSource(salts=TABLE_SALT).draw(key, study.caps)
    return root, a, b


def positions(result, channel):
    """`(start ticks, first payload column)` of the completed records on `channel`."""
    log = result.run.carry.log
    keep = (
        np.asarray(log.valid)
        & (np.asarray(log.channel) == channel)
        & (np.asarray(log.status) == RecordStatus.COMPLETED)
    )
    return np.asarray(log.start_tick)[keep], np.asarray(log.payload)[keep][:, 0]


@pytest.fixture(scope="module")
def runs():
    """Clean and leaky studies, each on the two truths."""
    out = {}
    for name, leaky in (("clean", False), ("leaky", True)):
        study = survey_22(leaky=leaky)
        root, a, b = _truths(study)
        out[name] = (run_one(study, a, root, N_STEPS), run_one(study, b, root, N_STEPS))
    return out


def test_the_swap_moves_the_target_and_no_view_before_its_release(runs):
    """Fails if the table reaches the policy before a record carries it, the truths
    differ in more than the table, or the positions do not come from the table:
    the realized positions differ in every record and equal the table at the start
    tick, while the fingerprints of all earlier decisions are identical.
    """
    a, b = runs["clean"]
    assert np.array_equal(a.truth.streams, b.truth.streams)
    assert not np.array_equal(a.truth.table.values, b.truth.table.values)
    (starts, pa), (_, pb) = positions(a, TARGET), positions(b, TARGET)
    assert len(starts) == 4
    assert np.all(pa != pb)
    for p, r in ((pa, a), (pb, b)):
        want = np.interp(starts, np.asarray(TICKS), np.asarray(r.truth.table.values))
        np.testing.assert_allclose(p, want, rtol=0, atol=1e-12)
    step = first_release(a, TARGET)
    assert step == first_release(b, TARGET)
    same_fp, full, n_decisions = views_before(a, b, step)
    assert same_fp and full and n_decisions >= 1
    # not vacuous: once a target record is released, the full views differ
    late = [i for i, s in enumerate(a.steps) if s >= step]
    assert late and not all(same_view(a.views[i], b.views[i]) for i in late)


def test_a_finder_that_releases_the_position_at_tick_zero_breaks_the_invariance(runs):
    """Fails if the invariance check cannot fail: the leaky study adds a finder
    realized and released at tick 0 with the table's position as its payload, so
    the views before the target's first release differ across the two truths.
    """
    a, b = runs["leaky"]
    for r in (a, b):
        _, finder = positions(r, FINDER)
        log = r.run.carry.log
        row = np.flatnonzero(
            np.asarray(log.valid) & (np.asarray(log.channel) == FINDER)
        )
        assert len(row) == 1 and int(log.released_tick[row[0]]) == 0
        np.testing.assert_allclose(finder, [r.truth.table.at(0)], rtol=0, atol=1e-12)
    assert not np.array_equal(positions(a, FINDER)[1], positions(b, FINDER)[1])
    step = first_release(a, TARGET)
    assert step == first_release(b, TARGET)
    same_fp, full, n_decisions = views_before(a, b, step)
    assert not full and n_decisions >= 1
    # the fingerprint holds integers only, so it cannot see the leaked float
    assert same_fp
