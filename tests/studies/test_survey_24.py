"""Survey 24: the latent is drawn under its own purpose and shown only by a record.

(`spaceodyssey.studies.survey_24`.) The draw is checked on its own: changing any other
purpose's salt leaves the table alone and moves that purpose's stream, changing its own
moves only the table. Two truths that differ only in the latent run on the host
executor: no policy view and no observer belief changes before the first realization,
and the observer belief does change once the first record is released.
"""

import jax.random as jr
import numpy as np
import pytest

from spaceodyssey.core.enums import EventKind, Purpose
from spaceodyssey.studies.survey_24 import (
    N_PURPOSES,
    TARGET,
    TrajectoryWorldSource,
    survey_24,
)
from tests.studies.trajectory_helpers import (
    first_release,
    first_step,
    run_one,
    views_before,
)

pytestmark = pytest.mark.slow(
    reason="one compiled structure shared by two host runs, about 8 s cold"
)

N_STEPS = 60
OWN = int(Purpose.TRAJECTORY)


def _salts(purpose):
    return tuple(1 if p == purpose else 0 for p in range(N_PURPOSES))


@pytest.fixture(scope="module")
def pair():
    """The study on two truths that differ only in the latent table."""
    study = survey_24()
    root = jr.PRNGKey(0)
    key = jr.fold_in(root, Purpose.WORLD)
    other = TrajectoryWorldSource(salts=_salts(OWN))
    a = study.world_source.draw(key, study.caps)
    b = other.draw(key, study.caps)
    return (
        run_one(study, a, root, N_STEPS),
        run_one(study, b, root, N_STEPS),
        study,
    )


def test_the_latent_moves_with_its_own_purpose_and_no_other():
    """Fails if another purpose's key moves the table, the table shares a stream with
    one, or a salt moves nothing: each of the five other purposes leaves the table
    unchanged and changes its own stream only; the table's own salt changes the table
    and no stream.
    """
    key = jr.fold_in(jr.PRNGKey(0), Purpose.WORLD)
    caps = survey_24().caps
    base = TrajectoryWorldSource().draw(key, caps)
    moved = 0
    for p in range(N_PURPOSES):
        truth = TrajectoryWorldSource(salts=_salts(p)).draw(key, caps)
        streams_equal = np.all(truth.streams == base.streams, axis=1)
        table_equal = np.array_equal(truth.table.values, base.table.values)
        if p == OWN:
            assert not table_equal and streams_equal.all()
        else:
            assert table_equal
            assert streams_equal.sum() == N_PURPOSES - 1 and not streams_equal[p]
        moved += 1
    assert moved == N_PURPOSES


def test_no_view_changes_before_the_first_realization_and_one_does_after(pair):
    """Fails if the latent reaches a policy view or the observer's belief before a
    channel realizes it, or the check is vacuous: the first record is realized (Done)
    before it is released, no policy view and no belief step before the release
    differ across the two truths, and the belief first differs at that release.
    """
    a, b, _ = pair
    assert not np.array_equal(a.truth.table.values, b.truth.table.values)
    assert np.array_equal(a.truth.streams, b.truth.streams)
    done = first_step(a, EventKind.DONE, TARGET)
    release = first_release(a, TARGET)
    assert done < release == first_release(b, TARGET)
    for step in (done, release):
        same_fp, full, n_decisions = views_before(a, b, step)
        assert same_fp and full and n_decisions >= 1
    ta, tb = a.run.trace, b.run.trace
    belief = (np.asarray(ta.belief_mean) == np.asarray(tb.belief_mean)).all(axis=-1)
    belief &= (np.asarray(ta.belief_aux) == np.asarray(tb.belief_aux)).all(axis=-1)
    assert belief[:release].all() and release > 0
    differs = np.flatnonzero(~belief)
    assert len(differs) > 0 and int(differs[0]) == release
