"""Survey 2 with orbit determination: an OFTI refit never changes a policy's actions."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("photomancy")

from spaceodyssey.core import HostExecutor
from spaceodyssey.core.executors.base import index_world
from spaceodyssey.studies.survey_2 import RevisitPolicy
from tests.adapters.test_photomancy_ofti import ofti_study, refit
from tests.core.helpers import worlds
from tests.studies.survey_helpers import action_projection

N_STEPS = 80
IGNORING = RevisitPolicy(ignore_belief=True)


def _with_policy(study, policy):
    agenda = dataclasses.replace(study.agenda, policies=(policy,))
    return dataclasses.replace(study, agenda=agenda)


def _run(study, boundary=None):
    truths, roots = worlds(1)
    return HostExecutor().run(
        study, index_world(truths, 0), roots[0], N_STEPS, refit=boundary
    )


@pytest.fixture(scope="module")
def runs():
    """The revisit survey with scripted astrometry: no refresh, and OFTI refresh."""
    with_ofti = _run(_with_policy(ofti_study(), IGNORING), refit())
    # a grid wider than the run: the same survey, never refreshed
    without = _run(_with_policy(ofti_study(every=10**6), IGNORING))
    return with_ofti, without


def _one(run):
    """The single world of a host run as a batch of one for the projection helper."""
    return dataclasses.replace(
        run,
        carry=_batched(run.carry),
        trace=_batched(run.trace),
    )


def _batched(tree):
    import jax

    return jax.tree_util.tree_map(lambda x: np.asarray(x)[None], tree)


@pytest.mark.slow  # one cold study compile and the OFTI scorer on the host
def test_an_ofti_refit_installs_particles_but_does_not_change_the_actions(runs):
    """Fails if the refit changes any decision, subject, acquisition or payload of the
    revisit policy, or never installs the orbit fit it was run for.
    """
    with_ofti, without = runs
    assert int(with_ofti.carry.belief.n[0]) == 3
    assert int(without.carry.belief.n[0]) == 0
    a = action_projection(_one(with_ofti), 0)
    b = action_projection(_one(without), 0)
    assert a[0] == b[0] and a[1] == b[1]
    assert np.array_equal(a[2], b[2])
    # the scripted astrometry wakes the policy at ticks 10, 40 and 70 as well
    assert len(a[0]) >= 5 and len(a[1]) == 2


@pytest.mark.slow  # a third cold study compile
def test_the_action_projection_sees_a_different_plan(runs):
    """Negative control: the same comparison fails for a policy with another plan,
    so the equality above is not a property of an insensitive projection.
    """
    _, without = runs
    other = RevisitPolicy(
        ignore_belief=True, plan=((1, 10), (0, 10), (1, 10), (0, 10), (1, 10))
    )
    changed = _run(_with_policy(ofti_study(every=10**6), other))
    a = action_projection(_one(without), 0)
    b = action_projection(_one(changed), 0)
    assert a[1] != b[1]
