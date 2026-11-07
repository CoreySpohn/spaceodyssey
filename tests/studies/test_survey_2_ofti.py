"""Survey 2: adaptive detections alongside checkpoint orbit determination."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("photomancy")

from spaceodyssey.core.enums import EffectKind
from spaceodyssey.studies.survey_2 import OrbitSearchObserver, RevisitPolicy
from tests.adapters.test_photomancy_ofti import REGISTRY, ofti_study, refit
from tests.core.helpers import worlds
from tests.studies.survey_helpers import (
    action_projection,
    run_with_actions,
    same_actions,
)

N_STEPS = 80


def _study(policy, every=100):
    study = ofti_study(every=every)
    observer = OrbitSearchObserver()
    return dataclasses.replace(
        study,
        caps=REGISTRY.lower(study.caps, observer),
        observer=observer,
        agenda=dataclasses.replace(study.agenda, policies=(policy,)),
    )


def _run(study, boundary=None):
    truths, roots = worlds(1)
    return run_with_actions("host", study, truths, roots, N_STEPS, refit=boundary)


@pytest.fixture(scope="module", params=[False, True], ids=["adaptive", "ignoring"])
def runs(request):
    """Both policies, each with and without an OFTI checkpoint."""
    policy = RevisitPolicy(ignore_belief=request.param)
    with_ofti = _run(_study(policy), refit(count="epochs"))
    without = _run(_study(policy, every=10**6))
    return request.param, with_ofti, without


@pytest.mark.slow  # cold host runs of two policies, shared by both tests
def test_an_ofti_refit_installs_particles_but_does_not_change_the_actions(runs):
    """Fails if adaptive search cannot read detections alongside an orbit fit,
    or a refresh changes its complete effects or acquisition payloads. The ignoring
    reduction is checked separately by the fixture's second policy.
    """
    ignoring, with_ofti, without = runs
    belief = with_ofti.result.carry.belief
    assert int(belief.epochs[0, 0]) == 3
    assert int(without.result.carry.belief.epochs[0, 0]) == 0
    samples = np.asarray(belief.samples[0, 0])
    assert np.isfinite(samples).all() and len({tuple(r) for r in samples}) > 1
    assert same_actions(with_ofti, without, 1)
    summary = with_ofti.result.carry.pbelief
    assert not hasattr(summary, "samples")
    assert int(np.asarray(summary.n).sum()) == 2
    np.testing.assert_array_equal(summary.n, belief.n)
    np.testing.assert_array_equal(summary.mean, belief.mean)
    decisions = with_ofti.decisions[0]
    assert len(decisions) >= 5
    # An adaptive decision at a measurement release must use the detection data.
    if not ignoring:
        study = _study(RevisitPolicy())
        observed = []
        for view, eff in decisions:
            if int(eff.kind[0]) != EffectKind.OBSERVE:
                continue
            n = np.asarray(view.belief.n)[:2]
            snr = np.asarray(view.belief.mean)[:2] * np.sqrt(n)
            snr /= float(study.channel_model.noise[0])
            detected = snr > float(study.objective.threshold)
            score = np.where(n == 0, np.inf, np.where(detected, -np.inf, snr))
            assert int(eff.subject[0]) == int(np.argmax(score))
            if n.any():
                observed.append(int(eff.subject[0]))
        assert 1 in observed


@pytest.mark.slow  # cold host runs of two policies, shared by both tests
def test_the_action_projection_sees_a_different_plan(runs):
    """Fails if an ignoring plan matches the adaptive path or another ignoring plan."""
    _, _, without = runs
    other = RevisitPolicy(
        ignore_belief=True, plan=((1, 10), (0, 10), (1, 10), (0, 10), (1, 10))
    )
    changed = _run(_study(other, every=10**6))
    assert not same_actions(without, changed, 1)
    assert action_projection(without, 0)[1] != action_projection(changed, 0)[1]
