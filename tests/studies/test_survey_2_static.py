"""Survey 2 builders and policy declarations: no run needed."""

from types import SimpleNamespace

import jax.numpy as jnp
import jax.random as jr
import pytest

from spaceodyssey.core.battery.result import tree_bits_equal
from spaceodyssey.core.enums import EffectKind
from spaceodyssey.studies import toy
from spaceodyssey.studies.survey_2 import (
    GaussianMean,
    RevisitPolicy,
    SearchMean,
    survey_2,
)


def test_the_plain_survey_is_survey_1_with_another_policy():
    """Fails if the plain survey changes more than its policy, or the ignoring
    policy requests different parameters (including inactive effect columns).
    """
    study, base = survey_2(), toy.survey_1()
    assert study.caps == base.caps
    assert study.channels.id.tolist() == base.channels.id.tolist()
    assert int(study.refresh.every_ticks) == int(base.refresh.every_ticks)
    assert type(study.agenda.policies[0]) is RevisitPolicy
    view = SimpleNamespace(tick=jnp.asarray(0), n_dec=jnp.asarray(0))
    ignoring = RevisitPolicy(ignore_belief=True)
    for pos in (0, 5):
        state = jnp.zeros(study.caps.n_pstate).at[0].set(pos)
        got = ignoring.decide(state, view, study, jr.PRNGKey(0))
        expected = base.agenda.policies[0].decide(state, view, base, jr.PRNGKey(0))
        assert tree_bits_equal(got, expected)


def test_a_refresh_grid_adds_the_refit_observer_and_a_schedule():
    """Fails if a positive period gives no schedule or a non-refit observer."""
    study = survey_2(100)
    assert study.refresh.every_ticks == 100
    assert isinstance(study.observer, SearchMean)
    assert study.observer.capabilities().refit


def test_the_policy_declares_only_observe_and_stop():
    """Fails if the policy declares an effect it does not use."""
    declared = RevisitPolicy().declare()
    assert declared.effects == frozenset((EffectKind.OBSERVE, EffectKind.STOP))


def test_the_posterior_standard_deviation_shrinks_with_samples():
    """Fails if the posterior spread is not the conjugate one: the prior's at zero
    samples and `1/sqrt(1/s^2 + n/noise^2)` after n.
    """
    observer = GaussianMean(prior_std=0.3, noise=0.5)
    assert float(observer.posterior_std(0)) == pytest.approx(0.3)
    expected = (1 / 0.3**2 + 4 / 0.5**2) ** -0.5
    assert float(observer.posterior_std(4)) == pytest.approx(expected)
    assert float(observer.posterior_std(jnp.asarray(9))) < float(
        observer.posterior_std(4)
    )
