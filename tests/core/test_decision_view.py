"""The decision view withholds the world source and carries the planning model."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from spaceodyssey import core
from spaceodyssey.core import EffectKind
from spaceodyssey.studies import builder
from spaceodyssey.studies.toy import ThresholdStopPolicy, ToyPhotometry
from tests.core.helpers import one_world

KEY = jax.random.PRNGKey(0)
PSTATE = jnp.zeros(1)


def view_with_snr(study, snr_over_planning_noise):
    """The policy view of a fresh run, with one belief mean and one count set."""
    truth, _ = one_world()
    carry = core.init_carry(study, truth)
    view = core.snapshot_view(carry, core.Audience.POLICY, study.caps)
    noise = (study.planning_model or study.channel_model).noise[0]
    mean = jnp.zeros_like(view.belief.mean).at[0].set(snr_over_planning_noise * noise)
    count = jnp.ones_like(view.belief.n)
    return eqx.tree_at(lambda v: (v.belief.mean, v.belief.n), view, (mean, count))


def test_the_view_withholds_the_world_source_and_uses_the_planning_model():
    """Fails if a policy or refit can reach the worlds or the generating model."""
    base = builder.survey_1()
    planning = ToyPhotometry(noise=2.0 * base.channel_model.noise)
    view = dataclasses.replace(base, planning_model=planning).decision_view()
    assert eqx.tree_equal(view.channel_model, planning)
    assert eqx.tree_equal(view.observer, base.observer)
    with pytest.raises(RuntimeError, match="withheld"):
        view.world_source.draw(jax.random.PRNGKey(0), base.caps)
    with pytest.raises(RuntimeError, match="withheld"):
        view.world_source.declared_population()


def test_without_a_planning_model_the_view_keeps_the_generating_model():
    """Fails if the default grant `model: truth` is not the generating model."""
    base = builder.survey_1()
    assert eqx.tree_equal(base.decision_view().channel_model, base.channel_model)
    assert base.decision_view().belief_population is None


def test_a_declared_belief_population_reaches_the_view():
    """Fails if a declared belief population is dropped from the decision view."""
    base = builder.survey_1()
    belief = ToyPhotometry(noise=base.channel_model.noise)
    study = dataclasses.replace(base, belief_population=belief)
    assert eqx.tree_equal(study.decision_view().belief_population, belief)


def test_a_policy_reads_the_planning_noise_not_the_generating_noise():
    """Fails if a decision on the view depends on the generating model."""
    policy = ThresholdStopPolicy(stop_snr=3.0)
    base = builder.survey_1(policies=(policy,))
    noise = base.channel_model.noise
    a = dataclasses.replace(base, planning_model=ToyPhotometry(noise=noise))
    b = dataclasses.replace(a, channel_model=ToyPhotometry(noise=3.0 * noise))
    view = view_with_snr(a, snr_over_planning_noise=4.5)

    def stop(s):
        out = policy.decide(PSTATE, view, s, KEY)
        return int(out[0].kind[0]) == EffectKind.STOP

    assert stop(a.decision_view()) and stop(b.decision_view())
    assert stop(a) != stop(b)
