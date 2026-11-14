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
from tests.core import refresh_helpers
from tests.core.helpers import one_world, worlds

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


def test_the_view_of_a_view_is_the_view():
    """Fails if a second `decision_view` call raises or builds a different study."""
    view = builder.survey_1().decision_view()
    assert view.decision_view() is view


def _require_withheld(study):
    if not isinstance(study.world_source, core.WithheldWorldSource):
        raise AssertionError("a decision received the full study")


class ProbePolicy(refresh_helpers.LeastSeenPolicy):
    """The least-seen policy, refusing any study that still holds the world source."""

    def decide(self, pstate, view, study, key):
        _require_withheld(study)
        return super().decide(pstate, view, study, key)


class ProbeRefit(core.AbstractRefitBoundary):
    """The masked refit, refusing any study that still holds the world source."""

    def __init__(self):
        self.inner = refresh_helpers.masked()
        self.calls = 0

    def refit_batch(self, study, requests):
        _require_withheld(study)
        self.calls += 1
        return self.inner.refit_batch(study, requests)


@pytest.mark.medium(reason="Runs a refreshed toy study on both executors")
@pytest.mark.parametrize("executor", ["compiled", "host_array", "host_python"])
def test_every_decision_and_refit_call_site_passes_the_view(executor):
    """Fails if an executor hands a policy or a refit the full study."""
    study = refresh_helpers.refresh_study(40, policies=(ProbePolicy(),))
    truths, roots = worlds(1)
    refit = ProbeRefit()
    if executor == "compiled":
        run = core.CompiledExecutor().run
        run(study, truths, roots, 24, refit=refit, chunk=5)
    else:
        form = "array" if executor == "host_array" else "python"
        truth, root = one_world()
        core.HostExecutor().run(study, truth, root, 24, refit=refit, policy_form=form)
    assert refit.calls > 0


class MisleadingWorldSource(core.AbstractWorldSource):
    """A user source with a similarly named ordinary class attribute."""

    withheld = True

    def draw(self, key, capacities):
        return jnp.asarray(1.0)


def test_a_user_sources_public_attribute_cannot_bypass_the_view():
    """Fails if a user-defined source's withheld flag exposes the full study."""
    study = dataclasses.replace(
        builder.survey_1(), world_source=MisleadingWorldSource()
    )
    view = study.decision_view()
    assert isinstance(view.world_source, core.WithheldWorldSource)
    with pytest.raises(RuntimeError, match="withheld"):
        view.world_source.draw(KEY, study.caps)
