"""Frozen network actions agree with eager evaluation on each recorded view."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from spaceodyssey.core import CompiledExecutor, HostExecutor
from spaceodyssey.core.battery.views import DecisionViews
from spaceodyssey.core.carry import init_carry
from spaceodyssey.core.enums import Audience, Disposition, EffectKind, Scope
from spaceodyssey.core.executors.base import histories_equal, index_world
from spaceodyssey.core.preflight import preflight
from spaceodyssey.core.snapshot import EFFECT_FIELDS, snapshot_view
from spaceodyssey.studies.survey_25 import NetworkPolicy, survey_25
from spaceodyssey.studies.toy import ToyTruth


def reference_actions(policy, view, study):
    # This eager reference deliberately does not call decide or a policy helper.
    inputs = np.concatenate(
        [
            view.belief.mean,
            view.belief.n,
            [
                float(view.tick) / study.horizon_tick,
                float(view.n_dec) / max(policy.n_obs, 1),
            ],
        ]
    ).astype(np.float64)
    logits = np.asarray(policy.network(jnp.asarray(inputs)))
    subject, channel = divmod(int(np.argmax(logits)), study.caps.n_channels)
    stop = int(view.n_dec) >= policy.n_obs
    columns = {
        name: np.zeros(study.caps.n_effects, dtype=np.int64) for name in EFFECT_FIELDS
    }
    columns["disruptive"] = np.zeros(study.caps.n_effects, dtype=bool)
    columns["kind"][0] = EffectKind.STOP if stop else EffectKind.OBSERVE
    columns["subject"][0] = 0 if stop else subject
    columns["channel_index"][0] = 0 if stop else channel
    columns["lane"][0] = 0 if stop else int(study.channels.lane[channel])
    columns["group"][0] = int(view.n_dec)
    if stop:
        columns["scope"][0] = Scope.CAMPAIGN
        columns["disposition"][0] = Disposition.DRAIN
    return columns


@pytest.mark.parametrize("seed", [0, 1])
def test_survey_25_actions_match_python_on_recorded_snapshots(seed):
    study = survey_25(key=jr.PRNGKey(seed))
    policy = study.agenda.policies[0]
    assert isinstance(policy, NetworkPolicy)
    assert isinstance(policy.network, eqx.nn.MLP)
    assert not preflight(study, executor="compiled")
    truth = ToyTruth(signal=jnp.array([0.7, 1.0, 1.3, 1.5]))
    root = jr.PRNGKey(19)
    views, effects = [], []
    host = HostExecutor().run(
        study,
        truth,
        root,
        40,
        policy_form="python",
        view_log=views,
        effects_log=effects,
    )
    capture = DecisionViews()
    compiled = CompiledExecutor().run_worlds(
        capture.instrument(study),
        jax.tree_util.tree_map(lambda a: a[None], truth),
        root[None],
        40,
    )
    recorded = capture.read(study, root, index_world(compiled.trace, 0))
    assert bool(host.complete) and bool(compiled.complete[0])
    assert histories_equal(host.history(), compiled.history(0)).go
    assert len(views) == len(recorded) == policy.n_obs + 1
    assert host.history().n_records == policy.n_obs
    for host_view, host_effect, pair in zip(views, effects, recorded, strict=True):
        assert pair is not None, "compiled decision was not captured"
        compiled_view, compiled_effect = pair
        for view, effect in ((host_view, host_effect), pair):
            expected = reference_actions(policy, view, study)
            for name in EFFECT_FIELDS:
                np.testing.assert_array_equal(getattr(effect, name), expected[name])
        for name in EFFECT_FIELDS:
            np.testing.assert_array_equal(
                getattr(host_effect, name), getattr(compiled_effect, name)
            )
        np.testing.assert_allclose(
            host_view.belief.mean, compiled_view.belief.mean, rtol=0, atol=1e-12
        )


def test_network_action_changes_with_released_belief():
    study = survey_25(key=jr.PRNGKey(0))
    network = eqx.nn.MLP(10, 12, width_size=8, depth=0, key=jr.PRNGKey(2))
    # A controlled network chooses subject 0 initially, subject 1 after its
    # released mean exceeds 1. A fixed plan cannot satisfy this counterfactual.
    network = eqx.tree_at(
        lambda n: (n.layers[0].weight, n.layers[0].bias),
        network,
        (jnp.zeros((12, 10)).at[3, 1].set(1), jnp.zeros(12).at[0].set(1)),
    )
    policy = NetworkPolicy(network=network)
    truth = ToyTruth(signal=jnp.ones(study.caps.n_subjects))
    carry = init_carry(study, truth)
    view = snapshot_view(carry, Audience.POLICY, study.caps)
    changed = dataclasses.replace(
        view,
        belief=dataclasses.replace(view.belief, mean=view.belief.mean.at[1].set(2)),
    )
    decide = eqx.filter_jit(policy.decide)
    before = decide(carry.pstate, view, study, jr.PRNGKey(0))[0]
    after = decide(carry.pstate, changed, study, jr.PRNGKey(0))[0]
    assert int(before.subject[0]) == 0 and int(after.subject[0]) == 1


@pytest.mark.parametrize("n_obs", [jnp.asarray(5), 5.0, True])
def test_network_policy_requires_a_python_int_limit(n_obs):
    network = survey_25().agenda.policies[0].network
    with pytest.raises(TypeError, match="Python int"):
        NetworkPolicy(network=network, n_obs=n_obs)
