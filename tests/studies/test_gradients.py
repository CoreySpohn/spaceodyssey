"""Structural tests of the three gradient estimators at small sizes."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.studies import builder
from tests.core import gradients as grad
from tests.core.helpers import history_of, worlds
from tests.studies.scenarios import toy_gradient


def test_d1_reverse_matches_fd_on_toy_sizes():
    r = grad.d1_reverse_and_fd(builder.survey_1(), *worlds(64))
    assert min(r["rel_err"].values()) <= 1e-6
    assert 1.5 <= r["slope"] <= 2.5 and abs(r["reference"]) >= 1e-3


def test_d1_detects_wrong_gradient():
    r = grad.d1_reverse_and_fd(builder.survey_1(), *worlds(64), grad_scale=1.1)
    assert min(r["rel_err"].values()) > 1e-3


def test_d1_slope_is_nan_without_the_fit_range():
    r = grad.d1_reverse_and_fd(builder.survey_1(), *worlds(64), hs=(1e-5, 1e-6))
    assert np.isnan(r["slope"]) and set(r["rel_err"]) == {1e-5, 1e-6}


def test_d2_two_world_enumeration_is_unbiased():
    est, truth = grad.d2_exact_two_world(
        p=float(jax.nn.sigmoid(0.3)), f=((0.0, 1.0), (1.0, 1.0))
    )
    assert est == pytest.approx(truth, rel=1e-12)


@pytest.mark.parametrize("seed", [0, 1], ids=["root-0", "root-1"])
def test_d2_statistic_shape(seed):
    r = grad.d2_reinforce_and_enumerated(
        toy_gradient.d2_study(), *worlds(1024, seed=seed)
    )
    assert r["n_flip"] > 0 and np.isfinite(r["z"]) and r["se_paired"] > 0
    assert r["all_done"]


@pytest.mark.slow(reason="REINFORCE at full size, 32768 worlds per seed set")
@pytest.mark.parametrize("seed", [0, 1], ids=["root-0", "root-1"])
def test_d2_reinforce_is_unbiased_at_full_size(seed):
    # 32768 worlds per seed set; the estimate must sit within 3 paired standard
    # errors of the enumerated gradient, resolved to a few percent.
    r = grad.d2_reinforce_and_enumerated(
        toy_gradient.d2_study(), *worlds(32768, seed=seed)
    )
    assert r["all_done"] and r["n_flip"] > 0
    assert abs(r["z"]) < 3
    assert r["power"] < 0.05


def test_d3_per_world_gap_monotone_in_width():
    r = grad.d3_pathwise_and_analytic(builder.survey_1(), *worlds(256))
    gaps = r["gaps"]
    assert gaps.shape == (4, 256) and r["all_done"]
    assert bool(np.all(gaps[1:] <= gaps[:-1] + 1e-15))


def test_gradient_leaf_is_continuous_only():
    dyn, _ = eqx.partition(builder.survey_1().design, eqx.is_inexact_array)
    assert len(jax.tree_util.tree_leaves(dyn)) == 1
    assert dyn.gain.dtype == jnp.float64


def test_replay_reproduces_the_recorded_history():
    study = builder.survey_1()
    truths, roots = worlds(1)
    replay = grad.replay_study(
        study, jax.tree_util.tree_map(lambda a: a[0], truths), roots[0]
    )
    a, b = history_of(study), history_of(replay)
    assert a.events == b.events and a.rows == b.rows


def test_stochastic_policy_observe_count_matches_acquisition_counts():
    study = toy_gradient.d2_study()
    truths, roots = worlds(32)
    cs, _ = CompiledExecutor().run(study, truths, roots, grad.N_STEPS)
    assert bool(jnp.all(cs.done))
    assert np.array_equal(
        np.asarray(cs.pstate[:, 0]), np.asarray(cs.acquisition_counts.sum(axis=1))
    )
    assert 0 < float(cs.pstate[:, 0].mean()) < 5


def test_forced_m_makes_m_observes():
    study = toy_gradient.d2_study()
    truths, roots = worlds(8)
    for m in range(6):
        s = eqx.tree_at(
            lambda x: x.agenda.policies,
            study,
            (
                toy_gradient.ReplayPolicy(
                    effects=toy_gradient.forced_effects(study.caps, study, m)
                ),
            ),
        )
        cs, _ = CompiledExecutor().run(s, truths, roots, grad.N_STEPS)
        assert bool(jnp.all(cs.done)) and int(cs.acquisition_counts.sum()) == m * 8


def test_paired_z_flags_a_bias():
    rng = np.random.default_rng(0)
    ref = rng.normal(1.0, 0.1, 4000)
    est = ref + rng.normal(0.0, 0.5, 4000)
    z0, se, power = grad.paired_z(est, ref)
    z1, _, _ = grad.paired_z(est + 0.1, ref)
    assert abs(z0) < 4 and z1 > 8 and power == pytest.approx(se / abs(ref.mean()))


def test_recorded_effects_do_not_depend_on_the_world():
    study = builder.survey_1()
    truths, roots = worlds(8)
    pick = lambda w: jax.tree_util.tree_map(lambda a: a[w], truths)  # noqa: E731
    t0 = grad.record_effects(study, pick(0), roots[0])
    t7 = grad.record_effects(study, pick(7), roots[7])
    assert all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(
            jax.tree_util.tree_leaves(t0), jax.tree_util.tree_leaves(t7), strict=False
        )
    )


def test_stochastic_credit_equals_forced_credit_at_its_own_m():
    """Handle a transaction state update.

    The enumerated reference assumes a world's credit depends on the draws only through
    m; check it world by world.
    """
    study = toy_gradient.d2_study()
    truths, roots = worlds(128)
    credit, done, m_w = grad._hard_values(study, truths, roots, grad.N_STEPS)
    forced = np.stack(
        [
            np.asarray(grad._forced_credit(study, truths, roots, m, grad.N_STEPS))
            for m in range(6)
        ]
    )
    own = forced[np.asarray(m_w).astype(int), np.arange(128)]
    assert np.array_equal(own, np.asarray(credit))
    assert bool(np.all(np.asarray(done)))
