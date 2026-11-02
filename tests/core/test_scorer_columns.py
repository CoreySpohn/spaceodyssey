"""Scorer-only payload columns never reach a snapshot or an observer."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from spaceodyssey.core import (
    Audience,
    CompiledExecutor,
    HostExecutor,
    index_world,
    snapshot_view,
)
from spaceodyssey.core.enums import Purpose
from spaceodyssey.studies import toy
from tests.core import record_helpers as rh
from tests.core.corpus import CORPUS, CORPUS_BY_NAME, PLAN5
from tests.core.helpers import N_STEPS_CORPUS, worlds


def _run(study, executor, truths, roots, n_worlds):
    """Run `n_worlds` worlds; returns the list of final carries."""
    if executor == "compiled":
        cs, _ = CompiledExecutor().run(study, truths, roots, N_STEPS_CORPUS)
        return [index_world(cs, w) for w in range(n_worlds)]
    out = []
    for w in range(n_worlds):
        carry, _ = HostExecutor().run(
            study, index_world(truths, w), roots[w], N_STEPS_CORPUS
        )
        out.append(carry)
    return out


def _hidden_columns_are_zero(carry, caps, reference=None):
    """The final carry's snapshots hide what the audience may not read.

    The snapshot is taken under `caps`; what may not be read is judged by `reference`
    (default `caps`), so a study with a looser layout can be checked against a stricter
    one.
    """
    reference = caps if reference is None else reference
    for audience in (Audience.POLICY, Audience.OBSERVER):
        view = snapshot_view(carry, audience, caps)
        table = np.asarray(reference.payload_audience)
        kind = np.clip(np.asarray(view.log.kind), 0, len(table) - 1)
        unreadable = (table[kind] & int(audience)) == 0
        payload = np.asarray(view.log.payload)
        if np.any(payload[unreadable] != 0):
            return False
    return True


@pytest.mark.parametrize("executor", ["host", "compiled"])
@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.name)
def test_snapshot_hides_scorer_columns(case, executor):
    rh.SEEN.clear()
    study = rh.spied(case.make_study())
    truths, roots = worlds(2)
    carries = _run(study, executor, truths, roots, 2)
    # every view a policy was handed, on the run itself
    assert rh.SEEN, "no decision ran"
    assert all(total == 0.0 for total in rh.SEEN)
    for carry in carries:
        # the scorer still has its column, so the check is not vacuous
        scored = np.asarray(carry.log.payload)[:, 1]
        assert np.any(scored != 0.0)
        assert _hidden_columns_are_zero(carry, study.caps)


def test_the_leaking_layout_is_caught_by_the_same_check():
    base = CORPUS_BY_NAME["baseline"].make_study()
    rh.SEEN.clear()
    study = rh.spied(rh.with_leaking_layout(base), reference=base.caps)
    truths, roots = worlds(2)
    carries = _run(study, "host", truths, roots, 2)
    assert rh.SEEN and any(total > 0.0 for total in rh.SEEN)
    # the snapshot check, judged against the protected layout, fails as well
    assert not _hidden_columns_are_zero(carries[0], study.caps, base.caps)


def _recovery(study, n_worlds=32):
    """RMS error of the policy's reconstruction and of the prior-mean guess."""
    roots = jax.vmap(lambda i: jr.fold_in(jr.PRNGKey(7), i))(jnp.arange(n_worlds))
    truths = jax.vmap(
        lambda r: toy.draw_truth(jr.fold_in(r, Purpose.WORLD), study.caps)
    )(roots)
    cs, _ = CompiledExecutor().run(study, truths, roots, N_STEPS_CORPUS)
    subject = np.asarray(cs.pstate[:, 3]).astype(int) - 1
    assert np.all(subject >= 0), "the policy never saw a released row"
    signal = np.asarray(truths.signal)[np.arange(n_worlds), subject]
    error = np.asarray(cs.pstate[:, 2]) - signal
    chance = 1.0 - signal  # a guess at the prior mean
    return float(np.sqrt(np.mean(error**2))), float(np.sqrt(np.mean(chance**2)))


def _reconstructing_study():
    return toy.survey_1(policies=(rh.ReconstructionPolicy(plan=PLAN5),))


def test_policy_cannot_recover_truth():
    err, chance = _recovery(_reconstructing_study())
    # the best it can do is the noisy value itself, which is worse than a prior guess
    assert err > chance


def test_the_leaking_layout_lets_the_same_policy_recover_truth():
    err, chance = _recovery(rh.with_leaking_layout(_reconstructing_study()))
    assert err < 1e-9 < chance


def test_the_log_keeps_the_scorer_column_for_the_scorer():
    study = CORPUS_BY_NAME["baseline"].make_study()
    truths, roots = worlds(1)
    carry, _ = HostExecutor().run(study, index_world(truths, 0), roots[0], 40)
    assert np.any(np.asarray(carry.log.payload)[:, 1] != 0.0)
    assert study.caps.payload_audience
