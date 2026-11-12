"""The outcome table and `run_arms` on the fixture run (oracle arm, real refit)."""

import dataclasses
import itertools
import types

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.adapters.photomancy import N_GRID, DecisionSummary
from spaceodyssey.core.enums import EffectKind
from spaceodyssey.studies import bracket as br

from .bracket_helpers import N_STEPS, N_WORLDS, build

BUDGET_D = 8.0
pytestmark = pytest.mark.slow(reason="compiles the chunked refresh driver once")


@pytest.fixture(scope="module")
def built():
    _, study, executor, truths, roots, run = build(budget_d=BUDGET_D)
    table = br.outcome_table(run, truths, study)
    return study, executor, truths, roots, run, table


def test_the_table_agrees_with_the_log_and_the_ledger(built):
    """Visits, telescope days and refusals in the table equal the raw records."""
    _, _, _, _, run, table = built
    log = run.carry.log
    kind, valid = np.asarray(log.kind), np.asarray(log.valid)
    visits = ((kind == br.KIND_VISIT) & valid).sum(axis=1)
    np.testing.assert_array_equal(table.star["n_visits"].sum(axis=1), visits)
    days = table.star["t_search"] + table.star["t_followup"] + table.star["t_block"]
    np.testing.assert_allclose(days.sum(axis=1), table.world["charged"], atol=1e-9)
    assert np.all(table.world["no_funds"] >= 1)
    assert np.all(table.slot["detected_again"] <= table.slot["detected"])
    assert np.all(table.slot["detected"] <= table.slot["present"])


def test_a_visit_waits_for_a_refit_that_read_the_previous_one(built):
    """A star's next visit starts on a later day than the release of its last row."""
    _, _, _, _, run, _ = built
    log = run.carry.log
    kind, valid = np.asarray(log.kind), np.asarray(log.valid)
    subject, start = np.asarray(log.subject), np.asarray(log.start_tick)
    released = np.asarray(log.released_tick)
    checked = 0
    for w in range(N_WORLDS):
        rows = np.flatnonzero((kind[w] == br.KIND_VISIT) & valid[w])
        for s in set(subject[w][rows]):
            mine = rows[subject[w][rows] == s]
            mine = mine[np.argsort(start[w][mine])]
            for a, b in itertools.pairwise(mine):
                day = br.TICKS_PER_DAY
                assert start[w, b] // day > released[w, a] // day
                checked += 1
    assert checked > 0, "no star was visited twice"


def test_run_arms_is_deterministic_and_a_self_pair_is_zero(built):
    """Two runs of one arm give bitwise equal tables; the self gap is zero."""
    study, executor, truths, roots, _, table = built
    arms = {"a": (study, executor.boundary), "b": (study, executor.boundary)}
    out = br.run_arms(arms, truths, roots, N_STEPS)
    for name in ("slot", "star", "world"):
        for key, value in getattr(out.table["a"], name).items():
            reference = getattr(table, name)[key]
            np.testing.assert_array_equal(value, reference)
            np.testing.assert_array_equal(getattr(out.table["b"], name)[key], reference)
    mean, se, _, n = out.gap("a", "b", "charged")
    assert (mean, se, n) == (0.0, 0.0, N_WORLDS)
    jax.clear_caches()


def test_the_table_reads_the_observer_summary_not_the_policy_summary(built):
    """Endpoints follow `carry.belief`: a policy-only claim does not count."""
    study, _, truths, _, run, _ = built
    zeros = jax.tree_util.tree_map(jnp.zeros_like, run.carry.belief)
    held = dataclasses.replace(
        zeros,
        n_det=zeros.n_det.at[:, 0, 0].set(3),
        p_class=zeros.p_class.at[:, 0, 0].set(1.0),
        revealed=zeros.revealed.at[:, 0, 0].set(br.CLASS_ONE + br.REVEAL_OFFSET),
    )
    observer_only = eqx.tree_at(
        lambda c: (c.belief, c.pbelief), run.carry, (held, zeros)
    )
    policy_only = eqx.tree_at(lambda c: (c.belief, c.pbelief), run.carry, (zeros, held))
    got = br.outcome_table(dataclasses.replace(run, carry=observer_only), truths, study)
    none = br.outcome_table(dataclasses.replace(run, carry=policy_only), truths, study)
    assert got.slot["claimed"][:, 0, 0].all()
    assert not none.slot["claimed"].any()
    total = got.world["confirmed"]
    np.testing.assert_array_equal(total, np.ones(N_WORLDS))
    real = np.asarray(truths.present & (truths.klass == 1))[:, 0, 0]
    np.testing.assert_array_equal(got.world["false_claims"], ~real)


def _decide(built, budget_d):
    """One decision of a policy whose only choice is a block on slot 0 of star 0."""
    study, _, _, _, _, _ = built
    caps = study.caps
    policy = dataclasses.replace(study.agenda.policies[0], budget_d=budget_d)
    n, slots = caps.n_subjects, policy.n_slots
    belief = DecisionSummary.zeros(n, slots, N_GRID, br.N_DETECT, br.N_BLOCK, n_an=1)
    belief = dataclasses.replace(
        belief,
        n_det=belief.n_det.at[0, 0].set(3),
        p_class=belief.p_class.at[0, 0].set(1.0),
        status=jnp.zeros_like(belief.status),
        p_block=belief.p_block.at[0, 0].set(1.0),
    )
    pstate = jnp.zeros(caps.n_pstate).at[n * (3 + slots) + 1].set(float(n))
    view = types.SimpleNamespace(
        belief=belief, tick=jnp.asarray(0), n_dec=jnp.asarray(1)
    )
    effects, new, _ = policy.decide(pstate, view, study, jax.random.key(0))
    return effects, np.asarray(new)[: n * (3 + slots)].reshape(n, 3 + slots)


def test_a_block_that_does_not_fit_is_not_started(built):
    """At the budget edge the block is not run, not even as a stand-in visit."""
    study = built[0]
    assert _decide(built, 1000.0)[1][0, 3] == 1.0, (
        "with the budget free the block starts"
    )
    left = 2.0
    effects, stars = _decide(built, left)
    assert float(study.channels.cost[0, br.SHARED_RESOURCE]) < left
    assert stars[:, 3:].sum() == 0.0, "a visit set a block ordinal"
    assert stars[:, 1].sum() == 0.0, "a stand-in visit was made"
    assert int(effects.kind[0]) == EffectKind.WAIT
