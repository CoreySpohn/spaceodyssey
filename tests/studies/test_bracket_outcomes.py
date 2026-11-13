"""The outcome table and `run_arms` on the fixture run (oracle arm, real refit)."""

import dataclasses
import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.studies import bracket as br

from .bracket_helpers import N_STEPS, N_WORLDS, build

# Enough telescope days for a second visit to one star in each world, and few
# enough that each world runs out of budget.
BUDGET_D = 9.0
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
    days = table.star["t_search"] + table.star["t_followup"]
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
    """Fails if imaging candidacy follows the policy summary instead of the observer."""
    study, _, truths, _, run, _ = built
    zeros = jax.tree_util.tree_map(jnp.zeros_like, run.carry.belief)
    held = dataclasses.replace(
        zeros,
        n_det=zeros.n_det.at[:, 0, 0].set(3),
    )
    observer_only = eqx.tree_at(
        lambda c: (c.belief, c.pbelief), run.carry, (held, zeros)
    )
    policy_only = eqx.tree_at(lambda c: (c.belief, c.pbelief), run.carry, (zeros, held))
    got = br.outcome_table(dataclasses.replace(run, carry=observer_only), truths, study)
    none = br.outcome_table(dataclasses.replace(run, carry=policy_only), truths, study)
    assert got.slot["candidate"][:, 0, 0].all()
    assert not none.slot["candidate"].any()
