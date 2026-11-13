"""The bracket policy waits only for refits of the visits a refit reads."""

import numpy as np
import pytest

from spaceodyssey.studies import bracket as br

from .bracket_helpers import build

pytestmark = pytest.mark.slow(reason="compiles the chunked refresh driver once")

BUDGET_D = 30.0


def test_an_arm_whose_policy_is_never_refit_still_spends_budget():
    """Fails if the never-refit policy waits for a refit instead of spending."""
    _, study, _, truths, _, run = build(
        arm="prior_planned",
        budget_d=BUDGET_D,
        p_new=0.3,
        occupancy=0.2,
        n_round=2**9,
        n_rounds=1,
        n_centers=16,
        n_constants=1,
        max_rounds=2,
        n_first=64,
        n_particles=128,
        ess_floor=1.0,
    )
    assert study.agenda.policies[0].refits_reach_policy is False
    table = br.outcome_table(run, truths, study)
    assert np.all(table.star["n_visits"].sum(axis=1) > 1)
    assert np.all(table.world["charged"] > 0.1 * BUDGET_D)


def test_an_arm_that_refits_the_policy_waits_for_it():
    """The oracle and posterior arms keep the wait; only the unrefit arm drops it."""
    for name in ("oracle", "replan", "prior_planned"):
        _, study, _, _, _, _ = build(arm=name, run=False, occupancy=0.2)
        reaches = study.agenda.policies[0].refits_reach_policy
        assert reaches == (name != "prior_planned"), name
