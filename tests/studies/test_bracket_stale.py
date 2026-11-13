"""The bracket policy waits only for refits of the visits a refit reads."""

import numpy as np
import pytest

from spaceodyssey.studies import bracket as br

from .bracket_helpers import build

pytestmark = pytest.mark.slow(reason="compiles the chunked refresh driver once")

# Enough telescope days that the fixture's first claimed planet gets a block.
BUDGET_D = 30.0
# With the exozodi in the host's own colour, the second world's candidate is served
# only by the 45-day block at the default requirement, which the budget cannot
# cover; at a requirement of 5 (a lower resolving power) the 5-day block serves it.
# That world's next visits fall after the default 120-day horizon.
BLOCK_SNR_REQUIRED = 5.0
HORIZON_D = 180.0


def test_a_star_that_got_a_block_is_visited_again():
    """After a block the policy still acts: a block is not a visit a refit reads."""
    _, study, _, truths, _, run = build(
        budget_d=BUDGET_D, horizon_d=HORIZON_D, block_snr_required=BLOCK_SNR_REQUIRED
    )
    log = run.carry.log
    kind, valid = np.asarray(log.kind), np.asarray(log.valid)
    start = np.asarray(log.start_tick)
    table = br.outcome_table(run, truths, study)
    assert np.all(table.star["n_block"].sum(axis=1) >= 1), "no block was started"
    for w in range(kind.shape[0]):
        blocks = start[w][(kind[w] == br.KIND_BLOCK) & valid[w]]
        visits = start[w][(kind[w] == br.KIND_VISIT) & valid[w]]
        assert np.any(visits > blocks.min())


def test_an_arm_whose_policy_is_never_refit_still_spends_budget():
    """The policy of `prior_planned` never waits for a refit that cannot come."""
    _, study, _, truths, _, run = build(
        arm="prior_planned", budget_d=BUDGET_D, p_new=0.3, occupancy=0.2
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
