"""The bracket study on the stored fixture: the oracle arm, real summary refit."""

import numpy as np
import pytest

from spaceodyssey.core.enums import AdmissionCode
from spaceodyssey.studies import bracket as br

from .bracket_helpers import N_WORLDS, build

pytestmark = pytest.mark.slow(reason="compiles the chunked refresh driver once")


@pytest.fixture(scope="module")
def built():
    return build()


def test_the_oracle_run_ends_at_the_horizon_without_overflow(built):
    """The run completes: no stall, overflow or invariant violation in any world."""
    carry = built[-1].carry
    assert bool(np.all(np.asarray(carry.done)))
    assert not bool(np.any(np.asarray(carry.overflow)))
    assert not bool(np.any(np.asarray(carry.stalled)))
    assert not bool(np.any(np.asarray(carry.violation)))


def test_the_shared_budget_binds_through_a_refused_admission(built):
    """Less than the cheapest visit is left and an admission is refused for funds."""
    _, study, _, _, _, run = built
    balance = np.asarray(run.carry.ledger.balance)[:, -1, br.SHARED_RESOURCE]
    cheapest = float(study.channels.cost[0, br.SHARED_RESOURCE])
    assert np.all(balance < cheapest)
    refused = np.asarray(run.trace.admit_code) == int(AdmissionCode.NO_FUNDS)
    assert np.all(refused.reshape(N_WORLDS, -1).sum(axis=1) >= 1)


def test_each_star_has_one_catalog_row_before_the_first_telescope_visit(built):
    """Kickoff observes the catalog mode of every star, ahead of any telescope mode."""
    source, study, _, _, _, run = built
    trace = run.trace
    catalog_id = br.FIRST_CHANNEL_ID + study.caps.n_channels - 1
    channel = np.asarray(trace.channel)
    subject = np.asarray(trace.subject)
    tick = np.asarray(trace.tick)
    for w in range(N_WORLDS):
        on_catalog = channel[w] == catalog_id
        on_telescope = (channel[w] >= br.FIRST_CHANNEL_ID) & ~on_catalog
        assert set(subject[w][on_catalog]) == set(range(len(source.stars)))
        assert tick[w][on_catalog].max() <= tick[w][on_telescope].min()
