"""Balance conservation under charge components, cancels and refunds.

`balance_conserved` holds three independent accounts to each other: the final balance,
the ledger's debit and refund counters, and the retained records. A refund receipt
carries a negative `charged` on its own row, so the records still net to the ledger.
"""

import dataclasses
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import HostExecutor
from spaceodyssey.core.conformance.invariants import balance_conserved
from spaceodyssey.core.enums import ChargePricing
from tests.core.helpers import one_world
from tests.core.program_helpers import cancel, observe, wait
from tests.core.refund_helpers import receipt, refund_study

pytestmark = pytest.mark.medium  # A cold host run compiles the executor.

CANCELLED = (observe(0, 0, 10, 0), wait(0, 60), cancel(1, 0))
SURCHARGE = dict(charge=10.0, pricing=ChargePricing.SURCHARGE, surcharge=2.0)
CASES = {
    "plain": dict(),
    "cancel_with_surcharge": dict(**SURCHARGE),
    "refunded": dict(**SURCHARGE, receipts=(receipt(200),)),
    "refunded_twice": dict(**SURCHARGE, receipts=(receipt(200), receipt(210))),
    "ignored_receipt": dict(**SURCHARGE, receipts=(receipt(200, index=7),)),
}


def _run(study, n_steps=80):
    truth, root = one_world(caps=study.caps)
    result = HostExecutor().run(study, truth, root, n_steps)
    return SimpleNamespace(study=study, carry=result.carry, trace=result.trace)


@pytest.fixture(scope="module")
def runs():
    return {n: _run(refund_study(CANCELLED, **kw)) for n, kw in CASES.items()}


@pytest.mark.parametrize("name", CASES)
def test_every_account_agrees_after_pricing_cancels_and_refunds(runs, name):
    """Fails if a charge component or a credit moves the balance without the ledger."""
    result = balance_conserved(runs[name])
    assert result.passed, result


@pytest.mark.parametrize("name", CASES)
def test_each_row_nets_debits_and_refunds(runs, name):
    """Final balance is initial minus debited plus refunded, in every row."""
    run = runs[name]
    ledger = run.carry.ledger
    expected = (
        np.asarray(run.study.budget)
        - np.asarray(ledger.debited)
        + np.asarray(ledger.refunded)
    )
    np.testing.assert_allclose(np.asarray(ledger.balance), expected, atol=1e-9)


def test_a_refund_nets_the_surcharge_out_of_the_credit(runs):
    """Debited 12, credited 5 on the cancel and 5 on the failure: 2 stays spent."""
    ledger = runs["refunded"].carry.ledger
    assert float(ledger.debited[0, 0]) == pytest.approx(12.0)
    assert float(ledger.refunded[0, 0]) == pytest.approx(10.0)


def test_a_forged_balance_is_caught(runs):
    """The ledger's counters and records cannot be satisfied by a moved balance."""
    run = runs["refunded"]
    ledger = dataclasses.replace(run.carry.ledger, balance=run.carry.ledger.balance + 1)
    forged = SimpleNamespace(
        study=run.study,
        carry=dataclasses.replace(run.carry, ledger=ledger),
        trace=run.trace,
    )
    assert not balance_conserved(forged).passed


def test_a_forged_consistent_balance_and_counter_is_caught_by_the_records(runs):
    """Balance and refund counter moved together still fail against the records."""
    run = runs["refunded"]
    ledger = run.carry.ledger
    forged_ledger = dataclasses.replace(
        ledger,
        balance=ledger.balance.at[0, 0].add(3.0),
        refunded=ledger.refunded.at[0, 0].add(3.0),
    )
    forged = SimpleNamespace(
        study=run.study,
        carry=dataclasses.replace(run.carry, ledger=forged_ledger),
        trace=run.trace,
    )
    assert not balance_conserved(forged).passed


def test_a_receipt_row_that_loses_its_negative_charge_is_caught(runs):
    """Without the credit on the receipt row the records overstate the net spend."""
    run = runs["refunded"]
    log = run.carry.log
    row = int(run.carry.n_log) - 1
    log = dataclasses.replace(
        log, charged=log.charged.at[row].set(jnp.zeros_like(log.charged[row]))
    )
    forged = SimpleNamespace(
        study=run.study,
        carry=dataclasses.replace(run.carry, log=log),
        trace=run.trace,
    )
    assert not balance_conserved(forged).passed


def test_the_check_still_holds_when_the_log_overflows():
    """A one-row log refuses the receipt; the trace accounts the lost completion."""
    study = refund_study(
        CANCELLED, **SURCHARGE, receipts=(receipt(200),), cap_kw=dict(n_log=1)
    )
    result = balance_conserved(_run(study))
    assert result.passed, result
