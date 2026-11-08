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

from spaceodyssey.core import HostExecutor, handlers
from spaceodyssey.core.conformance.invariants import balance_conserved
from spaceodyssey.core.enums import ChargePricing
from tests.core.helpers import one_world
from tests.core.program_helpers import cancel, observe, wait
from tests.core.refund_helpers import (
    apply_refund,
    before_release,
    mutate,
    receipt,
    refund_study,
)

pytestmark = pytest.mark.medium  # A cold host run compiles the executor.

# Every study here shares one compiled executor: two programs, and a log of three rows,
# which is the most a case with two receipts fills and the least that still overflows.
SHARED = dict(n_programs=2, n_log=3)
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


class _Runs(dict):
    """The runs by case, each made on first use."""

    def __missing__(self, name):
        self[name] = _run(refund_study(CANCELLED, **CASES[name], cap_kw=SHARED))
        return self[name]


@pytest.fixture(scope="module")
def runs():
    return _Runs()


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
    """A full log refuses the third receipt; the trace accounts the lost completion."""
    receipts = (receipt(200), receipt(210), receipt(220))
    study = refund_study(CANCELLED, **SURCHARGE, receipts=receipts, cap_kw=SHARED)
    run = _run(study)
    assert int(run.carry.n_log) == 3
    assert int(np.asarray(run.carry.overflow_mask)) != 0
    result = balance_conserved(run)
    assert result.passed, result


def _lost_completion_run():
    """Four completions of program 1 into a log of three rows: the fourth is lost."""
    script = (
        observe(0, 0, 10, 1),
        wait(1, 50),
        observe(2, 0, 10, 1),
        wait(3, 50),
        observe(4, 0, 10, 1),
        wait(5, 50),
        observe(6, 0, 10, 1),
    )
    study = refund_study(script, cap_kw=SHARED)
    return _run(study, n_steps=240)


def test_a_lost_completion_is_charged_to_the_program_that_paid_it():
    """Two programs, a full log: the refused completion belongs to program 1."""
    run = _lost_completion_run()
    study = run.study
    assert int(run.carry.n_log) == 3
    assert int(np.asarray(run.carry.overflow_mask)) != 0
    held = np.asarray(run.carry.ledger.balance)
    program_rows = held[study.caps.n_assets : study.caps.n_assets + 2, 1]
    np.testing.assert_allclose(program_rows, [100.0, 60.0])
    result = balance_conserved(run)
    assert result.passed, result


def test_a_debit_moved_to_the_wrong_program_is_caught_after_an_overflow():
    """Fails if the lost completion's program is read from the counters it checks."""
    run = _lost_completion_run()
    study = run.study
    row = study.caps.n_assets
    assert balance_conserved(run).passed
    ledger = run.carry.ledger
    moved = dataclasses.replace(
        ledger,
        debited=ledger.debited.at[row, 1].add(10.0).at[row + 1, 1].add(-10.0),
        balance=ledger.balance.at[row, 1].add(-10.0).at[row + 1, 1].add(10.0),
    )
    forged = SimpleNamespace(
        study=study, carry=dataclasses.replace(run.carry, ledger=moved), trace=run.trace
    )
    np.testing.assert_allclose(np.asarray(moved.balance)[row : row + 2, 1], [90, 70])
    assert not balance_conserved(forged).passed


# ---- mutants: the rule, changed, applied by hand to the carry before a release ------
#
# The real rule leaves the books conserved on the release it guards; each mutant
# applied to the same carry does not. See the matching tests in `test_refund.py`: the
# mutant is a changed `_apply_refund` run eagerly, so no program is compiled for it.


def _mutant_case(case, program, tick, old, new):
    script = (observe(0, 0, 10, program), wait(0, 60), cancel(1, 0))
    study = refund_study(script, **CASES[case], cap_kw=SHARED)
    before, slot = before_release(study, _run(study).trace, tick)
    real = apply_refund(handlers._apply_refund, before, slot)
    mutant = apply_refund(mutate(handlers._apply_refund, old, new), before, slot)
    return before, real, mutant


@pytest.mark.parametrize("program", [0, 1])
def test_mutant_a_refund_of_the_whole_charge_fails_conservation(program):
    """Mutant: the credit includes the surcharge; the ledger and receipt agree on it."""
    old = "log.charged[origin] - log.nonrefundable[origin]"
    before, real, mutant = _mutant_case(
        "refunded", program, 200, old, "log.charged[origin]"
    )
    assert int(before.carry.n_log) == 2  # the acquisition and the receipt
    assert balance_conserved(real).passed
    assert not balance_conserved(mutant).passed


@pytest.mark.parametrize("program", [0, 1])
def test_mutant_a_second_receipt_credit_fails_conservation(program):
    """Mutant: no once-only test, so two receipts credit one acquisition twice."""
    before, real, mutant = _mutant_case("refunded_twice", program, 210, " & ~paid", "")
    assert int(before.carry.n_log) == 3  # the acquisition and two receipts
    assert balance_conserved(real).passed
    assert not balance_conserved(mutant).passed
