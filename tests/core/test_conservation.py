"""Balance conservation under charge components, cancels and refunds.

`balance_conserved` holds three independent accounts to each other: the final balance,
the ledger's debit and refund counters, and the retained records. A refund receipt
carries a negative `charged` on its own row, so the records still net to the ledger.
"""

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import HostExecutor, handlers
from spaceodyssey.core.conformance.invariants import balance_conserved
from spaceodyssey.core.enums import ChargePricing, RemainderRule
from tests.core import award_helpers
from tests.core.helpers import one_world
from tests.core.program_helpers import cancel, end, observe, wait
from tests.core.refund_helpers import mutate, receipt, refund_study

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


# awards on, one compiled executor for all: (script, awards, receipts)
FOREVER = award_helpers.award(0, 2, 2, 0, 1000, 5, RemainderRule.FORFEIT)
EXPIRED = award_helpers.award(0, 2, 2, 0, 60, 5, RemainderRule.CARRY_OVER)
AWARD_CASES = {
    "cancel_under_award": (CANCELLED, (FOREVER,), ()),
    "observe_after_expiry": (
        (wait(0, 100), observe(1, 0, 10, 0), end(2)),
        (EXPIRED,),
        (),
    ),
}


class _Runs(dict):
    """The runs by case, each made on first use."""

    def __missing__(self, name):
        if name in AWARD_CASES:
            script, awards, receipts = AWARD_CASES[name]
            study = award_helpers.award_refund_study(
                script, awards, receipts, pricing=None
            )
        else:
            study = refund_study(CANCELLED, **CASES[name])
        self[name] = _run(study)
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


@pytest.mark.slow  # a one-row log is static: each study compiles, 17 s cold
def test_the_check_still_holds_when_the_log_overflows():
    """A one-row log refuses the receipt; the trace accounts the lost completion."""
    study = refund_study(
        CANCELLED, **SURCHARGE, receipts=(receipt(200),), cap_kw=dict(n_log=1)
    )
    result = balance_conserved(_run(study))
    assert result.passed, result


@pytest.mark.slow  # a one-row log is static: each study compiles, 17 s cold
def test_a_lost_completion_is_charged_to_the_program_that_paid_it():
    """Two programs, a one-row log: the refused completion belongs to program 1."""
    script = (observe(0, 0, 10, 1), wait(1, 200), observe(2, 0, 10, 1))
    study = refund_study(script, cap_kw=dict(n_programs=2, n_log=1))
    run = _run(study, n_steps=120)
    assert int(run.carry.n_log) == 1
    assert int(np.asarray(run.carry.overflow_mask)) != 0
    held = np.asarray(run.carry.ledger.balance)
    program_rows = held[study.caps.n_assets : study.caps.n_assets + 2, 1]
    np.testing.assert_allclose(program_rows, [100.0, 80.0])
    result = balance_conserved(run)
    assert result.passed, result


@pytest.mark.slow  # a one-row log is static: each study compiles, 17 s cold
def test_a_debit_moved_to_the_wrong_program_is_caught_after_an_overflow():
    """Fails if the lost completion's program is read from the counters it checks."""
    script = (observe(0, 0, 10, 1), wait(1, 200), observe(2, 0, 10, 1))
    study = refund_study(script, cap_kw=dict(n_programs=2, n_log=1))
    run = _run(study, n_steps=120)
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
    np.testing.assert_allclose(np.asarray(moved.balance)[row : row + 2, 1], [90, 90])
    assert not balance_conserved(forged).passed


def _mutant(monkeypatch, old, new):
    monkeypatch.setattr(
        handlers, "_apply_refund", mutate(handlers._apply_refund, old, new)
    )
    jax.clear_caches()  # the compiled step closed over the real handler


@pytest.mark.slow  # A mutant clears the compile caches and compiles the executor cold.
@pytest.mark.parametrize("program", [0, 1])
def test_mutant_a_refund_of_the_whole_charge_fails_conservation(monkeypatch, program):
    """Mutant: the credit includes the surcharge; the ledger and receipt agree on it."""
    _mutant(
        monkeypatch,
        "log.charged[origin] - log.nonrefundable[origin]",
        "log.charged[origin]",
    )
    try:
        script = (observe(0, 0, 10, program), wait(0, 60), cancel(1, 0))
        study = refund_study(
            script,
            **SURCHARGE,
            receipts=(receipt(200),),
            cap_kw=dict(n_programs=2),
        )
        run = _run(study)
        assert int(run.carry.n_log) == 2  # the acquisition and the receipt
        assert not balance_conserved(run).passed
    finally:
        jax.clear_caches()


@pytest.mark.slow  # A mutant clears the compile caches and compiles the executor cold.
@pytest.mark.parametrize("program", [0, 1])
def test_mutant_a_second_receipt_credit_fails_conservation(monkeypatch, program):
    """Mutant: no once-only test, so two receipts credit one acquisition twice."""
    _mutant(monkeypatch, " & ~paid", "")
    try:
        script = (observe(0, 0, 10, program), wait(0, 60), cancel(1, 0))
        study = refund_study(
            script,
            **SURCHARGE,
            receipts=(receipt(200), receipt(210)),
            cap_kw=dict(n_programs=2),
        )
        run = _run(study)
        assert int(run.carry.n_log) == 3  # the acquisition and two receipts
        assert not balance_conserved(run).passed
    finally:
        jax.clear_caches()


def test_a_cancellation_under_an_award_conserves(runs):
    """Fails if the replay drops a cancel's credit from the refunded account.

    The lane is cancelled at tick 60, half spent: the ledger credits 5, and the replay
    of the awards must expect the same credit in `refunded` and in the balance.
    """
    run = runs["cancel_under_award"]
    assert float(np.asarray(run.carry.ledger.refunded)[2, 1]) == 5.0
    assert balance_conserved(run).passed


def test_a_debit_after_its_window_expired_names_generation_zero(runs):
    """Fails if a debit on a row whose window has closed captures that row's generation.

    The carry-over award ends at tick 60, so the row is unwindowed at tick 100: the
    record names generation 0 and the rule NONE (not the row's generation 2), and the
    replay accepts it.
    """
    run = runs["observe_after_expiry"]
    log = run.carry.log
    assert int(np.asarray(run.carry.ledger.win_gen)[2]) == 2
    assert not np.any(np.asarray(log.award_gens)[0])
    assert not np.any(np.asarray(log.award_rules)[0])
    assert balance_conserved(run).passed


WINDOWED_SCRIPT = (
    wait(0, 40),
    observe(1, 0, 10, 0),
    wait(1, 270),
    wait(2, 140),
    wait(3, 100),
    end(5),
)


def _re_awarded(receipts):
    """Award A forfeits at 300, B (carry over) opens at 310; the refund is at 350."""
    first = award_helpers.award(0, 2, 2, 0, 300, 5, RemainderRule.FORFEIT)
    again = award_helpers.award(3, 2, 2, 310, 360, 7, RemainderRule.CARRY_OVER)
    study = award_helpers.award_refund_study(
        WINDOWED_SCRIPT, (first, again), receipts, pricing=None
    )
    return _run(study)


def test_a_refund_across_an_expiry_and_a_re_award_conserves():
    """Control: the late refund lapses, and the per-row account and the replay agree."""
    run = _re_awarded((receipt(350),))
    ledger = run.carry.ledger
    assert float(np.asarray(ledger.refunded)[2, 1]) == 10.0
    assert float(np.asarray(ledger.lapsed)[2, 1]) >= 10.0
    assert balance_conserved(run).passed


@pytest.mark.slow  # A mutant clears the compile caches and compiles the executor cold.
def test_mutant_crediting_an_old_awards_refund_to_the_newer_award_fails(monkeypatch):
    """Mutant: the refund always credits the row, so a closed award raises the new one.

    The ledger counters and the record agree with the credit; only the independent
    replay of the disposition refuses it.
    """
    _mutant(monkeypatch, "same = led.win_gen[rows] == gens", "same = rows >= 0")
    try:
        run = _re_awarded((receipt(350),))
        assert not balance_conserved(run).passed
    finally:
        jax.clear_caches()
