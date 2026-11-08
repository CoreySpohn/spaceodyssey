"""Failure receipts and refunds, then the windowed part (late refunds and awards).

A failure receipt is a process-written record of a kind a channel declares as a failure
cause. It names its target by semantic identity (subject, channel, acquisition index)
and credits the refundable remainder of that acquisition, once, when it is released to
the policy: in the release stage, never when the process writes it. Each check has a
control or a mutant of the rule it guards. The runs share one compiled executor.
"""

import dataclasses
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import spaceodyssey.core.snapshot as snapshot
from spaceodyssey.core import (
    Audience,
    EventKind,
    HostExecutor,
    handlers,
    snapshot_fingerprint,
    snapshot_view,
)
from spaceodyssey.core.conformance.invariants import balance_conserved
from spaceodyssey.core.enums import I64, ChargePricing, RemainderRule
from spaceodyssey.core.preflight import FindingCode, preflight
from spaceodyssey.core.records import masked_record, record_none
from spaceodyssey.studies import toy
from tests.core import award_helpers
from tests.core.helpers import one_world
from tests.core.program_helpers import cancel, end, observe, wait
from tests.core.refund_helpers import mutate, receipt, refund_study

pytestmark = pytest.mark.medium  # A cold host run compiles the executor.

N_STEPS = 80
CANCELLED = (observe(0, 0, 10, 0), wait(0, 60), cancel(1, 0))  # half spent at tick 60
COMPLETED = (observe(0, 0, 10, 0),)
# occupancy 10 plus a surcharge of 2: debited 12; cancelled halfway it retains 5 + 2
SURCHARGE = dict(charge=10.0, pricing=ChargePricing.SURCHARGE, surcharge=2.0)
OBSERVER_ONLY = int(Audience.OBSERVER)
N_ASSETS = refund_study().caps.n_assets


def _run(study, n_steps=N_STEPS):
    truth, root = one_world(caps=study.caps)
    return HostExecutor().run(study, truth, root, n_steps)


def _held(run):
    """The balance of the asset, program and shared rows, one resource each."""
    ledger = run.carry.ledger
    last = ledger.balance.shape[0] - 1
    return np.asarray(ledger.balance)[[0, N_ASSETS, last], [0, 1, 2]]


def _fingerprint(study, carry):
    view = snapshot_view(carry, Audience.POLICY, study.caps)
    return int(snapshot_fingerprint(view, study.observer.layout(study.caps)))


SCENARIOS = {
    "base": dict(),
    "applied": dict(receipts=(receipt(200),)),
    "duplicate": dict(receipts=(receipt(200), receipt(210))),
    "late": dict(receipts=(receipt(200),), deadline=100),
    "in_time": dict(receipts=(receipt(150),), deadline=100),
    "wrong_index": dict(receipts=(receipt(200, index=7),)),
    "wrong_subject": dict(receipts=(receipt(200, subject=1),)),
    "wrong_channel": dict(receipts=(receipt(200, channel=11),)),
    "held": dict(receipts=(receipt(200, held=True),)),
    "observer_only": dict(receipts=(receipt(200, audience=OBSERVER_ONLY),), failure=0),
    "private_visible_kind": dict(receipts=(receipt(200, audience=OBSERVER_ONLY),)),
    "private_elsewhere": dict(
        receipts=(receipt(200, audience=OBSERVER_ONLY, index=7),)
    ),
    "shifted": dict(receipts=(receipt(150, audience=OBSERVER_ONLY), receipt(200))),
}


@pytest.fixture(scope="module")
def runs():
    out = {}
    for name, kw in SCENARIOS.items():
        study = refund_study(CANCELLED, **SURCHARGE, **kw)
        out[name] = (study, _run(study))
    return out


def _ignored(run):
    return int(run.carry.ledger.refund_ignored)


def _receipt_row(run):
    """The receipt's log slot: the last row written."""
    return int(run.carry.n_log) - 1


# ---- the credit is the refundable remainder -----------------------------------------


def test_a_receipt_credits_the_remainder_without_the_surcharge(runs):
    """Fails if the surcharge comes back: debited 12, retained 7, refundable 5."""
    np.testing.assert_allclose(_held(runs["base"][1]), 93.0)
    study, run = runs["applied"]
    np.testing.assert_allclose(_held(run), 98.0)
    assert _ignored(run) == 0
    log = run.carry.log
    row = _receipt_row(run)
    np.testing.assert_allclose(np.asarray(log.charged)[row], -5.0)
    np.testing.assert_allclose(np.asarray(log.nonrefundable)[0], 2.0)
    # the ledger's own counter agrees: 5 credited on the cancel, 5 on the failure
    refunded = np.asarray(run.carry.ledger.refunded)
    np.testing.assert_allclose(refunded[[0, study.caps.n_assets], [0, 1]], 10.0)


def test_the_receipt_row_records_the_target_it_resolved(runs):
    """The internal id is on the row; the visible identity is the semantic index."""
    _, run = runs["applied"]
    log, row = run.carry.log, _receipt_row(run)
    assert int(log.refund_of[row]) == int(log.acquisition_id[0]) >= 0
    assert int(log.refund_of[0]) == -1  # the cancelled record names no target
    assert int(log.refund_index[row]) == int(log.acquisition_index[0])


def test_a_second_receipt_for_one_acquisition_is_ignored(runs):
    """Fails if a failure credits twice: the credit is applied once."""
    _, run = runs["duplicate"]
    np.testing.assert_allclose(_held(run), 98.0)
    assert _ignored(run) == 1


@pytest.mark.parametrize("name", ["late", "wrong_index", "wrong_subject"])
def test_a_receipt_that_does_not_apply_credits_nothing_and_counts(runs, name):
    """Expired claim, an acquisition that does not exist, another subject's."""
    _, run = runs[name]
    np.testing.assert_allclose(_held(run), 93.0)
    assert _ignored(run) == 1
    assert int(run.carry.log.refund_of[_receipt_row(run)]) == -1


def test_a_receipt_of_a_kind_the_channel_does_not_declare_credits_nothing(runs):
    """Channel 11 declares no failure kind, so the receipt is not a cause there."""
    _, run = runs["wrong_channel"]
    np.testing.assert_allclose(_held(run), 93.0)
    assert _ignored(run) == 1


def test_a_receipt_of_the_kind_of_its_own_origin_still_credits():
    """Fails if a receipt is ignored because its kind equals the origin's kind."""
    study = refund_study(
        CANCELLED, **SURCHARGE, receipts=(receipt(200),), failure=toy.RECORD_MEASUREMENT
    )
    run = _run(study)
    log = run.carry.log
    kinds = np.asarray(log.kind)[np.asarray(log.valid)]
    assert len(kinds) == 2 and kinds[0] == kinds[1] == toy.RECORD_MEASUREMENT
    np.testing.assert_allclose(_held(run), 98.0)
    assert _ignored(run) == 0


@pytest.mark.slow  # its own channel columns compile the executor, 19 s cold
def test_an_acquisition_of_the_failure_kind_is_not_a_receipt():
    """Fails if a second acquisition of the failure kind refunds the first."""
    script = (observe(0, 0, 10, 0), observe(1, 0, 10, 0))
    study = refund_study(script, charge=10.0, failure=toy.RECORD_MEASUREMENT)
    run = _run(study)
    log = run.carry.log
    valid = np.asarray(log.valid)
    assert valid.sum() == 2
    assert (np.asarray(log.subject)[valid] == 0).all()
    np.testing.assert_allclose(_held(run)[0], 80.0)
    assert _ignored(run) == 0


def test_control_a_receipt_inside_the_deadline_credits(runs):
    """Control for the expired claim: the target ended at tick 60, deadline 100."""
    _, run = runs["in_time"]
    np.testing.assert_allclose(_held(run), 98.0)
    assert _ignored(run) == 0


@pytest.mark.slow  # its own channel columns compile the executor, 19 s cold
def test_the_remainder_of_a_completed_acquisition_excludes_its_surcharge():
    """Completed, never cancelled: charged 12, of which 2 is fixed, so 10 returns."""
    study = refund_study(COMPLETED, **SURCHARGE, receipts=(receipt(200),))
    run = _run(study)
    np.testing.assert_allclose(_held(run), 98.0)
    assert _ignored(run) == 0


@pytest.mark.slow  # its own channel columns compile the executor, 19 s cold
def test_a_flat_price_is_refundable_on_a_failure_but_not_on_a_cancel():
    """Flat 4: the cancel returns nothing (96); the failure returns the 4 (100)."""
    flat = dict(charge=4.0, pricing=ChargePricing.FLAT_FRACTION)
    study = refund_study(CANCELLED, **flat)
    np.testing.assert_allclose(_held(_run(study)), 96.0)
    study = refund_study(CANCELLED, **flat, receipts=(receipt(200),))
    np.testing.assert_allclose(_held(_run(study)), 100.0)


# ---- the settlement boundary --------------------------------------------------------


def test_a_held_receipt_credits_nothing_and_changes_no_balance(runs):
    """Written but never released: no credit, no counter, no target resolved."""
    _, run = runs["held"]
    np.testing.assert_allclose(_held(run), 93.0)
    assert _ignored(run) == 0
    row = _receipt_row(run)
    assert int(run.carry.log.released_tick[row]) == -1
    assert int(run.carry.log.refund_of[row]) == -1


def test_an_observer_only_receipt_is_refused_by_preflight_and_moves_nothing(runs):
    """W002 refuses the kind; bypassed, the policy balance and fingerprint hold."""
    study, run = runs["observer_only"]
    codes = {f.code for f in preflight(study)}
    assert FindingCode.W002 in codes
    np.testing.assert_allclose(_held(run), 93.0)
    assert _ignored(run) == 0
    assert int(run.carry.log.refund_of[_receipt_row(run)]) == -1
    # another private receipt, of a kind the policy can read, changes nothing either
    other_study, other = runs["private_visible_kind"]
    assert _fingerprint(study, run.carry) == _fingerprint(other_study, other.carry)
    elsewhere_study, elsewhere = runs["private_elsewhere"]
    assert _fingerprint(study, run.carry) == _fingerprint(
        elsewhere_study, elsewhere.carry
    )
    np.testing.assert_allclose(_held(other), 93.0)


def test_control_the_same_receipt_released_to_the_policy_credits(runs):
    """Control: only the audience kept the observer-only receipt from crediting."""
    np.testing.assert_allclose(_held(runs["applied"][1]), 98.0)


def test_a_delayed_receipt_credits_at_its_release_tick_not_when_written():
    """Written at 150 and released at 190: the balance holds until the release."""
    study = refund_study(CANCELLED, **SURCHARGE, receipts=(receipt(150, delay=40),))
    full = _run(study)
    t = full.trace
    release = np.flatnonzero(
        np.asarray(t.handled)
        & (np.asarray(t.kind) == EventKind.RELEASE)
        & (np.asarray(t.tick) == 190)
    )
    assert len(release) == 1
    before = _run(study, int(release[0]))  # every step but the release
    np.testing.assert_allclose(_held(before), 93.0)
    assert int(before.carry.log.released_tick[_receipt_row(before)]) == -1
    np.testing.assert_allclose(_held(full), 98.0)
    assert int(full.carry.log.released_tick[_receipt_row(full)]) == 190


# ---- masking: no queue identity in any view -----------------------------------------


def test_a_view_shows_the_legacy_sentinel_and_the_semantic_index(runs):
    """`refund_of` is -1 where it is -1 today; the receipt shows only its index."""
    study, run = runs["applied"]
    assert int(run.carry.log.refund_of[_receipt_row(run)]) >= 0
    for audience in (Audience.POLICY, Audience.OBSERVER):
        view = snapshot_view(run.carry, audience, study.caps)
        n = int(view.n_rows)
        assert n == 2
        assert (np.asarray(view.log.refund_of)[:n] == -1).all()
        assert (np.asarray(view.log.refund_of)[n:] == 0).all()  # padding as before
        assert int(view.log.refund_index[n - 1]) == 0


def test_a_masked_record_hides_a_resolved_refund_target():
    """Fails if the observer's fold sees the queue-derived id of a refund target."""
    caps = refund_study(COMPLETED).caps
    rec = dataclasses.replace(record_none(caps), refund_of=jnp.asarray(5, I64))
    assert int(masked_record(rec, caps, Audience.OBSERVER).refund_of) == -1
    assert int(masked_record(record_none(caps), caps, Audience.POLICY).refund_of) == -1


def test_the_policy_fingerprint_ignores_a_private_event_before_the_refund(runs):
    """K6: an observer-only receipt shifts the target's internal id, not the view."""
    study, plain = runs["applied"]
    _, shifted = runs["shifted"]
    ids = [int(r.carry.log.acquisition_id[0]) for r in (plain, shifted)]
    assert ids[0] != ids[1]  # the queue-derived identity did move
    assert _fingerprint(study, plain.carry) == _fingerprint(study, shifted.carry)


def test_mutant_a_visible_internal_refund_id_moves_the_fingerprint(runs, monkeypatch):
    """Mutant: the view shows `refund_of` as stored. The K6 equality must fail."""
    study, plain = runs["applied"]
    _, shifted = runs["shifted"]
    monkeypatch.setattr(snapshot, "hide_refund_of", lambda x: x)
    assert _fingerprint(study, plain.carry) != _fingerprint(study, shifted.carry)


# ---- mutants (last: each clears the compile caches) ---------------------------------


def _with_apply(monkeypatch, apply):
    monkeypatch.setattr(handlers, "_apply_refund", apply)
    jax.clear_caches()  # the compiled step closed over the real handler


@pytest.mark.slow  # A mutant clears the compile caches and compiles the executor cold.
def test_mutant_a_refund_of_the_whole_charge_is_caught(monkeypatch):
    """Mutant: the credit is the retained charge, surcharge included (7, not 5)."""
    old = "log.charged[origin] - log.nonrefundable[origin]"
    _with_apply(monkeypatch, mutate(handlers._apply_refund, old, "log.charged[origin]"))
    try:
        study = refund_study(CANCELLED, **SURCHARGE, receipts=(receipt(200),))
        np.testing.assert_allclose(_held(_run(study)), 100.0)  # not 98
    finally:
        jax.clear_caches()


@pytest.mark.slow  # A mutant clears the compile caches and compiles the executor cold.
def test_mutant_a_refund_applied_twice_is_caught(monkeypatch):
    """Mutant: no once-only test, so a second receipt credits again."""
    _with_apply(monkeypatch, mutate(handlers._apply_refund, " & ~paid", ""))
    try:
        study = refund_study(
            CANCELLED, **SURCHARGE, receipts=(receipt(200), receipt(210))
        )
        np.testing.assert_allclose(_held(_run(study)), 103.0)  # not 98
    finally:
        jax.clear_caches()


@pytest.mark.slow  # A mutant clears the compile caches and compiles the executor cold.
def test_mutant_a_refund_past_its_deadline_is_caught(monkeypatch):
    """Mutant: no deadline test, so the expired claim credits."""
    _with_apply(monkeypatch, mutate(handlers._apply_refund, " & in_time", ""))
    try:
        study = refund_study(
            CANCELLED, **SURCHARGE, receipts=(receipt(200),), deadline=100
        )
        np.testing.assert_allclose(_held(_run(study)), 98.0)  # not 93
    finally:
        jax.clear_caches()


@pytest.mark.slow  # A mutant clears the compile caches and compiles the executor cold.
def test_mutant_a_refund_on_any_release_leaks_an_observer_only_receipt(monkeypatch):
    """Mutant: the credit applies on any release, not the policy's first."""
    old = "        first\n        & rec.valid\n"
    _with_apply(monkeypatch, mutate(handlers._apply_refund, old, "        rec.valid\n"))
    try:
        study = refund_study(
            CANCELLED,
            **SURCHARGE,
            receipts=(receipt(200, audience=OBSERVER_ONLY),),
            failure=0,
        )
        np.testing.assert_allclose(_held(_run(study)), 98.0)  # not 93
    finally:
        jax.clear_caches()


# ---- refunds across award windows -------------------------------------------------

# Program 0's row is balance row 2 and program 1's is 3; the program-scoped resource is
# column 1. Decision 0 awards at tick 0, decision 1 observes at tick 40 (the debit of
# 10 under that award), decision 2 (woken by the release at 170) only waits, and
# decision 3 acts at tick 310, after the first window ends at 300. The failure receipt
# for that observation is released at tick 350.
P0, P1, COL = 2, 3, 1
FIRST = award_helpers.award(0, 2, P0, 0, 300, 5, RemainderRule.FORFEIT)
WIN_SCRIPT = (
    wait(0, 40),
    observe(1, 0, 10, 0),
    wait(1, 270),
    wait(2, 140),
    wait(3, 100),
    end(5),
)


LATE = (receipt(350),)
INSIDE = (receipt(250),)
SECOND = award_helpers.award(3, 2, P0, 310, 360, 7, RemainderRule.CARRY_OVER)


def _then(rule, to=0):
    """The first award, then a second one on the same row after it, with `rule`."""
    return (FIRST, award_helpers.award(3, 2, P0, 310, 360, 7, rule, to))


# name -> (awards, receipt tick, overrides): every run shares one compiled executor
WINDOW_SCENARIOS = {
    "forfeited_then_none": (_then(RemainderRule.NONE), LATE, {}),
    "forfeited_then_carry": (_then(RemainderRule.CARRY_OVER), LATE, {}),
    "forfeited_then_forfeit": (_then(RemainderRule.FORFEIT), LATE, {}),
    "ends_always": (
        (FIRST, award_helpers.award(3, 2, P0, 310, 340, 20, RemainderRule.CARRY_OVER)),
        LATE,
        dict(charge=5.0),
    ),
    "idle": ((award_helpers.award(0, 2, P0, 0, 300, 5, RemainderRule.NONE),), LATE, {}),
    "none_then_newer": (
        (award_helpers.award(0, 2, P0, 0, 300, 5, RemainderRule.NONE), *_then(0)[1:]),
        LATE,
        {},
    ),
    "reallocated": (
        (award_helpers.award(0, 2, P0, 0, 300, 5, RemainderRule.REALLOCATE, P1),),
        LATE,
        {},
    ),
    "inside": (
        (award_helpers.award(0, 2, P0, 0, 600, 5, RemainderRule.FORFEIT),),
        INSIDE,
        {},
    ),
}


class _Windowed(dict):
    """The run of a scenario, and of its twin without the receipt, each made once."""

    def __missing__(self, key):
        name, with_receipt = key
        awards, receipts, overrides = WINDOW_SCENARIOS[name]
        study = award_helpers.award_refund_study(
            WIN_SCRIPT,
            awards,
            receipts if with_receipt else (),
            pricing=None,
            **overrides,
        )
        truth, root = one_world(caps=study.caps)
        result = HostExecutor().run(study, truth, root, 80)
        self[key] = SimpleNamespace(study=study, carry=result.carry, trace=result.trace)
        return self[key]


@pytest.fixture(scope="module")
def windowed():
    return _Windowed()


def _cell(run, name, row=P0):
    return float(np.asarray(getattr(run.carry.ledger, name))[row, COL])


def _change(windowed, name, counter, row=P0):
    """What the receipt changed in a ledger counter: the run with it less without."""
    return _cell(windowed[name, True], counter, row) - _cell(
        windowed[name, False], counter, row
    )


# scenario -> (balance change on P0, on P1, refunded change on the row that took it,
# lapsed change on P0); the credit is 10 (5 in "ends_always")
DISPOSITIONS = {
    # a forfeited award's refund lapses whatever the row holds now
    "forfeited_then_none": (0.0, 0.0, 10.0, 10.0),
    "forfeited_then_carry": (0.0, 0.0, 10.0, 10.0),
    "forfeited_then_forfeit": (0.0, 0.0, 10.0, 10.0),
    # a row left ALWAYS by a later award's expiry does not take an old refund
    "ends_always": (0.0, 0.0, 5.0, 5.0),
    # an IDLE row with no newer award takes the credit
    "idle": (10.0, 0.0, 10.0, 0.0),
    # an old NONE award's refund lapses under a newer award
    "none_then_newer": (0.0, 0.0, 10.0, 10.0),
    # a late REALLOCATE refund credits the destination row, not the expired one
    "reallocated": (0.0, 10.0, 10.0, 0.0),
    # control: released while the award is open, it credits the row (the release
    # wakes a decision earlier, so only the refund itself is compared with the twin)
    "inside": (None, None, 10.0, None),
}


@pytest.mark.parametrize("name", DISPOSITIONS)
def test_a_late_refund_is_disposed_by_the_window_that_charged_it(windowed, name):
    """Fails if a refund lands on a newer award, a later state or the expired row.

    Each scenario is compared with its twin run without the receipt, so the change is
    the refund alone; conservation holds against the independent replay.
    """
    on_p0, on_p1, refunded, lapsed = DISPOSITIONS[name]
    taker = P1 if name == "reallocated" else P0
    if on_p0 is not None:
        assert _change(windowed, name, "balance") == on_p0
        assert _change(windowed, name, "balance", P1) == on_p1
    assert _change(windowed, name, "refunded", taker) == refunded
    if on_p0 is not None:
        assert _change(windowed, name, "lapsed") == lapsed
    assert balance_conserved(windowed[name, True]).passed


def test_a_refund_of_five_to_a_row_left_always_keeps_its_balance(windowed):
    """The balance the row ends with is B's own 20: the refund of exactly 5 lapsed."""
    run = windowed["ends_always", True]
    assert _cell(run, "balance") == _cell(windowed["ends_always", False], "balance")
    assert (
        _cell(run, "refunded") - _cell(windowed["ends_always", False], "refunded")
        == 5.0
    )


@pytest.mark.slow  # a mutant of the disposition, outside the quick tier
def test_crediting_an_old_awards_refund_to_the_newer_award_is_caught():
    """Mutant: the refund raises the row's newer award from 7 to 8 units, unlapsed.

    The ledger moves the credit from `lapsed` to the row's balance with the record's
    own generation untouched, so only the replay of the disposition can refuse it.
    """
    run = _Windowed()["forfeited_then_carry", True]
    ledger = run.carry.ledger
    bump = jnp.zeros_like(ledger.balance).at[P0, COL].set(10.0)
    new = eqx.tree_at(
        lambda lg: (lg.balance, lg.lapsed),
        ledger,
        (ledger.balance + bump, ledger.lapsed - bump),
    )
    forged = SimpleNamespace(
        study=run.study,
        carry=eqx.tree_at(lambda c: c.ledger, run.carry, new),
        trace=run.trace,
    )
    assert not balance_conserved(forged).passed


@pytest.mark.slow  # a mutant of the disposition, outside the quick tier
def test_a_refund_to_the_wrong_row_after_expiry_is_caught():
    """Mutant: a REALLOCATE award's late refund credited to the expired row."""
    run = _Windowed()["reallocated", True]
    ledger = run.carry.ledger
    move = jnp.zeros_like(ledger.balance).at[P0, COL].set(10.0).at[P1, COL].set(-10.0)
    new = eqx.tree_at(
        lambda lg: (lg.balance, lg.refunded),
        ledger,
        (ledger.balance + move, ledger.refunded + move),
    )
    forged = SimpleNamespace(
        study=run.study,
        carry=eqx.tree_at(lambda c: c.ledger, run.carry, new),
        trace=run.trace,
    )
    assert not balance_conserved(forged).passed
