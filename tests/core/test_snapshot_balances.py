"""Balances in the policy snapshot and the QUERY_MIXED refusal (A7).

With `snapshot_balances` on, the policy view carries the ledger balance and its integer
count of quanta, the integer joins the policy fingerprint, and a decision that combines
a Query with any other effect is refused whole. With it off nothing exists and the old
behavior stands. Each rule is a checker function; the real tests run it on the shipped
code and the mutant tests run it on a broken copy and expect it to fail.

`Allocate` has no admission yet (it answers `UNSUPPORTED`), so the Query plus Allocate
cases use the declared effect kind as it is: the refusal does not depend on whether the
other effect could have been admitted.
"""

import dataclasses
import functools

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import spaceodyssey.core.handlers as handlers
import spaceodyssey.core.snapshot as snapshot
import spaceodyssey.core.study as study_module
from spaceodyssey import core
from spaceodyssey.core import transaction
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import (
    I64,
    AdmissionCode,
    Audience,
    ChargePricing,
    Disposition,
    EffectKind,
    Scope,
    effect_mask,
)
from spaceodyssey.core.events import empty_event_row
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.core.kinds import AbstractPolicy
from spaceodyssey.studies import oracle, toy
from tests.core.helpers import one_world, worlds
from tests.core.mutants import _clone
from tests.core.refund_helpers import (
    BALANCE,
    N_RECEIPTS,
    OFF,
    ScriptedReceipts,
    receipt,
    refund_study,
)

K, A = EffectKind, AdmissionCode
MIXED = [
    (K.QUERY, K.OBSERVE),
    (K.OBSERVE, K.QUERY),
    (K.QUERY, K.ALLOCATE),
    (K.ALLOCATE, K.QUERY),
    (K.WAIT, K.QUERY),
]
admit_j = eqx.filter_jit(core.admit)
OBSERVER, POLICY = Audience.OBSERVER, Audience.POLICY


@functools.cache
def _study(on=True, quantum=1.0, n_programs=1, permitted=None):
    caps = Capacities(
        snapshot_balances=on,
        balance_quantum=quantum if on else 1.0,
        n_programs=n_programs,
    )
    if permitted is None:
        return toy.survey_1(caps=caps)
    return toy.survey_1(caps=caps, permitted_effects=(permitted,))


@functools.cache
def _carry(on=True, quantum=1.0, n_programs=1, permitted=None):
    study = _study(on, quantum, n_programs, permitted)
    truth, _ = one_world(0, study.caps)
    return eqx.filter_jit(core.init_carry)(study, truth)


def _rows(*kinds):
    n = _study().caps.n_effects
    kind = jnp.asarray([*kinds, *[K.NONE] * (n - len(kinds))], I64)
    return dataclasses.replace(core.effects_none(_study().caps), kind=kind)


def _unchanged(a, b):
    r = core.tree_equal_report(a, b)
    return r["max_int_diff"] == 0 and r["max_abs"] == 0.0


def _requests(c):
    return int(np.asarray(c.ev.valid & (c.ev.kind == core.EventKind.REQUEST)).sum())


# ---- admission rules, one checker each ------------------------------------------


def _query_alone_passes(admit):
    study, c0 = _study(), _carry()
    c1, codes = admit(c0, _rows(K.QUERY), study)
    assert codes.tolist() == [A.OK, A.NONE]
    assert _requests(c1) == 1
    # the snapshot the host rebuilds is the one the request names
    (row,) = np.flatnonzero(
        np.asarray(c1.ev.valid & (c1.ev.kind == core.EventKind.REQUEST))
    )
    stamped = int(c1.ev.snapshot_fp[row])
    assert stamped == int(transaction.decision_fingerprint(c1, study)) != 0


def _mixed_is_refused_whole(admit, kinds):
    study, c0 = _study(), _carry()
    c1, codes = admit(c0, _rows(*kinds), study)
    assert codes.tolist() == [A.QUERY_MIXED, A.QUERY_MIXED]
    assert _requests(c1) == 0
    assert _unchanged(c0, c1)


def _off_keeps_the_old_behavior(admit):
    study, c0 = _study(False), _carry(False)
    c1, codes = admit(c0, _rows(K.QUERY, K.OBSERVE), study)
    assert codes.tolist() == [A.OK, A.OK]
    assert _requests(c1) == 1 and float(c1.ledger.debited.sum()) > 0.0
    _, codes = admit(c0, _rows(K.QUERY, K.ALLOCATE), study)
    assert codes.tolist() == [A.OK, A.UNSUPPORTED]


def _only_effects_are_mixed(admit):
    study = _study(permitted=effect_mask(K.QUERY, K.WAIT))
    c0 = _carry(permitted=effect_mask(K.QUERY, K.WAIT))
    # a row the phase forbids is not an effect of the decision
    _, codes = admit(c0, _rows(K.QUERY, K.OBSERVE), study)
    assert codes.tolist() == [A.OK, A.FORBIDDEN]
    # a second Query is refused for the slot it needs, not as a mix
    _, codes = admit(c0, _rows(K.QUERY, K.QUERY), study)
    assert codes.tolist() == [A.OK, A.NO_SLOT]


def test_a_query_alone_passes_with_balances_on():
    """Fails if the refusal catches a lone Query or the request names a stale view."""
    _query_alone_passes(admit_j)


@pytest.mark.parametrize("kinds", MIXED, ids=lambda k: f"{k[0].name}-{k[1].name}")
def test_a_query_mixed_with_another_effect_is_refused_whole(kinds):
    """Fails if either order lets a sibling debit run (the host would reject the
    rebuilt snapshot) or leaves a refused row with another code.
    """
    _mixed_is_refused_whole(admit_j, kinds)


def test_with_balances_off_a_mixed_decision_behaves_as_before():
    """Fails if the refusal is live without the flag: every older study would change."""
    _off_keeps_the_old_behavior(admit_j)


def test_a_forbidden_row_and_a_second_query_are_not_mixed_decisions():
    """Fails if the refusal counts a forbidden row, or a second Query, as an effect."""
    _only_effects_are_mixed(admit_j)


# ---- the view and the fingerprint -----------------------------------------------


def _fp(study, carry, audience=POLICY):
    """Eager, so a patched leaf list is read afresh (a jitted call is cached)."""
    view = snapshot.snapshot_view(carry, audience, study.caps)
    return int(snapshot.snapshot_fingerprint(view, study.observer.layout(study.caps)))


def _bumped(carry, delta):
    return eqx.tree_at(
        lambda c: c.ledger.balance, carry, carry.ledger.balance.at[0, 0].add(delta)
    )


def _fingerprint_rules():
    on, off = _study(True, 0.5), _study(False)
    c_on, c_off = _carry(True, 0.5), _carry(False)
    base = _fp(on, c_on)
    off_view = snapshot.snapshot_view(c_off, POLICY, off.caps)
    assert all(
        getattr(off_view, name) is None
        for name in ("balance", "balance_q", "balance_rows")
    )
    # the integer view joins the fingerprint only when balances are on
    assert base != _fp(off, c_off)
    assert _fp(off, _bumped(c_off, 3.0)) == _fp(off, c_off)
    # 10.0 over 0.5 is 20 quanta: 10.1 rounds back to 20, 10.5 is 21
    assert _fp(on, _bumped(c_on, 0.1)) == base
    assert _fp(on, _bumped(c_on, 0.5)) != base
    # the observer never sees balances
    for observer_view in (
        (on, c_on),
        (on, _bumped(c_on, 1.0)),
        (off, c_off),
    ):
        assert _fp(*observer_view, OBSERVER) == _fp(off, c_off, OBSERVER)
    observer = snapshot.snapshot_view(c_on, OBSERVER, on.caps)
    assert all(
        getattr(observer, name) is None
        for name in ("balance", "balance_q", "balance_rows")
    )


def test_balances_move_the_policy_fingerprint_only_when_on_and_never_the_observers():
    """Fails if the float is hashed, a sub-quantum change moves the digest, an off
    study's digest moves, or the observer's view carries a balance.
    """
    _fingerprint_rules()


def test_the_policy_view_lists_every_balance_row_in_quanta():
    """Fails if a row is missing (the program rows follow the assets, the shared row
    is last), or the integer is not the balance over the quantum, rounded.
    """
    study = _study(True, 0.5, 2)
    carry = _carry(True, 0.5, 2)
    carry = eqx.tree_at(
        lambda c: c.ledger.balance, carry, carry.ledger.balance.at[3, 1].set(7.26)
    )
    view = snapshot.snapshot_view(carry, POLICY, study.caps)
    n_rows = study.caps.n_assets + 2 + 1
    assert view.balance.shape == (n_rows, study.caps.n_resources)
    np.testing.assert_array_equal(view.balance, np.asarray(view.balance_q) * 0.5)
    expected = np.round(np.asarray(carry.ledger.balance) / 0.5).astype(np.int64)
    np.testing.assert_array_equal(view.balance_q, expected)
    program_1 = study.caps.n_assets + 1
    assert program_1 == 3 and int(view.balance_q[program_1, 1]) == 15


@pytest.mark.medium(reason="admission compiles twice, 4.9 s cold alone")
def test_an_admitted_debit_moves_the_fingerprint_only_with_balances_on():
    """Fails if the balance is not in the policy digest: the reason a Query cannot
    share a decision with an Observe.
    """
    for on in (True, False):
        study, c0 = _study(on), _carry(on)
        c1, codes = admit_j(c0, _rows(K.OBSERVE), study)
        assert codes[0] == A.OK
        moved = _fp(study, c1) != _fp(study, c0)
        assert moved is on


@pytest.mark.parametrize(
    "quantum", [0.0, -1.0, float("nan"), float("inf"), 2, True], ids=repr
)
def test_a_balance_quantum_that_is_not_a_positive_finite_float_is_refused(quantum):
    """Fails if a zero, negative, nonfinite or non-float quantum builds capacities."""
    with pytest.raises(ValueError, match="balance_quantum"):
        Capacities(balance_quantum=quantum)


# ---- the hosted path ------------------------------------------------------------


@pytest.mark.slow(reason="one study on both executors, 38 to 55 s cold")
def test_a_hosted_run_with_balances_on_answers_its_query_on_both_executors():
    """Fails if the rebuilt snapshot fingerprint differs from the request's, or the
    two executors disagree on a fingerprint that now holds the balance.
    """
    study, service = oracle.oracle_study(caps=Capacities(snapshot_balances=True))
    truths, roots = worlds(1, study.caps)
    host_service = oracle.ToyOracleService(service.kind)
    carry, trace = HostExecutor().run(
        study,
        core.index_world(truths, 0),
        roots[0],
        40,
        service=host_service,
    )
    run = CompiledExecutor().run(
        study, truths, roots, 40, service=oracle.ToyOracleService(service.kind)
    )
    assert len(host_service.requests) == 1
    assert not bool(carry.request.valid)
    layout = study.observer.layout(study.caps)
    host = core.canonical_history(carry, trace, layout)
    assert core.histories_equal(host, run.history(0)).ok
    assert np.array_equal(host.ints["snapshot_fp"], run.history(0).ints["snapshot_fp"])


# ---- mutants, one per rule (each compiles a broken copy of the code) -------------


def _mutant_admit(**replacements):
    return eqx.filter_jit(_clone(transaction.admit, **replacements))


def _no_rows(effects, mask, act):
    return jnp.zeros_like(effects.kind, dtype=bool)


def _every_nonempty_row(effects, mask, act):
    live = effects.kind != K.NONE
    mixed = (
        act
        & (live & (effects.kind == K.QUERY)).any()
        & (live & ~(effects.kind == K.QUERY)).any()
    )
    return live & mixed


def test_mutant_admission_that_never_refuses_fails_the_mixed_checks():
    """A refusal that selects no rows lets the sibling debit through, and the rebuilt
    fingerprint then differs from the one the request carries.
    """
    admit = _mutant_admit(_query_mixed_rows=_no_rows)
    for kinds in MIXED:
        with pytest.raises(AssertionError):
            _mixed_is_refused_whole(admit, kinds)
    study, c0 = _study(), _carry()
    c1, codes = admit(c0, _rows(K.QUERY, K.OBSERVE), study)
    assert codes.tolist() == [A.OK, A.OK]
    (row,) = np.flatnonzero(
        np.asarray(c1.ev.valid & (c1.ev.kind == core.EventKind.REQUEST))
    )
    assert int(c1.ev.snapshot_fp[row]) != int(
        transaction.decision_fingerprint(c1, study)
    )


def test_mutant_admission_that_refuses_without_the_flag_fails_the_off_check():
    """A refusal that ignores `snapshot_balances` changes every older study."""
    admit = _mutant_admit(_refuses_mixed=lambda caps: True)
    with pytest.raises(AssertionError):
        _off_keeps_the_old_behavior(admit)


def test_mutant_admission_that_counts_every_row_fails_the_effects_check():
    """A refusal that counts a forbidden row as an effect overrides its own code."""
    admit = _mutant_admit(_query_mixed_rows=_every_nonempty_row)
    with pytest.raises(AssertionError):
        _only_effects_are_mixed(admit)


def _leaves_float(view):
    extra = () if view.balance is None else (view.balance,)
    return (view.log.product, *extra)


def _leaves_none(view):
    return (view.log.product,)


def _every_view(carry, caps):
    balance = carry.ledger.balance
    rows = jnp.zeros((balance.shape[0], 2), I64)
    return balance, jnp.round(balance / caps.balance_quantum).astype(I64), rows


def _shown_to_all(carry, audience, caps):
    if not caps.snapshot_balances:
        return None, None, None
    return _every_view(carry, caps)


def _shown_when_off(carry, audience, caps):
    return _every_view(carry, caps)


@pytest.mark.parametrize(
    ("target", "broken"),
    [
        ("conditional_view_leaves", _leaves_float),
        ("conditional_view_leaves", _leaves_none),
        ("balance_view", _shown_to_all),
        ("balance_view", _shown_when_off),
    ],
    ids=["float-hashed", "integer-dropped", "observer-shown", "shown-when-off"],
)
def test_mutant_views_fail_the_fingerprint_rules(monkeypatch, target, broken):
    """Each broken view or leaf list must fail the fingerprint rules."""
    _fingerprint_rules()
    monkeypatch.setattr(snapshot, target, broken)
    with pytest.raises(AssertionError):
        _fingerprint_rules()


# ---- review round 1: a refund ahead of the request, row names, the int64 range ----


def _layout_study(n_assets, n_programs, budget=None, quantum=1.0):
    caps = Capacities(
        snapshot_balances=True,
        balance_quantum=quantum,
        n_assets=n_assets,
        n_programs=n_programs,
        n_lanes=2,
        n_channels=2,
    )
    rows = n_assets + n_programs + 1
    if budget is None:
        budget = jnp.arange(rows * caps.n_resources, dtype=jnp.float64)
        budget = budget.reshape(rows, caps.n_resources)
    lane_asset = [0, 0] if n_assets == 1 else [0, 1]
    return toy.survey_1(caps=caps, lane_asset=lane_asset, budget=budget)


def _layouts_are_told_apart():
    """Two layouts with five rows and one budget table must not share a digest."""
    one_three, two_two = _layout_study(1, 3), _layout_study(2, 2)
    fps = []
    for study in (one_three, two_two):
        truth, _ = one_world(0, study.caps)
        carry = eqx.filter_jit(core.init_carry)(study, truth)
        view = snapshot.snapshot_view(carry, POLICY, study.caps)
        np.testing.assert_array_equal(view.balance_q, np.asarray(study.budget))
        fps.append(_fp(study, carry))
    assert fps[0] != fps[1]


def test_the_view_names_its_rows_so_layouts_with_equal_row_counts_differ():
    """Fails if the digest sees only the table: with (1 asset, 3 programs) and (2
    assets, 2 programs) program 0 owns different cells yet the views were identical.
    """
    _layouts_are_told_apart()
    study = _layout_study(1, 3)
    truth, _ = one_world(0, study.caps)
    carry = eqx.filter_jit(core.init_carry)(study, truth)
    rows = snapshot.snapshot_view(carry, POLICY, study.caps).balance_rows
    np.testing.assert_array_equal(rows, [[0, 0], [1, 0], [1, 1], [1, 2], [2, 0]])


def _rows_dropped(view):
    program = () if view.log.program is None else (view.log.program,)
    return (view.log.product, *program, view.balance_q)


def test_mutant_leaves_without_row_names_fail_the_layout_check(monkeypatch):
    """The layout check must fail when the row names are not hashed."""
    _layouts_are_told_apart()
    monkeypatch.setattr(snapshot, "conditional_view_leaves", _rows_dropped)
    with pytest.raises(AssertionError):
        _layouts_are_told_apart()


@pytest.mark.parametrize(
    ("budget", "quantum", "accepted"),
    [
        (1.0e7, 1.0e-12, False),
        (float(2**53), 1.0, True),
        (float(2**53) * 2, 1.0, False),
        (float("inf"), 1.0, False),
        (float("nan"), 1.0, False),
        (1.0e7, 1.0e-8, True),
    ],
    ids=["1e-12-saturates", "2^53", "2^54", "inf", "nan", "1e-8-fits"],
)
def test_a_budget_the_integer_balance_cannot_hold_is_refused(budget, quantum, accepted):
    """Fails if a quantum is accepted that makes distinct balances share an integer
    (10_000_000 and 9_999_999 over 1e-12 both saturate int64) or leaves it unjudged.
    """
    table = jnp.full((5, 3), budget, jnp.float64)
    if accepted:
        _layout_study(2, 2, table, quantum)
        return
    with pytest.raises(ValueError, match="balance_quantum"):
        _layout_study(2, 2, table, quantum)


def test_a_study_that_shows_no_balances_judges_no_range():
    """The range rule belongs to the flag: an off study keeps any budget."""
    caps = Capacities(n_lanes=2, n_channels=2)
    table = jnp.full((5, 3), 1.0e30, jnp.float64)
    toy.survey_1(caps=caps, budget=table)


def test_mutant_study_without_the_range_check_fails_the_range_test(monkeypatch):
    """The range test must fail when the check is removed."""
    monkeypatch.setattr(study_module, "_check_balance_range", lambda *args: None)
    with pytest.raises(pytest.fail.Exception):
        test_a_budget_the_integer_balance_cannot_hold_is_refused(1.0e7, 1.0e-12, False)


@pytest.mark.parametrize(
    "cost", [-1.0e20, -2.0e20, float("nan")], ids=["-1e20", "-2e20", "nan"]
)
def test_a_negative_channel_cost_is_refused_while_balances_are_shown(cost):
    """Fails if a credit-like channel is accepted: with a zero budget one Observe
    reached INT64_MAX for both -1e20 and -2e20, and the two fingerprints were equal.
    """
    base = _layout_study(2, 2, jnp.zeros((5, 3), jnp.float64))
    bad = eqx.tree_at(
        lambda s: (s.channels.cost, s.channels.charge),
        base,
        (jnp.full_like(base.channels.cost, cost),) * 2,
    )
    with pytest.raises(ValueError, match="debit that is not negative"):
        study_module._check_balance_range(bad)
    with pytest.raises(ValueError, match="debit that is not negative"):
        toy.survey_1(
            caps=bad.caps,
            lane_asset=[0, 1],
            budget=bad.budget,
            channels=bad.channels,
        )


def test_mutant_study_without_the_sign_check_fails_the_sign_test(monkeypatch):
    """The sign test must fail when only the budget bound is checked."""
    real = study_module._check_balance_range

    def budget_only(study):
        ones = jnp.ones_like(study.channels.cost)
        clean = eqx.tree_at(
            lambda s: (s.channels.cost, s.channels.charge), study, (ones, ones)
        )
        return real(clean)

    monkeypatch.setattr(study_module, "_check_balance_range", budget_only)
    with pytest.raises(pytest.fail.Exception):
        test_a_negative_channel_cost_is_refused_while_balances_are_shown(-1.0e20)


def _priced(**kw):
    return refund_study(cap_kw={"snapshot_balances": True}, **kw)


NEGATIVE_PRICES = [
    dict(charge=0.0, pricing=ChargePricing.SURCHARGE, surcharge=-1.0e20),
    dict(charge=0.0, pricing=ChargePricing.SURCHARGE, surcharge=-2.0e20),
    dict(charge=0.0, pricing=ChargePricing.SURCHARGE, surcharge=-3.0),
    dict(charge=10.0, pricing=ChargePricing.SURCHARGE, surcharge=-11.0),
    dict(charge=10.0, pricing=ChargePricing.OCCUPANCY, surcharge=-1.0),
    dict(charge=-1.0, pricing=ChargePricing.FLAT_FRACTION),
]
VALID_PRICES = [
    dict(charge=10.0, pricing=ChargePricing.SURCHARGE, surcharge=2.0),
    dict(charge=0.0, pricing=ChargePricing.SURCHARGE, surcharge=2.0),
    dict(charge=4.0, pricing=ChargePricing.FLAT_FRACTION),
    dict(charge=4.0, pricing=ChargePricing.STATISTICAL),
    dict(charge=10.0),
]


def _negative_prices_are_refused():
    for kw in NEGATIVE_PRICES:
        with pytest.raises(ValueError, match="debit that is not negative"):
            _priced(**kw)
    for kw in VALID_PRICES:
        _priced(**kw)


def test_a_price_that_debits_a_negative_amount_is_refused_while_balances_are_shown():
    """Fails if a surcharge (or a charge) that makes the ledger a credit passes with
    cost = charge = 0 and a surcharge of -1e20, which `charge_parts` would debit as
    a credit; positive surcharge, flat and statistical prices stay valid.
    """
    _negative_prices_are_refused()


def test_a_negative_surcharge_is_valid_while_balances_are_off():
    """The rule belongs to the flag: refund prices are not judged otherwise."""
    refund_study(charge=0.0, pricing=ChargePricing.SURCHARGE, surcharge=-3.0)


def test_mutant_study_that_ignores_surcharge_prices_fails_the_price_test(monkeypatch):
    """The price test must fail when the surcharge and the priced debit go unchecked."""
    real = study_module._check_balance_range

    def no_prices(study):
        plain = dataclasses.replace(study.channels, surcharge=None, pricing=None)
        return real(eqx.tree_at(lambda s: s.channels, study, plain))

    monkeypatch.setattr(study_module, "_check_balance_range", no_prices)
    with pytest.raises(pytest.fail.Exception):
        _negative_prices_are_refused()


class _Script(AbstractPolicy):
    """One row of up to two effects per decision; the last row repeats."""

    rows: tuple = eqx.field(static=True)

    def declare(self):
        return Declaration(stops=((Scope.CAMPAIGN, Disposition.DRAIN),))

    def _index(self, view):
        return jnp.minimum(view.n_dec, len(self.rows) - 1)

    def decide(self, pstate, view, study, key):
        table = jnp.asarray(self.rows, I64)[self._index(view)]
        kind = table[:, 0]
        stop = kind == K.STOP
        eff = toy._effects(
            study.caps,
            kind=kind,
            tick=table[:, 1],
            channel_index=table[:, 2],
            group=view.n_dec,
            scope=jnp.where(stop, Scope.CAMPAIGN, 0),
            disposition=jnp.where(stop, Disposition.DRAIN, 0),
        )
        return eff, pstate, jnp.asarray(0.0)


class _Learner(_Script):
    """Follows the script to its Query at decision `ask_at`, then asks again at the next
    decision only if the snapshot says its previous Query was refused, and otherwise
    stops. Rows after the Query: the second Query, then Stop.
    """

    ask_at: int = eqx.field(default=2, static=True)

    def _index(self, view):
        n = view.n_dec
        refused = False if view.query_refused is None else view.query_refused == n - 1
        again = self.ask_at + 1
        return jnp.where(n <= self.ask_at, n, jnp.where(refused, again, again + 1))


def _script(*decisions, width=2):
    pad = (int(K.NONE), 0, 0)
    return tuple(
        tuple((int(k), t, c) for k, t, c in d) + (pad,) * (width - len(d))
        for d in decisions
    )


# Observe and Wait; then Wait and Cancel, whose refund is queued ahead of the next
# decision's request at its tick; then a lone Query (stale and refused while balances
# are shown); then the same Query again, which `_Learner` sends only if it saw the
# refusal; then Stop.
REFUND_AHEAD = _script(
    [(K.OBSERVE, 0, 0), (K.WAIT, 10, 0)],
    [(K.WAIT, 10, 0), (K.CANCEL, 0, 0)],
    [(K.QUERY, 0, 0)],
    [(K.QUERY, 0, 0)],
    [(K.STOP, 0, 0)],
)


def _refund_study(on=True, quantum=1.0):
    registry, kind = oracle.oracle_registry()
    caps = registry.lower(Capacities(snapshot_balances=on, balance_quantum=quantum))
    return toy.survey_1(caps=caps, policies=(_Learner(REFUND_AHEAD),)), kind


class _BalanceService(oracle.ToyOracleService):
    """Answers with the balance of ledger row 0, resource 0 as the snapshot shows."""

    def answer(self, request, view, layout):
        super().answer(request, view, layout)
        balance = 0.0 if view.balance is None else float(view.balance[0, 0])
        return np.asarray([balance, 0.0, 0.0])


def _refund_run(on, n_requests, n_stale, quantum=1.0):
    study, kind = _refund_study(on, quantum)
    truths, roots = worlds(1, study.caps)
    host_service = _BalanceService(kind)
    carry, trace = HostExecutor().run(
        study, core.index_world(truths, 0), roots[0], 80, service=host_service
    )
    compiled_service = _BalanceService(kind)
    run = CompiledExecutor().run(study, truths, roots, 80, service=compiled_service)
    assert not bool(carry.request.valid)
    layout = study.observer.layout(study.caps)
    host = core.canonical_history(carry, trace, layout)
    assert core.histories_equal(host, run.history(0)).ok
    for history, service in ((host, host_service), (run.history(0), compiled_service)):
        stale = int((history.ints["admit_code"] == A.QUERY_STALE).sum())
        assert (len(service.requests), stale) == (n_requests, n_stale)
    return host_service, host


@pytest.mark.slow(reason="one study on both executors, 38 to 55 s cold")
def test_a_query_made_stale_by_a_queued_refund_is_refused_and_asked_again():
    """Fails with 'the snapshot changed between the request and its answer' if the
    stale request is held, or if it is answered on a state the policy did not read.
    The first Query is refused (one QUERY_STALE row, no request reaches the host), the
    policy reads the refusal in its next snapshot and asks again, and that Query is
    answered, on both executors.
    """
    service, host = _refund_run(True, n_requests=1, n_stale=1)
    # the answer is computed on the state the asking decision read
    asked = service.requests[0]
    decisions = host.ints["snapshot_fp"][host.ints["snapshot_fp"] != 0]
    assert asked.snapshot_fp == decisions[3]


@pytest.mark.slow(reason="the balances-off control: two more compiles, 12 s cold alone")
def test_with_balances_off_the_same_refund_study_refuses_nothing():
    """The control: nothing is refused, the first Query is answered and the policy,
    which has no refusal to read, stops.
    """
    _refund_run(False, n_requests=1, n_stale=0)


@pytest.mark.slow(reason="two more compiles, 10 s cold alone")
def test_a_refund_inside_one_quantum_is_not_stale_because_the_view_cannot_show_it():
    """Fails if the host could read a balance the fingerprint does not hash: with a
    quantum of 10 the refund takes the balance from 9 to 10, one count either way, so
    the request is held and answered on exactly what the policy read.
    """
    _refund_run(True, n_requests=1, n_stale=0, quantum=10.0)


def _the_float_shows_only_the_hashed_integer():
    """Two carries that differ by less than a quantum read the same float, and the
    float is a whole number of quanta.
    """
    study = _study(True, 10.0)
    base = _carry(True, 10.0)
    near = _bumped(base, 3.0)
    one = snapshot.snapshot_view(base, POLICY, study.caps)
    two = snapshot.snapshot_view(near, POLICY, study.caps)
    assert _fp(study, base) == _fp(study, near)
    np.testing.assert_array_equal(one.balance, two.balance)
    np.testing.assert_array_equal(one.balance, np.asarray(one.balance_q) * 10.0)


def test_the_policy_reads_no_balance_the_fingerprint_does_not_hash():
    """Fails if the float view carries the raw ledger balance."""
    _the_float_shows_only_the_hashed_integer()


def test_mutant_view_with_the_raw_float_fails_the_float_rule(monkeypatch):
    """The float rule must fail when the view exposes the raw balance."""
    _the_float_shows_only_the_hashed_integer()
    real = snapshot.balance_view

    def raw(carry, audience, caps):
        _, quanta, rows = real(carry, audience, caps)
        return carry.ledger.balance, quanta, rows

    monkeypatch.setattr(snapshot, "balance_view", raw)
    with pytest.raises(AssertionError):
        _the_float_shows_only_the_hashed_integer()


def _refusal_is_readable_by_the_policy_only():
    study, carry = _study(True), _carry(True)
    refused = eqx.tree_at(lambda c: c.query_refused, carry, jnp.asarray(2, I64))
    for audience, shown in ((POLICY, True), (OBSERVER, False)):
        view = snapshot.snapshot_view(refused, audience, study.caps)
        assert (view.query_refused is not None) is shown
    assert int(snapshot.snapshot_view(refused, POLICY, study.caps).query_refused) == 2
    assert _fp(study, refused) != _fp(study, carry)
    assert _fp(study, refused, OBSERVER) == _fp(study, carry, OBSERVER)
    off = _carry(False)
    assert off.query_refused is None
    assert snapshot.snapshot_view(off, POLICY, _study(False).caps).query_refused is None


def test_the_policy_snapshot_carries_the_refusal_and_it_moves_its_fingerprint():
    """Fails if the refusal is absent from the policy view or digest, is shown to the
    observer, or exists with balances off.
    """
    _refusal_is_readable_by_the_policy_only()


@pytest.mark.parametrize("hide", ["view", "digest"])
def test_mutant_hidden_refusal_fails_the_visibility_rules(monkeypatch, hide):
    """The refusal rules must fail when the leaf is hidden from the view or digest."""
    _refusal_is_readable_by_the_policy_only()
    if hide == "view":
        real = snapshot.snapshot_view

        def hidden(carry, audience, caps):
            return dataclasses.replace(real(carry, audience, caps), query_refused=None)

        monkeypatch.setattr(snapshot, "snapshot_view", hidden)
    else:
        real = snapshot.conditional_view_leaves

        def undigested(view):
            leaves = real(view)
            return leaves[:-1] if view.query_refused is not None else leaves

        monkeypatch.setattr(snapshot, "conditional_view_leaves", undigested)
    with pytest.raises(AssertionError):
        _refusal_is_readable_by_the_policy_only()


def _held(study, carry, fp):
    ev = empty_event_row(study.caps)
    ev = dataclasses.replace(
        ev, snapshot_fp=jnp.asarray(fp, I64), decision=jnp.asarray(0, I64)
    )
    truth, root = one_world(0, study.caps)
    after, _, _ = handlers.h_request(carry, ev, study, truth, root)
    waits = lambda c: int((c.ev.valid & (c.ev.kind == core.EventKind.WAIT)).sum())  # noqa: E731
    return bool(after.request.valid), waits(after) - waits(carry)


def _stale_request_is_refused_with_a_wake():
    study, carry = _study(True), _carry(True)
    moved = _bumped(carry, 5.0)
    fresh = _fp(study, carry)
    assert _held(study, carry, fresh) == (True, 0)  # nothing moved: held, no wake
    assert _held(study, moved, fresh) == (False, 1)  # a refund moved it: refused
    # with balances off the request is held as before
    assert _held(_study(False), _carry(False), 1234) == (True, 0)


def test_a_request_whose_snapshot_moved_is_not_held_and_wakes_the_policy():
    """Fails if a request on a moved balance is held, a fresh one is refused, or an
    off study refuses anything.
    """
    _stale_request_is_refused_with_a_wake()


@pytest.mark.parametrize(
    "verdict", [False, True], ids=["answers-anyway", "refuses-all"]
)
def test_mutant_stale_checks_fail_the_refusal_rules(monkeypatch, verdict):
    """The refusal rules must fail when staleness is never or always found."""
    _stale_request_is_refused_with_a_wake()
    monkeypatch.setattr(handlers, "_stale_request", lambda c, ev, study: verdict)
    with pytest.raises(AssertionError):
        _stale_request_is_refused_with_a_wake()


# ---- a receipt released between a decision and its request ------------------------


class _AdmitStageReceipts(ScriptedReceipts):
    """Receipts fired at the admit stage, after the decide stage of their tick."""

    def seed(self, carry, caps):
        for i in range(self.rows.shape[0]):
            row = self.rows[i]
            carry = core.enqueue(
                carry,
                row[7] == 1,
                kind=core.EventKind.PROCESS,
                tick=row[0],
                stage=core.Stage.ADMIT,
                group=i,
                audience=row[6],
                caps=caps,
            )
        return carry


# Observe on channel 10; a receipt naming it is processed at tick 125, which wakes the
# policy before the receipt's release (queued behind it) runs. The policy asks, the
# release credits the refund ahead of the request, and the request is refused as stale.
RECEIPT_AHEAD = _script(
    [(K.OBSERVE, 0, 0), (K.WAIT, 200, 0)],
    [(K.QUERY, 0, 0)],
    [(K.WAIT, 10, 0)],
    [(K.QUERY, 0, 0)],
    [(K.STOP, 0, 0)],
    width=3,
)


class _ReceiptLearner(_Script):
    """Asks at decision 1; the receipt's own release wakes it once more, and it waits;
    the refusal wakes it again, and it asks a second time; then it stops.
    """

    def _index(self, view):
        return jnp.minimum(view.n_dec, 4)


@pytest.mark.slow(reason="one study on both executors, 38 to 55 s cold")
def test_a_receipt_released_between_a_decision_and_its_request_refuses_the_query():
    """Fails if a refund credited by a receipt released ahead of the request is
    answered on a state the policy did not read: the request is stale, refused once, and
    the policy, which reads the refusal, asks again and is answered.
    """
    registry, kind = oracle.oracle_registry()
    caps = registry.lower(Capacities(n_effects=3, refunds=True, snapshot_balances=True))
    ch = refund_study().channels
    study = toy.survey_1(
        caps=caps,
        channels=ch,
        budget=jnp.full((caps.n_balance_rows, caps.n_resources), BALANCE),
        policies=(_ReceiptLearner(RECEIPT_AHEAD),),
        process=_AdmitStageReceipts(rows=(receipt(125), *[OFF] * (N_RECEIPTS - 1))),
    )
    truths, roots = worlds(1, study.caps)
    host_service = _BalanceService(kind)
    carry, trace = HostExecutor().run(
        study, core.index_world(truths, 0), roots[0], 80, service=host_service
    )
    compiled_service = _BalanceService(kind)
    run = CompiledExecutor().run(study, truths, roots, 80, service=compiled_service)
    layout = study.observer.layout(study.caps)
    host = core.canonical_history(carry, trace, layout)
    assert core.histories_equal(host, run.history(0)).ok
    for history, service in ((host, host_service), (run.history(0), compiled_service)):
        stale = int((history.ints["admit_code"] == A.QUERY_STALE).sum())
        assert (len(service.requests), stale) == (1, 1)
    assert int(carry.ledger.refunded.sum()) > 0  # the receipt did credit a refund
