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
    assert snapshot.snapshot_view(c_off, POLICY, off.caps).balance is None
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
    assert snapshot.snapshot_view(c_on, OBSERVER, on.caps).balance is None


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
    np.testing.assert_array_equal(view.balance, carry.ledger.balance)
    expected = np.round(np.asarray(carry.ledger.balance) / 0.5).astype(np.int64)
    np.testing.assert_array_equal(view.balance_q, expected)
    program_1 = study.caps.n_assets + 1
    assert program_1 == 3 and int(view.balance_q[program_1, 1]) == 15


@pytest.mark.medium  # admission compiles twice, 4.9 s cold alone
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


@pytest.mark.medium  # one study, both executors, 13 s cold alone
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


@pytest.mark.slow  # compiles a copy of admission, 2 s cold alone
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


@pytest.mark.slow
def test_mutant_admission_that_refuses_without_the_flag_fails_the_off_check():
    """A refusal that ignores `snapshot_balances` changes every older study."""
    admit = _mutant_admit(_refuses_mixed=lambda caps: True)
    with pytest.raises(AssertionError):
        _off_keeps_the_old_behavior(admit)


@pytest.mark.slow
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
    monkeypatch.setattr(study_module, "_check_balance_range", lambda budget, caps: None)
    with pytest.raises(pytest.fail.Exception):
        test_a_budget_the_integer_balance_cannot_hold_is_refused(1.0e7, 1.0e-12, False)


class _Script(AbstractPolicy):
    """One row of up to two effects per decision; the last row repeats."""

    rows: tuple = eqx.field(static=True)

    def declare(self):
        return Declaration(stops=((Scope.CAMPAIGN, Disposition.DRAIN),))

    def decide(self, pstate, view, study, key):
        table = jnp.asarray(self.rows, I64)[jnp.minimum(view.n_dec, len(self.rows) - 1)]
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


def _script(*decisions):
    pad = (int(K.NONE), 0, 0)
    return tuple(
        tuple((int(k), t, c) for k, t, c in d) + (pad,) * (2 - len(d))
        for d in decisions
    )


# Observe and Wait; then Wait and Cancel, whose refund is queued ahead of the next
# decision's request at its tick; then a lone Query; then Stop.
REFUND_AHEAD = _script(
    [(K.OBSERVE, 0, 0), (K.WAIT, 10, 0)],
    [(K.WAIT, 10, 0), (K.CANCEL, 0, 0)],
    [(K.QUERY, 0, 0)],
    [(K.STOP, 0, 0)],
)


def _refund_study(on=True):
    registry, kind = oracle.oracle_registry()
    caps = registry.lower(Capacities(snapshot_balances=on))
    return toy.survey_1(caps=caps, policies=(_Script(REFUND_AHEAD),)), kind


@pytest.mark.medium  # two compiles, both executors
def test_a_lone_query_survives_a_refund_queued_ahead_of_it_on_both_executors():
    """Fails with 'the snapshot changed between the request and its answer' if the
    request keeps the fingerprint of the decision while a queued Cancel returns balance
    before the request is held.
    """
    study, kind = _refund_study(True)
    truths, roots = worlds(1, study.caps)
    host_service = oracle.ToyOracleService(kind)
    carry, trace = HostExecutor().run(
        study, core.index_world(truths, 0), roots[0], 60, service=host_service
    )
    run = CompiledExecutor().run(
        study, truths, roots, 60, service=oracle.ToyOracleService(kind)
    )
    assert len(host_service.requests) == 1
    assert not bool(carry.request.valid)
    layout = study.observer.layout(study.caps)
    assert core.histories_equal(
        core.canonical_history(carry, trace, layout), run.history(0)
    ).ok


def _held_request(study, carry, stale):
    ev = empty_event_row(study.caps)
    ev = dataclasses.replace(
        ev, snapshot_fp=jnp.asarray(stale, I64), decision=jnp.asarray(0, I64)
    )
    truth, root = one_world(0, study.caps)
    held, _, _ = handlers.h_request(carry, ev, study, truth, root)
    return int(held.request.snapshot_fp)


def _request_is_stamped_where_it_is_held():
    study, carry = _study(True), _carry(True)
    moved = _bumped(carry, 5.0)
    assert _held_request(study, carry, 1234) == _fp(study, carry)
    # the balance moved after the decision read it: the held stamp follows the state
    assert _held_request(study, moved, 1234) == _fp(study, moved)
    assert _fp(study, moved) != _fp(study, carry)
    # with balances off the stamp is left as the decision wrote it
    assert _held_request(_study(False), _carry(False), 1234) == 1234


def test_a_held_request_is_stamped_with_the_snapshot_it_is_answered_on():
    """Fails if the stamp is the decision's after the balance moved (the exception
    above), or if an off study rewrites a stamp.
    """
    _request_is_stamped_where_it_is_held()


def test_mutant_held_request_keeping_the_old_stamp_fails_the_stamp_check(monkeypatch):
    """The stamp check must fail when the request keeps the decision's stamp."""
    _request_is_stamped_where_it_is_held()
    monkeypatch.setattr(
        handlers, "snapshot_fingerprint", lambda view, layout: jnp.asarray(1234, I64)
    )
    with pytest.raises(AssertionError):
        _request_is_stamped_where_it_is_held()
