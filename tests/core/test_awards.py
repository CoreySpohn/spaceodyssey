"""Awards and windows: an Allocate grants a balance for a window, and the window ends.

An admitted Allocate adds its amount to the recipient row and opens a window
`[from, until)`; the row is valid only inside it. A WINDOW_END event fires at `until`
(stage ANALYZE, neutral tie) and applies the remainder rule the award declared. Each
check has a control or a mutant of the rule it guards. The runs are host runs that
share one compile.
"""

import dataclasses
import functools
from types import SimpleNamespace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import (
    CompiledExecutor,
    HostExecutor,
    handlers,
    histories_equal,
)
from spaceodyssey.core.conformance.invariants import balance_conserved
from spaceodyssey.core.enums import (
    N_TIES,
    REFRESH_TIE,
    TIE_NEUTRAL,
    AdmissionCode,
    EffectKind,
    EventKind,
    RemainderRule,
    Stage,
    WindowState,
)
from spaceodyssey.core.events import empty_event_row
from spaceodyssey.core.snapshot import EffectTable
from spaceodyssey.studies import toy
from tests.core.award_helpers import award, award_study
from tests.core.helpers import one_world, worlds
from tests.core.program_helpers import end, observe, wait

pytestmark = pytest.mark.medium  # A cold host run compiles the executor.

N_STEPS = 200
P0, P1 = 2, 3  # the balance rows of programs 0 and 1 (two assets come first)
PROGRAM_COL = 1  # the program-scoped resource; every acquisition costs 1 of each
OK = int(AdmissionCode.OK)
FORFEIT, CARRY, REALLOC = (
    RemainderRule.FORFEIT,
    RemainderRule.CARRY_OVER,
    RemainderRule.REALLOCATE,
)

# one observation at tick 40 (channel 10 completes at 150) inside the window [30, 200)
INSIDE = (wait(0, 40), observe(1, 0, 10, 0), end(5))


def _inside(rule, to=P1):
    return INSIDE, (award(0, 2, P0, 30, 200, 5, rule, to),)


SCENARIOS = {
    "forfeit": _inside(FORFEIT),
    "none": _inside(RemainderRule.NONE),
    "carry": _inside(CARRY),
    "realloc": _inside(REALLOC),
    # an observation before the window opens
    "early": (
        (wait(0, 10), observe(1, 0, 10, 0), end(5)),
        (award(0, 2, P0, 30, 200, 5),),
    ),
    # an observation after the window ended (the rule NONE keeps the balance)
    "late": (
        (wait(0, 100), observe(1, 0, 10, 0), end(5)),
        (award(0, 2, P0, 0, 60, 5),),
    ),
    # the window ends before the observation completes
    "completes_late": (
        INSIDE,
        (award(0, 2, P0, 30, 100, 5, FORFEIT),),
    ),
    # a second award onto a row whose window is still open
    "overlap": (
        (wait(0, 20), wait(1, 200), end(2)),
        (award(0, 2, P0, 0, 100, 5), award(1, 2, P0, 20, 150, 3)),
    ),
    # a new award at the tick the previous window ends
    "adjacent": (
        (wait(0, 60), wait(1, 100), end(2)),
        (award(0, 2, P0, 0, 60, 5, FORFEIT), award(1, 2, P0, 60, 120, 3, FORFEIT)),
    ),
    # a completion at the very tick the window ends (channel 12 from tick 10 ends at 65)
    "same_tick": (
        (wait(0, 10), observe(1, 0, 12, 0), wait(1, 200), end(2)),
        (award(0, 2, P0, 0, 65, 5, FORFEIT),),
    ),
}


def _host(study, n_steps=N_STEPS):
    truth, root = one_world(caps=study.caps)
    result = HostExecutor().run(study, truth, root, n_steps)
    return SimpleNamespace(study=study, carry=result.carry, trace=result.trace)


@functools.cache
def _run(name):
    script, awards = SCENARIOS[name]
    return _host(award_study(script, awards))


def _handled(run):
    return np.asarray(run.trace.handled)


def _codes(run, kind):
    """The admission codes of every handled event of `kind`, one row per event."""
    t = run.trace
    pick = _handled(run) & (np.asarray(t.kind) == kind)
    return np.asarray(t.admit_code)[pick]


def _ledger(run):
    return run.carry.ledger


def _col(array, row):
    return float(np.asarray(array)[row, PROGRAM_COL])


def _window_ends(run):
    t = run.trace
    return np.flatnonzero(_handled(run) & (np.asarray(t.kind) == EventKind.WINDOW_END))


def test_an_allocate_grants_the_amount_and_opens_the_window():
    """Fails if an admitted award does not credit its row or record its window."""
    run = _run("forfeit")
    t = run.trace
    first = int(np.flatnonzero(_handled(run))[0])
    slot = int(np.flatnonzero(np.asarray(t.eff_kind)[first] == EffectKind.ALLOCATE)[0])
    assert int(np.asarray(t.admit_code)[first, slot]) == OK
    assert np.asarray(t.alloc_amount)[first, slot, PROGRAM_COL] == 5.0
    assert int(np.asarray(t.alloc_recipient)[first, slot]) == P0
    assert int(np.asarray(t.alloc_from)[first, slot]) == 30
    assert int(np.asarray(t.alloc_until)[first, slot]) == 200
    assert _col(_ledger(run).granted, P0) == 5.0
    # the observation at tick 40 paid one unit of the award
    assert int(_codes(run, EventKind.WAIT)[1][0]) == OK


def test_a_window_end_event_exists_only_once_an_award_is_issued():
    """Fails if a study without awards queues an expiry, or one with them lacks it."""
    ends = _window_ends(_run("forfeit"))
    assert len(ends) == 1
    assert int(np.asarray(_run("forfeit").trace.tick)[ends[0]]) == 200
    off = _host(toy.survey_1())
    assert not np.any(np.asarray(off.trace.kind)[_handled(off)] == EventKind.WINDOW_END)
    assert _ledger(off).granted is None
    assert _ledger(off).win_gen is None
    assert off.trace.alloc_amount is None


def test_the_window_end_is_an_analyze_stage_neutral_tie_event():
    """Fails if the expiry moves to another stage, takes a new tie rank or is late."""
    run = _run("forfeit")
    t = run.trace
    (i,) = _window_ends(run)
    assert int(np.asarray(t.stage)[i]) == Stage.ANALYZE
    assert int(np.asarray(t.tie)[i]) == TIE_NEUTRAL


@pytest.mark.parametrize(
    ("name", "left", "lapsed", "state", "gained"),
    [
        ("none", 4.0, 0.0, WindowState.IDLE, 0.0),
        ("forfeit", 0.0, 4.0, WindowState.IDLE, 0.0),
        ("carry", 4.0, 0.0, WindowState.ALWAYS, 0.0),
        ("realloc", 0.0, 4.0, WindowState.IDLE, 4.0),
    ],
)
def test_each_remainder_rule_settles_the_unspent_balance(
    name, left, lapsed, state, gained
):
    """Fails if a rule keeps, lapses, reopens or moves the wrong amount."""
    ledger = _ledger(_run(name))
    assert _col(ledger.balance, P0) == left
    assert _col(ledger.lapsed, P0) == lapsed
    assert int(np.asarray(ledger.win_state)[P0]) == state
    assert _col(ledger.balance, P1) == gained
    assert _col(ledger.granted, P1) == gained
    assert int(np.asarray(ledger.win_gen)[P0]) == 2  # opened once, closed once


def test_an_observation_outside_the_window_is_refused():
    """Fails if a row is spent before it opens or after it ended."""
    early = _codes(_run("early"), EventKind.WAIT)[1][0]
    late = _codes(_run("late"), EventKind.WAIT)[1][0]
    assert int(early) == AdmissionCode.OUTSIDE_VALIDITY
    assert int(late) == AdmissionCode.OUTSIDE_VALIDITY


def test_an_acquisition_that_completes_after_expiry_still_settles():
    """Fails if a window end cancels, refunds or re-validates a running acquisition."""
    run = _run("completes_late")
    t = run.trace
    tick, kind = np.asarray(t.tick), np.asarray(t.kind)
    done = _handled(run) & (kind == EventKind.DONE)
    assert done.any()
    assert int(tick[done][0]) > 100  # after the window ended at tick 100
    assert _col(_ledger(run).balance, P0) == 0.0  # the award lapsed, the charge stood
    assert _col(_ledger(run).lapsed, P0) == 4.0


def test_an_award_onto_an_open_window_is_forbidden_and_changes_nothing():
    """Fails if a second window replaces the first or the refused amount is granted."""
    run = _run("overlap")
    t = run.trace
    kinds = np.asarray(t.eff_kind)
    codes = np.asarray(t.admit_code)
    allocs = _handled(run)[:, None] & (kinds == EffectKind.ALLOCATE)
    seen = codes[allocs]
    assert seen.tolist() == [OK, AdmissionCode.FORBIDDEN]
    assert _col(_ledger(run).granted, P0) == 5.0
    assert int(np.asarray(_ledger(run).win_gen)[P0]) == 1 + 1  # one open, one closed
    assert len(_window_ends(run)) == 1  # the refused award queued no expiry


def test_a_new_award_at_the_expiry_tick_is_admitted_after_the_old_window_ends():
    """Fails if the expiry runs after the same-tick decision (adjacent windows)."""
    run = _run("adjacent")
    ledger = _ledger(run)
    t = run.trace
    allocs = _handled(run)[:, None] & (np.asarray(t.eff_kind) == EffectKind.ALLOCATE)
    assert np.asarray(t.admit_code)[allocs].tolist() == [OK, OK]
    assert _col(ledger.granted, P0) == 8.0
    assert _col(ledger.lapsed, P0) == 8.0
    assert int(np.asarray(ledger.win_gen)[P0]) == 4
    assert int(np.asarray(ledger.win_state)[P0]) == WindowState.IDLE


def test_a_completion_at_the_expiry_tick_is_handled_before_the_window_end():
    """Fails if an expiry overtakes the settlement that falls on the same tick."""
    run = _run("same_tick")
    t = run.trace
    pick = _handled(run) & (np.asarray(t.tick) == 65)
    order = [
        (int(k), int(s))
        for k, s in zip(
            np.asarray(t.kind)[pick], np.asarray(t.stage)[pick], strict=True
        )
    ]
    kinds = [k for k, _ in order]
    assert EventKind.DONE in kinds and EventKind.WINDOW_END in kinds
    assert kinds.index(EventKind.DONE) < kinds.index(EventKind.WINDOW_END)
    stages = [s for _, s in order]
    assert stages == sorted(stages)


def test_the_expiry_tie_ranks_after_a_same_tick_refresh():
    """Fails if the neutral tie ranks before the refresh tie or a tie is added."""
    assert REFRESH_TIE < TIE_NEUTRAL
    assert N_TIES == 3  # the expiry added no tie rank


def _open_window(run):
    """The final carry with row P0 put back into an open FORFEIT window of gen 2."""
    c = run.carry
    ledger = c.ledger
    ledger = eqx.tree_at(
        lambda lg: (lg.win_state, lg.win_gen, lg.remainder_rule, lg.balance),
        ledger,
        (
            ledger.win_state.at[P0].set(WindowState.WINDOWED),
            ledger.win_gen.at[P0].set(2),
            ledger.remainder_rule.at[P0].set(FORFEIT),
            ledger.balance.at[P0, PROGRAM_COL].set(4.0),
        ),
    )
    return eqx.tree_at(lambda x: x.ledger, c, ledger)


def test_a_stale_expiry_after_a_re_award_changes_nothing():
    """Fails if an old expiry closes the window a later award opened."""
    run = _run("forfeit")
    study = run.study
    truth, root = one_world(caps=study.caps)
    base = _open_window(run)
    stale = dataclasses.replace(
        empty_event_row(study.caps),
        kind=jnp.asarray(EventKind.WINDOW_END, jnp.int64),
        lane=jnp.asarray(P0, jnp.int64),
        gen=jnp.asarray(1, jnp.int64),
        valid=jnp.asarray(True),
    )
    after, _, _ = handlers.h_window_end(base, stale, study, truth, root)
    assert np.array_equal(after.ledger.balance, base.ledger.balance)
    assert np.array_equal(after.ledger.win_gen, base.ledger.win_gen)
    assert np.array_equal(after.ledger.win_state, base.ledger.win_state)
    # control: the live generation does close it and lapses the balance
    live = dataclasses.replace(stale, gen=jnp.asarray(2, jnp.int64))
    closed, _, _ = handlers.h_window_end(base, live, study, truth, root)
    assert _col(closed.ledger.balance, P0) == 0.0
    assert _col(closed.ledger.lapsed, P0) == _col(base.ledger.lapsed, P0) + 4.0
    assert int(closed.ledger.win_gen[P0]) == 3


def test_the_allocate_decision_is_retained_in_the_trace():
    """Fails if the amount, window or recipient of an Allocate leaves the trace."""
    run = _run("realloc")
    t = run.trace
    amount = np.asarray(t.alloc_amount)
    assert amount.sum() == 5.0  # one Allocate, retained once
    (i, e) = np.argwhere(amount.sum(-1) > 0)[0]
    assert np.asarray(t.alloc_until)[i, e] == 200
    assert np.asarray(t.alloc_recipient)[i, e] == P0
    # a trace row of another decision holds no award
    assert not amount[~_handled(run)].any()


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_award_scenario_conserves_the_balance(name):
    """Fails if grants, lapses or moves leave the three accounts disagreeing."""
    result = balance_conserved(_run(name))
    assert result.passed, (name, result)


def _forged(run, **leaves):
    ledger = run.carry.ledger
    new = eqx.tree_at(
        lambda lg: tuple(getattr(lg, n) for n in leaves),
        ledger,
        tuple(leaves.values()),
    )
    carry = eqx.tree_at(lambda c: c.ledger, run.carry, new)
    return SimpleNamespace(study=run.study, carry=carry, trace=run.trace)


@pytest.mark.parametrize("name", ["forfeit", "realloc"])
def test_forged_award_counters_are_caught(name):
    """Fails if the invariant accepts an award counter the trace does not support."""
    run = _run(name)
    ledger = run.carry.ledger
    bump = jnp.zeros_like(ledger.granted).at[P0, PROGRAM_COL].set(1.0)
    assert not balance_conserved(_forged(run, granted=ledger.granted - bump))
    assert not balance_conserved(_forged(run, granted=ledger.granted + bump))
    assert not balance_conserved(_forged(run, lapsed=ledger.lapsed + 10 * bump))
    assert not balance_conserved(_forged(run, balance=ledger.balance + bump))


def test_the_effect_table_without_awards_holds_no_award_column():
    """Fails if a study without awards starts to carry the Allocate columns."""
    study = toy.survey_1()
    eff = dataclasses.fields(EffectTable)
    from spaceodyssey.core.snapshot import effects_none

    table = effects_none(study.caps)
    for f in eff:
        if f.name in (
            "amount",
            "window_from",
            "window_until",
            "recipient",
            "remainder",
            "realloc_to",
        ):
            assert getattr(table, f.name) is None


@pytest.mark.slow
def test_the_host_and_compiled_executors_make_the_same_history():
    """Fails if the compiled scan and the host loop disagree on an award's run."""
    script, awards = SCENARIOS["realloc"]
    study = award_study(script, awards)
    truths, roots = worlds(2, study.caps)
    compiled = CompiledExecutor().run_worlds(study, truths, roots, 40)
    host = HostExecutor().run_worlds(study, truths, roots, 40)
    assert host.status.tolist() == compiled.status.tolist()
    for w in range(2):
        report = histories_equal(compiled.history(w), host.history(w))
        assert report, report
