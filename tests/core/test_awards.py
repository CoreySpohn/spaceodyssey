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
import jax
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
from tests.core.award_helpers import (
    ScriptedAwards,
    award,
    award_refund_study,
    award_study,
)
from tests.core.helpers import one_world, worlds
from tests.core.program_helpers import cancel, end, observe, wait
from tests.core.refund_helpers import mutate, receipt

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


@pytest.mark.slow  # the awards-off control compiles a study, 12 s cold
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


# ---- review round 1: generation capture, expiry replay, shapes, range ---------------


def test_a_paired_forged_expiry_is_caught():
    """Fails if restoring a forfeited balance and erasing its lapse passes unnoticed.

    The pair keeps balance, granted and lapsed mutually consistent, so only a replay of
    the award, its window and the debits inside it can tell the lapse was owed.
    """
    run = _run("forfeit")
    ledger = run.carry.ledger
    assert float(np.asarray(ledger.lapsed)[P0, PROGRAM_COL]) > 0.0
    lapsed = np.asarray(ledger.lapsed)[P0, PROGRAM_COL]
    bump = jnp.zeros_like(ledger.balance).at[P0, PROGRAM_COL].set(lapsed)
    assert balance_conserved(run).passed
    forged = _forged(run, balance=ledger.balance + bump, lapsed=ledger.lapsed - bump)
    assert not balance_conserved(forged).passed


def test_the_paired_forged_expiry_is_caught_for_a_reallocation():
    """Fails if moving a reallocated balance back and un-granting it passes."""
    run = _run("realloc")
    ledger = run.carry.ledger
    moved = float(np.asarray(ledger.granted)[P1, PROGRAM_COL])
    assert moved > 0.0
    bump = jnp.zeros_like(ledger.balance).at[P0, PROGRAM_COL].set(moved)
    drop = jnp.zeros_like(ledger.balance).at[P1, PROGRAM_COL].set(moved)
    forged = _forged(
        run,
        balance=ledger.balance + bump - drop,
        lapsed=ledger.lapsed - bump,
        granted=ledger.granted - drop,
    )
    assert not balance_conserved(forged).passed


def _record_award_gens(run):
    log = run.carry.log
    return np.asarray(log.award_gens)[np.asarray(log.valid)].tolist()


@pytest.mark.slow  # its own refund study compiles the executor, 20 s cold
def test_a_completion_records_the_generation_of_every_row_it_drew_from():
    """Fails if the record keeps one generation instead of one per payer slot.

    Row 2 (program 0) is awarded for [0, 200), so its generation is 1 at the debit; the
    asset row 0 and the shared row 4 never held a window and stay at 0. The record
    holds a generation for each of the three resources.
    """
    study = award_study(
        (wait(0, 10), observe(1, 0, 10, 0), end(5)),
        (award(0, 2, P0, 0, 200, 5),),
        refunds=True,
    )
    run = _host(study)
    assert _record_award_gens(run) == [[0, 1, 0]]


@pytest.mark.slow  # its own refund study compiles the executor, 20 s cold
def test_two_awarded_payers_each_keep_their_own_generation():
    """Fails if the generation of a second awarded payer row is lost.

    The asset row 0 is awarded twice (generation 3 after its first window ends and a
    second opens) while program row 2 is awarded once (generation 1): the record keeps
    both, where a single number would drop one.
    """
    study = award_study(
        (wait(0, 5), wait(1, 100), wait(2, 40), observe(3, 0, 10, 0), end(5)),
        (
            award(0, 2, 0, 0, 20, 2, RemainderRule.NONE),
            award(0, 1, P0, 0, 300, 5),
            award(2, 2, 0, 100, 300, 2, RemainderRule.NONE),
        ),
        refunds=True,
    )
    run = _host(study)
    gens = _record_award_gens(run)
    assert len(gens) == 1
    assert gens[0][0] == 3 and gens[0][1] == 1  # asset row 0 twice awarded; row 2 once


class _Reshaped(ScriptedAwards):
    """The scripted awards, with the amount column reshaped to `shape`."""

    shape: tuple = eqx.field(static=True, default=(3,))

    def decide(self, pstate, view, study, key):
        eff, ps, u = super().decide(pstate, view, study, key)
        flat = eff.amount.sum(axis=1) if len(self.shape) == 1 else eff.amount[:, :1]
        return dataclasses.replace(eff, amount=flat), ps, u


def _reshaped(shape):
    class Shaped(_Reshaped):
        pass

    Shaped.__dataclass_fields__["shape"].default = shape
    return Shaped


@pytest.mark.slow  # each reshaped policy compiles the executor, 35 s cold each
@pytest.mark.parametrize("shape", [(3,), (3, 1)])
def test_a_malformed_amount_shape_is_refused_as_unsupported(shape):
    """Fails if a vector that broadcasts into every resource is credited."""
    study = award_study(
        (wait(0, 10), end(5)),
        (award(0, 2, P0, 0, 200, 5),),
        policy_cls=_reshaped(shape),
    )
    run = _host(study)
    t = run.trace
    allocs = _handled(run)[:, None] & (np.asarray(t.eff_kind) == EffectKind.ALLOCATE)
    assert np.asarray(t.admit_code)[allocs].tolist() == [AdmissionCode.UNSUPPORTED]
    assert float(np.asarray(run.carry.ledger.granted).sum()) == 0.0
    assert float(np.asarray(run.carry.ledger.balance)[P0, PROGRAM_COL]) == 0.0


@pytest.mark.slow  # static balance capacities: each variant compiles, 25 s cold
def test_an_award_past_the_representable_balance_range_is_refused():
    """Fails if an Allocate can take a row above S4's exact range (2**53 quanta).

    With balances in the snapshot, both 1e20 and 1e21 would read as the same clamped
    value. A small award is the control: it is admitted under the same capacities.
    """
    big = (
        (wait(0, 10), wait(1, 20), wait(2, 30), end(5)),
        (
            award(0, 2, P0, 0, 200, 1e20),
            award(1, 2, P0, 0, 200, 1e21),
            award(2, 2, P0, 0, 200, 3.0),
        ),
    )
    study = award_study(*big, caps_extra={"snapshot_balances": True})
    run = _host(study)
    t = run.trace
    allocs = _handled(run)[:, None] & (np.asarray(t.eff_kind) == EffectKind.ALLOCATE)
    codes = np.asarray(t.admit_code)[allocs].tolist()
    assert codes[0] == AdmissionCode.UNSUPPORTED
    assert codes[1] == AdmissionCode.UNSUPPORTED
    assert float(np.asarray(run.carry.ledger.balance)[P0].max()) < 2.0**53
    small = award_study(
        (wait(0, 10), end(5)),
        (award(0, 2, P0, 0, 200, 3.0),),
        caps_extra={"snapshot_balances": True},
    )
    srun = _host(small)
    sa = _handled(srun)[:, None] & (
        np.asarray(srun.trace.eff_kind) == EffectKind.ALLOCATE
    )
    assert np.asarray(srun.trace.admit_code)[sa].tolist() == [OK]


@pytest.mark.slow  # static balance capacities: each variant compiles, 25 s cold
@pytest.mark.parametrize("quantum", [1.0, 0.125, 10.0])
def test_opposite_budgets_do_not_offset_in_the_range_bound(quantum):
    """Fails if a negative budget row lowers the supply the range bound measures.

    Rows hold +B and -B of the program resource (B is 2**53 quanta), so their sum is
    zero while the positive row already sits at the limit: an award of 8 quanta on it
    would pass 2**53 quanta. A negative budget is refused whatever its size.
    """

    def codes(big):
        study = award_study(
            (wait(0, 10), end(5)),
            (award(0, 2, P0, 0, 200, 8.0 * quantum),),
            caps_extra={"snapshot_balances": True, "balance_quantum": quantum},
        )
        budget = np.asarray(study.budget).copy()
        budget[P0, PROGRAM_COL] = big
        budget[P1, PROGRAM_COL] = -big
        study = dataclasses.replace(study, budget=jnp.asarray(budget))
        run = _host(study)
        allocs = _handled(run)[:, None] & (
            np.asarray(run.trace.eff_kind) == EffectKind.ALLOCATE
        )
        return np.asarray(run.trace.admit_code)[allocs].tolist(), run

    top = 2.0**53 * quantum
    refused, run = codes(top)
    assert refused == [AdmissionCode.UNSUPPORTED]
    assert float(np.asarray(run.carry.ledger.granted).sum()) == 0.0
    small, _ = codes(100.0 * quantum)
    assert small == [AdmissionCode.UNSUPPORTED], "any negative budget is refused"


@pytest.mark.slow  # static balance capacities: each variant compiles, 25 s cold
@pytest.mark.parametrize("quantum", [1.0, 0.125, 10.0])
def test_a_negative_reallocation_cannot_create_headroom(quantum):
    """Fails if a negative balance reallocated twice lets an award pass 2**53 quanta.

    Rows start at B - 8q and -8q (B is 2**53 quanta); the negative row is moved to the
    other and back, which lowered the signed `granted` sum and admitted 16q on the
    positive row. Negative budgets are refused, so no award is admitted at all, and the
    positive row stays below B.
    """
    top = 2.0**53 * quantum
    study = award_study(
        (wait(0, 10), wait(1, 10), wait(2, 10), end(5)),
        (
            award(0, 2, P1, 0, 10, 0, RemainderRule.REALLOCATE, P0),
            award(1, 2, P0, 10, 20, 0, RemainderRule.REALLOCATE, P1),
            award(2, 2, P0, 20, 100, 16 * quantum),
        ),
        caps_extra={"snapshot_balances": True, "balance_quantum": quantum},
    )
    budget = np.zeros(np.asarray(study.budget).shape)
    budget[P0, PROGRAM_COL] = top - 8 * quantum
    budget[P1, PROGRAM_COL] = -8 * quantum
    study = dataclasses.replace(study, budget=jnp.asarray(budget))
    run = _host(study, n_steps=80)
    t = run.trace
    allocs = _handled(run)[:, None] & (np.asarray(t.eff_kind) == EffectKind.ALLOCATE)
    codes = np.asarray(t.admit_code)[allocs].tolist()
    assert all(code == AdmissionCode.UNSUPPORTED for code in codes)
    assert float(np.asarray(run.carry.ledger.balance)[P0, PROGRAM_COL]) < top


@pytest.mark.slow  # a study with a large award compiles the executor, 21 s cold
def test_a_large_award_without_snapshot_balances_is_admitted():
    """Control: the range rule binds only when balances are in the snapshot."""
    study = award_study((wait(0, 10), end(5)), (award(0, 2, P0, 0, 200, 1e20),))
    run = _host(study)
    t = run.trace
    allocs = _handled(run)[:, None] & (np.asarray(t.eff_kind) == EffectKind.ALLOCATE)
    assert np.asarray(t.admit_code)[allocs].tolist() == [OK]


@pytest.mark.slow  # compiles admission with balances in the snapshot, 15 s cold
def test_an_allocate_mixed_with_a_query_is_refused_whole():
    """Fails if S4's mixed-Query guard stops covering Allocate when awards are on."""
    from spaceodyssey import core

    study = award_study(
        (end(0),), (), caps_extra={"snapshot_balances": True, "balance_quantum": 1.0}
    )
    truth, _ = one_world(0, study.caps)
    c0 = eqx.filter_jit(core.init_carry)(study, truth)
    base = core.effects_none(study.caps)
    kind = jnp.asarray([EffectKind.ALLOCATE, EffectKind.QUERY, EffectKind.NONE])
    eff = dataclasses.replace(base, kind=kind.astype(base.kind.dtype))
    _, codes = eqx.filter_jit(core.admit)(c0, eff, study)
    assert codes.tolist()[:2] == [AdmissionCode.QUERY_MIXED] * 2


# ---- review round 2: refunds in the replay, pending debits, a program-row award ----

REFUND_SCRIPT = (wait(0, 40), observe(1, 0, 10, 0), wait(1, 400), end(5))
REFUND_AWARD = (award(0, 2, P0, 0, 300, 5, FORFEIT),)


@functools.cache
def _refund_run(receipt_tick):
    study = award_refund_study(
        REFUND_SCRIPT, REFUND_AWARD, (receipt(receipt_tick),), pricing=None
    )
    return _host(study, n_steps=60)


@pytest.mark.slow  # award runs with refunds compile once, 23 s cold
def test_a_refund_after_the_expiry_is_replayed_and_conserves():
    """Control: the replay credits the late refund and agrees with the ledger."""
    run = _refund_run(350)
    assert float(np.asarray(run.carry.ledger.refunded)[P0, PROGRAM_COL]) == 10.0
    assert float(np.asarray(run.carry.ledger.lapsed)[P0, PROGRAM_COL]) > 0.0
    assert balance_conserved(run).passed


@pytest.mark.slow  # award runs with refunds compile once, 23 s cold
def test_a_refund_before_the_expiry_is_replayed_and_conserves():
    """Control: a refund inside the window raises what the expiry forfeits.

    The same refund after the expiry lapses at once, so either way the row ends with
    the same lapsed total and balance.
    """
    run = _refund_run(250)
    assert balance_conserved(run).passed
    late = _refund_run(350)
    ledger, late_ledger = run.carry.ledger, late.carry.ledger
    for name in ("lapsed", "balance", "refunded"):
        assert np.allclose(
            np.asarray(getattr(ledger, name)), np.asarray(getattr(late_ledger, name))
        ), name


@pytest.mark.slow  # award runs with refunds compile once, 23 s cold
@pytest.mark.parametrize("tick", [250, 350])
def test_a_paired_forged_expiry_is_caught_when_the_run_holds_a_refund(tick):
    """Fails if a receipt in the log switches the expiry reconciliation off."""
    run = _refund_run(tick)
    ledger = run.carry.ledger
    lapsed = float(np.asarray(ledger.lapsed)[P0, PROGRAM_COL])
    bump = jnp.zeros_like(ledger.balance).at[P0, PROGRAM_COL].set(lapsed)
    forged = _forged(run, balance=ledger.balance + bump, lapsed=ledger.lapsed - bump)
    assert not balance_conserved(forged).passed


@pytest.mark.slow  # award runs with refunds compile once, 23 s cold
def test_a_refund_credited_to_another_programs_row_is_caught():
    """Fails if a refund that lands on the wrong program's row conserves."""
    run = _refund_run(350)
    ledger = run.carry.ledger
    move = jnp.zeros_like(ledger.balance)
    move = move.at[P0, PROGRAM_COL].set(-10.0).at[P1, PROGRAM_COL].set(10.0)
    refunded = ledger.refunded + move
    forged = _forged(run, balance=ledger.balance + move, refunded=refunded)
    assert not balance_conserved(forged).passed


def _forged_log(run, **columns):
    log = eqx.tree_at(
        lambda lg: tuple(getattr(lg, n) for n in columns),
        run.carry.log,
        tuple(jnp.asarray(v) for v in columns.values()),
    )
    return SimpleNamespace(
        study=run.study,
        carry=eqx.tree_at(lambda c: c.log, run.carry, log),
        trace=run.trace,
    )


TWO_PROGRAMS = (
    wait(0, 40),
    observe(1, 0, 10, 0),
    wait(1, 200),
    observe(2, 0, 10, 1),
    wait(2, 300),
    end(5),
)


@functools.cache
def _two_program_run():
    study = award_refund_study(
        TWO_PROGRAMS, REFUND_AWARD, (receipt(350),), pricing=None
    )
    return _host(study, n_steps=60)


@pytest.mark.slow  # award runs with refunds compile once, 23 s cold
def test_a_refund_routed_to_another_acquisition_in_both_places_is_caught():
    """Fails if the replay trusts the handler's `refund_of` to find the origin.

    The receipt names subject 0 on channel 10, index 0 (a program 0 acquisition; index 1
    is program 1's). The
    credit is moved to program 1's row in the balance and the refunded counter, and
    `refund_of` is repointed at program 1's acquisition, so every counter and the
    handler's own column agree with the forgery. Only the receipt's identity does not.
    """
    run = _two_program_run()
    assert balance_conserved(run).passed
    log = run.carry.log
    refs = np.asarray(log.refund_of).copy()
    ids = np.asarray(log.acquisition_id)
    valid = np.asarray(log.valid)
    (receipt_row,) = np.flatnonzero(valid & (refs >= 0))
    other = [
        j
        for j in np.flatnonzero(valid & (refs < 0))
        if int(np.asarray(log.channel)[j]) == 10
        and int(np.asarray(log.acquisition_index)[j]) == 1
    ]
    assert len(other) == 1
    refs[receipt_row] = ids[other[0]]
    ledger = run.carry.ledger
    move = jnp.zeros_like(ledger.balance)
    move = move.at[P0, PROGRAM_COL].set(-10.0).at[P1, PROGRAM_COL].set(10.0)
    forged = _forged(
        _forged_log(run, refund_of=refs),
        balance=ledger.balance + move,
        refunded=ledger.refunded + move,
    )
    assert not balance_conserved(forged).passed


@pytest.mark.slow  # award runs with refunds compile once, 23 s cold
@pytest.mark.parametrize(
    "receipts",
    [
        (receipt(160), receipt(170)),
        (receipt(160, index=7),),
        (receipt(160, held=True),),
        (receipt(160, channel=11),),
    ],
    ids=["duplicate", "unmatched", "held", "wrong-channel"],
)
def test_receipts_that_apply_nothing_do_not_fail_conservation(receipts):
    """Fails if an ignored, unmatched or held receipt is replayed as an acquisition."""
    study = award_refund_study(REFUND_SCRIPT, REFUND_AWARD, receipts, pricing=None)
    run = _host(study, n_steps=60)
    assert balance_conserved(run).passed
    log = run.carry.log
    kinds = np.asarray(log.channel)[np.asarray(log.valid)]
    assert (kinds == 50).any(), "the receipt reached the log"


@pytest.mark.slow  # award runs with refunds compile once, 23 s cold
def test_a_record_naming_a_newer_award_generation_is_caught():
    """Fails if the replay accepts a record whose payer generation is not the row's.

    A late refund is credited by the origin's award identity, so a record that names a
    newer award than the row held at its debit would send the credit to the wrong one.
    """
    run = _refund_run(350)
    log = run.carry.log
    gens = np.asarray(log.award_gens).copy()
    (row,) = np.flatnonzero(np.asarray(log.valid) & (np.asarray(log.refund_of) < 0))
    gens[row, PROGRAM_COL] += 1
    new = eqx.tree_at(lambda lg: lg.award_gens, log, jnp.asarray(gens))
    forged = SimpleNamespace(
        study=run.study,
        carry=eqx.tree_at(lambda c: c.log, run.carry, new),
        trace=run.trace,
    )
    assert not balance_conserved(forged).passed


@pytest.mark.slow  # stops a scenario run early, which compiles again, 20 s cold
def test_a_pending_acquisition_at_the_expiry_is_replayed_without_its_record():
    """Fails if an admission debit with no record yet is missing from the replay.

    The run stops just after the expiry with the settlement still queued: the lapse
    excludes the debited unit, and the replay finds it from the queued settlement.
    """
    script, awards = SCENARIOS["completes_late"]
    study = award_study(script, awards)
    full = _run("completes_late")
    stop = int(_window_ends(full)[0]) + 1  # just after the expiry, before the DONE
    run = _host(study, n_steps=stop)
    pending = run.carry.ev.valid & (run.carry.ev.kind == EventKind.DONE)
    assert bool(np.asarray(pending).any())
    assert not np.asarray(run.carry.log.valid).any()
    assert balance_conserved(run).passed


def test_an_award_to_a_program_row_does_not_crash_the_conservation_check():
    """Fails if a queued WINDOW_END's balance-row id is read as a lane number."""
    study = award_study(
        (wait(0, 10), end(5)),
        (award(0, 2, P1, 0, 200, 5, FORFEIT),),
    )
    run = _host(study, n_steps=1)
    assert (
        int(
            np.asarray(
                run.carry.ev.valid & (run.carry.ev.kind == EventKind.WINDOW_END)
            ).sum()
        )
        == 1
    )
    assert balance_conserved(run).passed


@pytest.mark.slow  # static balance capacities: each variant compiles, 25 s cold
def test_an_allocate_after_a_debit_cannot_pass_the_representable_supply():
    """Fails if refunds or reallocations can carry a row past 2**53 quanta.

    A row starts at 2**53 and is debited 1; the old rule (current balance plus the
    award) admitted an award of 1 elsewhere, after which the refund would return the
    unit. The rule is on supply (budget plus grants), so the award is refused.
    """
    top = 2.0**53
    study = award_study(
        (wait(0, 10), observe(1, 0, 10, 0), wait(2, 10), end(5)),
        (award(2, 2, P1, 0, 200, 1.0),),
        program_credit=top,
        caps_extra={"snapshot_balances": True},
    )
    run = _host(study)
    t = run.trace
    allocs = _handled(run)[:, None] & (np.asarray(t.eff_kind) == EffectKind.ALLOCATE)
    codes = np.asarray(t.admit_code)[allocs].tolist()
    assert codes == [AdmissionCode.UNSUPPORTED]


# ---- refunds across award windows -------------------------------------------------

# Program 0's row is balance row 2 and program 1's is 3; the program-scoped resource is
# column 1. Decision 0 awards at tick 0, decision 1 observes at tick 40 (the debit of
# 10 under that award), decision 2 (woken by the release at 170) only waits, and
# decision 3 acts at tick 310, after the first window ends at 300. The failure receipt
# for that observation is released at tick 350.
WIN_FIRST = award(0, 2, P0, 0, 300, 5, RemainderRule.FORFEIT)
WIN_SCRIPT = (
    wait(0, 40),
    observe(1, 0, 10, 0),
    wait(1, 270),
    wait(2, 140),
    wait(3, 100),
    end(5),
)


LATE = (receipt(350),)
RECEIPT_INSIDE = (receipt(250),)


def _then(rule, to=0):
    """The first award, then a second one on the same row after it, with `rule`."""
    return (WIN_FIRST, award(3, 2, P0, 310, 360, 7, rule, to))


# name -> (awards, receipt tick, overrides): every run shares one compiled executor
WINDOW_SCENARIOS = {
    "forfeited_then_none": (_then(RemainderRule.NONE), LATE, {}),
    "forfeited_then_carry": (_then(RemainderRule.CARRY_OVER), LATE, {}),
    "forfeited_then_forfeit": (_then(RemainderRule.FORFEIT), LATE, {}),
    "ends_always": (
        (WIN_FIRST, award(3, 2, P0, 310, 340, 20, RemainderRule.CARRY_OVER)),
        LATE,
        dict(charge=5.0),
    ),
    "idle": ((award(0, 2, P0, 0, 300, 5, RemainderRule.NONE),), LATE, {}),
    "none_then_newer": (
        (award(0, 2, P0, 0, 300, 5, RemainderRule.NONE), *_then(0)[1:]),
        LATE,
        {},
    ),
    "reallocated": (
        (award(0, 2, P0, 0, 300, 5, RemainderRule.REALLOCATE, P1),),
        LATE,
        {},
    ),
    "inside": (
        (award(0, 2, P0, 0, 600, 5, RemainderRule.FORFEIT),),
        RECEIPT_INSIDE,
        {},
    ),
}


class _Windowed(dict):
    """The run of a scenario, and of its twin without the receipt, each made once."""

    def __missing__(self, key):
        name, with_receipt = key
        awards, receipts, overrides = WINDOW_SCENARIOS[name]
        study = award_refund_study(
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


def _window_cell(run, name, row=P0):
    return float(np.asarray(getattr(run.carry.ledger, name))[row, PROGRAM_COL])


def _window_change(windowed, name, counter, row=P0):
    """What the receipt changed in a ledger counter: the run with it less without."""
    return _window_cell(windowed[name, True], counter, row) - _window_cell(
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
        assert _window_change(windowed, name, "balance") == on_p0
        assert _window_change(windowed, name, "balance", P1) == on_p1
    assert _window_change(windowed, name, "refunded", taker) == refunded
    if on_p0 is not None:
        assert _window_change(windowed, name, "lapsed") == lapsed
    assert balance_conserved(windowed[name, True]).passed


def test_a_refund_of_five_to_a_row_left_always_keeps_its_balance(windowed):
    """The balance the row ends with is B's own 20: the refund of exactly 5 lapsed."""
    run = windowed["ends_always", True]
    assert _window_cell(run, "balance") == _window_cell(
        windowed["ends_always", False], "balance"
    )
    assert (
        _window_cell(run, "refunded")
        - _window_cell(windowed["ends_always", False], "refunded")
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
    bump = jnp.zeros_like(ledger.balance).at[P0, PROGRAM_COL].set(10.0)
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
    move = (
        jnp.zeros_like(ledger.balance)
        .at[P0, PROGRAM_COL]
        .set(10.0)
        .at[P1, PROGRAM_COL]
        .set(-10.0)
    )
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


# ---- conservation across windows, on the same compiled study ----------------------

CANCELLED = (observe(0, 0, 10, 0), wait(0, 60), cancel(1, 0))  # half spent at tick 60
# name -> (script, awards): receipts none; the study is the one the scenarios share
CONSERVED = {
    "cancel_under_award": (CANCELLED, (award(0, 2, P0, 0, 1000, 5, FORFEIT),)),
    "observe_after_expiry": (
        (wait(0, 100), observe(1, 0, 10, 0), end(2)),
        (award(0, 2, P0, 0, 60, 5, CARRY),),
    ),
}


class _Conserved(dict):
    """The run of a conservation case, made on first use."""

    def __missing__(self, name):
        script, awards = CONSERVED[name]
        study = award_refund_study(script, awards, (), pricing=None)
        self[name] = _host(study, n_steps=80)
        return self[name]


@pytest.fixture(scope="module")
def conserved():
    return _Conserved()


def test_a_cancellation_under_an_award_conserves(conserved):
    """Fails if the replay drops a cancel's credit from the refunded account.

    The lane is cancelled at tick 60, half spent: the ledger credits 5, and the replay
    of the awards must expect the same credit in `refunded` and in the balance.
    """
    run = conserved["cancel_under_award"]
    assert _col(run.carry.ledger.refunded, P0) == 5.0
    assert balance_conserved(run).passed


def test_a_debit_after_its_window_expired_names_generation_zero(conserved):
    """Fails if a debit on a row whose window has closed captures that row's generation.

    The carry-over award ends at tick 60, so the row is unwindowed at tick 100: the
    record names generation 0 and the rule NONE (not the row's generation 2), and the
    replay accepts it.
    """
    run = conserved["observe_after_expiry"]
    log = run.carry.log
    assert int(np.asarray(run.carry.ledger.win_gen)[P0]) == 2
    assert not np.any(np.asarray(log.award_gens)[0])
    assert not np.any(np.asarray(log.award_rules)[0])
    assert balance_conserved(run).passed


def test_a_refund_across_an_expiry_and_a_re_award_conserves(windowed):
    """Control: the late refund lapses, and the per-row account and the replay agree."""
    run = windowed["forfeited_then_carry", True]
    ledger = run.carry.ledger
    assert _col(ledger.refunded, P0) == 10.0
    assert _col(ledger.lapsed, P0) >= 10.0
    assert balance_conserved(run).passed


def _mutant(monkeypatch, old, new):
    monkeypatch.setattr(
        handlers, "_apply_refund", mutate(handlers._apply_refund, old, new)
    )
    jax.clear_caches()  # the compiled step closed over the real handler


@pytest.mark.slow  # A mutant clears the compile caches and compiles the executor cold.
def test_mutant_crediting_an_old_awards_refund_to_the_newer_award_fails(monkeypatch):
    """Mutant: the refund always credits the row, so a closed award raises the new one.

    The ledger counters and the record agree with the credit; only the independent
    replay of the disposition refuses it.
    """
    _mutant(monkeypatch, "same = led.win_gen[rows] == gens", "same = rows >= 0")
    try:
        run = _Windowed()["forfeited_then_carry", True]
        assert not balance_conserved(run).passed
    finally:
        jax.clear_caches()
