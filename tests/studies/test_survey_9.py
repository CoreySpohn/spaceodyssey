"""Survey 9, interleaved programs: independent balances and a credit-free program.

Two programs share the lanes of the toy study. The merge-time guard (medium) is (b) and
(d): programs alternating on one channel keep independent balances, stopping one lets
the other continue, and every ledger row conserves. The rest is slow (the Slice 2b
quick-tier budget, K6): (a) the one-program history equals `survey_1`'s, (c) the
credit-free program, the controls, and both-executor parity.

The study is scripted, so the cadence is part of the evidence: a decision is woken by
each public release and by each wait, so the scripts below are written against the
ticks the runs print (a funded start on channel 12 at tick t occupies lane 2 until
t + 55; the credit-free channel 13 until t + 45).
"""

import dataclasses
from types import SimpleNamespace

import equinox as eqx
import jax.random as jr
import numpy as np
import pytest

from spaceodyssey.core import HostExecutor
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.conformance.invariants import balance_conserved
from spaceodyssey.core.enums import F64, AdmissionCode, Disposition, EffectKind, Scope
from spaceodyssey.core.executors.base import histories_equal
from spaceodyssey.studies import toy
from tests.core.helpers import one_world, worlds
from tests.core.program_helpers import (
    ScriptedPrograms,
    end,
    observe,
    stop,
    two_program_study,
    wait,
)
from tests.studies.survey_helpers import run_on, same_histories

N_STEPS = 200
CLOSED, NO_FUNDS, BUSY = (
    AdmissionCode.LANE_CLOSED,
    AdmissionCode.NO_FUNDS,
    AdmissionCode.LANE_BUSY,
)
PROGRAM_COL = 1  # the resource whose scope is the program (asset, program, shared)
FUNDED, FREE = 12, 13  # channels on lane 2, the second asset's lane

# (b) programs 0 and 1 alternate on channel 10; program 0 is stopped, then both ask
ALTERNATE = (
    observe(0, 0, 10, 0),
    observe(1, 1, 10, 1),
    observe(2, 0, 10, 0),
    observe(3, 1, 10, 1),
    stop(4, Scope.PROGRAM, Disposition.CLOSE, 0),
    wait(4, 20),
    observe(5, 0, 10, 0),
    observe(5, 1, 10, 1),
)
# (c) the funded program starts channel 12 at ticks 0 and 110; the credit-free program
# starts channel 13 at tick 60 (the lane is idle) and asks again at 110 (it is not)
FREE_RUN = (
    observe(0, 0, FUNDED, 0),
    wait(0, 60),
    observe(2, 1, FREE, 1),
    wait(2, 50),
    observe(4, 0, FUNDED, 0),
    observe(4, 1, FREE, 1),
    wait(4, 60),
    end(5),
)
# the same funded requests with no credit-free program, so the decision ticks match
BASE_RUN = (
    observe(0, 0, FUNDED, 0),
    wait(0, 60),
    wait(2, 50),
    observe(3, 0, FUNDED, 0),
    wait(3, 60),
    end(4),
)
# control: the credit-free start is placed at tick 80, across the funded request at 110
BAD_RUN = (
    observe(0, 0, FUNDED, 0),
    wait(0, 60),
    wait(2, 20),
    observe(3, 1, FREE, 1),
    wait(3, 30),
    observe(4, 0, FUNDED, 0),
    end(5),
)


def _four_channel_study(rows, *, free_cost=0.0, free_balance=0.0):
    """The toy study with channel 13 on lane 2 beside channel 12.

    Channel 13 costs `free_cost` and program 1 holds `free_balance`, so with the
    defaults program 1 has no credit and its channel debits nothing.
    """
    caps = Capacities(n_programs=2, n_effects=3, n_channels=4)
    study = toy.survey_1(caps=caps, policies=(ScriptedPrograms(script=rows),))
    cost = study.channels.cost.at[3].set(free_cost)
    channels = dataclasses.replace(study.channels, cost=cost, charge=cost)
    budget = study.budget.at[caps.n_assets + 1].set(free_balance)
    return dataclasses.replace(study, channels=channels, budget=budget)


def _run(study, n_steps=N_STEPS):
    truth, root = one_world(caps=study.caps)
    return HostExecutor().run(study, truth, root, n_steps)


def _requests(run):
    """Admission code of each Observe request, in decision order."""
    t = run.trace
    keep = np.asarray(t.handled)[:, None] & (
        np.asarray(t.eff_kind) == EffectKind.OBSERVE
    )
    return np.asarray(t.admit_code)[keep]


def _records(run, **columns):
    """The log rows of a run, columns named by keyword and read to the row count."""
    n = int(run.carry.n_log)
    return {k: np.asarray(getattr(run.carry.log, k))[:n] for k in columns} | {"n": n}


def _reconciled(study, run):
    """Each ledger row as the budget less what the retained records charged it.

    A record charges its lane's asset row, its program's row and the shared row; a
    cancelled record holds only what it kept. This reads the records and the budget,
    not the ledger's counters, and returns `(gap, n_charges)`: the largest difference
    from the final balance and the number of record-row charges it summed.
    """
    caps, log = study.caps, run.carry.log
    expected = np.asarray(study.budget, float).copy()
    n = int(run.carry.n_log)
    lane = np.asarray(log.lane)[:n]
    program = np.asarray(log.program)[:n]
    charged = np.asarray(log.charged)[:n]
    valid = np.asarray(log.valid)[:n]
    asset_of = np.asarray(study.assets.lane_asset)
    for i in np.flatnonzero(valid):
        rows = (asset_of[lane[i]], caps.n_assets + program[i], expected.shape[0] - 1)
        for col, row in enumerate(rows):  # resource k is charged to the k-th row
            expected[row, col] -= charged[i, col]
    gap = np.abs(expected - np.asarray(run.carry.ledger.balance)).max()
    return float(gap), int(valid.sum())


@pytest.fixture(scope="module")
def alternate():
    study = two_program_study(*ALTERNATE)
    return study, _run(study)


@pytest.mark.medium  # the merge-time guard: one study, one host run
def test_alternating_programs_keep_independent_balances_and_a_stop_spares_the_other(
    alternate,
):
    """B7 row 9 (b) and (d). Fails if two programs on one channel share a balance, if
    stopping one program stops the other, or if any ledger row stops conserving.
    """
    study, run = alternate
    rows = _records(run, program=0, start_tick=0)
    program, start = rows["program"], rows["start_tick"]
    codes = _requests(run)
    first_stop = int(
        np.flatnonzero(np.asarray(run.trace.eff_kind) == EffectKind.STOP)[0]
    )
    # evidence: both programs ran before the stop, and the other ran after it
    stop_tick = int(np.asarray(run.trace.tick)[first_stop // study.caps.n_effects])
    before = start < stop_tick
    assert (program[before] == 0).sum() == 2
    assert (program[before] == 1).sum() == 2
    assert (program[~before] == 1).sum() == 1
    assert (program[~before] == 0).sum() == 0
    assert (codes == CLOSED).sum() == 1  # the stopped program asked once more
    # independent balances: each program row pays for its own admissions only
    balance = np.asarray(run.carry.ledger.balance)
    first = study.caps.n_assets
    assert balance[first, PROGRAM_COL] == pytest.approx(100.0 - 2)
    assert balance[first + 1, PROGRAM_COL] == pytest.approx(100.0 - 3)
    # (d) the per-row account balances, and it held something to balance
    assert float(np.asarray(run.carry.ledger.debited).sum()) > 0
    gap, n_charged = _reconciled(study, run)
    assert n_charged == 5
    assert gap == pytest.approx(0.0, abs=1e-9)
    # the conformance invariant (program aware since S3a) agrees with the records above
    seen = SimpleNamespace(study=study, carry=run.carry, trace=run.trace)
    result = balance_conserved(seen)
    assert result.passed, result


@pytest.mark.slow  # a second host run of the same shapes (K6)
def test_control_one_pool_would_have_charged_one_row():
    """Control for (b). With every request in program 0 the program-1 row is untouched
    and program 0 alone pays, for the four admissions before its stop, so the split
    above is the programs' and the stop closed that program only.
    """
    rows = tuple((*r[:4], 0, *r[5:]) for r in ALTERNATE)
    study = two_program_study(*rows)
    run = _run(study)
    balance = np.asarray(run.carry.ledger.balance)
    first = study.caps.n_assets
    assert balance[first, PROGRAM_COL] == pytest.approx(100.0 - 4)
    assert balance[first + 1, PROGRAM_COL] == pytest.approx(100.0)
    assert (_requests(run) == CLOSED).sum() == 2  # its two later requests


class _PlanScript(ScriptedPrograms):
    """The script, with the state increments and uniform draws of `FixedPlanPolicy`.

    The plan position `pstate[0]` advances on every Observe and the policy returns its
    own uniform draw, so the two policies leave the same state and draws in the history.
    """

    n_plan: int = eqx.field(static=True, default=0)

    def decide(self, pstate, view, study, key):
        eff, _, _ = super().decide(pstate, view, study, key)
        advance = (eff.kind[0] == EffectKind.OBSERVE) & (view.n_dec < self.n_plan)
        new_ps = pstate.at[0].add(advance.astype(F64))
        return eff, new_ps, jr.uniform(key, dtype=F64)


@pytest.mark.slow  # two more compiled programs (K6)
def test_a_one_program_study_has_the_history_of_survey_1():
    """B7 row 9 (a). The plan of `survey_1` written as a script, with one program and
    no awards, makes the same history: ints exact, floats in the go band, with no leaf
    set aside. Fails if the program machinery changes what a one-program study does.
    """
    plan = ((0, 10), (1, 10), (0, 10), (1, 10), (0, 10))
    script = (*(observe(n, s, c, 0) for n, (s, c) in enumerate(plan)), end(5))
    caps = Capacities()
    policy = _PlanScript(script=script, n_plan=len(plan))
    scripted = toy.survey_1(caps=caps, policies=(policy,))
    plain = toy.survey_1(caps=caps)
    assert caps.n_programs == 1 and plain.caps.n_programs == 1
    truths, roots = worlds(1, caps)
    a = run_on("compiled", scripted, truths, roots, 60).history(0)
    b = run_on("compiled", plain, truths, roots, 60).history(0)
    assert a.n_records == b.n_records == len(plan)  # every request ran
    report = histories_equal(a, b)
    assert not report.vacuous
    assert report.go, report


@pytest.fixture(scope="module")
def credit_runs():
    out = {}
    for name, rows, kw in (
        ("free", FREE_RUN, {}),
        ("base", BASE_RUN, {}),
        ("bad", BAD_RUN, {}),
        ("priced", FREE_RUN, {"free_cost": 1.0}),
    ):
        study = _four_channel_study(rows, **kw)
        out[name] = (study, _run(study))
    return out


def _funded_starts(run):
    rec = _records(run, channel=0, start_tick=0)
    return tuple(int(t) for t in rec["start_tick"][rec["channel"] == FUNDED])


@pytest.mark.slow  # four host runs of one compiled study (K6)
def test_a_credit_free_program_runs_only_on_an_idle_lane_and_delays_no_funded_start(
    credit_runs,
):
    """B7 row 9 (c). Fails if the credit-free program is admitted on a busy lane, if
    it needs credit, or if the funded program starts later with it than without.
    """
    study, free = credit_runs["free"]
    _, base = credit_runs["base"]
    rec = _records(free, channel=0, start_tick=0, end_tick=0, program=0)
    on_free = rec["channel"] == FREE
    # evidence: it was admitted once and refused once (the lane was busy), unfunded
    assert on_free.sum() == 1
    assert (_requests(free) == BUSY).sum() == 1
    assert (rec["program"][on_free] == 1).all()
    first = study.caps.n_assets + 1
    balance = np.asarray(free.carry.ledger.balance)
    assert balance[first, PROGRAM_COL] == 0.0  # it never held credit, so it paid none
    # admitted only on an idle lane: no two lane-2 occupancies overlap
    order = np.argsort(rec["start_tick"])
    start, end_ = rec["start_tick"][order], rec["end_tick"][order]
    assert len(start) == 3
    assert (start[1:] >= end_[:-1]).all()
    # never delays a funded start: the same ticks as the run without it
    assert len(_funded_starts(base)) == 2
    assert _funded_starts(free) == _funded_starts(base)
    gap, n_charged = _reconciled(study, free)
    assert n_charged == 3
    assert gap == pytest.approx(0.0, abs=1e-9)


@pytest.mark.slow  # shares the credit runs (K6)
def test_controls_a_misplaced_credit_free_start_delays_funded_work_and_credit_is_needed(
    credit_runs,
):
    """Control for (c), two ways. Placed across the funded request, the credit-free
    start leaves it refused and its start tick lost; priced, the same channel is
    refused for funds, so the credit-free program was the reason it ran.
    """
    _, base = credit_runs["base"]
    _, bad = credit_runs["bad"]
    assert len(_funded_starts(base)) == 2
    assert _funded_starts(base) == (5, 115)
    assert _funded_starts(bad) == (5,)  # the first start survives, the second is lost
    # the credit-free admission that interfered: one start of channel 13, unfunded
    assert (_records(bad, channel=0)["channel"] == FREE).sum() == 1
    assert (_requests(bad) == BUSY).sum() == 1
    _, priced = credit_runs["priced"]
    assert (_requests(priced) == NO_FUNDS).sum() >= 1
    assert (_records(priced, channel=0)["channel"] == FREE).sum() == 0


@pytest.mark.slow  # the compiled executor, one more program per study (K6)
def test_both_executors_make_the_same_history_for_the_interleaved_studies(
    alternate, credit_runs
):
    """Parity for (b) and (c): ints exact, floats in the go band, on both executors."""
    for study, host in (alternate, credit_runs["free"]):
        truths, roots = worlds(1, study.caps)
        a = run_on("compiled", study, truths, roots, N_STEPS)
        b = run_on("host", study, truths, roots, N_STEPS)
        assert int(np.asarray(a.carry.n_log)[0]) == int(host.carry.n_log) > 0
        assert same_histories(a, b, 1)
