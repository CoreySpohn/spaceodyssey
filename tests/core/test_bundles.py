"""Bundles: observations that share one lane run as one occupancy.

The rows of one admission group that share a bundle ordinal are admitted whole, start
together, pay one transition, share one lane generation and keep the lane busy until the
longest member ends. Cancelling any member settles every pending member. Every run and
hand-built admission below shares one compiled study structure (programs, refunds,
awards, commitments and bundles on); scenarios differ only in script data.
"""

import dataclasses
from types import SimpleNamespace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey import core
from spaceodyssey.core import (
    CompiledExecutor,
    HostExecutor,
    admit,
    battery,
    histories_equal,
    init_carry,
)
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.conformance.invariants import balance_conserved
from spaceodyssey.core.enums import (
    I64,
    AdmissionCode,
    EffectKind,
    EventKind,
    RecordStatus,
)
from spaceodyssey.core.snapshot import effects_none
from spaceodyssey.studies import toy
from tests.core.award_helpers import PROGRAM_COL, ScriptedAwards, award
from tests.core.helpers import one_world, worlds
from tests.core.program_helpers import cancel, end, observe, wait

CAPS = Capacities(
    n_effects=4,
    n_channels=4,
    n_programs=2,
    awards=True,
    refunds=True,
    n_commitments=2,
    max_bundles=2,
    max_bundle_size=2,
)
# Channels 12 (50 ticks) and 13 (40 ticks) occupy lane 2 behind a 5 tick transition.
LANE = 2
LONG, SHORT = 12, 13
LONG_IX, SHORT_IX = 2, 3  # their rows of the channel table
OK = int(AdmissionCode.OK)
BUSY = int(AdmissionCode.LANE_BUSY)
UNSUPPORTED = int(AdmissionCode.UNSUPPORTED)
N_STEPS = 200
PROGRAM_ROW = CAPS.n_assets + 1  # the balance row of program 1


def _as_plan(rows):
    out = np.full((8, 4), -1, np.int64)
    for i, r in enumerate(rows):
        out[i] = r
    return jnp.asarray(out, I64)


def member(n, slot, ordinal=0, group_slot=None):
    """Effect row `slot` of decision `n` joins bundle `ordinal` of a group."""
    return (n, slot, slot if group_slot is None else group_slot, ordinal)


class ScriptedBundles(ScriptedAwards):
    """The scripted effects with the bundle ordinal and group of each `member` row."""

    plan: object = eqx.field(default=None, converter=lambda r: _as_plan(r or ()))

    def decide(self, pstate, view, study, key):
        """Emit the script rows, then name each planned row's bundle and group."""
        eff, ps, u = super().decide(pstate, view, study, key)
        n = study.caps.n_effects
        ordinal = jnp.full(n, -1, I64)
        group = eff.group
        for e in range(n):
            pick = (self.plan[:, 0] == view.n_dec) & (self.plan[:, 1] == e)
            r = self.plan[jnp.argmax(pick)]
            ordinal = ordinal.at[e].set(jnp.where(pick.any(), r[3], -1))
            joined = view.n_dec * n + r[2]
            group = group.at[e].set(jnp.where(pick.any(), joined, group[e]))
        return dataclasses.replace(eff, bundle=ordinal, group=group), ps, u


def bundle_study(script=(), plan=(), awards=()):
    """The toy study on `CAPS` with a scripted policy; program rows hold 50 credit."""
    policy = ScriptedBundles(script=script, awards=awards, plan=plan)
    study = toy.survey_1(caps=CAPS, policies=(policy,))
    budget = study.budget
    for p in range(CAPS.n_programs):
        budget = budget.at[CAPS.n_assets + p, PROGRAM_COL].set(50.0)
    return dataclasses.replace(study, budget=budget)


def run(study):
    """A host run of one world."""
    truth, root = one_world(caps=CAPS)
    result = HostExecutor().run(study, truth, root, N_STEPS)
    return SimpleNamespace(study=study, carry=result.carry, trace=result.trace)


def rows(result):
    """The valid log rows of a run, as dicts of column name to value."""
    log = result.carry.log
    keep = np.flatnonzero(np.asarray(log.valid))
    cols = ("channel", "status", "bundle", "start_tick", "end_tick", "program")
    return [{c: int(np.asarray(getattr(log, c))[i]) for c in cols} for i in keep]


def codes(result, decision):
    """The admission codes of the step that made decision `decision`."""
    trace = result.trace
    step = np.flatnonzero(
        np.asarray(trace.handled)
        & (np.asarray(trace.n_dec) == decision + 1)
        & (np.asarray(trace.admit_code) != AdmissionCode.NONE).any(axis=1)
    )[0]
    return np.asarray(trace.admit_code)[step].tolist()


# The pair on lane 2 at tick 0 starts at 5; the short member ends at 45 and the long one
# at 55. Each completion triggers a decision: the one at 45 asks for the lane again and
# is refused, the one at 55 starts a second pair (decision 2, bundle id 4).
PAIR = (member(0, 0), member(0, 1, group_slot=0))
REPEAT = (member(2, 0), member(2, 1, group_slot=0))
SCRIPTS = {
    "pair": (
        (
            observe(0, 0, LONG, 1),
            observe(0, 1, SHORT, 1),
            observe(1, 0, LONG, 1),
            observe(2, 0, LONG, 1),
            observe(2, 1, SHORT, 1),
            end(2),
        ),
        (*PAIR, *REPEAT),
        (),
    ),
    # cancelled at tick 20, both members still pending, with an award window open
    "cancel_all": (
        (
            observe(0, 0, LONG, 1),
            observe(0, 1, SHORT, 1),
            wait(0, 20),
            cancel(1, LANE),
            end(1),
        ),
        PAIR,
        (award(0, 3, PROGRAM_ROW, 0, 100, 20.0),),
    ),
    # cancelled at tick 45, just after the short member completed
    "cancel_rest": (
        (observe(0, 0, LONG, 1), observe(0, 1, SHORT, 1), cancel(1, LANE), end(1)),
        PAIR,
        (),
    ),
}


@pytest.fixture(scope="module")
def runs():
    """One host run per scenario, all on the one compiled structure."""
    return {name: run(bundle_study(*spec)) for name, spec in SCRIPTS.items()}


@pytest.fixture(scope="module")
def hand():
    """The study, a fresh carry and a jitted `admit` for hand-built decisions."""
    study = bundle_study()
    truth, _ = one_world(caps=CAPS)
    carry = eqx.filter_jit(init_carry)(study, truth)
    return study, carry, eqx.filter_jit(admit)


def table(members, *, start=0, kind=EffectKind.OBSERVE):
    """Effect rows for `members`: (channel row, group, ordinal, program) each."""
    eff = effects_none(CAPS)
    cols = {k: np.asarray(getattr(eff, k)).copy() for k in ("kind", "channel_index")}
    for k in ("group", "bundle", "program", "start_tick"):
        cols[k] = np.asarray(getattr(eff, k)).copy()
    for e, (channel, group, ordinal, program) in enumerate(members):
        cols["kind"][e] = kind
        cols["channel_index"][e] = channel
        cols["group"][e] = group
        cols["bundle"][e] = ordinal
        cols["program"][e] = program
        cols["start_tick"][e] = start
    return dataclasses.replace(eff, **{k: jnp.asarray(v, I64) for k, v in cols.items()})


def done_rows(carry):
    """Indices of the queued Done events."""
    return np.flatnonzero(
        np.asarray(carry.ev.valid & (carry.ev.kind == EventKind.DONE))
    )


@pytest.mark.medium(reason="three host runs on one compiled structure, 7 s cold")
def test_members_start_together_and_pay_one_transition(runs):
    """Fails if a bundle's members start one after another (a second transition) or
    carry different bundle ids, or if two decisions' first bundles share an id.
    """
    first, second = rows(runs["pair"])[:2], rows(runs["pair"])[2:]
    assert [r["status"] for r in first + second] == [RecordStatus.COMPLETED] * 4
    for pair in (first, second):
        assert len({r["start_tick"] for r in pair}) == 1
        assert len({r["bundle"] for r in pair}) == 1
    assert first[0]["start_tick"] == 5  # one 5 tick transition, not two
    assert sorted(r["end_tick"] for r in first) == [45, 55]
    # ordinal 0 of decision 0 and of decision 2 are different bundles
    assert (first[0]["bundle"], second[0]["bundle"]) == (0, 2 * CAPS.max_bundles)


@pytest.mark.medium(reason="three host runs on one compiled structure, 7 s cold")
def test_the_lane_frees_only_at_the_last_members_done(runs):
    """Fails if the lane frees when the shorter member completes (45) instead of the
    longer one (55), so a decision at 45 is admitted onto a busy lane.
    """
    result = runs["pair"]
    assert codes(result, 1)[0] == BUSY
    assert codes(result, 2)[:2] == [OK, OK]
    assert min(r["start_tick"] for r in rows(result)[2:]) == 55 + 5


@pytest.mark.parametrize("order", [(LONG_IX, SHORT_IX), (SHORT_IX, LONG_IX)])
def test_a_shorter_member_committed_second_leaves_the_longer_end(hand, order):
    """Fails if the busy window follows the last member committed (45) instead of the
    longest (55): the stored window is only ever extended.
    """
    study, c0, step = hand
    a, b = order
    c1, out = step(c0, table([(a, 0, 0, 1), (b, 0, 0, 1)]), study)
    assert out[:2].tolist() == [OK, OK]
    assert int(c1.ledger.lane_busy_until[LANE]) == 55
    # one lane generation for the pair, named by both Done events
    done = done_rows(c1)
    assert len(done) == 2
    assert int(c1.ledger.lane_gen[LANE]) == 1
    assert set(np.asarray(c1.ev.gen)[done].tolist()) == {1}
    assert set(np.asarray(c1.ev.bundle)[done].tolist()) == {0}


def test_a_bundle_pays_one_transition_in_ticks(hand):
    """Fails if each member pays its own transition: with one 5 tick transition the
    Done rows fall at 5 + 40 and 5 + 50, a second transition would push them later.
    """
    study, c0, step = hand
    c1, _ = step(c0, table([(LONG_IX, 0, 0, 1), (SHORT_IX, 0, 0, 1)]), study)
    done = done_rows(c1)
    assert sorted(np.asarray(c1.ev.tick)[done].tolist()) == [5 + 40, 5 + 50]
    assert int(c1.ledger.lane_gen[LANE]) == 1  # not 2: one occupancy


REFUSALS = {
    "two lanes": ([(LONG_IX, 0, 0, 1), (0, 0, 0, 1)], 0),
    "two groups": ([(LONG_IX, 0, 0, 1), (SHORT_IX, 1, 0, 1)], 0),
    "two programs": ([(LONG_IX, 0, 0, 1), (SHORT_IX, 0, 0, 0)], 0),
    "ordinal at the capacity": ([(LONG_IX, 0, CAPS.max_bundles, 1)], 0),
    "over the size": ([(LONG_IX, 0, 0, 1)] * 3, 0),
    "future start": ([(LONG_IX, 0, 0, 1), (SHORT_IX, 0, 0, 1)], 9),
}


@pytest.mark.parametrize("name", list(REFUSALS))
def test_a_bundle_the_lane_cannot_take_whole_is_refused_untouched(hand, name):
    """Fails if a bundle mixing lanes, groups or programs, naming an ordinal past the
    capacity, holding too many members or starting later is admitted, or if refusing
    it changes the carry.
    """
    study, c0, step = hand
    members, start = REFUSALS[name]
    c1, out = step(c0, table(members, start=start), study)
    assert out[: len(members)].tolist() == [UNSUPPORTED] * len(members)
    report = core.tree_equal_report(
        (c0.ledger, c0.ev, c0.log), (c1.ledger, c1.ev, c1.log)
    )
    assert report["max_int_diff"] == 0 and report["max_abs"] == 0.0


@pytest.mark.medium(reason="three host runs on one compiled structure, 7 s cold")
def test_cancelling_a_member_settles_every_pending_member(runs):
    """Fails if a cancel settles only the member it hits: the other keeps its Done and
    its debit, or a record carries a different bundle id, or the refund is lost.
    """
    result = runs["cancel_all"]
    done = rows(result)
    assert [r["status"] for r in done] == [RecordStatus.PARTIAL] * 2
    assert len({r["bundle"] for r in done}) == 1 and done[0]["bundle"] >= 0
    assert not done_rows(result.carry).size  # nothing left pending
    assert (
        int(result.carry.ledger.lane_gen[LANE]) == 2
    )  # one bump to start, one to cancel
    assert len(done) == 2  # exactly the two Done rows were settled
    ledger = result.carry.ledger
    own = [(1, 0), (PROGRAM_ROW, 1), (CAPS.n_assets + CAPS.n_programs, 2)]
    charged = np.asarray([ledger.debited[r, k] for r, k in own])
    refunded = np.asarray([ledger.refunded[r, k] for r, k in own])
    # each member debited its own unit cost once, and refunded its unused share once:
    # cancelled at 20, the long member has 35 of 50 ticks left and the short 25 of 40
    np.testing.assert_allclose(charged, 2.0)
    np.testing.assert_allclose(refunded, 35 / 50 + 25 / 40)
    assert balance_conserved(result).passed  # refunds, programs and the award window


@pytest.mark.medium(reason="three host runs on one compiled structure, 7 s cold")
def test_a_cancel_settles_the_members_still_pending_and_no_other(runs):
    """Fails if a cancel after the short member completed re-settles it, or leaves the
    long member running.
    """
    status = {r["channel"]: r["status"] for r in rows(runs["cancel_rest"])}
    assert status == {SHORT: RecordStatus.COMPLETED, LONG: RecordStatus.PARTIAL}
    assert balance_conserved(runs["cancel_rest"]).passed


@pytest.mark.medium(reason="the bundle study on both executors, 12 s cold")
def test_both_executors_agree_on_bundles():
    """Fails if the compiled scan and the host loop disagree on a bundle run."""
    study = bundle_study(*SCRIPTS["pair"])
    truths, roots = worlds(2, CAPS)
    compiled = CompiledExecutor().run_worlds(study, truths, roots, N_STEPS)
    host = HostExecutor().run_worlds(study, truths, roots, N_STEPS)
    assert host.status.tolist() == compiled.status.tolist()
    for w in range(2):
        report = histories_equal(compiled.history(w), host.history(w))
        assert report, report


@pytest.mark.medium(reason="the bundle study through the fork battery, 15 s cold")
def test_the_fork_battery_passes_on_bundles():
    """Fails if a bundle id makes a forked branch differ from its trunk before the
    fork, or in the records the fork does not touch.
    """
    study = bundle_study(*SCRIPTS["pair"])
    truths, roots = worlds(2, CAPS)
    result = battery.fork_check(study, truths, roots, N_STEPS, at=1)
    assert result, result.diagnostic
