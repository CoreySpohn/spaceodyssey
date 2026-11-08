"""The allocation committee: a host service answers a level-3 policy, awards flow.

A level-3 array policy asks the toy committee a `Query` at two ticks; the committee
reads the masked snapshot, splits a pool between two programs by the proposals released
to the policy, and answers with an award record; the policy turns each record into
`Allocate` effects and spends them. The guard (medium) runs the study on both executors
and replays it. The perturbation sweep and the paired controls are slow.
"""

import dataclasses
import functools
from types import SimpleNamespace

import equinox as eqx
import jax
import numpy as np
import pytest

from spaceodyssey.core import RunStatus
from spaceodyssey.core.conformance.invariants import balance_conserved
from spaceodyssey.core.enums import AdmissionCode, Audience, EffectKind
from spaceodyssey.core.executors.base import index_world
from spaceodyssey.core.preflight import preflight
from spaceodyssey.studies import allocator as al
from tests.core.helpers import worlds
from tests.studies.survey_helpers import run_with_actions, same_histories

N_STEPS = 200
OK = int(AdmissionCode.OK)
N_DECISIONS = 15  # waits and releases wake the policy too


@functools.cache
def _study(proposals=al.DEFAULT_PROPOSALS):
    return al.allocator_study(proposals)


@functools.cache
def _world(proposals=al.DEFAULT_PROPOSALS):
    study = _study(proposals)
    return study, *worlds(1, study.caps)


@functools.cache
def _run(executor, proposals=al.DEFAULT_PROPOSALS, committee="toy"):
    """`(AllocatorRun, committee service, batched result)` of one world."""
    study, truths, roots = _world(proposals)
    service = (
        al.ToyCommittee() if committee == "toy" else al.OmniscientCommittee(proposals)
    )
    acted = run_with_actions(executor, study, truths, roots, N_STEPS, service=service)
    result = acted.result
    run = al.AllocatorRun(
        jax.tree_util.tree_map(lambda x: x[0], result.carry),
        index_world(result.trace, 0),
        acted.decisions[0],
        roots[0],
    )
    return run, service, result


def _awards(run):
    """The award records in the log, in order, as payload rows."""
    log = run.carry.log
    keep = np.asarray(log.valid) & (np.asarray(log.kind) == al.AWARD_KIND_INDEX)
    return np.asarray(log.payload)[keep]


def _allocate_codes(run):
    t = run.trace
    live = np.asarray(t.handled)[:, None] & (
        np.asarray(t.eff_kind) == EffectKind.ALLOCATE
    )
    return np.asarray(t.admit_code)[live].tolist()


def test_split_pool_is_proportional_in_whole_quanta():
    assert al.split_pool([6.0, 3.0]).tolist() == [8.0, 4.0]
    assert al.split_pool([8.0, 8.0]).tolist() == [6.0, 6.0]
    assert al.split_pool([0.0, 0.0]).tolist() == [0.0, 0.0]
    assert al.split_pool([1.0, 2.0]).tolist() == [4.0, 8.0]
    assert al.split_pool([1.0, 3.0]).sum() <= al.POOL


def test_the_study_has_no_preflight_error():
    assert not [f for f in preflight(_study()) if f.severity == "error"]


def _check_answers(run, service):
    assert [r.tick for r in service.requests] == list(al.QUERY_TICKS)
    awards = _awards(run)
    assert awards.shape[0] == 2
    # (round, window from, window until, amount per program)
    assert awards[0, :5].tolist() == [0, 200, 450, 8.0, 4.0]
    assert awards[1, :5].tolist() == [1, 500, 750, 6.0, 6.0]
    assert not bool(run.carry.request.valid)
    # every award became two admitted Allocate effects and two funded observations
    assert _allocate_codes(run) == [OK] * 4
    t = run.trace
    live = np.asarray(t.handled)[:, None] & (
        np.asarray(t.eff_kind) == EffectKind.OBSERVE
    )
    assert np.asarray(t.admit_code)[live].tolist() == [OK] * 4
    granted = np.asarray(run.carry.ledger.granted)
    assert granted[2:4, al.PROGRAM_COLUMN].tolist() == [14.0, 10.0]
    assert not bool(run.carry.violation)
    held = SimpleNamespace(study=_study(), carry=run.carry, trace=run.trace)
    assert not balance_conserved(held)


@pytest.mark.medium  # one study, one host run
def test_the_committee_answers_the_policy_on_the_host():
    run, service, result = _run("host")
    assert result.history(0).status == RunStatus.COMPLETE
    _check_answers(run, service)
    report = al.replay_allocation(_study(), run)
    assert report.ok, report.failures
    assert (report.n_decisions, report.n_awards, report.n_allocations) == (
        N_DECISIONS,
        2,
        4,
    )


@pytest.mark.slow  # the compiled executor: a second program and the capture callback
def test_the_committee_answers_on_both_executors():
    host, hs, hr = _run("host")
    comp, cs, cr = _run("compiled")
    assert cr.history(0).status == RunStatus.COMPLETE
    assert same_histories(hr, cr, 1)
    assert [r.tick for r in hs.requests] == [r.tick for r in cs.requests]
    _check_answers(comp, cs)
    assert np.array_equal(_awards(host), _awards(comp))
    # the replay of the compiled run, from the views its callback captured
    report = al.replay_allocation(_study(), comp)
    assert report.ok, report.failures
    assert report.n_decisions == N_DECISIONS


def _drop_award(run, k):
    log = run.carry.log
    rows = np.flatnonzero(
        np.asarray(log.valid) & (np.asarray(log.kind) == al.AWARD_KIND_INDEX)
    )
    valid = np.asarray(log.valid).copy()
    valid[rows[k]] = False
    carry = eqx.tree_at(lambda c: c.log.valid, run.carry, valid)
    return dataclasses.replace(run, carry=carry)


def _drop_allocate(run, which):
    """Remove the `which`-th recorded Allocate effect from the decisions."""
    seen = -1
    decisions = []
    for view, eff in run.decisions:
        kind = np.asarray(eff.kind).copy()
        for slot in np.flatnonzero(kind == EffectKind.ALLOCATE):
            seen += 1
            if seen == which:
                kind[slot] = EffectKind.NONE
        decisions.append((view, dataclasses.replace(eff, kind=kind)))
    return dataclasses.replace(run, decisions=tuple(decisions))


def _flip_admission(run):
    """Record the first Allocate as refused."""
    t = run.trace
    step = int(
        np.flatnonzero(np.asarray(t.eff_kind == EffectKind.ALLOCATE).any(axis=1))[0]
    )
    slot = int(np.flatnonzero(np.asarray(t.eff_kind)[step] == EffectKind.ALLOCATE)[0])
    code = np.asarray(t.admit_code).copy()
    code[step, slot] = int(AdmissionCode.FORBIDDEN)
    return dataclasses.replace(run, trace=eqx.tree_at(lambda x: x.admit_code, t, code))


@pytest.mark.medium  # numpy edits of the host run
def test_replay_fails_when_the_run_is_altered():
    run, _, _ = _run("host")
    study = _study()
    for k in (0, 1):
        report = al.replay_allocation(study, _drop_award(run, k))
        assert not report.ok and report.n_awards == 2, k
    assert any(
        "recorded awards" in f
        for f in al.replay_allocation(study, _drop_award(run, 1)).failures
    )
    for which in range(4):
        assert not al.replay_allocation(study, _drop_allocate(run, which)).ok, which
    assert not al.replay_allocation(study, _flip_admission(run)).ok
    # a committee that read more than the view cannot reproduce the recorded awards
    assert not al.replay_allocation(study, run, al.OmniscientCommittee()).ok


def _perturbed(index, request):
    p = list(al.DEFAULT_PROPOSALS)
    t, program, _, audience = p[index]
    p[index] = (t, program, request, audience)
    return tuple(p)


OBSERVER_ONLY = [
    i for i, p in enumerate(al.DEFAULT_PROPOSALS) if p[3] == Audience.OBSERVER
]
UNRELEASED = [
    i for i, p in enumerate(al.DEFAULT_PROPOSALS) if p[0] > al.QUERY_TICKS[-1]
]
RELEASED = 1  # the proposal at tick 80


def _viewed_requests(run):
    seen = set()
    for view, _ in run.decisions:
        n = int(view.n_rows)
        kind = np.asarray(view.log.kind)[:n]
        seen |= set(
            np.asarray(view.log.payload)[:n][kind == al.PROPOSAL_KIND_INDEX][:, 0]
        )
    return seen


@pytest.mark.slow  # three more host runs of one study shape, each its own compile
def test_the_committee_reads_only_released_records():
    base, svc, _ = _run("host")
    ref = _awards(base)
    assert len(svc.requests) == 2 and ref.size
    # the observer-only and the post-query proposals never reach a policy view
    assert 40.0 not in _viewed_requests(base)
    assert 90.0 in _viewed_requests(base)  # released at 700, after both queries
    for index in (*OBSERVER_ONLY, *UNRELEASED):
        proposals = _perturbed(index, 1000.0)
        run, service, _ = _run("host", proposals)
        assert len(service.requests) == 2
        assert np.array_equal(_awards(run), ref), index
    # releasing a proposal does move the awards
    moved, _, _ = _run("host", _perturbed(RELEASED, 9.0))
    assert not np.array_equal(_awards(moved), ref)
    assert _awards(moved).shape == ref.shape


@pytest.mark.slow  # the control: the same perturbations, a committee that cheats
def test_a_committee_that_reads_the_script_fails_the_perturbation():
    base, _, _ = _run("host", committee="omniscient")
    ref = _awards(base)
    assert ref.size
    for index in (*OBSERVER_ONLY, *UNRELEASED):
        proposals = _perturbed(index, 1000.0)
        run, _, _ = _run("host", proposals, committee="omniscient")
        assert not np.array_equal(_awards(run), ref), index
