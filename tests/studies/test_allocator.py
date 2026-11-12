"""The allocation committee: a host service answers a level-3 policy, awards flow.

A level-3 array policy asks the toy committee a `Query` at two ticks; the committee
reads the masked snapshot, splits a pool between two programs by the proposals released
to the policy, and answers with an award record; the policy turns each record into
`Allocate` effects and spends them.

The quick tier calls the committee and the policy directly on small masked snapshots
and checks the recorded-input checks of the replay. Everything that runs the study (the
host run and its replay, the executor parity, the perturbation sweep and the control)
is slow, because compiling the study dominates.
"""

import dataclasses
import functools
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import RunStatus
from spaceodyssey.core import service as host_service
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.conformance.invariants import balance_conserved
from spaceodyssey.core.enums import AdmissionCode, Audience, EffectKind, EventKind
from spaceodyssey.core.executors.base import index_world
from spaceodyssey.core.preflight import preflight
from spaceodyssey.core.service import Request
from tests.core.helpers import worlds
from tests.studies.scenarios import allocator as al
from tests.studies.survey_helpers import run_with_actions, same_histories

N_STEPS = 200
OK = int(AdmissionCode.OK)
P, A, M = al.PROPOSAL_KIND_INDEX, al.AWARD_KIND_INDEX, 1  # kinds; 1 is a measurement
WIDTH = 5
ROWS = 8
FIELDS = ("amount", "window_from", "window_until", "recipient", "program", "remainder")


@functools.cache
def _study(proposals=al.DEFAULT_PROPOSALS):
    return al.allocator_study(proposals)


# ---------------------------------------------------------------- quick: direct calls


class _Log(eqx.Module):
    kind: jax.Array
    payload: jax.Array
    valid: jax.Array
    audience: jax.Array
    released_tick: jax.Array


class _View(eqx.Module):
    log: _Log
    n_rows: jax.Array
    tick: jax.Array
    n_dec: jax.Array
    release_version: jax.Array


def _view(rows, n_rows=None, tick=200, n_dec=3):
    """A masked snapshot of `rows`: `(kind, payload, valid, audience, released)`."""
    pad = (0, (0,) * WIDTH, False, 0, -1)
    rows = [*rows, *[pad] * (ROWS - len(rows))]
    log = _Log(
        kind=jnp.asarray([r[0] for r in rows], jnp.int64),
        payload=jnp.asarray([list(r[1]) for r in rows], jnp.float64),
        valid=jnp.asarray([r[2] for r in rows]),
        audience=jnp.asarray([r[3] for r in rows], jnp.int64),
        released_tick=jnp.asarray([r[4] for r in rows], jnp.int64),
    )
    n = len(rows) if n_rows is None else n_rows
    return _View(
        log,
        jnp.asarray(n, jnp.int64),
        jnp.asarray(tick, jnp.int64),
        jnp.asarray(n_dec, jnp.int64),
        jnp.asarray(0, jnp.int64),
    )


def _proposal(program, request, valid=True, released=10):
    pay = (request, program, 0, 0, 0)
    return (P, pay, valid, int(Audience.POLICY), released)


def _award(rnd, a0, a1, start=200, valid=True, released=200):
    pay = (rnd, start, start + al.WINDOW, a0, a1)
    return (A, pay, valid, int(Audience.POLICY), released)


@functools.cache
def _decide():
    return eqx.filter_jit(al.AllocationPolicy().decide)


def _effects(view, asked, done):
    study = _study()
    ps = jnp.zeros(study.caps.n_pstate).at[0].set(asked).at[1].set(done)
    eff, _, _ = _decide()(ps, view, study, jax.random.PRNGKey(0))
    return eff


@pytest.mark.parametrize("n_effects", (5, 6))
def test_the_policy_fills_the_effect_table_of_any_capacity(n_effects):
    study = al.allocator_study(
        caps=Capacities(n_programs=2, n_effects=n_effects, awards=True)
    )
    view = _view([_award(0, 8.0, 4.0)])
    ps = jnp.zeros(study.caps.n_pstate).at[0].set(1)
    eff, _, _ = al.AllocationPolicy().decide(ps, view, study, jax.random.PRNGKey(0))
    assert eff.kind.shape == (n_effects,) and eff.amount.shape[0] == n_effects
    assert not np.asarray(eff.kind)[5:].any()  # rows past the policy's own are empty
    assert len(_allocates(eff)) == 2


def test_the_policy_refuses_a_table_narrower_than_its_own():
    study = al.allocator_study(caps=Capacities(n_programs=2, n_effects=5, awards=True))
    narrow = dataclasses.replace(
        study, caps=dataclasses.replace(study.caps, n_effects=4)
    )
    with pytest.raises(ValueError, match=r"emits 5 effect rows.*n_effects is 4"):
        al.AllocationPolicy().decide(
            jnp.zeros(study.caps.n_pstate), _view([]), narrow, jax.random.PRNGKey(0)
        )


def _allocates(eff):
    return [
        tuple(np.asarray(getattr(eff, f))[s] for f in FIELDS)
        for s in np.flatnonzero(np.asarray(eff.kind) == EffectKind.ALLOCATE)
    ]


def _request(tick):
    return Request(tick, 0, 0, int(Audience.POLICY), 0, 0, 3, 0)


def test_split_pool_is_proportional_in_whole_quanta():
    assert al.split_pool([6.0, 3.0]).tolist() == [8.0, 4.0]
    assert al.split_pool([8.0, 8.0]).tolist() == [6.0, 6.0]
    assert al.split_pool([0.0, 0.0]).tolist() == [0.0, 0.0]
    assert al.split_pool([1.0, 2.0]).tolist() == [4.0, 8.0]
    assert al.split_pool([1.0, 3.0]).sum() <= al.POOL


def test_the_study_has_no_preflight_error():
    assert not [f for f in preflight(_study()) if f.severity == "error"]


def test_the_committee_reads_only_the_rows_the_view_shows():
    seen = [_proposal(0, 6.0), _proposal(1, 3.0), _award(0, 8.0, 4.0)]
    committee = al.ToyCommittee()
    view = _view(seen)
    row = committee.answer(_request(200), view, None).tolist()
    assert row == [1, 200, 450, 8.0, 4.0]  # round 1: one award is already in view
    # rows the snapshot does not show (invalid, past `n_rows`) and other kinds are inert
    noise = [_proposal(0, 500.0, valid=False), (M, (500.0,) * WIDTH, True, 3, 5)]
    padded = _view([*seen, *noise, _proposal(1, 700.0)], n_rows=len(seen) + 2)
    assert committee.answer(_request(200), padded, None).tolist() == row
    # a row inside the view does move it
    more = _view([*seen, _proposal(1, 9.0)])
    assert committee.answer(_request(200), more, None).tolist() != row


def test_the_policy_turns_an_award_into_its_allocations():
    n_assets = _study().caps.n_assets
    view = _view([_award(0, 8.0, 4.0)])
    eff = _effects(view, asked=1, done=0)
    got = _allocates(eff)
    assert len(got) == 2
    for p, (amount, start, until, recipient, program, _) in enumerate(got):
        assert amount.tolist() == [0.0, (8.0, 4.0)[p], 0.0]
        assert (start, until) == (200, 450)
        assert (recipient, program) == (n_assets + p, p)
    awards = [np.asarray(view.log.payload)[0]]
    assert al.allocation_failures(n_assets, 3, awards, got) == []
    # each field of a recorded Allocate is checked against the award
    for i, name in enumerate(FIELDS):
        bad = [list(g) for g in got]
        bad[0][i] = bad[0][i] + 1 if name != "program" else 1 - bad[0][i]
        failed = al.allocation_failures(n_assets, 3, awards, [tuple(b) for b in bad])
        assert failed, name
    assert al.allocation_failures(n_assets, 3, awards, got[:1])
    # a program with no award funds nothing; a stale, invalid or hidden award is ignored
    assert len(_allocates(_effects(_view([_award(0, 8.0, 0.0)]), 1, 0))) == 1
    for rows, n in (
        ([_award(1, 8.0, 4.0)], None),  # another round
        ([_award(0, 8.0, 4.0, valid=False)], None),
        ([_award(0, 8.0, 4.0)], 0),  # past n_rows
    ):
        eff = _effects(_view(rows, n_rows=n), asked=1, done=0)
        assert not _allocates(eff)
    assert not _allocates(_effects(view, asked=0, done=0))  # nothing was asked


def test_dropping_a_recorded_award_fails_the_check():
    committee = al.ToyCommittee()
    v1 = _view([_proposal(0, 6.0), _proposal(1, 3.0)], tick=200)
    first = committee.answer(_request(200), v1, None)
    v2 = _view([_proposal(0, 6.0), _proposal(1, 3.0), _award(0, 8.0, 4.0)], tick=500)
    second = committee.answer(_request(500), v2, None)
    asked = [(_request(200), v1), (_request(500), v2)]
    recorded = [(first, 200), (second, 500)]
    assert al.award_failures(committee, asked, recorded) == []
    assert al.award_failures(committee, asked, recorded[1:])  # first dropped
    assert al.award_failures(committee, asked, recorded[:1])  # second dropped
    assert al.award_failures(committee, asked, [(first, 200), (second + 1, 500)])


def test_awards_must_be_visible_to_the_policy_as_recorded():
    shown = _view([_proposal(0, 6.0), _award(0, 8.0, 4.0)], tick=300)

    def log(award):
        rows = [_proposal(0, 6.0), award]
        return SimpleNamespace(
            kind=np.asarray([r[0] for r in rows]),
            payload=np.asarray([list(r[1]) for r in rows]),
            valid=np.asarray([r[2] for r in rows]),
            audience=np.asarray([r[3] for r in rows]),
            released_tick=np.asarray([r[4] for r in rows]),
        )

    def failures(award, step=9, released=(2, 5)):
        return al.visibility_failures(
            shown, log(award), step, 300, np.asarray(released)
        )

    assert failures(_award(0, 8.0, 4.0)) == []
    observer_only = (A, _award(0, 8.0, 4.0)[1], True, int(Audience.OBSERVER), 200)
    assert failures(observer_only)
    assert failures(_award(0, 8.0, 4.0, released=-1))  # never stamped released
    assert failures(_award(0, 8.0, 4.0, released=400))  # stamped after the decision
    assert failures(_award(0, 8.0, 4.0), released=(2, -1))  # no release step
    assert failures(_award(0, 8.0, 4.0), released=(2, 12))  # released after this step
    assert failures(_award(0, 9.0, 4.0))  # the view's award is not the logged one


# ------------------------------------------------------------------ slow: study runs


@functools.cache
def _world(proposals=al.DEFAULT_PROPOSALS):
    study = _study(proposals)
    return study, *worlds(1, study.caps)


@functools.cache
def _holding(service, monkeypatch):
    """Hand the carry to a carry-reading committee at every snapshot the host takes."""
    snapshot = host_service._snapshot_for

    def spy(carry, audience, decision, study):
        service.hold(carry)
        return snapshot(carry, audience, decision, study)

    monkeypatch.setattr(host_service, "_snapshot_for", spy)


def _run(executor, proposals=al.DEFAULT_PROPOSALS, committee="toy", monkeypatch=None):
    """`(AllocatorRun, committee service, batched result)` of one world."""
    study, truths, roots = _world(proposals)
    service = al.ToyCommittee() if committee == "toy" else al.CarryCommittee()
    if committee != "toy":
        _holding(service, monkeypatch)
    acted = run_with_actions(executor, study, truths, roots, N_STEPS, service=service)
    result = acted.result
    run = al.AllocatorRun(
        jax.tree_util.tree_map(lambda x: x[0], result.carry),
        index_world(result.trace, 0),
        acted.decisions[0],
        roots[0],
        jax.tree_util.tree_map(lambda x: x[0], truths),
    )
    return run, service, result


def _awards(run):
    """The award records in the log, in order, as payload rows."""
    log = run.carry.log
    keep = np.asarray(log.valid) & (np.asarray(log.kind) == A)
    return np.asarray(log.payload)[keep]


def _recorded_allocates(run):
    out = []
    for _, eff in run.decisions:
        for s in np.flatnonzero(np.asarray(eff.kind) == EffectKind.ALLOCATE):
            out.append(tuple(np.asarray(getattr(eff, f))[s] for f in FIELDS))
    return out


def _follows_awards(run, study=None):
    study = study or _study()
    caps = study.caps
    return al.allocation_failures(
        caps.n_assets, caps.n_resources, _awards(run), _recorded_allocates(run)
    )


def _check_answers(run, service):
    assert [r.tick for r in service.requests] == list(al.QUERY_TICKS)
    awards = _awards(run)
    assert awards.shape[0] == 2
    # (round, window from, window until, amount per program)
    assert awards[0, :5].tolist() == [0, 200, 450, 8.0, 4.0]
    assert awards[1, :5].tolist() == [1, 500, 750, 6.0, 6.0]
    assert not bool(run.carry.request.valid)
    # every award became two admitted Allocate effects and two funded observations
    t = run.trace
    handled = np.asarray(t.handled)[:, None]
    for kind in (EffectKind.ALLOCATE, EffectKind.OBSERVE):
        live = handled & (np.asarray(t.eff_kind) == kind)
        assert np.asarray(t.admit_code)[live].tolist() == [OK] * 4
    assert _follows_awards(run) == []
    granted = np.asarray(run.carry.ledger.granted)
    assert granted[2:4, al.PROGRAM_COLUMN].tolist() == [14.0, 10.0]
    assert not bool(run.carry.violation)
    held = SimpleNamespace(study=_study(), carry=run.carry, trace=run.trace)
    assert not balance_conserved(held)


def _pending_row(run):
    """The log row of the proposal written at 150 and released at 700."""
    log = run.carry.log
    hit = np.asarray(log.valid) & (np.asarray(log.kind) == P)
    hit &= np.asarray(log.payload)[:, 0] == 90.0
    (row,) = np.flatnonzero(hit)
    return row


@pytest.mark.slow(reason="one host run of the study: the compile dominates")
def test_the_committee_answers_the_policy_on_the_host():
    run, service, result = _run("host")
    assert result.history(0).status == RunStatus.COMPLETE
    _check_answers(run, service)
    # the pending proposal is in the log before the first query and released after both
    row = _pending_row(run)
    log = run.carry.log
    assert int(log.start_tick[row]) < al.QUERY_TICKS[0]
    assert int(log.released_tick[row]) > al.QUERY_TICKS[-1]
    report = al.replay_allocation(_study(), run)
    assert report.ok, report.failures
    assert (report.n_awards, report.n_allocations) == (2, 4)
    assert report.n_decisions == len(run.decisions) > 10


def _with_effects(run, edit):
    decisions = tuple((view, edit(eff)) for view, eff in run.decisions)
    return dataclasses.replace(run, decisions=decisions)


def _drop_award(run, k):
    log = run.carry.log
    rows = np.flatnonzero(np.asarray(log.valid) & (np.asarray(log.kind) == A))
    valid = np.asarray(log.valid).copy()
    valid[rows[k]] = False
    return dataclasses.replace(
        run, carry=eqx.tree_at(lambda c: c.log.valid, run.carry, valid)
    )


def _set_award(run, field, value):
    log = run.carry.log
    row = np.flatnonzero(np.asarray(log.valid) & (np.asarray(log.kind) == A))[0]
    column = np.asarray(getattr(log, field)).copy()
    column[row] = value
    return dataclasses.replace(
        run, carry=eqx.tree_at(lambda c: getattr(c.log, field), run.carry, column)
    )


def _drop_allocate(run, which):
    seen = [-1]

    def edit(eff):
        kind = np.asarray(eff.kind).copy()
        for slot in np.flatnonzero(kind == EffectKind.ALLOCATE):
            seen[0] += 1
            if seen[0] == which:
                kind[slot] = EffectKind.NONE
        return dataclasses.replace(eff, kind=kind)

    return _with_effects(run, edit)


def _corrupt(run, kind, field):
    """Change one field of the first recorded effect of `kind`."""
    done = [False]

    def edit(eff):
        slots = np.flatnonzero(np.asarray(eff.kind) == kind)
        if done[0] or not len(slots):
            return eff
        done[0] = True
        column = np.asarray(getattr(eff, field)).copy()
        s = slots[0]
        column[s] = column[s] + 1 if field != "program" else 1 - column[s]
        return dataclasses.replace(eff, **{field: column})

    return _with_effects(run, edit)


def _set_slot(run, slot):
    """Point the first release step at log slot `slot`."""
    t = run.trace
    kind = np.asarray(t.kind)
    step = int(np.flatnonzero(np.asarray(t.handled) & (kind == EventKind.RELEASE))[0])
    column = np.asarray(t.log_slot).copy()
    column[step] = slot
    return dataclasses.replace(run, trace=eqx.tree_at(lambda x: x.log_slot, t, column))


def _flip_admission(run):
    t = run.trace
    step = int(
        np.flatnonzero((np.asarray(t.eff_kind) == EffectKind.ALLOCATE).any(1))[0]
    )
    slot = int(np.flatnonzero(np.asarray(t.eff_kind)[step] == EffectKind.ALLOCATE)[0])
    code = np.asarray(t.admit_code).copy()
    code[step, slot] = int(AdmissionCode.FORBIDDEN)
    return dataclasses.replace(run, trace=eqx.tree_at(lambda x: x.admit_code, t, code))


@pytest.mark.slow(reason="edits of the host run")
def test_replay_fails_when_the_run_is_altered():
    run, _, _ = _run("host")
    study = _study()
    ref = al.reference_execution(study, run)
    assert al.replay_allocation(study, run, reference=ref).ok
    for k in (0, 1):
        report = al.replay_allocation(study, _drop_award(run, k), reference=ref)
        assert not report.ok and report.n_awards == 2, k
    for which in range(4):
        assert not al.replay_allocation(
            study, _drop_allocate(run, which), reference=ref
        ).ok, which
        assert _follows_awards(_drop_allocate(run, which)), which
    for field in FIELDS:
        altered = _corrupt(run, EffectKind.ALLOCATE, field)
        assert not al.replay_allocation(study, altered, reference=ref).ok, field
        assert _follows_awards(altered), field
    # every other effect the policy emitted is compared with the replayed one too
    for kind, field in (
        (EffectKind.OBSERVE, "program"),
        (EffectKind.OBSERVE, "subject"),
        (EffectKind.OBSERVE, "channel_index"),
        (EffectKind.WAIT, "tick"),
        (EffectKind.QUERY, "subject"),
        (EffectKind.STOP, "scope"),
    ):
        altered = _corrupt(run, kind, field)
        assert not al.replay_allocation(study, altered, reference=ref).ok, (kind, field)
    # an award the policy was never shown, as the log records it
    for field, value in (("audience", int(Audience.OBSERVER)), ("released_tick", -1)):
        report = al.replay_allocation(
            study, _set_award(run, field, value), reference=ref
        )
        assert not report.ok, field
    assert not al.replay_allocation(study, _flip_admission(run), reference=ref).ok
    # a release that names a log slot past the end of the log is a failed report
    for slot in (len(np.asarray(run.carry.log.kind)), 10**6):
        report = al.replay_allocation(study, _set_slot(run, slot), reference=ref)
        assert not report.ok, slot
    # a committee that read more than the view cannot reproduce the recorded awards
    assert not al.replay_allocation(study, run, al.CarryCommittee()).ok


@pytest.mark.slow(
    reason="the compiled executor: a second program and the capture callback"
)
def test_the_committee_answers_on_both_executors():
    host, hs, hr = _run("host")
    comp, cs, cr = _run("compiled")
    assert cr.history(0).status == RunStatus.COMPLETE
    assert same_histories(hr, cr, 1)
    assert [r.tick for r in hs.requests] == [r.tick for r in cs.requests]
    _check_answers(comp, cs)
    assert np.array_equal(_awards(host), _awards(comp))
    report = al.replay_allocation(_study(), comp)
    assert report.ok, report.failures
    assert report.n_decisions == len(comp.decisions)


def _perturbed(index, request):
    p = list(al.DEFAULT_PROPOSALS)
    p[index] = (p[index][0], p[index][1], request, *p[index][3:])
    return tuple(p)


def _released_at(index, tick):
    p = list(al.DEFAULT_PROPOSALS)
    p[index] = (*p[index][:4], tick)
    return tuple(p)


OBSERVER_ONLY = [
    i for i, p in enumerate(al.DEFAULT_PROPOSALS) if p[3] == Audience.OBSERVER
]
PENDING = [i for i, p in enumerate(al.DEFAULT_PROPOSALS) if p[4] > al.QUERY_TICKS[-1]]
RELEASED = 1  # the proposal at tick 80


def _viewed_requests(run):
    seen = set()
    for view, _ in run.decisions:
        n = int(view.n_rows)
        kind = np.asarray(view.log.kind)[:n]
        pay = np.asarray(view.log.payload)[:n]
        seen |= set(pay[kind == P][:, 0])
    return seen


@pytest.mark.slow(reason="four more host runs of one study shape, each its own compile")
def test_the_committee_reads_only_released_records():
    base, svc, _ = _run("host")
    ref = _awards(base)
    assert len(svc.requests) == 2 and ref.size
    # the observer-only proposal never reaches a policy view; the pending one reaches
    # it only after both queries
    assert 40.0 not in _viewed_requests(base)
    assert 90.0 in _viewed_requests(base)
    assert not any(
        90.0 in np.asarray(v.log.payload)[: int(v.n_rows)][:, 0]
        for v, _ in base.decisions
        if int(v.tick) <= al.QUERY_TICKS[-1]
    )
    for index in (*OBSERVER_ONLY, *PENDING):
        run, service, _ = _run("host", _perturbed(index, 1000.0))
        assert len(service.requests) == 2
        assert np.array_equal(_awards(run), ref), index
        assert _follows_awards(run) == []
    # releasing the pending proposal before the first query moves the awards, and the
    # allocations follow the new awards
    early = _released_at(PENDING[0], 180)
    moved, _, _ = _run("host", early)
    assert not np.array_equal(_awards(moved), ref)
    assert _awards(moved).shape == ref.shape
    assert _follows_awards(moved, _study(early)) == []
    assert al.replay_allocation(_study(early), moved).ok
    assert _awards(moved)[0, 3:5].tolist() != ref[0, 3:5].tolist()


@pytest.mark.slow(
    reason="the control: the same perturbations, a carry-reading committee"
)
def test_a_committee_that_reads_unreleased_records_fails_the_perturbation(monkeypatch):
    base, _, _ = _run("host", committee="carry", monkeypatch=monkeypatch)
    ref = _awards(base)
    assert ref.size
    for index in (*OBSERVER_ONLY, *PENDING):
        proposals = _perturbed(index, 1000.0)
        run, _, _ = _run("host", proposals, committee="carry", monkeypatch=monkeypatch)
        assert not np.array_equal(_awards(run), ref), index
