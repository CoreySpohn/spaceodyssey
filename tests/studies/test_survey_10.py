"""Survey 10, the time-allocation committee: awards from released proposals only.

A level-3 array policy asks the toy committee a `Query` at two fixed ticks; the
committee answers with an award record read from the masked snapshot, and the policy
turns each award into `Allocate` effects. One module-scoped run store holds, for each
executor, the default run and four runs of the same study structure that differ in
data only (proposal values and release ticks are traced numbers, so none of them
compiles again):

- `perturbed`: the observer-only proposal and the proposal still unreleased at both
  queries are made enormous (a);
- `early`: the unreleased proposal is released before the first query (b);
- `carry_base`, `carry_perturbed`: the carry-reading committee on the default and on
  the perturbed script (c).

Every assertion reads those runs, and each count it relies on must be nonzero. Study
runs are medium tests (the Slice 2b budget, K6): compiling the study once per executor
dominates, and the replay of (d) is eager.
"""

import dataclasses
from types import SimpleNamespace

import equinox as eqx
import jax
import numpy as np
import pytest

from spaceodyssey.core import RunStatus
from spaceodyssey.core.enums import I64, AdmissionCode, Audience, EffectKind
from spaceodyssey.core.executors.base import index_world
from spaceodyssey.core.lift import traced
from spaceodyssey.studies import allocator as al
from tests.core.helpers import worlds
from tests.studies.survey_helpers import run_with_actions, same_histories

N_STEPS = 200
OK = int(AdmissionCode.OK)
P, A = al.PROPOSAL_KIND_INDEX, al.AWARD_KIND_INDEX
FIELDS = ("amount", "window_from", "window_until", "recipient", "program", "remainder")
HUGE = 1000.0
EARLY = 180  # the unreleased proposal reaches the policy before the first query at 200
LAST_QUERY = al.QUERY_TICKS[-1]
SEALED = tuple(  # the proposals no award may read: observer-only, or unreleased
    i
    for i, p in enumerate(al.DEFAULT_PROPOSALS)
    if p[3] == Audience.OBSERVER or p[4] > LAST_QUERY
)
PENDING = next(i for i in SEALED if al.DEFAULT_PROPOSALS[i][4] > LAST_QUERY)


class Proposals(al.ProposalRecords):
    """`ProposalRecords` with the release ticks traced, so a release is data."""

    releases: tuple[int, ...] = eqx.field(
        default=(), metadata=traced(I64, sequence=True)
    )


def _study(values=None, releases=None):
    """The allocator study; `values` and `releases` replace the script's numbers."""
    study = al.allocator_study()
    old = study.process
    fields = {f.name: getattr(old, f.name) for f in dataclasses.fields(old)}
    fields["values"] = tuple(values) if values is not None else old.values
    fields["releases"] = tuple(releases) if releases is not None else old.releases
    return dataclasses.replace(study, process=Proposals(**fields))


def _script(index, *, value=None, release=None):
    values = [p[2] for p in al.DEFAULT_PROPOSALS]
    releases = [p[4] for p in al.DEFAULT_PROPOSALS]
    if value is not None:
        for i in np.atleast_1d(index):
            values[i] = value
    if release is not None:
        releases[index] = release
    return _study(values, releases)


def _run(executor, study, committee):
    truths, roots = worlds(1, study.caps)
    acted = run_with_actions(executor, study, truths, roots, N_STEPS, service=committee)
    result = acted.result
    return SimpleNamespace(
        study=study,
        service=committee,
        result=result,
        run=al.AllocatorRun(
            jax.tree_util.tree_map(lambda x: x[0], result.carry),
            index_world(result.trace, 0),
            acted.decisions[0],
            roots[0],
        ),
    )


def _carry_run(executor, study):
    """A run of the committee that is handed the carry at each snapshot."""
    from spaceodyssey.core import service as host_service

    committee = al.CarryCommittee()
    snapshot = host_service._snapshot_for

    def spy(carry, audience, decision, study):
        committee.hold(carry)
        return snapshot(carry, audience, decision, study)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(host_service, "_snapshot_for", spy)
        return _run(executor, study, committee)


def _runs(executor):
    sealed = _script(list(SEALED), value=HUGE)
    return SimpleNamespace(
        base=_run(executor, _study(), al.ToyCommittee()),
        perturbed=_run(executor, sealed, al.ToyCommittee()),
        early=_run(executor, _script(PENDING, release=EARLY), al.ToyCommittee()),
        carry_base=_carry_run(executor, _study()),
        carry_perturbed=_carry_run(executor, sealed),
    )


@pytest.fixture(scope="module")
def host_runs():
    return _runs("host")


@pytest.fixture(scope="module")
def compiled_runs():
    return _runs("compiled")


@pytest.fixture(scope="module", params=("host", "compiled"))
def runs(request):
    return request.getfixturevalue(f"{request.param}_runs")


# --------------------------------------------------------------------------- readers


def _positive(**counts):
    """Assert each counted piece of evidence is nonzero."""
    empty = [name for name, n in counts.items() if not n]
    assert not empty, f"no evidence: {empty}"


def _awards(r):
    """The award records in the log, in order, as payload rows."""
    log = r.run.carry.log
    keep = np.asarray(log.valid) & (np.asarray(log.kind) == A)
    return np.asarray(log.payload)[keep]


def _allocates(r):
    """Each recorded `Allocate` effect, in decision order, as a tuple of its fields."""
    return [
        tuple(np.asarray(getattr(eff, f))[s].tolist() for f in FIELDS)
        for _, eff in r.run.decisions
        for s in np.flatnonzero(np.asarray(eff.kind) == EffectKind.ALLOCATE)
    ]


def _follow(r):
    caps = r.study.caps
    return al.allocation_failures(
        caps.n_assets, caps.n_resources, _awards(r), _allocates(r)
    )


def _admitted(r, kind):
    """Admission codes of every `kind` effect the run requested."""
    t = r.run.trace
    live = np.asarray(t.handled)[:, None] & (np.asarray(t.eff_kind) == kind)
    return np.asarray(t.admit_code)[live].tolist()


def _proposal_rows(r, *, request):
    log = r.run.carry.log
    keep = np.asarray(log.valid) & (np.asarray(log.kind) == P)
    return np.flatnonzero(keep & (np.asarray(log.payload)[:, 0] == request))


def _viewed(r, request, *, until=None):
    """How many policy views, up to tick `until`, show a proposal of `request`."""
    n = 0
    for view, _ in r.run.decisions:
        if until is not None and int(view.tick) > until:
            continue
        rows = int(view.n_rows)
        kind = np.asarray(view.log.kind)[:rows]
        pay = np.asarray(view.log.payload)[:rows]
        n += int(np.isin(request, pay[kind == P][:, 0]))
    return n


# ------------------------------------------------------------------------- the tests


@pytest.mark.medium(
    reason="one study compiled once per executor, five runs on it: 9.3 s cold, shared"
)
def test_awards_ignore_unreleased_and_observer_only_proposals(runs):
    """(a) Perturbing a proposal the policy never sees leaves the awards identical."""
    base, moved = runs.base, runs.perturbed
    assert base.study.agenda.levels.tolist() == [3]
    assert base.result.history(0).status == RunStatus.COMPLETE
    reference = _awards(base)
    log = moved.run.carry.log
    rows = _proposal_rows(moved, request=HUGE)
    n_observer = int((np.asarray(log.audience)[rows] == int(Audience.OBSERVER)).sum())
    n_unreleased = int((np.asarray(log.released_tick)[rows] > LAST_QUERY).sum())
    _positive(
        queries=len(base.service.requests),
        awards=reference.shape[0],
        perturbed_rows=len(rows),
        observer_only_rows=n_observer,
        unreleased_rows=n_unreleased,
        views=len(base.run.decisions),
        allocates=len(_allocates(moved)),
        views_after_the_last_query=_viewed(base, 90.0),
    )
    assert [q.tick for q in moved.service.requests] == list(al.QUERY_TICKS)
    assert np.array_equal(_awards(moved), reference)
    assert _allocates(moved) == _allocates(base)
    assert _follow(moved) == []
    # the huge request is in the log but in no view before the last query; the
    # observer-only one is in no view at all, and the unreleased one is viewed only
    # after both queries
    assert _viewed(moved, 40.0) == 0
    assert _viewed(moved, HUGE, until=LAST_QUERY) == 0
    assert _viewed(moved, HUGE) > 0
    assert _viewed(base, 90.0, until=LAST_QUERY) == 0 and _viewed(base, 90.0) > 0


@pytest.mark.medium(reason="reads the five-run store, 9.3 s cold fixture shared")
def test_releasing_a_proposal_moves_the_award_the_expected_way(runs):
    """(b) Releasing the unreleased proposal early raises its program's share."""
    base, early = runs.base, runs.early
    ref, new = _awards(base), _awards(early)
    asked = al.DEFAULT_PROPOSALS[PENDING]
    program = asked[1]
    other = 1 - program
    _positive(awards=len(ref), moved_awards=len(new), allocates=len(_allocates(early)))
    assert new.shape == ref.shape
    # the first award, from the proposals released by tick 200 plus the early one
    totals = np.zeros(2)
    for i, (tick, prog, request, audience, release) in enumerate(al.DEFAULT_PROPOSALS):
        release = EARLY if i == PENDING else release
        if audience == Audience.POLICY and max(tick, release) <= al.QUERY_TICKS[0]:
            totals[prog] += request
    want = al.split_pool(totals)
    assert new[0, 3:5].tolist() == want.tolist()
    assert new[0, 3 + program] > ref[0, 3 + program]
    assert new[0, 3 + other] < ref[0, 3 + other]
    assert not np.array_equal(new, ref)
    assert _follow(early) == []
    assert _admitted(early, EffectKind.ALLOCATE) == [OK] * 4
    granted = np.asarray(early.run.carry.ledger.granted)[2:4, al.PROGRAM_COLUMN]
    assert granted.tolist() == new[:, 3:5].sum(axis=0).tolist()


@pytest.mark.medium(reason="reads the five-run store, 9.3 s cold fixture shared")
def test_a_committee_reading_the_carry_fails_the_perturbation(runs):
    """(c) The carry-reading committee's awards move under the same perturbation."""
    base, moved = runs.carry_base, runs.carry_perturbed
    toy = _awards(runs.base)
    _positive(
        carry_awards=len(_awards(base)),
        perturbed_awards=len(_awards(moved)),
        queries=len(moved.service.requests),
    )
    assert not np.array_equal(_awards(moved), _awards(base))  # fails (a)
    # it reads the sealed proposals even unperturbed, so it disagrees with the toy
    assert not np.array_equal(_awards(base), toy)
    assert _awards(moved)[0, 3:5].tolist() != _awards(base)[0, 3:5].tolist()


def _drop_award(r, k):
    log = r.carry.log
    rows = np.flatnonzero(np.asarray(log.valid) & (np.asarray(log.kind) == A))
    valid = np.asarray(log.valid).copy()
    valid[rows[k]] = False
    return dataclasses.replace(
        r, carry=eqx.tree_at(lambda c: c.log.valid, r.carry, valid)
    )


def _edit_effects(r, edit):
    return dataclasses.replace(
        r, decisions=tuple((view, edit(eff)) for view, eff in r.decisions)
    )


def _drop_allocate(r):
    """Remove the first recorded `Allocate` request."""
    done = [False]

    def edit(eff):
        kind = np.asarray(eff.kind).copy()
        slots = np.flatnonzero(kind == EffectKind.ALLOCATE)
        if len(slots) and not done[0]:
            kind[slots[0]] = EffectKind.NONE
            done[0] = True
        return dataclasses.replace(eff, kind=kind)

    return _edit_effects(r, edit)


def _bend_allocate(r):
    """Add one unit to the amount of the first recorded `Allocate`."""
    done = [False]

    def edit(eff):
        slots = np.flatnonzero(np.asarray(eff.kind) == EffectKind.ALLOCATE)
        if len(slots) and not done[0]:
            amount = np.asarray(eff.amount).copy()
            amount[slots[0], al.PROGRAM_COLUMN] += 1.0
            done[0] = True
            return dataclasses.replace(eff, amount=amount)
        return eff

    return _edit_effects(r, edit)


def _flip_admission(r):
    t = r.trace
    step = int(
        np.flatnonzero((np.asarray(t.eff_kind) == EffectKind.ALLOCATE).any(1))[0]
    )
    slot = int(np.flatnonzero(np.asarray(t.eff_kind)[step] == EffectKind.ALLOCATE)[0])
    code = np.asarray(t.admit_code).copy()
    code[step, slot] = int(AdmissionCode.FORBIDDEN)
    return dataclasses.replace(r, trace=eqx.tree_at(lambda x: x.admit_code, t, code))


@pytest.mark.medium(
    reason="replays the five-run store eagerly: 1.7 s call, 9.3 s fixture shared"
)
def test_replay_reproduces_the_run_and_fails_without_a_recorded_award(runs):
    """(d) The recorded run replays exactly; dropping one recorded award fails."""
    base = runs.base
    study, run = base.study, base.run
    report = al.replay_allocation(study, run)
    assert report.ok, report.failures
    _positive(
        decisions=report.n_decisions,
        awards=report.n_awards,
        allocations=report.n_allocations,
        admitted_allocates=len(_admitted(base, EffectKind.ALLOCATE)),
        admitted_observations=len(_admitted(base, EffectKind.OBSERVE)),
    )
    assert report.n_decisions == len(run.decisions)
    assert (report.n_awards, report.n_allocations) == (2, 4)
    assert _admitted(base, EffectKind.ALLOCATE) == [OK] * 4
    assert _admitted(base, EffectKind.OBSERVE) == [OK] * 4
    assert _awards(base)[:, :5].tolist() == [
        [0, 200, 450, 8.0, 4.0],
        [1, 500, 750, 6.0, 6.0],
    ]
    # either recorded award dropped: the committee's answer finds no record
    for k in (0, 1):
        dropped = al.replay_allocation(study, _drop_award(run, k))
        assert not dropped.ok and dropped.n_awards == 2, k
        assert any(m.startswith("query") for m in dropped.failures), k
    # the Allocate effects and the admission outcomes are compared too
    for name, edited in (
        ("allocate removed", _drop_allocate(run)),
        ("allocate amount", _bend_allocate(run)),
        ("admission code", _flip_admission(run)),
    ):
        assert not al.replay_allocation(study, edited).ok, name
    # a committee that read the carry cannot reproduce the recorded awards
    assert not al.replay_allocation(study, run, al.CarryCommittee()).ok


@pytest.mark.medium(
    reason="reads both executors' stores, 18.5 s cold for the two builds"
)
def test_both_executors_make_the_same_awards(host_runs, compiled_runs):
    """The host and compiled runs agree on histories, requests and awards."""
    for name in ("base", "perturbed", "early"):
        host, comp = getattr(host_runs, name), getattr(compiled_runs, name)
        assert same_histories(host.result, comp.result, 1), name
        assert [q.tick for q in host.service.requests] == [
            q.tick for q in comp.service.requests
        ]
        assert np.array_equal(_awards(host), _awards(comp)), name
        assert _allocates(host) == _allocates(comp), name
