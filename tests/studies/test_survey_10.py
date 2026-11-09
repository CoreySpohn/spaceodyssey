"""Survey 10, the time-allocation committee: awards from released proposals only.

A level-3 array policy asks the toy committee a `Query` at two fixed ticks; the
committee answers with an award record read from the masked snapshot, and the policy
turns each award into `Allocate` effects. The study is the production allocator study.
One module-scoped run store holds, for each executor, the default run and the runs that
differ from it in data only (proposal requests are traced numbers, so they share its
program), and one run that releases a proposal early, which has a program of its own
(release ticks are static in `ProposalRecords`):

- `observer`, `unreleased`: the observer-only proposal, and the proposal still
  unreleased at both queries, each made enormous in its own run (a);
- `early`: the unreleased proposal is released before the first query (b);
- `carry_base`, `carry_observer`, `carry_unreleased`: the carry-reading committee on
  the default script and on each perturbation (c).

Every assertion reads those runs, and each count it relies on must be nonzero. Study
runs are medium tests (the Slice 2b budget, K6): compiling the study twice per
executor dominates, and the replay of (d) is eager.
"""

import dataclasses

import equinox as eqx
import jax
import numpy as np
import pytest

from spaceodyssey.core import RunStatus
from spaceodyssey.core.enums import AdmissionCode, Audience, EffectKind
from spaceodyssey.core.executors.base import index_world
from spaceodyssey.studies import allocator as al
from spaceodyssey.studies import toy
from tests.core.helpers import worlds
from tests.studies.survey_helpers import run_with_actions, same_histories

N_STEPS = 200
OK = int(AdmissionCode.OK)
P, A = al.PROPOSAL_KIND_INDEX, al.AWARD_KIND_INDEX
FIELDS = ("amount", "window_from", "window_until", "recipient", "program", "remainder")
HUGE = 1000.0
EARLY = 180  # the unreleased proposal reaches the policy before the first query at 200
LAST_QUERY = al.QUERY_TICKS[-1]
# proposal rows are identified by their scripted index (the acquisition index), never
# by value: the observer-only proposal, and the one unreleased through both queries
OBSERVER = next(
    i for i, p in enumerate(al.DEFAULT_PROPOSALS) if p[3] == Audience.OBSERVER
)
PENDING = next(i for i, p in enumerate(al.DEFAULT_PROPOSALS) if p[4] > LAST_QUERY)
# first-principles awards, whole quarters rounded down from a pool of 12
# default, round 0: requests 6 and 3 -> 12 * 6/9 = 8 and 12 * 3/9 = 4
# default, round 1: requests 8 and 8 -> 6 and 6
# early, round 0: requests 6 and 3 + 90 = 93 -> 12 * 6/99 = 0.727 -> 0.5,
#   12 * 93/99 = 11.27 -> 11.25 (nearest gives 0.75 and 11.25, ceiling 0.75 and 11.5)
# early, round 1: requests 8 and 98 -> 12 * 8/106 = 0.906 -> 0.75,
#   12 * 98/106 = 11.09 -> 11
DEFAULT_AMOUNTS = [[8.0, 4.0], [6.0, 6.0]]
EARLY_AMOUNTS = [[0.5, 11.25], [0.75, 11.0]]
# mixed: the early script with the program-0 request written at tick 400 raised from 2
# to 20, so round 1 has requests 26 and 98 -> 12 * 26/124 = 2.52 -> 2.5 and
# 12 * 98/124 = 9.48 -> 9.25. Program 0 holds 0.5 at the first query and 2.5 at the
# second; an observation costs 1, so its first request is refused for funds and its
# second is admitted.
LATE = next(i for i, p in enumerate(al.DEFAULT_PROPOSALS) if p[0] == 400)
MIXED_AMOUNTS = [[0.5, 11.25], [2.5, 9.25]]


def _study(requests=None, releases=None):
    """The allocator study; requests and release ticks replaced by script index."""
    proposals = [list(p) for p in al.DEFAULT_PROPOSALS]
    for i, value in (requests or {}).items():
        proposals[i][2] = value
    for i, tick in (releases or {}).items():
        proposals[i][4] = tick
    return al.allocator_study(tuple(tuple(p) for p in proposals))


def _run(executor, study, committee):
    """A run of `study` on `executor`."""
    truths, roots = worlds(1, study.caps)
    acted = run_with_actions(executor, study, truths, roots, N_STEPS, service=committee)
    result = acted.result
    run = al.AllocatorRun(
        jax.tree_util.tree_map(lambda x: x[0], result.carry),
        index_world(result.trace, 0),
        acted.decisions[0],
        roots[0],
        jax.tree_util.tree_map(lambda x: x[0], truths),
    )
    return _Run(study, committee, result, run)


class _Run:
    """A run of one study, with the host re-execution of it made on first use."""

    def __init__(self, study, committee, result, run):
        self.study, self.service, self.result, self.run = study, committee, result, run
        self.kept = None

    @property
    def ref(self):
        """The reference execution of the run (the host re-run of its study)."""
        if self.kept is None:
            self.kept = al.reference_execution(
                self.study, self.run, type(self.service)()
            )
        return self.kept


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


class _Runs:
    """The runs of one executor, each built on first use and then kept.

    A test pays only for the runs it reads, so a fresh process running one test does
    not build the rest.
    """

    def __init__(self, executor):
        self.executor = executor
        self.kept = {}

    def __getattr__(self, name):
        if name not in self.kept:
            self.kept[name] = self._build(name)
        return self.kept[name]

    def _build(self, name):
        toy_ = al.ToyCommittee
        observer = _study({OBSERVER: HUGE})
        pending = _study({PENDING: HUGE})
        early = _study(releases={PENDING: EARLY})
        mixed = _study({LATE: 20.0}, releases={PENDING: EARLY})
        make = {
            "base": lambda: _run(self.executor, _study(), toy_()),
            "observer": lambda: _run(self.executor, observer, toy_()),
            "unreleased": lambda: _run(self.executor, pending, toy_()),
            "early": lambda: _run(self.executor, early, toy_()),
            "mixed": lambda: _run(self.executor, mixed, toy_()),
            "carry_base": lambda: _carry_run(self.executor, _study()),
            "carry_observer": lambda: _carry_run(self.executor, observer),
            "carry_unreleased": lambda: _carry_run(self.executor, pending),
        }
        return make[name]()


@pytest.fixture(scope="module")
def host_runs():
    return _Runs("host")


@pytest.fixture(scope="module")
def compiled_runs():
    return _Runs("compiled")


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


def _proposal_row(r, index):
    """The log row of the proposal scripted at `index`."""
    log = r.run.carry.log
    hit = np.asarray(log.valid) & (np.asarray(log.kind) == P)
    hit &= np.asarray(log.acquisition_index) == index
    (row,) = np.flatnonzero(hit)
    return int(row)


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


REASON = "two programs per executor, eight runs: about 15 s cold fixture, shared"


@pytest.mark.medium(reason=REASON)
def test_awards_ignore_unreleased_and_observer_only_proposals(runs):
    """(a) Perturbing a proposal the policy never sees leaves the awards identical."""
    base = runs.base
    assert base.study.agenda.levels.tolist() == [3]
    assert base.result.history(0).status == RunStatus.COMPLETE
    reference = _awards(base)
    _positive(
        queries=len(base.service.requests),
        awards=reference.shape[0],
        views=len(base.run.decisions),
    )
    assert [q.tick for q in base.service.requests] == list(al.QUERY_TICKS)
    for name, index in (("observer", OBSERVER), ("unreleased", PENDING)):
        moved = getattr(runs, name)
        log = moved.run.carry.log
        row = _proposal_row(moved, index)
        assert np.asarray(log.payload)[row, 0] == HUGE
        if name == "observer":
            sealed = np.asarray(log.audience)[row] == int(Audience.OBSERVER)
        else:
            sealed = np.asarray(log.released_tick)[row] > LAST_QUERY
        _positive(
            sealed_row=int(sealed),
            requests=len(moved.service.requests),
            allocates=len(_allocates(moved)),
        )
        assert [q.tick for q in moved.service.requests] == list(al.QUERY_TICKS)
        assert np.array_equal(_awards(moved), reference), name
        assert _allocates(moved) == _allocates(base), name
        assert _follow(moved) == [], name
        assert _viewed(moved, HUGE, until=LAST_QUERY) == 0, name
    # the observer-only request is in no policy view; the unreleased one reaches the
    # views only after both queries (so a view could have shown it)
    assert _viewed(runs.observer, HUGE) == 0
    assert _viewed(runs.unreleased, HUGE) > 0


@pytest.mark.medium(reason=REASON)
def test_releasing_a_proposal_moves_the_award_the_expected_way(runs):
    """(b) Releasing the unreleased proposal early raises its program's share."""
    base, early = runs.base, runs.early
    ref, new = _awards(base), _awards(early)
    program = al.DEFAULT_PROPOSALS[PENDING][1]
    _positive(awards=len(ref), moved_awards=len(new), allocates=len(_allocates(early)))
    assert ref[:, 3:5].tolist() == DEFAULT_AMOUNTS
    assert new[:, 3:5].tolist() == EARLY_AMOUNTS
    for award in (*ref, *new):
        assert award[3:5].sum() <= al.POOL
    assert new[0, 3 + program] > ref[0, 3 + program]
    assert new[0, 4 - program] < ref[0, 4 - program]
    assert _follow(early) == []
    assert _admitted(early, EffectKind.ALLOCATE) == [OK] * 4
    granted = np.asarray(early.run.carry.ledger.granted)[2:4, al.PROGRAM_COLUMN]
    assert granted.tolist() == new[:, 3:5].sum(axis=0).tolist()


@pytest.mark.medium(reason=REASON)
def test_a_committee_reading_the_carry_fails_the_perturbation(runs):
    """(c) The carry-reading committee's awards move under each perturbation."""
    base = runs.carry_base
    _positive(
        carry_awards=len(_awards(base)),
        queries=len(base.service.requests),
        toy_awards=len(_awards(runs.base)),
    )
    for name in ("carry_observer", "carry_unreleased"):
        moved = getattr(runs, name)
        assert len(_awards(moved)) == len(_awards(base)), name
        assert not np.array_equal(_awards(moved), _awards(base)), name  # fails (a)
        assert _awards(moved)[0, 3:5].tolist() != _awards(base)[0, 3:5].tolist(), name
    # it reads the sealed proposals even unperturbed, so it disagrees with the toy
    assert not np.array_equal(_awards(base), _awards(runs.base))


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


def _flip_admission(r, kind):
    """Mark the first admitted `kind` request refused in the trace."""
    t = r.trace
    step = int(np.flatnonzero((np.asarray(t.eff_kind) == kind).any(1))[0])
    slot = int(np.flatnonzero(np.asarray(t.eff_kind)[step] == kind)[0])
    code = np.asarray(t.admit_code).copy()
    code[step, slot] = int(AdmissionCode.FORBIDDEN)
    return dataclasses.replace(r, trace=eqx.tree_at(lambda x: x.admit_code, t, code))


def _slots(r, kind):
    """The `(step, slot)` of every requested `kind` effect, in decision order."""
    t = r.trace
    hit = np.asarray(t.eff_kind) == kind
    hit &= np.asarray(t.handled)[:, None]
    return [tuple(map(int, x)) for x in np.argwhere(hit)]


def _set_codes(r, kind, codes):
    """Replace the admission code of the n-th `kind` request for each `n: code`."""
    t = r.trace
    column = np.asarray(t.admit_code).copy()
    slots = _slots(r, kind)
    for n, code in codes.items():
        column[slots[n]] = int(code)
    return dataclasses.replace(r, trace=eqx.tree_at(lambda x: x.admit_code, t, column))


def _set_observe_codes(r, codes):
    return _set_codes(r, EffectKind.OBSERVE, codes)


def _bend_trace(r, field):
    """Add one to a field of the first retained `Allocate` row of the trace."""
    t = r.trace
    step = int(
        np.flatnonzero((np.asarray(t.eff_kind) == EffectKind.ALLOCATE).any(1))[0]
    )
    slot = int(np.flatnonzero(np.asarray(t.eff_kind)[step] == EffectKind.ALLOCATE)[0])
    column = np.asarray(getattr(t, field)).copy()
    column[(step, slot) + ((al.PROGRAM_COLUMN,) if column.ndim == 3 else ())] += 1
    return dataclasses.replace(
        r, trace=eqx.tree_at(lambda x: getattr(x, field), t, column)
    )


def _drop_measurement(r):
    """Invalidate the first measurement record, which an admitted request started."""
    log = r.carry.log
    (first, *_) = np.flatnonzero(
        np.asarray(log.valid) & (np.asarray(log.kind) == toy.RECORD_MEASUREMENT)
    )
    valid = np.asarray(log.valid).copy()
    valid[first] = False
    return dataclasses.replace(
        r, carry=eqx.tree_at(lambda c: c.log.valid, r.carry, valid)
    )


def _repeat_measurement(r):
    """Copy the first measurement record into a free log row."""
    log = r.carry.log
    valid = np.asarray(log.valid)
    (first, *_) = np.flatnonzero(
        valid & (np.asarray(log.kind) == toy.RECORD_MEASUREMENT)
    )
    free = int(np.flatnonzero(~valid)[0])
    rows = valid.shape[0]

    def copy(x):
        x = np.asarray(x)
        if x.ndim and x.shape[0] == rows:
            x = x.copy()
            x[free] = x[first]
        return x

    again = jax.tree_util.tree_map(copy, log)
    return dataclasses.replace(r, carry=eqx.tree_at(lambda c: c.log, r.carry, again))


def _swap_codes(r, kind_a, kind_b):
    """Give the first `kind_a` request the code of the first `kind_b` request."""
    t = r.trace
    code = np.asarray(t.admit_code).copy()
    eff = np.asarray(t.eff_kind)

    def first(kind):
        step = int(np.flatnonzero((eff == kind).any(1))[0])
        return step, int(np.flatnonzero(eff[step] == kind)[0])

    a, b = first(kind_a), first(kind_b)
    code[a] = int(AdmissionCode.FORBIDDEN)
    code[b] = int(AdmissionCode.NO_FUNDS)
    return dataclasses.replace(r, trace=eqx.tree_at(lambda x: x.admit_code, t, code))


def _bend_record(r, kind, column, by=1):
    """Add `by` to a column of the first valid log row of `kind`."""
    log = r.carry.log
    (row, *_) = np.flatnonzero(np.asarray(log.valid) & (np.asarray(log.kind) == kind))
    values = np.asarray(getattr(log, column)).copy()
    values[row] += by
    return dataclasses.replace(
        r, carry=eqx.tree_at(lambda c: getattr(c.log, column), r.carry, values)
    )


def _bend_payload(r, by):
    """Add `by` to the first payload column of the first measurement record."""
    log = r.carry.log
    (row, *_) = np.flatnonzero(
        np.asarray(log.valid) & (np.asarray(log.kind) == toy.RECORD_MEASUREMENT)
    )
    payload = np.asarray(log.payload).copy()
    payload[row, 0] += by
    return dataclasses.replace(
        r, carry=eqx.tree_at(lambda c: c.log.payload, r.carry, payload)
    )


def _bend_granted(r):
    """Add one to the ledger's granted total of the first program."""
    granted = np.asarray(r.carry.ledger.granted).copy()
    granted[2, al.PROGRAM_COLUMN] += 1.0
    return dataclasses.replace(
        r, carry=eqx.tree_at(lambda c: c.ledger.granted, r.carry, granted)
    )


def _bend_proposal(r, index):
    """Change the request of the recorded proposal scripted at `index`."""
    log = r.carry.log
    hit = np.asarray(log.valid) & (np.asarray(log.kind) == P)
    (row,) = np.flatnonzero(hit & (np.asarray(log.acquisition_index) == index))
    payload = np.asarray(log.payload).copy()
    payload[row, 0] += 1.0
    return dataclasses.replace(
        r, carry=eqx.tree_at(lambda c: c.log.payload, r.carry, payload)
    )


# the log columns of each record kind that the replay must compare with the re-execution
RECORD_COLUMNS = {
    "measurement": (
        "channel",
        "status",
        "program",
        "audience",
        "released_tick",
        "acquisition_index",
        "acquisition_id",
    ),
    "award": ("status", "program", "acquisition_index", "audience", "released_tick"),
}


@pytest.mark.medium(reason="replays the store eagerly: about 2 s call, shared fixture")
def test_replay_reproduces_the_run_and_fails_without_a_recorded_award(runs):
    """(d) The recorded run replays exactly; dropping one recorded award fails."""
    base = runs.base
    study, run = base.study, base.run
    ref = base.ref
    report = al.replay_allocation(study, run, reference=ref)
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
        dropped = al.replay_allocation(study, _drop_award(run, k), reference=ref)
        assert not dropped.ok and dropped.n_awards == 2, k
        assert any(m.startswith("query") for m in dropped.failures), k
    # the Allocate effects and the admission outcomes are compared too
    for name, edited in (
        ("allocate removed", _drop_allocate(run)),
        ("allocate amount", _bend_allocate(run)),
        ("allocate admission", _flip_admission(run, EffectKind.ALLOCATE)),
        ("observe admission", _flip_admission(run, EffectKind.OBSERVE)),
        ("query admission", _flip_admission(run, EffectKind.QUERY)),
        ("wait admission", _flip_admission(run, EffectKind.WAIT)),
        ("stop admission", _flip_admission(run, EffectKind.STOP)),
        ("trace amount", _bend_trace(run, "alloc_amount")),
        ("trace window start", _bend_trace(run, "alloc_from")),
        ("trace window end", _bend_trace(run, "alloc_until")),
        ("trace recipient", _bend_trace(run, "alloc_recipient")),
        ("recorded proposal", _bend_proposal(run, 0)),
        ("measurement record dropped", _drop_measurement(run)),
        ("granted total", _bend_granted(run)),
        *(
            (f"{label} {column}", _bend_record(run, kind, column))
            for label, kind in (("measurement", toy.RECORD_MEASUREMENT), ("award", A))
            for column in RECORD_COLUMNS[label]
        ),
        ("measurement payload beyond the go band", _bend_payload(run, 1e-9)),
        ("measurement record repeated", _repeat_measurement(run)),
        (
            "swapped query and wait codes",
            _swap_codes(run, EffectKind.QUERY, EffectKind.WAIT),
        ),
    ):
        assert not al.replay_allocation(study, edited, reference=ref).ok, name
    # a one-ULP float difference is inside the go band: not a failure
    nudged = _bend_payload(run, np.spacing(1.0))
    assert al.replay_allocation(study, nudged, reference=ref).ok
    # without a precomputed reference the replay re-executes the study itself, and the
    # re-execution does not read the run's trace: an altered outcome is still found
    altered = _flip_admission(run, EffectKind.ALLOCATE)
    own = al.reference_execution(study, altered)
    assert np.array_equal(own.trace.admit_code, ref.trace.admit_code)
    assert not al.replay_allocation(study, altered).ok
    # a committee that read the carry cannot reproduce the recorded awards
    assert not al.replay_allocation(study, run, al.CarryCommittee()).ok


@pytest.mark.medium(reason="the early and mixed host runs and their re-executions")
def test_legitimate_refusals_replay_and_each_outcome_is_the_cores(host_runs):
    """Legitimate refusals replay; each outcome is reconciled with its own request."""
    runs = host_runs
    refused, ok = int(AdmissionCode.NO_FUNDS), OK
    early = runs.early
    assert _admitted(early, EffectKind.OBSERVE) == [refused, ok, refused, ok]
    assert al.replay_allocation(early.study, early.run, reference=early.ref).ok
    mixed = runs.mixed
    assert _awards(mixed)[:, 3:5].tolist() == MIXED_AMOUNTS
    codes = _admitted(mixed, EffectKind.OBSERVE)
    assert codes == [refused, ok, ok, ok]
    report = al.replay_allocation(mixed.study, mixed.run, reference=mixed.ref)
    assert report.ok, report.failures
    _positive(refusals=codes.count(refused), admissions=codes.count(ok))
    study, run = mixed.study, mixed.run
    for name, edit in (
        ("swapped outcomes", {0: ok, 2: refused}),
        ("refusal reason", {0: AdmissionCode.FORBIDDEN}),
        ("refused though admitted", {1: refused}),
        ("admitted though refused", {0: ok}),
    ):
        edited = _set_observe_codes(run, edit)
        assert not al.replay_allocation(study, edited, reference=mixed.ref).ok, name


def _edge_studies():
    """Four data variants of the default study that exercise other admission rules."""
    base = al.allocator_study()
    proposals = [list(q) for q in al.DEFAULT_PROPOSALS]
    proposals[4][2] = HUGE  # the request of one program swamps the other's award
    permitted = sum(1 << int(k) for k in EffectKind if k != EffectKind.ALLOCATE)
    return {
        "outside validity": al.allocator_study(tuple(map(tuple, proposals))),
        "transition equal to occupancy": al.allocator_study(
            transition_ticks=base.assets.transition_ticks.at[0].set(100)
        ),
        "asset budget short": al.allocator_study(budget=base.budget.at[0, 0].set(0)),
        "allocate forbidden": al.allocator_study(permitted_effects=(permitted,)),
    }


@pytest.mark.medium(reason="four host runs that share the default program, replayed")
def test_edge_rules_replay_and_their_outcomes_cannot_be_changed():
    """Legitimate refusals and boundary cases replay; changing an outcome fails."""
    refusals = {int(AdmissionCode.NO_FUNDS), int(AdmissionCode.OUTSIDE_VALIDITY)}
    for name, study in _edge_studies().items():
        r = _run("host", study, al.ToyCommittee())
        assert not bool(r.run.carry.violation), name
        report = al.replay_allocation(study, r.run, reference=r.ref)
        assert report.ok, (name, report.failures)
        codes = _admitted(r, EffectKind.OBSERVE)
        _positive(**{name: len(codes)})
        # every outcome is the core's: each refusal made admitted and each admission
        # refused for another reason must be rejected, for Observe and Allocate alike
        for kind in (EffectKind.OBSERVE, EffectKind.ALLOCATE):
            for n, code in enumerate(_admitted(r, kind)):
                other = int(AdmissionCode.FORBIDDEN) if code == OK else OK
                edited = _set_codes(r.run, kind, {n: other})
                assert not al.replay_allocation(study, edited, reference=r.ref).ok, (
                    name,
                    kind,
                    n,
                )
        if name == "allocate forbidden":
            forbidden = [int(AdmissionCode.FORBIDDEN)] * 4
            assert _admitted(r, EffectKind.ALLOCATE) == forbidden
        if name == "outside validity":
            assert int(AdmissionCode.OUTSIDE_VALIDITY) in codes, codes
            swapped = {
                n: AdmissionCode.NO_FUNDS
                for n, c in enumerate(codes)
                if c == AdmissionCode.OUTSIDE_VALIDITY
            }
            edited = _set_observe_codes(r.run, swapped)
            assert not al.replay_allocation(study, edited, reference=r.ref).ok
        if name == "asset budget short":
            assert int(AdmissionCode.NO_FUNDS) in codes, codes
        assert set(codes) - {OK} <= refusals, (name, codes)


@pytest.mark.medium(
    reason="reads both executors' stores: the two builds, about 27 s cold"
)
def test_both_executors_make_the_same_awards(host_runs, compiled_runs):
    """The host and compiled runs agree on histories, requests and awards."""
    for name in ("base", "early", "mixed"):
        host, comp = getattr(host_runs, name), getattr(compiled_runs, name)
        assert same_histories(host.result, comp.result, 1), name
        assert [q.tick for q in host.service.requests] == [
            q.tick for q in comp.service.requests
        ]
        assert np.array_equal(_awards(host), _awards(comp)), name
        assert _allocates(host) == _allocates(comp), name
