"""Programs per activity: a Stop closes one program, and every charge has an owner.

With `n_programs > 1` an effect names a program. The program is copied onto the queued
Done row and the record, the ledger charges the row `n_assets + program`, a program Stop
sets that program's closed flag (its lanes stay open) and a drain or an abort acts only
on live occupancies of that program. With one program none of this exists: the program
columns are absent and the compiled scan is the one it was. Each check below has a
control, a mutant of the rule it guards, or both.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import (
    Audience,
    HostExecutor,
    handlers,
    init_carry,
    snapshot_fingerprint,
    snapshot_view,
)
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.carry import _tree_set
from spaceodyssey.core.enums import (
    I64,
    AdmissionCode,
    Disposition,
    EffectKind,
    RecordStatus,
    Scope,
)
from spaceodyssey.core.events import empty_event_row, empty_events
from spaceodyssey.core.ledger import _resource_rows
from spaceodyssey.core.preflight import FindingCode, preflight
from spaceodyssey.core.records import empty_log, masked_record, record_none
from spaceodyssey.core.snapshot import effects_none
from spaceodyssey.studies import toy
from tests.core.helpers import one_world
from tests.core.program_helpers import (
    ScriptedPrograms,
    cancel,
    observe,
    stop,
    two_program_study,
    wait,
)

N_STEPS = 60
CLOSE, DRAIN, ABORT = Disposition.CLOSE, Disposition.DRAIN, Disposition.ABORT
PROGRAM = Scope.PROGRAM
FAR = 120  # past the first occupancy on channel 10, which settles at tick 110

# Programs 0 and 1 share lane 0; the first closes while its occupancy is done.
STOP_ONE = (
    observe(0, 0, 10, 0),
    wait(0, FAR),
    stop(1, PROGRAM, CLOSE, 0),
    wait(1, 60),
    observe(2, 1, 10, 0),
    observe(2, 1, 10, 1),
)
# Both lanes busy at tick 20, then program 0 is stopped with `disp`.
TWO_LIVE = (observe(0, 0, 10, 0), observe(0, 1, 11, 1), wait(0, 20))
# One occupancy of program 1 on lane 0, cancelled at tick 20.
CANCEL_OWNED = (observe(0, 0, 10, 1), wait(0, 20), cancel(1, 0))
# Two acquisitions on the lane, the programs swapped between two scripts.
RELABEL = (observe(0, 0, 10, 0), wait(0, FAR), observe(2, 1, 10, 1))
SWAPPED = (observe(0, 0, 10, 1), wait(0, FAR), observe(2, 1, 10, 0))


def _run(*rows, n_programs=2):
    study = two_program_study(*rows, n_programs=n_programs)
    truth, root = one_world(caps=study.caps)
    return study, HostExecutor().run(study, truth, root, N_STEPS)


def _live(log, n):
    return {
        k: np.asarray(getattr(log, k))[:n] for k in ("program", "status", "subject")
    }


def _rows(run):
    """Per-record columns of the log, in write order."""
    return _live(run.carry.log, int(run.carry.n_log))


def _codes(run):
    """Admission codes of each decision, in decision order."""
    t = run.trace
    keep = np.asarray(t.handled)[:, None] & (np.asarray(t.eff_kind) != EffectKind.NONE)
    return np.asarray(t.admit_code)[keep]


@pytest.fixture(scope="module")
def stop_one():
    return _run(*STOP_ONE)


@pytest.fixture(scope="module")
def abort_one():
    return _run(*TWO_LIVE, stop(1, PROGRAM, ABORT, 0))


@pytest.fixture(scope="module")
def drain_one():
    return _run(*TWO_LIVE, stop(1, PROGRAM, DRAIN, 0))


@pytest.fixture(scope="module")
def cancelled():
    return _run(*CANCEL_OWNED)


def _row(study, program):
    return study.caps.n_assets + program


# ---- the ledger row of a program ----------------------------------------------------


def test_the_ledger_charges_the_row_of_the_named_program():
    """Fails if the program row is not n_assets + program (one program: the one row)."""
    many = two_program_study(n_programs=3)
    rows = [int(r) for r in _resource_rows(many, 1, 2)]
    assert rows == [int(many.assets.lane_asset[1]), many.caps.n_assets + 2, 5]
    one = toy.survey_1()
    assert [int(r) for r in _resource_rows(one, 1, 0)][1] == one.caps.n_assets
    assert [int(r) for r in _resource_rows(one, 1)][1] == one.caps.n_assets


def test_control_a_program_that_is_ignored_charges_the_first_row():
    """Control: the row differs by program, so a rule that drops it would be seen."""
    many = two_program_study(n_programs=3)
    assert int(_resource_rows(many, 0, 1)[1]) != int(_resource_rows(many, 0, 0)[1])


# ---- one program: nothing exists ----------------------------------------------------


def test_one_program_holds_no_program_leaf():
    """Fails if a study with one program carries a program column or flag."""
    study = toy.survey_1()
    truth, _ = one_world()
    carry = eqx.filter_jit(init_carry)(study, truth)
    assert carry.ev.program is None and carry.request.program is None
    assert carry.refresh.program is None and carry.log.program is None
    assert carry.ledger.program_closed is None
    assert effects_none(study.caps).program is None
    assert record_none(study.caps).program is None
    assert len(jax.tree_util.tree_leaves(carry)) == N_DEFAULT_CARRY_LEAVES


def test_control_two_programs_add_exactly_the_program_leaves():
    """Control: more than one program adds one leaf to each table and the ledger."""
    caps = Capacities(n_programs=2)
    assert empty_events(caps).program.shape == (caps.n_events,)
    assert empty_event_row(caps).program.shape == ()
    assert empty_log(caps).program.shape == (caps.n_log,)
    assert record_none(caps).program.shape == ()
    assert effects_none(caps).program.shape == (caps.n_effects,)
    study = two_program_study()
    truth, _ = one_world(caps=study.caps)
    carry = eqx.filter_jit(init_carry)(study, truth)
    assert carry.ledger.program_closed.shape == (2,)
    # event table, request row, refresh row, log and the closed flags
    assert len(jax.tree_util.tree_leaves(carry)) == N_DEFAULT_CARRY_LEAVES + 5


# The leaves of a default carry, as on `main` before programs (the scan's StableHLO is
# the same text; the change record holds the digest comparison).
N_DEFAULT_CARRY_LEAVES = 134


# ---- P001 ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_programs", [1, 2, 3])
def test_a_program_outside_the_held_range_is_p001(n_programs):
    """Fails if P001 misses the first id past the range or flags the last one in it."""
    inside, outside = n_programs - 1, n_programs
    ok = two_program_study(observe(0, 0, 10, inside), n_programs=n_programs)
    bad = two_program_study(observe(0, 0, 10, outside), n_programs=n_programs)
    assert FindingCode.P001 not in {f.code for f in preflight(ok)}
    found = [f for f in preflight(bad) if f.code == FindingCode.P001]
    assert [(f.subject, f.index, f.member) for f in found] == [
        ("policy", 0, "programs")
    ]


def test_a_program_stop_names_its_program_for_p001():
    """Fails if the program a Stop names is not checked like an Observe's."""
    study = two_program_study(stop(0, PROGRAM, CLOSE, 5), n_programs=2)
    assert FindingCode.P001 in {f.code for f in preflight(study)}


# ---- a program Stop closes only its program -----------------------------------------


@pytest.mark.medium  # A cold host run compiles the executor, over the 5 s unit budget.
def test_a_program_stop_closes_only_its_program_and_the_lane_stays_open(stop_one):
    """Fails if the Stop closes a lane, or a program other than the one it names."""
    _study, run = stop_one
    c = run.carry
    assert list(np.asarray(c.ledger.program_closed)) == [True, False]
    assert not np.asarray(c.ledger.lane_closed).any()
    codes = _codes(run)
    # the closed program is refused on a free lane, the other is admitted on that lane
    assert (AdmissionCode.LANE_CLOSED in codes) and not (
        codes == AdmissionCode.LANE_BUSY
    ).any()
    rows = _rows(run)
    assert list(rows["program"]) == [0, 1] and list(rows["subject"]) == [0, 1]


# ---- drain and abort act only on the program's live occupancies ---------------------


@pytest.mark.medium  # A cold host run compiles the executor, over the 5 s unit budget.
def test_an_abort_cancels_only_the_live_occupancy_of_its_program(abort_one):
    """Fails if the abort touches the other program's occupancy or refunds wrongly."""
    study, run = abort_one
    c = run.carry
    rows = _rows(run)
    by = {int(p): int(s) for p, s in zip(rows["program"], rows["status"], strict=True)}
    assert by == {0: int(RecordStatus.PARTIAL), 1: int(RecordStatus.COMPLETED)}
    balance = np.asarray(c.ledger.balance)
    # program 0 spent a tenth of a unit (ticks 10 to 20 of 100) and got 0.9 back
    assert balance[_row(study, 0), 1] == pytest.approx(99.9)
    assert balance[_row(study, 1), 1] == pytest.approx(99.0)
    assert list(np.asarray(c.ledger.lane_gen)) == [2, 1, 0]
    assert list(np.asarray(c.ledger.program_closed)) == [True, False]


@pytest.mark.medium  # A cold host run compiles the executor, over the 5 s unit budget.
def test_a_drain_leaves_the_programs_occupancy_to_finish_and_refunds_nothing(drain_one):
    """Fails if a program drain cancels, refunds, or closes the other program."""
    study, run = drain_one
    c = run.carry
    rows = _rows(run)
    assert set(rows["status"]) == {int(RecordStatus.COMPLETED)}
    assert list(c.ledger.program_closed) == [True, False]
    assert np.asarray(c.ledger.balance)[_row(study, 0), 1] == pytest.approx(99.0)
    assert not np.asarray(c.ledger.refunded).any()


# ---- every charge has an owner ------------------------------------------------------


@pytest.mark.medium  # A cold host run compiles the executor, over the 5 s unit budget.
def test_completion_and_cancel_charge_the_row_of_the_admitted_program(
    stop_one, cancelled
):
    """Fails if a debit, completion or refund lands on a row other than the owner's."""
    study, run = stop_one
    balance = np.asarray(run.carry.ledger.balance)
    # each program paid for its own acquisition and nothing else
    assert balance[_row(study, 0), 1] == pytest.approx(99.0)
    assert balance[_row(study, 1), 1] == pytest.approx(99.0)
    study, run = cancelled
    balance = np.asarray(run.carry.ledger.balance)
    assert _rows(run)["program"][0] == 1
    assert balance[_row(study, 0)].tolist() == [100.0, 100.0, 100.0]
    assert balance[_row(study, 1), 1] == pytest.approx(99.9)
    refunded = np.asarray(run.carry.ledger.refunded)
    assert refunded[_row(study, 1), 1] == pytest.approx(0.9)
    assert not refunded[_row(study, 0)].any()


# ---- who sees a program -------------------------------------------------------------


def _fingerprints(study, run):
    layout = study.observer.layout(study.caps)
    out = {}
    for audience in (Audience.POLICY, Audience.OBSERVER):
        view = snapshot_view(run.carry, audience, study.caps)
        out[audience] = (view, int(snapshot_fingerprint(view, layout)))
    return out


@pytest.mark.medium  # A cold host run compiles the executor, over the 5 s unit budget.
def test_the_policy_sees_program_ids_and_the_observer_sees_none(stop_one):
    """Fails if the observer view shows a program id, or the policy view hides one."""
    study, run = stop_one
    views = _fingerprints(study, run)
    policy, observer = views[Audience.POLICY][0], views[Audience.OBSERVER][0]
    shown = np.asarray(policy.log.program)[: int(policy.n_rows)]
    assert list(shown) == [0, 1]
    assert not np.asarray(observer.log.program).any()
    full = jax.tree_util.tree_map(lambda x: x[0], run.carry.log)
    row = dataclasses.replace(record_none(study.caps), program=jnp.asarray(1, I64))
    assert int(masked_record(row, study.caps, Audience.OBSERVER).program) == 0
    assert int(masked_record(row, study.caps, Audience.POLICY).program) == 1
    assert full is not None


@pytest.mark.medium  # Two runs share one compiled program; the second is warm.
def test_relabelling_programs_moves_the_policy_fingerprint_and_not_the_observer_s():
    """Fails if a program label reaches the observer's identity, or not the policy's."""
    a_study, a = _run(*RELABEL)
    b_study, b = _run(*SWAPPED)
    fa, fb = _fingerprints(a_study, a), _fingerprints(b_study, b)
    assert fa[Audience.POLICY][1] != fb[Audience.POLICY][1]
    assert fa[Audience.OBSERVER][1] == fb[Audience.OBSERVER][1]


@pytest.mark.medium  # Two runs share one compiled program; the second is warm.
def test_control_an_observer_view_that_shows_the_label_moves_its_fingerprint():
    """Control: a leaked label changes the observer identity, so the check bites."""
    a_study, a = _run(*RELABEL)
    b_study, b = _run(*SWAPPED)
    layout = a_study.observer.layout(a_study.caps)
    leaked = []
    for study, run in ((a_study, a), (b_study, b)):
        shown = snapshot_view(run.carry, Audience.POLICY, study.caps).log.program
        view = snapshot_view(run.carry, Audience.OBSERVER, study.caps)
        view = eqx.tree_at(lambda v: v.log.program, view, shown)
        leaked.append(int(snapshot_fingerprint(view, layout)))
    assert leaked[0] != leaked[1]


def test_a_scripted_policy_is_a_policy():
    """Fails if the helper policy stops being a Declaration-bearing toy policy."""
    policy = ScriptedPrograms(script=(observe(0, 0, 10, 1),))
    assert policy.declare().programs == (1,)


# ---- mutants (last: each clears the compile caches) ---------------------------------


@pytest.mark.medium  # Each mutant is a cold compile of the host executor.
def test_mutant_a_stop_that_closes_every_lane_is_caught(monkeypatch):
    """Mutant: with programs, a program Stop closes all lanes. The check must fail."""
    real = handlers.h_stop

    def closes_all(c, ev, study, truth, root):
        out, emit, rec = real(c, ev, study, truth, root)
        closed = jnp.where(ev.scope == Scope.PROGRAM, True, out.ledger.lane_closed)
        return _tree_set(out, **{"ledger.lane_closed": closed}), emit, rec

    table = tuple(closes_all if h is real else h for h in handlers.HANDLERS)
    monkeypatch.setattr(handlers, "HANDLERS", table)
    jax.clear_caches()  # the compiled step closed over the real handler table
    try:
        _, run = _run(*STOP_ONE)
        assert np.asarray(run.carry.ledger.lane_closed).all()
        assert list(_rows(run)["program"]) == [0]  # program 1 was refused on its lane
    finally:
        jax.clear_caches()


@pytest.mark.medium  # Each mutant is a cold compile of the host executor.
def test_mutant_a_stop_that_never_sets_the_flag_admits_the_closed_program(monkeypatch):
    """Mutant: the flag is not set, so the closed program is admitted again."""
    real = handlers.h_stop

    def forgets(c, ev, study, truth, root):
        out, emit, rec = real(c, ev, study, truth, root)
        flag = c.ledger.program_closed
        return _tree_set(out, **{"ledger.program_closed": flag}), emit, rec

    table = tuple(forgets if h is real else h for h in handlers.HANDLERS)
    monkeypatch.setattr(handlers, "HANDLERS", table)
    jax.clear_caches()
    try:
        _, run = _run(*STOP_ONE)
        assert not np.asarray(run.carry.ledger.program_closed).any()
        # program 0 is admitted again on its free lane, so program 1 finds it busy
        assert list(_rows(run)["program"]) == [0, 0]
    finally:
        jax.clear_caches()


@pytest.mark.medium  # Each mutant is a cold compile of the host executor.
def test_mutant_an_abort_that_ignores_the_program_cancels_both(monkeypatch):
    """Mutant: the abort drops its program filter, so it cancels every live lane."""
    real = handlers._cancel_lane

    def any_program(*args, **kwargs):
        kwargs.pop("program", None)
        kwargs.pop("by_program", None)
        return real(*args, **kwargs)

    monkeypatch.setattr(handlers, "_cancel_lane", any_program)
    jax.clear_caches()
    try:
        _, run = _run(*TWO_LIVE, stop(1, PROGRAM, ABORT, 0))
        assert set(_rows(run)["status"]) == {int(RecordStatus.PARTIAL)}
    finally:
        jax.clear_caches()


@pytest.mark.medium  # Each mutant is a cold compile of the host executor.
def test_mutant_a_refund_to_the_first_program_row_is_caught(monkeypatch):
    """Mutant: the cancel credits program 0 whoever owns the occupancy."""
    real = handlers._resource_rows

    def first_row(study, lane, program=0):
        return real(study, lane, 0)

    monkeypatch.setattr(handlers, "_resource_rows", first_row)
    jax.clear_caches()
    try:
        study, run = _run(*CANCEL_OWNED)
        balance = np.asarray(run.carry.ledger.balance)
        assert balance[_row(study, 1), 1] == pytest.approx(99.0)  # never credited
        assert balance[_row(study, 0), 1] > 100.0  # credited to the wrong owner
    finally:
        jax.clear_caches()
