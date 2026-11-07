"""Pre-mission tables and their import: identities, release protocol, one kickoff."""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey import core
from spaceodyssey.codec import dump_study, load_study
from spaceodyssey.core import (
    CompiledExecutor,
    HostExecutor,
    histories_equal,
    index_world,
    snapshot_fingerprint,
    snapshot_view,
)
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import (
    PUBLIC,
    Audience,
    Capacity,
    EventKind,
    RunStatus,
    Stage,
)
from spaceodyssey.core.executors.base import run_status
from spaceodyssey.core.keys import fold_key
from spaceodyssey.core.records import KindRegistry, PayloadColumn, RecordKind
from spaceodyssey.studies import toy
from spaceodyssey.studies.premission import (
    PAD,
    PREMISSION_KINDS,
    ImportDemand,
    PreMissionImporter,
    PreMissionTable,
    authored_table,
    split_rows,
)
from tests.codec.test_document import TOY_KINDS, assert_same_pytree
from tests.core.helpers import transition_j, worlds

OBS, POL = int(Audience.OBSERVER), int(Audience.POLICY)
MEAS = toy.RECORD_MEASUREMENT
PRIOR = 3  # the derived kind registered after the toy kinds
REGISTRY = KindRegistry(
    (
        RecordKind("none", ()),
        toy.MEASUREMENT_KIND,
        toy.AUXILIARY_KIND,
        RecordKind("prior", (PayloadColumn("value", PUBLIC),), derived=True),
    )
)
CAPS = REGISTRY.lower(Capacities())


def table(namespace=(0, 0, 0, 0), audience=(3, 3, OBS, 3), order=(0, 1, 2, 3)):
    """Two public raw rows, an observer-only raw row and a public prior.

    Row 2 is the observer-only row; `namespace` pads rows with `PAD`. `order`
    permutes the rows.
    """
    rows = dict(
        subject=[0, 1, 2, 0],
        local_index=[0, 1, 2, 0],
        channel=[60, 60, 60, 61],
        kind=[MEAS, MEAS, MEAS, PRIOR],
        audience=list(audience),
        payload=[[1.0], [2.0], [3.0], [0.5]],
        start_offset=[-5, -5, -5, -10],
        end_offset=[-1, -1, -1, 0],
        namespace=list(namespace),
        revision=[0, 2, 0, 0],
    )
    rows = {k: [v[i] for i in order] for k, v in rows.items()}
    return authored_table(**rows)


def study_of(t):
    return toy.survey_1(CAPS, process=PreMissionImporter(table=t))


def one(caps=CAPS):
    truths, roots = worlds(1, caps)
    return index_world(truths, 0), roots[0]


def written(log, n):
    """(kind, subject, channel, acquisition index, revision, audience) per row."""
    cols = (log.kind, log.subject, log.channel, log.acquisition_index)
    cols += (log.revision, log.audience)
    return [tuple(int(np.asarray(c)[i]) for c in cols) for i in range(n)]


# ---- the table ---------------------------------------------------------------------


def test_the_table_refuses_inconsistent_shapes_and_unknown_audiences():
    """Fails if a table with a short column, a payload of the wrong rank, a pedigree
    of another row count or a row released to nobody (or to the scorer) is built.
    """
    good = table()
    with pytest.raises(ValueError, match="rows"):
        dataclasses.replace(good, channel=good.channel[:3]).__check_init__()
    with pytest.raises(ValueError, match="payload"):
        dataclasses.replace(good, payload=good.payload[:, 0]).__check_init__()
    with pytest.raises(ValueError, match="pedigree"):
        dataclasses.replace(good, external=good.external[:3]).__check_init__()
    for bad in (0, int(Audience.SCORER), 7):
        with pytest.raises(ValueError, match="audience"):
            table(audience=(3, 3, bad, 3))
    # a padding row is never imported, so its audience is not read
    table(namespace=(0, 0, PAD, 0), audience=(3, 3, 0, 3))


def test_import_demand_counts_two_events_and_three_ids_per_written_row():
    """Fails if a public prior is not counted twice, a padding row is counted, or the
    cost is not 2 events, 3 ids and one log row per written row.
    """
    assert list(np.asarray(split_rows(table(), CAPS))) == [False, False, False, True]
    need = ImportDemand.of(table(), CAPS)
    assert (need.rows, need.events, need.ids) == (5, 10, 15)
    assert (need.queue_peak, need.steps) == (6, 11)
    padded = ImportDemand.of(table(namespace=(0, 0, PAD, PAD)), CAPS)
    assert (padded.rows, padded.events, padded.ids) == (2, 4, 6)
    # without a derived kind in the registry nothing is split
    assert ImportDemand.of(table(), Capacities()).rows == 4


# ---- seed and handle, one row at a time ------------------------------------------


def test_seed_queues_one_process_event_per_written_row_and_retargets_the_kickoff():
    """Fails if a row is queued at another tick, stage, group or audience, if the
    public prior is not queued once per audience, if a padding row is queued, if the
    kickoff wait stays in the admit stage, or if the queue differs from the plain
    study's in anything but that and the import rows.
    """
    truth, _ = one()
    c = core.init_carry(study_of(table(namespace=(0, 0, 0, 0))), truth)
    ev = c.ev
    rows = np.flatnonzero(np.asarray(ev.valid & (ev.kind == EventKind.PROCESS)))
    assert len(rows) == 5
    assert set(np.asarray(ev.tick)[rows]) == {0}
    assert set(np.asarray(ev.stage)[rows]) == {int(Stage.RELEASE)}
    pairs = [(int(ev.group[i]), int(ev.audience[i])) for i in rows]
    assert pairs == [(0, 3), (1, 3), (2, OBS), (3, OBS), (3, POL)]
    kick = np.flatnonzero(np.asarray(ev.valid & (ev.kind == EventKind.WAIT)))
    assert len(kick) == 1 and int(ev.stage[kick[0]]) == int(Stage.RELEASE)
    assert int(ev.sid[kick[0]]) == 0 and int(c.next_sid) == 6
    # padding rows queue nothing and take no id
    padded = core.init_carry(study_of(table(namespace=(PAD,) * 4)), truth)
    plain = core.init_carry(toy.survey_1(CAPS), truth)
    moved = dataclasses.replace(
        plain.ev,
        stage=jnp.where(plain.ev.kind == EventKind.WAIT, Stage.RELEASE, plain.ev.stage),
    )
    assert_same_pytree(padded, dataclasses.replace(plain, ev=moved))


def _handled(c, study, group, audience):
    row = dataclasses.replace(
        core.empty_event_row(),
        kind=jnp.asarray(EventKind.PROCESS),
        stage=jnp.asarray(Stage.RELEASE),
        group=jnp.asarray(group),
        audience=jnp.asarray(audience),
        valid=jnp.asarray(True),
    )
    return core.handlers.h_process(c, row, study, None, None)


def test_a_row_is_written_unset_with_its_supplied_index_and_relative_span():
    """Fails if the importer stamps a row itself, writes another acquisition index
    than the supplied local index, takes the revision or span from elsewhere, or
    takes no stable id; and if the release it queues is not at tick 0 for the row.
    """
    study = study_of(table())
    truth, _ = one()
    c = core.init_carry(study, truth)
    before = int(c.next_sid)
    c, _, rec = _handled(c, study, 1, 3)
    assert int(rec.released_tick) == -1 and int(rec.accepted_tick) == -1
    assert (int(rec.subject), int(rec.channel)) == (1, 60)
    assert (int(rec.acquisition_index), int(rec.revision)) == (1, 2)
    assert (int(rec.start_tick), int(rec.end_tick), int(rec.available_tick)) == (
        -5,
        -1,
        0,
    )
    assert float(rec.payload[0]) == 2.0 and int(rec.acquisition_id) == before
    # row 3's local index is 0: the written index is the supplied one, not the row's
    _, _, prior = _handled(c, study, 3, OBS)
    assert int(prior.acquisition_index) == 0 and int(prior.kind) == PRIOR
    assert int(c.next_sid) == before + 2  # the row's id and its release's
    release = np.flatnonzero(np.asarray(c.ev.valid & (c.ev.kind == EventKind.RELEASE)))
    assert len(release) == 1
    assert int(c.ev.tick[release[0]]) == 0
    assert int(c.ev.slot[release[0]]) == 0 and int(c.ev.audience[release[0]]) == 3


def test_an_overflow_of_the_queue_the_log_or_the_ids_is_reported():
    """Fails if a table one row too large for `n_events` (or a full log, or no
    stable id left) is imported silently instead of setting its capacity bit.
    """
    truth, _ = one()
    t = table()
    need = ImportDemand.of(t, CAPS)
    for n_events, bit in ((need.queue_peak, 0), (need.queue_peak - 1, 1)):
        caps = dataclasses.replace(CAPS, n_events=n_events)
        c = core.init_carry(
            toy.survey_1(caps, process=PreMissionImporter(table=t)), truth
        )
        assert int(c.overflow_mask) & Capacity.QUEUE_SLOTS == bit * Capacity.QUEUE_SLOTS
        assert bool(c.overflow) == bool(bit)
    study = study_of(t)
    c = core.init_carry(study, truth)
    full = eqx.tree_at(lambda x: x.n_log, c, jnp.asarray(CAPS.n_log))
    out, _, _ = _handled(full, study, 0, 3)
    assert int(out.overflow_mask) & Capacity.LOG_ROWS and int(out.n_log) == CAPS.n_log
    spent = eqx.tree_at(lambda x: x.next_sid, c, jnp.asarray(CAPS.max_sid))
    out, _, rec = _handled(spent, study, 0, 3)
    assert int(out.overflow_mask) & Capacity.STABLE_IDS and not bool(rec.valid)


def test_the_importer_round_trips_through_the_codec_kinds():
    """Fails if the importer, its table or its source identities have no kind
    reference, or if a reload (inline or with file references) is not exact.
    """
    study = study_of(table())
    kinds = {**TOY_KINDS, **PREMISSION_KINDS}
    text = dump_study(study, kinds).to_json()
    assert_same_pytree(study, load_study(text, kinds))
    assert '"premission.table"' in text and '"premission.source_id"' in text
    assert isinstance(load_study(text, kinds).process.table, PreMissionTable)


def test_identity_is_namespace_producer_subject_and_index_never_revision():
    """Fails if equal source indices of different namespaces, producers or subjects
    are taken as one source, or if two revisions of one datum are two sources.
    """
    from spaceodyssey.core.preflight import preflight

    base = dict(
        subject=[0, 0],
        local_index=[0, 1],
        channel=[60, 60],
        kind=[MEAS, MEAS],
        audience=[3, 3],
        payload=[[1.0], [2.0]],
        source_index=[5, 5],
    )
    for differs in (
        dict(namespace=[0, 1]),
        dict(producer=[60, 61]),
        dict(subject=[0, 1]),
    ):
        t = authored_table(**{**base, **differs})
        assert preflight(study_of(t)) == []
    t = authored_table(**base, revision=[0, 1])
    found = preflight(study_of(t))
    assert [(str(f.code), f.member) for f in found] == [
        ("D006", "table.source.source_index")
    ]


# ---- runs ------------------------------------------------------------------------


@pytest.mark.medium  # one compile per executor of the import study
def test_import_rows_are_stamped_folded_once_and_seen_by_the_one_first_decision():
    """B1 on both executors. Fails if a row is delivered unstamped, folded twice or
    by the wrong audience, if the first decision fires before the last import or more
    than once at tick 0, if the step budget is not exactly the counted demand, or if
    the executors disagree.
    """
    study = study_of(table())
    truth, root = one()
    need = ImportDemand.of(study.process.table, CAPS)
    host = HostExecutor()
    short, _ = host.run(study, truth, root, need.steps - 1)
    assert int(run_status(short)) == RunStatus.TRUNCATED and int(short.n_dec) == 0
    assert int(short.next_sid) == 1 + need.ids
    first, _ = host.run(study, truth, root, need.steps)
    assert int(first.n_dec) == 1 and int(first.n_log) == need.rows
    log, n = first.log, need.rows
    aud = np.asarray(log.audience)[:n]
    accepted, released = (
        np.asarray(log.accepted_tick)[:n],
        np.asarray(log.released_tick)[:n],
    )
    assert list(aud) == [3, 3, OBS, OBS, POL]
    assert list(accepted) == [0 if a & OBS else -1 for a in aud]
    assert list(released) == [0 if a & POL else -1 for a in aud]
    # raw rows fold once per audience that reads them; the prior rows never fold
    assert list(np.asarray(first.belief.n)) == [1, 1, 1, 0]
    assert list(np.asarray(first.pbelief.n)) == [1, 1, 0, 0]
    assert list(np.asarray(first.belief.mean)[:3]) == [1.0, 2.0, 3.0]
    assert not bool(first.dup_fold)
    assert list(np.asarray(first.release_version)) == [4, 3]

    run, trace = host.run(study, truth, root, 60)
    truths, roots = worlds(1, CAPS)
    compiled = CompiledExecutor().run(study, truths, roots, 60)
    assert int(run_status(run)) == RunStatus.COMPLETE
    assert bool(np.asarray(compiled.complete)[0])
    report = histories_equal(
        core.canonical_history(run, trace, study.observer.layout(CAPS)),
        compiled.history(0),
    )
    assert report.ok, report
    for carry, tr in ((run, trace), (index_world(compiled.carry, 0), None)):
        tr = index_world(compiled.trace, 0) if tr is None else tr
        wakes = np.flatnonzero(np.asarray(tr.wake))
        at_zero = [i for i in wakes if int(tr.tick[i]) == 0]
        assert len(at_zero) == 1
        assert int(tr.n_log[at_zero[0]]) == need.rows
        assert not bool(carry.dup_fold)


def _decisions_at_zero(study, carry, truth, root):
    """Step to the first event after tick 0: (wakes at tick 0, carry before it)."""
    wakes = 0
    for i in range(40):
        nxt, tr = transition_j(carry, jnp.asarray(i), root, study, truth)
        if int(tr.tick) > 0:
            return wakes, carry
        wakes += int(tr.wake)
        carry = nxt
    raise AssertionError("the run never left tick 0")


@pytest.mark.medium  # compiles the shared transition for two studies
def test_one_initial_decision_with_and_without_a_public_zero_information_import():
    """Fails if a public import of a row nobody folds adds a decision at tick 0, or
    changes the stateful plan policy's state, the decision count or the within-tick
    ordinal after tick 0. Control: the same import with the kickoff left in the admit
    stage makes a second initial decision.
    """
    truth, root = one()
    plain = toy.survey_1(CAPS)
    zero = authored_table(
        subject=[0],
        local_index=[0],
        channel=[61],
        kind=[PRIOR],
        audience=[3],
        payload=[[0.0]],
    )
    imported = study_of(zero)
    base_wakes, base = _decisions_at_zero(
        plain, core.init_carry(plain, truth), truth, root
    )
    c0 = core.init_carry(imported, truth)
    wakes, after = _decisions_at_zero(imported, c0, truth, root)
    assert base_wakes == wakes == 1
    assert int(after.n_log) == 2 and list(np.asarray(after.release_version)) == [1, 1]
    for name in ("pstate", "n_dec", "dec_tick", "dec_ordinal", "belief", "pbelief"):
        assert_same_pytree(getattr(base, name), getattr(after, name))
    # control: the kickoff left where `init_carry` put it wakes a second decision
    kick = (c0.ev.kind == EventKind.WAIT) & c0.ev.valid
    unmoved = eqx.tree_at(
        lambda c: c.ev.stage, c0, jnp.where(kick, Stage.ADMIT, c0.ev.stage)
    )
    wakes, after = _decisions_at_zero(imported, unmoved, truth, root)
    assert wakes == 2 and int(after.dec_ordinal) == 2
    assert not np.array_equal(np.asarray(after.pstate), np.asarray(base.pstate))


def _policy_rows(carry, study):
    view = snapshot_view(carry, Audience.POLICY, study.caps)
    n = int(view.n_rows)
    return written(view.log, n), view


@pytest.mark.medium  # one compiled program, three tables of one shape
def test_an_observer_only_row_moves_no_public_identity_and_order_moves_none():
    """Fails if inserting an observer-only source changes a public row's identity or
    fold key, a policy snapshot fingerprint, a decision or the final policy view; or
    if permuting the table changes the identity of any written row.
    """
    truths, roots = worlds(1, CAPS)
    run = CompiledExecutor().run
    hidden = study_of(table())
    shown = study_of(table(namespace=(0, 0, PAD, 0)))
    permuted = study_of(table(order=(3, 2, 0, 1)))
    a, b, p = (run(s, truths, roots, 60) for s in (hidden, shown, permuted))
    ca, cb, cp = (index_world(r.carry, 0) for r in (a, b, p))
    ta, tb = index_world(a.trace, 0), index_world(b.trace, 0)
    wa, wb = np.asarray(ta.wake), np.asarray(tb.wake)
    assert wa.sum() == wb.sum() > 1
    assert list(np.asarray(ta.snapshot_fp)[wa]) == list(np.asarray(tb.snapshot_fp)[wb])
    assert np.array_equal(np.asarray(ta.eff_kind)[wa], np.asarray(tb.eff_kind)[wb])
    rows_a, view_a = _policy_rows(ca, hidden)
    rows_b, view_b = _policy_rows(cb, shown)
    assert rows_a == rows_b
    layout = hidden.observer.layout(CAPS)
    assert int(snapshot_fingerprint(view_a, layout)) == int(
        snapshot_fingerprint(view_b, layout)
    )
    # the fold keys of the public import rows (slots 0 and 1 in both logs)
    for slot in (0, 1):
        ra = core.handlers._log_rec(ca.log, slot)
        rb = core.handlers._log_rec(cb.log, slot)
        assert np.array_equal(
            np.asarray(fold_key(roots[0], ra, hidden)),
            np.asarray(fold_key(roots[0], rb, shown)),
        )
    # the observer-only row is there, and only for the observer
    assert int(ca.belief.n[2]) - int(cb.belief.n[2]) == 1
    assert_same_pytree(ca.pbelief, cb.pbelief)
    need = ImportDemand.of(hidden.process.table, CAPS).rows
    assert sorted(written(ca.log, need)) == sorted(written(cp.log, need))


def test_only_in_table_pedigree_entries_must_name_a_row_of_the_table():
    """Fails if an external pedigree entry (provenance only) is checked against the
    table, if an in-table entry naming a present row is refused, or if an in-table
    entry naming a padding row is accepted.
    """
    from spaceodyssey.core.preflight import preflight

    def with_pedigree(entry, namespace=(0, 0)):
        return authored_table(
            subject=[0, 0],
            local_index=[0, 1],
            channel=[60, 61],
            kind=[MEAS, PRIOR],
            audience=[3, 3],
            payload=[[1.0], [0.5]],
            namespace=list(namespace),
            pedigree=[(), (entry,)],
        )

    absent = (3, 70, 0, 9)
    assert preflight(study_of(with_pedigree((*absent, True)))) == []
    assert preflight(study_of(with_pedigree((0, 60, 0, 0, False)))) == []
    found = preflight(study_of(with_pedigree((*absent, False))))
    assert [(str(f.code), f.entry) for f in found] == [("D005", (1, 0))]
    padded = preflight(study_of(with_pedigree((0, 60, 0, 0, False), (PAD, 0))))
    assert [str(f.code) for f in padded] == ["D005"]
