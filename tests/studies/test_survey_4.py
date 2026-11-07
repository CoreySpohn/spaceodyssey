"""Survey 4, RV precursor: exported upstream, imported downstream, lineage verified.

The merge-time guard is a unit test with no executor compile: a small synthetic
upstream log (with a correction) is exported, the table is stored as a file-backed
document and its lineage verified, the import's tick-0 events are handled eagerly,
and the first decision is taken on the policy's snapshot. The full survey runs are
slow (the Slice 2b quick-tier budget, K6): one real upstream run on the compiled
executor, then the survey with, without and with a zero-information precursor on
both executors, all of one shape (one compiled program per executor), and the epoch
relation, which wraps the process.
"""

import dataclasses
import functools
import hashlib
import json

import jax.random as jr
import numpy as np
import pytest

from spaceodyssey.codec import dump_study, load_study
from spaceodyssey.codec.preflight import resolve_pointer
from spaceodyssey.core import (
    CompiledExecutor,
    HostExecutor,
    battery,
    init_carry,
    snapshot_view,
)
from spaceodyssey.core.enums import (
    PUBLIC,
    Audience,
    EffectKind,
    EventKind,
    RecordStatus,
)
from spaceodyssey.core.events import select_next
from spaceodyssey.core.executors.base import index_world
from spaceodyssey.core.handlers import h_process, h_release, h_wait
from spaceodyssey.core.preflight import preflight
from spaceodyssey.core.preflight.checks import PreflightContext
from spaceodyssey.core.records import empty_log
from spaceodyssey.studies import toy
from spaceodyssey.studies.precursor import verify_lineage
from spaceodyssey.studies.premission import PREMISSION_KINDS
from spaceodyssey.studies.survey_4 import (
    N_OBSERVATIONS,
    PRECURSOR_CHANNEL,
    SURVEY_4_KINDS,
    TARGETS,
    export_precursor,
    survey_4,
    survey_4_upstream,
    without_precursor,
    zero_information,
)
from tests.codec.test_document import TOY_KINDS
from tests.core.helpers import worlds
from tests.studies.survey_helpers import run_on, same_histories

OBS, POL = int(Audience.OBSERVER), int(Audience.POLICY)
MEAS = toy.RECORD_MEASUREMENT
KINDS = {**TOY_KINDS, **PREMISSION_KINDS, **SURVEY_4_KINDS}
UPSTREAM = dump_study(survey_4_upstream(), TOY_KINDS)
OTHER = dump_study(survey_4_upstream(horizon_tick=900), TOY_KINDS)
DECLARED = "/study/process/table/upstream_hash"
PAYLOAD = "/study/process/table/payload/file"
HORIZON = int(survey_4_upstream().horizon_tick)
N_STEPS = 40
N_WORLDS = 2
EXECUTORS = ("compiled", "host")


# ---- the synthetic upstream log -----------------------------------------------------


def _row(subject, index, revision, tick, audience=PUBLIC):
    """An upstream measurement on channel 10, released at `tick` to `audience`."""
    return {
        "kind": MEAS,
        "channel": 10,
        "status": int(RecordStatus.COMPLETED),
        "valid": True,
        "subject": subject,
        "acquisition_index": index,
        "revision": revision,
        "audience": audience,
        "accepted_tick": tick if audience & OBS else -1,
        "released_tick": tick if audience & POL else -1,
        "start_tick": tick - 20,
        "end_tick": tick - 5,
        "payload": [10.0 * index + revision + 1.0, 0.5, 1.0, 0.0],
    }


def _log(*rows):
    base = empty_log(survey_4_upstream().caps)
    cols = {f.name: np.array(getattr(base, f.name)) for f in dataclasses.fields(base)}
    for i, row in enumerate(rows):
        for name, value in row.items():
            cols[name][i] = value
    return dataclasses.replace(base, **cols)


# planet 1, the exported one: datum 0 corrected (revision 1 at 300), datum 3 once,
# datum 2 released only after the upstream horizon; planet 0 is not exported
LOG = _log(
    _row(1, 0, 0, 200),
    _row(1, 0, 1, 300),
    _row(1, 3, 0, 450),
    _row(1, 2, 0, HORIZON + 100),
    _row(0, 0, 0, 200),
)


def _imported(study):
    """The carry after the import's tick-0 events, handled one by one, eagerly."""
    handlers = {
        int(EventKind.WAIT): h_wait,
        int(EventKind.PROCESS): h_process,
        int(EventKind.RELEASE): h_release,
    }
    truth = toy.draw_truth(jr.PRNGKey(0), study.caps)
    c = init_carry(study, truth)
    while True:
        nxt, ev, has = select_next(c, study.caps)
        if not bool(has) or int(ev.tick) > 0 or int(ev.kind) not in handlers:
            return c
        c = handlers[int(ev.kind)](nxt, ev, study, truth, jr.PRNGKey(0))[0]


def _codes(found):
    return [(str(f.code), str(f.severity), f.pointer) for f in found]


def test_a_precursor_exports_imports_and_verifies_its_lineage(tmp_path):
    """B7 row 4 (d), (e), (f), the import half of (b), and its first decision. Fails
    if a corrected datum is exported twice or at a revision other than the latest
    released by the end of the upstream run (or a datum released after it is
    exported); if the lineage report does not name the declared hash and each
    file's sha256, or `verify_lineage` does not pass with the upstream document and
    fail with `D004` at the declared hash with another; if a tampered table file is
    not refused at its pointer; if an imported row differs from its table row; or
    if the policy's first decision does not read the precursor (it observes the
    other target with the precursor, the precursor's target without it).
    """
    export = export_precursor(LOG, UPSTREAM.content_hash)
    table = export.table
    # (f) one row per datum, the correction at revision 1
    assert np.asarray(table.source.source_index).tolist() == [0, 3]
    assert np.asarray(table.source.revision).tolist() == [1, 0]
    reasons = {(r.source_index, r.revision): r.reason for r in export.revisions}
    assert reasons == {
        (0, 0): "superseded",
        (0, 1): "exported",
        (3, 0): "exported",
        (2, 0): "not released",
    }
    study = survey_4(table)
    context = PreflightContext(upstream_document=UPSTREAM)
    assert preflight(study, context=context) == []

    # (d) the lineage of the stored document
    files = tmp_path / "files"
    text = dump_study(study, KINDS, files=files, max_inline=2).to_json()
    assert load_study(text, KINDS, files=files).process.table.n_rows == 2
    report = verify_lineage(text, UPSTREAM.to_json(), files=files)
    assert report.status == "verified" and report.findings == ()
    (lineage,) = report.tables
    assert lineage.upstream_hash == UPSTREAM.content_hash
    named = {ref.pointer: ref for ref in lineage.files}
    assert PAYLOAD in named
    for ref in named.values():
        raw = (files / ref.name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == ref.sha256
    wrong = verify_lineage(text, OTHER, files=files)
    assert _codes(wrong.findings) == [("D004", "error", DECLARED)]
    # (e) a tampered table file is refused at its pointer
    path = files / resolve_pointer(json.loads(text), PAYLOAD)["name"]
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    tampered = verify_lineage(text, UPSTREAM, files=files)
    assert _codes(tampered.findings) == [("L001", "error", f"{PAYLOAD}/sha256")]

    # (b) the imported rows are the table's, folded once by both audiences
    c = _imported(study)
    log = c.log
    rows = np.flatnonzero(
        np.asarray(log.valid) & (np.asarray(log.channel) == PRECURSOR_CHANNEL)
    )
    assert rows.size == table.n_rows
    columns = (
        (log.subject, table.source.subject),
        (log.acquisition_index, table.local_index),
        (log.revision, table.source.revision),
        (log.start_tick, table.start_offset),
        (log.end_tick, table.end_offset),
        (log.audience, table.audience),
        (log.released_tick, np.zeros(table.n_rows, int)),
    )
    for got, want in columns:
        assert np.asarray(got)[rows].tolist() == np.asarray(want).tolist()
    width = np.shape(table.payload)[1]
    assert np.array_equal(np.asarray(log.payload)[rows, :width], table.payload)
    target = int(table.source.subject[0])
    assert int(c.belief.n[target]) == int(c.pbelief.n[target]) == table.n_rows

    # the first decision reads the precursor through the policy's snapshot
    def first_subject(s, carry):
        view = snapshot_view(carry, POL, s.caps)
        policy = s.agenda.policies[0]
        eff, _, _ = policy.decide(carry.pstate, view, s, jr.PRNGKey(0))
        assert int(eff.kind[0]) == int(EffectKind.OBSERVE)
        return int(eff.subject[0])

    bare = survey_4(without_precursor(table))
    assert first_subject(study, c) == 1 - target
    assert first_subject(bare, _imported(bare)) == target


# ---- the full survey runs (slow) ---------------------------------------------------


@functools.cache
def _precursor(audience=PUBLIC):
    """The precursor exported from one real upstream run (world 0, compiled)."""
    upstream = survey_4_upstream()
    truths, roots = worlds(1, upstream.caps)
    run = CompiledExecutor().run_worlds(upstream, truths, roots, N_STEPS)
    return export_precursor(
        index_world(run.carry, 0), UPSTREAM.content_hash, audience=audience
    ).table


@functools.cache
def _run(executor, variant):
    table = _precursor(OBS if variant == "observer" else PUBLIC)
    table = {
        "with": table,
        "observer": table,
        "without": without_precursor(table),
        "zero": zero_information(table),
    }[variant]
    study = survey_4(table)
    truths, roots = worlds(N_WORLDS, study.caps)
    run = run_on(executor, study, truths, roots, N_STEPS)
    assert np.asarray(run.complete).all(), variant
    return study, run


def projection(run, w, until=None):
    """Decisions (tick, effect kinds, admission codes) and the non-imported records
    keyed by (subject, channel, acquisition index, revision), up to tick `until`."""
    trace, log = index_world(run.trace, w), index_world(run.carry.log, w)
    tick = np.asarray(trace.tick)
    keep = np.asarray(trace.handled) & np.asarray(trace.wake)
    rows = np.asarray(log.valid) & (np.asarray(log.channel) != PRECURSOR_CHANNEL)
    if until is not None:
        keep &= tick < until
        rows &= np.asarray(log.start_tick) < until
    decisions = tuple(
        (int(t), tuple(map(int, k)), tuple(map(int, a)))
        for t, k, a in zip(
            tick[keep],
            np.asarray(trace.eff_kind)[keep],
            np.asarray(trace.admit_code)[keep],
            strict=True,
        )
    )
    cols = (log.subject, log.channel, log.acquisition_index, log.revision)
    records = tuple(zip(*(np.asarray(c)[rows].tolist() for c in cols), strict=True))
    return decisions, records


def shown(run, w):
    """Ticks of the decisions whose snapshot showed a precursor row."""
    trace, log = index_world(run.trace, w), index_world(run.carry.log, w)
    keep = np.asarray(trace.handled) & np.asarray(trace.wake)
    released = np.asarray(log.released_tick)
    imported = (
        np.asarray(log.valid)
        & (np.asarray(log.channel) == PRECURSOR_CHANNEL)
        & (released >= 0)
    )
    first = released[imported].min() if imported.any() else np.inf
    return [int(t) for t in np.asarray(trace.tick)[keep] if t >= first]


@pytest.mark.slow  # one real upstream run and the survey on both executors (K6)
def test_the_precursor_raises_the_value_by_its_known_amount():
    """B7 row 4 (a), with pre-flight and executor agreement. Fails if, with the same
    truths and seeds, the value with the precursor less the value without it is not
    the toy's `(n_obs + k) // 2 - n_obs // 2` for `k` precursor rows of one target
    (positive), in every world and on both executors; if a variant fails pre-flight
    with its upstream document; or if the executors' histories differ.
    """
    k = _precursor().n_rows
    assert k == 2
    want = (N_OBSERVATIONS + k) // 2 - N_OBSERVATIONS // 2
    context = PreflightContext(upstream_document=UPSTREAM)
    for variant in ("with", "without", "zero", "observer"):
        study, host = _run("host", variant)
        assert preflight(study, context=context) == [], variant
        assert same_histories(host, _run("compiled", variant)[1], N_WORLDS), variant
    for executor in EXECUTORS:
        study, with_ = _run(executor, "with")
        _, without = _run(executor, "without")
        for w in range(N_WORLDS):
            value = study.objective.evaluate(index_world(with_.carry, w), study)
            base = study.objective.evaluate(index_world(without.carry, w), study)
            assert int(value) - int(base) == want > 0, (executor, w)
        # the value is the least-known target's count of measurements the observer
        # accepted, read from the log (the observer-only precursor leaves the counts
        # unequal: the policy never sees it)
        for variant in ("with", "without", "observer"):
            study, run = _run(executor, variant)
            for w in range(N_WORLDS):
                carry = index_world(run.carry, w)
                assert int(study.objective.evaluate(carry, study)) == _least_known(
                    carry.log
                ), (executor, variant, w)


def _least_known(log):
    """The fewest accepted measurements of a target, counted from the log."""
    taken = (
        np.asarray(log.valid)
        & (np.asarray(log.kind) == MEAS)
        & (np.asarray(log.accepted_tick) >= 0)
    )
    subjects = np.asarray(log.subject)[taken]
    return min(int(np.sum(subjects == t)) for t in TARGETS)


@pytest.mark.slow  # shares the runs of the value test (K6)
def test_the_precursor_moves_only_decisions_that_saw_it():
    """B7 row 4 (b) and (c), both executors. Fails if the projection of decisions and
    non-imported records differs from the run without the precursor before the first
    decision whose snapshot showed a precursor row (counted, nonzero, and the
    projection must then differ); if an observer-only precursor, which no decision
    sees, moves any of it; or if a zero-information precursor moves any of it or
    changes the number of decisions at tick 0.
    """
    for executor in EXECUTORS:
        _, without = _run(executor, "without")
        _, with_ = _run(executor, "with")
        _, hidden = _run(executor, "observer")
        _, zero = _run(executor, "zero")
        for w in range(N_WORLDS):
            ticks = shown(with_, w)
            assert ticks, "no decision saw the precursor"
            first = ticks[0]
            assert projection(with_, w, first) == projection(without, w, first)
            assert projection(with_, w) != projection(without, w)
            assert shown(hidden, w) == []
            assert projection(hidden, w) == projection(without, w)
            # evidence: the zero-information rows were imported, public, and
            # stamped at tick 0 for both audiences, before the projections compare
            log = index_world(zero.carry.log, w)
            imported = np.flatnonzero(
                np.asarray(log.valid)
                & (np.asarray(log.channel) == PRECURSOR_CHANNEL)
                & (np.asarray(log.kind) == toy.RECORD_AUXILIARY)
            )
            assert imported.size == _precursor().n_rows == 2
            assert np.asarray(log.audience)[imported].tolist() == [PUBLIC] * 2
            assert np.asarray(log.released_tick)[imported].tolist() == [0, 0]
            assert np.asarray(log.accepted_tick)[imported].tolist() == [0, 0]
            assert projection(zero, w) == projection(without, w)
            decisions, records = projection(without, w)
            assert len(decisions) == N_OBSERVATIONS + 1
            assert len(records) == N_OBSERVATIONS
            at_zero = [d for d in decisions if d[0] == 0]
            assert len(at_zero) == 1
            assert [d for d in projection(zero, w)[0] if d[0] == 0] == at_zero
            # the hidden precursor still reached the observer
            hidden_n = index_world(hidden.carry, w).belief.n
            bare_n = index_world(without.carry, w).belief.n
            assert int(np.sum(hidden_n)) == int(np.sum(bare_n)) + 2


@pytest.mark.slow  # the epoch wrapper compiles its own program per executor (K6)
def test_the_epoch_relation_holds_with_an_exported_precursor():
    """B6 on survey 4. Fails if shifting the epoch origin moves any decision, record
    or belief other than by the shift, on either executor, for the survey importing
    a precursor whose times are all before the start; or if the relation compares
    no imported row or decision.
    """
    study = survey_4(_precursor())
    truths, roots = worlds(N_WORLDS, study.caps)
    for executor in (CompiledExecutor(), HostExecutor()):
        result = battery.shift_epoch(
            study, truths, roots, N_STEPS, 1000, executor=executor
        )
        assert result, f"{type(executor).__name__}: {result.diagnostic}"
        assert result.detail["n_before_start"] == N_WORLDS * 2
        assert result.detail["n_views"] == N_WORLDS * (N_OBSERVATIONS + 1)
        assert result.detail["n_records"] == N_WORLDS * (2 + N_OBSERVATIONS)
