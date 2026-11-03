"""Invariants every conformance case must hold, on either executor.

Each invariant is a function of a `CaseRun` and returns a `CheckResult`: its name, a
pass flag and, when it fails, a diagnostic naming the first offending item. They read
the final carry, the trace and the decision views, never the study's hand history, so
a case cannot satisfy them by being pinned.

Two invariants are conditioned on the run having kept all its records. A run that ran
out of log capacity (`Capacity.LOG_ROWS` set) dropped records by construction. Balance
and acquisition-index checks cannot account for them; those two report `skipped`
in their detail and judge only what they can.
"""

import numpy as np

from spaceodyssey.core.battery.result import make_result
from spaceodyssey.core.conformance.corpus import public_events
from spaceodyssey.core.enums import Audience, Capacity, EventKind, RecordStatus
from spaceodyssey.core.snapshot import snapshot_view

ATOL = 1e-9


def _valid_rows(run):
    return np.flatnonzero(np.asarray(run.carry.log.valid))


def _pending_settlements(run):
    """Slots of queued completions that still belong to a live occupancy."""
    ev, ledger = run.carry.ev, run.carry.ledger
    live = (
        np.asarray(ev.valid)
        & (np.asarray(ev.kind) == EventKind.DONE)
        & (np.asarray(ev.gen) == np.asarray(ledger.lane_gen)[np.asarray(ev.lane)])
    )
    return np.flatnonzero(live)


def order_keys_monotone(run):
    """Handled rows come out in non-decreasing (tick, stage, micro, tie, id) order."""
    t = run.trace
    handled = np.asarray(t.handled)
    keys = np.stack(
        [
            np.asarray(getattr(t, n))[handled]
            for n in ("tick", "stage", "micro", "tie", "sid")
        ],
        axis=1,
    )
    failures = []
    for i in range(1, len(keys)):
        if tuple(keys[i]) < tuple(keys[i - 1]):
            failures.append(
                f"handled step {i} has key {tuple(keys[i].tolist())} after "
                f"{tuple(keys[i - 1].tolist())}"
            )
            break
    return make_result("order_keys_monotone", failures, n_steps=len(keys))


def exactly_once_settlement(run):
    """No completion settled twice: `double_settle` is clear, each id settles once."""
    log, failures = run.carry.log, []
    if bool(run.carry.double_settle):
        failures.append("double_settle is set")
    rows = _valid_rows(run)
    first = [int(log.acquisition_id[i]) for i in rows if int(log.revision[i]) == 0]
    repeated = sorted({a for a in first if first.count(a) > 1})
    if repeated:
        failures.append(f"acquisition {repeated[0]} has two revision-0 records")
    return make_result("exactly_once_settlement", failures, n_records=len(rows))


def one_fold_per_acquisition(run):
    """No record folded twice, and no belief holds more folds than acquisitions."""
    c, failures = run.carry, []
    if not run.folds:
        failures.append("fold counts were not recorded")
    for audience, (ids, counts) in enumerate(run.folds):
        repeated = np.flatnonzero(counts > 1)
        if repeated.size:
            i = repeated[0]
            failures.append(
                f"audience {audience}: acquisition {ids[i]} folded {counts[i]} times"
            )
    if bool(c.dup_fold):
        failures.append("dup_fold is set")
    log = c.log
    rows = _valid_rows(run)
    done = RecordStatus.COMPLETED
    seen = {
        int(log.acquisition_id[i])
        for i in rows
        if int(log.accepted_tick[i]) >= 0 and int(log.status[i]) == done
    }
    folds = int(np.asarray(c.belief.n).sum())
    if folds > len(seen):
        failures.append(f"{folds} folds for {len(seen)} accepted acquisitions")
    released = {
        int(log.acquisition_id[i])
        for i in rows
        if int(log.released_tick[i]) >= 0 and int(log.status[i]) == done
    }
    pfolds = int(np.asarray(c.pbelief.n).sum())
    if pfolds > len(released):
        failures.append(f"{pfolds} policy folds for {len(released)} released")
    return make_result("one_fold_per_acquisition", failures, folds=folds)


def no_violation(run):
    """The run raised no transaction violation."""
    failures = ["violation is set"] if bool(run.carry.violation) else []
    return make_result("no_violation", failures)


def _balance_rows(study, lane):
    caps = study.caps
    asset = int(np.asarray(study.assets.lane_asset)[lane])
    return (asset, caps.n_assets, caps.n_balance_rows - 1)


def balance_conserved(run):
    """Initial balance minus what was spent equals the final balance, row by row.

    What was spent is the net charge of every logged record (a cancelled acquisition
    has had its unused cost refunded already) plus the full cost of each acquisition
    still running when the run ended.
    """
    study, c = run.study, run.carry
    spent = np.zeros(np.asarray(study.budget).shape)
    log, cost = c.log, np.asarray(study.channels.cost)
    for i in _valid_rows(run):
        rows = _balance_rows(study, int(log.lane[i]))
        for r, row in enumerate(rows):
            spent[row, r] += float(np.asarray(log.charged)[i, r])
    ev = c.ev
    for i in _pending_settlements(run):
        rows = _balance_rows(study, int(ev.lane[i]))
        for r, row in enumerate(rows):
            spent[row, r] += float(cost[int(ev.channel_index[i]), r])
    expected = np.asarray(study.budget) - spent
    final = np.asarray(c.ledger.balance)
    if not np.all(np.isfinite(final)):
        return make_result("balance_conserved", ["the final balance is not finite"])
    if int(c.overflow_mask) & Capacity.LOG_ROWS:
        return make_result("balance_conserved", [], skipped="overflow")
    gap = np.abs(final - expected)
    failures = []
    if not np.all(np.isfinite(gap) & (gap <= ATOL)):
        row, res = np.unravel_index(int(np.argmax(gap)), gap.shape)
        failures.append(
            f"balance row {row} resource {res} is {final[row, res]} "
            f"but initial minus spent is {expected[row, res]}"
        )
    return make_result("balance_conserved", failures, max_gap=float(gap.max()))


def unique_acquisition_id(run):
    """No two records share an acquisition id and a revision."""
    log, failures = run.carry.log, []
    seen = {}
    for i in _valid_rows(run):
        key = (int(log.acquisition_id[i]), int(log.revision[i]))
        if key in seen:
            failures.append(
                f"slots {seen[key]} and {int(i)} share acquisition "
                f"{key[0]} revision {key[1]}"
            )
            break
        seen[key] = int(i)
    return make_result("unique_acquisition_id", failures, n_records=len(seen))


def dense_acquisition_index(run):
    """Per subject, admitted acquisitions hold the indices 0 to count - 1, each once.

    Acquisitions are the logged records of a kind some channel produces, one per
    acquisition id (revisions share an index), plus the completions still queued. A
    subject with a count of zero was never admitted anything and is not judged, so
    records a process writes directly are left alone.
    """
    study, c, failures = run.study, run.carry, []
    log = c.log
    counts = np.asarray(c.acquisition_counts)
    kinds = set(np.asarray(study.channels.kind).tolist())
    held = {}
    for i in _valid_rows(run):
        if int(log.kind[i]) in kinds:
            held.setdefault(
                int(log.acquisition_id[i]),
                (int(log.subject[i]), int(log.acquisition_index[i])),
            )
    ev = c.ev
    pending = {
        ("pending", int(i)): (int(ev.subject[i]), int(ev.acquisition_index[i]))
        for i in _pending_settlements(run)
    }
    per_subject = {}
    for subject, index in [*held.values(), *pending.values()]:
        per_subject.setdefault(subject, []).append(index)
    for subject, count in enumerate(counts):
        n = int(count)
        got = per_subject.get(subject, [])
        if n == 0:
            continue
        if len(got) != len(set(got)):
            failures.append(f"subject {subject} repeats an index in {sorted(got)}")
        elif got and max(got) >= n:
            failures.append(f"subject {subject} has index {max(got)} with count {n}")
        elif not (int(c.overflow_mask) & Capacity.LOG_ROWS) and sorted(got) != list(
            range(n)
        ):
            failures.append(f"subject {subject} holds {sorted(got)}, not 0 to {n - 1}")
    return make_result("dense_acquisition_index", failures, counts=counts.tolist())


def release_not_before_available(run):
    """A record is never released before it is available."""
    log, failures = run.carry.log, []
    for i in _valid_rows(run):
        released = int(log.released_tick[i])
        if released >= 0 and released < int(log.available_tick[i]):
            failures.append(
                f"slot {int(i)} released at {released} before available "
                f"{int(log.available_tick[i])}"
            )
            break
    return make_result("release_not_before_available", failures)


def _readable_leak(view, table, audience):
    kind = np.clip(np.asarray(view.log.kind), 0, len(table) - 1)
    unreadable = (table[kind] & int(audience)) == 0
    return bool(np.any(np.asarray(view.log.payload)[unreadable] != 0))


def no_scorer_column_in_snapshot(run):
    """No payload column an audience may not read is nonzero in its snapshot.

    Judged on the snapshot of the final state for the policy and the observer, and on
    every decision view the run kept, against `reference_caps` (default: the study's
    own layout).
    """
    reference = run.study.caps if run.reference_caps is None else run.reference_caps
    table = np.asarray(reference.payload_audience)
    failures = []
    for audience in (Audience.POLICY, Audience.OBSERVER):
        view = snapshot_view(run.carry, audience, run.study.caps)
        if _readable_leak(view, table, audience):
            failures.append(
                f"the final {audience.name.lower()} snapshot shows a hidden column"
            )
    for n, view in enumerate(run.views):
        if _readable_leak(view, table, Audience.POLICY):
            failures.append(f"decision view {n} shows a hidden column")
            break
    return make_result("no_scorer_column_in_snapshot", failures, n_views=len(run.views))


def no_extra_wakes_under_private_events(run):
    """Private events add no wake and no decision, and move no other row."""
    if run.probe is None:
        return make_result(
            "no_extra_wakes_under_private_events", ["the probe was not run"]
        )
    wakes, decisions, events = run.probe
    failures = []
    if wakes != run.wakes:
        failures.append(f"{wakes} wakes with private events, {run.wakes} without")
    if decisions != run.decisions:
        failures.append(
            f"{decisions} decisions with private events, {run.decisions} without"
        )
    if events != public_events(run.trace):
        failures.append("the handled rows other than the private events differ")
    return make_result("no_extra_wakes_under_private_events", failures, wakes=run.wakes)


INVARIANTS = (
    order_keys_monotone,
    exactly_once_settlement,
    one_fold_per_acquisition,
    no_violation,
    balance_conserved,
    unique_acquisition_id,
    dense_acquisition_index,
    release_not_before_available,
    no_scorer_column_in_snapshot,
    no_extra_wakes_under_private_events,
)


def check_run(run, invariants=INVARIANTS):
    """Every invariant over one run; a tuple of `CheckResult`."""
    return tuple(inv(run) for inv in invariants)


def failures_of(results):
    """The failed results among `results`."""
    return [r for r in results if not r.passed]
