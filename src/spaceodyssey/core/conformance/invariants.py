"""Invariants every conformance case must hold, on either executor.

Each invariant is a function of a `CaseRun` and returns a `CheckResult`: its name, a
pass flag and, when it fails, a diagnostic naming the first offending item. They read
the final carry, the trace and the decision views, never the study's hand history, so
a case cannot satisfy them by being pinned.

A run that ran out of log capacity (`Capacity.LOG_ROWS` set) dropped records by
construction. The balance, acquisition-index and fold checks still hold there: they
reconcile what survives the overflow (the ledger's debit and refund counters, the
completions the trace handled, the queued settlements) against the retained records.
"""

from collections import Counter

import numpy as np

from spaceodyssey.core.battery.result import make_result
from spaceodyssey.core.conformance.corpus import public_events
from spaceodyssey.core.conformance.observer_kit import double_fold_failures
from spaceodyssey.core.enums import (
    AdmissionCode,
    Audience,
    Capacity,
    EffectKind,
    EventKind,
    RecordStatus,
)
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


def _fold_key(log, i):
    """The identity `CountingObserver` gives the acquisition a log row holds."""
    return (
        int(log.channel[i]) * (1 << 40)
        + int(log.subject[i]) * (1 << 20)
        + int(log.acquisition_index[i])
    )


def _expected_folds(log, rows, ticks):
    """One key per acquisition id with a row whose `ticks` entry is set."""
    first = {}
    for i in rows:
        if int(ticks[i]) >= 0:
            first.setdefault(int(log.acquisition_id[i]), _fold_key(log, i))
    return Counter(first.values())


def one_fold_per_acquisition(run):
    """Each audience commits exactly one fold for each acquisition it accepted.

    The observer folds an acquisition when its first revision is accepted
    (`accepted_tick` set on a row of that acquisition id); the policy belief folds it
    when its first revision is released (`released_tick`). The committed folds the
    counting observer recorded must equal that set, key for key and count for count,
    so a dropped, doubled or misattributed commit fails.
    """
    c, failures = run.carry, []
    if len(run.folds) != 2:
        failures.append("fold counts were not recorded")
    log = c.log
    rows = _valid_rows(run)
    sources = (("observer", log.accepted_tick), ("policy", log.released_tick))
    for (name, ticks), (ids, counts) in zip(sources, run.folds, strict=False):
        got = Counter(
            {int(k): int(n) for k, n in zip(ids, counts, strict=True) if n > 0}
        )
        want = _expected_folds(log, rows, ticks)
        for key in sorted(got):
            if got[key] > 1:
                failures.append(
                    f"{name}: acquisition key {key} folded {got[key]} times"
                )
                break
        if got != want:
            missing = sorted((want - got).elements())
            extra = sorted((got - want).elements())
            failures.append(
                f"{name}: committed folds differ from accepted acquisitions "
                f"(missing {missing[:3]}, unexpected {extra[:3]})"
            )
    failures += double_fold_failures(c)
    done = RecordStatus.COMPLETED
    seen = {
        int(log.acquisition_id[i])
        for i in rows
        if int(log.accepted_tick[i]) >= 0 and int(log.status[i]) == done
    }
    layout = run.study.observer.layout(run.study.caps)
    folds = int(np.asarray(layout.count(c.belief)).sum())
    if folds > len(seen):
        failures.append(f"{folds} folds for {len(seen)} accepted acquisitions")
    released = {
        int(log.acquisition_id[i])
        for i in rows
        if int(log.released_tick[i]) >= 0 and int(log.status[i]) == done
    }
    pfolds = int(np.asarray(layout.count(c.pbelief)).sum())
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


def _retained_net_spend(run):
    """Net charge of the retained records plus full cost of each live occupancy."""
    study, c = run.study, run.carry
    spent = np.zeros(np.asarray(study.budget).shape)
    log, cost = c.log, np.asarray(study.channels.cost)
    charged = np.asarray(log.charged)
    for i in _valid_rows(run):
        rows = _balance_rows(study, int(log.lane[i]))
        for r, row in enumerate(rows):
            spent[row, r] += float(charged[i, r])
    ev = c.ev
    for i in _pending_settlements(run):
        rows = _balance_rows(study, int(ev.lane[i]))
        for r, row in enumerate(rows):
            spent[row, r] += float(cost[int(ev.channel_index[i]), r])
    return spent


def _refused_completions(run):
    """Full cost of settlements the trace handled whose record the log refused.

    A completion handled with no row written lost its record to log capacity; its
    debit stands and nothing credits it back. Cancellations are not counted here.
    """
    study, t = run.study, run.trace
    ids = np.asarray(study.channels.id).tolist()
    cost = np.asarray(study.channels.cost)
    lost = np.zeros(np.asarray(study.budget).shape)
    refused = (
        np.asarray(t.handled)
        & (np.asarray(t.kind) == EventKind.DONE)
        & (np.asarray(t.n_written) == 0)
    )
    for i in np.flatnonzero(refused):
        rows = _balance_rows(study, int(t.lane[i]))
        for r, row in enumerate(rows):
            lost[row, r] += float(cost[ids.index(int(t.channel[i])), r])
    return lost


def balance_conserved(run):
    """Initial balance minus what was spent equals the final balance, row by row.

    Three independent accounts must agree, with or without log overflow: the final
    balance, the ledger's own debit and refund counters, and the retained records plus
    the settlements still queued. Records the log refused are accounted from the trace
    (the completions it handled that wrote no row), so exhausting the log does not
    switch the check off. A cancellation whose record was refused cannot be priced from
    evidence that survives; such a run fails the residual check rather than passing.
    """
    study, c = run.study, run.carry
    budget = np.asarray(study.budget)
    final = np.asarray(c.ledger.balance)
    if not np.all(np.isfinite(final)):
        return make_result("balance_conserved", ["the final balance is not finite"])
    failures = []
    net = np.asarray(c.ledger.debited) - np.asarray(c.ledger.refunded)
    gap = np.abs(final - (budget - net))
    if not np.all(np.isfinite(gap) & (gap <= ATOL)):
        row, res = np.unravel_index(int(np.argmax(gap)), gap.shape)
        failures.append(
            f"balance row {row} resource {res} is {final[row, res]} but initial "
            f"minus the ledger's net debits is {budget[row, res] - net[row, res]}"
        )
    kept = _retained_net_spend(run)
    overflowed = bool(int(c.overflow_mask) & Capacity.LOG_ROWS)
    lost = _refused_completions(run) if overflowed else np.zeros_like(kept)
    residual = np.abs(net - kept - lost)
    if not np.all(np.isfinite(residual) & (residual <= ATOL)):
        row, res = np.unravel_index(int(np.argmax(residual)), residual.shape)
        failures.append(
            f"balance row {row} resource {res}: net debits {net[row, res]} but "
            f"records and queued settlements account for "
            f"{kept[row, res] + lost[row, res]}"
        )
    return make_result(
        "balance_conserved",
        failures,
        max_gap=float(gap.max()),
        max_residual=float(residual.max()),
        overflow=overflowed,
    )


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


def _admission_evidence(run):
    """Admissions seen outside the counters, as {acquisition id: (subject, index)}.

    An admitted observation queues one completion whose sid is its acquisition id, so
    the completions the trace handled and the ones still queued name every admission
    that did not end in a cancellation; a cancelled one is named by its retained
    partial or failed record. Records a process wrote directly are not admissions.
    """
    study, c, t = run.study, run.carry, run.trace
    named, failures = {}, []

    def note(acq, subject, index, source):
        if named.setdefault(acq, (subject, index)) != (subject, index):
            failures.append(
                f"acquisition {acq} is named {named[acq]} and "
                f"{(subject, index)} ({source})"
            )

    done = np.asarray(t.handled) & (np.asarray(t.kind) == EventKind.DONE)
    for i in np.flatnonzero(done):
        note(int(t.sid[i]), int(t.subject[i]), int(t.acquisition_index[i]), "trace")
    ev = c.ev
    queued = np.asarray(ev.valid) & (np.asarray(ev.kind) == EventKind.DONE)
    for i in np.flatnonzero(queued):
        note(int(ev.sid[i]), int(ev.subject[i]), int(ev.acquisition_index[i]), "queue")
    log = c.log
    kinds = set(np.asarray(study.channels.kind).tolist())
    cut = (RecordStatus.PARTIAL, RecordStatus.FAILED)
    for i in _valid_rows(run):
        acq = int(log.acquisition_id[i])
        if int(log.kind[i]) in kinds and (acq in named or int(log.status[i]) in cut):
            note(acq, int(log.subject[i]), int(log.acquisition_index[i]), "log")
    return named, failures


def dense_acquisition_index(run):
    """Per subject, admitted acquisitions hold the indices 0 to count - 1, each once.

    The counters are reconciled with admissions observed independently of them: the
    observe rows the trace shows admitted must sum to the counters, and every
    admitted acquisition (see `_admission_evidence`) must hold an index in
    `[0, count)` of its subject, no two sharing one. Without log overflow the indices
    must also be exactly `0 .. count - 1`; with it, the cancellation records the log
    refused may leave gaps, but never an index outside the range. A counter of zero
    next to an admitted acquisition fails.
    """
    c = run.carry
    counts = np.asarray(c.acquisition_counts)
    failures = []
    if np.any(counts < 0):
        failures.append(f"a subject counter is negative: {counts.tolist()}")
    t = run.trace
    admitted = int(
        (
            (np.asarray(t.eff_kind) == EffectKind.OBSERVE)
            & (np.asarray(t.admit_code) == AdmissionCode.OK)
        ).sum()
    )
    if admitted != int(counts.sum()):
        failures.append(
            f"the trace admitted {admitted} observations but the counters sum to "
            f"{int(counts.sum())}"
        )
    named, naming = _admission_evidence(run)
    failures += naming
    per_subject = {}
    for acq, (subject, index) in named.items():
        if not 0 <= subject < len(counts):
            failures.append(f"acquisition {acq} names subject {subject}")
        elif not 0 <= index < int(counts[subject]):
            failures.append(
                f"acquisition {acq} of subject {subject} has index {index} "
                f"with count {int(counts[subject])}"
            )
        per_subject.setdefault(subject, []).append(index)
    overflowed = bool(int(c.overflow_mask) & Capacity.LOG_ROWS)
    for subject, got in sorted(per_subject.items()):
        if len(got) != len(set(got)):
            failures.append(f"subject {subject} repeats an index in {sorted(got)}")
    for subject, count in enumerate(counts):
        n = int(count)
        got = sorted(per_subject.get(subject, []))
        if not overflowed and got != list(range(n)):
            failures.append(f"subject {subject} holds {got}, not 0 to {n - 1}")
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


def _view_failures(views, expected, table, label):
    """Hidden-column and capture failures of the views one policy form received."""
    failures = []
    if len(views) != expected:
        failures.append(
            f"{label}: {len(views)} decision views captured for {expected} decisions"
        )
    for n, view in enumerate(views):
        if view is None:
            failures.append(f"{label}: decision view {n} was not captured")
            break
        if _readable_leak(view, table, Audience.POLICY):
            failures.append(f"{label}: decision view {n} shows a hidden column")
            break
    return failures


def no_scorer_column_in_snapshot(run):
    """No payload column an audience may not read is nonzero in what it was handed.

    Judged on every view `policy.decide` (and `decide_py` on the host) actually
    received, one for each decision the trace woke, on either executor; and on the
    snapshot of the final state for the policy and the observer. The reference is
    `reference_caps` (default: the study's own layout).
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
    forms = {"array": run.views, **run.form_views}
    for form in run.case.policy_forms if run.executor == "host" else ("array",):
        if form not in forms:
            failures.append(f"{form}: no decision views were captured")
    for form, views in forms.items():
        failures += _view_failures(views, run.wakes, table, form)
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
