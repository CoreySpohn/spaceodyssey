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
    INSTALLING_STATUSES,
    AdmissionCode,
    Audience,
    Capacity,
    ChargePricing,
    EffectKind,
    EventKind,
    RecordStatus,
    RemainderRule,
)
from spaceodyssey.core.records import pedigree_of
from spaceodyssey.core.snapshot import snapshot_view

ATOL = 1e-9


def _valid_rows(run):
    return np.flatnonzero(np.asarray(run.carry.log.valid))


def _pending_settlements(run):
    """Slots of queued completions that still belong to a live occupancy."""
    ev, ledger = run.carry.ev, run.carry.ledger
    done = np.asarray(ev.valid) & (np.asarray(ev.kind) == EventKind.DONE)
    lane_gen = np.asarray(ledger.lane_gen)
    # only a DONE row names a lane; a WINDOW_END names a balance row in the same column
    lane = np.clip(np.asarray(ev.lane), 0, lane_gen.shape[0] - 1)
    live = done & (np.asarray(ev.gen) == lane_gen[lane])
    # a commitment not yet started is debited and settles later, and it stays queued
    # only while pending (a cancel removes its start)
    pending = np.asarray(ev.valid) & (np.asarray(ev.kind) == EventKind.COMMIT_START)
    return np.flatnonzero(live | pending)


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


def _expected_folds(log, rows, ticks, derived=(), incremental=True):
    """One key per acquisition id with a row whose `ticks` entry is set.

    A derived row (a product or a declared summary) is never expected to be folded, so
    a derived acquisition id has no key; an observer that is not incremental changes
    only at checkpoints and folds nothing.
    """
    first = {}
    if not incremental:
        return Counter()
    for i in rows:
        if int(ticks[i]) >= 0 and int(log.kind[i]) not in derived:
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
        want = _expected_folds(
            log,
            rows,
            ticks,
            run.study.caps.derived_kinds,
            run.study.observer.capabilities().incremental,
        )
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
    derived = run.study.caps.derived_kinds
    seen = {
        int(log.acquisition_id[i])
        for i in rows
        if int(log.accepted_tick[i]) >= 0
        and int(log.status[i]) == done
        and int(log.kind[i]) not in derived
    }
    layout = run.study.observer.layout(run.study.caps)
    folds = int(np.asarray(layout.count(c.belief)).sum())
    if folds > len(seen):
        failures.append(f"{folds} folds for {len(seen)} accepted acquisitions")
    released = {
        int(log.acquisition_id[i])
        for i in rows
        if int(log.released_tick[i]) >= 0
        and int(log.status[i]) == done
        and int(log.kind[i]) not in derived
    }
    pfolds = int(np.asarray(layout.count(c.pbelief)).sum())
    if pfolds > len(released):
        failures.append(f"{pfolds} policy folds for {len(released)} released")
    return make_result("one_fold_per_acquisition", failures, folds=folds)


def no_violation(run):
    """The run raised no transaction violation."""
    failures = ["violation is set"] if bool(run.carry.violation) else []
    return make_result("no_violation", failures)


def _balance_rows(study, lane, program=0):
    caps = study.caps
    asset = int(np.asarray(study.assets.lane_asset)[lane])
    owner = caps.n_assets + (program if caps.n_programs > 1 else 0)
    return (asset, owner, caps.n_balance_rows - 1)


def _charge_totals(study, ops=0):
    """What an admission debits per channel and resource, read from the declarations.

    The cost, until the capacities hold refunds; then the charge plus the fixed part
    of a SURCHARGE column. With a per-state cost table the variable part is its row for
    the asset's ops state at admission (`ops`). Written out here, not read from the
    ledger's own helper.
    """
    ch = study.channels
    by_ops = None if ch.cost_by_ops is None else np.asarray(ch.cost_by_ops)[ops]
    if not study.caps.refunds:
        return np.asarray(ch.cost) if by_ops is None else by_ops
    base = np.asarray(ch.charge) if by_ops is None else by_ops
    if ch.pricing is None or ch.surcharge is None:
        return base
    fixed = np.asarray(ch.surcharge) * (
        np.asarray(ch.pricing) == ChargePricing.SURCHARGE
    )
    return base + fixed


def _log_program(log, i):
    return 0 if log.program is None else int(np.asarray(log.program)[i])


def _is_receipt(study, log, i):
    """Whether row `i` is a failure receipt, by the rule the handler applies.

    A receipt is a record of a declared failure kind written on a channel other than
    the one it names (its `source`); an acquisition of the failure kind is not one.
    """
    kinds = study.channels.failure_kind
    if kinds is None or log.refund_index is None:
        return False
    kind, channel = int(np.asarray(log.kind)[i]), int(np.asarray(log.channel)[i])
    return kind in np.asarray(kinds).tolist() and channel != int(
        np.asarray(log.source)[i]
    )


def _refund_credit(study, log, i, valid, credited):
    """The origin of applied receipt `i`, its credit, and whether `refund_of` agrees.

    The origin is resolved from the receipt's semantic identity alone (its `subject`,
    the channel it names in `source` and the origin's acquisition index in
    `refund_index`), the way the handler picks it, not from the `refund_of` the handler
    wrote; `agrees` says whether that column names the same acquisition. The credit is
    the refundable remainder of the origin, `charged` less the `nonrefundable` part,
    and only the first receipt for an origin earns it. The receipt's own `charged`
    column is not read, so a handler that over-credits is caught by the ledger's
    counters.
    """
    ids, refs = np.asarray(log.acquisition_id), np.asarray(log.refund_of)
    subject, channel = np.asarray(log.subject), np.asarray(log.channel)
    index = np.asarray(log.acquisition_index)
    want = (
        int(subject[i]),
        int(np.asarray(log.source)[i]),
        int(np.asarray(log.refund_index)[i]),
    )
    hits = [
        j
        for j in valid
        if j != i and (int(subject[j]), int(channel[j]), int(index[j])) == want
    ]
    if not hits:
        return None, 0.0, False
    j = hits[0]
    agrees = int(refs[i]) == int(ids[j])
    if int(ids[j]) in credited:
        return None, 0.0, agrees
    credited.add(int(ids[j]))
    charged = np.asarray(log.charged)[j]
    fixed = 0.0 if log.nonrefundable is None else np.asarray(log.nonrefundable)[j]
    return j, np.maximum(charged - fixed, 0.0), agrees


def _receipt_mismatches(run):
    """Applied receipts whose `charged` is not the negative of the credit they earn."""
    log, bad = run.carry.log, []
    valid, credited = _valid_rows(run), set()
    refs, charged = np.asarray(log.refund_of), np.asarray(log.charged)
    for i in valid:
        if int(refs[i]) < 0:
            continue
        _, credit, agrees = _refund_credit(run.study, log, i, valid, credited)
        if not agrees or np.any(np.abs(charged[i] + credit) > ATOL):
            bad.append(int(i))
    return bad


def _retained_net_spend(run, refunds=None):
    """Net charge of the retained records plus full cost of each live occupancy.

    With awards on, `refunds` is the replay's per-row credit (a refund may lapse or move
    to another row), and the receipts are not credited here.
    """
    study, c = run.study, run.carry
    spent = np.zeros(np.asarray(study.budget).shape)
    log = c.log
    charged = np.asarray(log.charged)
    refs = np.asarray(log.refund_of)
    valid = _valid_rows(run)
    credited = set()
    for i in valid:
        if int(refs[i]) >= 0:
            if refunds is not None:
                continue
            j, credit, _ = _refund_credit(run.study, log, i, valid, credited)
            if j is None:
                continue
            rows = _balance_rows(
                study, int(np.asarray(log.lane)[j]), _log_program(log, j)
            )
            for r, row in enumerate(rows):
                spent[row, r] -= float(credit[r])
            continue
        rows = _balance_rows(study, int(np.asarray(log.lane)[i]), _log_program(log, i))
        for r, row in enumerate(rows):
            spent[row, r] += float(charged[i, r])
    ev = c.ev
    for i in _pending_settlements(run):
        program = 0 if ev.program is None else int(np.asarray(ev.program)[i])
        rows = _balance_rows(study, int(ev.lane[i]), program)
        # a queued settlement is priced at the ops state it was admitted under
        queued = _charge_totals(study, 0 if ev.ops is None else int(ev.ops[i]))
        for r, row in enumerate(rows):
            spent[row, r] += float(queued[int(ev.channel_index[i]), r])
    # a disruptive activation is debited from the asset's own row and has no record
    assets = study.assets
    used = np.asarray(c.ledger.disruptive_used)
    for a in np.flatnonzero(used):
        spent[a, int(np.asarray(assets.disruptive_resource)[a])] += float(used[a])
    if refunds is not None:
        spent -= refunds
    return spent


def _refused_completions(run):
    """Full cost of settlements the trace handled whose record the log refused.

    A completion handled with no row written lost its record to log capacity; its
    debit stands and nothing credits it back. Cancellations are not counted here. The
    trace names the program the settlement was charged under when the study holds more
    than one.
    """
    study, t = run.study, run.trace
    ids = np.asarray(study.channels.id).tolist()
    if study.caps.ops and study.channels.cost_by_ops is not None:
        raise NotImplementedError(
            "a completion the log refused cannot be priced: the trace has no ops state"
        )
    cost = _charge_totals(study)
    lost = np.zeros(np.asarray(study.budget).shape)
    refused = (
        np.asarray(t.handled)
        & (np.asarray(t.kind) == EventKind.DONE)
        & (np.asarray(t.n_written) == 0)
    )
    for i in np.flatnonzero(refused):
        program = 0 if t.program is None else int(np.asarray(t.program)[i])
        rows = _balance_rows(study, int(t.lane[i]), program)
        for r, row in enumerate(rows):
            lost[row, r] += float(cost[ids.index(int(t.channel[i])), r])
    return lost


def _award_timeline(run):
    """The money events of a run in the order the executor handled them.

    Each entry is `(tick, stage, seq, kind, data)`. Stage 1 holds the credit of an
    applied receipt (at its `released_tick`), stage 2 the expiries the trace handled,
    stage 3 everything a decision does: grants, admission debits (a settled or
    cancelled record gives its tick as `start_tick` less the lane's transition, a
    queued settlement as its due tick less occupancy and transition) and a
    cancellation's credit (at its `end_tick`). Nothing here reads the ledger's
    balances or counters. The second result lists the problems found on the way: an
    applied receipt whose `refund_of` disagrees with the origin its identity names.
    """
    study, c, t = run.study, run.carry, run.trace
    log, assets, ch = c.log, study.assets, study.channels
    occupancy = np.asarray(ch.occupancy_ticks)
    transition = np.asarray(assets.transition_ticks)
    out, seq = [], 0
    kind, tick, eff = np.asarray(t.kind), np.asarray(t.tick), np.asarray(t.eff_kind)
    ok = np.asarray(t.admit_code) == AdmissionCode.OK
    for i in np.flatnonzero(np.asarray(t.handled)):
        if int(kind[i]) == EventKind.WINDOW_END:
            out.append((int(tick[i]), 2, seq, "expire", int(np.asarray(t.lane)[i])))
            seq += 1
        for e in np.flatnonzero((eff[i] == EffectKind.ALLOCATE) & ok[i]):
            data = tuple(
                np.asarray(x)[i, e]
                for x in (
                    t.alloc_recipient,
                    t.alloc_amount,
                    t.alloc_until,
                    t.alloc_remainder,
                    t.alloc_to,
                )
            )
            out.append((int(tick[i]), 3, seq, "grant", data))
            seq += 1
    valid, credited = _valid_rows(run), set()
    refs, status = np.asarray(log.refund_of), np.asarray(log.status)
    charged, lane = np.asarray(log.charged), np.asarray(log.lane)
    problems = []
    for i in valid:
        if int(refs[i]) >= 0:
            j, credit, agrees = _refund_credit(study, log, i, valid, credited)
            if not agrees:
                problems.append(
                    f"receipt row {int(i)} names acquisition {int(refs[i])}, not the "
                    "origin its subject, channel and index identify"
                )
            if j is not None:
                rows = _balance_rows(study, int(lane[j]), _log_program(log, j))
                when = int(np.asarray(log.released_tick)[i])
                captured = tuple(
                    None if col is None else np.asarray(col)[j]
                    for col in (log.award_gens, log.award_rules, log.award_tos)
                )
                out.append((when, 1, seq, "refund", (rows, credit, captured)))
                seq += 1
            continue
        if _is_receipt(study, log, i):
            continue  # ignored, unmatched or held: it moves no money
        index = int(np.asarray(log.channel_index)[i])
        rows = _balance_rows(study, int(lane[i]), _log_program(log, i))
        priced = _charge_totals(study, 0 if log.ops is None else int(log.ops[i]))
        start = int(np.asarray(log.start_tick)[i]) - int(transition[int(lane[i])])
        gens = tuple(
            None if col is None else np.asarray(col)[i]
            for col in (log.award_gens, log.award_rules, log.award_tos)
        )
        cancelled = int(status[i]) in (RecordStatus.PARTIAL, RecordStatus.CANCELLED)
        debit = priced[index] if cancelled else charged[i]
        out.append((start, 3, seq, "debit", (rows, debit, gens)))
        seq += 1
        if cancelled:
            back = np.maximum(priced[index] - charged[i], 0.0)
            end = int(np.asarray(log.end_tick)[i])
            out.append((end, 3, seq, "credit", (rows, back, None)))
            seq += 1
    ev = c.ev
    for i in _pending_settlements(run):
        index = int(np.asarray(ev.channel_index)[i])
        program = 0 if ev.program is None else int(np.asarray(ev.program)[i])
        rows = _balance_rows(study, int(ev.lane[i]), program)
        start = int(np.asarray(ev.tick)[i]) - int(occupancy[index])
        start -= int(transition[int(ev.lane[i])])
        queued = _charge_totals(study, 0 if ev.ops is None else int(ev.ops[i]))
        out.append((start, 3, seq, "debit", (rows, queued[index], None)))
        seq += 1
    return sorted(out, key=lambda x: x[:3]), problems


def _replay_awards(run):
    """Replay every credit, debit, grant and expiry; return what the ledger must hold.

    The result is `(balance, granted, lapsed, gen, problems, refunded, receipts)`, where
    `receipts` is the part of `refunded` that applied receipts credited. An expiry finds
    the award that opened its window (same row and expiry tick, earlier in the trace)
    and applies that award's remainder rule to what the replayed row holds then: FORFEIT
    lapses it, REALLOCATE lapses it and grants it to the declared row, NONE and
    CARRY_OVER move nothing. A record's `award_gens`, `award_rules` and `award_tos`, one
    entry per payer slot, must be the generation, rule and destination of the window the
    paying row held at its debit (a grant on that row in the same tick may come before
    or after the debit, so either is accepted; outside a window the generation is 0 and
    the rule NONE).
    A refund is disposed from the origin's captured identity against the replayed row:
    it credits the row when the generation is unchanged, or when the rule is NONE or
    CARRY_OVER and no window is open on the row; it credits the captured destination
    under REALLOCATE; any other credit lapses (refunded and lapsed both rise, the
    balance does not move).
    """
    budget = np.asarray(run.study.budget)
    balance, granted = budget.astype(np.float64).copy(), np.zeros_like(budget)
    lapsed, gen = np.zeros_like(budget), np.zeros(budget.shape[0], np.int64)
    refunded = np.zeros_like(budget)
    receipts = np.zeros_like(budget)
    opened = []
    timeline, problems = _award_timeline(run)
    by_group = {}
    for entry in timeline:
        by_group.setdefault(entry[:2], []).append(entry)

    def holds(row):
        """The rule and destination of the window open on `row`, or None."""
        return {w[0]: (w[2], w[3]) for w in opened}.get(row)

    for (now, _), group in by_group.items():
        before = gen.copy()
        windows = {row: holds(row) for row in range(budget.shape[0])}
        for _, _, _, what, data in group:
            if what == "grant":
                row, amount, until, rule, to = data
                row = int(row)
                balance[row] += amount
                granted[row] += amount
                gen[row] += 1
                opened = [w for w in opened if w[0] != row]
                opened.append((row, int(until), int(rule), int(to)))
            elif what == "credit":
                rows, amount, _ = data
                for r, row in enumerate(rows):
                    balance[row, r] += float(amount[r])
                    refunded[row, r] += float(amount[r])
            elif what == "refund":
                rows, amount, (gens, rules, tos) = data
                for r, row in enumerate(rows):
                    credit = float(amount[r])
                    receipts[row, r] += credit
                    rule = int(rules[r])
                    open_now = any(w[0] == row for w in opened)
                    same = int(gens[r]) == int(gen[row])
                    keeps = rule in (RemainderRule.NONE, RemainderRule.CARRY_OVER)
                    if same or (keeps and not open_now):
                        balance[row, r] += credit
                        refunded[row, r] += credit
                    elif rule == RemainderRule.REALLOCATE:
                        balance[int(tos[r]), r] += credit
                        refunded[int(tos[r]), r] += credit
                        receipts[row, r] -= credit
                        receipts[int(tos[r]), r] += credit
                    else:
                        refunded[row, r] += credit
                        lapsed[row, r] += credit
            elif what == "debit":
                rows, amount, _ = data
                for r, row in enumerate(rows):
                    balance[row, r] -= float(amount[r])
            else:
                row = int(data)
                hit = [w for w in opened if w[0] == row and w[1] == now]
                if not hit:
                    continue
                _, _, rule, to = hit[-1]
                opened.remove(hit[-1])
                gen[row] += 1
                if rule in (RemainderRule.FORFEIT, RemainderRule.REALLOCATE):
                    held = balance[row].copy()
                    balance[row] -= held
                    lapsed[row] += held
                    if rule == RemainderRule.REALLOCATE:
                        balance[to] += held
                        granted[to] += held
        for _, _, _, what, data in group:
            if what != "debit" or data[2] is None or data[2][0] is None:
                continue
            rows, _, (seen, rules, tos) = data
            after = {row: holds(row) for row in range(budget.shape[0])}
            for r, row in enumerate(rows):
                # outside a window the generation is 0 and the rule NONE
                allowed = {
                    int(before[row]) if windows[row] else 0,
                    int(gen[row]) if after[row] else 0,
                }
                if int(seen[r]) not in allowed:
                    problems.append(
                        f"a record names generation {int(seen[r])} for row {row}"
                    )
                none = (int(RemainderRule.NONE), 0)
                named = (int(rules[r]), int(tos[r]))
                if named not in (windows[row] or none, after[row] or none):
                    problems.append(
                        f"a record names rule {named[0]} and destination "
                        f"{named[1]} for row {row}, which held {windows[row]} "
                        "at its debit"
                    )
    return balance, granted, lapsed, gen, problems, refunded, receipts


def _award_mismatches(run, replay=None):
    """What the ledger's award counters and balances hold that the replay does not.

    Skipped only when the log overflowed: a debit or credit may then have no record to
    replay it from, and the three-way account in `balance_conserved` still binds.
    """
    c = run.carry
    if int(c.overflow_mask) & Capacity.LOG_ROWS:
        return []
    balance, granted, lapsed, gen, bad, refunded, _ = replay or _replay_awards(run)
    led = c.ledger
    for name, want, have in (
        ("balance", balance, led.balance),
        ("granted", granted, led.granted),
        ("lapsed", lapsed, led.lapsed),
        ("refunded", refunded, led.refunded),
    ):
        diff = np.abs(np.asarray(have) - want)
        if np.any(diff > ATOL):
            row, res = np.unravel_index(int(np.argmax(diff)), diff.shape)
            bad.append(
                f"the replay of the trace and records puts {name} at row {row} "
                f"resource {res} at {want[row, res]}, the ledger holds "
                f"{np.asarray(have)[row, res]}"
            )
    if np.any(np.asarray(led.win_gen) != gen):
        bad.append("a row's window generation differs from the replay")
    return bad


def balance_conserved(run):
    """Initial balance plus awards less what was spent equals the final balance.

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
    awarded = np.zeros_like(budget)
    if c.ledger.granted is not None:
        awarded = np.asarray(c.ledger.granted) - np.asarray(c.ledger.lapsed)
    gap = np.abs(final - (budget + awarded - net))
    if not np.all(np.isfinite(gap) & (gap <= ATOL)):
        row, res = np.unravel_index(int(np.argmax(gap)), gap.shape)
        failures.append(
            f"balance row {row} resource {res} is {final[row, res]} but initial "
            f"plus awards less the ledger's net debits is "
            f"{budget[row, res] + awarded[row, res] - net[row, res]}"
        )
    replay = None if c.ledger.granted is None else _replay_awards(run)
    kept = _retained_net_spend(run, None if replay is None else replay[6])
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
    for i in _receipt_mismatches(run):
        failures.append(f"receipt row {i} does not carry the credit it is owed")
    if c.ledger.granted is not None:
        failures.extend(_award_mismatches(run, replay))
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
    admission = set(np.asarray(study.channels.id).tolist())
    cut = (RecordStatus.PARTIAL, RecordStatus.FAILED)
    for i in _valid_rows(run):
        acq = int(log.acquisition_id[i])
        # a row on a channel outside the admission table was written by a process, a
        # service or a refresh: it names no admission
        if int(log.channel[i]) not in admission:
            continue
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


def _product_slots(run):
    """Valid log rows of the refresh product kind (none without the protocol)."""
    study = run.study
    if study.caps.rows_per_subject <= 0:
        return []
    log = run.carry.log
    return [i for i in _valid_rows(run) if int(log.kind[i]) == int(study.refresh.kind)]


def _received_sets(run):
    """Per checkpoint completion: the product slots and what each audience had received.

    Walks the handled steps of the trace in order, from the log and the trace alone: a
    handled `RELEASE` or `RESULT` step delivers its row to the observer and the policy
    as the event's audience and the row's audience allow, and a `REFRESH` step that
    wrote rows is a completion whose product rows are the last `n_written` slots it
    names. Yields `(step, hold_step, slots, received)`, `received` being one list of
    slots per audience (observer, policy) at that moment.
    """
    t = run.trace
    log = run.carry.log
    kind, ev_aud = np.asarray(t.kind), np.asarray(t.audience)
    slot, wrote = np.asarray(t.log_slot), np.asarray(t.n_written)
    row_aud, status = np.asarray(log.audience), np.asarray(log.status)
    received, hold, out = ([], []), None, []
    for step in np.flatnonzero(np.asarray(t.handled)):
        k = int(kind[step])
        if k in (EventKind.RELEASE, EventKind.RESULT):
            j = int(slot[step])
            if int(status[j]) == RecordStatus.UNRELEASED:
                continue
            reach = int(ev_aud[step]) & int(row_aud[j])
            for a, flag in enumerate((Audience.OBSERVER, Audience.POLICY)):
                if reach & int(flag) and j not in received[a]:
                    received[a].append(j)
        elif k == EventKind.REFRESH and int(wrote[step]) == 0:
            hold = int(step)
        elif k == EventKind.REFRESH:
            last = int(slot[step])
            slots = list(range(last - int(wrote[step]) + 1, last + 1))
            out.append((int(step), hold, slots, ([*received[0]], [*received[1]])))
    return out


def _expected_revision_set(run, slots, subject):
    """The triples of the revision set of `subject` among the received log `slots`."""
    log = run.carry.log
    caps = run.study.caps
    current = {}
    for j in slots:
        if int(log.subject[j]) != subject:
            continue
        key = int(log.acquisition_id[j])
        if key not in current or int(log.revision[j]) > int(log.revision[current[key]]):
            current[key] = j
    return sorted(
        (
            int(log.channel[j]),
            int(log.acquisition_index[j]),
            int(log.revision[j]),
        )
        for j in current.values()
        if int(log.kind[j]) in caps.refit_kinds
        and int(log.status[j]) in caps.refit_statuses
    )


def refit_inputs_equal_revision_set(run):
    """Each product row's pedigree is the revision set its audience had at that time.

    The pedigree is read from the product column. The expected set is rebuilt from the
    trace order and the log alone: the current revision of each acquisition among the
    refit-kind rows the audience had received before the checkpoint completed, dropped
    when its status is not a refit status. A study without the refresh protocol passes
    with no rows checked.
    """
    failures, checked = [], 0
    log = run.carry.log
    for step, _, slots, received in _received_sets(run):
        for p in slots:
            a = 0 if int(log.audience[p]) == Audience.OBSERVER else 1
            want = _expected_revision_set(run, received[a], int(log.subject[p]))
            got = pedigree_of(log, p)[2]
            checked += 1
            if list(got) != want:
                failures.append(
                    f"product row {p} (step {step}, audience {a}, subject "
                    f"{int(log.subject[p])}) names {list(got)} but the audience had "
                    f"received {want}"
                )
                break
    if not _product_slots(run):
        return make_result("refit_inputs_equal_revision_set", [], n_checked=0)
    if not checked:
        failures.append("product rows exist but no checkpoint step wrote them")
    return make_result("refit_inputs_equal_revision_set", failures, n_checked=checked)


def installs_match_product_rows(run):
    """Each install has one product row that says what was installed, and nothing else.

    Per product row: its channel is its audience's product channel, its pedigree names
    the same audience, status and row count as the row's payload, it sits at its
    checkpoint's grid ordinal, and no audience, subject and ordinal repeat. The last
    row of each audience and subject carries the status the carry recorded. An observer
    entry whose status does not install leaves the trace's belief projection of that
    subject as it was when the checkpoint held the world.
    """
    study, c = run.study, run.carry
    log, failures = c.log, []
    slots = _product_slots(run)
    if not slots:
        return make_result("installs_match_product_rows", [], n_products=0)
    sched = study.refresh
    channels = (int(sched.observer_channel), int(sched.policy_channel))
    flags = (int(Audience.OBSERVER), int(Audience.POLICY))
    last, seen = {}, set()
    for p in slots:
        flag = int(log.audience[p])
        a = flags.index(flag) if flag in flags else -1
        subject, ordinal = int(log.subject[p]), int(log.acquisition_index[p])
        who = f"product row {p} (audience {a}, subject {subject})"
        pedigree = pedigree_of(log, p)
        status, n_rows = int(log.payload[p, 0]), int(log.payload[p, 1])
        if a < 0 or int(log.channel[p]) != channels[a]:
            failures.append(f"{who} is on channel {int(log.channel[p])}")
        if pedigree[0] != flag or pedigree[1] != status:
            failures.append(
                f"{who}: the pedigree says {pedigree[:2]}, not {(flag, status)}"
            )
        if len(pedigree[2]) != n_rows:
            failures.append(
                f"{who} counts {n_rows} rows but its pedigree names {len(pedigree[2])}"
            )
        if (a, subject, ordinal) in seen:
            failures.append(f"{who} repeats ordinal {ordinal}")
        seen.add((a, subject, ordinal))
        if (a, subject) not in last or ordinal >= last[a, subject][0]:
            last[a, subject] = (ordinal, status)
    recorded = np.asarray(c.refresh_status)
    for (a, subject), (_, status) in sorted(last.items()):
        if int(recorded[a, subject]) != status:
            failures.append(
                f"audience {a} subject {subject}: the last product row says status "
                f"{status}, the carry {int(recorded[a, subject])}"
            )
    t = run.trace
    mean = np.asarray(t.belief_mean)
    every, offset = int(sched.every_ticks), int(sched.offset_ticks)
    for step, hold, slots_at, _ in _received_sets(run):
        tick = int(t.tick[step])
        # a product is indexed by the grid ordinal of its checkpoint, at revision 0
        # and at that checkpoint's tick, whatever the log order
        ordinal = (tick - offset) // every if every > 0 else 0
        for p in slots_at:
            who = f"product row {p}"
            if int(log.acquisition_index[p]) != ordinal:
                failures.append(
                    f"{who} is indexed {int(log.acquisition_index[p])}, but its "
                    f"checkpoint at tick {tick} is grid ordinal {ordinal}"
                )
            if int(log.revision[p]) != 0 or int(log.available_tick[p]) != tick:
                failures.append(
                    f"{who} has revision {int(log.revision[p])} at tick "
                    f"{int(log.available_tick[p])}, not revision 0 at tick {tick}"
                )
        for p in slots_at:
            subject = int(log.subject[p])
            installs = int(log.payload[p, 0]) in {int(x) for x in INSTALLING_STATUSES}
            if int(log.audience[p]) == Audience.OBSERVER and not installs:
                if hold is None or not np.array_equal(
                    mean[step, subject], mean[hold, subject]
                ):
                    failures.append(
                        f"observer subject {subject}: status {int(log.payload[p, 0])} "
                        "does not install but the belief moved"
                    )
    return make_result("installs_match_product_rows", failures, n_products=len(slots))


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
    refit_inputs_equal_revision_set,
    installs_match_product_rows,
)


def check_run(run, invariants=INVARIANTS):
    """Every invariant over one run; a tuple of `CheckResult`."""
    return tuple(inv(run) for inv in invariants)


def failures_of(results):
    """The failed results among `results`."""
    return [r for r in results if not r.passed]
