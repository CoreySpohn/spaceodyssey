"""Transaction."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.carry import _tree_set, empty_trace
from spaceodyssey.core.enums import (
    I64,
    PUBLIC,
    TIE_NEUTRAL,
    AdmissionCode,
    Audience,
    CommitState,
    Disposition,
    EffectKind,
    EventKind,
    RemainderRule,
    Stage,
    WindowState,
    version_slot,
)
from spaceodyssey.core.events import enqueue, order_key, select_next, where_tree
from spaceodyssey.core.handlers import step, stop_reaches
from spaceodyssey.core.keys import policy_key
from spaceodyssey.core.ledger import (
    AssetTable,
    ChannelTable,
    _resource_rows,
    charge_parts,
)
from spaceodyssey.core.lift import lifting_jit
from spaceodyssey.core.snapshot import (
    award_columns,
    snapshot_fingerprint,
    snapshot_view,
    start_columns,
)
from spaceodyssey.core.study import _EXACT_QUANTA


def active(c):
    """Every write in `advance`, `admit` and `close_decision` is gated by this.

    A run waiting for the host to answer a request, or to install a refresh
    checkpoint, does nothing until it does.
    """
    return (
        ~c.done
        & ~c.stalled
        & ~c.overflow
        & ~c.idle
        & ~c.request.valid
        & ~c.refresh.valid
    )


def wake_by_audience(carry, ev):
    """Wake only when the event is visible to the policy audience."""
    return (
        carry.woke
        & ~carry.done
        & ~carry.draining
        & ((ev.audience & Audience.POLICY) != 0)
    )


def decision_due(carry, caps):
    """Whether no queued row is left at this tick ahead of the decide stage."""
    limit = order_key(carry.tick, Stage.DECIDE, 0, 0, caps, 0)
    ev = carry.ev
    keys = order_key(ev.tick, ev.stage, ev.micro, ev.sid, caps, ev.tie)
    return ~jnp.any(ev.valid & (keys < limit))


def advance(carry, root, study, truth, *, wake_fn=wake_by_audience):
    """Select the next event, handle it, consume it. Returns (carry, trace, wake).

    Wakes coalesce per tick. An event that wakes the policy marks the wake pending; the
    decision fires once, on the event after which no queued row remains at or before the
    current tick ahead of the decide stage, so every record released at that tick is
    in the decision's view. Rows the decision itself enqueues run after it and may wake
    a further decision at the next microstep, bounded by the microstep guard.
    """
    caps = study.caps
    act = active(carry)
    c1, ev, has = select_next(carry, caps)
    handled = act & has
    c2, emit, rec = step(c1, ev, study, truth, root)
    c3 = where_tree(handled, c2, carry)
    drainable = c3.ev.valid & (
        (c3.ev.kind == EventKind.DONE)
        | (c3.ev.kind == EventKind.RELEASE)
        | (c3.ev.kind == EventKind.REQUEST)
        | (c3.ev.kind == EventKind.RESULT)
        | (c3.ev.kind == EventKind.REFRESH)
    )
    if caps.commitments:
        # a commitment still to start is work in flight: a drain waits for its start
        drainable = drainable | (c3.ev.valid & (c3.ev.kind == EventKind.COMMIT_START))
    drained = (
        handled & c3.draining & ~drainable.any() & ~c3.request.valid & ~c3.refresh.valid
    )
    idle = carry.idle | (act & ~has)
    c4 = _tree_set(c3, idle=idle, done=c3.done | drained | idle)
    raised = handled & wake_fn(c4, ev)
    pending = c4.pending_wake | raised
    wake = handled & pending & decision_due(c4, caps) & ~c4.done & ~c4.draining
    c5 = _tree_set(c4, woke=jnp.asarray(False), pending_wake=pending & ~wake)
    z = jnp.asarray(0, I64)

    def g(x):
        return jnp.where(handled, x, z)

    belief_mean, belief_aux = study.observer.layout(caps).project(c5.belief)

    trace = dataclasses.replace(
        empty_trace(caps),
        handled=handled,
        kind=g(ev.kind),
        tick=g(ev.tick),
        stage=g(ev.stage),
        micro=g(ev.micro),
        tie=g(ev.tie),
        sid=g(ev.sid),
        lane=g(ev.lane),
        channel=g(ev.channel),
        subject=g(ev.subject),
        acquisition_index=g(ev.acquisition_index),
        log_slot=g(ev.slot),
        audience=g(ev.audience),
        n_written=g(emit.n_written),
        rec_status=g(rec.status),
        wake=wake,
        pending_wake=c5.pending_wake,
        n_dec=c5.n_dec,
        event_fp=g(ev.snapshot_fp),
        release_version=c5.release_version,
        belief_mean=belief_mean,
        belief_aux=belief_aux,
        n_log=c5.n_log,
        program=None if ev.program is None else g(ev.program),
    )
    if caps.snapshot_balances:
        # a request refused as stale was not held: the row records the refusal
        stale = handled & (ev.kind == EventKind.REQUEST) & ~c5.request.valid
        code = jnp.where(stale, AdmissionCode.QUERY_STALE, trace.admit_code[0])
        trace = dataclasses.replace(
            trace, admit_code=trace.admit_code.at[0].set(code.astype(I64))
        )
    return c5, trace, wake


def phase_at(carry, study):
    """Select the phase after accounting for elapsed boundaries."""
    bounds = study.agenda.phase_until[:-1]
    if bounds.shape[0] == 0:
        return carry.phase
    passed = jnp.sum(carry.tick >= jnp.asarray(bounds, I64)).astype(I64)
    return jnp.maximum(carry.phase, passed)


def decision_ordinal(carry):
    """Decisions already made at the current tick."""
    return jnp.where(carry.dec_tick == carry.tick, carry.dec_ordinal, 0)


def decision_key(carry, root, study):
    """The policy key of the decision about to run at the current tick."""
    return policy_key(
        root,
        phase_at(carry, study),
        carry.tick,
        decision_ordinal(carry),
        coupling=study.coupling,
    )


def _policy_view(carry, study):
    return eqx.filter_jit(snapshot_view)(carry, Audience.POLICY, study.caps)


def decide(carry, root, study, view=None):
    """Run the policy of the current phase on the policy-audience view.

    `view` is that view of `carry` when the caller already built it.
    """
    view = _policy_view(carry, study) if view is None else view
    ph = phase_at(carry, study)
    pstate = jnp.where(ph != carry.phase, jnp.zeros_like(carry.pstate), carry.pstate)
    key = decision_key(carry, root, study)

    def run_policy(ps, v, k, p):
        out = p.decide(ps, v, study, k)
        # checked per policy, before the switch asks its branches to agree
        _check_rows(out[0], study.caps, f" (policy: {p})")
        return out

    branches = tuple(
        (lambda ps, v, k, p=p: run_policy(ps, v, k, p)) for p in study.agenda.policies
    )
    if len(branches) == 1:
        return branches[0](pstate, view, key)
    return lax.switch(
        jnp.clip(ph, 0, len(branches) - 1).astype(jnp.int32),
        branches,
        pstate,
        view,
        key,
    )


def _policy_fingerprint(view, study):
    return snapshot_fingerprint(view, study.observer.layout(study.caps))


def decision_fingerprint(carry, study, view=None):
    """Fingerprint of the snapshot the policy reads at the current state.

    `view` is the policy-audience view of `carry` when the caller already built it.
    """
    view = _policy_view(carry, study) if view is None else view
    return lifting_jit(_policy_fingerprint)(view, study)


def _with_award_columns(effects, caps):
    """The effect table with its Allocate columns, zeroed where a producer left none.

    A column of the wrong shape (an amount that is not one entry per effect row and
    resource would broadcast into every resource) is replaced by zeros and the
    recipient of every row by -1, so each Allocate of the decision is refused as
    unsupported. The shapes are static, so the check costs nothing at run time.
    """
    held = {
        "amount": effects.amount,
        "window_from": effects.window_from,
        "window_until": effects.window_until,
        "recipient": effects.recipient,
        "remainder": effects.remainder,
        "realloc_to": effects.realloc_to,
    }
    cols = award_columns(caps)
    bad = {k: v is not None and jnp.shape(v) != cols[k].shape for k, v in held.items()}
    out = {k: v for k, v in cols.items() if held[k] is None or bad[k]}
    if any(bad.values()):
        out["recipient"] = jnp.full(caps.n_effects, -1, I64)
    return dataclasses.replace(effects, **out)


def _with_start_column(effects, caps):
    """The effect table with a start-tick column, zero (immediate) if none was given."""
    if effects.start_tick is not None:
        return effects
    return dataclasses.replace(effects, **start_columns(caps))


def _check_rows(effects, caps, source=""):
    """Refuse an effect table whose row count is not `caps.n_effects`.

    The shapes are static, so the check costs nothing at run time. `source` names the
    producer, when the caller knows it.
    """
    columns = (
        ("kind", effects.kind),
        ("subject", effects.subject),
        ("channel_index", effects.channel_index),
        ("group", effects.group),
        ("tick", effects.tick),
        ("lane", effects.lane),
        ("scope", effects.scope),
        ("disposition", effects.disposition),
        ("disruptive", effects.disruptive),
        ("program", effects.program),
        ("start_tick", effects.start_tick),
    )
    for name, column in columns:
        if column is None:
            continue
        rows = jnp.shape(column)
        if rows != (caps.n_effects,):
            raise ValueError(
                f"effect table column {name!r} has shape {rows}, but "
                f"Capacities.n_effects is {caps.n_effects}{source}: a policy must "
                "emit exactly n_effects rows"
            )


def mask_effects(effects, wake):
    """Disable effect rows for an unwoken decision."""
    return dataclasses.replace(
        effects, kind=jnp.where(wake, effects.kind, EffectKind.NONE)
    )


def _phase_mask(c, study):
    """Bitmask of the effects the current phase may ask for, its level included."""
    masks = jnp.asarray(study.agenda.effective_effects(), I64)
    return masks[jnp.clip(c.phase, 0, masks.shape[0] - 1)]


def _permitted(mask, kind):
    return ((mask >> kind) & 1) == 1


def _program_of(eff, e, caps):
    """Program of effect row e: its column with several programs, else the one.

    A producer that leaves the column `None` names no owner; the row is program 0.
    """
    if caps.n_programs > 1 and eff.program is not None:
        return eff.program[e]
    return jnp.asarray(0, I64)


def _start_of(eff, e, caps, t):
    """The tick effect row e starts at: its start tick, never before the decision's."""
    if not caps.commitments:
        return t
    return jnp.maximum(eff.start_tick[e], t)


def _frozen(c, study, lane, start, disruptive, owed=None):
    """Whether a start on `lane` is a frozen insertion and whether the row may break it.

    An asset with a horizon freezes every start within `freeze_ticks` of now. A row
    breaks the freeze when it is disruptive and the asset's disruptive resource (a
    resource column of the asset's own balance row) holds an activation beyond what the
    row's own charge takes from that cell (`owed`, the row's charge per resource, when
    the row charges the asset row).
    """
    assets = study.assets
    asset = assets.lane_asset[lane]
    horizon = assets.freeze_ticks[asset]
    inserts = (horizon > 0) & (start - c.tick < horizon)
    resource = assets.disruptive_resource[asset]
    column = jnp.clip(resource, 0, study.caps.n_resources - 1)
    held = c.ledger.balance[asset, column]
    if owed is not None:
        held = held - jnp.where(owed[1][column] == asset, owed[0][column], 0.0)
    return inserts, disruptive & (resource >= 0) & (held >= 1.0)


def _debit_activation(c, study, lane, spend):
    """Debit one disruptive activation of the lane's asset, where `spend`."""
    assets = study.assets
    asset = assets.lane_asset[lane]
    resource = jnp.clip(
        assets.disruptive_resource[asset], 0, study.caps.n_resources - 1
    )
    one = jnp.where(spend, 1.0, 0.0)
    return _tree_set(
        c,
        **{
            "ledger.balance": c.ledger.balance.at[asset, resource].add(-one),
            "ledger.debited": c.ledger.debited.at[asset, resource].add(one),
            "ledger.disruptive_used": c.ledger.disruptive_used.at[asset].add(
                spend.astype(I64)
            ),
        },
    )


def _commit_observe(c, eff, e, ok, study):
    """Debit, occupy the lane and queue the settlement of effect row e where `ok`.

    A row that starts later reserves the lane instead: it takes a commitment slot and
    queues the event that starts it.
    """
    caps, channels = study.caps, study.channels
    t = c.tick
    channel_index = eff.channel_index[e]
    lane = channels.lane[channel_index]
    start = _start_of(eff, e, caps, t)
    done_t = (
        start
        + study.assets.transition_ticks[lane]
        + channels.occupancy_ticks[channel_index]
    )
    future = ok & (start > t) if caps.commitments else jnp.asarray(False)
    ok_now = ok & ~future if caps.commitments else ok
    program = _program_of(eff, e, caps)
    rows = _resource_rows(study, lane, program)
    cols = jnp.arange(caps.n_resources)
    gen = c.ledger.lane_gen[lane] + ok_now.astype(I64)
    subject = eff.subject[e]
    acquisition_index = c.acquisition_counts[subject]
    c1 = _tree_set(
        c,
        **{
            "ledger.balance": c.ledger.balance.at[rows, cols].add(
                jnp.where(ok, -charge_parts(study, channel_index)[0], 0.0)
            ),
            "ledger.debited": c.ledger.debited.at[rows, cols].add(
                jnp.where(ok, charge_parts(study, channel_index)[0], 0.0)
            ),
            "ledger.lane_busy_until": c.ledger.lane_busy_until.at[lane].set(
                jnp.where(ok_now, done_t, c.ledger.lane_busy_until[lane])
            ),
            "ledger.lane_gen": c.ledger.lane_gen.at[lane].set(gen),
            "acquisition_counts": c.acquisition_counts.at[subject].add(ok.astype(I64)),
        },
    )
    if caps.commitments:
        inserts, _ = _frozen(c, study, lane, start, eff.disruptive[e])
        c1 = _debit_activation(c1, study, lane, ok & inserts)
        led = c1.ledger
        slot = jnp.argmax(led.cm_state == CommitState.NONE)

        def put(column, value):
            return column.at[slot].set(jnp.where(future, value, column[slot]))

        c1 = _tree_set(
            c1,
            **{
                "ledger.cm_state": put(led.cm_state, CommitState.PENDING),
                "ledger.cm_lane": put(led.cm_lane, lane),
                "ledger.cm_program": put(led.cm_program, program),
                "ledger.cm_start": put(led.cm_start, start),
                "ledger.cm_end": put(led.cm_end, done_t),
                "ledger.cm_index": put(led.cm_index, led.cm_count[program, lane]),
                "ledger.cm_count": led.cm_count.at[program, lane].add(
                    future.astype(I64)
                ),
            },
        )
        c1 = enqueue(
            c1,
            future,
            kind=EventKind.COMMIT_START,
            tick=start,
            stage=Stage.RELEASE,
            tie=TIE_NEUTRAL,
            lane=lane,
            channel_index=channel_index,
            channel=channels.id[channel_index],
            subject=subject,
            acquisition_index=acquisition_index,
            group=eff.group[e],
            audience=channels.audience[channel_index],
            slot=slot,
            scope=c.phase,
            acquisition_id=c.next_sid,
            program=program,
            caps=caps,
        )
    if caps.awards:
        # the generation, at debit, of each row that pays one resource of the charge,
        # with the rule and destination of its window; a debit outside a window has
        # generation 0 and the rule NONE
        led = c.ledger
        windowed = led.win_state[rows] == WindowState.WINDOWED
        captured = {
            "ledger.lane_award_gens": (
                c1.ledger.lane_award_gens,
                jnp.where(windowed, led.win_gen[rows], 0),
            ),
            "ledger.lane_award_rules": (
                c1.ledger.lane_award_rules,
                jnp.where(windowed, led.remainder_rule[rows], RemainderRule.NONE),
            ),
            "ledger.lane_award_tos": (
                c1.ledger.lane_award_tos,
                jnp.where(windowed, led.realloc_row[rows], 0),
            ),
        }
        c1 = _tree_set(
            c1,
            **{
                name: old.at[lane].set(jnp.where(ok, new, old[lane]))
                for name, (old, new) in captured.items()
            },
        )
    return enqueue(
        c1,
        ok_now,
        kind=EventKind.DONE,
        tick=done_t,
        stage=Stage.SETTLE,
        lane=lane,
        channel_index=channel_index,
        channel=channels.id[channel_index],
        subject=subject,
        acquisition_index=acquisition_index,
        gen=gen,
        group=eff.group[e],
        audience=channels.audience[channel_index],
        scope=c.phase,
        acquisition_id=c.next_sid,
        program=program,
        caps=caps,
    )


def _commit_allocate(c, eff, e, want, study):
    """Admit an Allocate row: grant the amount, open the window, queue its expiry.

    Returns the carry and the admission code. A row is malformed (UNSUPPORTED) when its
    recipient or reallocation row is out of range, its amount is negative or not
    finite, its window is empty, already over or past the last tick, or its rule is
    unknown; an award onto a row that already holds a window is FORBIDDEN (one window
    per row), and a full queue is NO_SLOT. While the balance view is on, an award that
    would take a resource's supply (budget plus grants) past 2**53 quanta is unsupported
    too.
    """
    caps = study.caps
    rows = caps.n_balance_rows
    t = c.tick
    r = jnp.clip(eff.recipient[e], 0, rows - 1)
    start, until = eff.window_from[e], eff.window_until[e]
    rule, to = eff.remainder[e], eff.realloc_to[e]
    amount = eff.amount[e]
    fine = (
        (eff.recipient[e] >= 0)
        & (eff.recipient[e] < rows)
        & jnp.all(jnp.isfinite(amount) & (amount >= 0.0))
        & (start >= 0)
        & (start < until)
        & (until > t)
        & (until <= caps.max_tick)
        & (rule >= RemainderRule.NONE)
        & (rule <= RemainderRule.REALLOCATE)
        & ((rule != RemainderRule.REALLOCATE) | ((to >= 0) & (to < rows) & (to != r)))
    )
    if caps.snapshot_balances:
        # The balance view counts `balance_quantum` units exactly up to 2**53 (the bound
        # `Study` holds the budget to). With every budget entry non-negative, rows never
        # go negative, and a debit, a refund, a lapse and a REALLOCATE all move quantity
        # without adding any; only the budget and the awards add. So a resource's budget
        # over all rows plus everything granted so far (a reallocation counts again,
        # which only refuses early) bounds every row at every time, whichever refund or
        # reallocation lands where. A study with a negative budget entry is refused
        # (UNSUPPORTED) rather than bounded: a negative row could be reallocated to
        # create headroom the sums do not see. `granted` is summed by absolute value
        # all the same, so a signed transfer can never lower the supply.
        budget = jnp.asarray(study.budget)
        supply = jnp.abs(budget).sum(axis=0) + jnp.abs(c.ledger.granted).sum(axis=0)
        fine = fine & jnp.all(budget >= 0.0)
        fine = fine & jnp.all((supply + amount) / caps.balance_quantum <= _EXACT_QUANTA)
    open_ = c.ledger.win_state[r] == WindowState.WINDOWED
    room = (~c.ev.valid).any() & (c.next_sid < caps.max_sid)
    code = jnp.select(
        [~fine, open_, ~room],
        [AdmissionCode.UNSUPPORTED, AdmissionCode.FORBIDDEN, AdmissionCode.NO_SLOT],
        AdmissionCode.OK,
    ).astype(I64)
    ok = want & (code == AdmissionCode.OK)
    gen = c.ledger.win_gen[r] + 1
    c1 = _tree_set(
        c,
        **{
            "ledger.balance": c.ledger.balance.at[r].add(jnp.where(ok, amount, 0.0)),
            "ledger.granted": c.ledger.granted.at[r].add(jnp.where(ok, amount, 0.0)),
            "ledger.win_state": c.ledger.win_state.at[r].set(
                jnp.where(ok, WindowState.WINDOWED, c.ledger.win_state[r])
            ),
            "ledger.win_gen": c.ledger.win_gen.at[r].set(
                jnp.where(ok, gen, c.ledger.win_gen[r])
            ),
            "ledger.valid_from_tick": c.ledger.valid_from_tick.at[r].set(
                jnp.where(ok, start, c.ledger.valid_from_tick[r])
            ),
            "ledger.valid_until_tick": c.ledger.valid_until_tick.at[r].set(
                jnp.where(ok, until, c.ledger.valid_until_tick[r])
            ),
            "ledger.remainder_rule": c.ledger.remainder_rule.at[r].set(
                jnp.where(ok, rule, c.ledger.remainder_rule[r])
            ),
            "ledger.realloc_row": c.ledger.realloc_row.at[r].set(
                jnp.where(ok, to, c.ledger.realloc_row[r])
            ),
        },
    )
    c2 = enqueue(
        c1,
        ok,
        kind=EventKind.WINDOW_END,
        tick=until,
        stage=Stage.ANALYZE,
        tie=TIE_NEUTRAL,
        lane=r,
        gen=gen,
        audience=PUBLIC,
        caps=caps,
    )
    return c2, jnp.where(want, code, AdmissionCode.NONE)


def _group_code(c, eff, member, study):
    """Admission result of the rows in `member`, judged as one group.

    The group is admitted on its summed demand per resource row, on lane availability
    (two rows on one lane cannot both run) and on queue room for every row. The result
    is the first refusal in row order, or OK when every row passes.
    """
    caps, channels = study.caps, study.channels
    n = caps.n_effects
    t = c.tick
    idx = jnp.arange(n)

    def inputs(f):
        lane = channels.lane[eff.channel_index[f]]
        program = _program_of(eff, f, caps)
        rows = _resource_rows(study, lane, program)
        cost = charge_parts(study, eff.channel_index[f])[0]
        return lane, program, rows, cost

    lanes, programs, rows, cost = jax.vmap(inputs)(idx)
    n_members = member.astype(I64).sum()
    room = ((~c.ev.valid).sum() >= n_members) & (c.next_sid + n_members <= caps.max_sid)
    if caps.commitments:
        later = member & (jax.vmap(lambda f: _start_of(eff, f, caps, t))(idx) > t)
        room = room & ((c.ledger.cm_state == CommitState.NONE).sum() >= later.sum())
    cols = jnp.arange(caps.n_resources)
    if caps.commitments:
        starts = jax.vmap(lambda f: _start_of(eff, f, caps, t))(idx)
        assets_of = study.assets.lane_asset[lanes].astype(I64)
        dcols = jnp.clip(
            study.assets.disruptive_resource[assets_of], 0, caps.n_resources - 1
        )
        # a row that starts inside its asset's horizon spends one activation
        spends = member & jax.vmap(
            lambda f: _frozen(c, study, lanes[f], starts[f], eff.disruptive[f])[0]
        )(idx)

    def code_of(f):
        ci = eff.channel_index[f]
        start = _start_of(eff, f, caps, t)
        done_t = (
            start
            + study.assets.transition_ticks[lanes[f]]
            + channels.occupancy_ticks[ci]
        )
        clash = jnp.any(member & (idx < f) & (lanes == lanes[f]))

        def add(g, demand):
            return demand + jnp.where(member[g] & (rows[g] == rows[f]), cost[g], 0.0)

        # rows are added in index order, one at a time, as the floating-point
        # result depends on the order
        demand = lax.fori_loop(0, n, add, jnp.zeros(caps.n_resources))
        funds = jnp.all(c.ledger.balance[rows[f], cols] >= demand)
        if caps.commitments:
            # the activations of the group come out of the asset row's cell as well, on
            # top of whatever the group's charges take from that cell; a row that
            # spends one needs the whole demand on the cell covered
            def cell(g, total):
                same = (assets_of[g] == assets_of[f]) & (dcols[g] == dcols[f])
                paid = member[g] & (rows[g][dcols[f]] == assets_of[f])
                return (
                    total
                    + jnp.where(spends[g] & same, 1.0, 0.0)
                    + jnp.where(paid, cost[g][dcols[f]], 0.0)
                )

            need = lax.fori_loop(0, n, cell, jnp.asarray(0.0))
            held = c.ledger.balance[assets_of[f], dcols[f]]
            funds = funds & (~spends[f] | (held >= need))
        closed = c.ledger.lane_closed[lanes[f]] | c.draining
        if caps.n_programs > 1:
            # a closed program refuses its own admissions; its lanes stay open
            closed = closed | c.ledger.program_closed[programs[f]]
        conds = [
            closed,
            (c.ledger.lane_busy_until[lanes[f]] > start) | clash,
            done_t > study.horizon_tick,
            ~funds,
            ~room,
        ]
        codes_ = [
            AdmissionCode.LANE_CLOSED,
            AdmissionCode.LANE_BUSY,
            AdmissionCode.HORIZON,
            AdmissionCode.NO_FUNDS,
            AdmissionCode.NO_SLOT,
        ]
        if caps.commitments:
            # a start may not overlap a reservation of the lane
            led = c.ledger
            conds[1] = conds[1] | jnp.any(
                (led.cm_state == CommitState.PENDING)
                & (led.cm_lane == lanes[f])
                & (led.cm_start < done_t)
                & (led.cm_end > start)
            )
            inserts, may_break = _frozen(
                c, study, lanes[f], start, eff.disruptive[f], (cost[f], rows[f])
            )
            conds.insert(3, inserts & ~may_break)
            codes_.insert(3, AdmissionCode.FROZEN_HORIZON)
        if caps.awards:
            # a payer row that is not ALWAYS admits only inside its window
            payer = rows[f]
            outside = jnp.any(
                (cost[f] > 0.0)
                & (c.ledger.win_state[payer] != WindowState.ALWAYS)
                & (
                    (t < c.ledger.valid_from_tick[payer])
                    | (t >= c.ledger.valid_until_tick[payer])
                )
            )
            conds.insert(3, outside)
            codes_.insert(3, AdmissionCode.OUTSIDE_VALIDITY)
        code = jnp.select(conds, codes_, AdmissionCode.OK).astype(I64)
        return jnp.where(member[f], code, AdmissionCode.OK)

    codes = jax.vmap(code_of)(idx)
    refused = codes != AdmissionCode.OK
    return jnp.where(
        refused.any(), codes[jnp.argmax(refused)], AdmissionCode.OK
    ).astype(I64)


class _AdmissionInputs(eqx.Module):
    """Admission inputs independent of policy and model implementations."""

    caps: Capacities = eqx.field(static=True)
    assets: AssetTable
    channels: ChannelTable
    horizon_tick: jax.Array
    # the study's initial balances, read by the award range bound only
    budget: jax.Array | None = None


@eqx.filter_jit
def _admit_rows(c, eff, mask, study, fp):
    """Admit the rows of one decision, Observe rows of one group in or out whole."""
    caps = study.caps
    first_sid = c.next_sid
    n = caps.n_effects
    out = jnp.full(n, AdmissionCode.NONE, I64)
    decided = jnp.zeros(n, jnp.bool_)
    idx = jnp.arange(n)

    def row(e, state):
        c, out, decided = state
        act = active(c)
        kind = eff.kind[e]
        allowed = _permitted(mask, kind)
        obs = act & (kind == EffectKind.OBSERVE) & allowed
        # the first undecided row of a group admits every row of that group
        start = obs & ~decided[e]
        member = (
            start
            & (idx >= e)
            & (eff.kind == EffectKind.OBSERVE)
            & _permitted(mask, eff.kind)
            & (eff.group == eff.group[e])
        )
        gcode = eqx.filter_jit(_group_code)(c, eff, member, study)
        ok = gcode == AdmissionCode.OK

        def commit(f, c):
            return eqx.filter_jit(_commit_observe)(
                c, eff, f.astype(I64), member[f] & ok, study
            )

        c = lax.fori_loop(0, n, commit, c)
        out = jnp.where(member, gcode, out)
        decided = decided | member
        t = c.tick
        w = eff.tick[e]
        c = enqueue(
            c,
            act & allowed & (kind == EffectKind.WAIT),
            kind=EventKind.WAIT,
            tick=jnp.maximum(w, t),
            stage=jnp.where(w > t, Stage.RELEASE, Stage.ADMIT),
            group=eff.group[e],
            audience=PUBLIC,
            caps=caps,
        )
        c = enqueue(
            c,
            act & allowed & (kind == EffectKind.STOP),
            kind=EventKind.STOP,
            tick=t,
            stage=Stage.ADMIT,
            lane=eff.lane[e],
            group=eff.group[e],
            audience=PUBLIC,
            scope=eff.scope[e],
            disposition=eff.disposition[e],
            program=_program_of(eff, e, caps),
            caps=caps,
        )
        cancels = act & allowed & (kind == EffectKind.CANCEL)
        cancel_extra = {}
        cancel_code = AdmissionCode.OK
        if caps.commitments:
            # a Cancel that starts after now names the pending commitment of its lane
            # that starts then; inside the horizon it is refused like an insertion
            named = eff.start_tick[e] > t
            hit = (
                (c.ledger.cm_state == CommitState.PENDING)
                & (c.ledger.cm_lane == eff.lane[e])
                & (c.ledger.cm_start == eff.start_tick[e])
            )
            slot = jnp.argmax(hit)
            # a commitment already named by a Cancel waiting in the queue has no
            # second cancellation to give
            # nor does one that an abort already queued for its scope will cancel
            ev = c.ev
            owner = c.ledger.cm_program[slot] if caps.n_programs > 1 else 0
            stop_program = ev.program if ev.program is not None else 0
            abort_first = (
                (ev.kind == EventKind.STOP)
                & (ev.disposition == Disposition.ABORT)
                & stop_reaches(
                    study, ev.scope, ev.lane, stop_program, eff.lane[e], owner
                )
            )
            again = jnp.any(
                ev.valid
                & (((ev.kind == EventKind.CANCEL) & (ev.slot == slot)) | abort_first)
            )
            inserts, may_break = _frozen(
                c, study, eff.lane[e], eff.start_tick[e], eff.disruptive[e]
            )
            queued = (~c.ev.valid).any() & (c.next_sid < caps.max_sid)
            cancel_code = jnp.select(
                [
                    named & (~hit.any() | again),
                    named & inserts & ~may_break,
                    named & ~queued,
                ],
                [
                    AdmissionCode.UNSUPPORTED,
                    AdmissionCode.FROZEN_HORIZON,
                    AdmissionCode.NO_SLOT,
                ],
                AdmissionCode.OK,
            ).astype(I64)
            cancels = cancels & (cancel_code == AdmissionCode.OK)
            # the activation is spent only by a Cancel that is queued
            c = _debit_activation(c, study, eff.lane[e], cancels & named & inserts)
            cancel_extra = {"slot": jnp.where(named, slot, -1)}
        c = enqueue(
            c,
            cancels,
            kind=EventKind.CANCEL,
            tick=t,
            stage=Stage.ADMIT,
            lane=eff.lane[e],
            gen=c.ledger.lane_gen[eff.lane[e]] + eff.tick[e],
            group=eff.group[e],
            audience=PUBLIC,
            **cancel_extra,
            caps=caps,
        )
        asked = act & allowed & (kind == EffectKind.QUERY)
        outstanding = c.request.valid | jnp.any(
            c.ev.valid
            & ((c.ev.kind == EventKind.REQUEST) | (c.ev.kind == EventKind.RESULT))
        )
        room = (~c.ev.valid).any() & (c.next_sid < caps.max_sid) & ~outstanding
        c = enqueue(
            c,
            asked & room,
            kind=EventKind.REQUEST,
            tick=t,
            stage=Stage.ADMIT,
            subject=eff.subject[e],
            channel_index=eff.channel_index[e],
            group=eff.group[e],
            audience=Audience.POLICY,
            snapshot_fp=fp,
            version=c.release_version[version_slot(Audience.POLICY)],
            decision=c.n_dec,
            caps=caps,
        )
        alloc_code = AdmissionCode.UNSUPPORTED
        if caps.awards:
            c, alloc_code = _commit_allocate(
                c, eff, e, act & allowed & (kind == EffectKind.ALLOCATE), study
            )
        conds = [
            ~act | (kind == EffectKind.NONE),
            ~allowed,
            obs,
            kind == EffectKind.ALLOCATE,
            (kind == EffectKind.QUERY) & ~room,
        ]
        codes = [
            AdmissionCode.NONE,
            AdmissionCode.FORBIDDEN,
            out[e],
            alloc_code,
            AdmissionCode.NO_SLOT,
        ]
        if caps.commitments:
            conds.append(kind == EffectKind.CANCEL)
            codes.append(cancel_code)
        code_e = jnp.select(conds, codes, AdmissionCode.OK).astype(I64)
        return c, out.at[e].set(code_e), decided

    c, out, decided = lax.fori_loop(0, n, row, (c, out, decided))
    # A question is answered before any sibling event at this position runs,
    # including siblings listed before the Query in the effect table.
    ev = c.ev
    added = ev.valid & (ev.sid >= first_sid)
    asks = added & (ev.kind == EventKind.REQUEST)
    siblings = (
        added
        & (ev.kind != EventKind.REQUEST)
        & (ev.tick == c.tick)
        & (ev.stage == c.stage)
    )
    c = _tree_set(c, **{"ev.micro": ev.micro + jnp.where(asks.any() & siblings, 2, 0)})
    return c, out


def _refuses_mixed(caps):
    """Whether a decision that mixes a Query with another effect is refused."""
    return caps.snapshot_balances


def _query_mixed_rows(effects, mask, act):
    """Rows of a decision refused for mixing a Query with another effect.

    The rows are the permitted, non-empty ones of a decision that holds a Query and at
    least one other such row; none otherwise. A row the phase mask forbids is not an
    effect of the decision and keeps its own code.
    """
    live = _permitted(mask, effects.kind) & (effects.kind != EffectKind.NONE)
    query = live & (effects.kind == EffectKind.QUERY)
    mixed = act & query.any() & (live & ~query).any()
    return live & mixed


def admit(carry, effects, study, snapshot_fp=None):
    """Admit a decision's effect rows. Returns (carry, admission code per row).

    Rows run in order, each against the balance left by the rows before it, except that
    Observe rows sharing a `group` id are judged together: summed demand per resource
    row, lane availability and queue room for all of them, then every row is admitted or
    none is, and a refused group takes no acquisition index. An effect whose bit is
    clear in the phase's effective mask (`Agenda.effective_effects`: the phase mask
    and its level mask) is refused as forbidden and changes nothing.

    A Query is not charged. It queues a request row at the current tick in the admit
    stage, addressed to the policy audience and stamped with the fingerprint of the
    snapshot the decision read (`snapshot_fp`, computed here when not given) and the
    decision's index. An `Allocate` answers unsupported unless awards are on, where it
    grants its amount to the recipient row and opens a window (`_commit_allocate`).

    While `caps.snapshot_balances` is on, a decision that combines a Query with any
    other effect is refused whole, every such row with `AdmissionCode.QUERY_MIXED`,
    and changes nothing: the request carries the fingerprint of a snapshot that
    holds the balances, which the debits of a sibling row would move before the host
    answers. With the flag off the check does not exist.
    """
    _check_rows(effects, study.caps)
    fp = decision_fingerprint(carry, study) if snapshot_fp is None else snapshot_fp
    supported = (
        (effects.kind == EffectKind.OBSERVE)
        | (effects.kind == EffectKind.WAIT)
        | (effects.kind == EffectKind.STOP)
        | (effects.kind == EffectKind.CANCEL)
        | (effects.kind == EffectKind.QUERY)
    )
    if study.caps.awards:
        effects = _with_award_columns(effects, study.caps)
        supported = supported | (effects.kind == EffectKind.ALLOCATE)
    if study.caps.commitments:
        effects = _with_start_column(effects, study.caps)
    phase = phase_at(carry, study)
    mask = _phase_mask(_tree_set(carry, phase=phase), study)
    mixed = None
    if _refuses_mixed(study.caps):
        mixed = _query_mixed_rows(effects, mask, active(carry))
        supported = supported & ~mixed
        effects = dataclasses.replace(
            effects, kind=jnp.where(mixed, EffectKind.NONE, effects.kind)
        )
    any_eff = active(carry) & (supported & _permitted(mask, effects.kind)).any()
    c = _tree_set(carry, stage=jnp.where(any_eff, Stage.ADMIT, carry.stage))
    c = _tree_set(c, phase=jnp.where(any_eff, phase, c.phase))
    inputs = _AdmissionInputs(
        study.caps,
        study.assets,
        study.channels,
        study.horizon_tick,
        study.budget if study.caps.awards and study.caps.snapshot_balances else None,
    )
    c, codes = _admit_rows(c, effects, mask, inputs, fp)
    if mixed is not None:
        codes = jnp.where(mixed, AdmissionCode.QUERY_MIXED, codes).astype(I64)
    return c, codes


def close_decision(carry, effects, pstate, wake, study):
    """Commit a woken decision: policy state, decision count, stage; clear the wake."""
    w = wake & active(carry)
    return _tree_set(
        carry,
        pstate=jnp.where(w, pstate, carry.pstate),
        n_dec=carry.n_dec + w.astype(I64),
        dec_tick=jnp.where(w, carry.tick, carry.dec_tick),
        dec_ordinal=jnp.where(w, decision_ordinal(carry) + 1, carry.dec_ordinal),
        woke=jnp.asarray(False),
        stage=jnp.where(w, Stage.ADMIT, carry.stage),
        phase=jnp.where(w, phase_at(carry, study), carry.phase),
    )


def finish_trace(trace, effects, codes, u, wake, carry, study, snapshot_fp=0):
    """Attach decision outputs and final belief state to a trace row.

    The belief state is the observer layout's trace projection of the full belief.
    `snapshot_fp` is the fingerprint of the snapshot the decision read; it is recorded
    for a woken step and zero otherwise.
    """
    belief_mean, belief_aux = study.observer.layout(study.caps).project(carry.belief)
    if study.caps.snapshot_balances:
        # a step that decided carries its decision's codes; any other step keeps the
        # refusal `advance` wrote for a stale request
        codes = jnp.where(wake, codes, trace.admit_code)
    extra = {}
    if study.caps.awards:
        effects = _with_award_columns(effects, study.caps)
        ask = effects.kind == EffectKind.ALLOCATE
        extra = {
            "alloc_amount": jnp.where(ask[:, None], effects.amount, 0.0),
            "alloc_from": jnp.where(ask, effects.window_from, 0),
            "alloc_until": jnp.where(ask, effects.window_until, 0),
            "alloc_recipient": jnp.where(ask, effects.recipient, 0),
            "alloc_remainder": jnp.where(ask, effects.remainder, 0),
            "alloc_to": jnp.where(ask, effects.realloc_to, 0),
        }
    return dataclasses.replace(
        trace,
        **extra,
        n_dec=carry.n_dec,
        snapshot_fp=jnp.where(wake, jnp.asarray(snapshot_fp, I64), 0),
        eff_kind=effects.kind,
        eff_disruptive=effects.disruptive,
        admit_code=codes,
        pol_u=jnp.where(wake, u, 0.0),
        belief_mean=belief_mean,
        belief_aux=belief_aux,
        n_log=carry.n_log,
    )


def transition(carry, i, root, study, truth, *, wake_fn=wake_by_audience):
    """advance, then decide, admit and close under a mask on the wake."""
    c, trace, wake = advance(carry, root, study, truth, wake_fn=wake_fn)
    view = _policy_view(c, study)
    eff, pstate, u = decide(c, root, study, view)
    eff = mask_effects(eff, wake)
    fp = decision_fingerprint(c, study, view)
    c, codes = admit(c, eff, study, fp)
    c = close_decision(c, eff, pstate, wake, study)
    return c, finish_trace(trace, eff, codes, u, wake, c, study, fp)
