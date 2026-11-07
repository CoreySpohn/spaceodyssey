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
    AdmissionCode,
    Audience,
    EffectKind,
    EventKind,
    Stage,
    version_slot,
)
from spaceodyssey.core.events import enqueue, order_key, select_next, where_tree
from spaceodyssey.core.handlers import step
from spaceodyssey.core.keys import policy_key
from spaceodyssey.core.ledger import (
    AssetTable,
    ChannelTable,
    _resource_rows,
    charge_parts,
)
from spaceodyssey.core.snapshot import snapshot_fingerprint, snapshot_view


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


def decide(carry, root, study):
    """Run the policy of the current phase on the policy-audience view."""
    view = eqx.filter_jit(snapshot_view)(carry, Audience.POLICY, study.caps)
    ph = phase_at(carry, study)
    pstate = jnp.where(ph != carry.phase, jnp.zeros_like(carry.pstate), carry.pstate)
    key = decision_key(carry, root, study)
    branches = tuple(
        (lambda ps, v, k, p=p: p.decide(ps, v, study, k)) for p in study.agenda.policies
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


def decision_fingerprint(carry, study):
    """Fingerprint of the snapshot the policy reads at the current state."""
    return eqx.filter_jit(_policy_fingerprint)(
        eqx.filter_jit(snapshot_view)(carry, Audience.POLICY, study.caps), study
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


def _commit_observe(c, eff, e, ok, study):
    """Debit, occupy the lane and queue the settlement of effect row e where `ok`."""
    caps, channels = study.caps, study.channels
    t = c.tick
    channel_index = eff.channel_index[e]
    lane = channels.lane[channel_index]
    done_t = (
        t
        + study.assets.transition_ticks[lane]
        + channels.occupancy_ticks[channel_index]
    )
    program = _program_of(eff, e, caps)
    rows = _resource_rows(study, lane, program)
    cols = jnp.arange(caps.n_resources)
    gen = c.ledger.lane_gen[lane] + ok.astype(I64)
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
                jnp.where(ok, done_t, c.ledger.lane_busy_until[lane])
            ),
            "ledger.lane_gen": c.ledger.lane_gen.at[lane].set(gen),
            "acquisition_counts": c.acquisition_counts.at[subject].add(ok.astype(I64)),
        },
    )
    return enqueue(
        c1,
        ok,
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


def _group_code(c, eff, member, study):
    """Admission result of the rows in `member`, judged as one group.

    The group is admitted on its summed demand per resource row, on lane availability
    (two rows on one lane cannot both run) and on queue room for every row. The result
    is the first refusal in row order, or OK when every row passes.
    """
    caps, channels = study.caps, study.channels
    n = caps.n_effects
    t = c.tick
    lanes = [channels.lane[eff.channel_index[f]] for f in range(n)]
    programs = [_program_of(eff, f, caps) for f in range(n)]
    rows = [_resource_rows(study, lanes[f], programs[f]) for f in range(n)]
    cost = [charge_parts(study, eff.channel_index[f])[0] for f in range(n)]
    n_members = sum(member[f].astype(I64) for f in range(n))
    room = ((~c.ev.valid).sum() >= n_members) & (c.next_sid + n_members <= caps.max_sid)
    cols = jnp.arange(caps.n_resources)
    codes = []
    for f in range(n):
        ci = eff.channel_index[f]
        done_t = (
            t + study.assets.transition_ticks[lanes[f]] + channels.occupancy_ticks[ci]
        )
        clash = jnp.asarray(False)
        for g in range(f):
            clash = clash | (member[g] & (lanes[g] == lanes[f]))
        demand = jnp.zeros(caps.n_resources)
        for g in range(n):
            demand = demand + jnp.where(member[g] & (rows[g] == rows[f]), cost[g], 0.0)
        funds = jnp.all(c.ledger.balance[rows[f], cols] >= demand)
        closed = c.ledger.lane_closed[lanes[f]] | c.draining
        if caps.n_programs > 1:
            # a closed program refuses its own admissions; its lanes stay open
            closed = closed | c.ledger.program_closed[programs[f]]
        code = jnp.select(
            [
                closed,
                (c.ledger.lane_busy_until[lanes[f]] > t) | clash,
                done_t > study.horizon_tick,
                ~funds,
                ~room,
            ],
            [
                AdmissionCode.LANE_CLOSED,
                AdmissionCode.LANE_BUSY,
                AdmissionCode.HORIZON,
                AdmissionCode.NO_FUNDS,
                AdmissionCode.NO_SLOT,
            ],
            AdmissionCode.OK,
        ).astype(I64)
        codes.append(jnp.where(member[f], code, AdmissionCode.OK))
    first = jnp.asarray(AdmissionCode.OK, I64)
    for f in reversed(range(n)):
        first = jnp.where(codes[f] != AdmissionCode.OK, codes[f], first)
    return first


class _AdmissionInputs(eqx.Module):
    """Admission inputs independent of policy and model implementations."""

    caps: Capacities = eqx.field(static=True)
    assets: AssetTable
    channels: ChannelTable
    horizon_tick: jax.Array


@eqx.filter_jit
def _admit_rows(c, eff, mask, study, fp):
    """Admit the rows of one decision, Observe rows of one group in or out whole."""
    caps = study.caps
    first_sid = c.next_sid
    n = caps.n_effects
    out = [jnp.asarray(AdmissionCode.NONE, I64)] * n
    decided = jnp.zeros(n, jnp.bool_)
    for e in range(n):
        act = active(c)
        kind = eff.kind[e]
        allowed = _permitted(mask, kind)
        obs = act & (kind == EffectKind.OBSERVE) & allowed
        # the first undecided row of a group admits every row of that group
        start = obs & ~decided[e]
        member = jnp.stack(
            [
                start
                & (f >= e)
                & (eff.kind[f] == EffectKind.OBSERVE)
                & _permitted(mask, eff.kind[f])
                & (eff.group[f] == eff.group[e])
                for f in range(n)
            ]
        )
        gcode = eqx.filter_jit(_group_code)(c, eff, member, study)
        ok = gcode == AdmissionCode.OK
        for f in range(n):
            c = eqx.filter_jit(_commit_observe)(
                c, eff, jnp.asarray(f, I64), member[f] & ok, study
            )
            out[f] = jnp.where(member[f], gcode, out[f])
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
        c = enqueue(
            c,
            act & allowed & (kind == EffectKind.CANCEL),
            kind=EventKind.CANCEL,
            tick=t,
            stage=Stage.ADMIT,
            lane=eff.lane[e],
            gen=c.ledger.lane_gen[eff.lane[e]] + eff.tick[e],
            group=eff.group[e],
            audience=PUBLIC,
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
        out[e] = jnp.select(
            [
                ~act | (kind == EffectKind.NONE),
                ~allowed,
                obs,
                kind == EffectKind.ALLOCATE,
                (kind == EffectKind.QUERY) & ~room,
            ],
            [
                AdmissionCode.NONE,
                AdmissionCode.FORBIDDEN,
                out[e],
                AdmissionCode.UNSUPPORTED,
                AdmissionCode.NO_SLOT,
            ],
            AdmissionCode.OK,
        ).astype(I64)
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
    return c, jnp.stack(out)


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
    decision's index; `Allocate` is declared and answers unsupported.

    While `caps.snapshot_balances` is on, a decision that combines a Query with any
    other effect is refused whole, every such row with `AdmissionCode.QUERY_MIXED`,
    and changes nothing: the request carries the fingerprint of a snapshot that
    holds the balances, which the debits of a sibling row would move before the host
    answers. With the flag off the check does not exist.
    """
    fp = decision_fingerprint(carry, study) if snapshot_fp is None else snapshot_fp
    supported = (
        (effects.kind == EffectKind.OBSERVE)
        | (effects.kind == EffectKind.WAIT)
        | (effects.kind == EffectKind.STOP)
        | (effects.kind == EffectKind.CANCEL)
        | (effects.kind == EffectKind.QUERY)
    )
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
        study.caps, study.assets, study.channels, study.horizon_tick
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
    return dataclasses.replace(
        trace,
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
    eff, pstate, u = decide(c, root, study)
    eff = mask_effects(eff, wake)
    fp = decision_fingerprint(c, study)
    c, codes = admit(c, eff, study, fp)
    c = close_decision(c, eff, pstate, wake, study)
    return c, finish_trace(trace, eff, codes, u, wake, c, study, fp)
