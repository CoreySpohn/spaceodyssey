"""Transaction."""

import dataclasses

import jax.numpy as jnp
from jax import lax

from spaceodyssey.core.carry import _tree_set, empty_trace
from spaceodyssey.core.enums import (
    I64,
    PUBLIC,
    AdmissionCode,
    Audience,
    EffectKind,
    EventKind,
    Stage,
)
from spaceodyssey.core.events import enqueue, select_next, where_tree
from spaceodyssey.core.handlers import step
from spaceodyssey.core.keys import policy_key
from spaceodyssey.core.ledger import _resource_rows
from spaceodyssey.core.snapshot import snapshot_view


def active(c):
    """Every write in `advance`, `admit` and `close_decision` is gated by this."""
    return ~c.done & ~c.stalled & ~c.overflow & ~c.idle


def wake_by_audience(carry, ev):
    """Wake only when the event is visible to the policy audience."""
    return (
        carry.woke
        & ~carry.done
        & ~carry.draining
        & ((ev.audience & Audience.POLICY) != 0)
    )


def advance(carry, root, study, truth, *, wake_fn=wake_by_audience):
    """Select the next event, handle it, consume it. Returns (carry, trace, wake)."""
    caps = study.caps
    act = active(carry)
    c1, ev, has = select_next(carry, caps)
    handled = act & has
    c2, emit, rec = step(c1, ev, study, truth, root)
    c3 = where_tree(handled, c2, carry)
    drainable = c3.ev.valid & (
        (c3.ev.kind == EventKind.DONE) | (c3.ev.kind == EventKind.RELEASE)
    )
    drained = handled & c3.draining & ~drainable.any()
    idle = carry.idle | (act & ~has)
    c4 = _tree_set(c3, idle=idle, done=c3.done | drained | idle)
    wake = handled & wake_fn(c4, ev)
    c5 = _tree_set(c4, woke=wake)
    z = jnp.asarray(0, I64)

    def g(x):
        return jnp.where(handled, x, z)

    trace = dataclasses.replace(
        empty_trace(caps),
        handled=handled,
        kind=g(ev.kind),
        tick=g(ev.tick),
        stage=g(ev.stage),
        micro=g(ev.micro),
        sid=g(ev.sid),
        lane=g(ev.lane),
        channel=g(ev.channel),
        subject=g(ev.subject),
        acquisition_index=g(ev.acquisition_index),
        audience=g(ev.audience),
        n_written=g(emit.n_written),
        rec_status=g(rec.status),
        wake=wake,
        n_dec=c5.n_dec,
        belief_mean=c5.belief.mean,
        belief_aux=c5.belief.aux,
        n_log=c5.n_log,
    )
    return c5, trace, wake


def phase_at(carry, study):
    """Select the phase after accounting for elapsed boundaries."""
    bounds = study.agenda.phase_until[:-1]
    if not bounds:
        return carry.phase
    passed = jnp.sum(carry.tick >= jnp.asarray(bounds, I64)).astype(I64)
    return jnp.maximum(carry.phase, passed)


def decide(carry, root, study):
    """Run the policy of the current phase on the policy-audience view."""
    view = snapshot_view(carry, Audience.POLICY, study.caps)
    ph = phase_at(carry, study)
    pstate = jnp.where(ph != carry.phase, jnp.zeros_like(carry.pstate), carry.pstate)
    key = policy_key(root, carry.n_dec)
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


def mask_effects(effects, wake):
    """Disable effect rows for an unwoken decision."""
    return dataclasses.replace(
        effects, kind=jnp.where(wake, effects.kind, EffectKind.NONE)
    )


def _admit_row(c, eff, e, study):
    caps, channels = study.caps, study.channels
    act = active(c)
    kind = eff.kind[e]
    t = c.tick
    channel_index = eff.channel_index[e]
    lane = channels.lane[channel_index]
    done_t = (
        t
        + study.assets.transition_ticks[lane]
        + channels.occupancy_ticks[channel_index]
    )
    rows = _resource_rows(study, lane)
    cols = jnp.arange(caps.n_resources)
    funds = jnp.all(c.ledger.balance[rows, cols] >= channels.cost[channel_index])
    room = (~c.ev.valid).any() & (c.next_sid < caps.max_sid)
    code = jnp.select(
        [
            c.ledger.lane_closed[lane] | c.draining,
            c.ledger.lane_busy_until[lane] > t,
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
    obs = act & (kind == EffectKind.OBSERVE)
    ok = obs & (code == AdmissionCode.OK)
    gen = c.ledger.lane_gen[lane] + ok.astype(I64)
    subject = eff.subject[e]
    acquisition_index = c.acquisition_counts[subject]
    c1 = _tree_set(
        c,
        **{
            "ledger.balance": c.ledger.balance.at[rows, cols].add(
                jnp.where(ok, -channels.cost[channel_index], 0.0)
            ),
            "ledger.lane_busy_until": c.ledger.lane_busy_until.at[lane].set(
                jnp.where(ok, done_t, c.ledger.lane_busy_until[lane])
            ),
            "ledger.lane_gen": c.ledger.lane_gen.at[lane].set(gen),
            "acquisition_counts": c.acquisition_counts.at[subject].add(ok.astype(I64)),
        },
    )
    c1 = enqueue(
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
        caps=caps,
    )
    w = eff.tick[e]
    c1 = enqueue(
        c1,
        act & (kind == EffectKind.WAIT),
        kind=EventKind.WAIT,
        tick=jnp.maximum(w, t),
        stage=jnp.where(w > t, Stage.RELEASE, Stage.ADMIT),
        group=eff.group[e],
        audience=PUBLIC,
        caps=caps,
    )
    c1 = enqueue(
        c1,
        act & (kind == EffectKind.STOP),
        kind=EventKind.STOP,
        tick=t,
        stage=Stage.ADMIT,
        lane=eff.lane[e],
        group=eff.group[e],
        audience=PUBLIC,
        scope=eff.scope[e],
        disposition=eff.disposition[e],
        caps=caps,
    )
    c1 = enqueue(
        c1,
        act & (kind == EffectKind.CANCEL),
        kind=EventKind.CANCEL,
        tick=t,
        stage=Stage.ADMIT,
        lane=eff.lane[e],
        gen=c.ledger.lane_gen[eff.lane[e]] + eff.tick[e],
        group=eff.group[e],
        audience=PUBLIC,
        caps=caps,
    )
    reserved = (kind == EffectKind.ALLOCATE) | (kind == EffectKind.QUERY)
    out = jnp.select(
        [~act | (kind == EffectKind.NONE), obs, reserved],
        [AdmissionCode.NONE, code, AdmissionCode.UNSUPPORTED],
        AdmissionCode.OK,
    ).astype(I64)
    return c1, out


def admit(carry, effects, study):
    """Admit effect rows against the balance left by earlier rows.

    Admit a decision's effect rows in order, each against the balance left by the rows
    before it. Returns (carry, admission code per row).
    """
    caps = study.caps
    supported = (
        (effects.kind == EffectKind.OBSERVE)
        | (effects.kind == EffectKind.WAIT)
        | (effects.kind == EffectKind.STOP)
        | (effects.kind == EffectKind.CANCEL)
    )
    any_eff = active(carry) & supported.any()
    c = _tree_set(carry, stage=jnp.where(any_eff, Stage.ADMIT, carry.stage))
    c = _tree_set(c, phase=jnp.where(any_eff, phase_at(c, study), c.phase))
    codes = []
    for e in range(caps.n_effects):
        c, code = _admit_row(c, effects, e, study)
        codes.append(code)
    return c, jnp.stack(codes)


def close_decision(carry, effects, pstate, wake, study):
    """Commit a woken decision: policy state, decision count, stage; clear the wake."""
    w = wake & active(carry)
    return _tree_set(
        carry,
        pstate=jnp.where(w, pstate, carry.pstate),
        n_dec=carry.n_dec + w.astype(I64),
        woke=jnp.asarray(False),
        stage=jnp.where(w, Stage.ADMIT, carry.stage),
        phase=jnp.where(w, phase_at(carry, study), carry.phase),
    )


def finish_trace(trace, effects, codes, u, wake, carry):
    """Attach decision outputs and final belief state to a trace row."""
    return dataclasses.replace(
        trace,
        n_dec=carry.n_dec,
        eff_kind=effects.kind,
        admit_code=codes,
        pol_u=jnp.where(wake, u, 0.0),
        belief_mean=carry.belief.mean,
        belief_aux=carry.belief.aux,
        n_log=carry.n_log,
    )


def transition(carry, i, root, study, truth, *, wake_fn=wake_by_audience):
    """advance, then decide, admit and close under a mask on the wake."""
    c, trace, wake = advance(carry, root, study, truth, wake_fn=wake_fn)
    eff, pstate, u = decide(c, root, study)
    eff = mask_effects(eff, wake)
    c, codes = admit(c, eff, study)
    c = close_decision(c, eff, pstate, wake, study)
    return c, finish_trace(trace, eff, codes, u, wake, c)
