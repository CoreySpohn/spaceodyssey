"""Carry."""

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.enums import (
    F64,
    I64,
    PUBLIC,
    AdmissionCode,
    B,
    ChargeRule,
    EventKind,
    RemainderRule,
    Stage,
)
from spaceodyssey.core.events import (
    EventTable,
    empty_event_row,
    empty_events,
    enqueue,
)
from spaceodyssey.core.ledger import Ledger
from spaceodyssey.core.records import RecordLog, empty_log
from spaceodyssey.core.snapshot import MeanBelief


class Carry(eqx.Module):
    """Complete immutable state of a transaction run."""

    tick: jax.Array
    stage: jax.Array
    micro: jax.Array
    phase: jax.Array
    ev: EventTable
    log: RecordLog
    n_log: jax.Array
    ledger: Ledger
    belief: eqx.Module  # the observer's full belief (`BeliefLayout.belief`)
    pbelief: eqx.Module  # the policy summary (`BeliefLayout.summary`)
    acquisition_counts: jax.Array
    pstate: jax.Array
    woke: jax.Array
    pending_wake: jax.Array
    done: jax.Array
    draining: jax.Array
    stalled: jax.Array
    idle: jax.Array
    overflow: jax.Array
    overflow_mask: jax.Array
    violation: jax.Array
    double_settle: jax.Array
    dup_fold: jax.Array
    n_dec: jax.Array
    dec_tick: jax.Array
    dec_ordinal: jax.Array
    next_sid: jax.Array
    release_version: jax.Array  # I64[2]: releases received by (observer, policy)
    request: EventTable  # the host request being awaited; `valid` while unanswered
    refresh: EventTable  # the checkpoint held for the host; `valid` until installed
    refresh_dirty: jax.Array  # B[2, n_subjects]: (observer, policy) subjects to refit
    refresh_status: jax.Array  # I64[2, n_subjects]: `RefitStatus` of the last refit


def _zero_belief(caps):
    return MeanBelief(
        mean=jnp.zeros(caps.n_subjects, F64),
        n=jnp.zeros(caps.n_subjects, I64),
        aux=jnp.zeros(caps.n_an, F64),
    )


def empty_carry(caps, balance=None):
    """A zeroed carry at tick 0, stage Stage.SETTLE, with an empty queue and log.

    Both beliefs are zeroed `MeanBelief`s; `init_carry` replaces them with the
    observer's own types.
    """
    z = jnp.asarray(0, I64)
    f = jnp.asarray(False)
    br, nr = caps.n_balance_rows, caps.n_resources
    # the refresh leaves are zero-size unless the capacities carry the protocol
    n_refresh = caps.n_subjects if caps.rows_per_subject > 0 else 0
    return Carry(
        tick=z,
        stage=z,
        micro=z,
        phase=z,
        ev=empty_events(caps),
        log=empty_log(caps),
        n_log=z,
        ledger=Ledger(
            balance=jnp.zeros((br, nr), F64) if balance is None else balance,
            lane_busy_until=jnp.zeros(caps.n_lanes, I64),
            lane_gen=jnp.zeros(caps.n_lanes, I64),
            lane_closed=jnp.zeros(caps.n_lanes, B),
            program_charge_rule=jnp.full(caps.n_programs, ChargeRule.RECEIPT, I64),
            valid_from_tick=jnp.zeros(br, I64),
            valid_until_tick=jnp.full(br, caps.max_tick, I64),
            remainder_rule=jnp.full(br, RemainderRule.NONE, I64),
            disruptive_used=jnp.zeros(caps.n_assets, I64),
            debited=jnp.zeros((br, nr), F64),
            refunded=jnp.zeros((br, nr), F64),
            program_closed=jnp.zeros(caps.n_programs, B)
            if caps.n_programs > 1
            else None,
            refund_ignored=z if caps.refunds else None,
            win_state=jnp.zeros(br, I64) if caps.awards else None,
            win_gen=jnp.zeros(br, I64) if caps.awards else None,
            realloc_row=jnp.zeros(br, I64) if caps.awards else None,
            granted=jnp.zeros((br, nr), F64) if caps.awards else None,
            lapsed=jnp.zeros((br, nr), F64) if caps.awards else None,
        ),
        belief=_zero_belief(caps),
        pbelief=_zero_belief(caps),
        acquisition_counts=jnp.zeros(caps.n_subjects, I64),
        pstate=jnp.zeros(caps.n_pstate, F64),
        woke=f,
        pending_wake=f,
        done=f,
        draining=f,
        stalled=f,
        idle=f,
        overflow=f,
        overflow_mask=z,
        violation=f,
        double_settle=f,
        dup_fold=f,
        n_dec=z,
        dec_tick=jnp.asarray(-1, I64),
        dec_ordinal=z,
        next_sid=z,
        release_version=jnp.zeros(2, I64),
        request=empty_event_row(caps),
        refresh=empty_event_row(caps),
        refresh_dirty=jnp.zeros((2, n_refresh), B),
        refresh_status=jnp.full((2, n_refresh), -1, I64),
    )


class StepEmit(eqx.Module):
    """A handler's bookkeeping output."""

    n_written: jax.Array
    last_slot: jax.Array


class Trace(eqx.Module):
    """One per executor step."""

    handled: jax.Array
    kind: jax.Array
    tick: jax.Array
    stage: jax.Array
    micro: jax.Array
    tie: jax.Array
    sid: jax.Array
    lane: jax.Array
    channel: jax.Array
    subject: jax.Array
    acquisition_index: jax.Array
    log_slot: jax.Array  # log row named by a handled release or result
    audience: jax.Array
    n_written: jax.Array
    rec_status: jax.Array
    wake: jax.Array
    pending_wake: jax.Array
    n_dec: jax.Array
    snapshot_fp: jax.Array  # fingerprint of the snapshot a decision read, else 0
    event_fp: jax.Array  # fingerprint a handled request or result names, else 0
    release_version: jax.Array  # I64[2] after the step
    eff_kind: jax.Array
    eff_disruptive: jax.Array
    admit_code: jax.Array
    belief_mean: jax.Array
    belief_aux: jax.Array
    n_log: jax.Array
    pol_u: jax.Array
    # The program an event's debit or settlement is charged under; absent (None, no
    # leaf) unless the capacities hold more than one program.
    program: jax.Array | None = None
    # What each Allocate row of the decision asked: the amount per resource, the window
    # and the recipient row. Absent (None, no leaf) unless awards are on, so no other
    # trace, history or fingerprint changes.
    alloc_amount: jax.Array | None = None  # F64[E, R]
    alloc_from: jax.Array | None = None  # I64[E]
    alloc_until: jax.Array | None = None  # I64[E]
    alloc_recipient: jax.Array | None = None  # I64[E]


def absent(trace, name):
    """Whether `trace` holds no leaf `name`: a conditional leaf whose feature is off.

    Walks over every trace field skip these; the names are spelled out so no field is
    reached by a dynamic lookup.
    """
    return (
        (name == "program" and trace.program is None)
        or (name == "alloc_amount" and trace.alloc_amount is None)
        or (name == "alloc_from" and trace.alloc_from is None)
        or (name == "alloc_until" and trace.alloc_until is None)
        or (name == "alloc_recipient" and trace.alloc_recipient is None)
    )


# Trace fields filled by the observer layout's projection of the full belief.
PROJECTION_FIELDS = ("belief_mean", "belief_aux")


def empty_trace(caps):
    """Return a zeroed trace row."""
    z = jnp.asarray(0, I64)
    f = jnp.asarray(False)
    return Trace(
        handled=f,
        kind=z,
        tick=z,
        stage=z,
        micro=z,
        tie=z,
        sid=z,
        lane=z,
        channel=z,
        subject=z,
        acquisition_index=z,
        log_slot=z,
        audience=z,
        n_written=z,
        rec_status=z,
        wake=f,
        pending_wake=f,
        n_dec=z,
        snapshot_fp=z,
        event_fp=z,
        release_version=jnp.zeros(2, I64),
        eff_kind=jnp.zeros(caps.n_effects, I64),
        eff_disruptive=jnp.zeros(caps.n_effects, B),
        admit_code=jnp.full(caps.n_effects, AdmissionCode.NONE, I64),
        belief_mean=jnp.zeros(caps.n_subjects, F64),
        belief_aux=jnp.zeros(caps.n_an, F64),
        n_log=z,
        pol_u=jnp.asarray(0.0, F64),
        program=z if caps.n_programs > 1 else None,
        alloc_amount=jnp.zeros((caps.n_effects, caps.n_resources), F64)
        if caps.awards
        else None,
        alloc_from=jnp.zeros(caps.n_effects, I64) if caps.awards else None,
        alloc_until=jnp.zeros(caps.n_effects, I64) if caps.awards else None,
        alloc_recipient=jnp.zeros(caps.n_effects, I64) if caps.awards else None,
    )


def _tree_set(carry, **paths):
    """Set carry leaves by dotted name, e.g. `_tree_set(c, **{"ledger.balance": x})`."""
    names = list(paths)

    def getter(c):
        out = []
        for n in names:
            o = c
            for part in n.split("."):
                o = getattr(o, part)
            out.append(o)
        return tuple(out)

    return eqx.tree_at(getter, carry, tuple(paths.values()))


def init_carry(study, truth):
    """Initialize the budget and public kickoff wake at tick 0."""
    caps = study.caps
    c = empty_carry(caps, balance=study.budget)
    c = _tree_set(
        c, belief=study.observer.init(caps), pbelief=study.observer.init_summary(caps)
    )
    c = enqueue(
        c,
        True,
        kind=EventKind.WAIT,
        tick=0,
        stage=Stage.ADMIT,
        audience=PUBLIC,
        caps=caps,
    )
    return study.process.seed(c, caps)
