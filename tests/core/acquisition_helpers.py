"""Admission-index checker for scripted and generated effect sequences."""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from spaceodyssey import core
from spaceodyssey.core.enums import EffectKind, EventKind, RecordStatus
from spaceodyssey.studies import builder
from tests.core.helpers import CAPS, admit_j, advance_j, one_world

STUDY = builder.survey_1()
TRUTH, ROOT = one_world()
N_SUBJECTS = 2

# ---- a driver that feeds random effects straight into admission ----------------


def effect_table(op):
    """The effect table of one op: observe, group, cancel, wait or stop."""
    e = core.effects_none(CAPS)
    kind = op[0]
    if kind == "observe":
        _, subject, ch = op
        return dataclasses.replace(
            e,
            kind=e.kind.at[0].set(EffectKind.OBSERVE),
            subject=e.subject.at[0].set(subject),
            channel_index=e.channel_index.at[0].set(ch),
        )
    if kind == "group":
        _, subject_a, ch_a, subject_b, ch_b = op
        return dataclasses.replace(
            e,
            kind=jnp.full(CAPS.n_effects, EffectKind.OBSERVE, jnp.int64),
            subject=jnp.asarray([subject_a, subject_b], jnp.int64),
            channel_index=jnp.asarray([ch_a, ch_b], jnp.int64),
            group=jnp.asarray([3, 3], jnp.int64),
        )
    if kind == "cancel":
        return dataclasses.replace(
            e,
            kind=e.kind.at[0].set(EffectKind.CANCEL),
            lane=e.lane.at[0].set(op[1]),
        )
    if kind == "wait":
        return dataclasses.replace(
            e, kind=e.kind.at[0].set(EffectKind.WAIT), tick=e.tick.at[0].set(20)
        )
    _, scope, disposition = op
    return dataclasses.replace(
        e,
        kind=e.kind.at[0].set(EffectKind.STOP),
        lane=e.lane.at[0].set(1),
        scope=e.scope.at[0].set(scope),
        disposition=e.disposition.at[0].set(disposition),
    )


def new_settlements(before, after):
    """(subject, index) of the Done rows queued between two carries, in sid order."""
    ev = after.ev
    fresh = np.asarray(
        ev.valid & (ev.kind == EventKind.DONE) & (ev.sid >= before.next_sid)
    )
    order = np.argsort(np.asarray(ev.sid)[fresh])
    subjects = np.asarray(ev.subject)[fresh][order].tolist()
    indices = np.asarray(ev.acquisition_index)[fresh][order].tolist()
    return list(zip(subjects, indices, strict=True))


def drive(ops, advance=advance_j, steps=()):
    """Admit each op, then advance a few events; return the violations found."""
    c = eqx.filter_jit(core.init_carry)(STUDY, TRUTH)
    problems, admitted = [], {s: [] for s in range(N_SUBJECTS)}
    for n, op in enumerate(ops):
        before = c
        c, _codes = admit_j(c, effect_table(op), STUDY)
        counts_before = np.asarray(before.acquisition_counts)
        counts_after = np.asarray(c.acquisition_counts)
        fresh = new_settlements(before, c)
        expected = counts_before.copy()
        for subject, index in fresh:
            if index != expected[subject]:
                problems.append((n, "index", subject, index, int(expected[subject])))
            expected[subject] += 1
            admitted[subject].append(index)
        if not np.array_equal(counts_after, expected):
            problems.append((n, "counts", counts_after.tolist(), expected.tolist()))
        for _ in range(steps[n] if n < len(steps) else 1):
            if not bool(c.ev.valid.any()):
                break
            counts_pre = np.asarray(c.acquisition_counts)
            c, _, _ = advance(c, ROOT, STUDY, TRUTH, wake_fn=core.wake_by_audience)
            if np.any(np.asarray(c.acquisition_counts) < counts_pre):
                problems.append((n, "decrement", counts_pre.tolist()))
    for subject, got in admitted.items():
        if got != list(range(len(got))):
            problems.append(("dense", subject, got))
    log = c.log
    rows = [
        (int(log.subject[i]), int(log.acquisition_index[i]))
        for i in range(int(c.n_log))
        if int(log.status[i]) != RecordStatus.FAILED
    ]
    if len(rows) != len(set(rows)):
        problems.append(("duplicate record index", rows))
    return problems
