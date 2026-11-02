"""Builders for the record tests: scripted records, spies and the leaking layout."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core import (
    AbstractPolicy,
    AbstractProcess,
    Audience,
    EventKind,
    KindRegistry,
    Stage,
    enqueue,
    record_none,
)
from spaceodyssey.core.enums import F64, I64, PUBLIC, RecordStatus
from spaceodyssey.core.records import PayloadColumn, RecordKind
from spaceodyssey.studies import toy


class ScriptedRecords(AbstractProcess):
    """A process that writes one scripted measurement record per fire.

    Fire i happens at `ticks[i]`, in the release stage, and logs a record of subject 0
    with the i-th status, acquisition id, revision and value. The record is released at
    its own tick to the process audience.
    """

    ticks: tuple[int, ...] = eqx.field(static=True)
    values: tuple[float, ...] = eqx.field(static=True)
    ids: tuple[int, ...] = eqx.field(static=True)
    revisions: tuple[int, ...] = eqx.field(static=True)
    statuses: tuple[int, ...] = eqx.field(static=True)
    audience: int = eqx.field(default=PUBLIC)

    def seed(self, carry, caps):
        """Queue one process event per scripted record."""
        for i, tick in enumerate(self.ticks):
            carry = enqueue(
                carry,
                True,
                kind=EventKind.PROCESS,
                tick=tick,
                stage=Stage.RELEASE,
                group=i,
                audience=self.audience,
                caps=caps,
            )
        return carry

    def handle(self, carry, ev, study):
        """Write the scripted record of this fire."""
        base = record_none(study.caps)
        pick = lambda xs, dt: jnp.asarray(xs, dt)[ev.group]  # noqa: E731
        rec = dataclasses.replace(
            base,
            kind=jnp.asarray(toy.RECORD_MEASUREMENT, I64),
            start_tick=ev.tick,
            end_tick=ev.tick,
            available_tick=ev.tick,
            status=pick(self.statuses, I64),
            acquisition_id=pick(self.ids, I64),
            revision=pick(self.revisions, I64),
            released_tick=jnp.asarray(-1, I64),
            accepted_tick=jnp.asarray(-1, I64),
            superseded_by=jnp.asarray(-1, I64),
            payload=base.payload.at[0].set(pick(self.values, F64)),
            valid=jnp.asarray(True),
        )
        return carry, rec


def scripted_study(ticks, values, ids, revisions=None, statuses=None, **kw):
    """A toy study with a silent policy and a `ScriptedRecords` process."""
    n = len(ticks)
    process = ScriptedRecords(
        ticks=tuple(ticks),
        values=tuple(values),
        ids=tuple(ids),
        revisions=tuple(revisions or (0,) * n),
        statuses=tuple(statuses or (int(RecordStatus.COMPLETED),) * n),
        **kw,
    )
    return toy.survey_1(policies=(toy.NullPolicy(),), process=process)


def leaking_registry():
    """The old layout: the raw noise variate is a column every audience can read."""
    leaking = RecordKind(
        "measurement",
        (
            PayloadColumn("value", PUBLIC),
            PayloadColumn("variate", PUBLIC),
            PayloadColumn("fraction", PUBLIC),
        ),
    )
    return KindRegistry((RecordKind("none", ()), leaking, toy.AUXILIARY_KIND))


def with_leaking_layout(study):
    """The same study with the leaking layout lowered into its capacities."""
    caps = leaking_registry().lower(
        dataclasses.replace(study.caps, payload_audience=())
    )
    return dataclasses.replace(study, caps=caps)


def scorer_only_columns(caps):
    """bool[K, P]: columns only the scorer may read."""
    table = np.asarray(caps.payload_audience, dtype=np.int64)
    return table == int(Audience.SCORER)


SEEN = []


def _note(total):
    SEEN.append(float(total))


class SpyPolicy(AbstractPolicy):
    """Wraps a policy and reports the mass of `hidden` payload columns in its views.

    `hidden` is bool[K, P] over record kinds and columns, fixed when the spy is built so
    that a study with a different layout is judged against the same columns.
    """

    inner: AbstractPolicy
    hidden: tuple[tuple[bool, ...], ...] = eqx.field(static=True)

    def decide(self, pstate, view, study, key):
        """Report the hidden columns of the view, then defer to the inner policy."""
        hidden = jnp.asarray(self.hidden)
        kind = jnp.clip(view.log.kind, 0, hidden.shape[0] - 1)
        total = jnp.sum(jnp.abs(view.log.payload) * hidden[kind])
        jax.debug.callback(_note, total)
        return self.inner.decide(pstate, view, study, key)


def spied(study, reference=None):
    """The study with every policy wrapped in a `SpyPolicy`.

    The hidden columns are the scorer-only columns of `reference` (default: the
    study's own capacities).
    """
    mask = scorer_only_columns(study.caps if reference is None else reference)
    hidden = tuple(tuple(bool(x) for x in row) for row in mask)
    return eqx.tree_at(
        lambda s: s.agenda.policies,
        study,
        tuple(SpyPolicy(p, hidden) for p in study.agenda.policies),
    )


class ReconstructionPolicy(toy.FixedPlanPolicy):
    """Tries to recover the true signal from the first released row it sees.

    It inverts the channel model: signal = (value - noise * variate) / gain, which is
    exact when the raw variate is readable. The estimate and its subject (plus one) are
    kept in `pstate[2]` and `pstate[3]`.
    """

    def decide(self, pstate, view, study, key):
        """Plan as the fixed policy does, and record a reconstruction."""
        eff, ps, u = super().decide(pstate, view, study, key)
        row = view.log
        noise = study.channel_model.noise[row.channel_index[0]]
        est = (row.payload[0, 0] - noise * row.payload[0, 1]) / study.design.gain
        take = (view.n_rows > 0) & (ps[3] == 0)
        ps = ps.at[2].set(jnp.where(take, est, ps[2]))
        ps = ps.at[3].set(jnp.where(take, row.subject[0].astype(F64) + 1.0, ps[3]))
        return eff, ps, u
