"""A process that writes scripted measurement records, for revision studies."""

import dataclasses

import equinox as eqx
import jax.numpy as jnp

from spaceodyssey.core import (
    AbstractProcess,
    EventKind,
    Stage,
    enqueue,
    record_none,
)
from spaceodyssey.core.enums import F64, I64, PUBLIC, RecordStatus
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


__all__ = ["ScriptedRecords", "scripted_study"]
