"""A process that writes scripted measurement records, for revision studies."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core import (
    AbstractProcess,
    EventKind,
    Stage,
    enqueue,
    record_none,
)
from spaceodyssey.core.capacities import PLAIN_VALUE
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import F64, I64, PUBLIC, Capacity, RecordStatus
from spaceodyssey.plain import plain_array
from spaceodyssey.studies import toy

# The channel scripted records are written on by default: neither an admission channel
# of the toy studies (10 to 13), a refresh product channel (90, 91) nor a service
# channel.
SCRIPTED_CHANNEL = 50


def _i64_row(x):
    return plain_array(I64, x, row=True)


def _f64_row(x):
    return plain_array(F64, x, row=True)


class ScriptedRecords(AbstractProcess):
    """A process that writes one scripted measurement record per fire.

    Fire i happens at `ticks[i]`, in the release stage, and logs a record of subject 0
    with the i-th status, acquisition id, revision and value. The record is released at
    its own tick to the process audience.

    A scripted record is identified by the process's own coordinates, not by the
    admission counters: its channel is the declared `channel` (never an admission
    channel) and its acquisition index is its scripted acquisition id, kept across the
    revisions of that acquisition. Repeated fires therefore never share the identity
    `(channel, subject, acquisition index, revision)`.
    """

    # The values are arrays, so scripts that differ only in numbers share one compiled
    # program; the number of fires is the array length and stays static. A document
    # writes each as the tuple it was built from.
    ticks: jax.Array = eqx.field(converter=_i64_row, metadata=PLAIN_VALUE)
    values: jax.Array = eqx.field(converter=_f64_row, metadata=PLAIN_VALUE)
    ids: jax.Array = eqx.field(converter=_i64_row, metadata=PLAIN_VALUE)
    revisions: jax.Array = eqx.field(converter=_i64_row, metadata=PLAIN_VALUE)
    statuses: jax.Array = eqx.field(converter=_i64_row, metadata=PLAIN_VALUE)
    audience: int = eqx.field(default=PUBLIC)
    channel: int = eqx.field(default=SCRIPTED_CHANNEL, static=True)
    # the kind of each fire's record; empty means every record is a measurement
    kinds: jax.Array = eqx.field(default=(), converter=_i64_row, metadata=PLAIN_VALUE)

    def __check_init__(self):
        """Refuse two fires that name one (scripted id, revision) pair."""
        seen = set()
        pairs = zip(self.ids.tolist(), self.revisions.tolist(), strict=True)
        for pair in pairs:
            if pair in seen:
                raise ValueError(
                    f"scripted records repeat (id, revision) = {pair}; a correction "
                    "needs a distinct revision of its acquisition"
                )
            seen.add(pair)

    def declare(self):
        """The record channel this process writes on, for pre-flight."""
        return Declaration(channels=(self.channel,))

    def seed(self, carry, caps):
        """Queue one process event per scripted record."""
        for i in range(self.ticks.shape[0]):
            carry = enqueue(
                carry,
                True,
                kind=EventKind.PROCESS,
                tick=self.ticks[i],
                stage=Stage.RELEASE,
                group=i,
                audience=self.audience,
                caps=caps,
            )
        return carry

    def handle(self, carry, ev, study):
        """Write the scripted record of this fire."""
        caps = study.caps
        base = record_none(caps)
        pick = lambda xs, dt: jnp.asarray(xs, dt)[ev.group]  # noqa: E731
        index = pick(self.ids, I64)
        # The scripted id is the semantic index. The internal id comes from the shared
        # allocator: the first fire of an id takes the next one, a correction reuses
        # the id of the record it corrects, so no producer shares an internal id.
        log = carry.log
        same = (
            log.valid & (log.channel == self.channel) & (log.acquisition_index == index)
        )
        found = same.any()
        room = carry.next_sid < caps.max_sid
        internal = jnp.where(
            found, log.acquisition_id[jnp.argmax(same)], carry.next_sid
        )
        take = ~found & room
        full = ~found & ~room
        carry = eqx.tree_at(
            lambda c: (c.next_sid, c.overflow, c.overflow_mask),
            carry,
            (
                carry.next_sid + take.astype(I64),
                carry.overflow | full,
                carry.overflow_mask | jnp.where(full, Capacity.STABLE_IDS, 0),
            ),
        )
        rec = dataclasses.replace(
            base,
            kind=(
                pick(self.kinds, I64)
                if self.kinds.size
                else jnp.asarray(toy.RECORD_MEASUREMENT, I64)
            ),
            channel=jnp.asarray(self.channel, I64),
            acquisition_index=index,
            start_tick=ev.tick,
            end_tick=ev.tick,
            available_tick=ev.tick,
            status=pick(self.statuses, I64),
            acquisition_id=internal,
            revision=pick(self.revisions, I64),
            released_tick=jnp.asarray(-1, I64),
            accepted_tick=jnp.asarray(-1, I64),
            superseded_by=jnp.asarray(-1, I64),
            payload=base.payload.at[0].set(pick(self.values, F64)),
            valid=~full,
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


def scripted_refresh_study(
    ticks, values, ids, revisions=None, statuses=None, *, every=100, **kw
):
    """A scripted study on a refresh grid: a refit-only observer, a silent policy.

    `kw` goes to `toy.survey_refresh` (an `observer`, capacities, a `refresh`); the
    process is the scripted one.
    """
    n = len(ticks)
    process = ScriptedRecords(
        ticks=tuple(ticks),
        values=tuple(values),
        ids=tuple(ids),
        revisions=tuple(revisions or (0,) * n),
        statuses=tuple(statuses or (int(RecordStatus.COMPLETED),) * n),
    )
    return toy.survey_refresh(
        every, policies=(toy.NullPolicy(),), process=process, **kw
    )


__all__ = [
    "SCRIPTED_CHANNEL",
    "ScriptedRecords",
    "scripted_refresh_study",
    "scripted_study",
]
