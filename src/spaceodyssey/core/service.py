"""Host protocol: how a request to the host is answered inside the transaction.

A `Query` effect becomes a request row. When the transaction reaches that row it holds
the run (`Carry.request`) until the host answers. The answer comes back as a record
written to the log and a `Result` row queued at the request's own tick and stage; the
transaction then handles it like any other row, so it is released and woken on in the
same stage. The host sees the request, the snapshot the policy asked on and nothing
else: no truth, no queue.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.carry import _tree_set
from spaceodyssey.core.enums import F64, I64, Audience, EventKind, RecordStatus
from spaceodyssey.core.events import empty_event_row, enqueue
from spaceodyssey.core.records import record_none, write_log
from spaceodyssey.core.snapshot import snapshot_fingerprint, snapshot_view


@dataclasses.dataclass(frozen=True)
class Request:
    """What the host is asked: plain integers read off the pending request row."""

    tick: int
    stage: int
    micro: int
    service: int  # the `channel_index` of the Query effect
    subject: int
    audience: int  # who the snapshot was cut for, and who may read the answer
    snapshot_fp: int
    release_version: int
    decision: int  # index of the decision that asked
    group: int


class AbstractHostService:
    """A host-side answerer. `kind` is the record kind index its answers carry.

    `answer` gets the request and the snapshot it names (the audience's masked view,
    as numpy arrays) and returns the payload columns of the answer. It must be a
    function of those two alone, and the same on every executor.
    """

    kind: int = 0

    def answer(self, request, view):
        """Return the answer's payload as a float vector of at most `payload_width`."""
        raise NotImplementedError


def read_request(carry):
    """The pending request of one world as a `Request`, or None when none is held."""
    row = carry.request
    if not bool(row.valid):
        return None
    return Request(
        tick=int(row.tick),
        stage=int(row.stage),
        micro=int(row.micro),
        service=int(row.channel_index),
        subject=int(row.subject),
        audience=int(row.audience),
        snapshot_fp=int(row.snapshot_fp),
        release_version=int(row.version),
        decision=int(row.decision),
        group=int(row.group),
    )


def commit_result(carry, payload, kind, study):
    """Write the host's answer and queue its `Result` row beside the request.

    The row takes the request's tick and stage, so it sorts straight after the request
    within the same stage and no later work runs before it. The record is logged now and
    released when the row is handled. Clears the pending request; does nothing when
    none is held.
    """
    caps = study.caps
    req = carry.request
    base = record_none(caps)
    rec = dataclasses.replace(
        base,
        kind=jnp.asarray(kind, I64),
        subject=req.subject,
        channel=req.channel_index,
        channel_index=req.channel_index,
        start_tick=req.tick,
        end_tick=req.tick,
        available_tick=req.tick,
        status=jnp.asarray(RecordStatus.COMPLETED, I64),
        phase=carry.phase,
        acquisition_id=carry.next_sid,
        released_tick=jnp.asarray(-1, I64),
        accepted_tick=jnp.asarray(-1, I64),
        superseded_by=jnp.asarray(-1, I64),
        audience=req.audience,
        source=req.channel_index,
        vintage=carry.phase,
        payload=jnp.asarray(payload, F64),
        valid=req.valid,
    )
    c1, slot = write_log(carry, rec, caps)
    wrote = c1.n_log > carry.n_log
    c2 = enqueue(
        c1,
        wrote,
        kind=EventKind.RESULT,
        tick=req.tick,
        stage=req.stage,
        subject=req.subject,
        channel_index=req.channel_index,
        channel=req.channel_index,
        group=req.group,
        audience=req.audience,
        slot=slot,
        acquisition_id=carry.next_sid,
        snapshot_fp=req.snapshot_fp,
        version=req.version,
        decision=req.decision,
        caps=caps,
    )
    return _tree_set(c2, request=empty_event_row())


@eqx.filter_jit
def _snapshot_for(carry, audience, decision, caps):
    view = snapshot_view(carry, audience, caps)
    view = dataclasses.replace(view, n_dec=decision)
    return view, snapshot_fingerprint(view)


_commit = eqx.filter_jit(commit_result)


def answer_pending(carry, study, service):
    """Answer the pending request of one world through `service`; returns the carry.

    The snapshot handed over is rebuilt from the carry for the request's audience at
    the request's decision. Its fingerprint must equal the one on the request, which is
    the check that nothing visible to that audience changed between the question and
    the answer.
    """
    request = read_request(carry)
    if request is None:
        return carry
    caps = study.caps
    view, fp = _snapshot_for(
        carry, Audience(request.audience), np.int64(request.decision), caps
    )
    if int(fp) != request.snapshot_fp:
        raise RuntimeError("the snapshot changed between the request and its answer")
    payload = np.zeros(caps.payload_width, dtype=np.float64)
    given = np.atleast_1d(
        np.asarray(
            service.answer(request, jax.tree_util.tree_map(np.asarray, view)),
            dtype=np.float64,
        )
    )
    if given.shape[0] > caps.payload_width:
        raise ValueError("the answer is wider than the payload")
    payload[: given.shape[0]] = given
    return _commit(carry, payload, np.int64(service.kind), study)
