"""Replay a recorded run log into an observer belief."""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.enums import Audience, EventKind
from spaceodyssey.core.records import LOG_INT_FIELDS, Record, masked_record


@eqx.filter_jit
def _fold(study, belief, rec):
    return study.observer.analyze(study.observer.fold(belief, rec, study.caps))


def _row(log, i):
    fields = {n: jnp.asarray(getattr(log, n)[i]) for n in LOG_INT_FIELDS}
    fields.update({n: jnp.asarray(getattr(log, n)[i]) for n in ("cost", "charged")})
    fields["payload"] = jnp.asarray(log.payload[i])
    fields["valid"] = jnp.asarray(True)
    return Record(**fields)  # the full row; callers mask it for their audience


def replay_belief(study, carry, trace):
    """Rebuild the observer belief of one world from its recorded log alone.

    Reads only the record log and the order in which the run released rows (the
    handled release steps of the trace). Each acquisition is folded once, in release
    order, through the study's observer; later revisions are not folded. With the
    executor's own observer on the same rows in the same order the result equals
    `carry.belief` bit for bit.
    """
    log = dataclasses.replace(
        carry.log,
        **{n: np.asarray(getattr(carry.log, n)) for n in (*LOG_INT_FIELDS, "valid")},
    )
    handled = np.asarray(trace.handled) & (np.asarray(trace.kind) == EventKind.RELEASE)
    steps = zip(
        *(
            np.asarray(getattr(trace, n))[handled].tolist()
            for n in ("tick", "subject", "acquisition_index", "channel")
        ),
        strict=True,
    )
    belief = study.observer.init(study.caps)
    used = np.zeros(log.valid.shape[0], dtype=bool)
    folded = set()
    for tick, subject, acquisition_index, channel in steps:
        hit = (
            log.valid
            & ~used
            & (log.accepted_tick == tick)
            & (log.subject == subject)
            & (log.acquisition_index == acquisition_index)
            & (log.channel == channel)
        )
        if not hit.any():
            continue
        i = int(np.argmax(hit))
        used[i] = True
        acquisition = int(log.acquisition_id[i])
        if acquisition in folded:
            continue
        folded.add(acquisition)
        row = masked_record(_row(carry.log, i), study.caps, Audience.OBSERVER)
        belief = _fold(study, belief, row)
    return belief
