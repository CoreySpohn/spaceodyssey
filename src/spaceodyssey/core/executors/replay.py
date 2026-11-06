"""Replay a recorded run log into an observer belief."""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.enums import Audience, EventKind, RecordStatus
from spaceodyssey.core.keys import fold_key
from spaceodyssey.core.records import LOG_INT_FIELDS, Record, masked_record


@eqx.filter_jit
def _fold(study, belief, rec, key=None):
    return study.observer.analyze(study.observer.fold(belief, rec, study.caps, key=key))


def _row(log, i):
    fields = {n: jnp.asarray(getattr(log, n)[i]) for n in LOG_INT_FIELDS}
    fields.update({n: jnp.asarray(getattr(log, n)[i]) for n in ("cost", "charged")})
    fields["payload"] = jnp.asarray(log.payload[i])
    fields["valid"] = jnp.asarray(True)
    return Record(**fields)  # the full row; callers mask it for their audience


def replay_belief(study, carry, trace, root=None):
    """Rebuild the observer belief of one world from its recorded log alone.

    Reads only the record log and the order in which the run released rows (the
    handled release and result steps of the trace). Each acquisition is folded once, in
    release order, through the study's observer; later revisions are not folded. With
    the executor's own observer on the same rows in the same order the result equals
    `carry.belief` bit for bit. Each fold gets the record's fold key under the
    world's `root` key (`keys.fold_key`, as in the run); without `root` the folds
    get no key, which only an observer that draws nothing accepts. A release that
    names no matching log row raises `ValueError`.
    """
    log = dataclasses.replace(
        carry.log,
        **{n: np.asarray(getattr(carry.log, n)) for n in (*LOG_INT_FIELDS, "valid")},
    )
    kind = np.asarray(trace.kind)
    released = (kind == EventKind.RELEASE) | (kind == EventKind.RESULT)
    handled = np.asarray(trace.handled) & released
    steps = zip(
        *(
            np.asarray(getattr(trace, n))[handled].tolist()
            for n in ("log_slot", "tick", "subject", "acquisition_index", "channel")
        ),
        strict=True,
    )
    belief = study.observer.init(study.caps)
    used = np.zeros(log.valid.shape[0], dtype=bool)
    folded = set()
    for i, tick, subject, acquisition_index, channel in steps:
        if not (
            0 <= i < len(log.valid)
            and log.valid[i]
            and log.subject[i] == subject
            and log.acquisition_index[i] == acquisition_index
            and log.channel[i] == channel
        ):
            raise ValueError(f"release at tick {tick} matches no log row at slot {i}")
        if log.status[i] == RecordStatus.UNRELEASED:
            continue
        if int(log.kind[i]) in study.caps.derived_kinds:
            # a derived row (a product or a declared summary) is never folded
            continue
        if not int(log.audience[i]) & Audience.OBSERVER:
            # a row the observer may not read is never folded into its belief
            continue
        if not used[i] and log.accepted_tick[i] != tick:
            raise ValueError(
                f"release at tick {tick} matches no accepted row at slot {i}"
            )
        used[i] = True
        acquisition = int(log.acquisition_id[i])
        if acquisition in folded:
            continue
        folded.add(acquisition)
        full = _row(carry.log, i)
        key = None if root is None else fold_key(root, full, study)
        belief = _fold(
            study, belief, masked_record(full, study.caps, Audience.OBSERVER), key
        )
    return belief
