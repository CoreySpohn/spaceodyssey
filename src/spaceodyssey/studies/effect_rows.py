"""Effect-table rows and the channel lookup used by the toy kinds and the builder."""

import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.enums import (
    I64,
)
from spaceodyssey.core.snapshot import (
    EffectTable,
)


def effects(
    caps,
    kind,
    subject=0,
    channel_index=0,
    group=0,
    tick=0,
    lane=0,
    scope=0,
    disposition=0,
    disruptive=False,
    start_tick=None,
):
    """An `EffectTable` of `caps.n_effects` rows; short columns are padded."""
    n = caps.n_effects

    def fit(v, dtype=I64):
        # a column of fewer rows than the capacity is completed with empty rows
        a = jnp.asarray(v, dtype)
        if a.ndim == 1 and a.shape[0] < n:
            a = jnp.concatenate([a, jnp.zeros(n - a.shape[0], dtype)])
        return a

    def col(v, dtype=I64):
        if np.ndim(v) == 0:
            return jnp.asarray(v, dtype) * jnp.ones(n, dtype)
        return fit(v, dtype)

    return EffectTable(
        kind=fit(kind),
        subject=col(subject),
        channel_index=col(channel_index),
        group=col(group),
        tick=col(tick),
        lane=col(lane),
        scope=col(scope),
        disposition=col(disposition),
        disruptive=col(disruptive, jnp.bool_),
        start_tick=col(0 if start_tick is None else start_tick)
        if caps.commitments
        else None,
    )


def chan_index(study, channel_id):
    """Index of the channel with id `channel_id` in the study's channel table."""
    return jnp.argmax(study.channels.id == channel_id).astype(I64)
