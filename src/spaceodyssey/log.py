"""The log of a run and the functions that write it.

Two functions build rows: ``grant_rows`` the rows of a grant and ``step`` each
observed row. Every row's ``model_slot`` is read from the design's mode table through
its action's mode.
"""

import jax.numpy as jnp
from jaxtyping import PyTree

from spaceodyssey.design import Mode
from spaceodyssey.types import Grant, Row


def grant_rows(
    grant: Grant, payload: tuple[PyTree, ...], modes: tuple[Mode, ...]
) -> Row:
    """The log rows of a grant: released at time zero and charged nothing.

    Each row keeps its action's star, mode and exposure, and carries the model slot
    of its mode.

    Args:
        grant: the grant.
        payload: one payload buffer per model, leading axis over the granted rows, as
            ``SealedWorld.reveal`` returns them.
        modes: the design's mode table, static.

    Returns:
        The granted rows, stacked along a leading axis.
    """
    actions = grant.actions
    zero_d = jnp.zeros_like(actions.exposure_d)
    zero_i = jnp.zeros_like(actions.star)
    slots = [modes[int(m)].model_slot for m in jnp.ravel(actions.mode)]
    return Row(
        star=actions.star,
        mode=actions.mode,
        time_d=zero_d,
        exposure_d=actions.exposure_d,
        charged_d=zero_d,
        release_d=zero_d,
        refusal=zero_i,
        model_slot=jnp.asarray(slots, dtype=zero_i.dtype).reshape(zero_i.shape),
        payload=payload,
    )
