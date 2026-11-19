"""The log of a run and the functions that write it.

Two functions build rows: ``grant_rows`` the rows of a grant and ``step`` each
observed row. Every row's ``model_slot`` is read from the design's mode table through
its action's mode. ``start_log`` starts a log from the grant's rows and ``execute``
runs a policy against a sealed world, one ``step`` per action.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import PRNGKeyArray, PyTree

from spaceodyssey.design import Design, Mode, Resources
from spaceodyssey.policy import AbstractPolicy, public_view
from spaceodyssey.types import (
    ACCEPTED,
    OVER_BUDGET,
    AbstractObservationModel,
    Action,
    Grant,
    ObservationLog,
    Row,
    clock_d,
    spent_d,
)
from spaceodyssey.world import SealedWorld, row_key, run_keys


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
        refusal=zero_i + ACCEPTED,
        model_slot=jnp.asarray(slots, dtype=zero_i.dtype).reshape(zero_i.shape),
        payload=payload,
    )


def start_log(rows: Row, capacity: int) -> ObservationLog:
    """A log of capacity ``W`` that starts with ``rows``, zeros after them.

    Args:
        rows: the grant's rows (``grant_rows``), stacked along a leading axis.
        capacity: the log's row capacity ``W``, static.

    Returns:
        The log, its count the number of rows.

    Raises:
        ValueError: if the rows do not fit the capacity.
    """
    n_row = rows.star.shape[0]
    if n_row > capacity:
        raise ValueError(f"{n_row} granted rows exceed the log capacity {capacity}")

    def pad(column):
        return (
            jnp.zeros((capacity, *column.shape[1:]), column.dtype)
            .at[:n_row]
            .set(column)
        )

    return ObservationLog(jax.tree.map(pad, rows), n_row)


def step(
    key: PRNGKeyArray,
    log: ObservationLog,
    world: SealedWorld,
    models: tuple[AbstractObservationModel, ...],
    action: Action,
    mode: Mode,
    resources: Resources,
) -> ObservationLog:
    """Take one action: draw its row from the world and write it at ``count``.

    The row starts on the calendar at the later of the action's ``not_before_d`` and
    the time the instrument is next free (``clock_d``), so waiting costs no budget. An
    action whose exposure fits the budget left (the budget less ``spent_d``) is
    accepted: it is charged its exposure, its payload is drawn
    by the world under the model of its mode with ``row_key(key, star, count)``, and
    it is released ``mode.latency_d`` after its exposure ends. One that does not fit is
    refused (``OVER_BUDGET``): charged nothing, its payload zeros. The payload sits in
    its model's buffer and the other buffers hold zeros. The model is picked by the
    mode's static slot, so nothing switches on a traced value, and under ``jax.vmap``
    over worlds (``log`` and ``world`` stacked) one program serves every world.

    Args:
        key: the run's noise key (``run_keys``).
        log: the log so far.
        world: the sealed world.
        models: the design's model table, static.
        action: the action.
        mode: the action's mode, from the design's mode table.
        resources: the run's budget.

    Returns:
        The log with the new row and its count one higher.

    Raises:
        EquinoxRuntimeError: if the log is full.
    """
    index = eqx.error_if(
        log.count, log.count >= log.rows.star.shape[0], "the log is full"
    )
    time_d = jnp.maximum(action.not_before_d, clock_d(log))
    accepted = action.exposure_d <= resources.budget_d - spent_d(log)
    drawn = world.observe(
        row_key(key, action.star, index), models[mode.model_slot], action
    )
    buffers = tuple(
        jax.tree.map(lambda x: jnp.where(accepted, x, 0), drawn)
        if slot == mode.model_slot
        else jax.tree.map(lambda column: jnp.zeros_like(column[0]), buffer)
        for slot, buffer in enumerate(log.rows.payload)
    )
    charged_d = jnp.where(accepted, action.exposure_d, 0.0)
    row = Row(
        star=action.star,
        mode=action.mode,
        time_d=time_d,
        exposure_d=action.exposure_d,
        charged_d=charged_d,
        release_d=time_d + charged_d + jnp.where(accepted, mode.latency_d, 0.0),
        refusal=jnp.where(accepted, ACCEPTED, OVER_BUDGET),
        model_slot=mode.model_slot,
        payload=buffers,
    )
    rows = jax.tree.map(
        lambda column, x: lax.dynamic_update_index_in_dim(column, x, index, 0),
        log.rows,
        row,
    )
    return ObservationLog(rows, index + 1)


_step = eqx.filter_jit(step)
_public_view = eqx.filter_jit(public_view)


def execute(
    key: PRNGKeyArray,
    policy: AbstractPolicy,
    world: SealedWorld,
    design: Design,
    grant: Grant,
) -> ObservationLog:
    """Run a policy against a sealed world until it stops.

    The run key is split first (``run_keys``); the world was drawn from its world key,
    every row's noise comes from its noise key, and each decision's key is the policy
    key folded with the number of actions taken. The grant's rows are revealed at time
    zero and start the log, and the policy's state starts from ``policy.init``. Then,
    on the host, the policy is shown the public view of the log and its action is
    taken by one compiled ``step``, with the model picked from the design's static mode
    table, until the policy stops. The policy's state passes from one decision to the
    next unread: the host reads only the action and the stop flag.

    Args:
        key: the run key.
        policy: the policy.
        world: the sealed world, one world (no world axis).
        design: the run's design.
        grant: the rows given at time zero.

    Returns:
        The run's log.

    Raises:
        ValueError: if the grant does not fit the log, or the policy asks for a mode
            not in the design or for more rows than the log holds.
    """
    _, noise_key, policy_key = run_keys(key)
    payload = world.reveal(noise_key, grant, design.models, design.modes)
    log = start_log(grant_rows(grant, payload, design.modes), design.capacity)
    n_granted = log.count
    public = _public_view(log, n_granted, design.resources)
    state = policy.init(public)
    while True:
        decision_key = jax.random.fold_in(policy_key, public.n_taken)
        action, stop, state = policy.decide(decision_key, state, public)
        if bool(stop):
            return log
        mode_index = int(action.mode)
        if not 0 <= mode_index < len(design.modes):
            raise ValueError(f"mode {mode_index} is not in the design's mode table")
        if int(log.count) >= design.capacity:
            raise ValueError(f"the policy asks for more than {design.capacity} rows")
        mode = design.modes[mode_index]
        log = _step(
            noise_key, log, world, design.models, action, mode, design.resources
        )
        public = _public_view(log, n_granted, design.resources)
