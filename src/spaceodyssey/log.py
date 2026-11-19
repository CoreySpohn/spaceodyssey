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

from spaceodyssey.design import (
    Design,
    Keepout,
    Mode,
    Resources,
    fits,
    model_slots,
    next_open,
)
from spaceodyssey.policy import AbstractPolicy, public_view
from spaceodyssey.types import (
    ACCEPTED,
    OVER_BUDGET,
    PAST_END,
    WAITED,
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
    slots = model_slots(modes, actions.mode)
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
    keepout: Keepout,
) -> ObservationLog:
    """Take one action: draw its row from the world and write it at ``count``.

    The row is ready on the calendar at the later of the action's ``not_before_d`` and
    the time the instrument is next free (``clock_d``), so waiting costs no budget; an
    observation asked for while its star is in keepout waits for the star to open
    (``next_open``). An action of zero exposure is a wait (``WAITED``): the instrument
    idles until its start, which moves the clock; keepout does not delay it; it is
    charged nothing, carries no payload and is released at its start, so no replay
    applies it. A row that would start after the calendar's end
    (``Resources.end_d``), or whose star never opens again, is refused
    (``PAST_END``). Any other action whose exposure fits the budget left (the budget
    less ``spent_d``, by the one rule ``fits``) is accepted: it is charged its
    exposure, its payload is drawn by the world under the model of its mode with
    ``row_key(key, star, count)`` from the action at its realized start (its
    ``not_before_d`` set to the start), and it is released ``mode.latency_d`` after
    its exposure ends. One that does not fit is refused (``OVER_BUDGET``): charged
    nothing, its payload zeros. A refused row still starts on the calendar, so a
    refused row that asked to wait moves the clock to its requested start: the
    refusal is a verdict made at that start, and waiting costs nothing, so nothing is
    gained or lost. The payload sits in its model's buffer and the other buffers hold
    zeros. The model is picked by the mode's static slot, so nothing switches on a
    traced value, and under ``jax.vmap`` over worlds (``log`` and ``world`` stacked)
    one program serves every world.

    Args:
        key: the run's noise key (``run_keys``).
        log: the log so far.
        world: the sealed world.
        models: the design's model table, static.
        action: the action.
        mode: the action's mode, from the design's mode table.
        resources: the run's budget and the calendar's end.
        keepout: the design's keepout table.

    Returns:
        The log with the new row and its count one higher.

    Raises:
        EquinoxRuntimeError: if the log is full.
    """
    index = eqx.error_if(
        log.count, log.count >= log.rows.star.shape[0], "the log is full"
    )
    ready_d = jnp.maximum(action.not_before_d, clock_d(log))
    waited = action.exposure_d <= 0.0
    opens_d = jnp.where(waited, ready_d, next_open(keepout, action.star, ready_d))
    late = ~(opens_d <= resources.end_d)  # never opening is late too
    time_d = jnp.where(jnp.isfinite(opens_d), opens_d, ready_d)
    left_d = resources.budget_d - spent_d(log)
    accepted = ~waited & ~late & fits(action.exposure_d, left_d)
    started = Action(action.star, action.mode, action.exposure_d, time_d)
    drawn = world.observe(
        row_key(key, action.star, index), models[mode.model_slot], started
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
        refusal=jnp.select(
            [late, waited, accepted], [PAST_END, WAITED, ACCEPTED], OVER_BUDGET
        ),
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


@eqx.filter_jit
def _decide(policy, key, state, public):
    """A policy's decision, compiled once per policy structure."""
    return policy.decide(key, state, public)


def execute(
    key: PRNGKeyArray,
    grant: Grant,
    world: SealedWorld,
    design: Design,
    policy: AbstractPolicy,
) -> ObservationLog:
    """Run a policy against a sealed world until it stops.

    The run key is split first (``run_keys``); the world was drawn from its world key,
    every row's noise comes from its noise key, and each decision's key is the policy
    key folded with the number of actions taken. The grant's rows are revealed at time
    zero and start the log, and the policy's state starts from ``policy.init``. Then,
    on the host, the policy is shown the public view of the log, decides through one
    compiled ``decide`` (called directly for a ``host_only`` policy), and its action
    is taken by one compiled ``step``, with the model picked from the design's static
    mode table, until the policy stops. The policy's state passes from one decision to
    the next unread: the host reads only the action and the stop flag.

    The arguments follow ``step``: the key, the state the run starts from (the
    grant), the world, the design and what chooses the actions.

    Args:
        key: the run key.
        grant: the rows given at time zero.
        world: the sealed world, one world (no world axis).
        design: the run's design.
        policy: the policy.

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
    public = _public_view(log, n_granted, design.resources, design.keepout)
    state = policy.init(public)
    while True:
        decision_key = jax.random.fold_in(policy_key, public.n_taken)
        if policy.host_only:
            action, stop, state = policy.decide(decision_key, state, public)
        else:
            action, stop, state = _decide(policy, decision_key, state, public)
        if bool(stop):
            return log
        mode_index = int(action.mode)
        if not 0 <= mode_index < len(design.modes):
            raise ValueError(f"mode {mode_index} is not in the design's mode table")
        if int(log.count) >= design.capacity:
            raise ValueError(f"the policy asks for more than {design.capacity} rows")
        mode = design.modes[mode_index]
        log = _step(
            noise_key,
            log,
            world,
            design.models,
            action,
            mode,
            design.resources,
            design.keepout,
        )
        public = _public_view(log, n_granted, design.resources, design.keepout)
