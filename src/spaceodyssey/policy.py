"""Policies: what to observe next, decided from the public record of a run.

A policy never sees the sealed world. It is handed a ``PublicView``, reductions of the
log at the current time: the budget left, the rows in flight and the rows released,
their payloads only once released. ``execute`` builds the view and asks the policy for
an action until the policy stops, threading the policy's state between decisions.
"""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from spaceodyssey.design import Resources
from spaceodyssey.types import Action, ObservationLog, as_float, as_index, spent_d


class PublicView(eqx.Module):
    """What a policy may see of a run at the current time.

    Attributes:
        log: the log, with the payload of every row not yet released zeroed: a row in
            flight shows its star, mode, times and charge, not its data.
        time_d: the current time, days: the time charged so far.
        budget_left_d: the budget not yet charged, days.
        in_flight: the rows taken and not yet released, shape ``(W,)``.
        n_taken: the actions taken so far, refused ones included: the rows after the
            grant.
    """

    log: ObservationLog
    time_d: Float[Array, ""] = eqx.field(converter=as_float)
    budget_left_d: Float[Array, ""] = eqx.field(converter=as_float)
    in_flight: Bool[Array, " W"] = eqx.field(converter=jnp.asarray)
    n_taken: Int[Array, ""] = eqx.field(converter=as_index)


def public_view(
    log: ObservationLog, n_granted: Int[Array, ""], resources: Resources
) -> PublicView:
    """The public view of a run at the time charged so far.

    Args:
        log: the log, grant rows first.
        n_granted: the number of grant rows at the head of the log.
        resources: the run's budget.

    Returns:
        The view a policy decides from.
    """
    rows = log.rows
    time_d = spent_d(log)
    real = jnp.arange(rows.star.shape[0]) < log.count
    released = real & (rows.release_d <= time_d)

    def hide(x):
        return jnp.where(released.reshape(-1, *([1] * (x.ndim - 1))), x, 0)

    payload = jax.tree.map(hide, rows.payload)
    public = ObservationLog(eqx.tree_at(lambda r: r.payload, rows, payload), log.count)
    return PublicView(
        log=public,
        time_d=time_d,
        budget_left_d=resources.budget_d - time_d,
        in_flight=real & ~released,
        n_taken=log.count - n_granted,
    )


class AbstractPolicy(eqx.Module):
    """A rule for the next action, from the public view alone.

    A policy is a record and a pair of functions on it. ``init`` gives the policy's
    state after the grant; ``decide`` takes a key, that state and the public view and
    returns the next action, whether to stop instead, and the new state. The state is
    a record (an ``eqx.Module``) that ``execute`` threads from one decision to the next
    without reading it, so a policy carries what it learns between decisions in its
    state and never in itself. ``stop`` is a traced boolean: when it is true the run
    ends and the action is ignored.

    ``execute`` never reads a belief: it shows the policy the public view and takes the
    action it returns. A policy that plans on a belief keeps that belief, and anything
    it caches from it (each star's frontier, the last solved price), in its state. It
    keeps them current through ``absorb``, applying the rows the view shows released
    since its last decision, and never replays rows it has not been shown.

    A planning policy reads the forecast and the verdict as two beliefs, each from its
    own replay of the log. Whether the forecast's replay follows the decision time (a
    learning forecast) or stays at time zero (a frozen forecast, the replay at
    ``time_d = 0``, which holds the grant's rows only) is a static field of the planning
    policy, ``forecast_updates``, its only reader. With the forecast frozen the planner
    maximizes a frozen-forecast surrogate, which is not a martingale.
    """

    @abc.abstractmethod
    def init(self, public: PublicView) -> eqx.Module:
        """The policy's state after the grant.

        Args:
            public: the public view of the run after the grant.

        Returns:
            The state, a record.
        """

    @abc.abstractmethod
    def decide(
        self, key: PRNGKeyArray, state: eqx.Module, public: PublicView
    ) -> tuple[Action, Bool[Array, ""], eqx.Module]:
        """The next action, whether to stop, and the new state.

        Args:
            key: the decision's key, folded from the run's policy key (``run_keys``)
                and the number of actions taken.
            state: the policy's state.
            public: the public view of the run.

        Returns:
            The action, the stop flag (when true the run ends and the action is not
            taken) and the new state.
        """


class TableState(eqx.Module):
    """Where a ``TablePolicy`` is in its table.

    Attributes:
        cursor: index of the table's next action.
    """

    cursor: Int[Array, ""] = eqx.field(converter=as_index)


class TablePolicy(AbstractPolicy):
    """Replays a fixed table of actions in order, then stops.

    The table is a schedule made outside the run (an EXOSIMS, AYO or M4OPT schedule,
    or an oracle's): action ``i`` is taken as the run's ``i``-th action, whatever the
    rows before it showed, and a refused action is not retried. The state is the index
    of the next action, a traced counter, so a decision reads nothing concrete.

    Attributes:
        actions: an ``Action`` whose leaves have a leading axis over the table.
    """

    actions: Action

    def init(self, public: PublicView) -> TableState:
        """The cursor at the table's first action."""
        return TableState(0)

    def decide(
        self, key: PRNGKeyArray, state: TableState, public: PublicView
    ) -> tuple[Action, Bool[Array, ""], TableState]:
        """The table's next action; stop past its end."""
        n_action = self.actions.star.shape[0]
        if n_action == 0:  # a static shape: an empty table stops at once
            return Action(0, 0, 0.0), jnp.asarray(True), state
        index = jnp.minimum(state.cursor, n_action - 1)
        action = jax.tree.map(lambda leaf: leaf[index], self.actions)
        return action, state.cursor >= n_action, TableState(state.cursor + 1)
