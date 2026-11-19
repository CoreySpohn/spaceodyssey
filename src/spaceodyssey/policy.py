"""Policies: what to observe next, decided from the public record of a run.

A policy never sees the sealed world. It is handed a ``PublicView``, reductions of the
log at the current time: the budget left, the rows in flight and the rows released,
their payloads only once released. ``execute`` builds the view and asks the policy for
an action until the policy stops.
"""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

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

    ``execute`` never reads a belief: it shows the policy the public view and takes the
    action it returns. A policy that plans on a belief holds that belief itself and
    keeps it current through ``absorb``, applying the rows the view shows released
    since its last decision; it never replays rows it has not been shown.
    """

    @abc.abstractmethod
    def decide(self, public: PublicView) -> Action | None:
        """The next action, or ``None`` to end the run.

        Args:
            public: the public view of the run.

        Returns:
            The action, or ``None`` when the policy stops.
        """


class TablePolicy(AbstractPolicy):
    """Replays a fixed table of actions in order, then stops.

    The table is a schedule made outside the run (an EXOSIMS, AYO or M4OPT schedule,
    or an oracle's): action ``i`` is taken as the run's ``i``-th action, whatever the
    rows before it showed, and a refused action is not retried.

    Attributes:
        actions: an ``Action`` whose leaves have a leading axis over the table.
    """

    actions: Action

    def decide(self, public: PublicView) -> Action | None:
        """The table's next action, ``None`` past its end."""
        index = int(public.n_taken)
        if index >= self.actions.star.shape[0]:
            return None
        return jax.tree.map(lambda leaf: leaf[index], self.actions)
