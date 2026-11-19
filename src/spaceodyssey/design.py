"""The design of a run: its modes, the table from mode to model, budget and calendar.

A mode is what a policy chooses when it observes (an instrument setting); an observation
model is what inverts the rows a mode returns. Several modes may share one model. The
design holds the one table from mode to model: every row's ``model_slot`` and every
menu entry's model are read from it, so no record states a slot of its own.

The calendar has an end (``Resources.end_d``), and each star can be observed only out
of keepout (``Keepout``). One budget rule serves every reader: ``fits``.

The host unpacks a design into narrow pieces (the model table, one mode, the budget,
the keepout table); no kernel takes a ``Design``.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from spaceodyssey.types import AbstractObservationModel, as_float

BUDGET_TOL_D = 1e-9
"""The budget rule's tolerance, days: an exposure fits the budget left when it exceeds
it by no more than this (``fits``), so a schedule whose exposures sum to the budget in
decimal is never refused its last row by rounding. A run's realized spend is at most
its budget plus this."""


def fits(
    cost_d: Float[Array, "..."], left_d: Float[Array, "..."]
) -> Bool[Array, "..."]:
    """The one budget rule: whether a cost fits the budget left (``BUDGET_TOL_D``).

    ``step`` accepts a row, ``dispatch`` hands out a tied star's upgrade and a planner
    offers an option by this rule alone.

    Args:
        cost_d: the time asked for, days.
        left_d: the budget left, days.

    Returns:
        True where ``cost_d <= left_d + BUDGET_TOL_D``.
    """
    return cost_d <= left_d + BUDGET_TOL_D


class Mode(eqx.Module):
    """One observing mode: the model that inverts its rows and when they are released.

    A row of this mode starts when it is taken, occupies the instrument for its
    exposure, and is released ``latency_d`` after the exposure ends. Until then it is
    in flight: in the log, but not yet read by any replay.

    Attributes:
        model_slot: index of the mode's observation model in the design's model table,
            static.
        latency_d: days from the end of an exposure to the release of its row.
    """

    model_slot: int = eqx.field(static=True)
    latency_d: Float[Array, ""] = eqx.field(converter=as_float)


class Resources(eqx.Module):
    """What a run may spend: observing time and calendar time.

    Attributes:
        budget_d: the observing time of the run, days. The grant is charged nothing.
        end_d: the calendar's end, days from the start of the survey: no row starts
            after it (``PAST_END``).
    """

    budget_d: Float[Array, ""] = eqx.field(converter=as_float)
    end_d: Float[Array, ""] = eqx.field(converter=as_float)


class Keepout(eqx.Module):
    """When each star can be observed: a table of closed calendar bins per star.

    Bin ``b`` covers ``[b step_d, (b + 1) step_d)``; past the last bin the last bin
    holds, so a table that covers the calendar says everything. A star is open at a
    time when its bin there is not closed (``open_at``); an observation asked for in
    keepout starts when its star next opens (``next_open``).

    Attributes:
        closed: whether each star is in keepout in each bin, shape ``(S, T)``.
        step_d: the width of a bin, days.
    """

    closed: Bool[Array, "S T"] = eqx.field(converter=jnp.asarray)
    step_d: Float[Array, ""] = eqx.field(converter=as_float)

    def __check_init__(self):
        """Check that the table has a star axis and a time axis."""
        if self.closed.ndim != 2 or self.closed.shape[-1] < 1:
            raise ValueError(f"keepout is (S, T), not {self.closed.shape}")


def no_keepout(n_star: int) -> Keepout:
    """A keepout table that closes no star at any time.

    Args:
        n_star: the number of stars, ``S``.

    Returns:
        The table, one open bin per star.
    """
    return Keepout(np.zeros((n_star, 1), bool), 1.0)


def _bin(keepout: Keepout, time_d: Float[Array, ""]) -> Int[Array, ""]:
    """The table's bin at a time, the last one past the table's end."""
    n_bin = keepout.closed.shape[-1]
    index = jnp.floor(time_d / keepout.step_d)
    return jnp.clip(jnp.nan_to_num(index, posinf=n_bin), 0, n_bin - 1).astype(int)


def open_at(
    keepout: Keepout, star: Int[Array, ""], time_d: Float[Array, ""]
) -> Bool[Array, ""]:
    """Whether a star is out of keepout at a time.

    Args:
        keepout: the keepout table.
        star: the star's index.
        time_d: the time, days from the start of the survey.

    Returns:
        True when the star can be observed then.
    """
    return ~keepout.closed[star, _bin(keepout, time_d)]


def next_open(
    keepout: Keepout, star: Int[Array, ""], time_d: Float[Array, ""]
) -> Float[Array, ""]:
    """The earliest time at or after ``time_d`` at which a star is out of keepout.

    Args:
        keepout: the keepout table.
        star: the star's index.
        time_d: the time, days from the start of the survey.

    Returns:
        ``time_d`` if the star is open then, else the start of its next open bin;
        infinite if it never opens again.
    """
    first = _bin(keepout, time_d)
    bins = jnp.arange(keepout.closed.shape[-1])
    ok = (bins >= first) & ~keepout.closed[star]
    opens = jnp.argmax(ok)
    start = jnp.where(opens == first, time_d, opens * keepout.step_d)
    return jnp.where(jnp.any(ok), jnp.maximum(start, time_d), jnp.inf)


class Design(eqx.Module):
    """The modes of a run, the models that invert their rows, the budget and calendar.

    Attributes:
        models: the model table, static in structure; ``Row.model_slot`` indexes it,
            and so does every mode's ``model_slot``.
        modes: the mode table; ``Action.mode`` indexes it.
        resources: the budget and the calendar's end.
        keepout: when each star can be observed, one row per star.
        capacity: the log's row capacity ``W``, static; a run never derives it from
            its budget. Waits (``WAITED``) and refused rows take rows too.

    Raises:
        ValueError: if a mode names a model slot not in ``models``, or the capacity is
            not positive.
    """

    models: tuple[AbstractObservationModel, ...]
    modes: tuple[Mode, ...]
    resources: Resources
    keepout: Keepout
    capacity: int = eqx.field(static=True)

    def __check_init__(self):
        """Check that every mode's model is in the table and the log has room."""
        if any(not 0 <= mode.model_slot < len(self.models) for mode in self.modes):
            raise ValueError("a mode's model slot is not in the model table")
        if self.capacity < 1:
            raise ValueError(f"the log capacity is positive, not {self.capacity}")


def mode_models(design: Design) -> tuple[AbstractObservationModel, ...]:
    """The observation model of each mode, read from the design's one table.

    Args:
        design: the design.

    Returns:
        One model per mode, in mode order; ``Menu.mode`` indexes it.
    """
    return tuple(design.models[mode.model_slot] for mode in design.modes)


def model_slots(modes: tuple[Mode, ...], mode: Int[Array, "..."]) -> list[int]:
    """The model slot of each mode in ``mode``, read from the one mode table.

    Args:
        modes: the design's mode table, static.
        mode: concrete mode indices, any shape; read in order.

    Returns:
        The model slot of each mode, in order.

    Raises:
        ValueError: if a mode is not in the mode table.
    """
    indices = [int(m) for m in np.ravel(mode)]
    if any(not 0 <= m < len(modes) for m in indices):
        raise ValueError("a mode is not in the mode table")
    return [modes[m].model_slot for m in indices]
