"""The design of a run: its modes, the one table from mode to model, and the budget.

A mode is what a policy chooses when it observes (an instrument setting); an observation
model is what inverts the rows a mode returns. Several modes may share one model. The
design holds the one table from mode to model: every row's ``model_slot`` and every
menu entry's model are read from it, so no record states a slot of its own.

The host unpacks a design into narrow pieces (the model table, one mode, the budget);
no kernel takes a ``Design``.
"""

import equinox as eqx
import numpy as np
from jaxtyping import Array, Float, Int

from spaceodyssey.types import AbstractObservationModel, as_float


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
    """What a run may spend.

    Attributes:
        budget_d: the observing time of the run, days. The grant is charged nothing.
    """

    budget_d: Float[Array, ""] = eqx.field(converter=as_float)


class Design(eqx.Module):
    """The modes of a run, the models that invert their rows, and the budget.

    Attributes:
        models: the model table, static in structure; ``Row.model_slot`` indexes it,
            and so does every mode's ``model_slot``.
        modes: the mode table; ``Action.mode`` indexes it.
        resources: the budget.
        capacity: the log's row capacity ``W``, static; a run never derives it from
            its budget.

    Raises:
        ValueError: if a mode names a model slot not in ``models``, or the capacity is
            not positive.
    """

    models: tuple[AbstractObservationModel, ...]
    modes: tuple[Mode, ...]
    resources: Resources
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
