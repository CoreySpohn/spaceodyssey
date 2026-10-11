"""The observing-context seam: what the agent may know about observability.

A context provider assembles the :class:`planit_py.vocabulary.ObservingContext`
record the policy consumes: which targets are observable now, and what it
costs to get to them. The reference geometry-backed provider (keepout,
visibility windows, slew) arrives with the orbital-mechanics adapter; further
constraint families (thermal, communications, operations) enter as more
registry entries on this seam, never as signature changes.
"""

import abc
from typing import ClassVar, final

import equinox as eqx
import jax.numpy as jnp
from planit_py.vocabulary import ObservingContext

from spaceodyssey.registry import SeamInfo, register


class AbstractContextProvider(eqx.Module):
    """Assembles the agent-visible observing context at one epoch."""

    PROTOCOL_VERSION: ClassVar[int] = 1

    @abc.abstractmethod
    def context(
        self, catalog, time_d: float, observatory_state=None
    ) -> ObservingContext:
        """Return the observing context at the given epoch.

        Args:
            catalog: The target catalog (provider-specific; may be None for
                providers that carry their own target list).
            time_d: Epoch in days of mission time.
            observatory_state: Optional observatory state for providers that
                model time-dependent constraints; reference implementations
                ignore it.
        """


@final
@register(
    "context",
    "always_observable",
    SeamInfo(
        operators={"sample", "propagate", "integrate"},
        produces=("planit_py.ObservingContext",),
        version="1",
        fidelity="none",
        cost_hint="low",
    ),
)
class AlwaysObservable(AbstractContextProvider):
    """The null context: every target observable, constant slew cost.

    Useful as the no-constraints baseline and for closed-form configurations
    where observability windows are handled analytically or not at all.

    Attributes:
        target_ids: The catalog target identifiers to report.
        slew_d: Constant slew-plus-settle time in days charged per target.
    """

    target_ids: tuple[int, ...] = eqx.field(static=True)
    slew_d: float = 0.0

    def context(
        self, catalog, time_d: float, observatory_state=None
    ) -> ObservingContext:
        """Return an all-observable context over the configured targets."""
        n = len(self.target_ids)
        return ObservingContext(
            time_d=time_d,
            target_ids=jnp.asarray(self.target_ids),
            observable=jnp.ones(n, dtype=bool),
            slew_d=jnp.full(n, self.slew_d),
        )
