"""Roles: the priors under which the same weighted draws are read.

A role is a prior with its own weights over the draws of one refit. The forecast
guides the planner; the verdict members judge. Every role reads the same cloud, so a
role is carried by its prior density alone, taken on the cloud's measure.
"""

import equinox as eqx
from jaxtyping import Array, Float

from spaceodyssey.types import Atoms


class AbstractPrior(eqx.Module):
    """A role prior: a density over a cloud's atoms on a declared measure.

    The density is taken in the coordinates named by ``measure``. A prior and a cloud
    agree on a measure, or the change of variables (the log Jacobian) is missing from
    one side and the role weights are tilted.

    Attributes:
        measure: the coordinates in which the density is taken, a static label.
    """

    measure: str = eqx.field(static=True)

    def log_density(self, atoms: Atoms, star) -> Float[Array, " A"]:
        """Log prior density of each draw on ``measure``.

        A draw outside the prior's support returns ``-inf`` (through ``jnp.where``, so
        the gradient stays finite), never ``log(0)``.

        Args:
            atoms: the weighted draws.
            star: a pytree of scalar leaves describing the star.

        Returns:
            Log densities of shape ``(A,)``.
        """
        raise NotImplementedError


class Roles(eqx.Module):
    """The forecast and the verdict members.

    The verdict members are one prior whose array leaves are stacked along a leading
    axis of length ``R - 1``; they share one static structure (and one ``measure``).
    In the role axis ``R`` the forecast is row 0 and the members are rows 1 to
    ``R - 1``.

    Attributes:
        forecast: the prior that guides the planner.
        verdict: the stacked prior of the verdict members.
    """

    forecast: AbstractPrior
    verdict: AbstractPrior
