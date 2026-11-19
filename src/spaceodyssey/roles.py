"""Roles: the priors under which the same weighted draws are read.

A role is a prior with its own weights over the draws of one refit. The truth draws the
worlds; the forecast guides the planner; the verdict members judge. Every role reads
the same cloud, so a role is carried by its prior density alone, taken on the cloud's
chart, and one prior type serves all three (a forecast may be the truth's prior).
"""

import abc

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from spaceodyssey.types import Atoms


class AbstractRolePrior(eqx.Module):
    """A role prior: a joint density over whole draws on a declared chart.

    The density is taken in the coordinates named by ``chart``. A prior and a cloud
    agree on a chart, or the change of variables (the log Jacobian) is missing from
    one side and the role weights are tilted. The density covers a whole draw (its
    planets, their states and gas blocks, and which slots exist) and need not factor
    over them, which a law over one quantity does not express.

    Attributes:
        chart: the coordinates in which the density is taken, a static label.
    """

    chart: str = eqx.field(static=True)

    @abc.abstractmethod
    def log_prob(self, atoms: Atoms, star) -> Float[Array, " A"]:
        """Log prior density of each draw on ``chart``.

        A draw outside the prior's support returns ``-inf`` (through ``jnp.where``, so
        the gradient stays finite), never ``log(0)``.

        Args:
            atoms: the weighted draws.
            star: a pytree of scalar leaves describing the star.

        Returns:
            Log densities of shape ``(A,)``.
        """

    @abc.abstractmethod
    def sample(self, key: PRNGKeyArray, star) -> Atoms:
        """Draw one point state of a star from the prior.

        Args:
            key: the key of the draw.
            star: a pytree of scalar leaves describing the star.

        Returns:
            Atoms holding one draw (``A = 1``) with a point gas block (``x_var = 0``),
            on this prior's ``chart``.
        """


class Roles(eqx.Module):
    """The forecast and the verdict members.

    The verdict members are one prior whose array leaves are stacked along a leading
    axis of length ``R - 1``; they share one static structure (and one ``chart``).
    In the role axis ``R`` the forecast is row 0 and the members are rows 1 to
    ``R - 1``.

    Attributes:
        forecast: the prior that guides the planner.
        verdict: the stacked prior of the verdict members.
    """

    forecast: AbstractRolePrior
    verdict: AbstractRolePrior
