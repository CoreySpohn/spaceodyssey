"""Roles: the priors under which the same weighted draws are read.

A role is a prior with its own weights over the draws of one refit. The truth draws the
worlds; the forecast guides the planner; the verdict members judge. Every role reads
the same cloud, so a role is carried by its prior density alone, read as a ratio to
the refit's own prior on one chart, and one prior type (``AbstractRolePrior``) serves
all three (a forecast may be the truth's prior).
"""

import equinox as eqx

from spaceodyssey.types import AbstractRolePrior


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
