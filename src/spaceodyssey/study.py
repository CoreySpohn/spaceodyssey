"""A study: the claim, the truth and the roles, each star's starting draws, the design.

A study is declared once and run in arms. An arm is the data given at time zero (a
grant); what a planner does with the rows after it is the planner's own
configuration.
"""

import equinox as eqx
from jaxtyping import Array, Float, PyTree

from spaceodyssey.claim import Claim
from spaceodyssey.design import Design
from spaceodyssey.roles import Roles
from spaceodyssey.types import (
    AbstractRefitter,
    AbstractRolePrior,
    Grant,
    Pool,
    Refit,
    as_float,
)


class Arm(eqx.Module):
    """One arm of a study: its grant.

    Whether a planner's forecast learns from the released rows is the planner's own
    static field (see ``AbstractPolicy``), not the arm's: a table reads no forecast.

    Attributes:
        grant: the rows given at time zero.
    """

    grant: Grant


class Study(eqx.Module):
    """A declared study, shared by its arms.

    The forecast is a role prior like any other: a forecast equal to the truth's prior
    is written ``forecast=truth`` and is the same record. The roles are assembled here,
    once, with the forecast in row 0 and the verdict members after it.

    Attributes:
        claim: the claim the study scores.
        truth: the truth's role prior, from which the worlds are drawn.
        roles: the forecast and the stacked verdict members.
        refits: each star's refit at time zero, leaves with a leading star axis ``S``
            (``star_index`` the star's own index).
        pool: each star's undetected planets at time zero, leading axis ``S``.
        stars: the stars, a pytree whose leaves have a leading axis ``S``.
        design: the modes, the model table, the budget and the log capacity.
        prior_mass_outside: each role's prior mass outside each star's refit box,
            shape ``(S, R)``.
        alpha: the risk of a certificate, static.
        refitter: the producer of a star's refit when its rows find a planet its
            cloud holds no track for (``AbstractRefitter``); ``score`` refits with it,
            and a planner is handed the same one. None for a study whose clouds hold
            every planet a run can detect.

    Raises:
        ValueError: if the refits, the pools, ``prior_mass_outside`` and the design's
            keepout table disagree on the number of stars, the pools and
            ``prior_mass_outside`` on the number of roles, or the refits' row masks
            are not of the log's capacity.
    """

    claim: Claim
    truth: AbstractRolePrior
    roles: Roles
    refits: Refit
    pool: Pool
    stars: PyTree
    design: Design
    prior_mass_outside: Float[Array, "S R"]
    alpha: float = eqx.field(static=True)
    refitter: AbstractRefitter | None

    def __init__(
        self,
        claim: Claim,
        truth: AbstractRolePrior,
        forecast: AbstractRolePrior,
        verdict: AbstractRolePrior,
        refits: Refit,
        pool: Pool,
        stars: PyTree,
        design: Design,
        prior_mass_outside: Float[Array, "S R"],
        alpha: float,
        *,
        refitter: AbstractRefitter | None = None,
    ):
        """Declare a study; the forecast and the verdict members become its roles."""
        self.claim = claim
        self.truth = truth
        self.roles = Roles(forecast, verdict)
        self.refits = refits
        self.pool = pool
        self.stars = stars
        self.design = design
        self.prior_mass_outside = as_float(prior_mass_outside)
        self.alpha = float(alpha)
        self.refitter = refitter

    def __check_init__(self):
        """Check one ``S`` and one ``R`` across the per-star fields, and W."""
        n_star = {
            self.refits.star_index.shape[0],
            self.pool.log_mass.shape[0],
            self.prior_mass_outside.shape[0],
            self.design.keepout.closed.shape[0],
        }
        if len(n_star) != 1:
            raise ValueError(f"per-star fields disagree on S: {n_star}")
        n_role = {self.pool.log_mass.shape[-1], self.prior_mass_outside.shape[-1]}
        if len(n_role) != 1:
            raise ValueError(f"pool and prior_mass_outside disagree on R: {n_role}")
        if self.refits.row_mask.shape[-1] != self.design.capacity:
            raise ValueError(
                f"refit row masks hold {self.refits.row_mask.shape[-1]} rows, the log "
                f"{self.design.capacity}"
            )
