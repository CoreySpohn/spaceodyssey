"""A study: the claim, the truth and the roles, each star's starting draws, the design.

A study is declared once and run in arms. An arm is the data given at time zero (a
grant) and whether the planner's forecast learns from the rows released after it.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from spaceodyssey.claim import Claim
from spaceodyssey.design import Design
from spaceodyssey.roles import AbstractRolePrior, Roles
from spaceodyssey.types import Grant, Pool, Refit


class Arm(eqx.Module):
    """One arm of a study: its grant and whether its forecast learns.

    The planner reads the forecast and the verdict as two beliefs, each from its own
    replay of the log: the verdict's at the decision time, the forecast's at the
    decision time when ``forecast_updates`` and at time zero otherwise. A frozen
    forecast is therefore the replay at ``time_d = 0``, which holds the grant's rows
    only, and released rows never touch its weights or its draws. With it frozen the
    planner maximizes a frozen-forecast surrogate, which is not a martingale.

    Attributes:
        grant: the rows given at time zero.
        forecast_updates: whether the forecast's replay follows the decision time,
            static.
    """

    grant: Grant
    forecast_updates: bool = eqx.field(static=True)


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

    Raises:
        ValueError: if the refits, the pools and ``prior_mass_outside`` disagree on the
            number of stars, or the refits' row masks are not of the log's capacity.
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
    ):
        """Declare a study; the forecast and the verdict members become its roles."""
        self.claim = claim
        self.truth = truth
        self.roles = Roles(forecast, verdict)
        self.refits = refits
        self.pool = pool
        self.stars = stars
        self.design = design
        self.prior_mass_outside = jnp.asarray(prior_mass_outside, float)
        self.alpha = float(alpha)

    def __check_init__(self):
        """Check that every per-star field has one star axis and the log's capacity."""
        n_star = {
            self.refits.star_index.shape[0],
            self.pool.log_mass.shape[0],
            self.prior_mass_outside.shape[0],
        }
        if len(n_star) != 1:
            raise ValueError(f"per-star fields disagree on S: {n_star}")
        if self.refits.row_mask.shape[-1] != self.design.capacity:
            raise ValueError(
                f"refit row masks hold {self.refits.row_mask.shape[-1]} rows, the log "
                f"{self.design.capacity}"
            )
