"""Value and spend: what each star is worth to the planner at a given price of time.

The planner prices exposure time at ``lam`` (reward per day). Each star offers a padded
menu of options, each with a reward and a cost in days, and always the null option
(observe nothing, reward and cost zero). At price ``lam`` a star is worth its best net
reward,

    V_i(lam) = max(0, max_m mask_m (r_m - lam c_m)),

a convex, nonincreasing, piecewise-linear function of ``lam``. Its negative slope is the
time the star would spend at that price. Axis names: ``S`` stars, ``M`` menu entries.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


class Frontier(eqx.Module):
    """One padded menu of options per star.

    Menus are padded to a shared ``M`` with ``mask``, so stars and worlds with shorter
    menus reuse the same compiled program. Padded entries are ignored; their reward and
    cost should still be finite. The null option is implied and is not stored.

    Attributes:
        reward: reward of each option, shape ``(S, M)``.
        cost_d: cost of each option in days, shape ``(S, M)``.
        mask: true where the option exists, shape ``(S, M)``.
    """

    reward: Float[Array, "S M"] = eqx.field(converter=jnp.asarray)
    cost_d: Float[Array, "S M"] = eqx.field(converter=jnp.asarray)
    mask: Bool[Array, "S M"] = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Check that the three arrays share one ``(S, M)`` shape."""
        if not self.reward.shape == self.cost_d.shape == self.mask.shape:
            raise ValueError(
                "reward, cost_d and mask must share one shape, got "
                f"{self.reward.shape}, {self.cost_d.shape} and {self.mask.shape}"
            )
        if self.reward.ndim != 2:
            raise ValueError(f"a frontier is (S, M), got shape {self.reward.shape}")


def _values(lam: Float[Array, ""], f: Frontier) -> Float[Array, " S"]:
    """Per-star value at price ``lam``, with the null option as column zero."""
    net = jnp.where(f.mask, f.reward - lam * f.cost_d, -jnp.inf)
    null = jnp.zeros((*net.shape[:-1], 1), net.dtype)
    return jnp.max(jnp.concatenate([null, net], axis=-1), axis=-1)


def value_and_spend(
    lam: Float[Array, ""], f: Frontier
) -> tuple[Float[Array, " S"], Float[Array, " S"]]:
    """Per-star value at price ``lam`` and the time each star spends there.

    The spend is ``-dV_i/dlam``, taken as one forward tangent of the value itself
    (``jax.jvp``), so the two cannot drift apart. Away from a kink it is the cost of the
    option the star chooses. At a kink, where several options (the null among them)
    tie for the best net reward, ``max`` differentiates by averaging, so the spend is
    the mean cost of the tied options: one point of the subdifferential, not a proof
    that the budget clears (Boyd and Vandenberghe, Convex Optimization, sec. 3.2.3
    and 5.1, for the pointwise maximum and the dual function).

    Args:
        lam: the price of time, reward per day, ``lam >= 0``.
        f: the frontier.

    Returns:
        The value of each star and its spend in days, each of shape ``(S,)``.
    """
    lam = jnp.asarray(lam, jnp.result_type(f.reward.dtype, f.cost_d.dtype, float))
    values, slope = jax.jvp(lambda x: _values(x, f), (lam,), (jnp.ones_like(lam),))
    return values, -slope
