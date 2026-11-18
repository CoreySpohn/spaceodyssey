"""The price of time: the Lagrangian dual of the budgeted survey and its tie rule.

Choosing at most one option per star within a time budget ``B`` is relaxed by pricing
time at ``lam >= 0``. The dual function

    D(lam) = lam B + sum_i V_i(lam)

bounds the value of every schedule that fits the budget, for every ``lam >= 0`` (weak
duality), and is convex and piecewise linear. Its kinks sit where two options of one
star (the null option among them) have equal net reward, so its minimum over
``lam >= 0`` is attained at zero or at one of those breakpoints. ``solve_price``
evaluates ``D`` on all of them and takes the ``argmin`` (of near-ties within
roundoff, the largest price): exact, with static shapes, and vmappable over worlds.
``dispatch`` turns a price into the options bought. Reference: Boyd and Vandenberghe,
Convex Optimization, ch. 5.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from spaceodyssey.value import Frontier, value_and_spend

# Net rewards within this many units of roundoff of the best are a tie.
_TIE_ULPS = 64
# Candidate prices are evaluated in batches of this size to bound memory.
_BATCH = 256


def dual(
    lam: Float[Array, ""], f: Frontier, capacity_d: Float[Array, ""]
) -> Float[Array, ""]:
    """The dual function ``lam * capacity_d + sum_i V_i(lam)``.

    Args:
        lam: the price of time, reward per day, ``lam >= 0``.
        f: the frontier.
        capacity_d: the time budget in days.

    Returns:
        An upper bound on the value of any schedule within the budget.
    """
    values, _ = value_and_spend(lam, f)
    return lam * capacity_d + jnp.sum(values)


def _with_null(f: Frontier):
    """Reward, cost and mask with the null option prepended as column zero."""
    reward = jnp.pad(f.reward, ((0, 0), (1, 0)))
    cost = jnp.pad(f.cost_d, ((0, 0), (1, 0)))
    mask = jnp.pad(f.mask, ((0, 0), (1, 0)), constant_values=True)
    return reward, cost, mask


def _candidates(f: Frontier) -> Float[Array, " P"]:
    """Every breakpoint of the dual, clipped at zero and joined with zero.

    The breakpoint of options ``m`` and ``n`` of one star is the price at which their
    net rewards are equal, ``(r_m - r_n) / (c_m - c_n)``. Each unordered pair is taken
    once (``c_m > c_n``); pairs with equal cost have no breakpoint and are masked out,
    as are pairs with a padded option. Masked entries become zero, already a candidate.
    """
    reward, cost, mask = _with_null(f)
    d_reward = reward[:, :, None] - reward[:, None, :]
    d_cost = cost[:, :, None] - cost[:, None, :]
    pair = mask[:, :, None] & mask[:, None, :] & (d_cost > 0)
    lam = jnp.where(pair, d_reward / jnp.where(pair, d_cost, 1.0), 0.0)
    return jnp.concatenate([jnp.zeros(1, lam.dtype), jnp.maximum(lam, 0.0).ravel()])


def solve_price(
    f: Frontier, capacity_d: Float[Array, ""]
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Minimize the dual over ``lam >= 0``.

    The dual can be flat over an interval of prices, and roundoff then decides which
    end an ``argmin`` lands on. Of the candidates whose dual is within roundoff of the
    least, the largest price is taken: there the dual rises to the right, so the
    options each star takes without a tie, plus the cheapest of its tied ones, fit the
    budget, and ``dispatch`` returns a schedule within it.

    Args:
        f: the frontier.
        capacity_d: the time budget in days.

    Returns:
        The minimizing price and the dual at it. The latter is the relaxation bound: no
        schedule within the budget earns more. When the budget cannot all be used
        profitably the price is zero.
    """
    candidates = _candidates(f)
    duals = jax.lax.map(
        lambda lam: dual(lam, f, capacity_d), candidates, batch_size=_BATCH
    )
    least = jnp.min(duals)
    # Roundoff in D is at most a few ulps of lam B + sum_i |r_i|, and both are bounded
    # by |D| plus the largest reward magnitude of each star.
    reward_scale = jnp.sum(jnp.max(jnp.where(f.mask, jnp.abs(f.reward), 0.0), axis=-1))
    tol = _TIE_ULPS * jnp.finfo(duals.dtype).eps * (jnp.abs(least) + reward_scale)
    best = jnp.argmax(jnp.where(duals <= least + tol, candidates, -jnp.inf))
    return candidates[best], duals[best]


def dispatch(
    lam: Float[Array, ""], f: Frontier, capacity_d: Float[Array, ""]
) -> Bool[Array, "S M"]:
    """The options bought at price ``lam`` with budget ``capacity_d``.

    Each star takes an option of best net reward ``r - lam c``, the null option
    included. A star whose best is unique takes it. Where several options tie, the star
    first takes the cheapest of them (the null option when it ties); the budget left
    after those choices is then handed out in order of star index, lowest first, each
    tied star taking its most expensive tied option that still fits. At a zero price
    tied options have equal reward, so a dearer one gains nothing and none is taken.

    At the minimizing price the untied choices plus the cheapest tied options fit the
    budget (the right derivative of the dual is nonnegative there), so the result is a
    schedule within the budget. At any other price the untied choices are bought
    whatever they cost.

    Args:
        lam: the price of time, reward per day.
        f: the frontier.
        capacity_d: the time budget in days.

    Returns:
        True for each bought option, at most one per star, shape ``(S, M)``.
    """
    reward, cost, mask = _with_null(f)
    dtype = jnp.result_type(reward.dtype, cost.dtype, float)
    reward, cost = reward.astype(dtype), cost.astype(dtype)
    lam = jnp.asarray(lam, dtype)
    net = jnp.where(mask, reward - lam * cost, -jnp.inf)
    best = jnp.max(net, axis=-1, keepdims=True)
    scale = jnp.where(mask, jnp.abs(reward) + lam * cost, 0.0)
    tol = _TIE_ULPS * jnp.finfo(dtype).eps * jnp.max(scale, axis=-1, keepdims=True)
    tied = mask & (net >= best - tol)
    base = jnp.argmin(jnp.where(tied, cost, jnp.inf), axis=-1)
    base_cost = jnp.take_along_axis(cost, base[:, None], axis=-1)[:, 0]
    capacity_d = jnp.asarray(capacity_d, dtype)
    slack = _TIE_ULPS * jnp.finfo(dtype).eps * jnp.abs(capacity_d)

    def upgrade(left, star):
        tied_s, cost_s, base_s, base_cost_s = star
        fits = tied_s & (lam > 0) & (cost_s - base_cost_s <= left + slack)
        dearest = jnp.argmax(jnp.where(fits, cost_s, -jnp.inf))
        pick = jnp.where(fits.any(), dearest, base_s)
        return left - (cost_s[pick] - base_cost_s), pick

    left = capacity_d - jnp.sum(base_cost)
    _, choice = jax.lax.scan(upgrade, left, (tied, cost, base, base_cost))
    bought = choice[:, None] == jnp.arange(cost.shape[-1])
    return bought[:, 1:]
