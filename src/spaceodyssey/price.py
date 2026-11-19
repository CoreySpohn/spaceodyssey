"""The price of time: the Lagrangian dual of the budgeted survey and its tie rule.

Choosing at most one option per star within a time budget ``B`` is relaxed by pricing
time at ``lam >= 0``. The dual function

    D(lam) = lam B + sum_i V_i(lam)

bounds the value of every schedule that fits the budget, for every ``lam >= 0`` (weak
duality), and is convex and piecewise linear. Its kinks sit where two options of one
star (the null option among them) have equal net reward, so its minimum over
``lam >= 0`` is attained at zero or at one of those breakpoints. Between two
neighboring breakpoints every ``V_i`` is linear, so the slope of ``D`` there is the
budget less the spend at any interior price, and that slope never decreases.
``solve_price`` sorts the breakpoints and bisects on its sign: exact, with static
shapes, vmappable over worlds, and ``O(log)`` spend evaluations. ``dispatch`` turns a
price into the options bought. Reference: Boyd and Vandenberghe, Convex Optimization,
ch. 5.

An option whose reward or cost is not finite (NaN: a role the reweighting gate refused,
upstream) is refused, never priced: ``solve_price`` and ``dispatch`` drop it and return
the stars that offered one, so the caller can refit them, and the other stars are priced
as if it were absent. One refused star never stops the batch.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from spaceodyssey._numerics import TIE_ULPS
from spaceodyssey.value import Frontier, value_and_spend


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


def _finite(f: Frontier) -> tuple[Frontier, Bool[Array, " S"]]:
    """The frontier with every non-finite option refused, and the stars that had one.

    An offered option whose reward or cost is not finite is masked out and its entries
    set to zero, so it is neither a breakpoint nor a choice.
    """
    usable = f.mask & jnp.isfinite(f.reward) & jnp.isfinite(f.cost_d)
    refused = jnp.any(f.mask & ~usable, axis=-1)
    clean = Frontier(
        jnp.where(usable, f.reward, 0.0), jnp.where(usable, f.cost_d, 0.0), usable
    )
    return clean, refused


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
) -> tuple[Float[Array, ""], Float[Array, ""], Bool[Array, " S"]]:
    """Minimize the dual over ``lam >= 0``.

    Only finite options are priced: an option whose reward or cost is not finite is
    refused (the stars that offered one are returned), and the price and the bound are
    those of the frontier without it.

    The sorted breakpoints ``p_0 = 0 <= p_1 <= ...`` split the prices into intervals
    on which ``D`` is linear with slope ``capacity_d - spend``, the spend read at a
    probe inside the interval (past the last breakpoint for the last one). The slope
    never decreases, so bisection finds the first breakpoint whose interval to the right
    spends less than the budget. That breakpoint minimizes ``D``: the dual falls (or is
    flat) up to it and rises after it. On a flat stretch it is the right end, where the
    options each star takes just above the price fit the budget, so ``dispatch`` returns
    a schedule within it. A probe that lands on a repeated breakpoint reads an averaged
    spend inside the subdifferential, so the spend at the probes still never increases.

    Args:
        f: the frontier.
        capacity_d: the time budget in days.

    Returns:
        The minimizing price; the dual at it, the relaxation bound: no schedule of the
        finite options within the budget earns more; and the refused stars, true for a
        star with an offered option that is not finite, shape ``(S,)``. When the budget
        cannot all be used profitably the price is zero.
    """
    f, refused = _finite(f)
    prices = jnp.sort(_candidates(f))
    after = jnp.concatenate([prices[1:], 2.0 * prices[-1:] + 1.0])
    probe = 0.5 * (prices + after)
    capacity_d = jnp.asarray(capacity_d, prices.dtype)

    def halve(_, bracket):
        low, high = bracket
        mid = (low + high) // 2
        under = jnp.sum(value_and_spend(probe[mid], f)[1]) < capacity_d
        low = jnp.where(under, low, jnp.minimum(mid + 1, high))
        high = jnp.where(under, mid, high)
        return low, high

    n_price = prices.shape[0]
    first, _ = jax.lax.fori_loop(0, n_price.bit_length(), halve, (0, n_price - 1))
    lam = prices[first]
    return lam, dual(lam, f, capacity_d), refused


def dispatch(
    lam: Float[Array, ""], f: Frontier, capacity_d: Float[Array, ""]
) -> tuple[Bool[Array, "S M"], Bool[Array, " S"]]:
    """The options bought at price ``lam`` with budget ``capacity_d``.

    An option whose reward or cost is not finite is refused, never bought; the stars
    that offered one are returned beside the purchase, as by ``solve_price``.

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
        True for each bought option, at most one per star, shape ``(S, M)``, and the
        refused stars, shape ``(S,)``.
    """
    f, refused = _finite(f)
    reward, cost, mask = _with_null(f)
    dtype = jnp.result_type(reward.dtype, cost.dtype, float)
    reward, cost = reward.astype(dtype), cost.astype(dtype)
    lam = jnp.asarray(lam, dtype)
    net = jnp.where(mask, reward - lam * cost, -jnp.inf)
    best = jnp.max(net, axis=-1, keepdims=True)
    scale = jnp.where(mask, jnp.abs(reward) + lam * cost, 0.0)
    tol = TIE_ULPS * jnp.finfo(dtype).eps * jnp.max(scale, axis=-1, keepdims=True)
    tied = mask & (net >= best - tol)
    base = jnp.argmin(jnp.where(tied, cost, jnp.inf), axis=-1)
    base_cost = jnp.take_along_axis(cost, base[:, None], axis=-1)[:, 0]
    capacity_d = jnp.asarray(capacity_d, dtype)
    slack = TIE_ULPS * jnp.finfo(dtype).eps * jnp.abs(capacity_d)

    def upgrade(left, star):
        tied_s, cost_s, base_s, base_cost_s = star
        fits = tied_s & (lam > 0) & (cost_s - base_cost_s <= left + slack)
        dearest = jnp.argmax(jnp.where(fits, cost_s, -jnp.inf))
        pick = jnp.where(fits.any(), dearest, base_s)
        return left - (cost_s[pick] - base_cost_s), pick

    left = capacity_d - jnp.sum(base_cost)
    _, choice = jax.lax.scan(upgrade, left, (tied, cost, base, base_cost))
    bought = choice[:, None] == jnp.arange(cost.shape[-1])
    return bought[:, 1:], refused
