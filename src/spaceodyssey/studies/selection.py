"""Select one package per star using a threshold price for each resource."""

# Ruff parses jaxtyping dimensions as forward-reference Python expressions.
# ruff: noqa: F722

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Bool, Float, Int

from spaceodyssey.core.enums import AdmissionCode


class Selection(eqx.Module):
    """Resource prices, flat ``g * V + v`` choices (-1 for none), and costs.

    Prices have units of gain per resource unit; selected_time uses each
    resource column's declared units, which need not all be days.
    """

    price: Float[jax.Array, " R"]
    choice: Int[jax.Array, " N"]
    selected_time: Float[jax.Array, " R"]


@eqx.filter_jit
def select(
    gain: Float[jax.Array, "N G V"],
    time: Float[jax.Array, "N G V R"],
    capacity: Float[jax.Array, " R"],
    n_bisect: int = 48,
    sweeps: int | None = None,
) -> Selection:
    """Choose positive-gain packages with strictly positive net value.

    Costs must be nonnegative, in the same units as capacity. Non-finite
    costs or gains exclude an option. At equal net value, none wins a zero
    tie; positive ties use the first flat option. Infinite capacities are
    unpriced. Each coordinate uses the feasible upper endpoint of a fixed
    bisection, with ``sweeps`` passes (default: resource count).

    Coordinate updates can violate earlier limits. After pricing, remove
    the lowest-gain selected package consuming an oversubscribed resource
    until all columns fit; ties remove the lowest star index. This final
    pruning is conservative and does not promise a multi-resource optimum.
    """
    n_stars, _, _ = gain.shape
    n_resources = capacity.shape[0]
    flat_gain = gain.reshape(n_stars, -1)
    flat_time = time.reshape(n_stars, -1, n_resources)
    valid = (
        (flat_gain > 0)
        & jnp.isfinite(flat_gain)
        & jnp.all(jnp.isfinite(flat_time) & (flat_time >= 0), axis=-1)
    )
    safe_time = jnp.where(valid[..., None], flat_time, 0.0)
    safe_gain = jnp.where(valid, flat_gain, 0.0)
    stars = jnp.arange(n_stars)

    def costs_of(choice):
        costs = safe_time[stars, jnp.maximum(choice, 0)]
        return jnp.where((choice >= 0)[:, None], costs, 0.0)

    def choices_at(price):
        net = safe_gain - jnp.sum(safe_time * price, axis=-1)
        net = jnp.where(valid, net, -jnp.inf)
        choice = jnp.argmax(net, axis=-1).astype(jnp.int64)
        return jnp.where(jnp.max(net, axis=-1) > 0, choice, -1)

    def coordinate(index, price):
        column = index % n_resources
        cost = safe_time[..., column]
        ratio = safe_gain / jnp.where(cost > 0, cost, 1.0)
        ratio = jnp.where(cost > 0, ratio, 0.0)
        upper = jnp.max(jnp.minimum(ratio, jnp.finfo(gain.dtype).max))
        upper = jnp.minimum(jnp.nextafter(upper, jnp.inf), jnp.finfo(gain.dtype).max)
        unpriced = price.at[column].set(0.0)

        def fits(value):
            choice = choices_at(unpriced.at[column].set(value))
            return jnp.sum(costs_of(choice)[:, column]) <= capacity[column]

        def bisect(_, bounds):
            lower, upper = bounds
            midpoint = lower + (upper - lower) * 0.5
            feasible = fits(midpoint)
            return jnp.where(feasible, lower, midpoint), jnp.where(
                feasible, midpoint, upper
            )

        _, threshold = jax.lax.fori_loop(
            0, n_bisect, bisect, (jnp.asarray(0.0, gain.dtype), upper)
        )
        return unpriced.at[column].set(jnp.where(fits(0.0), 0.0, threshold))

    price = jax.lax.fori_loop(
        0,
        (n_resources if sweeps is None else sweeps) * n_resources,
        coordinate,
        jnp.zeros_like(capacity),
    )
    choice = choices_at(price)

    def prune(_, choice):
        costs = costs_of(choice)
        violated = costs.sum(axis=0) > capacity
        removable = (choice >= 0) & jnp.any((costs > 0) & violated, axis=-1)
        values = safe_gain[stars, jnp.maximum(choice, 0)]
        star = jnp.argmin(jnp.where(removable, values, jnp.inf))
        return choice.at[star].set(jnp.where(jnp.any(removable), -1, choice[star]))

    choice = jax.lax.fori_loop(0, n_stars, prune, choice)
    return Selection(
        price=price, choice=choice, selected_time=costs_of(choice).sum(axis=0)
    )


def act(
    selection: Selection,
    gain: Float[jax.Array, "N G V"],
    time: Float[jax.Array, "N G V R"],
    current_node: Int[jax.Array, " N"],
    *,
    ranking_resource: int,
) -> tuple[Int[jax.Array, ""], Int[jax.Array, ""], Bool[jax.Array, ""]]:
    """Start the selected option at its current node with highest gain/cost.

    ``ranking_resource`` explicitly names the cost column used for ranking;
    callers with telescope costs should pass their named telescope_days
    column. Zero-cost positive gain ranks first. Ties use the lowest star
    index. Waiting returns ``(-1, -1, False)``; option is flat ``g * V + v``.
    """
    n_stars, _, n_variants = gain.shape
    choice = selection.choice
    safe_choice = jnp.maximum(choice, 0)
    stars = jnp.arange(n_stars)
    value = gain.reshape(n_stars, -1)[stars, safe_choice]
    cost = time.reshape(n_stars, -1, time.shape[-1])[
        stars, safe_choice, ranking_resource
    ]
    eligible = (choice >= 0) & (choice // n_variants == current_node)
    ratio = jnp.where(cost > 0, value / jnp.where(cost > 0, cost, 1.0), jnp.inf)
    star = jnp.argmax(jnp.where(eligible, ratio, -jnp.inf)).astype(jnp.int64)
    starts = jnp.any(eligible)
    return jnp.where(starts, star, -1), jnp.where(starts, choice[star], -1), starts


def capacity_of(
    balance: Float[jax.Array, " R"],
    days_left: Float[jax.Array, ""],
    telescope_column: int,
) -> Float[jax.Array, " R"]:
    """Cap the named telescope balance (days) by remaining mission days."""
    return balance.at[telescope_column].set(
        jnp.minimum(balance[telescope_column], days_left)
    )


def trace_scalars(
    trace: Float[jax.Array, "T 6"],
    *,
    final_balance_d: Float[jax.Array, ""],
    admission_code: Int[jax.Array, " A"],
    admission_late: Bool[jax.Array, " A"],
) -> tuple[Int[jax.Array, ""], Float[jax.Array, ""]]:
    """Return late NO_FUNDS count and unspent days with open opportunity.

    Trace columns are price, cumulative charged days, selected days, window
    days, acted, and cumulative value. A positive window-days entry records
    open opportunity. The final ledger telescope balance supplies stranded_d
    when any such opportunity exists, otherwise zero. ``admission_late`` is
    the caller's explicit late-admission mask (padded entries must be false),
    aligned with admission codes; waits alone are not evidence of refusal.
    The historical name early_overspend denotes a count, not days.
    """
    early_overspend = jnp.sum(
        admission_late & (admission_code == AdmissionCode.NO_FUNDS), dtype=jnp.int64
    )
    stranded_d = jnp.where(
        jnp.any(trace[:, 3] > 0), jnp.maximum(final_balance_d, 0.0), 0.0
    )
    return early_overspend, stranded_d
