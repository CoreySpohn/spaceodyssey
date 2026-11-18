"""Belief: weighted draws, a weight vector per role, and the probabilities read off."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gammainc
from jaxtyping import Array, Float, Int

from spaceodyssey.claim import Claim
from spaceodyssey.roles import Roles
from spaceodyssey.types import Atoms, Pool


class Belief(eqx.Module):
    """What one role set believes about one star: draws, role weights, and the pool.

    Attributes:
        atoms: the weighted draws of the refit.
        log_prior: log prior density of each draw under each role, shape ``(R, A)``.
        log_like: summed log likelihood of the rows each role has seen, ``(R, A)``.
        pool: the undetected planets.
    """

    atoms: Atoms
    log_prior: Float[Array, "R A"] = eqx.field(converter=jnp.asarray)
    log_like: Float[Array, "R A"] = eqx.field(converter=jnp.asarray)
    pool: Pool


def _role_log_weights(atoms: Atoms, log_prior, log_like) -> Float[Array, "R A"]:
    """Unnormalized log weights of every role over the leading axis of ``atoms``.

    A draw with zero refit weight, zero prior or zero likelihood has log weight
    ``-inf``. The ``jnp.where`` keeps it there even when ``log_ref_density`` is also
    ``-inf`` (which would otherwise give ``-inf - (-inf) = nan``); the sum is linear,
    so the untaken branch passes no nan into the gradient.
    """
    cloud = atoms.cloud
    dead = (
        jnp.isneginf(cloud.log_weight)
        | jnp.isneginf(log_prior)
        | jnp.isneginf(log_like)
    )
    total = cloud.log_weight + log_prior - cloud.log_ref_density + log_like
    return jnp.where(dead, -jnp.inf, total)


def _role_weights(atoms: Atoms, log_prior, log_like) -> Float[Array, "R A"]:
    """Normalized weights of every role over the leading axis of ``atoms``."""
    return jax.nn.softmax(_role_log_weights(atoms, log_prior, log_like), axis=-1)


def role_log_weights(
    atoms: Atoms,
    roles: Roles,
    log_like: Float[Array, "R A"],
    star,
    pool: Pool,
) -> Belief:
    """Fill each role's prior over the draws and return the belief.

    The weight of role ``r`` is
    ``log w_r = log_weight + log_prior[r] - log_ref_density + log_like[r]``: the refit
    weight, the role prior, minus the refit target ``p0``, plus the role's likelihood.

    Preconditions: ``log_weight`` is against ``p0`` times the incoming likelihood, and
    that likelihood is the same law for every role; every prior and ``p0`` share a
    measure (checked here); the proposal covers the support of each role's prior times
    its likelihood. A draw outside a prior's support has zero weight under that role.

    Args:
        atoms: the weighted draws of one refit.
        roles: the forecast and the stacked verdict members.
        log_like: summed log likelihood of the rows each role has seen, ``(R, A)``.
        star: a pytree of scalar leaves describing the star.
        pool: the undetected planets, which carry their own per-role mark weights.

    Returns:
        The belief with ``log_prior`` of shape ``(R, A)``, forecast in row 0.

    Raises:
        ValueError: if a prior's ``measure`` differs from the cloud's.
    """
    for name, prior in (("forecast", roles.forecast), ("verdict", roles.verdict)):
        if prior.measure != atoms.cloud.measure:
            raise ValueError(
                f"the {name} prior is on measure {prior.measure!r} but the cloud is "
                f"on {atoms.cloud.measure!r}"
            )
    forecast = roles.forecast.log_density(atoms, star)
    verdict = eqx.filter_vmap(
        lambda prior: prior.log_density(atoms, star), in_axes=eqx.if_array(0)
    )(roles.verdict)
    log_prior = jnp.concatenate([forecast[None], verdict], axis=0)
    return Belief(atoms, log_prior, log_like, pool)


def weights(b: Belief) -> Float[Array, "R A"]:
    """Normalized role weights over the draws.

    The unnormalized log weight of role ``r`` is
    ``log_weight + log_prior[r] - log_ref_density + log_like[r]``, and each row is
    normalized to sum to one.

    Args:
        b: the belief.

    Returns:
        Weights of shape ``(R, A)``.
    """
    return _role_weights(b.atoms, b.log_prior, b.log_like)


# Full float32 precision for every product of weights: on a GPU the default precision
# of a float32 dot or convolution can drop to TF32, about three digits.
_EXACT = lax.Precision.HIGHEST


def _ratio(num, den):
    """``num / den``, defined as zero where ``den`` is zero, with finite gradients."""
    safe = jnp.where(den > 0, den, 1.0)
    return jnp.where(den > 0, num / safe, 0.0)


def _conjunct_probs(atoms: Atoms, claim: Claim, star) -> Float[Array, "C A K"]:
    """Probability of each conjunct per draw and planet; zero where no planet exists.

    The ``jnp.where`` keeps a padded slot's state out of the values. The result has the
    gas block's float dtype (or a wider one a conjunct returns), so a conjunct that
    returns booleans is counted as 0 and 1.
    """
    probs = jnp.stack([c.prob(atoms, star) for c in claim.conjuncts])
    return jnp.where(atoms.exists, probs, jnp.zeros((), atoms.x_mean.dtype))


def _draw_joint(b: Belief, claim: Claim, star, planet) -> Float[Array, "C1 A"]:
    """Per draw, the probability that the first ``c`` conjuncts hold, ``c = 0 .. C``."""
    probs = _conjunct_probs(b.atoms, claim, star)  # (C, A, K)
    one_planet = jnp.take(probs, planet, axis=2)  # (C, A)
    joint = jnp.cumprod(one_planet, axis=0)  # (C, A)
    return jnp.concatenate([jnp.ones_like(joint[:1]), joint], axis=0)


def _joint_mass(b: Belief, claim: Claim, star, planet) -> Float[Array, "R C1"]:
    """Probability that the first ``c`` conjuncts hold, ``c = 0 .. C``, per role."""
    return jnp.matmul(
        weights(b), _draw_joint(b, claim, star, planet).T, precision=_EXACT
    )


def q(b: Belief, claim: Claim, star, planet: Int[Array, ""]) -> Float[Array, " R"]:
    """Probability that the claim holds on one planet, per role.

    This is the weighted share of draws in which the claim holds; a draw whose slot
    ``planet`` does not exist counts as claim-false. For a conditional claim it is
    ``P(claim and given) / P(given)``.

    Args:
        b: the belief.
        claim: the claim.
        star: a pytree of scalar leaves describing the star.
        planet: index of the planet along the ``K`` axis.

    Returns:
        ``q`` per role, shape ``(R,)``.
    """
    mass = _joint_mass(b, claim, star, planet)
    return _ratio(mass[:, -1], mass[:, claim.given])


def chain(b: Belief, claim: Claim, star, planet: Int[Array, ""]) -> Float[Array, "R C"]:
    """Conditional probability of each conjunct given the ones before it, per role.

    For an unconditional claim the product over the conjuncts is ``q``. For a
    conditional claim the first ``given`` entries multiply to ``P(given)`` and the rest
    to ``q``.

    Args:
        b: the belief.
        claim: the claim.
        star: a pytree of scalar leaves describing the star.
        planet: index of the planet along the ``K`` axis.

    Returns:
        Probabilities of shape ``(R, C)``.
    """
    mass = _joint_mass(b, claim, star, planet)
    return _ratio(mass[:, 1:], mass[:, :-1])


def _poisson_binomial(p: Float[Array, "A K"], n_bins: int) -> Float[Array, "A N"]:
    """Count distribution of Bernoulli planets per draw, lumped at ``n_bins - 1``."""
    start = jnp.zeros((p.shape[0], n_bins), dtype=p.dtype).at[:, 0].set(1.0)

    def add_planet(pmf, p_k):
        up = jnp.concatenate([jnp.zeros_like(pmf[:, :1]), pmf[:, :-1]], axis=1)
        up = up.at[:, -1].add(pmf[:, -1])  # mass at the cap stays at the cap
        return pmf * (1.0 - p_k[:, None]) + up * p_k[:, None], None

    pmf, _ = lax.scan(add_planet, start, p.T)
    return pmf


def _poisson(lam: Float[Array, ""], n_bins: int) -> Float[Array, " N"]:
    """Poisson pmf on ``0 .. n_bins - 2`` with the remaining mass in the last bin.

    The head is ``exp(-lam) lam^n / n!`` built by the running product of ``lam / n``,
    which has a finite derivative at ``lam = 0``. The tail ``P(N >= n_bins - 1)`` is the
    regularized lower incomplete gamma function ``P(n_bins - 1, lam)``, not one minus
    the head, which cancels to rounding error for a small ``lam`` (in float32, one
    minus the head is about 360 times the tail at ``lam = 1e-3``). For one tail count
    the tail is ``-expm1(-lam)``, whose derivative is finite at ``lam = 0`` where the
    incomplete gamma function's is not.
    """
    if n_bins == 1:
        return jnp.ones(1, lam.dtype)
    k = jnp.arange(1, n_bins - 1, dtype=lam.dtype)
    head = jnp.exp(-lam) * jnp.concatenate(
        [jnp.ones(1, lam.dtype), jnp.cumprod(lam / k)]
    )
    if n_bins == 2:
        tail = -jnp.expm1(-lam)
    else:
        tail = gammainc(jnp.asarray(n_bins - 1, lam.dtype), lam)
    return jnp.append(head, tail)


def pool_rate(
    pool: Pool, claim: Claim, star, time_zero: bool = False
) -> Float[Array, " R"]:
    """Expected number of claim-true planets in the pool, per role.

    The rate of role ``r`` is ``exp(log_mass[r]) * pbar[r]``, where ``pbar[r]`` is the
    claim probability of a mark averaged with role ``r``'s own mark weights. With
    ``time_zero`` it is the same at time zero, from ``log_mass0`` and ``log_like0``.

    Args:
        pool: the undetected planets.
        claim: an unconditional claim.
        star: a pytree of scalar leaves describing the star.
        time_zero: read the pool's state at time zero, static.

    Returns:
        The rate per role, shape ``(R,)``.

    Raises:
        ValueError: if the claim is conditional.
    """
    if claim.given != 0:
        raise ValueError("pool_rate needs an unconditional claim (claim.given == 0)")
    log_mass, log_like = (
        (pool.log_mass0, pool.log_like0)
        if time_zero
        else (pool.log_mass, pool.log_like)
    )
    p_mark = jnp.prod(_conjunct_probs(pool.marks, claim, star), axis=0)[:, 0]  # (Am,)
    mark_weights = _role_weights(pool.marks, pool.log_prior, log_like)  # (R, Am)
    return jnp.exp(log_mass) * jnp.matmul(mark_weights, p_mark, precision=_EXACT)


def count_pmf(b: Belief, claim: Claim, star, n_max: int) -> Float[Array, "R N"]:
    """Distribution of the number of claim-true planets at a star, per role.

    The count is the sum of two independent parts: the draws' planets (a
    Poisson-binomial per draw, mixed with the role weights) and the pool (a Poisson
    count with mean ``exp(log_mass) * pbar``, where ``pbar`` is the claim probability
    averaged over the marks with the role's own mark weights). Only slots that exist
    are counted, among the draws' planets and the marks. Mass at or beyond
    ``n_max`` is lumped into the last bin, so each row has ``n_max + 1`` bins and sums
    to one.

    Independence between the pool and the draws' planets is exact only for an
    unpruned Poisson process.

    Args:
        b: the belief.
        claim: an unconditional claim (``claim.given == 0``).
        star: a pytree of scalar leaves describing the star.
        n_max: the count at which mass is lumped, static.

    Returns:
        Probabilities of shape ``(R, n_max + 1)``.

    Raises:
        ValueError: if the claim is conditional.
    """
    if claim.given != 0:
        raise ValueError("count_pmf needs an unconditional claim (claim.given == 0)")
    n_bins = n_max + 1

    p_draw = jnp.prod(_conjunct_probs(b.atoms, claim, star), axis=0)  # (A, K)
    per_draw = _poisson_binomial(p_draw, n_bins)  # (A, N)
    from_draws = jnp.matmul(weights(b), per_draw, precision=_EXACT)  # (R, N)

    lam = pool_rate(b.pool, claim, star)  # (R,)
    from_pool = jax.vmap(_poisson, in_axes=(0, None))(lam, n_bins)  # (R, N)

    full = jax.vmap(partial(jnp.convolve, precision=_EXACT))(from_draws, from_pool)
    return jnp.concatenate(
        [full[:, : n_bins - 1], full[:, n_bins - 1 :].sum(1, keepdims=True)], 1
    )
