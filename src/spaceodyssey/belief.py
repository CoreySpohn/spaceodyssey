"""Belief: weighted draws, a weight vector per role, and the probabilities read off."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gammaln, xlogy
from jaxtyping import Array, Float, Int

from spaceodyssey.claim import Claim
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


def _role_weights(atoms: Atoms, log_prior, log_like) -> Float[Array, "R A"]:
    """Normalized weights of every role over the leading axis of ``atoms``."""
    cloud = atoms.cloud
    log_w = cloud.log_weight + log_prior - cloud.log_ref_density + log_like
    return jax.nn.softmax(log_w, axis=-1)


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


def _ratio(num, den):
    """``num / den``, defined as zero where ``den`` is zero, with finite gradients."""
    safe = jnp.where(den > 0, den, 1.0)
    return jnp.where(den > 0, num / safe, 0.0)


def _joint_mass(b: Belief, claim: Claim, star, planet) -> Float[Array, "R C1"]:
    """Probability that the first ``c`` conjuncts hold, ``c = 0 .. C``, per role."""
    probs = jnp.stack([c.prob(b.atoms, star) for c in claim.conjuncts])  # (C, A, K)
    one_planet = jnp.take(probs, planet, axis=2)  # (C, A)
    joint = jnp.cumprod(one_planet, axis=0)  # (C, A)
    mass = weights(b) @ joint.T  # (R, C)
    return jnp.concatenate([jnp.ones_like(mass[:, :1]), mass], axis=1)


def q(b: Belief, claim: Claim, star, planet: Int[Array, ""]) -> Float[Array, " R"]:
    """Probability that the claim holds on one planet, per role.

    This is the weighted share of draws in which the claim holds. For a conditional
    claim it is ``P(claim and given) / P(given)``.

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


def _poisson(lam, n_bins: int) -> Float[Array, " N"]:
    """Poisson pmf on ``0 .. n_bins - 2`` with the remaining mass in the last bin."""
    n = jnp.arange(n_bins - 1)
    head = jnp.exp(xlogy(n, lam) - lam - gammaln(n + 1.0))
    return jnp.append(head, jnp.maximum(1.0 - head.sum(), 0.0))


def count_pmf(b: Belief, claim: Claim, star, n_max: int) -> Float[Array, "R N"]:
    """Distribution of the number of claim-true planets at a star, per role.

    The count is the sum of two independent parts: the draws' planets (a
    Poisson-binomial per draw, mixed with the role weights) and the pool (a Poisson
    count with mean ``exp(log_mass) * pbar``, where ``pbar`` is the claim probability
    averaged over the marks with the role's own mark weights). Mass at or beyond
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

    p_draw = jnp.prod(
        jnp.stack([c.prob(b.atoms, star) for c in claim.conjuncts]), axis=0
    )
    per_draw = _poisson_binomial(p_draw, n_bins)  # (A, N)
    from_draws = weights(b) @ per_draw  # (R, N)

    marks = b.pool.marks
    p_mark = jnp.prod(
        jnp.stack([c.prob(marks, star) for c in claim.conjuncts]), axis=0
    )[:, 0]
    mark_weights = _role_weights(marks, b.pool.log_prior, b.pool.log_like)  # (R, Am)
    lam = jnp.exp(b.pool.log_mass) * (mark_weights @ p_mark)  # (R,)
    from_pool = jax.vmap(_poisson, in_axes=(0, None))(lam, n_bins)  # (R, N)

    full = jax.vmap(jnp.convolve)(from_draws, from_pool)  # (R, 2N - 1)
    return jnp.concatenate(
        [full[:, : n_bins - 1], full[:, n_bins - 1 :].sum(1, keepdims=True)], 1
    )
