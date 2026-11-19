"""Belief: weighted draws, a weight vector per role, and the probabilities read off.

The belief at a star is read off a replay of the log into one refit: ``replay``
applies every row due at a query time that the refit does not already represent,
``absorb`` brings a replay up to a later query time, and ``replay_belief`` reads the
belief off it under a role set. A planner reads two beliefs, the forecast's and the
verdict's, each off its own replay (``planning_beliefs``).
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import gammainc, ndtr, ndtri
from jaxtyping import Array, Bool, Float, Int, PyTree

from spaceodyssey._numerics import EXACT, safe_ratio
from spaceodyssey._standin.diagnostics import kish_ess, pareto_khat
from spaceodyssey.claim import Claim, GasAbove
from spaceodyssey.roles import Roles
from spaceodyssey.types import (
    ACCEPTED,
    AbstractObservationModel,
    AbstractRolePrior,
    Atoms,
    GasNodes,
    ObservationLog,
    Pool,
    Refit,
    real_rows,
)


class Belief(eqx.Module):
    """What one role set believes about one star: draws, role weights, and the pool.

    Attributes:
        atoms: the weighted draws; after a replay, their gas block carries the
            applied rows.
        log_prior_ratio: log of each role's prior over the refit's prior
            (``log pi_r - log pi_0``) at each draw, shape ``(R, A)``; a row of NaN for
            a role the reweighting gate refused (``role_belief``).
        log_like: summed log likelihood of the rows each role has seen, ``(R, A)``.
        pool: the undetected planets.

    Raises:
        ValueError: if ``log_prior_ratio``, ``log_like`` and the pool disagree on the
            number of roles, or the draws on their number.
    """

    atoms: Atoms
    log_prior_ratio: Float[Array, "R A"] = eqx.field(converter=jnp.asarray)
    log_like: Float[Array, "R A"] = eqx.field(converter=jnp.asarray)
    pool: Pool

    def __check_init__(self):
        """Check the static shapes: one ``R`` and one ``A`` across the fields."""
        n_draw = self.atoms.cloud.log_weights.shape[-1]
        roles = {
            self.log_prior_ratio.shape[-2],
            self.log_like.shape[-2],
            self.pool.log_mass.shape[-1],
        }
        draws = {self.log_prior_ratio.shape[-1], self.log_like.shape[-1], n_draw}
        if len(roles) != 1 or len(draws) != 1:
            raise ValueError(
                "belief fields disagree on R or A (log_prior_ratio "
                f"{self.log_prior_ratio.shape}, log_like {self.log_like.shape}, pool R "
                f"{self.pool.log_mass.shape[-1]}, {n_draw} draws)"
            )


def _role_log_weights(atoms: Atoms, log_prior_ratio, log_like) -> Float[Array, "R A"]:
    """Unnormalized log weights of every role over the leading axis of ``atoms``.

    A draw with zero refit weight, zero prior ratio or zero likelihood has log weight
    ``-inf``. The ``jnp.where`` keeps it there even when another term is ``+inf``; the
    sum is linear, so the untaken branch passes no nan into the gradient. A NaN row (a
    refused role) stays NaN.
    """
    log_weights = atoms.cloud.log_weights
    dead = (
        jnp.isneginf(log_weights)
        | jnp.isneginf(log_prior_ratio)
        | jnp.isneginf(log_like)
    )
    total = log_weights + log_prior_ratio + log_like
    return jnp.where(dead, -jnp.inf, total)


def role_belief(
    atoms: Atoms,
    roles: Roles,
    log_like: Float[Array, "R A"],
    star,
    pool: Pool,
    *,
    min_n_eff,
    max_khat,
) -> Belief:
    """The belief of a role set over one refit's draws, each role's prior filled in.

    The weight of role ``r`` (read with ``log_weights`` of the belief) is
    ``log w_r = log_weights + log pi_r - log pi_0 + log_like[r]``: the refit weight,
    the ratio of the role's prior ``pi_r`` to the refit's own prior ``pi_0`` (the
    cloud's ``prior``), both evaluated on the draws, plus the role's likelihood. No old
    likelihood term is evaluated: the refit's rows are in ``log_weights`` already.

    The reweighting is gated per role, the thresholds required: a role whose reweighted
    cloud ``log_weights + log pi_r - log pi_0`` has a Kish size below ``min_n_eff`` or
    an upper-tail Pareto shape above ``max_khat`` (a failed fit is NaN and fails) gets
    a row of NaN, and so does everything read from it; the other roles are unchanged,
    and under ``vmap`` each slice is gated on its own. A role with zero total weight
    fails too. The shape is taken on the reweighted cloud, not on the ratios alone: the
    draws follow the refit's proposal, not ``pi_0``, so the tail that sets the error is
    that of the new weights (for draws of uniform weight the two agree). The gate
    covers the prior swap only; the rows each role has seen are not part of it.
    ``min_n_eff=0, max_khat=inf`` keeps every row (the Pareto fit is then skipped), for
    a caller that reads the diagnostics itself.

    Preconditions: ``log_weights`` is against ``pi_0`` times the incoming likelihood,
    and that likelihood is the same model for every role; every prior and ``pi_0``
    share a chart (checked here); the proposal covers the support of each role's prior
    times its likelihood. A draw outside a prior's support, or outside ``pi_0``'s, has
    zero weight under that role.

    Args:
        atoms: the weighted draws of one refit.
        roles: the forecast and the stacked verdict members.
        log_like: summed log likelihood of the rows each role has seen, ``(R, A)``.
        star: a pytree of scalar leaves describing the star.
        pool: the undetected planets, which carry their own per-role mark weights.
        min_n_eff: the least Kish size of a role's weights.
        max_khat: the largest Pareto shape of a role's reweighted cloud.

    Returns:
        The belief with ``log_prior_ratio`` of shape ``(R, A)``, forecast in row 0.

    Raises:
        ValueError: if a role prior's ``chart`` differs from that of the refit's prior.
    """
    log_prior_ratio = _role_log_prior_ratio(atoms, roles, star, min_n_eff, max_khat)
    return Belief(atoms, log_prior_ratio, log_like, pool)


def _gate(log_weights, ratio, min_n_eff, max_khat, exempt=None) -> Float[Array, "R A"]:
    """``ratio``, with NaN rows for the roles the reweighting gate refuses.

    Both statistics are taken on the reweighted cloud ``log_weights + ratio``, the
    draws of zero weight left out. An infinite ``max_khat`` turns the Pareto check
    off, a failed fit (NaN, too few exceedances) included; a Python ``inf`` is static,
    so the fit is skipped, and a traced one gives the same rows. A row of ``exempt``
    (``(R,)``, or None for none) is kept whatever its statistics.
    """
    dead = jnp.isneginf(log_weights) | jnp.isneginf(ratio)
    total = jnp.where(dead, -jnp.inf, log_weights + ratio)
    ok = kish_ess(total) >= min_n_eff
    if not (isinstance(max_khat, float) and max_khat == math.inf):
        khat_ok = jax.vmap(pareto_khat)(total) <= max_khat
        ok = ok & (khat_ok | jnp.isposinf(max_khat))
    if exempt is not None:
        ok = ok | exempt
    return jnp.where(ok[:, None], ratio, jnp.nan)


def _prior_ratio_rows(
    atoms: Atoms,
    prior: AbstractRolePrior,
    star,
    name: str,
    stacked: bool,
    min_n_eff,
    max_khat,
    exempt=None,
) -> Float[Array, "R A"]:
    """Gated ``log pi_r - log pi_0`` of each draw, a row per member when stacked.

    Checks the chart against that of the refit's prior. A draw outside either prior's
    support gets ``-inf``. Rows in ``exempt`` are not gated.
    """
    base = atoms.cloud.prior
    if prior.chart != base.chart:
        raise ValueError(
            f"the {name} prior is on chart {prior.chart!r} but the refit's prior is "
            f"on {base.chart!r}"
        )
    if stacked:
        log_p = eqx.filter_vmap(
            lambda member: member.log_prob(atoms, star), in_axes=eqx.if_array(0)
        )(prior)
    else:
        log_p = prior.log_prob(atoms, star)[None]
    log_p0 = base.log_prob(atoms, star)
    outside = jnp.isneginf(log_p) | jnp.isneginf(log_p0)
    ratio = jnp.where(outside, -jnp.inf, log_p - log_p0)
    return _gate(atoms.cloud.log_weights, ratio, min_n_eff, max_khat, exempt)


def _role_log_prior_ratio(
    atoms: Atoms, roles: Roles, star, min_n_eff, max_khat, exempt=None
) -> Float[Array, "R A"]:
    """Every role's gated prior ratio of each draw, forecast in row 0.

    ``exempt`` (``(R,)``, or None) names the rows that are not gated.
    """
    head, tail = (None, None) if exempt is None else (exempt[:1], exempt[1:])
    gate = (min_n_eff, max_khat)
    return jnp.concatenate(
        [
            _prior_ratio_rows(
                atoms, roles.forecast, star, "forecast", False, *gate, head
            ),
            _prior_ratio_rows(atoms, roles.verdict, star, "verdict", True, *gate, tail),
        ],
        axis=0,
    )


def role_pool(
    marks: Atoms,
    roles: Roles,
    log_like: Float[Array, "R Am"],
    log_mass: Float[Array, " R"],
    star,
    *,
    min_n_eff,
    max_khat,
) -> Pool:
    """The pool of a role set over the marks of one refit, each role's prior filled in.

    A mark's weight under role ``r`` is formed as a draw's is (``role_belief``):
    ``log_weights + log pi_r - log pi_0 + log_like[r]``, with ``pi_0`` the prior of the
    marks' cloud. The prior ratio is built by the same function and gate as a
    belief's: a role whose reweighted marks fail the gate gets a NaN row, and its rate
    then reads NaN, never zero. A role whose pool is empty (``log_mass = -inf``) is
    never refused: it holds no planet to misread, and its rate is zero.

    Args:
        marks: the marks, one planet each (``K = 1``), on the refit prior's chart.
        roles: the forecast and the stacked verdict members.
        log_like: log likelihood of each mark under each role, ``(R, Am)``.
        log_mass: log expected number of undetected planets per role, ``(R,)``.
        star: a pytree of scalar leaves describing the star.
        min_n_eff: the least Kish size of a role's mark weights.
        max_khat: the largest Pareto shape of a role's reweighted marks.

    Returns:
        The pool, forecast in row 0.

    Raises:
        ValueError: if a role prior's chart differs from that of the marks' prior.
    """
    log_mass = jnp.asarray(log_mass)
    empty = jnp.isneginf(log_mass)
    ratio = _role_log_prior_ratio(marks, roles, star, min_n_eff, max_khat, empty)
    return Pool(marks, ratio, log_like, log_mass)


def log_weights(b: Belief) -> Float[Array, "R A"]:
    """Unnormalized log role weights over the draws.

    The log weight of role ``r`` is ``log_weights + log_prior_ratio[r] + log_like[r]``,
    the first term the cloud's. A draw with zero refit weight, zero prior ratio or zero
    likelihood has log weight ``-inf``; a refused role's row is NaN.

    Args:
        b: the belief.

    Returns:
        Log weights of shape ``(R, A)``.
    """
    return _role_log_weights(b.atoms, b.log_prior_ratio, b.log_like)


def weights(b: Belief) -> Float[Array, "R A"]:
    """Normalized role weights over the draws: ``log_weights`` normalized per role.

    Each row sums to one. A role under which every draw has log weight ``-inf`` cannot
    be normalized: its row is NaN, on purpose, and so is everything read from that
    role (its ``q``, and its pool rate when the same holds for the marks).

    Args:
        b: the belief.

    Returns:
        Weights of shape ``(R, A)``.
    """
    return jax.nn.softmax(log_weights(b), axis=-1)


def mark_weights(pool: Pool) -> Float[Array, "R Am"]:
    """Normalized weights of the pool's marks, each role with its own.

    A mark's log weight under role ``r`` is formed like a draw's, ``log_weights +
    log_prior_ratio[r] + log_like[r]``, and each row is normalized over the marks.

    Args:
        pool: the undetected planets.

    Returns:
        Weights of shape ``(R, Am)``.
    """
    log_w = _role_log_weights(pool.marks, pool.log_prior_ratio, pool.log_like)
    return jax.nn.softmax(log_w, axis=-1)


def _conjunct_probs(atoms: Atoms, claim: Claim, star) -> Float[Array, "C A K"]:
    """Probability of each conjunct per draw and planet; zero where no planet exists.

    The ``jnp.where`` keeps a padded slot's state out of the values. The result has the
    gas block's float dtype (or a wider one a conjunct returns), so a conjunct that
    returns booleans is counted as 0 and 1.
    """
    probs = jnp.stack([c.prob(atoms, star) for c in claim.conjuncts])
    return jnp.where(atoms.exists, probs, jnp.zeros((), atoms.x_mean.dtype))


def draw_claim_mass(
    b: Belief, claim: Claim, star, planet: Int[Array, ""]
) -> Float[Array, "C1 A"]:
    """Per draw, the probability that the first ``c`` conjuncts hold, ``c = 0 .. C``.

    Row 0 is one; row ``C`` is the claim's probability on the draw, and row
    ``claim.given`` its condition's. Every claim probability is these rows weighted by
    a role.

    Args:
        b: the belief.
        claim: the claim.
        star: a pytree of scalar leaves describing the star.
        planet: index of the planet along the ``K`` axis.

    Returns:
        Probabilities of shape ``(C + 1, A)``.
    """
    probs = _conjunct_probs(b.atoms, claim, star)  # (C, A, K)
    one_planet = jnp.take(probs, planet, axis=2)  # (C, A)
    joint = jnp.cumprod(one_planet, axis=0)  # (C, A)
    return jnp.concatenate([jnp.ones_like(joint[:1]), joint], axis=0)


def _joint_mass(b: Belief, claim: Claim, star, planet) -> Float[Array, "R C1"]:
    """Probability that the first ``c`` conjuncts hold, ``c = 0 .. C``, per role."""
    return jnp.matmul(
        weights(b), draw_claim_mass(b, claim, star, planet).T, precision=EXACT
    )


def q(b: Belief, claim: Claim, star, planet: Int[Array, ""]) -> Float[Array, " R"]:
    """Probability that the claim holds on one planet, per role.

    This is the weighted share of draws in which the claim holds; a draw whose slot
    ``planet`` does not exist counts as claim-false. For a conditional claim it is
    ``P(claim and given) / P(given)``, and 0 where ``P(given)`` is 0. That 0 is a
    convention, not a confident negative: read ``q`` of a conditional claim only
    beside its condition mass (``Diagnostics.cond_mass``).

    Args:
        b: the belief.
        claim: the claim.
        star: a pytree of scalar leaves describing the star.
        planet: index of the planet along the ``K`` axis; a negative index counts
            from the end, and one past the end gives NaN.

    Returns:
        ``q`` per role, shape ``(R,)``.
    """
    mass = _joint_mass(b, claim, star, planet)
    return safe_ratio(mass[:, -1], mass[:, claim.given])


def chain(b: Belief, claim: Claim, star, planet: Int[Array, ""]) -> Float[Array, "R C"]:
    """Conditional probability of each conjunct given the ones before it, per role.

    For an unconditional claim the product over the conjuncts is ``q``. For a
    conditional claim the first ``given`` entries multiply to ``P(given)`` and the rest
    to ``q``. An entry whose preceding conjuncts have zero mass is 0, as in ``q``.

    Args:
        b: the belief.
        claim: the claim.
        star: a pytree of scalar leaves describing the star.
        planet: index of the planet along the ``K`` axis.

    Returns:
        Probabilities of shape ``(R, C)``.
    """
    mass = _joint_mass(b, claim, star, planet)
    return safe_ratio(mass[:, 1:], mass[:, :-1])


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


def _pool_masses(pool: Pool, claim: Claim, star) -> Float[Array, "R C1"]:
    """Expected number of pool planets meeting the first ``c`` conjuncts, ``c = 0..C``.

    Per role, the pool's expected number times the mark-weighted probability that the
    first ``c`` conjuncts hold; column 0 is the expected number itself.
    """
    probs = _conjunct_probs(pool.marks, claim, star)[:, :, 0]  # (C, Am)
    joint = jnp.concatenate([jnp.ones_like(probs[:1]), jnp.cumprod(probs, axis=0)])
    weighted = jnp.matmul(mark_weights(pool), joint.T, precision=EXACT)  # (R, C + 1)
    return jnp.exp(pool.log_mass)[:, None] * weighted


def pool_rate(pool: Pool, claim: Claim, star) -> Float[Array, " R"]:
    """Expected number of claim-true planets in the pool, per role.

    The rate of role ``r`` is ``exp(log_mass[r]) * pbar[r]``, where ``pbar[r]`` is the
    probability that a mark meets every conjunct of the claim, averaged with role
    ``r``'s own mark weights. For a conditional claim that is the rate of planets that
    meet the condition and the claim: a claim-true planet is one the claim holds for,
    and it holds only where its condition does. The rate at time zero is this function
    of the pool at time zero.

    Args:
        pool: the undetected planets.
        claim: the claim.
        star: a pytree of scalar leaves describing the star.

    Returns:
        The rate per role, shape ``(R,)``.
    """
    return _pool_masses(pool, claim, star)[:, -1]


def count_pmf(b: Belief, claim: Claim, star, n_max: int) -> Array:
    """Distribution of the number of claim-true planets at a star, per role.

    The count is the sum of two independent parts: the draws' planets (a
    Poisson-binomial per draw, mixed with the role weights) and the pool (a Poisson
    count with mean ``exp(log_mass) * pbar``, where ``pbar`` is the claim probability
    averaged over the marks with the role's own mark weights). Only slots that exist
    are counted, among the draws' planets and the marks. Mass at or beyond
    ``n_max`` is lumped into the last bin, so each axis has ``n_max + 1`` bins.

    A conditional claim (``claim.given > 0``, condition ``C``, claim ``Z``, with ``Z``
    holding only where ``C`` does) counts two things: ``N``, the planets for which
    ``Z`` holds, and ``F``, those that meet ``C`` and fail ``Z``. The result is their
    joint distribution, ``F`` along axis ``-2`` and ``N`` along axis ``-1``. On a draw
    planet ``k`` adds one to ``F`` with probability ``c_k - z_k`` and one to ``N`` with
    ``z_k``, with ``c_k = P(C)`` and ``z_k = P(C, Z)`` on the draw; the pool adds the
    independent Poisson counts ``F ~ Poisson(Lam (cbar - zbar))`` and
    ``N ~ Poisson(Lam zbar)`` (a Poisson process marked by the claim):

        P(F = f, N = n) = sum_a w_a [PB2_a * Pool2](f, n),

    with ``PB2_a`` the two-count Poisson-binomial of draw ``a`` and ``*`` the
    convolution in both counts, each lumped at ``n_max``. The planets that meet the
    condition number ``M = F + N``; a planner scores ``N`` given ``M`` (``LogScore``),
    so it is paid for what it learns about the claim among the planets its condition
    admits, and not for learning the condition alone (for one planet with ``P(C) = c``
    and ``P(Z | C) = z``, learning ``C`` leaves the expected score ``-c h(z)``
    unchanged).

    Independence between the pool and the draws' planets is exact only for an
    unpruned Poisson process.

    Args:
        b: the belief.
        claim: the claim.
        star: a pytree of scalar leaves describing the star.
        n_max: the count at which mass is lumped, static.

    Returns:
        Probabilities of shape ``(R, n_max + 1)`` for an unconditional claim, each row
        summing to one; of shape ``(R, n_max + 1, n_max + 1)`` (``F``, then ``N``) for
        a conditional claim, each role's table summing to one.
    """
    n_bins = n_max + 1
    probs = _conjunct_probs(b.atoms, claim, star)  # (C, A, K)
    p_draw = jnp.prod(probs, axis=0)  # (A, K)
    rates = _pool_masses(b.pool, claim, star)  # (R, C + 1)

    if claim.given == 0:
        per_draw = _poisson_binomial(p_draw, n_bins)  # (A, N)
        from_draws = jnp.matmul(weights(b), per_draw, precision=EXACT)  # (R, N)
        from_pool = jax.vmap(_poisson, in_axes=(0, None))(rates[:, -1], n_bins)
        return _add_counts(from_draws, from_pool)

    p_cond = jnp.prod(probs[: claim.given], axis=0)  # (A, K)
    per_draw = _pair_poisson_binomial(p_cond - p_draw, p_draw, n_bins)  # (A, N, N)
    from_draws = jnp.einsum("ra,afn->rfn", weights(b), per_draw, precision=EXACT)
    poisson = jax.vmap(_poisson, in_axes=(0, None))
    failed = poisson(rates[:, claim.given] - rates[:, -1], n_bins)  # (R, N)
    true = poisson(rates[:, -1], n_bins)
    from_pool = failed[:, :, None] * true[:, None, :]
    return _add_pair_counts(from_draws, from_pool)


def _pair_poisson_binomial(
    p_f: Float[Array, "A K"], p_n: Float[Array, "A K"], n_bins: int
) -> Float[Array, "A N N"]:
    """Joint distribution of two counts per draw, each planet adding to at most one.

    Planet ``k`` adds one to the first count with probability ``p_f``, one to the
    second with ``p_n`` and nothing otherwise; each count is lumped at
    ``n_bins - 1``.
    """
    n_draw = p_f.shape[0]
    start = jnp.zeros((n_draw, n_bins, n_bins), p_f.dtype).at[:, 0, 0].set(1.0)

    def up(pmf, axis):
        moved = jnp.roll(pmf, 1, axis=axis)
        first = [slice(None)] * 3
        first[axis] = 0
        last = [slice(None)] * 3
        last[axis] = -1
        moved = moved.at[tuple(first)].set(0.0)
        return moved.at[tuple(last)].add(pmf[tuple(last)])  # the cap keeps its mass

    def add_planet(pmf, p_k):
        f, n = p_k[0][:, None, None], p_k[1][:, None, None]
        return pmf * (1.0 - f - n) + up(pmf, 1) * f + up(pmf, 2) * n, None

    pmf, _ = lax.scan(add_planet, start, (p_f.T, p_n.T))
    return pmf


def _add_counts(a: Float[Array, "... N"], b: Float[Array, "... N"]) -> Array:
    """Distribution of the sum of two independent counts, lumped at the last bin.

    ``out[n] = sum_{j + k = n} a[j] b[k]`` for ``n < N - 1`` and the rest of the mass,
    ``j + k >= N - 1``, in the last bin: the convolution written out over the static
    ``N x N`` grid of pairs, each pair's product added to the bin
    ``min(j + k, N - 1)`` by one product with a fixed 0/1 matrix. ``N`` is short and
    static, so this is a few hundred multiplications, and under ``vmap`` it batches as
    a plain matrix product (a batched ``jnp.convolve`` becomes a grouped convolution,
    quadratic in the batch on the CPU in float64).
    """
    n_bins = a.shape[-1]
    pair_bin = np.minimum(
        np.add.outer(np.arange(n_bins), np.arange(n_bins)), n_bins - 1
    )
    to_bin = np.eye(n_bins)[pair_bin.reshape(-1)]  # (N N, N), a 0/1 matrix
    pairs = (a[..., :, None] * b[..., None, :]).reshape(*a.shape[:-1], n_bins**2)
    return jnp.matmul(pairs, jnp.asarray(to_bin, a.dtype), precision=EXACT)


def _add_pair_counts(a: Float[Array, "... N N"], b: Float[Array, "... N N"]) -> Array:
    """Distribution of the sum of two independent pairs of counts, each count lumped.

    ``out[f, n] = sum a[f1, n1] b[f2, n2]`` over ``min(f1 + f2, N - 1) = f`` and
    ``min(n1 + n2, N - 1) = n``: the convolution of ``_add_counts`` in both counts,
    by one contraction with its fixed 0/1 table.
    """
    n_bins = a.shape[-1]
    pair_bin = np.minimum(
        np.add.outer(np.arange(n_bins), np.arange(n_bins)), n_bins - 1
    )
    to_bin = jnp.asarray(np.eye(n_bins)[pair_bin], a.dtype)  # (N, N, N), a 0/1 table
    return jnp.einsum(
        "...ij,...kl,ikf,jln->...fn", a, b, to_bin, to_bin, precision=EXACT
    )


class Replay(eqx.Module):
    """The rows of one star replayed into its refit, held as summed statistics.

    ``replay`` applies every row of the log due at a query time, ``absorb`` brings a
    replay up to a later query time, and ``replay_belief`` reads the belief off it.
    Each model slot keeps the leafwise sum of its applied rows' ``stats``; the draws
    stay as the refit returned them and every belief is finalized from them afresh, so
    a replay kept current by ``absorb`` gives the belief that ``replay`` gives on the
    whole log, to rounding. A replay holds no role: one replay serves every role.

    Attributes:
        refit: the refit's draws, the rows they represent and their star.
        stats: one pytree per model slot, the summed statistics of the rows applied
            in that slot, shaped as the model's ``stats`` returns them.
        applied: the log rows applied, shape ``(W,)``, none of them in
            ``refit.row_mask``.
    """

    refit: Refit
    stats: tuple[PyTree, ...]
    applied: Bool[Array, " W"]


def _due(log: ObservationLog, refit: Refit, time_d) -> Bool[Array, " W"]:
    """The log rows a replay of ``refit`` applies at the query time ``time_d``.

    A row is due when it is real (``index < count``), released by the query time
    (``release_d <= time_d``), accepted (``refusal == ACCEPTED``), of the refit's
    star, and not among the rows the refit represents. A row in flight stays in the
    log and falls due once the query time reaches its release.
    """
    rows = log.rows
    return (
        real_rows(log)
        & (rows.release_d <= time_d)
        & (rows.refusal == ACCEPTED)
        & (rows.star == refit.star_index)
        & ~refit.row_mask
    )


def _add_stats(
    total: PyTree,
    model: AbstractObservationModel,
    payload: PyTree,
    keep: Bool[Array, ""],
    atoms: Atoms,
    star,
) -> PyTree:
    """``total`` plus one row's statistics where ``keep``, exact zeros elsewhere.

    A row that is not kept is read as a zero payload, so a model's checks on a row's
    payload see only the rows applied in its slot.
    """
    payload = jax.tree.map(lambda x: jnp.where(keep, x, jnp.zeros_like(x)), payload)
    stats = model.stats(payload, atoms, star)
    return jax.tree.map(
        lambda t, s: t + jnp.where(keep, s, jnp.zeros_like(s)), total, stats
    )


def _summed_stats(
    model: AbstractObservationModel,
    payload: PyTree,
    keep: Bool[Array, " W"],
    atoms: Atoms,
    star,
) -> PyTree:
    """The statistics of the kept rows of one model slot, summed by one scan."""
    one_row = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), payload
    )
    shapes = jax.eval_shape(model.stats, one_row, atoms, star)
    zero = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), shapes)

    def add(total, row):
        return _add_stats(total, model, *row, atoms, star), None

    total, _ = lax.scan(add, zero, (payload, keep))
    return total


def replay(
    refit: Refit,
    log: ObservationLog,
    models: tuple[AbstractObservationModel, ...],
    star,
    time_d: Float[Array, ""],
) -> Replay:
    """Replay the rows of a log due at a query time into a refit.

    A row is applied when it is released by ``time_d`` (``release_d <= time_d``),
    accepted, of the refit's star and outside its ``row_mask``. A row in flight, a row
    in ``row_mask`` (the cloud's weights already carry it), a refused row, a row of
    another star and a row past the log's ``count`` contribute exactly zero. Rows enter
    through their models' summed statistics and ``finalize``, never as a sum of per-row
    likelihoods, so rows reading one uncertain quantity are exact.

    Each model slot is one scan over the log's payload buffer of that slot, statically
    unrolled over ``models``; nothing switches on a traced slot. The sum is
    commutative and does not depend on where in the log a row sits. The scan reads the
    log's whole capacity, so a row slot costs the size of its model's statistics: held
    without the draw axis where they do not depend on the draw (both toy models), an
    empty or masked slot costs a few operations, not the draws.

    Args:
        refit: the refit's draws, the rows they represent and their star.
        log: the run's log, all stars; grant rows first, then observed rows.
        models: the run's model table, static; ``Row.model_slot`` indexes it.
        star: a pytree of scalar leaves describing the star.
        time_d: the query time, days from the start of the survey.

    Returns:
        The replay.

    Raises:
        ValueError: if the model table and the log's payload buffers differ in
            number, or ``row_mask`` does not have the log's capacity.
    """
    rows = log.rows
    if len(models) != len(rows.payload):
        raise ValueError(
            f"{len(models)} models for {len(rows.payload)} payload buffers in the log"
        )
    if refit.row_mask.shape != rows.star.shape:
        raise ValueError(
            f"row_mask {refit.row_mask.shape} does not match the log capacity "
            f"{rows.star.shape}"
        )
    applied = _due(log, refit, time_d)
    stats = tuple(
        _summed_stats(
            model,
            rows.payload[slot],
            applied & (rows.model_slot == slot),
            refit.atoms,
            star,
        )
        for slot, model in enumerate(models)
    )
    return Replay(refit, stats, applied)


def absorb(
    r: Replay,
    log: ObservationLog,
    models: tuple[AbstractObservationModel, ...],
    star,
    time_d: Float[Array, ""],
) -> Replay:
    """Bring a replay up to a later query time: apply the rows due and not applied.

    The new rows are applied one at a time in log order, under the rule and with the
    arithmetic of ``replay``, and only they are read, so a call costs its new rows,
    not the log. ``replay_belief`` of the result equals that of ``replay`` on the same
    log and time to rounding, bit for bit when the rows fell due in log order. A row is
    never applied twice: with no new row due, the replay is returned unchanged. The
    log must extend the one the replay has read, and query times must not decrease.
    Every input is an array or static structure, so under ``eqx.filter_jit`` a second
    row, a second star, a later time or a longer log of the same capacity traces
    nothing new. The loop over new rows is a ``lax.while_loop``, which forward-mode
    differentiation passes and reverse mode does not.

    Args:
        r: the replay.
        log: the run's log, all stars.
        models: the run's model table, static.
        star: a pytree of scalar leaves describing the replay's star.
        time_d: the query time, days from the start of the survey.

    Returns:
        The replay with every row due at ``time_d`` applied.
    """
    atoms = r.refit.atoms
    due = _due(log, r.refit, time_d)

    def new(carry):
        return due & ~carry[1]

    def apply_first(carry):
        stats, applied = carry
        index = jnp.argmax(new(carry))
        row = jax.tree.map(lambda x: x[index], log.rows)
        stats = tuple(
            _add_stats(
                total, model, row.payload[slot], row.model_slot == slot, atoms, star
            )
            for slot, (model, total) in enumerate(zip(models, stats, strict=True))
        )
        return stats, applied.at[index].set(True)

    stats, applied = lax.while_loop(
        lambda carry: jnp.any(new(carry)), apply_first, (r.stats, r.applied)
    )
    return Replay(r.refit, stats, applied)


def replay_belief(
    r: Replay,
    roles: Roles,
    models: tuple[AbstractObservationModel, ...],
    star,
    pool: Pool,
    *,
    min_n_eff,
    max_khat,
) -> Belief:
    """The belief a replay holds under a role set.

    The slots are finalized in table order, each against the draws the slot before it
    returned, and their log marginal likelihoods add: the joint likelihood of all
    applied rows by the chain rule. That needs every model to meet the condition on
    ``stats`` stated in ``AbstractObservationModel``. The likelihood is the same for
    every role (observation models are equal across roles), so ``log_like`` is one row
    broadcast to ``(R, A)``. The role weights are those of ``role_belief``, with
    each role's prior ratio taken on the refit's draws and gated there.

    The belief's draws are ``Atoms``, not a ``Refit``: their gas block carries the
    applied rows, so they cannot be replayed a second time.

    Open point: the role weight ``log pi_r - log pi_0`` is a ratio of point densities.
    On a draw whose gas block is a distribution (``x_var > 0``) it is exact only when
    every role's gas prior equals the refit's; a role whose gas prior differs
    needs the ratio of the two priors integrated over the block, which no role prior
    expresses yet.

    Args:
        r: the replay.
        roles: the forecast and the stacked verdict members.
        models: the run's model table, static.
        star: a pytree of scalar leaves describing the replay's star.
        pool: the undetected planets, carried unchanged.
        min_n_eff: the least Kish size of a role's weights (``role_belief``).
        max_khat: the largest Pareto shape of a role's reweighted cloud.

    Returns:
        The belief, with the draws updated by every applied row.

    Raises:
        ValueError: if a role prior's chart differs from that of the refit's prior.
    """
    ratio = _role_log_prior_ratio(r.refit.atoms, roles, star, min_n_eff, max_khat)
    atoms, log_like = _finalized(r, models, star)
    return Belief(atoms, ratio, jnp.broadcast_to(log_like, ratio.shape), pool)


def _finalized(
    r: Replay, models: tuple[AbstractObservationModel, ...], star
) -> tuple[Atoms, Float[Array, " A"]]:
    """The draws a replay holds and the joint log likelihood of its applied rows."""
    atoms = r.refit.atoms
    log_like = jnp.zeros_like(atoms.cloud.log_weights)
    for model, stats in zip(models, r.stats, strict=True):
        slot_log_like, atoms = model.finalize(stats, atoms, star)
        log_like = log_like + slot_log_like
    return atoms, log_like


MAX_NODES = 2**22
"""The most quadrature nodes one belief may carry, ``A K G n`` (``quadrature_draws``):
about 32 MiB per float64 array, and a planner's cells hold ``M B`` times that."""


_NEAR_FLOOR = 0.3
"""The near block of each side runs from the floor over this fraction of the minority
side's mass."""
_GRADED_UNTIL = 0.05
"""The graded block of each side ends at this fraction of the side's mass (or at twice
the near block's end, whichever is further)."""


def _block_counts(h: int) -> tuple[int, int, int]:
    """Nodes in the near, graded and far blocks of a side of ``h`` nodes.

    The near block grows with ``h`` (a quarter of the side, at least two), so the
    quadrature converges in ``n``: at a moderate claim probability the near block spans
    about 0.4 sd, which a reading resolves once its S/N is above about 20, and a fixed
    two nodes there leave an error that more nodes elsewhere never remove.
    """
    if h < 4:
        return 0, 0, h
    near = max(2, h // 4) if h >= 6 else 1
    graded = (h - near) // 2
    return near, graded, h - near - graded


def _floor_nodes(
    mean: Float[Array, "*f"],
    var: Float[Array, "*f"],
    split: Float[Array, "*f"],
    n: int,
) -> tuple[Float[Array, "*f n"], Float[Array, "*f n"]]:
    """Nodes and weights of each factor ``N(mean, var)`` split at ``split``.

    ``n / 2`` nodes on each side of the split, placed by the probability ``e`` between
    the node and the split. A side of mass ``M`` is cut, in ``e``, into three
    Gauss-Legendre blocks: a near block ``[0, a]``, uniform in ``e``; a graded block
    ``[a, b]``, uniform in ``log e``; a far block ``[b, M]``, uniform in ``e``. Here
    ``a`` is 0.3 times the minority side's mass and ``b`` the larger of 0.05 ``M`` and
    ``2 a``. Near certainty the posterior of every outcome cell is concentrated on the
    minority side and within about the minority mass of the floor on the dominant
    side: a uniform block over the dominant side puts no node there, the near block
    puts a quarter of the side's nodes (at least two), and the graded block carries the
    nodes from that scale out to the bulk.
    With fewer than 4 nodes per side, one uniform block.

    The node is ``mean -/+ sd Phi^-1(p)`` with ``p`` the probability beyond the node
    on its near or far tail, whichever is smaller, so that no small probability is
    taken as one minus a number. A point factor (``var = 0``) gives ``n`` nodes at its
    mean, weights summing to one; a side with no mass gives nodes at the mean with
    weight zero, so every node stays finite.
    """
    counts = _block_counts(n // 2)
    spread = var > 0
    sd = jnp.sqrt(jnp.where(spread, var, 1.0))
    z = jnp.where(spread, (split - mean) / sd, 0.0)
    below, above = ndtr(z), ndtr(-z)
    tiny = jnp.finfo(z.dtype).tiny  # a normal number: XLA flushes subnormals to zero
    minority = jnp.minimum(below, above)
    nodes, weights = [], []
    for mass, other, sign in ((below, above, -1.0), (above, below, 1.0)):
        a = jnp.maximum(_NEAR_FLOOR * minority, tiny)
        b = jnp.maximum(_GRADED_UNTIL * mass, 2.0 * a)
        zero = jnp.zeros_like(mass)
        ends = (zero, a, b, mass) if counts[0] else (zero, mass)
        blocks = zip(ends[:-1], ends[1:], [k for k in counts if k], strict=True)
        e, w = [], []
        for i, (e0, e1, k) in enumerate(blocks):
            t, gw = np.polynomial.legendre.leggauss(k)
            f, gw = (1.0 + t) / 2.0, gw / 2.0
            if counts[0] and i == 1:  # graded: uniform in log e
                v0, v1 = jnp.log(e0)[..., None], jnp.log(e1)[..., None]
                share = jnp.exp((v1 - v0) * (f - 1.0))  # e / b, in (a / b, 1)
                wk = gw * share  # normalized below: the block's mass b - a is exact
                e.append(e1[..., None] * share)
                w.append((e1 - e0)[..., None] * wk / jnp.sum(wk, -1, keepdims=True))
            else:
                e.append(e0[..., None] + (e1 - e0)[..., None] * f)
                w.append((e1 - e0)[..., None] * gw)
        e, w = jnp.concatenate(e, -1), jnp.concatenate(w, -1)
        live = (mass > 0)[..., None]
        # Beyond the node toward the split and the far side: other + e; away: mass - e.
        toward, away = other[..., None] + e, mass[..., None] - e
        near = toward < away
        p = jnp.where(live, jnp.where(near, toward, away), 0.5)
        step = sd[..., None] * ndtri(jnp.clip(p, tiny, 1.0))
        x = mean[..., None] + sign * jnp.where(near, step, -step)
        nodes.append(jnp.where(live & spread[..., None], x, mean[..., None]))
        weights.append(jnp.where(live, w, 0.0))
    return jnp.concatenate(nodes, -1), jnp.concatenate(weights, -1)


def _split_points(
    atoms: Atoms, claim: Claim, models: tuple[AbstractObservationModel, ...]
) -> Float[Array, "A K G"]:
    """Where each factor's quadrature splits: the claim's floor of its gas.

    A gas the claim does not read splits at the step of a model that steps in the
    abundance (``gas_step``), or else at each factor's mean. A model's step on a gas
    the claim reads must equal the claim's floor there (checked; the floor may be
    traced, so the check runs under ``jit`` too).
    """
    floors = {}
    for conjunct in claim.conjuncts:
        if isinstance(conjunct, GasAbove):
            if conjunct.gas in floors:
                raise ValueError(
                    f"the claim reads gas {conjunct.gas} twice; one quadrature split "
                    "per gas cannot serve two floors"
                )
            floors[conjunct.gas] = conjunct.floor
    steps = {m.gas_step for m in models if m.gas_step is not None}
    if len(steps) > 1:
        raise ValueError(f"models step at different abundances {sorted(steps)}")
    step = steps.pop() if steps else None
    n_gas = atoms.x_mean.shape[-1]
    if any(not 0 <= g < n_gas for g in floors):
        raise ValueError(f"the claim reads a gas outside the block of {n_gas}")
    dtype = atoms.x_mean.dtype
    columns = []
    for g in range(n_gas):
        if g in floors:
            floor = jnp.asarray(floors[g], dtype)
            if step is not None:
                floor = eqx.error_if(
                    floor,
                    floor != step,
                    "a model's step in the abundance differs from the claim's floor",
                )
            columns.append(jnp.broadcast_to(floor, atoms.x_mean.shape[:-1]))
        elif step is not None:
            columns.append(jnp.full(atoms.x_mean.shape[:-1], step, dtype))
        else:
            columns.append(atoms.x_mean[..., g])
    return jnp.stack(columns, axis=-1)


def quadrature_draws(
    b: Belief,
    claim: Claim,
    n: int = 24,
    models: tuple[AbstractObservationModel, ...] = (),
) -> Belief:
    """The belief with each draw's gas block on quadrature nodes, factor by factor.

    Every factor of the block, one planet and one gas ``N(m, v)``, is carried by ``n``
    point nodes: ``n / 2`` below the split and ``n / 2`` above it, ``x = m + sqrt(v)
    Phi^-1(u)`` with weight ``omega``, in three Gauss-Legendre blocks per side graded
    toward the split (``_floor_nodes``). The split
    of a gas is the floor of the claim's ``GasAbove`` on it, so a claim that steps at
    the floor keeps its probability (the mass on each side is exact) and the
    integrand of every outcome is smooth on each side; a gas the claim does not read
    splits at the step of a model in ``models`` (``gas_step``), or at the factor's
    mean. Given a draw the factors are independent, so the nodes are held per factor
    (``Atoms.nodes``, ``(A, K, G, n)``) and stand for the product grid of ``n ** (K
    G)`` points without forming it: the draws, their weights and the pool are
    unchanged, and the cost is linear in ``n K G``. A point factor's nodes coincide.

    On these nodes a planner's update by an outcome cell is exact reweighting of the
    draws and the nodes (``update_by_cell``): the cells of rows are independent given
    the points. The error is that of the quadrature: at the default 24 nodes the
    lookahead value of a reading binned at the floor is within 0.9 percent of a dense
    quadrature at depth 1 and 2, for a prior probability of the claim from 1e-6 to
    1 - 1e-6 and S/N 2 to 50; within 0.3 percent at 32 nodes and 0.06 percent at 48 or
    more. At 16 nodes (two near the floor) it is 2.7 percent to S/N 20 and 5.4 percent
    at S/N 50. A bin edge
    away from the split is not graded for: an edge 0.3 or 0.5 dex from the floor
    measured 1.4 to 2.2 percent at 24 nodes near certainty.

    Every factor is carried, read or not: the cost is linear in the factors, and which
    gases a model covers is a traced value (its S/N per gas), not a static one.

    Args:
        b: the belief, one star (no leading star axis).
        claim: the claim; the floor of each of its ``GasAbove`` conjuncts is the split
            of that gas, one per gas.
        n: nodes per factor, static: even, or 1 to leave the draws as they are (for
            a belief whose blocks are already point states).
        models: the models a planner reads, static; one that steps in the abundance
            (``gas_step``) must step at the claim's floor of every gas the claim
            reads, and sets the split of the others.

    Returns:
        The belief with ``atoms.nodes`` set.

    Raises:
        ValueError: if ``n`` is neither 1 nor a positive even number, if the claim
            reads one gas twice or a gas outside the block, if models step at
            different abundances, or if the nodes would exceed ``MAX_NODES``.
        EquinoxRuntimeError: if a model's step differs from the claim's floor.
    """
    if n == 1:
        return b
    if n < 2 or n % 2:
        raise ValueError(f"n is 1 or a positive even number, not {n}")
    atoms = b.atoms
    if atoms.x_mean.size * n > MAX_NODES:
        raise ValueError(
            f"{atoms.x_mean.shape} factors of {n} nodes exceed MAX_NODES = {MAX_NODES}"
        )
    split = _split_points(atoms, claim, models)
    nodes = GasNodes(*_floor_nodes(atoms.x_mean, atoms.x_var, split, n))
    atoms = Atoms(atoms.cloud, atoms.x_mean, atoms.x_var, atoms.exists, nodes)
    return Belief(atoms, b.log_prior_ratio, b.log_like, b.pool)


def select_roles(b: Belief, rows: slice) -> Belief:
    """The belief of some of ``b``'s roles: its rows ``rows``, the pool's with them.

    Leading axes (a stack of stars) are kept.

    Args:
        b: the belief.
        rows: the role rows to keep, a slice of the role axis ``R``.

    Returns:
        The belief of those roles, on the same draws.
    """
    pool = _pool_rows(b.pool, rows)
    return Belief(
        b.atoms, b.log_prior_ratio[..., rows, :], b.log_like[..., rows, :], pool
    )


def _pool_rows(pool: Pool, rows: slice) -> Pool:
    """The pool of some roles: its rows ``rows``, leading axes kept."""
    return Pool(
        pool.marks,
        pool.log_prior_ratio[..., rows, :],
        pool.log_like[..., rows, :],
        pool.log_mass[..., rows],
    )


def planning_beliefs(
    forecast: Replay,
    verdict: Replay,
    roles: Roles,
    models: tuple[AbstractObservationModel, ...],
    star,
    forecast_pool: Pool,
    verdict_pool: Pool,
    claim: Claim,
    n: int = 24,
    *,
    min_n_eff,
    max_khat,
) -> tuple[Belief, Belief]:
    """The two beliefs a planner reads: the forecast's and the verdict's.

    Each is read off its own replay of the log (``replay_belief``): the forecast's
    belief, one role (``R = 1``), off ``forecast`` under the forecast prior, and the
    verdict's, one role per member, off ``verdict`` under the members; each replay is
    read under its own roles only. The two replays may hold different rows: a frozen
    forecast is the replay at ``time_d = 0`` while the verdict's follows the decision
    time. Each reads its own pool, the forecast row 0 of ``forecast_pool`` and the
    verdict rows 1 on of ``verdict_pool``. The pools' role prior ratios are formed
    here, as the draws' are, by the same function and gate (``role_pool``): the
    ratios the pools carry in are not read, so a pool's marks are weighed under
    exactly the priors and the gate the draws are, and a role whose marks fail the
    gate reads NaN (an empty pool is never refused). Each belief carries its own
    draws, so a gas block the verdict's rows update never reaches the forecast. Each
    is then put on quadrature nodes split at the claim's floors (``quadrature_draws``,
    with ``models`` for their steps), on which a hypothetical outcome updates the
    belief exactly by reweighting.

    Args:
        forecast: the forecast's replay.
        verdict: the verdict's replay.
        roles: the forecast and the stacked verdict members.
        models: the run's model table, static.
        star: a pytree of scalar leaves describing the star.
        forecast_pool: the undetected planets the forecast reads, every role.
        verdict_pool: the undetected planets the verdict reads, every role.
        claim: the claim; its ``GasAbove`` floors split the quadrature
            (``quadrature_draws``).
        n: quadrature nodes per gas factor, a static planning setting.
        min_n_eff: the least Kish size of a role's weights (``role_belief``).
        max_khat: the largest Pareto shape of a role's reweighted cloud.

    Returns:
        The forecast's belief (``R = 1``) and the verdict's (``R`` the members), on
        quadrature nodes; a role the gate refused has NaN rows.

    Raises:
        ValueError: if a role prior's chart differs from that of the refit's prior, or
            the quadrature refuses the claim, the models or the size
            (``quadrature_draws``).
    """

    def read(r, prior, name, stacked, pool):
        gate = (min_n_eff, max_khat)
        ratio = _prior_ratio_rows(r.refit.atoms, prior, star, name, stacked, *gate)
        empty = jnp.isneginf(pool.log_mass)
        marks = _prior_ratio_rows(pool.marks, prior, star, name, stacked, *gate, empty)
        pool = Pool(pool.marks, marks, pool.log_like, pool.log_mass)
        atoms, log_like = _finalized(r, models, star)
        log_like = jnp.broadcast_to(log_like, ratio.shape)
        return quadrature_draws(Belief(atoms, ratio, log_like, pool), claim, n, models)

    return (
        read(
            forecast,
            roles.forecast,
            "forecast",
            False,
            _pool_rows(forecast_pool, slice(0, 1)),
        ),
        read(
            verdict,
            roles.verdict,
            "verdict",
            True,
            _pool_rows(verdict_pool, slice(1, None)),
        ),
    )
