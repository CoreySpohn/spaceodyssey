"""Belief: weighted draws, a weight vector per role, and the probabilities read off.

The belief at a star is read off a replay of the log into one refit: ``replay``
applies every row due at a query time that the refit does not already represent,
``absorb`` brings a replay up to a later query time, and ``replay_belief`` reads the
belief off it under a role set.
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gammainc
from jaxtyping import Array, Bool, Float, Int, PyTree

from spaceodyssey._numerics import EXACT, safe_ratio
from spaceodyssey.claim import Claim
from spaceodyssey.roles import Roles
from spaceodyssey.types import (
    AbstractObservationModel,
    Atoms,
    ObservationLog,
    Pool,
    Refit,
)


class Belief(eqx.Module):
    """What one role set believes about one star: draws, role weights, and the pool.

    Attributes:
        atoms: the weighted draws; after a replay, their gas block carries the
            applied rows.
        log_prior: log prior density of each draw under each role, shape ``(R, A)``.
        log_like: summed log likelihood of the rows each role has seen, ``(R, A)``.
        pool: the undetected planets.

    Raises:
        ValueError: if ``log_prior``, ``log_like`` and the pool disagree on the number
            of roles, or the draws on their number.
    """

    atoms: Atoms
    log_prior: Float[Array, "R A"] = eqx.field(converter=jnp.asarray)
    log_like: Float[Array, "R A"] = eqx.field(converter=jnp.asarray)
    pool: Pool

    def __check_init__(self):
        """Check the static shapes: one ``R`` and one ``A`` across the fields."""
        n_draw = self.atoms.cloud.log_weights.shape[-1]
        roles = {
            self.log_prior.shape[-2],
            self.log_like.shape[-2],
            self.pool.log_mass.shape[-1],
        }
        draws = {self.log_prior.shape[-1], self.log_like.shape[-1], n_draw}
        if len(roles) != 1 or len(draws) != 1:
            raise ValueError(
                f"belief fields disagree on R or A (log_prior {self.log_prior.shape}, "
                f"log_like {self.log_like.shape}, pool R "
                f"{self.pool.log_mass.shape[-1]}, {n_draw} draws)"
            )


def _require_ref_density(cloud) -> None:
    """Refuse a cloud whose reference density is not declared."""
    if cloud.log_ref_density is None:
        raise ValueError(
            "a cloud without a declared log_ref_density (log p0 on its chart) cannot "
            "be reweighted to a role"
        )


def _role_log_weights(atoms: Atoms, log_prior, log_like) -> Float[Array, "R A"]:
    """Unnormalized log weights of every role over the leading axis of ``atoms``.

    A draw with zero refit weight, zero prior or zero likelihood has log weight
    ``-inf``. The ``jnp.where`` keeps it there even when ``log_ref_density`` is also
    ``-inf`` (which would otherwise give ``-inf - (-inf) = nan``); the sum is linear,
    so the untaken branch passes no nan into the gradient.

    Raises:
        ValueError: if the cloud does not declare ``log_ref_density``.
    """
    cloud = atoms.cloud
    _require_ref_density(cloud)
    dead = (
        jnp.isneginf(cloud.log_weights)
        | jnp.isneginf(log_prior)
        | jnp.isneginf(log_like)
    )
    total = cloud.log_weights + log_prior - cloud.log_ref_density + log_like
    return jnp.where(dead, -jnp.inf, total)


def role_log_weights(
    atoms: Atoms,
    roles: Roles,
    log_like: Float[Array, "R A"],
    star,
    pool: Pool,
) -> Belief:
    """Fill each role's prior over the draws and return the belief.

    The weight of role ``r`` is
    ``log w_r = log_weights + log_prior[r] - log_ref_density + log_like[r]``: the refit
    weight, the role prior, minus the refit target ``p0``, plus the role's likelihood.

    Preconditions: ``log_weights`` is against ``p0`` times the incoming likelihood, and
    that likelihood is the same model for every role; every prior and ``p0`` share a
    chart (checked here); the proposal covers the support of each role's prior times
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
        ValueError: if the cloud declares no chart or no ``log_ref_density``, or a
            prior's ``chart`` differs from the cloud's.
    """
    return Belief(atoms, _role_log_prior(atoms, roles, star), log_like, pool)


def _role_log_prior(atoms: Atoms, roles: Roles, star) -> Float[Array, "R A"]:
    """Every role's log prior of each draw, forecast in row 0; checks the charts."""
    if atoms.cloud.chart is None:
        raise ValueError(
            "a cloud without a declared chart cannot be reweighted to a role"
        )
    _require_ref_density(atoms.cloud)
    for name, prior in (("forecast", roles.forecast), ("verdict", roles.verdict)):
        if prior.chart != atoms.cloud.chart:
            raise ValueError(
                f"the {name} prior is on chart {prior.chart!r} but the cloud is "
                f"on {atoms.cloud.chart!r}"
            )
    forecast = roles.forecast.log_prob(atoms, star)
    verdict = eqx.filter_vmap(
        lambda prior: prior.log_prob(atoms, star), in_axes=eqx.if_array(0)
    )(roles.verdict)
    return jnp.concatenate([forecast[None], verdict], axis=0)


def log_weights(b: Belief) -> Float[Array, "R A"]:
    """Unnormalized log role weights over the draws.

    The log weight of role ``r`` is
    ``log_weights + log_prior[r] - log_ref_density + log_like[r]``, the first and
    third terms the cloud's. A draw with zero refit weight, zero prior or zero
    likelihood has log weight ``-inf``.

    Args:
        b: the belief.

    Returns:
        Log weights of shape ``(R, A)``.

    Raises:
        ValueError: if the cloud does not declare ``log_ref_density``.
    """
    return _role_log_weights(b.atoms, b.log_prior, b.log_like)


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
    log_prior[r] - log_ref_density + log_like[r]``, and each row is normalized over the
    marks.

    Args:
        pool: the undetected planets.

    Returns:
        Weights of shape ``(R, Am)``.
    """
    log_w = _role_log_weights(pool.marks, pool.log_prior, pool.log_like)
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


def pool_rate(pool: Pool, claim: Claim, star) -> Float[Array, " R"]:
    """Expected number of claim-true planets in the pool, per role.

    The rate of role ``r`` is ``exp(log_mass[r]) * pbar[r]``, where ``pbar[r]`` is the
    claim probability of a mark averaged with role ``r``'s own mark weights. The rate
    at time zero is this function of the pool at time zero.

    Args:
        pool: the undetected planets.
        claim: an unconditional claim.
        star: a pytree of scalar leaves describing the star.

    Returns:
        The rate per role, shape ``(R,)``.

    Raises:
        ValueError: if the claim is conditional.
    """
    if claim.given != 0:
        raise ValueError("pool_rate needs an unconditional claim (claim.given == 0)")
    p_mark = jnp.prod(_conjunct_probs(pool.marks, claim, star), axis=0)[:, 0]  # (Am,)
    weighted = jnp.matmul(mark_weights(pool), p_mark, precision=EXACT)  # (R,)
    return jnp.exp(pool.log_mass) * weighted


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
    from_draws = jnp.matmul(weights(b), per_draw, precision=EXACT)  # (R, N)

    lam = pool_rate(b.pool, claim, star)  # (R,)
    from_pool = jax.vmap(_poisson, in_axes=(0, None))(lam, n_bins)  # (R, N)

    full = jax.vmap(partial(jnp.convolve, precision=EXACT))(from_draws, from_pool)
    return jnp.concatenate(
        [full[:, : n_bins - 1], full[:, n_bins - 1 :].sum(1, keepdims=True)], 1
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
            in that slot, leaves with leading axis ``A``.
        applied: the log rows applied, shape ``(W,)``, none of them in
            ``refit.row_mask``.
    """

    refit: Refit
    stats: tuple[PyTree, ...]
    applied: Bool[Array, " W"]


def _due(log: ObservationLog, refit: Refit, time_d) -> Bool[Array, " W"]:
    """The log rows a replay of ``refit`` applies at the query time ``time_d``.

    A row is due when it is real (``index < count``), released by the query time
    (``release_d <= time_d``), accepted (``refusal == 0``), of the refit's star, and
    not among the rows the refit represents. A row in flight stays in the log and
    falls due once the query time reaches its release.
    """
    rows = log.rows
    real = jnp.arange(rows.star.shape[0]) < log.count
    return (
        real
        & (rows.release_d <= time_d)
        & (rows.refusal == 0)
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
    commutative and does not depend on where in the log a row sits.

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
) -> Belief:
    """The belief a replay holds under a role set.

    The slots are finalized in table order, each against the draws the slot before it
    returned, and their log marginal likelihoods add: the joint likelihood of all
    applied rows by the chain rule. That needs every model to meet the condition on
    ``stats`` stated in ``AbstractObservationModel``. The likelihood is the same for
    every role (observation models are equal across roles), so ``log_like`` is one row
    broadcast to ``(R, A)``. The role weights are those of ``role_log_weights``, with
    each role's prior taken on the refit's draws.

    The belief's draws are ``Atoms``, not a ``Refit``: their gas block carries the
    applied rows, so they cannot be replayed a second time.

    Open point: the role weight ``log_prior - log_ref_density`` is a ratio of point
    densities. On a draw whose gas block is a distribution (``x_var > 0``) it is exact
    only when every role's gas prior equals the cloud's; a role whose gas prior differs
    needs the ratio of the two priors integrated over the block, which no role prior
    expresses yet.

    Args:
        r: the replay.
        roles: the forecast and the stacked verdict members.
        models: the run's model table, static.
        star: a pytree of scalar leaves describing the replay's star.
        pool: the undetected planets, carried unchanged.

    Returns:
        The belief, with the draws updated by every applied row.

    Raises:
        ValueError: if the cloud declares no chart or no ``log_ref_density``, or a
            role prior's chart differs from the cloud's.
    """
    atoms = r.refit.atoms
    log_prior = _role_log_prior(atoms, roles, star)
    log_like = jnp.zeros_like(atoms.cloud.log_weights)
    for model, stats in zip(models, r.stats, strict=True):
        slot_log_like, atoms = model.finalize(stats, atoms, star)
        log_like = log_like + slot_log_like
    return Belief(atoms, log_prior, jnp.broadcast_to(log_like, log_prior.shape), pool)
