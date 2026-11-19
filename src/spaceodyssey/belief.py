"""Belief: weighted draws, a weight vector per role, and the probabilities read off.

The belief at a star is inferred from one refit cloud and the log: ``infer`` replays
every row the cloud does not already represent, and ``absorb`` reads one new row into a
``Replay``, the same computation held open.
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gammainc
from jaxtyping import Array, Bool, Float, Int, PyTree

from spaceodyssey.claim import Claim
from spaceodyssey.roles import Roles
from spaceodyssey.types import (
    AbstractObservationModel,
    Atoms,
    ObservationLog,
    Pool,
    Row,
)


class Belief(eqx.Module):
    """What one role set believes about one star: draws, role weights, and the pool.

    Attributes:
        atoms: the weighted draws of the refit.
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
        n_draw = self.atoms.cloud.log_weight.shape[-1]
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
    that likelihood is the same model for every role; every prior and ``p0`` share a
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
    return Belief(atoms, _role_log_prior(atoms, roles, star), log_like, pool)


def _role_log_prior(atoms: Atoms, roles: Roles, star) -> Float[Array, "R A"]:
    """Every role's log prior of each draw, forecast in row 0; checks the measures."""
    for name, prior in (("forecast", roles.forecast), ("verdict", roles.verdict)):
        if prior.measure != atoms.cloud.measure:
            raise ValueError(
                f"the {name} prior is on measure {prior.measure!r} but the cloud is "
                f"on {atoms.cloud.measure!r}"
            )
    forecast = roles.forecast.log_prob(atoms, star)
    verdict = eqx.filter_vmap(
        lambda prior: prior.log_prob(atoms, star), in_axes=eqx.if_array(0)
    )(roles.verdict)
    return jnp.concatenate([forecast[None], verdict], axis=0)


def weights(b: Belief) -> Float[Array, "R A"]:
    """Normalized role weights over the draws.

    The unnormalized log weight of role ``r`` is
    ``log_weight + log_prior[r] - log_ref_density + log_like[r]``, and each row is
    normalized to sum to one. A role under which every draw has log weight ``-inf``
    cannot be normalized: its row is NaN, on purpose, and so is everything read from
    that role (its ``q``, and its pool rate when the same holds for the marks).

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
    return _ratio(mass[:, -1], mass[:, claim.given])


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
    mark_weights = _role_weights(pool.marks, pool.log_prior, pool.log_like)  # (R, Am)
    return jnp.exp(pool.log_mass) * jnp.matmul(mark_weights, p_mark, precision=_EXACT)


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


def _as_index(x) -> Int[Array, ""]:
    """``x`` as the default integer type, never weakly typed.

    A replay built from Python integers would otherwise hold weakly typed indices, and
    the strongly typed ones a compiled ``absorb`` returns would trace it a second time.
    """
    return jnp.asarray(x, dtype=int)


class Replay(eqx.Module):
    """The rows of one star read into one refit cloud, held as summed statistics.

    A replay is ``infer`` held open: ``replay`` reads a whole log into it, ``absorb``
    reads the next row, and ``replay_belief`` turns it into the belief. Each model
    slot keeps the leafwise sum of its rows' ``stats``; the draws stay as the refit
    returned them, and every belief is finalized from them afresh, so a belief kept
    current row by row equals the belief inferred from the whole log, to rounding.

    Attributes:
        atoms: the refit's draws, before any row is applied.
        log_prior: log prior of each draw under each role, ``(R, A)``.
        stats: one pytree per model slot, the summed statistics of the rows applied
            in that slot, leaves with leading axis ``A``.
        star_index: the star whose rows are applied.
        count: the number of log rows read, applied or not.
        pool: the undetected planets, carried unchanged.
    """

    atoms: Atoms
    log_prior: Float[Array, "R A"] = eqx.field(converter=jnp.asarray)
    stats: tuple[PyTree, ...]
    star_index: Int[Array, ""] = eqx.field(converter=_as_index)
    count: Int[Array, ""] = eqx.field(converter=_as_index)
    pool: Pool


def _applies(rows: Row, index, count, represented, star_index) -> Bool[Array, "*rows"]:
    """Whether log rows are applied to a cloud of one star.

    A row is applied when it is released (``index < count``), belongs to the star, was
    accepted (``refusal == 0``) and is not among the rows the cloud already represents.
    Its date plays no part.
    """
    released = index < count
    return released & (rows.star == star_index) & (rows.refusal == 0) & ~represented


def _add_stats(
    total: PyTree,
    model: AbstractObservationModel,
    payload: PyTree,
    keep: Bool[Array, ""],
    atoms: Atoms,
    star,
) -> PyTree:
    """``total`` plus one row's statistics where ``keep``, exact zeros elsewhere."""
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
    roles: Roles,
    atoms: Atoms,
    log: ObservationLog,
    models: tuple[AbstractObservationModel, ...],
    star,
    star_index: Int[Array, ""],
    pool: Pool,
) -> Replay:
    """Read every row of a log into a replay of one star's refit cloud.

    Each model slot is one scan over the log's payload buffer of that slot, statically
    unrolled over ``models``; nothing switches on a traced slot. A row contributes to
    its slot's sum when it is applied (see ``infer``) and exactly zero otherwise, so the
    sum is commutative and does not depend on where in the log a row sits.

    Args:
        roles: the forecast and the stacked verdict members.
        atoms: the refit's draws, with the rows they represent in ``row_mask``.
        log: the run's log, all stars.
        models: the run's model table, static; ``Row.model_slot`` indexes it.
        star: a pytree of scalar leaves describing the star.
        star_index: the star's index in the log's ``star`` column.
        pool: the undetected planets.

    Returns:
        The replay, with ``count`` equal to the log's.

    Raises:
        ValueError: if a role prior's measure differs from the cloud's, the model
            table and the log's payload buffers differ in number, or ``row_mask`` does
            not have the log's capacity.
    """
    rows = log.rows
    if len(models) != len(rows.payload):
        raise ValueError(
            f"{len(models)} models for {len(rows.payload)} payload buffers in the log"
        )
    if atoms.row_mask.shape != rows.star.shape:
        raise ValueError(
            f"row_mask {atoms.row_mask.shape} does not match the log capacity "
            f"{rows.star.shape}"
        )
    index = jnp.arange(rows.star.shape[0])
    applies = _applies(rows, index, log.count, atoms.row_mask, star_index)
    stats = tuple(
        _summed_stats(
            model, rows.payload[slot], applies & (rows.model_slot == slot), atoms, star
        )
        for slot, model in enumerate(models)
    )
    log_prior = _role_log_prior(atoms, roles, star)
    return Replay(atoms, log_prior, stats, star_index, log.count, pool)


def absorb(
    r: Replay,
    log: ObservationLog,
    models: tuple[AbstractObservationModel, ...],
    star,
) -> Replay:
    """Read the next row of the log, row ``r.count``, into a replay.

    The row's statistics are added to its slot's sum when it applies, under the same
    rule and with the same arithmetic as ``replay``, so ``replay_belief`` of a replay
    advanced row by row equals ``infer`` on the whole log to rounding. When the log
    holds no row past ``r.count`` the replay is returned unchanged; a row is never read
    twice. The log must extend the one the replay has read. Every input is an array or
    static structure, so under ``eqx.filter_jit`` a second row, a second star or a
    longer log of the same capacity traces nothing new.

    Args:
        r: the replay.
        log: the run's log, all stars.
        models: the run's model table, static.
        star: a pytree of scalar leaves describing the replay's star.

    Returns:
        The replay with the row read.
    """
    index = r.count
    at = jnp.minimum(index, log.rows.star.shape[0] - 1)
    row = jax.tree.map(lambda x: x[at], log.rows)
    applies = _applies(row, index, log.count, r.atoms.row_mask[at], r.star_index)
    stats = tuple(
        _add_stats(
            total,
            model,
            row.payload[slot],
            applies & (row.model_slot == slot),
            r.atoms,
            star,
        )
        for slot, (model, total) in enumerate(zip(models, r.stats, strict=True))
    )
    count = index + (index < log.count).astype(index.dtype)
    return eqx.tree_at(lambda x: (x.stats, x.count), r, (stats, count))


def replay_belief(
    r: Replay, models: tuple[AbstractObservationModel, ...], star
) -> Belief:
    """The belief a replay holds: every slot's sum finalized, then the role weights.

    The slots are finalized in order, each against the draws the slot before it
    returned, and their log marginal likelihoods add: the joint likelihood of all
    applied rows by the chain rule. This needs a slot's ``stats`` to read nothing of a
    draw that an earlier slot's ``finalize`` changes, since the statistics were taken
    against the refit's draws (both toy models qualify: the Gaussian model's statistics
    read no draw, the step report's read only point states, which nothing changes).

    The likelihood is the same for every role (observation models are equal across
    roles), so ``log_like`` is one row broadcast to ``(R, A)``.

    Args:
        r: the replay.
        models: the run's model table, static.
        star: a pytree of scalar leaves describing the replay's star.

    Returns:
        The belief, with the draws updated by every applied row and the refit's
        ``row_mask`` and cloud.
    """
    atoms = r.atoms
    log_like = jnp.zeros_like(atoms.cloud.log_weight)
    for model, stats in zip(models, r.stats, strict=True):
        slot_log_like, atoms = model.finalize(stats, atoms, star)
        log_like = log_like + slot_log_like
    log_like = jnp.broadcast_to(log_like, r.log_prior.shape)
    return Belief(atoms, r.log_prior, log_like, r.pool)


def infer(
    roles: Roles,
    atoms: Atoms,
    log: ObservationLog,
    models: tuple[AbstractObservationModel, ...],
    star,
    star_index: Int[Array, ""],
    pool: Pool,
) -> Belief:
    """The belief at one star from a refit cloud and the log.

    Every released row of the star outside the cloud's ``row_mask`` is applied once,
    whatever its date; a row in ``row_mask`` (the cloud's weights already carry it), a
    refused row, a row of another star and a row past the log's ``count`` contribute
    exactly zero. Rows enter through their models' summed statistics and ``finalize``,
    never as a sum of per-row likelihoods, so rows reading one uncertain quantity are
    exact. The role weights are those of ``role_log_weights``, with every role's
    likelihood the one ``finalize`` returns. This is ``replay_belief`` of ``replay``.

    Always infer from the refit's draws. The belief's draws carry a gas block updated
    by the applied rows but keep the refit's ``row_mask``, so reading them back into
    ``infer`` would apply those rows twice.

    Open point: the role weight ``log_prior - log_ref_density`` is a ratio of point
    densities. On a draw whose gas block is a distribution (``x_var > 0``) it is exact
    only when every role's gas prior equals the cloud's; a role whose gas prior differs
    needs the ratio of the two priors integrated over the block, which no role prior
    expresses yet.

    Args:
        roles: the forecast and the stacked verdict members.
        atoms: the refit's draws, with the rows they represent in ``row_mask``.
        log: the run's log, all stars; grant rows first, then observed rows.
        models: the run's model table, static; ``Row.model_slot`` indexes it.
        star: a pytree of scalar leaves describing the star.
        star_index: the star's index in the log's ``star`` column.
        pool: the undetected planets, carried unchanged.

    Returns:
        The belief.

    Raises:
        ValueError: as ``replay``.
    """
    return replay_belief(
        replay(roles, atoms, log, models, star, star_index, pool), models, star
    )
