"""Information and certificates.

Information is the change of a log score on the claim-true count between beliefs. The
information an action is expected to bring is read off its outcome cells: each cell
updates the belief, and the verdict's log score after it is averaged over the cells the
forecast expects. A certificate states that a claim holds (or fails) under every verdict
member, with its Monte Carlo and tail diagnostics.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from spaceodyssey._numerics import EXACT, guarded_log, safe_ratio
from spaceodyssey._standin.diagnostics import KHAT_MAX, kish_ess, mc_se, pareto_khat
from spaceodyssey.belief import (
    Belief,
    chain,
    draw_claim_mass,
    log_weights,
    q,
    weights,
)
from spaceodyssey.claim import Claim
from spaceodyssey.types import AbstractObservationModel, Action, Atoms

# Certificate status codes.
POSITIVE = 0
NEGATIVE = 1
INAPPLICABLE = 2
UNRESOLVED = 3


def log_score(pmf: Float[Array, " N"], count: Int[Array, ""]) -> Float[Array, ""]:
    """Log of the probability a count distribution gives to the realized count.

    A count beyond the last bin is scored in the last bin, which holds the lumped tail;
    a negative count is scored in the first.

    Args:
        pmf: the count distribution.
        count: the realized count.

    Returns:
        The log score in nats.
    """
    return jnp.log(pmf[jnp.clip(count, 0, pmf.shape[0] - 1)])


def information(
    pmf_before: Float[Array, " N"], pmf_after: Float[Array, " N"], count: Int[Array, ""]
) -> Float[Array, ""]:
    """Change of the log score of the realized count, in nats.

    Args:
        pmf_before: the count distribution before the stage.
        pmf_after: the count distribution after the stage.
        count: the realized count.

    Returns:
        ``log_score(after) - log_score(before)``.
    """
    return log_score(pmf_after, count) - log_score(pmf_before, count)


def chain_gain(
    b_before: Belief,
    b_after: Belief,
    claim: Claim,
    star,
    planet: Int[Array, ""],
    role: int,
) -> Float[Array, " C"]:
    """Split of a claim-true planet's information over the conjuncts.

    Args:
        b_before: the belief before the stage.
        b_after: the belief after the stage.
        claim: the claim.
        star: a pytree of scalar leaves describing the star.
        planet: index of the planet along the ``K`` axis.
        role: row of the role whose chain is read, static.

    Returns:
        ``ln chain_after - ln chain_before`` per conjunct, in nats. An entry that is 0
        before and after (a conjunct behind a condition of zero mass) is NaN.
    """
    after = chain(b_after, claim, star, planet)[role]
    before = chain(b_before, claim, star, planet)[role]
    return jnp.log(after) - jnp.log(before)


def expected_log_score(
    pmf: Float[Array, " N"], forecast: Float[Array, " N"]
) -> Float[Array, ""]:
    """The log score ``pmf`` earns, expected over draws from ``forecast``.

    ``sum_n forecast(n) ln pmf(n)``, the negative cross entropy of ``forecast``
    against ``pmf`` (Cover and Thomas, Elements of Information Theory, ch. 2). An
    outcome ``forecast`` gives no mass adds nothing; an outcome it expects and ``pmf``
    rules out gives ``-inf``. A NaN in ``forecast`` (a refused role) gives NaN, never
    a confident zero.

    Args:
        pmf: the distribution that is scored.
        forecast: the distribution the outcome is drawn from.

    Returns:
        The expected log score in nats.
    """
    expected = ~(forecast <= 0)  # a NaN forecast is expected, so the NaN propagates
    return jnp.sum(
        jnp.where(expected, forecast * jnp.where(expected, guarded_log(pmf), 0), 0)
    )


def outcome_prob(b: Belief, cells: Float[Array, "A Y"]) -> Float[Array, "R Y"]:
    """Probability of each outcome cell under each role: the cells mixed by its weights.

    Args:
        b: the belief.
        cells: probability of each cell under each draw, from a model's ``cells``.

    Returns:
        Probabilities of shape ``(R, Y)``; each row sums to one.
    """
    return jnp.matmul(weights(b), cells, precision=EXACT)


class OutcomeCells(eqx.Module):
    """The outcome cells of one action on one belief's draws, per draw and gas factor.

    A cell's probability under a draw is ``draw[a, y]`` times, for every gas factor
    ``f`` (one planet and one gas), the factor's probability of its bin
    ``codes[y, f]`` averaged over the factor's quadrature nodes:
    ``P(y | a) = draw[a, y] prod_f sum_j omega[a, f, j] node[a, f, j, codes[y, f]]``,
    exact because the factors are independent given the draw. A model whose cells read
    no gas has ``draw`` its cells and one sure bin per factor; a model that reads gas
    (``gas_cells``) has ``draw`` one and its bins per factor and node. The node
    probabilities are fixed by the nodes, so a planner reads them once; only the node
    weights ``omega`` change with the outcomes.

    Leading axes (one per menu entry, say) are kept.

    Attributes:
        draw: the part of each cell's probability that reads no gas, ``(A, Y)``.
        node: each factor's probability of each bin at each node, ``(A, K, G, n, B)``.
        codes: each cell's bin of each factor, ``(Y, K, G)``.
    """

    draw: Float[Array, "*batch A Y"]
    node: Float[Array, "*batch A K G n B"]
    codes: Int[Array, "*batch Y K G"]


def _node_weights(atoms: Atoms) -> Float[Array, "A K G n"]:
    """The weights of the gas block's nodes; one node of weight one at a point."""
    if atoms.nodes is None:
        return jnp.ones((*atoms.x_mean.shape, 1), atoms.x_mean.dtype)
    return atoms.nodes.weights


def _bin_probs(atoms: Atoms, cells: OutcomeCells) -> Float[Array, "A K G B"]:
    """Each factor's probability of each bin, averaged over its nodes' weights."""
    return jnp.sum(_node_weights(atoms)[..., None] * cells.node, axis=-2)


def _factor_index(codes: Int[Array, "... K G"]):
    """Planet and gas indices that broadcast against ``codes``, for a gather."""
    n_planet, n_gas = codes.shape[-2:]
    return jnp.arange(n_planet)[:, None], jnp.arange(n_gas)[None, :]


def outcome_cells(
    b: Belief, model: AbstractObservationModel, action: Action
) -> OutcomeCells:
    """The cells of an action on a belief's draws, per draw, gas factor and node.

    A model whose cells read the gas block (``gas_cells`` is not None) is read at each
    node of each factor (a point block is one node), the cells coded mixed radix as
    ``gas_cells`` states; any other model through ``cells``. On a belief on nodes such
    a model is handed the gas block as NaN, since ``x_mean`` there is the block the
    nodes were built from and not the belief: a model that reads the gas through
    ``cells`` alone is refused with an error, never valued at that stale mean.

    Args:
        b: the belief; its gas block a point state or on nodes
            (``require_point_draws``).
        model: the observation model of the action.
        action: the action.

    Returns:
        The ``OutcomeCells``.

    Raises:
        EquinoxRuntimeError: if a model without ``gas_cells`` reads the gas block of
            a belief on nodes (its cells come out NaN).
    """
    atoms = b.atoms
    point = Atoms(
        atoms.cloud, atoms.x_mean, jnp.zeros_like(atoms.x_var), atoms.exists
    )  # the nodes stripped: a model reads point blocks
    node_x = atoms.x_mean[..., None] if atoms.nodes is None else atoms.nodes.x
    n_draw, n_planet, n_gas, n_node = node_x.shape
    if jax.eval_shape(model.gas_cells, point, action) is None:
        if atoms.nodes is None:
            draw = model.cells(point, action)
        else:  # the block is on the nodes: x_mean is stale, so no model may read it
            unread = jnp.full_like(atoms.x_mean, jnp.nan)
            draw = model.cells(eqx.tree_at(lambda a: a.x_mean, point, unread), action)
            draw = eqx.error_if(
                draw,
                jnp.any(jnp.isnan(draw)),
                "a model without gas_cells read the gas block of a belief on nodes",
            )
        node = jnp.ones((n_draw, n_planet, n_gas, n_node, 1), draw.dtype)
        codes = jnp.zeros((draw.shape[-1], n_planet, n_gas), int)
        return OutcomeCells(draw, node, codes)

    def at_node(x):
        return model.gas_cells(eqx.tree_at(lambda a: a.x_mean, point, x), action)

    node = jax.vmap(at_node, in_axes=-1, out_axes=-2)(node_x)  # (A, K, G, n, B)
    n_factor = n_planet * n_gas
    codes = np.indices((node.shape[-1],) * n_factor).reshape(n_factor, -1).T
    codes = jnp.asarray(codes.reshape(-1, n_planet, n_gas))
    return OutcomeCells(jnp.ones((n_draw, codes.shape[0]), node.dtype), node, codes)


def cell_probs(b: Belief, cells: OutcomeCells) -> Float[Array, "A Y"]:
    """Each draw's probability of each cell, its factors' nodes averaged.

    Args:
        b: the belief the cells were read on.
        cells: the cells (``outcome_cells``).

    Returns:
        Probabilities of shape ``(A, Y)``; each row sums to one.
    """
    planet, gas = _factor_index(cells.codes)
    picked = _bin_probs(b.atoms, cells)[:, planet, gas, cells.codes]  # (A, Y, K, G)
    return cells.draw * jnp.prod(picked, axis=(-2, -1))


def mask_cells(cells: OutcomeCells, keep: Bool[Array, " Y"]) -> OutcomeCells:
    """``cells`` with every cell outside ``keep`` made sure under every draw and node.

    A sure cell updates nothing (``update_by_cell``), so a cell the forecast cannot
    give is read as harmless, and its NaN weights never reach a derivative.

    Args:
        cells: the cells.
        keep: the cells kept as they are, shape ``(Y,)``.

    Returns:
        The cells with one more bin per factor, a sure one, which every cell outside
        ``keep`` reads.
    """
    sure = cells.node.shape[-1]
    node = jnp.concatenate([cells.node, jnp.ones_like(cells.node[..., :1])], axis=-1)
    codes = jnp.where(keep[:, None, None], cells.codes, sure)
    return OutcomeCells(jnp.where(keep, cells.draw, 1.0), node, codes)


def update_by_cell(b: Belief, cells: OutcomeCells, cell: Int[Array, ""]) -> Belief:
    """The belief after an outcome falls in one cell, without a refit.

    Each draw's weight is multiplied by its probability of the cell (``cell_probs``),
    and the nodes of each gas factor by the factor's probability of its bin at each
    node, over the factor's mean probability of the bin. Given a draw the factors are
    independent and each node is a point, so this is the exact Bayes update of the
    draws and their nodes, at a cost linear in the nodes per factor: the product grid
    of the factors' nodes is never formed. A point block (no nodes) has one node,
    which the draw's weight alone carries. The probability is the same model for every
    role (observation models are equal across roles), so it joins every row of
    ``log_like``. The pool is left as it is.

    Args:
        b: the belief, its gas block a point state or on nodes.
        cells: the cells of the action, read off ``b``'s own draws
            (``outcome_cells``).
        cell: the cell the outcome falls in.

    Returns:
        The updated belief. A cell no draw of a role can give leaves that role's
        weights NaN (see ``weights``); the nodes of a factor that cannot give its bin
        (at a draw that cannot give the cell) are left as they were.
    """
    planet, gas = _factor_index(cells.codes)
    code = cells.codes[cell]  # (K, G)
    mean = _bin_probs(b.atoms, cells)[:, planet, gas, code]  # (A, K, G)
    prob = cells.draw[:, cell] * jnp.prod(mean, axis=(-2, -1))
    b = eqx.tree_at(lambda x: x.log_like, b, b.log_like + guarded_log(prob))
    if b.atoms.nodes is None:
        return b
    pick = jax.nn.one_hot(code, cells.node.shape[-1], dtype=cells.node.dtype)
    at_nodes = jnp.sum(cells.node * pick[:, :, None, :], axis=-1)  # (A, K, G, n)
    live = mean > 0
    omega = b.atoms.nodes.weights
    moved = omega * at_nodes / jnp.where(live, mean, 1.0)[..., None]
    return eqx.tree_at(
        lambda x: x.atoms.nodes.weights, b, jnp.where(live[..., None], moved, omega)
    )


def require_point_draws(b: Belief) -> Belief:
    """``b``, checked to hold point gas blocks: on quadrature nodes, or ``x_var = 0``.

    A hypothetical outcome updates a belief by reweighting its draws and its nodes
    (``update_by_cell``), which is exact on point states only: on a draw whose gas
    block is a distribution, a binned outcome would have to update the block, and no
    single reading stands for a bin. Put such a belief on nodes first
    (``quadrature_draws``). The check runs under ``jit`` as well.

    Args:
        b: the belief.

    Returns:
        ``b`` unchanged.

    Raises:
        EquinoxRuntimeError: if the block is not on nodes and a draw's gas block has
            a nonzero variance.
    """
    if b.atoms.nodes is not None:
        return b
    return eqx.error_if(
        b,
        jnp.any(b.atoms.x_var != 0),
        "a planner reads point draws; put an uncertain gas block on quadrature "
        "nodes with quadrature_draws first",
    )


def _bit(q_claim: Float[Array, "..."]) -> Float[Array, "... 2"]:
    """The distribution of a claim bit, ``(1 - q, q)``."""
    return jnp.stack([1.0 - q_claim, q_claim], axis=-1)


def claim_information(
    forecast: Belief,
    verdict: Belief,
    model: AbstractObservationModel,
    action: Action,
    claim: Claim,
    star,
    planet: Int[Array, ""],
) -> Float[Array, " R"]:
    """Expected gain of each verdict member's log score on one planet's claim bit.

    Row ``r`` is ``E_F[ln p_r(Z | Y) - ln p_r(Z)]``: ``Z`` the claim bit, ``Y`` the
    action's outcome cell, ``p_r`` the verdict's role ``r`` and the expectation over
    the forecast's joint distribution of the two. Each belief is updated by the cell
    on its own draws and nodes (``update_by_cell``); both must hold point gas blocks
    or nodes (``require_point_draws``). Passing the forecast as
    the verdict gives the forecast's own expected information, the mutual information
    of ``Z`` and ``Y`` under the forecast (Lindley 1956); a verdict's gain as the
    forecast expects it equals that mutual information only when the two agree, and
    can be negative when they do not. A cell the forecast gives no mass adds nothing;
    a refused role (a NaN row) gives NaN.

    The gain is on the cells the model reports. Merging cells can raise a verdict
    row, so no bound on the gain of a finer outcome follows from coarser cells.

    Args:
        forecast: the forecast's belief, one role (``R = 1``).
        verdict: the verdict's belief, one role per member.
        model: the observation model of the action.
        action: the action.
        claim: an unconditional claim (``claim.given == 0``).
        star: a pytree of scalar leaves describing the star.
        planet: index of the planet along the ``K`` axis.

    Returns:
        The expected gain per verdict member in nats, shape ``(R,)``.

    Raises:
        ValueError: if the claim is conditional; its bit is defined only where the
            condition holds.
    """
    if claim.given != 0:
        raise ValueError("claim_information needs an unconditional claim")
    forecast, verdict = require_point_draws(forecast), require_point_draws(verdict)
    cells_f = outcome_cells(forecast, model, action)
    cells_v = outcome_cells(verdict, model, action)
    p_cell = outcome_prob(forecast, cell_probs(forecast, cells_f))[0]  # (Y,)
    seen = ~(p_cell <= 0)  # a refused forecast (NaN) reads NaN, not zero
    # A cell the forecast cannot give is made sure: its gain is dropped below, and its
    # NaN weights would otherwise reach the derivative.
    safe_f, safe_v = mask_cells(cells_f, seen), mask_cells(cells_v, seen)

    def q_after(b, cells):
        def one(cell):
            return q(update_by_cell(b, cells, cell), claim, star, planet)

        return jax.vmap(one)(jnp.arange(p_cell.shape[0]))

    q_f, q_v = q_after(forecast, safe_f)[:, 0], q_after(verdict, safe_v)  # (Y,), (Y, R)
    score = jax.vmap(expected_log_score, in_axes=(0, None))
    after = jax.vmap(lambda qv, qf: score(_bit(qv), _bit(qf)))(q_v, q_f)  # (Y, R)
    before = score(
        _bit(q(verdict, claim, star, planet)), _bit(q(forecast, claim, star, planet)[0])
    )
    expected_after = jnp.sum(jnp.where(seen[:, None], p_cell[:, None] * after, 0), 0)
    return expected_after - before


class Diagnostics(eqx.Module):
    """What the weights say about one claim's probability, per role.

    Attributes:
        q: the claim probability.
        mc_se: delta-method Monte Carlo standard error of ``q``; for a conditional
            claim, the error of the ratio ``P(claim and given) / P(given)``.
        khat: Pareto shape of the weight tail; ``-inf`` when the weights have no
            tail (constant weights), NaN when the fit failed.
        n_eff: Kish size of the role weights; a warning, and the clip resolution.
        n_cond: Kish size of the condition-weighted draws, the weights times each
            draw's probability of the condition (``n_eff`` for an unconditional
            claim): the effective number of conditioning draws, the ``n`` of the
            boundary bound of ``certify``; zero when no draw meets the condition.
        cond_mass: ``P(given)``; 1 for an unconditional claim.
        prior_mass_outside: the prior mass of the role outside the refit's support.
            This is a prior mass: the posterior mass outside can be larger and is not
            observable from the draws.
    """

    q: Float[Array, " R"] = eqx.field(converter=jnp.asarray)
    mc_se: Float[Array, " R"] = eqx.field(converter=jnp.asarray)
    khat: Float[Array, " R"] = eqx.field(converter=jnp.asarray)
    n_eff: Float[Array, " R"] = eqx.field(converter=jnp.asarray)
    n_cond: Float[Array, " R"] = eqx.field(converter=jnp.asarray)
    cond_mass: Float[Array, " R"] = eqx.field(converter=jnp.asarray)
    prior_mass_outside: Float[Array, " R"] = eqx.field(converter=jnp.asarray)


def diagnose(
    b: Belief,
    claim: Claim,
    star,
    planet: Int[Array, ""],
    prior_mass_outside: Float[Array, " R"],
) -> Diagnostics:
    """Read ``q`` and its diagnostics off the role weights.

    The diagnostics see a heavy weight tail (``khat``) and a small sample (``n_eff``).
    They cannot see mass the cloud never reached: an unvisited mode shows in neither.

    Args:
        b: the belief.
        claim: the claim.
        star: a pytree of scalar leaves describing the star.
        planet: index of the planet along the ``K`` axis.
        prior_mass_outside: the prior mass of each role outside the support box of the
            refit, shape ``(R,)``, from whoever knows the box and the priors.

    Returns:
        The diagnostics, one entry per role.
    """
    log_w = log_weights(b)  # (R, A)
    per_draw = draw_claim_mass(b, claim, star, planet)  # (C + 1, A)
    h, g = per_draw[-1], per_draw[claim.given]
    joint = jnp.matmul(jax.nn.softmax(log_w, axis=-1), per_draw.T, precision=EXACT)
    cond_mass = joint[:, claim.given]  # one, to rounding, for an unconditional claim
    # No draw meeting the condition leaves no conditioning draw, n_cond = 0 (NaN for a
    # refused role, whose mass is NaN); the Kish size of all -inf would be NaN, so the
    # double where keeps it out of a gradient.
    met = (cond_mass > 0)[:, None]
    unknown = jnp.where(jnp.isnan(cond_mass), jnp.nan, 0.0)
    n_cond = kish_ess(jnp.where(met, log_w + guarded_log(g), log_w))
    return Diagnostics(
        q=safe_ratio(joint[:, -1], cond_mass),
        mc_se=jax.vmap(mc_se, in_axes=(0, None, None))(log_w, h, g),
        khat=jax.vmap(pareto_khat)(log_w),
        n_eff=kish_ess(log_w),
        n_cond=jnp.where(cond_mass > 0, n_cond, unknown),
        cond_mass=cond_mass,
        prior_mass_outside=jnp.broadcast_to(prior_mass_outside, cond_mass.shape),
    )


def clip_q(q: Float[Array, "..."], resolution: Float[Array, "..."]) -> Array:
    """Clip a probability to ``[resolution, 1 - resolution]`` before any logarithm.

    A cloud cannot resolve a probability finer than about one draw in ``2 n_eff``, so
    the usual resolution is ``1 / (2 n_eff)``; a claim every draw agrees on is then
    scored as ``1 - 1 / (2 n_eff)``, not as certain.

    Args:
        q: probabilities.
        resolution: the clip resolution, usually ``0.5 / n_eff``.

    Returns:
        The clipped probabilities.
    """
    return jnp.clip(q, resolution, 1.0 - resolution)


class Certificate(eqx.Module):
    """A statement about a claim that holds under every verdict member.

    Attributes:
        status: ``POSITIVE`` (0), ``NEGATIVE`` (1), ``INAPPLICABLE`` (2: the condition
            has no mass a cloud can resolve) or ``UNRESOLVED`` (3: no certificate, from
            a heavy weight tail, an interval across a threshold, or members that
            disagree).
        q_lo: the smallest lower bound over the members.
        q_hi: the largest upper bound over the members.
    """

    status: Int[Array, ""] = eqx.field(converter=jnp.asarray)
    q_lo: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    q_hi: Float[Array, ""] = eqx.field(converter=jnp.asarray)


def widen_by_mass_outside(
    lo: Float[Array, "..."], hi: Float[Array, "..."], mass: Float[Array, "..."]
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """An interval on a claim probability, widened by the prior mass outside the box.

    The draws see a role's prior only inside the refit's box. The mass ``m`` outside
    may hold the claim or not, so a probability ``p`` read off the draws bounds the
    role's own to ``[(1 - m) p, (1 - m) p + m]``; an interval's ends widen to
    ``(1 - m) lo`` and ``(1 - m) hi + m``.

    Args:
        lo: the lower ends.
        hi: the upper ends.
        mass: the prior mass outside the box, ``m``.

    Returns:
        The widened lower and upper ends.
    """
    return (1.0 - mass) * lo, (1.0 - mass) * hi + mass


def _bounds(d: Diagnostics, z: float):
    """Per-row interval: the wider of the delta-method and Wilson (1927) intervals.

    The Wilson interval stays inside ``(0, 1)`` when every draw agrees, which is where
    the delta-method error is exactly zero. Its ``n`` is the effective number of
    draws that meet the condition, the Kish size of the condition-weighted draws
    ``n_cond`` (``n_eff`` for an unconditional claim), floored at the inapplicable
    limit of one half. ``n_eff * cond_mass`` would overstate it when the weights
    correlate with the condition. The interval is
    clipped to ``[0, 1]``, where a probability lies: a member whose condition no draw
    meets has an infinite ``mc_se`` and the bounds 0 and 1. The prior mass ``m``
    outside the support then widens the interval to its conservative ends,
    ``(1 - m) lo`` and ``(1 - m) hi + m``.
    """
    n = jnp.maximum(d.n_cond, 0.5)
    p = jnp.clip(d.q, 0.0, 1.0)
    z2n = z**2 / n
    center = (p + z2n / 2.0) / (1.0 + z2n)
    half = z / (1.0 + z2n) * jnp.sqrt(p * (1.0 - p) / n + z2n / (4.0 * n))
    lo = jnp.clip(jnp.minimum(d.q - z * d.mc_se, center - half), 0.0, 1.0)
    hi = jnp.clip(jnp.maximum(d.q + z * d.mc_se, center + half), 0.0, 1.0)
    return widen_by_mass_outside(lo, hi, d.prior_mass_outside)


def _members(d: Diagnostics, z: float):
    """The verdict rows of ``d``, their bounds and whether any is inapplicable."""
    if d.q.shape[0] < 2:
        raise ValueError("a certificate needs at least one verdict member (R >= 2)")
    members = jax.tree.map(lambda x: x[1:], d)
    lo, hi = _bounds(members, z)
    inapplicable = jnp.any(members.cond_mass < 0.5 / members.n_eff)
    return members, lo, hi, inapplicable


def certify(
    d: Diagnostics, alpha: float, z: float = 2.0, khat_max: float = KHAT_MAX
) -> Certificate:
    """Issue a certificate on a claim from the verdict members (rows 1 to ``R - 1``).

    Rules, in order:

    - inapplicable when any member's ``cond_mass`` is below its clip resolution
      ``1 / (2 n_eff)``;
    - positive when every member's lower bound is at least ``1 - alpha``, and
      negative when every member's upper bound is at most ``alpha``, both only when
      every member's ``khat <= khat_max`` (a NaN ``khat`` fails);
    - unresolved otherwise.

    Each member's interval is the wider of ``q -+ z mc_se`` and the Wilson score
    interval at ``n = n_cond``, the Kish size of the condition-weighted draws, then
    widened by the member's prior mass
    outside the support. The forecast (row 0) is not a member.

    Args:
        d: the diagnostics of every role.
        alpha: the risk of the certificate.
        z: the width of the interval in standard errors.
        khat_max: the largest acceptable Pareto shape; pass ``khat_threshold(S)`` for
            a cloud of fewer than about 2,200 draws.

    Returns:
        The certificate.

    Raises:
        ValueError: if there are no verdict members.
    """
    members, lo, hi, inapplicable = _members(d, z)
    q_lo, q_hi = jnp.min(lo), jnp.max(hi)
    tail_ok = jnp.all(members.khat <= khat_max)
    status = jnp.select(
        [inapplicable, tail_ok & (q_lo >= 1.0 - alpha), tail_ok & (q_hi <= alpha)],
        [INAPPLICABLE, POSITIVE, NEGATIVE],
        UNRESOLVED,
    )
    return Certificate(status, q_lo, q_hi)


def refit_trigger(
    d: Diagnostics, alpha: float, z: float = 2.0, khat_max: float = KHAT_MAX
) -> Bool[Array, ""]:
    """Whether a refit could resolve the certificate ``certify(d, alpha, z)``.

    True when a member's ``khat`` fails ``khat <= khat_max`` (NaN included) or a
    member's own interval straddles ``alpha`` or ``1 - alpha``; false for an
    inapplicable claim. Members whose intervals sit on opposite sides of a threshold
    disagree on the prior, which more draws cannot settle, so that alone does not
    trigger. The caller states a cap on refit rounds and marks a capped result
    unresolved.

    Args:
        d: the diagnostics of every role.
        alpha: the risk of the certificate.
        z: the width of the interval in standard errors.
        khat_max: the largest acceptable Pareto shape; pass ``khat_threshold(S)`` for
            a cloud of fewer than about 2,200 draws.

    Returns:
        The trigger.

    Raises:
        ValueError: if there are no verdict members.
    """
    members, lo, hi, inapplicable = _members(d, z)
    heavy = ~jnp.all(members.khat <= khat_max)
    straddles = ((lo < 1.0 - alpha) & (hi >= 1.0 - alpha)) | (
        (lo <= alpha) & (hi > alpha)
    )
    return ~inapplicable & (heavy | jnp.any(straddles))
