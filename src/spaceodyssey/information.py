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
from spaceodyssey.types import AbstractObservationModel, Action

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
    rules out gives ``-inf``.

    Args:
        pmf: the distribution that is scored.
        forecast: the distribution the outcome is drawn from.

    Returns:
        The expected log score in nats.
    """
    expected = forecast > 0
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


def update_by_cell(b: Belief, cell_prob: Float[Array, " A"]) -> Belief:
    """The belief after an outcome falls in one cell, without a refit.

    Each draw's weight is multiplied by its probability of the cell. That probability
    is the same model for every role (laws are equal across roles), so it joins every
    row of ``log_like``. No row is needed and the pool is left as it is.

    This is the exact Bayes update of the weights. The draws themselves are not
    updated, so the update is exact for what a claim reads only where that is
    independent of the cell given the draw: a point gas block (``x_var = 0``), or a
    cell that reads nothing the claim reads. A Gaussian gas block that the cell reads
    keeps its prior spread.

    Args:
        b: the belief.
        cell_prob: probability of the cell under each draw, shape ``(A,)``.

    Returns:
        The updated belief. A cell no draw of a role can give leaves that role's
        weights NaN (see ``weights``).
    """
    return eqx.tree_at(lambda x: x.log_like, b, b.log_like + guarded_log(cell_prob))


def _bit(q_claim: Float[Array, "..."]) -> Float[Array, "... 2"]:
    """The distribution of a claim bit, ``(1 - q, q)``."""
    return jnp.stack([1.0 - q_claim, q_claim], axis=-1)


def claim_information(
    b: Belief,
    model: AbstractObservationModel,
    action: Action,
    claim: Claim,
    star,
    planet: Int[Array, ""],
) -> Float[Array, " R"]:
    """Expected gain of each role's log score on one planet's claim bit from an action.

    Row ``r`` is ``E_F[ln p_r(Z | Y) - ln p_r(Z)]``: ``Z`` the claim bit, ``Y`` the
    action's outcome cell, and the expectation over the forecast's joint distribution
    of the two (row 0). Each cell updates the weights by ``update_by_cell``. Row 0 is
    the forecast's own expected information, the mutual information of ``Z`` and ``Y``
    under the forecast (Lindley 1956); a verdict row is the verdict's gain as the
    forecast expects it, which equals that mutual information only when the two agree,
    and can be negative when they do not. A cell the forecast gives no mass adds
    nothing.

    The gain is on the cells the model reports. Merging cells can raise a verdict
    row, so no bound on the gain of a finer outcome follows from coarser cells.

    Args:
        b: the belief.
        model: the observation model of the action.
        action: the action.
        claim: an unconditional claim (``claim.given == 0``).
        star: a pytree of scalar leaves describing the star.
        planet: index of the planet along the ``K`` axis.

    Returns:
        The expected gain per role in nats, shape ``(R,)``.

    Raises:
        ValueError: if the claim is conditional; its bit is defined only where the
            condition holds.
    """
    if claim.given != 0:
        raise ValueError("claim_information needs an unconditional claim")
    cells = model.cells(b.atoms, action)  # (A, Y)
    p_cell = outcome_prob(b, cells)[0]  # (Y,), the forecast's
    seen = p_cell > 0
    # A cell the forecast cannot give is replaced by a harmless one: its gain is
    # dropped below, and its NaN weights would otherwise reach the derivative.
    safe = jnp.where(seen, cells, 1.0)
    q_after = jax.vmap(
        lambda c: q(update_by_cell(b, c), claim, star, planet), in_axes=1
    )(safe)  # (Y, R)
    q_before = q(b, claim, star, planet)  # (R,)
    score = jax.vmap(expected_log_score, in_axes=(0, None))
    after = jax.vmap(lambda qy: score(_bit(qy), _bit(qy[0])))(q_after)  # (Y, R)
    before = score(_bit(q_before), _bit(q_before[0]))
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
        n_eff: Kish size of the role weights; a warning, and the cap ``n`` in the
            boundary bound of ``certify``.
        cond_mass: ``P(given)``; 1 for an unconditional claim.
        prior_mass_outside: the prior mass of the role outside the refit's support.
            This is a prior mass: the posterior mass outside can be larger and is not
            observable from the draws.
    """

    q: Float[Array, " R"] = eqx.field(converter=jnp.asarray)
    mc_se: Float[Array, " R"] = eqx.field(converter=jnp.asarray)
    khat: Float[Array, " R"] = eqx.field(converter=jnp.asarray)
    n_eff: Float[Array, " R"] = eqx.field(converter=jnp.asarray)
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
    return Diagnostics(
        q=safe_ratio(joint[:, -1], cond_mass),
        mc_se=jax.vmap(mc_se, in_axes=(0, None, None))(log_w, h, g),
        khat=jax.vmap(pareto_khat)(log_w),
        n_eff=kish_ess(log_w),
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


def _bounds(d: Diagnostics, z: float):
    """Per-row interval: the wider of the delta-method and Wilson (1927) intervals.

    The Wilson interval stays inside ``(0, 1)`` when every draw agrees, which is where
    the delta-method error is exactly zero. Its ``n`` is the effective number of
    draws that meet the condition, ``n_eff * cond_mass`` (``n_eff`` for an
    unconditional claim), floored at the inapplicable limit of one half. The prior
    mass ``m`` outside the support then widens the interval to its conservative ends,
    ``(1 - m) lo`` and ``(1 - m) hi + m``.
    """
    n = jnp.maximum(d.n_eff * d.cond_mass, 0.5)
    p = jnp.clip(d.q, 0.0, 1.0)
    z2n = z**2 / n
    center = (p + z2n / 2.0) / (1.0 + z2n)
    half = z / (1.0 + z2n) * jnp.sqrt(p * (1.0 - p) / n + z2n / (4.0 * n))
    lo = jnp.minimum(d.q - z * d.mc_se, center - half)
    hi = jnp.maximum(d.q + z * d.mc_se, center + half)
    m = d.prior_mass_outside
    return (1.0 - m) * lo, (1.0 - m) * hi + m


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
    interval at ``n = n_eff * cond_mass``, then widened by the member's prior mass
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
