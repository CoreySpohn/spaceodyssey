"""Information: the change of a log score on the claim-true count between beliefs."""

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from spaceodyssey.belief import Belief, chain
from spaceodyssey.claim import Claim


def log_score(pmf: Float[Array, " N"], count: Int[Array, ""]) -> Float[Array, ""]:
    """Log of the probability a count distribution gives to the realized count.

    A count beyond the last bin is scored in the last bin, which holds the lumped tail.

    Args:
        pmf: the count distribution.
        count: the realized count.

    Returns:
        The log score in nats.
    """
    return jnp.log(pmf[jnp.minimum(count, pmf.shape[0] - 1)])


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
        ``ln chain_after - ln chain_before`` per conjunct, in nats.
    """
    after = chain(b_after, claim, star, planet)[role]
    before = chain(b_before, claim, star, planet)[role]
    return jnp.log(after) - jnp.log(before)
