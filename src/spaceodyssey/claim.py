"""Claims: predicates on one planet, built from conjuncts with a probability each."""

import abc

import equinox as eqx
import jax.numpy as jnp
from jax.scipy.special import ndtr
from jaxtyping import Array, Float

from spaceodyssey.types import Atoms


class AbstractConjunct(eqx.Module):
    """One factor of a claim: the probability that it holds for each draw and planet."""

    @abc.abstractmethod
    def prob(self, atoms: Atoms, star) -> Float[Array, "A K"]:
        """Probability that the conjunct holds, per draw and planet.

        Args:
            atoms: the weighted draws.
            star: a pytree of scalar leaves describing the star (an empty pytree in
                the toy).

        Returns:
            Probabilities in ``[0, 1]``, shape ``(A, K)``.
        """


class GasAbove(AbstractConjunct):
    """A gas abundance above a floor, read from the gas block.

    The abundance of each draw and planet is Gaussian, so the probability is
    ``Phi((x_mean - floor) / sqrt(x_var))``. A point state (``x_var = 0``) gives the
    strict indicator ``x_mean > floor``: 0 at ``x_mean == floor``, where the limit
    ``x_var -> 0`` is 0.5.

    Attributes:
        gas: index of the gas in the gas block, static.
        floor: abundance floor, in the units of ``x_mean``.
    """

    gas: int = eqx.field(static=True)
    floor: Float[Array, ""] = eqx.field(converter=jnp.asarray)

    def prob(self, atoms, star):
        """Probability of exceeding the floor; the indicator for a point state."""
        mean = atoms.x_mean[:, :, self.gas]
        var = atoms.x_var[:, :, self.gas]
        spread = var > 0
        # The double where keeps the gradient finite at x_var = 0.
        sd = jnp.sqrt(jnp.where(spread, var, 1.0))
        gaussian = ndtr((mean - self.floor) / sd)
        return jnp.where(spread, gaussian, (mean > self.floor).astype(gaussian.dtype))


class Claim(eqx.Module):
    """A conjunction of conjuncts, optionally conditioned on its leading conjuncts.

    Attributes:
        conjuncts: the conjuncts, in the order used by ``chain``; at least one.
        given: how many leading conjuncts form the condition; 0 means none, static. At
            least one conjunct stays outside the condition, so ``0 <= given <
            len(conjuncts)``.

    Raises:
        ValueError: if there are no conjuncts or ``given`` is out of range.
    """

    conjuncts: tuple[AbstractConjunct, ...]
    given: int = eqx.field(static=True, default=0)

    def __check_init__(self):
        """Reject an empty claim and a condition that covers every conjunct."""
        n = len(self.conjuncts)
        if n == 0:
            raise ValueError("a claim needs at least one conjunct")
        if not 0 <= self.given < n:
            raise ValueError(
                f"given must be in [0, {n}) for {n} conjuncts, not {self.given}"
            )
