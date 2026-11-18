"""Claims: predicates on one planet, built from conjuncts with a probability each."""

import equinox as eqx
import jax.numpy as jnp
from jax.scipy.special import ndtr
from jaxtyping import Array, Float

from spaceodyssey.types import Atoms


class AbstractConjunct(eqx.Module):
    """One factor of a claim: the probability that it holds for each draw and planet."""

    def prob(self, atoms: Atoms, star) -> Float[Array, "A K"]:
        """Probability that the conjunct holds, per draw and planet.

        Args:
            atoms: the weighted draws.
            star: a pytree of scalar leaves describing the star (an empty pytree in
                the toy).

        Returns:
            Probabilities in ``[0, 1]``, shape ``(A, K)``.
        """
        raise NotImplementedError


class GasAbove(AbstractConjunct):
    """A gas abundance above a floor, read from the gas block.

    The abundance of each draw and planet is Gaussian, so the probability is
    ``Phi((x_mean - floor) / sqrt(x_var))``. A point state (``x_var = 0``) gives the
    indicator ``x_mean > floor``.

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
        conjuncts: the conjuncts, in the order used by ``chain``.
        given: how many leading conjuncts form the condition; 0 means none, static.
    """

    conjuncts: tuple[AbstractConjunct, ...]
    given: int = eqx.field(static=True, default=0)
