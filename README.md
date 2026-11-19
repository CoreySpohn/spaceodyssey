# spaceodyssey

A JAX and Equinox simulator for Habitable Worlds Observatory mission studies. Beliefs over
planetary systems are weighted clouds of draws; claims are predicates on a planet; information is
the change of a log score on the count of claim-true planets.

This is a from-scratch rewrite and is built layer by layer. The current layers are the types, the
claim, the belief and the information, then the roles that reweight one cloud and the certificates
read off their weights, the value of an observation, and the survey loop: a policy runs against a
sealed world, and the log it leaves is scored against the unsealed truth.

A gas abundance with mean 0 and variance 1 above a floor of 0 holds with probability one half:

```python
import jax
import jax.numpy as jnp

from spaceodyssey.claim import GasAbove
from spaceodyssey.types import AbstractRolePrior, Atoms, WeightedCloud

jax.config.update("jax_enable_x64", True)  # a cloud's log_target is float64


class Flat(AbstractRolePrior):
    """A constant prior density, the refit prior of this one-draw cloud."""

    def log_prob(self, atoms, star):
        return jnp.zeros(atoms.x_mean.shape[0])

    def sample(self, key, star):
        raise NotImplementedError


cloud = WeightedCloud({}, jnp.zeros(1), jnp.zeros(1), jnp.nan, Flat("toy"))
atoms = Atoms(
    cloud,
    x_mean=jnp.zeros((1, 1, 1)),
    x_var=jnp.ones((1, 1, 1)),
    exists=jnp.ones((1, 1), dtype=bool),
)
assert float(GasAbove(0, 0.0).prob(atoms, ())[0, 0]) == 0.5
```
