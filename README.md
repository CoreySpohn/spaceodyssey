# spaceodyssey

A JAX and Equinox simulator for Habitable Worlds Observatory mission studies. Beliefs over
planetary systems are weighted clouds of draws; claims are predicates on a planet; information is
the change of a log score on the count of claim-true planets.

This is a from-scratch rewrite and is built layer by layer. The current layers are the types, the
claim, the belief and the information, then the roles that reweight one cloud and the certificates
read off their weights.

A gas abundance with mean 0 and variance 1 above a floor of 0 holds with probability one half:

```python
import jax.numpy as jnp

from spaceodyssey.claim import GasAbove
from spaceodyssey.types import Atoms, WeightedCloud

cloud = WeightedCloud({}, jnp.zeros(1), jnp.nan, jnp.zeros(1), chart="toy")
atoms = Atoms(
    cloud,
    x_mean=jnp.zeros((1, 1, 1)),
    x_var=jnp.ones((1, 1, 1)),
    exists=jnp.ones((1, 1), dtype=bool),
)
assert float(GasAbove(0, 0.0).prob(atoms, ())[0, 0]) == 0.5
```
