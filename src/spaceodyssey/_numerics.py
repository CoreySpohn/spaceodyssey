"""Guarded numerics shared by the package: one place for each guard.

Each guard keeps a value defined where the plain expression is not, and keeps its
derivative finite there: a ``jnp.where`` alone passes the NaN of the untaken branch into
the gradient, so the argument is made safe first. This module imports nothing from the
package.
"""

import jax.numpy as jnp
from jax import lax

EXACT = lax.Precision.HIGHEST
"""Precision of every product of weights. On a GPU the default precision of a float32
dot or convolution can drop to TF32, about three digits."""


def guarded_log(x):
    """``ln x``, ``-inf`` at zero, with a finite derivative there."""
    positive = x > 0
    return jnp.where(positive, jnp.log(jnp.where(positive, x, 1.0)), -jnp.inf)


def safe_ratio(num, den):
    """``num / den``, zero where ``den`` is not positive, with a finite gradient."""
    positive = den > 0
    return jnp.where(positive, num / jnp.where(positive, den, 1.0), 0.0)
