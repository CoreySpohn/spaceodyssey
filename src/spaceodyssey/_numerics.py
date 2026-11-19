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
    """``ln x``, ``-inf`` at zero, with a finite derivative there.

    A NaN ``x`` (an unknown, such as the mass of a refused role) gives NaN, never
    ``-inf``: a refusal is never read as a zero.
    """
    positive = x > 0
    off = jnp.where(x <= 0, -jnp.inf, jnp.nan)
    return jnp.where(positive, jnp.log(jnp.where(positive, x, 1.0)), off)


def safe_ratio(num, den):
    """``num / den``, zero where ``den`` is zero or negative, with a finite gradient.

    A NaN ``den`` (an unknown, such as the mass of a refused role) gives NaN, never
    zero.
    """
    positive = den > 0
    unknown = jnp.where(jnp.isnan(den), jnp.nan, 0.0)
    return jnp.where(positive, num / jnp.where(positive, den, 1.0), unknown)
