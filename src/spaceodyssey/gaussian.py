"""Deterministic Gaussian probability integrals."""

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr, ndtri

from spaceodyssey.errors import UnsupportedProfile


def _standard_bound(bound, mean, scale):
    # Keep infinities out of differentiated arithmetic, including scale VJPs.
    finite = jnp.isfinite(bound)
    value = (jnp.where(finite, bound, 0.0) - mean) / scale
    return jnp.where(finite, value, bound)


def normal_rectangle_probability(mean, covariance, lower, upper, *, nodes=32):
    """Integrate a normal density over a rectangle using fixed quadrature.

    Uses the sequential-conditioning transformation of Genz (1992) and a
    tensor product of Gauss-Legendre nodes on the unit cube. No random
    numbers or adaptive stopping are used. The result supports JIT and
    differentiation with respect to finite bounds, mean, and covariance.
    Accuracy depends on the geometry; compare node orders to assess error.
    Enable JAX float64 for tolerances finer than single precision.

    Args:
        mean: Normal mean, shape (D,).
        covariance: Positive definite covariance, shape (D, D).
        lower: Lower bounds, shape (D,); negative infinity is allowed.
        upper: Upper bounds, shape (D,); positive infinity is allowed.
        nodes: Fixed positive integer node count per integration dimension.
            Must be static under JIT. There are nodes**(D - 1) evaluations.

    Returns:
        Scalar rectangle probability. Empty rectangles have probability zero.

    Raises:
        UnsupportedProfile: If D is outside the supported range 1 through 4.
        ValueError: If shapes or the node count are invalid.
    """
    mean, covariance, lower, upper = map(jnp.asarray, (mean, covariance, lower, upper))
    if mean.ndim != 1:
        raise ValueError("mean must have shape (D,)")
    dimension = mean.shape[0]
    if not 1 <= dimension <= 4:
        raise UnsupportedProfile("normal rectangle quadrature supports 1 <= D <= 4")
    if (
        covariance.shape != (dimension, dimension)
        or lower.shape != mean.shape
        or upper.shape != mean.shape
    ):
        raise ValueError("covariance and bounds must share mean's dimension D")
    if type(nodes) is not int or nodes < 1:
        raise ValueError("nodes must be a positive integer")
    roots, weights = np.polynomial.legendre.leggauss(nodes)
    if dimension == 1:
        grid = jnp.empty((1, 0))
        quadrature_weights = jnp.ones(1)
    else:
        grid = jnp.asarray(
            np.stack(np.meshgrid(*[(roots + 1) / 2] * (dimension - 1)), -1).reshape(
                -1, dimension - 1
            )
        )
        quadrature_weights = jnp.asarray(
            np.prod(
                np.stack(np.meshgrid(*[weights / 2] * (dimension - 1)), -1), -1
            ).reshape(-1)
        )
    cholesky = jnp.linalg.cholesky(covariance)
    latent = jnp.zeros((grid.shape[0], dimension), dtype=cholesky.dtype)
    integrand = jnp.ones(grid.shape[0], dtype=cholesky.dtype)
    for i in range(dimension):
        conditional_mean = mean[i] + latent[:, :i] @ cholesky[i, :i]
        a = _standard_bound(lower[i], conditional_mean, cholesky[i, i])
        b = _standard_bound(upper[i], conditional_mean, cholesky[i, i])
        reflect = a > 0
        cdf_low = ndtr(jnp.where(reflect, -b, a))
        cdf_high = ndtr(jnp.where(reflect, -a, b))
        width = jnp.maximum(cdf_high - cdf_low, 0.0)
        integrand = integrand * width
        if i < dimension - 1:
            quantile = cdf_low + grid[:, i] * width
            eps = jnp.finfo(quantile.dtype).eps
            z = ndtri(jnp.clip(quantile, eps, 1 - eps))
            latent = latent.at[:, i].set(jnp.where(reflect, -z, z))
    return jnp.sum(quadrature_weights * integrand)
