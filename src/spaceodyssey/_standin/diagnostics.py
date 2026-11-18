"""Weight diagnostics: Kish size, Pareto tail shape and Monte Carlo standard error.

Stand-in for the photomancy particle diagnostics. The whole module is replaced when
those land; it holds no package imports so that the swap is a change of import.

Every function takes the log weights of one cloud along the last axis; the weights
need not be normalized.
"""

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

# Vehtari et al. (2024): above this Pareto shape an importance-sampling estimate is
# unreliable at any sample size.
KHAT_MAX = 0.7

# The weakly informative prior on the shape in the generalized-Pareto fit of Vehtari
# et al. (2024): a pseudo-sample of this size centered on this shape.
_PRIOR_SIZE = 10
_PRIOR_SHAPE = 0.5


def kish_ess(log_w: Float[Array, "... A"]) -> Float[Array, "..."]:
    """Kish effective sample size ``(sum w)^2 / sum w^2`` (Kish 1965).

    Stand-in for the photomancy diagnostics.

    Args:
        log_w: log weights, unnormalized, along the last axis.

    Returns:
        ``exp(2 logsumexp(log_w) - logsumexp(2 log_w))``.
    """
    lse = jax.nn.logsumexp
    return jnp.exp(2.0 * lse(log_w, axis=-1) - lse(2.0 * log_w, axis=-1))


def khat_threshold(n_draws: int) -> float:
    """Sample-size threshold on the Pareto shape (Vehtari et al. 2024).

    Below about 2,200 draws the shape that still gives a reliable estimate is smaller
    than 0.7: ``min(1 - 1 / log10(S), 0.7)``.

    Args:
        n_draws: the number of draws ``S``.

    Returns:
        The largest acceptable shape.
    """
    return min(1.0 - 1.0 / math.log10(n_draws), KHAT_MAX)


def _tail_size(n_draws: int) -> int:
    """Draws in the fitted tail: ``min(0.2 S, 3 sqrt(S))``, rounded up."""
    return min(math.ceil(0.2 * n_draws), math.ceil(3.0 * math.sqrt(n_draws)))


def _gpd_shape(x: Float[Array, " M"]) -> Float[Array, ""]:
    """Generalized-Pareto shape of sorted exceedances (Zhang and Stephens 2009).

    The profile posterior of the scale-to-shape ratio is approximated on a fixed grid
    and averaged; the shape is then shrunk toward the prior of Vehtari et al. (2024).
    A tail with no spread (largest exceedance zero) has no shape and gives NaN.
    """
    n = x.shape[0]
    n_grid = 30 + int(math.sqrt(n))
    j = jnp.arange(1, n_grid + 1, dtype=x.dtype)
    quartile = x[int(n / 4 + 0.5) - 1]
    b = 1.0 / x[-1] + (1.0 - jnp.sqrt(n_grid / (j - 0.5))) / (3.0 * quartile)
    k = jnp.mean(jnp.log1p(-b[:, None] * x[None, :]), axis=1)
    profile = n * (jnp.log(-b / k) - k - 1.0)
    b_post = jax.nn.softmax(profile) @ b
    k_post = jnp.mean(jnp.log1p(-b_post * x))
    shrunk = (n * k_post + _PRIOR_SIZE * _PRIOR_SHAPE) / (n + _PRIOR_SIZE)
    return jnp.where(x[-1] > 0, shrunk, jnp.nan)


def pareto_khat(log_w: Float[Array, " A"]) -> Float[Array, ""]:
    """Pareto shape of the upper tail of the weights (Vehtari et al. 2024).

    Stand-in for the photomancy diagnostics. A generalized-Pareto distribution is
    fitted (Zhang and Stephens 2009) to the excess of the ``min(0.2 S, 3 sqrt(S))``
    largest weights over the next largest. A shape above ``khat_threshold(S)`` means
    the weight tail is too heavy for a reliable estimate. Constant weights have no
    tail and give NaN, which every gate treats as a failure.

    Args:
        log_w: log weights of one cloud, unnormalized, shape ``(A,)``.

    Returns:
        The estimated shape.
    """
    n_tail = _tail_size(log_w.shape[0])
    top, _ = jax.lax.top_k(log_w, n_tail + 1)  # descending
    scale = top[0]
    tail = jnp.flip(top[:n_tail])  # ascending
    x = jnp.exp(tail - scale) - jnp.exp(top[n_tail] - scale)
    return _gpd_shape(x)


def mc_se(
    log_w: Float[Array, " A"],
    h: Float[Array, " A"],
    g: Float[Array, " A"] | float = 1.0,
) -> Float[Array, ""]:
    """Delta-method Monte Carlo standard error of a self-normalized estimate.

    Stand-in for the photomancy diagnostics. For ``q = sum w h / sum w g`` the error
    is ``sqrt(sum w^2 (h - q g)^2) / sum w g``; with ``g = 1`` (an unconditional
    estimate) it is ``sqrt(sum w^2 (h - q)^2) / sum w``. It is exactly zero when every
    draw agrees.

    Args:
        log_w: log weights, unnormalized, shape ``(A,)``.
        h: the per-draw value whose weighted mean is estimated (for a conditional
            estimate, the joint of the condition and the value).
        g: the per-draw probability of the condition; 1 for an unconditional estimate.

    Returns:
        The standard error; infinite when the condition has zero weight.
    """
    w = jax.nn.softmax(log_w)
    g = jnp.broadcast_to(jnp.asarray(g, dtype=w.dtype), w.shape)
    den = w @ g
    ok = den > 0
    safe_den = jnp.where(ok, den, 1.0)
    q = (w @ h) / safe_den
    se = jnp.sqrt(jnp.sum(w**2 * (h - q * g) ** 2)) / safe_den
    return jnp.where(ok, se, jnp.inf)
