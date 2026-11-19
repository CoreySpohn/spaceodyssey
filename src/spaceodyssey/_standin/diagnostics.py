"""Weight diagnostics: Kish size, Pareto tail shape and Monte Carlo standard error.

Stand-in for the photomancy particle diagnostics. The whole module is replaced when
those land; it holds no package imports so that the swap is a change of import.

Every function takes the log weights of one cloud along the last axis; the weights
need not be normalized.
"""

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

# Vehtari et al. (2024): above this Pareto shape an importance-sampling estimate is
# unreliable at any sample size.
KHAT_MAX = 0.7

# The weakly informative prior on the shape in the generalized-Pareto fit of Vehtari
# et al. (2024): a pseudo-sample of this size centered on this shape.
_PRIOR_SIZE = 10
_PRIOR_SHAPE = 0.5

# Fewest exceedances a generalized-Pareto fit is attempted on (as in Vehtari et al.).
_MIN_EXCEEDANCES = 5

# Level of the upper confidence bound that the largest exceedance puts on a Pareto
# tail index: the bound is below the true index with this probability.
_MAX_BOUND_LEVEL = 1e-3

# Two log weights within this many float epsilons (relative, floored at one) are tied.
_TIE_ULPS = 64


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


def _gpd_shape(x: Float[Array, " M"], n_fit: Int[Array, ""]) -> Float[Array, ""]:
    """Generalized-Pareto shape of sorted exceedances (Zhang and Stephens 2009).

    ``x`` is ascending; its first ``M - n_fit`` entries are zero (draws tied with the
    cutoff, which carry no exceedance) and are left out of the fit. The profile
    posterior of the scale-to-shape ratio is approximated on a fixed grid and
    averaged; the shape is then shrunk toward the prior of Vehtari et al. (2024).
    """
    n = x.shape[0]
    n_grid = 30 + int(math.sqrt(n))
    m = jnp.maximum(n_fit, 1).astype(x.dtype)
    j = jnp.arange(1, n_grid + 1, dtype=x.dtype)
    # First quartile of the fitted values, which are the last n_fit entries.
    quartile = x[jnp.clip(n - n_fit + (n_fit + 2) // 4 - 1, 0, n - 1)]
    b = 1.0 / x[-1] + (1.0 - jnp.sqrt(n_grid / (j - 0.5))) / (3.0 * quartile)
    # A zero entry adds log1p(0) = 0, so the sums run over the fitted values only.
    k = jnp.sum(jnp.log1p(-b[:, None] * x[None, :]), axis=1) / m
    profile = m * (jnp.log(-b / k) - k - 1.0)
    b_post = jax.nn.softmax(profile) @ b
    k_post = jnp.sum(jnp.log1p(-b_post * x)) / m
    return (m * k_post + _PRIOR_SIZE * _PRIOR_SHAPE) / (m + _PRIOR_SIZE)


def _ties(diff: Float[Array, "..."], value: Float[Array, "..."]) -> Bool[Array, "..."]:
    """Whether a log-weight difference is zero within float tolerance at ``value``."""
    eps = jnp.finfo(diff.dtype).eps
    return diff <= _TIE_ULPS * eps * jnp.maximum(1.0, jnp.abs(value))


def pareto_khat(log_w: Float[Array, " A"]) -> Float[Array, ""]:
    """Pareto shape of the upper tail of the weights (Vehtari et al. 2024).

    Stand-in for the photomancy diagnostics. A generalized-Pareto distribution is
    fitted (Zhang and Stephens 2009) to the excess of the ``min(0.2 S, 3 sqrt(S))``
    largest weights over the next largest. A shape above ``khat_threshold(S)`` means
    the weight tail is too heavy for a reliable estimate.

    Weights with no tail give ``-inf``, which passes every gate: all finite log
    weights equal within float tolerance (exact posterior draws), or the largest
    weights a plateau that no draw exceeds (a few discrete weight levels, whose Kish
    size is then at least the tail size). Draws tied with the cutoff are left out of
    the fit. ``NaN`` means the fit failed or is impossible (fewer than five
    exceedances, or no finite weight) and fails every gate.

    The fitted shape is capped by the bound that the largest weight puts on a Pareto
    tail: with ``n`` exceedances, cutoff ``u`` and largest weight ``w_max``, a tail of
    index ``k`` gives ``w_max / u <= r`` with probability ``(1 - r**(-1/k))**n``, so
    ``log(w_max / u) / -log(1 - 0.001**(1/n))`` is a 99.9 percent upper bound on the
    index. The cap matters when the cutoff falls inside a cluster of nearly equal
    weights: the exceedances then span many decades and the fit alone reads a heavy
    tail on bounded weights. The cap tests a Pareto tail in the weights themselves; a
    heavy component whose scale is far below the cutoff can pass, and carries a
    correspondingly small share of the total weight.

    Args:
        log_w: log weights of one cloud, unnormalized, shape ``(A,)``.

    Returns:
        The estimated shape, ``-inf`` for no tail, or ``NaN`` for a failed fit.
    """
    n_draws = log_w.shape[0]
    finite = jnp.isfinite(log_w)
    top_w = jnp.max(jnp.where(finite, log_w, -jnp.inf))
    low_w = jnp.min(jnp.where(finite, log_w, jnp.inf))
    flat = _ties(top_w - low_w, jnp.maximum(jnp.abs(top_w), jnp.abs(low_w)))
    n_tail = min(_tail_size(n_draws), n_draws - 1)
    if n_tail < 1:  # one draw: no spread
        return jnp.where(jnp.any(finite), -jnp.inf, jnp.nan)
    top, _ = jax.lax.top_k(log_w, n_tail + 1)  # descending
    tail = jnp.flip(top[:n_tail])  # ascending
    cutoff = top[n_tail]
    # A NaN difference (both -inf) is not an exceedance either.
    exceeds = ~_ties(tail - cutoff, tail) & ~jnp.isnan(tail - cutoff)
    x = jnp.where(exceeds, jnp.exp(tail - top[0]) - jnp.exp(cutoff - top[0]), 0.0)
    n_fit = jnp.sum(exceeds)
    khat = _gpd_shape(jnp.sort(x), n_fit)
    # A Pareto tail of index k puts the largest of n_fit exceedances at or below r
    # times the cutoff with probability (1 - r**(-1/k))**n_fit, so the observed
    # maximum bounds the index from above. A cutoff of -inf leaves the shape as fitted.
    level = _MAX_BOUND_LEVEL ** (1.0 / jnp.maximum(n_fit, 1))
    khat = jnp.minimum(khat, (top[0] - cutoff) / -jnp.log1p(-level))
    return jnp.select(
        [~jnp.any(finite), flat | (n_fit == 0), n_fit < _MIN_EXCEEDANCES],
        [jnp.nan, -jnp.inf, jnp.nan],
        khat,
    )


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
