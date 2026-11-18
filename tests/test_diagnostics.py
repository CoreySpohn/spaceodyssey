"""Diagnostics and certificates: boundary precision, the certify rule and the Pareto
tail fit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from spaceodyssey._standin.diagnostics import khat_threshold, pareto_khat
from spaceodyssey.claim import Claim
from spaceodyssey.information import (
    INAPPLICABLE,
    NEGATIVE,
    POSITIVE,
    UNRESOLVED,
    Diagnostics,
    certify,
    clip_q,
    diagnose,
    refit_trigger,
)

PLANET = jnp.asarray(0)


def _unit_weight_belief(toy, n_true, n):
    """`n` draws of unit weight, the first `n_true` claim-true (one planet each)."""
    hz = (jnp.arange(n) < n_true).astype(float)[:, None]
    return toy.belief(toy.atoms(hz, jnp.ones_like(hz), jnp.ones(n)))


@pytest.mark.parametrize("n", [60, 400, 1900])
def test_boundary_precision(toy, n):
    """Unit weights with q = 0.95: mc_se / (1 - q) = sqrt(q / ((1 - q) n)) =
    sqrt(19 / n), which is 0.563, 0.218 and 0.100 at n = 60, 400 and 1900. The count
    behind the clip resolution, n_eff, is returned and equals n; clipping at
    1 / (2 n_eff) leaves 0.95 alone and scores a unanimous claim as 1 - 1 / (2 n).

    Fails if mc_se is not the binomial error at the boundary (a missing square on the
    weights, or no centering on q), or n_eff is not the Kish size of the weights.
    """
    b = _unit_weight_belief(toy, round(0.95 * n), n)
    d = diagnose(b, Claim((toy.InHz(),)), toy.star, PLANET, jnp.zeros(1))
    assert_allclose(d.q, 0.95, rtol=1e-12)
    assert_allclose(d.mc_se / (1.0 - d.q), np.sqrt(19.0 / n), rtol=1e-10)
    assert_allclose(d.n_eff, n, rtol=1e-10)
    resolution = 0.5 / d.n_eff
    assert_allclose(clip_q(d.q, resolution), 0.95, rtol=1e-12)
    assert_allclose(clip_q(jnp.ones(1), resolution), 1.0 - 0.5 / n, rtol=1e-12)


def _diag(q, n_eff, khat=0.0, cond_mass=1.0, outside=0.0):
    """Diagnostics for the forecast (row 0, q = 0.5) and the members in `q`.

    Every draw agrees for a member at q = 0 or 1, so its mc_se is zero; otherwise
    the standard error is the binomial one at n_eff.
    """
    member = jnp.asarray(q, float)
    q_all = jnp.concatenate([jnp.full(1, 0.5), member])
    shape = q_all.shape
    full = lambda v: jnp.broadcast_to(jnp.asarray(v, float), shape)  # noqa: E731
    n = full(n_eff)
    return Diagnostics(
        q_all,
        jnp.sqrt(q_all * (1.0 - q_all) / n),
        full(khat),
        n,
        full(cond_mass),
        full(outside),
    )


ALPHA = 0.05


@pytest.mark.parametrize(
    ("diag", "status"),
    [
        pytest.param(_diag([1.0], 5), UNRESOLVED, id="unanimous-n5"),
        pytest.param(_diag([1.0], 50), UNRESOLVED, id="unanimous-n50"),
        pytest.param(_diag([1.0], 5000), POSITIVE, id="unanimous-n5000"),
        pytest.param(_diag([0.0], 50), UNRESOLVED, id="none-n50"),
        pytest.param(_diag([0.0], 5000), NEGATIVE, id="none-n5000"),
        pytest.param(_diag([1.0], 5000, khat=jnp.nan), UNRESOLVED, id="nan-khat"),
        pytest.param(_diag([1.0], 5000, cond_mass=5e-5), INAPPLICABLE, id="no-cond"),
        pytest.param(_diag([1.0], 5000, outside=0.06), UNRESOLVED, id="outside"),
        pytest.param(_diag([0.97], 5000), POSITIVE, id="one-member-097"),
        pytest.param(_diag([0.97, 0.92], 5000), UNRESOLVED, id="members-disagree"),
    ],
)
def test_certify_at_the_boundary(diag, status):
    """Unanimous draws (mc_se = 0) at alpha = 0.05 and z = 2: only n_eff = 5000
    certifies (bounds in the next test), and the negative side is symmetric. A NaN
    khat is unresolved; a condition mass of 5e-5, below 1 / (2 * 5000), is
    inapplicable; a prior mass 0.06 outside the support lowers the bound below 0.95.
    Members at 0.97 and 0.92 are unresolved though 0.97 alone certifies. The forecast
    (row 0, q = 0.5) never blocks a certificate.

    Fails if the interval is the delta-method one alone (it certifies at n_eff = 5
    once every draw agrees), if a NaN khat passes the tail gate, if the condition or
    the outside mass is ignored, if one member suffices, or if row 0 is a member.
    """
    assert int(certify(diag, ALPHA).status) == status


def test_certify_bounds_at_unanimity():
    """With every draw agreeing the bound is Wilson's: n / (n + z^2) = 0.556, 0.926
    and 0.9992 at n_eff = 5, 50 and 5000 (z = 2), and 4 / (n + 4) on the negative
    side. A prior mass m = 0.06 outside the support moves the lower bound to its
    conservative end (1 - m) * 0.9992 = 0.939 and the upper one to (1 - m) * 1 + m.

    Fails if the Wilson center or half-width is miscomputed, or the outside mass is
    applied to the wrong end.
    """
    n = np.array([5.0, 50.0, 5000.0])
    lows = np.array([float(certify(_diag([1.0], k), ALPHA).q_lo) for k in n])
    highs = np.array([float(certify(_diag([0.0], k), ALPHA).q_hi) for k in n])
    assert_allclose(lows, n / (n + 4.0), rtol=1e-12)
    assert_allclose(lows, [0.556, 0.926, 0.9992], atol=5e-4)
    assert_allclose(highs, 4.0 / (n + 4.0), rtol=1e-12)
    outside = certify(_diag([1.0], 5000.0, outside=0.06), ALPHA)
    assert_allclose(outside.q_lo, 0.94 * 5000.0 / 5004.0, rtol=1e-12)
    assert_allclose(outside.q_hi, 1.0, rtol=1e-12)


def test_pareto_khat_recovers_a_known_tail():
    """Weights w = U^-k with U uniform have a Pareto tail of shape k. One estimate at
    S = 5000 (a 213-draw tail) scatters by about 0.14, so the test averages 64
    replicates: N = (4 * 0.14 / 0.1)^2 = 31 replicates puts 0.1 at 4 sigma; 64 gives
    the mean a spread of 0.018. Both k = 0.5 and k = 1.0 (above the 0.7 gate) are
    recovered within 0.1. Constant weights give NaN, and through diagnose and certify
    the refit trigger is true. Below about 2,200 draws the threshold drops: 0.5 at
    S = 100.

    Fails if the fit's sign convention is flipped, the tail is taken from the wrong
    end, the exceedances are not over the next-largest weight, or constant weights
    pass the tail gate.
    """
    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    for k in (0.5, 1.0):
        log_w = -k * jnp.log(
            jax.vmap(lambda key: jax.random.uniform(key, (5000,)))(keys)
        )
        khat = jax.vmap(pareto_khat)(log_w)
        assert abs(float(jnp.mean(khat)) - k) < 0.1, (k, jnp.mean(khat))

    assert jnp.isnan(pareto_khat(jnp.zeros(5000)))
    assert khat_threshold(100) == pytest.approx(0.5)
    assert khat_threshold(5000) == pytest.approx(0.7)


def test_constant_weights_trigger_a_refit(toy):
    """A unanimous claim on 5000 draws of equal weight: khat is NaN, so the
    certificate is unresolved and the refit trigger is true; the same diagnostics
    with a finite khat certify and do not trigger.

    Fails if a NaN khat reaches a certificate or does not trigger a refit.
    """
    b = toy.belief(
        toy.atoms(jnp.ones((5000, 1)), jnp.ones((5000, 1)), jnp.ones(5000)), n_roles=2
    )
    d = diagnose(b, Claim((toy.InHz(),)), toy.star, PLANET, jnp.zeros(2))
    assert jnp.all(jnp.isnan(d.khat))
    cert = certify(d, ALPHA)
    assert int(cert.status) == UNRESOLVED
    assert bool(refit_trigger(d, cert, ALPHA))

    finite = Diagnostics(
        d.q, d.mc_se, jnp.zeros(2), d.n_eff, d.cond_mass, d.prior_mass_outside
    )
    cert = certify(finite, ALPHA)
    assert int(cert.status) == POSITIVE
    assert not bool(refit_trigger(finite, cert, ALPHA))
