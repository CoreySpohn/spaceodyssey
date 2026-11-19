"""Diagnostics and certificates: boundary precision, the certify rule and the Pareto
tail fit.

Every belief here has N_DRAW draws (a smaller cloud is padded with zero-weight draws,
which every diagnostic treats as absent) and two roles, and the kernels run through the
shared compiled wrappers (`toy.jit`), so the file compiles each kernel once.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from spaceodyssey._standin.diagnostics import khat_threshold
from spaceodyssey.claim import Claim
from spaceodyssey.information import (
    INAPPLICABLE,
    NEGATIVE,
    POSITIVE,
    UNRESOLVED,
    Diagnostics,
    clip_q,
)

PLANET = jnp.asarray(0)
N_DRAW = 5000  # every belief's draw count; smaller clouds are padded
NO_MASS = jnp.zeros(2)  # no prior mass outside the support, two roles


def _belief(toy, hz, size, w=None):
    """A two-role belief of flat priors over `len(hz)` draws of weight `w` (unit by
    default), padded to N_DRAW draws of zero weight.
    """
    n = len(hz)
    w = np.ones(n) if w is None else np.asarray(w)
    pad = np.zeros(N_DRAW - n)
    hz, size = (
        np.concatenate([np.asarray(x, float), pad])[:, None] for x in (hz, size)
    )
    return toy.belief(toy.atoms(hz, size, np.concatenate([w, pad])), n_roles=2)


def _unit_weight_belief(toy, n_true, n):
    """`n` draws of unit weight, the first `n_true` claim-true (one planet each)."""
    return _belief(toy, np.arange(n) < n_true, np.ones(n))


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
    d = toy.jit.diagnose(b, Claim((toy.InHz(),)), toy.star, PLANET, NO_MASS)
    assert_allclose(d.q, 0.95, rtol=1e-12)
    assert_allclose(d.mc_se / (1.0 - d.q), np.sqrt(19.0 / n), rtol=1e-10)
    assert_allclose(d.n_eff, n, rtol=1e-10)
    resolution = 0.5 / d.n_eff
    assert_allclose(clip_q(d.q, resolution), 0.95, rtol=1e-12)
    assert_allclose(clip_q(jnp.ones(2), resolution), 1.0 - 0.5 / n, rtol=1e-12)


def _diag(q, n_eff, khat=0.0, cond_mass=1.0, outside=0.0):
    """Diagnostics for the forecast (row 0, q = 0.5) and the members in `q`.

    Every draw agrees for a member at q = 0 or 1, so its mc_se is zero; otherwise
    the standard error is the binomial one at n_eff. The draws have unit weights and
    a condition that each draw meets or not, so the Kish size of the
    condition-weighted draws is the count of draws that meet it, n_eff * cond_mass.
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
        n * full(cond_mass),
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
        pytest.param(_diag([1.0], 5000, cond_mass=2e-3), UNRESOLVED, id="few-cond"),
        pytest.param(_diag([1.0], 5000, outside=0.06), UNRESOLVED, id="outside"),
        pytest.param(_diag([0.97], 5000), POSITIVE, id="one-member-097"),
        pytest.param(_diag([0.97, 0.92], 5000), UNRESOLVED, id="members-disagree"),
    ],
)
def test_certify_at_the_boundary(toy, diag, status):
    """Unanimous draws (mc_se = 0) at alpha = 0.05 and z = 2: only n_eff = 5000
    certifies (bounds in the next test), and the negative side is symmetric. A NaN
    khat is unresolved; a condition mass of 5e-5, below 1 / (2 * 5000), is
    inapplicable; a condition mass of 2e-3 leaves 5000 x 2e-3 = 10 conditioning
    draws, whose Wilson bound 10 / 14 = 0.714 is unresolved; a prior mass 0.06
    outside the support lowers the bound below 0.95. Members at 0.97 and 0.92 are
    unresolved though 0.97 alone certifies. The forecast (row 0, q = 0.5) never
    blocks a certificate.

    Fails if the interval is the delta-method one alone (it certifies at n_eff = 5
    once every draw agrees), if a NaN khat passes the tail gate, if the condition or
    the outside mass is ignored, if the Wilson n of a conditional claim is n_eff and
    not the draws that meet the condition, if one member suffices, or if row 0 is a
    member.
    """
    assert int(toy.jit.certify(diag, ALPHA).status) == status


def test_certify_bounds_at_unanimity(toy):
    """With every draw agreeing the bound is Wilson's: n / (n + z^2) = 0.556, 0.926
    and 0.9992 at n_eff = 5, 50 and 5000 (z = 2), and 4 / (n + 4) on the negative
    side. A prior mass m = 0.06 outside the support moves the lower bound to its
    conservative end (1 - m) * 0.9992 = 0.939 and the upper one to (1 - m) * 1 + m.

    Fails if the Wilson center or half-width is miscomputed, or the outside mass is
    applied to the wrong end.
    """
    n = np.array([5.0, 50.0, 5000.0])
    certify = toy.jit.certify
    lows = np.array([float(certify(_diag([1.0], k), ALPHA).q_lo) for k in n])
    highs = np.array([float(certify(_diag([0.0], k), ALPHA).q_hi) for k in n])
    assert_allclose(lows, n / (n + 4.0), rtol=1e-12)
    assert_allclose(lows, [0.556, 0.926, 0.9992], atol=5e-4)
    assert_allclose(highs, 4.0 / (n + 4.0), rtol=1e-12)
    outside = certify(_diag([1.0], 5000.0, outside=0.06), ALPHA)
    assert_allclose(outside.q_lo, 0.94 * 5000.0 / 5004.0, rtol=1e-12)
    assert_allclose(outside.q_hi, 1.0, rtol=1e-12)


def test_pareto_khat_recovers_a_known_tail(toy):
    """Weights w = U^-k with U uniform have a Pareto tail of shape k. One estimate at
    S = 5000 (a 213-draw tail) scatters by about 0.09 at k = 0.5 and 0.13 at k = 1,
    so the test averages 64 replicates: N = (4 * 0.13 / 0.1)^2 = 27 replicates puts
    0.1 at 4 sigma; 64 gives the mean a spread of at most 0.016. Both k = 0.5 and
    k = 1.0 (above the 0.7 gate) are recovered within 0.1. Below about 2,200 draws
    the threshold drops: 0.5 at S = 100.

    Fails if the fit's sign convention is flipped, the tail is taken from the wrong
    end, or the exceedances are not over the next-largest weight.
    """
    u = np.random.default_rng(0).uniform(size=(2, 64, N_DRAW))
    for k, u_k in zip((0.5, 1.0), u, strict=True):
        khat = toy.jit.pareto_khat(-k * np.log(u_k))
        assert abs(float(jnp.mean(khat)) - k) < 0.1, (k, jnp.mean(khat))

    assert khat_threshold(100) == pytest.approx(0.5)
    assert khat_threshold(5000) == pytest.approx(0.7)


def test_pareto_khat_without_a_tail(toy):
    """Weights with no tail give -inf: 5000 equal weights, the same with a quarter
    of the draws at zero weight, equal weights off by one ulp in three draws, and two
    weight levels (1,405 and 595 draws, ratio 0.6). Fifty draws at twice the weight of
    1,950 others fit a bounded tail (khat below zero). A fit that cannot be made is
    NaN: three draws above an otherwise flat cloud (fewer than five exceedances), or no
    finite weight. Every case is 5000 draws, a smaller one padded with zero weights,
    and all run as one batch.

    Fails if constant or two-level weights give NaN (a cloud of exact posterior draws
    could then never be certified), if draws tied with the cutoff enter the fit, if
    float noise counts as a tail, or if an impossible fit returns a number.
    """
    flat = np.zeros(N_DRAW)

    def padded(*parts):
        log_w = np.concatenate(parts)
        return np.concatenate([log_w, np.full(N_DRAW - len(log_w), -np.inf)])

    with_ulp = flat.copy()
    with_ulp[:3] = np.finfo(flat.dtype).eps
    three_up = flat.copy()
    three_up[:3] = np.log(2.0)
    khat = toy.jit.pareto_khat(
        np.stack(
            [
                flat,
                padded(np.zeros(3750)),
                with_ulp,
                padded(np.zeros(1405), np.full(595, np.log(0.6))),
                padded(np.full(50, np.log(2.0)), np.zeros(1950)),
                three_up,
                np.full(N_DRAW, -np.inf),
            ]
        )
    )
    assert np.all(np.isneginf(khat[:4]))
    assert float(khat[4]) < 0.0
    assert np.all(np.isnan(khat[5:]))


def test_constant_weights_certify(toy):
    """A unanimous claim on 5000 draws of equal weight: khat is -inf (no tail), so
    the tail gate passes, the certificate is positive and no refit is triggered.

    Fails if a cloud of equal weights (exact posterior draws) cannot be certified.
    """
    b = _belief(toy, np.ones(N_DRAW), np.ones(N_DRAW))
    d = toy.jit.diagnose(b, Claim((toy.InHz(),)), toy.star, PLANET, NO_MASS)
    assert jnp.all(jnp.isneginf(d.khat))
    assert int(toy.jit.certify(d, ALPHA).status) == POSITIVE
    assert not bool(toy.jit.refit_trigger(d, ALPHA))


def test_a_failed_fit_is_unresolved(toy):
    """The diagnostics of the previous test with a NaN khat on the member (a failed
    fit, set directly) are unresolved and trigger a refit.

    Fails if a NaN khat passes the tail gate or does not trigger a refit.
    """
    b = _belief(toy, np.ones(N_DRAW), np.ones(N_DRAW))
    d = toy.jit.diagnose(b, Claim((toy.InHz(),)), toy.star, PLANET, NO_MASS)
    failed = eqx.tree_at(lambda x: x.khat, d, d.khat.at[1].set(jnp.nan))
    assert int(toy.jit.certify(failed, ALPHA).status) == UNRESOLVED
    assert bool(toy.jit.refit_trigger(failed, ALPHA))


def test_conditional_error_and_bound(toy):
    """Unit weights, claim (size, hz) given size. 1000 draws, 200 Earth-sized, 150 of
    them in the HZ: q = 0.75, cond_mass = 0.2 and mc_se is the binomial error on the
    200 conditioning draws, sqrt(0.75 * 0.25 / 200) = 0.0306. With 10 Earth-sized
    draws out of 5000, all in the HZ, q = 1 but the Wilson bound at n = 10 (the Kish
    size of the 10 equal condition-weighted draws) is 10 / 14 = 0.714: unresolved,
    and the interval straddles 0.95 so a refit triggers.

    Weights that correlate with the condition: 500 draws, all in the HZ, 250
    Earth-sized at weight 0.7 and 250 not at weight 0.3. n_eff = 250^2 / (250 x 0.58)
    = 431.0 and cond_mass = 0.7, but the condition-weighted draws are 250 of equal
    weight, so n_cond = 250 (not 431.0 x 0.7 = 301.7) and the bound is 250 / 254.

    With no Earth-sized draw at all the condition has no mass: n_cond is 0, the
    certificate is inapplicable and its bounds are the whole interval, 0 and 1.

    Fails if the conditional mc_se is not centered on q g or not divided by
    P(given), if the Wilson n of a conditional claim is n_eff (5000 would certify on
    10 draws) or n_eff * cond_mass (301.7, more draws than meet the condition), or if
    a condition no draw meets gives a NaN n_cond (and NaN bounds).
    """
    claim = Claim((toy.InSize(), toy.InHz()), given=1)
    diagnose, certify = toy.jit.diagnose, toy.jit.certify

    def belief(n, n_size, n_both, w=None):
        draw = np.arange(n)
        return _belief(toy, draw < n_both, draw < n_size, w)

    d = diagnose(belief(1000, 200, 150), claim, toy.star, PLANET, NO_MASS)
    assert_allclose(d.q, 0.75, rtol=1e-12)
    assert_allclose(d.cond_mass, 0.2, rtol=1e-12)
    assert_allclose(d.mc_se, np.sqrt(0.75 * 0.25 / 200), rtol=1e-10)

    d = diagnose(belief(5000, 10, 10), claim, toy.star, PLANET, NO_MASS)
    assert_allclose(d.n_cond, 10.0, rtol=1e-10)
    cert = certify(d, ALPHA)
    assert int(cert.status) == UNRESOLVED
    assert_allclose(cert.q_lo, 10.0 / 14.0, rtol=1e-12)
    assert bool(toy.jit.refit_trigger(d, ALPHA))

    w = np.where(np.arange(500) < 250, 0.7, 0.3)
    d = diagnose(belief(500, 250, 500, w), claim, toy.star, PLANET, NO_MASS)
    assert_allclose(d.n_eff, 250**2 / (250 * 0.58), rtol=1e-10)
    assert_allclose(d.cond_mass, 0.7, rtol=1e-12)
    assert_allclose(d.n_cond, 250.0, rtol=1e-10)
    assert_allclose(certify(d, ALPHA).q_lo, 250.0 / 254.0, rtol=1e-12)

    d = diagnose(belief(500, 0, 0), claim, toy.star, PLANET, NO_MASS)
    assert_allclose(d.n_cond, 0.0)
    cert = certify(d, ALPHA)
    assert int(cert.status) == INAPPLICABLE
    assert (float(cert.q_lo), float(cert.q_hi)) == (0.0, 1.0)


def test_refit_trigger_reads_each_member(toy):
    """Members at 0.97 and 0.92 on 5000 draws are unresolved, but each member's own
    interval sits on one side of 0.95, so no refit is triggered: more draws cannot
    settle a disagreement between priors. A member at 0.95 on 400 draws straddles
    0.95 and triggers; an inapplicable claim never does.

    Fails if the trigger reads the span over members instead of each member's
    interval, or fires on an inapplicable claim.
    """
    certify, refit_trigger = toy.jit.certify, toy.jit.refit_trigger
    disagree = _diag([0.97, 0.92], 5000)
    assert int(certify(disagree, ALPHA).status) == UNRESOLVED
    assert not bool(refit_trigger(disagree, ALPHA))
    assert bool(refit_trigger(_diag([0.95], 400), ALPHA))
    assert not bool(refit_trigger(_diag([0.95], 400, cond_mass=1e-4), ALPHA))
