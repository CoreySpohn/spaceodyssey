"""Numbers from the framework memo, in one place, written as closed-form
expectations.
"""

from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import ndtri
from numpy.testing import assert_allclose

from spaceodyssey.belief import chain, count_pmf, q
from spaceodyssey.claim import Claim, GasAbove
from spaceodyssey.information import chain_gain, information
from spaceodyssey.toy import GAS_FLOOR, GAS_PRIOR_SD, GaussianGasLaw

PLANET = jnp.asarray(0)


@pytest.fixture(scope="module")
def stages(toy):
    """The four beliefs of the E track, all for one planet and two gases (water,
    oxygen).
    """
    hz4, size4 = [[0.0], [0.0], [1.0], [1.0]], [[0.0], [1.0], [0.0], [1.0]]
    s0 = toy.atoms(hz4, size4, [0.25] * 4)
    # Orbit resolved: hz = 1 in both draws; the size bit splits 0.7 / 0.3.
    kw = {"hz": [[1.0], [1.0]], "size": [[1.0], [0.0]], "w": [0.7, 0.3]}
    s1 = toy.atoms(**kw)
    # A detected gas is a point state above the floor: x_var = 0, x_mean > 0.
    s2 = toy.atoms(**kw, mean=[1.0, 0.0], var=[0.0, 1.0])
    s3 = toy.atoms(**kw, mean=[1.0, 1.0], var=[0.0, 0.0])
    return [toy.belief(a) for a in (s0, s1, s2, s3)]


@pytest.fixture(scope="module")
def claim(toy):
    floor = 0.0
    return Claim((toy.InHz(), toy.InSize(), GasAbove(0, floor), GasAbove(1, floor)))


def test_e_stage_values(stages, claim, toy):
    """The E track's q at the four stages, and the chain at each.

    Stage 0: 0.25 * 0.5 * 0.5 = 0.0625 (chain 0.5, 0.5, 0.5, 0.5). Stage 1: 0.7 * 0.25 =
    0.175 (chain 1, 0.7, 0.5, 0.5). Stage 2 (water detected): 0.7 * 0.5 = 0.35. Stage 3
    (oxygen detected): 0.70.

    Fails if q or chain misreads a weight, a gas probability or a conjunct order.
    """
    expected_q = [0.0625, 0.175, 0.35, 0.70]
    expected_chain = [[0.5] * 4, [1, 0.7, 0.5, 0.5], [1, 0.7, 1, 0.5], [1, 0.7, 1, 1]]
    for b, want_q, want_chain in zip(stages, expected_q, expected_chain, strict=True):
        assert_allclose(q(b, claim, toy.star, PLANET), [want_q], rtol=1e-12)
        assert_allclose(chain(b, claim, toy.star, PLANET), [want_chain], rtol=1e-12)


def test_e_and_l_information(stages, claim, toy):
    """E gains ln(0.70 / 0.0625) = ln 11.2 = 2.416; L loses |ln(0.30 / 0.9375)| = 1.139.

    E is the claim-true test planet (count 1, score ln q); L is the look-alike (count 0,
    score ln(1 - q)); both are scored on the count distribution of one planet.

    Fails if the count distribution of one planet is not [1 - q, q] or the score or its
    sign is wrong.
    """
    pmf = [count_pmf(b, claim, toy.star, 1)[0] for b in stages]
    assert_allclose(pmf[0], [0.9375, 0.0625], rtol=1e-12)
    gain_e = information(pmf[0], pmf[3], jnp.asarray(1))
    gain_l = information(pmf[0], pmf[3], jnp.asarray(0))
    assert_allclose(gain_e, np.log(0.70 / 0.0625), rtol=1e-12)
    assert_allclose(gain_l, np.log(0.30 / 0.9375), rtol=1e-12)
    assert round(float(gain_e), 2) == 2.42
    assert round(float(gain_l), 2) == -1.14


def test_chain_gain_split(stages, claim, toy):
    """The three transitions split E's information over the conjuncts.

    Orbit: [ln 2, ln 1.4, 0, 0]. Water: [0, 0, ln 2, 0]. Oxygen: [0, 0, 0, ln 2]. They
    sum to [0.693, 0.336, 0.693, 0.693] (0.69 + 0.34 + 0.69 + 0.69) and to 2.416 in
    total.

    Fails if the split is taken against the wrong stage or does not sum to the total.
    """
    ln2, ln14 = np.log(2.0), np.log(1.4)
    gains = [chain_gain(a, b, claim, toy.star, PLANET, 0) for a, b in pairwise(stages)]
    assert_allclose(gains[0], [ln2, ln14, 0, 0], atol=1e-12)
    assert_allclose(gains[1], [0, 0, ln2, 0], atol=1e-12)
    assert_allclose(gains[2], [0, 0, 0, ln2], atol=1e-12)
    total = sum(gains)
    assert_allclose(total, [ln2, ln14, ln2, ln2], atol=1e-12)
    assert_allclose(total.sum(), np.log(0.70 / 0.0625), rtol=1e-12)


def test_joint_versus_product(toy):
    """A verdict member: water probability 0.5 on Earth-sized draws, 0.8 on others.

    Two draws, weights 0.7 (size 1) and 0.3 (size 0), oxygen ignored. Jointly (size,
    water): q = 0.7 * 0.5 = 0.350. With water as the condition: P(water) = 0.7 * 0.5 +
    0.3 * 0.8 = 0.59 and P(size | water) = 0.35 / 0.59 = 0.593. The product of marginals
    0.70 * 0.59 = 0.413 is not q.

    Fails if a conditional claim returns the product of marginals, or the joint is not
    read.
    """
    # With unit variance and floor 0, x_mean = 0 gives P = 0.5; ndtri(0.8) gives 0.8.
    water = jnp.array([0.0, float(ndtri(0.8))])
    atoms = toy.atoms(
        [[1.0], [1.0]], [[1.0], [0.0]], [0.7, 0.3], mean=water[:, None], n_gas=1
    )
    b = toy.belief(atoms)
    joint = Claim((toy.InSize(), GasAbove(0, 0.0)), given=0)
    given = Claim((GasAbove(0, 0.0), toy.InSize()), given=1)
    assert_allclose(q(b, joint, toy.star, PLANET), [0.35], rtol=1e-12)
    p_water = 0.7 * 0.5 + 0.3 * 0.8
    assert_allclose(chain(b, given, toy.star, PLANET)[0, 0], p_water, rtol=1e-12)
    assert_allclose(q(b, given, toy.star, PLANET), [0.35 / p_water], rtol=1e-12)
    assert (
        abs(0.70 * p_water - 0.35 / p_water) > 0.1
    )  # the product of marginals is 0.413


def test_two_planet_count(toy):
    """K = 2, two draws of weight 0.5, planet 0 claim-true in one and planet 1 in the
    other.

    Each draw holds exactly one claim-true planet, so P(N = 1) = 1 and the log score is
    0. Two independent track bits would score 2 ln 0.5 = -1.39 and the count from
    independent bits ln 0.5 = -0.69.

    Fails if the count is taken per track, or the draws' planets are treated as
    independent across draws.
    """
    atoms = toy.atoms([[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]], [0.5, 0.5])
    claim = Claim((toy.InHz(), toy.InSize()))
    pmf = count_pmf(toy.belief(atoms), claim, toy.star, 2)[0]
    assert_allclose(pmf, [0.0, 1.0, 0.0], atol=1e-12)
    assert_allclose(jnp.log(pmf[1]), 0.0, atol=1e-12)
    # Test-only comparisons.
    assert_allclose(2 * np.log(0.5), -1.386, atol=1e-3)
    assert_allclose(np.log(0.5), -0.693, atol=1e-3)


# The gas law at the toy constants: floor 0 dex, the verdict's gas prior N(0, 1) dex,
# sigma = 2.5 / (S/N) dex. The numbers below hold for these constants only.
GAS_LAW = GaussianGasLaw(snr_one_day=[1.0])


def _expected_row(snr, u):
    """Payload (1, G, 2) of one planet whose readings sit at their expected value."""
    snr = jnp.asarray(snr, float)
    return jnp.stack([snr, jnp.where(snr > 0, u, 0.0)], axis=-1)[None]


def _p_gas(toy, snr, y):
    """P(gas above its floor) after one reading y at S/N `snr`, from the prior."""
    prior = toy.atoms([[1.0]], [[1.0]], [1.0], GAS_FLOOR, GAS_PRIOR_SD**2, n_gas=1)
    posterior = GAS_LAW.update(_expected_row([snr], y), prior)
    return GasAbove(0, GAS_FLOOR).prob(posterior, toy.star)[0, 0]


def test_gas_law_numbers_at_snr_5(stages, claim, toy):
    """The E track with the Gaussian gas law at S/N 5, the datum at y = u = 1 dex.

    sigma = 0.5, gain 1 / (1 + 0.25) = 0.8, posterior 0.80 +- 0.447 dex, P(gas) =
    Phi(0.8 / 0.447) = 0.963. Stage values q = 0.0625, 0.175, 0.337, 0.649; E's
    information +1.03, +0.66, +0.66 (total 2.34) and L's -0.13, -0.22, -0.64 (total
    -0.98). P(gas) = 0.76, 0.96, 1.00 at S/N 2.5, 5, 10. A gas 0.2 dex above its floor:
    0.64 at S/N 5 and 0.95 at S/N 20.7.

    Fails if the posterior gain, mean or width is wrong (for example a gain of
    1 / (1 + sigma)), or the update is applied to the wrong gas.
    """
    water, oxygen = _expected_row([5.0, 0.0], 1.0), _expected_row([0.0, 5.0], 1.0)
    after_water = GAS_LAW.update(water, stages[1].atoms)
    after_oxygen = GAS_LAW.update(oxygen, after_water)
    assert_allclose(after_water.x_mean[:, 0, 0], 0.8, rtol=1e-12)
    assert_allclose(np.sqrt(after_water.x_var[:, 0, 0]), 0.447, atol=5e-4)
    assert_allclose(after_water.x_mean[:, 0, 1], 0.0)

    beliefs = [stages[0], stages[1], toy.belief(after_water), toy.belief(after_oxygen)]
    values = [q(b, claim, toy.star, PLANET)[0] for b in beliefs]
    assert_allclose(values, [0.0625, 0.175, 0.337, 0.649], atol=5e-4)

    pmf = [count_pmf(b, claim, toy.star, 1)[0] for b in beliefs]
    gains_e = [information(a, b, jnp.asarray(1)) for a, b in pairwise(pmf)]
    gains_l = [information(a, b, jnp.asarray(0)) for a, b in pairwise(pmf)]
    assert_allclose(gains_e, [1.03, 0.66, 0.66], atol=5e-3)
    assert_allclose(gains_l, [-0.13, -0.22, -0.64], atol=5e-3)
    assert round(float(sum(gains_e)), 2) == 2.34
    assert round(float(sum(gains_l)), 2) == -0.98

    p = [_p_gas(toy, s, 1.0) for s in (2.5, 5.0, 10.0)]
    assert_allclose(p, [0.76, 0.963, 1.00], atol=5e-3)
    assert_allclose(_p_gas(toy, 5.0, 0.2), 0.64, atol=5e-3)
    assert_allclose(_p_gas(toy, 20.7, 0.2), 0.95, atol=1e-3)


def _pass_rate(toy, snr, u, n=2000):
    """Share of noise draws whose row passes P(gas 1) P(gas 2) >= 0.95.

    Each gas gets independent noise eps ~ N(0, 1) and reading y = u + sigma eps. The
    draws are the n midpoint quantiles of N(0, 1) per gas, crossed, so the share is a
    deterministic midpoint rule with error of order 1 / n.
    """
    eps = ndtri((jnp.arange(n) + 0.5) / n)
    sigma = 2.5 / snr
    p = jax.vmap(lambda y: _p_gas(toy, snr, y))(u + sigma * eps)
    return float(jnp.mean(p[:, None] * p[None, :] >= 0.95))


def test_requirement_table(toy):
    """The S/N at which a row passes the both-gas rule, for the toy constants only.

    Pass rule: P(gas 1) P(gas 2) >= 0.95, the conditional certificate at alpha = 0.05.
    The expected row (y = u) passes at S/N 5.4, 10.1, 24.6 for u = 1, 0.5, 0.2 dex
    (P(both) 0.9506, 0.9507, 0.9504; S_f = u / sigma = 2.16, 2.02, 1.97) and fails 0.1
    below each. The 90-percent rule is met at S/N 8.5, 16.6, 41 (pass rates 0.903,
    0.898, 0.892; exact 0.9030, 0.8976, 0.8925). At S/N 5 and u = 1 the pass rate is
    0.266 (exact 0.2654), and about 0.80 at S_f = 3 (S/N 7.5, exact 0.786).

    Fails if the posterior gain or width is mis-scaled, which moves the S/N at which the
    expected row passes and the share of noise draws that pass.
    """
    for u, snr, both in [(1.0, 5.4, 0.9506), (0.5, 10.1, 0.9507), (0.2, 24.6, 0.9504)]:
        assert_allclose(_p_gas(toy, snr, u) ** 2, both, atol=1e-4)
        assert _p_gas(toy, snr - 0.1, u) ** 2 < 0.95
    for u, snr, rate in [(1.0, 8.5, 0.903), (0.5, 16.6, 0.898), (0.2, 41.0, 0.892)]:
        assert_allclose(_pass_rate(toy, snr, u), rate, atol=1e-3)
    assert_allclose(_pass_rate(toy, 5.0, 1.0), 0.266, atol=1e-3)
    assert_allclose(_pass_rate(toy, 7.5, 1.0), 0.80, atol=0.02)
