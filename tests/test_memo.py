"""Numbers from the framework memo, in one place, written as closed-form
expectations.
"""

from itertools import pairwise

import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import ndtri
from numpy.testing import assert_allclose

from spaceodyssey.belief import Belief, chain, count_pmf, q
from spaceodyssey.claim import Claim, GasAbove
from spaceodyssey.information import chain_gain, diagnose, information

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


def test_members_by_ratio_on_a_resample(toy):
    """Three verdict members multiply the look-alike weight by w = 0.6, 1 and 1.6 on a
    resample of 2,000 equal draws. The memo quotes psi = 0.80, 0.70, 0.60 and Kish
    sizes 1,917, 2,000, 1,897. A split of 1,405 Earth-box draws and 595 look-alikes
    reproduces all six (only 1,404 to 1,406 Earth-box draws do): psi = 1405 / (1405 +
    595 w) and Kish = (1405 + 595 w)^2 / (1405 + 595 w^2).

    Fails if q or Kish is read off the wrong row of the role axis, or Kish is
    inverted.
    """
    n_earth, n_look = 1405, 595
    size = jnp.concatenate([jnp.ones(n_earth), jnp.zeros(n_look)])[:, None]
    atoms = toy.atoms(jnp.ones_like(size), size, jnp.ones(n_earth + n_look))
    w = jnp.array([1.0, 0.6, 1.0, 1.6])  # forecast, then the three members
    log_prior = jnp.where(size[:, 0] == 0.0, jnp.log(w)[:, None], 0.0)
    flat = toy.belief(atoms, n_roles=4)
    b = Belief(atoms, log_prior, flat.log_like, flat.pool)
    claim = Claim((toy.InHz(), toy.InSize()))
    d = diagnose(b, claim, toy.star, PLANET, jnp.zeros(4))
    assert_allclose(np.round(d.q[1:], 2), [0.80, 0.70, 0.60])
    assert_allclose(np.round(d.n_eff[1:]), [1917.0, 2000.0, 1897.0])
