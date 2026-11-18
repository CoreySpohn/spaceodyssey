"""Closed-form and constructed-fixture tests of weights, q, chain and count_pmf."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_max_ulp

from spaceodyssey.belief import Belief, chain, count_pmf, q, weights
from spaceodyssey.claim import Claim, GasAbove

PLANET = jnp.asarray(0)


def _random_belief(toy):
    """Eight draws with arbitrary weights, bits and gas probabilities (seed 0)."""
    rng = np.random.default_rng(0)
    w = rng.uniform(0.1, 1.0, 8)
    atoms = toy.atoms(
        rng.integers(0, 2, (8, 1)),
        rng.integers(0, 2, (8, 1)),
        w / w.sum(),
        mean=rng.normal(size=(8, 2)),
        var=rng.uniform(0.2, 2.0, (8, 2)),
    )
    return toy.belief(atoms), w / w.sum()


def test_weights_normalize_per_role(toy):
    """Role weights are the normalized exp(log_weight + log_prior - log_ref + log_like).

    Cloud weight (1, 3), reference density (1, 2). Role 0: prior (1, 1), likelihood (1,
    1) gives (1/1, 3/2) = (1, 1.5), normalized (0.4, 0.6). Role 1: prior (3, 1),
    likelihood (1, 2) gives (3, 3 * 2 / 2) = (3, 3), normalized (0.5, 0.5).

    Fails if a term enters with the wrong sign (the reference density divides), is
    dropped, or a row is not normalized.
    """
    atoms = toy.atoms([[1.0], [0.0]], [[1.0], [1.0]], [1.0, 3.0])
    atoms = eqx.tree_at(
        lambda a: a.cloud.log_ref_density, atoms, jnp.log(jnp.array([1.0, 2.0]))
    )
    b = toy.belief(atoms, n_roles=2)
    log_prior = jnp.log(jnp.array([[1.0, 1.0], [3.0, 1.0]]))
    log_like = jnp.log(jnp.array([[1.0, 1.0], [1.0, 2.0]]))
    b = Belief(atoms, log_prior, log_like, b.pool)
    assert_allclose(weights(b), [[0.4, 0.6], [0.5, 0.5]], rtol=1e-12)


def test_chain_multiplies_to_q_and_given_splits_it(toy):
    """The chain multiplies to q; with `given`, the condition entries multiply to
    P(given).

    The expected P(given) is the weighted share of draws satisfying the first conjuncts,
    computed here directly from the draws; q is the joint over that share.

    Fails if q and chain are computed from different masses, or `given` does not split
    the chain.
    """
    b, w = _random_belief(toy)
    conj = (toy.InHz(), toy.InSize(), GasAbove(0, 0.0), GasAbove(1, 0.0))
    plain = Claim(conj)
    probs = np.stack([np.asarray(c.prob(b.atoms, toy.star))[:, 0] for c in conj])
    q_plain = q(b, plain, toy.star, PLANET)
    assert_allclose(q_plain, [(w * probs.prod(0)).sum()], rtol=1e-12)
    assert_array_max_ulp(
        np.asarray(chain(b, plain, toy.star, PLANET).prod(axis=1)),
        np.asarray(q_plain),
        maxulp=4,
    )
    for given in (1, 2, 3):
        claim = Claim(conj, given=given)
        c = np.asarray(chain(b, claim, toy.star, PLANET))[0]
        p_given = (w * probs[:given].prod(0)).sum()
        assert_allclose(c[:given].prod(), p_given, rtol=1e-12)
        assert_allclose(c[given:].prod(), q(b, claim, toy.star, PLANET)[0], rtol=1e-12)
        assert_allclose(
            q(b, claim, toy.star, PLANET)[0], q_plain[0] / p_given, rtol=1e-12
        )


def test_pool_count_pmf_by_hand(toy):
    """A pool with no draws' planets: the count is Poisson with mean mass * pbar per
    role.

    Role 0 mark weights (0.5, 0.5), role 1 weights (0.2, 0.8), claim probabilities (1,
    0), mass 2. Then lambda_0 = 2 * 0.5 = 1.0 and lambda_1 = 2 * 0.2 = 0.4. For role 0
    with n_max = 3: P(0) = e^-1 = 0.3679, P(1) = 0.3679, P(2) = 0.1839, P(3 or more) = 1
    - 0.9197 = 0.0803. Role 1 has P(0) = e^-0.4 = 0.6703. Each row sums to one.

    Fails if the pool uses the wrong role's mark weights, forgets the claim probability
    of a mark, or does not lump the tail.
    """
    atoms = toy.atoms(
        np.zeros((1, 0)), np.zeros((1, 0)), [1.0]
    )  # K = 0: no draws' planets
    pool = toy.pool([1.0, 0.0], [[0.5, 0.5], [0.2, 0.8]], np.log(2.0), n_roles=2)
    pmf = count_pmf(
        toy.belief(atoms, pool, n_roles=2), Claim((toy.InHz(),)), toy.star, 3
    )
    e1 = np.exp(-1.0)
    assert_allclose(pmf[0], [e1, e1, e1 / 2, 1 - 2.5 * e1], rtol=1e-12)
    assert_allclose(pmf[0, 3], 0.0803, atol=1e-4)
    e04 = np.exp(-0.4)
    assert_allclose(pmf[1], [e04, 0.4 * e04, 0.08 * e04, 1 - 1.48 * e04], rtol=1e-12)
    assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-12)


def test_count_pmf_convolves_draws_and_pool(toy):
    """One certain planet in every draw plus a pool of mean 1 shifts the pool's pmf up
    by one.

    The draws contribute exactly one claim-true planet, so the count is 1 + Poisson(1):
    with n_max = 3, P = (0, e^-1, e^-1, 1 - 2 e^-1).

    Fails if the draws' count and the pool's count are combined by addition of the pmfs
    instead of convolution, or the lump is applied before the convolution.
    """
    atoms = toy.atoms([[1.0]], [[1.0]], [1.0])
    pool = toy.pool([1.0], [[1.0]], 0.0)
    pmf = count_pmf(toy.belief(atoms, pool), Claim((toy.InHz(),)), toy.star, 3)
    e1 = np.exp(-1.0)
    assert_allclose(pmf[0], [0.0, e1, e1, 1 - 2 * e1], atol=1e-12)


def test_count_pmf_rejects_a_conditional_claim(toy):
    """A conditional claim has no count distribution.

    Fails if count_pmf silently ignores `given`.
    """
    atoms = toy.atoms([[1.0]], [[1.0]], [1.0])
    with pytest.raises(ValueError, match="unconditional"):
        count_pmf(
            toy.belief(atoms), Claim((toy.InHz(), toy.InSize()), given=1), toy.star, 2
        )
