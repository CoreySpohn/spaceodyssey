"""Closed-form and constructed-fixture tests of weights, q, chain and count_pmf."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from numpy.testing import assert_allclose, assert_array_max_ulp

from spaceodyssey.belief import Belief, chain, count_pmf, q, weights
from spaceodyssey.claim import Claim, GasAbove
from spaceodyssey.information import log_score

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


@pytest.mark.parametrize(("n_conjuncts", "given"), [(0, 0), (2, 2), (2, 3), (2, -1)])
def test_claim_rejects_a_malformed_condition(toy, n_conjuncts, given):
    """A claim needs a conjunct, and a condition that leaves one conjunct outside it.

    With two conjuncts, `given` = 2, 3 and -1 each index the last column of the joint
    mass (by equality, by the clamp of a static index, by wrap-around), which makes q
    exactly 1.

    Fails if an empty claim or a `given` outside [0, C) is accepted.
    """
    conjuncts = (toy.InHz(), toy.InSize())[:n_conjuncts]
    with pytest.raises(ValueError, match="conjunct"):
        Claim(conjuncts, given=given)


def test_count_pmf_rejects_a_conditional_claim(toy):
    """A conditional claim has no count distribution.

    Fails if count_pmf silently ignores `given`.
    """
    atoms = toy.atoms([[1.0]], [[1.0]], [1.0])
    with pytest.raises(ValueError, match="unconditional"):
        count_pmf(
            toy.belief(atoms), Claim((toy.InHz(), toy.InSize()), given=1), toy.star, 2
        )


POOL_WEIGHTS = jnp.array([[0.5, 0.5], [0.2, 0.8]])


def _pool_belief(toy, log_mass, mark_weights, like=1.0, mark_hz=(1.0, 0.0), draws=True):
    """Two roles over a two-mark pool, with or without one draws' planet.

    With `draws`, two equal draws hold one planet that is claim-true in one of them;
    without, the draws hold no planets (K = 0) and the count is the pool's alone.
    """
    if draws:
        atoms = toy.atoms([[1.0], [0.0]], [[1.0], [1.0]], [0.5, 0.5])
    else:
        atoms = toy.atoms(np.zeros((1, 0)), np.zeros((1, 0)), [1.0])
    pool = toy.pool(mark_hz, mark_weights, log_mass, n_roles=2, like=like)
    return toy.belief(atoms, pool, n_roles=2)


@pytest.mark.parametrize("mass", [2.0, 2e-3])
def test_count_pmf_gradients_reach_the_pool(toy, mass):
    """Derivatives of count_pmf and of its log score with respect to the pool's mass
    and its marks' prior and likelihood match finite differences, forward and reverse.

    At mass 2e-3 the pool rate of role 0 is 1e-3, where the tail bin is 1.7e-10 and the
    log score of the tail count reads it directly.

    Fails if a derivative through the pool rate raises (an integer tangent in the pmf)
    or is wrong, or if the tail bin is formed by a cancellation that its derivative
    inherits.
    """
    claim = Claim((toy.InHz(),))
    like = jnp.array([[1.0, 1.0], [0.3, 0.7]])

    def pmf(log_mass, mark_weights, like):
        b = _pool_belief(toy, log_mass, mark_weights, like)
        return count_pmf(b, claim, toy.star, 3)

    def score(log_mass, mark_weights, like):
        return log_score(pmf(log_mass, mark_weights, like)[0], jnp.asarray(3))

    args = (jnp.log(mass), POOL_WEIGHTS, like)
    tol = {"eps": 1e-4, "atol": 1e-5, "rtol": 1e-5}
    check_grads(pmf, args, order=1, modes=("fwd", "rev"), **tol)
    check_grads(score, args, order=1, modes=("fwd", "rev"), **tol)


@pytest.mark.parametrize("n_max", [1, 3])
def test_count_pmf_derivative_at_a_zero_pool_rate(toy, n_max):
    """With every mark claim-false the pool rate is 0, and the derivative is finite.

    The pool rate of role r is lam_r = 2 sum_j w_rj hz_j, so at hz = 0 the Jacobian of
    the pmf with respect to the marks' claim bits is dP/dlam * 2 w_rj, with dP/dlam =
    (-1, 1, 0, ...) at lam = 0. It also matches a one-sided finite difference (the rate
    cannot go below 0). An empty pool (mass 0, log mass -inf) has zero, finite
    gradients. Both tail forms are covered: n_max = 1 (one tail count) and 3.

    Fails if the derivative is NaN at lam = 0 (a log of lam, or the incomplete gamma
    function's derivative at one tail count), or is not the closed form.
    """
    claim = Claim((toy.InHz(),))

    def pmf(mark_hz):
        b = _pool_belief(toy, jnp.log(2.0), POOL_WEIGHTS, mark_hz=mark_hz, draws=False)
        return count_pmf(b, claim, toy.star, n_max)

    zero = jnp.zeros(2)
    dp_dlam = np.zeros(n_max + 1)
    dp_dlam[:2] = [-1.0, 1.0]
    want = dp_dlam[None, :, None] * 2.0 * np.asarray(POOL_WEIGHTS)[:, None, :]
    assert_allclose(jax.jacfwd(pmf)(zero), want, atol=1e-12)
    assert_allclose(jax.jacrev(pmf)(zero), want, atol=1e-12)
    step = 1e-7
    finite = np.stack([(pmf(step * e) - pmf(zero)) / step for e in jnp.eye(2)], axis=-1)
    assert_allclose(finite, want, atol=1e-6)
    grad_score = jax.grad(lambda h: log_score(pmf(h)[0], jnp.asarray(0)))(zero)
    assert_allclose(grad_score, -2.0 * POOL_WEIGHTS[0], atol=1e-12)

    def empty(log_mass, mark_weights):
        b = _pool_belief(toy, log_mass, mark_weights, draws=False)
        return (count_pmf(b, claim, toy.star, n_max) * jnp.arange(n_max + 1)).sum()

    grads = jax.grad(empty, argnums=(0, 1))(jnp.asarray(-jnp.inf), POOL_WEIGHTS)
    for g in grads:
        assert_allclose(g, 0.0, atol=0.0)


def test_pool_tail_is_accurate_in_float32(toy):
    """The tail bin of a rare pool keeps its float64 value in float32.

    Pool rate 1e-3, n_max = 3: P(N >= 3) = 1.6654e-10. In float32, one minus the head
    rounds to 5.96e-8, a log score too high by 5.9 nats.

    Fails if the tail is formed as one minus the head, or the output leaves the input's
    precision.
    """
    claim = Claim((toy.InHz(),))
    b64 = _pool_belief(toy, jnp.log(2e-3), POOL_WEIGHTS, draws=False)
    b32 = jax.tree.map(
        lambda x: x.astype(jnp.float32) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        b64,
    )
    pmf64 = count_pmf(b64, claim, toy.star, 3)
    pmf32 = count_pmf(b32, claim, toy.star, 3)
    assert pmf32.dtype == jnp.float32
    assert_allclose(pmf64[0, 3], 1.6654e-10, rtol=1e-4)
    assert_allclose(pmf32, pmf64, rtol=1e-5, atol=1e-7)
    assert_allclose(pmf32[0, 3], pmf64[0, 3], rtol=1e-5)
