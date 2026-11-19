"""Closed-form and constructed-fixture tests of weights, q, chain and count_pmf."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from jaxtyping import TypeCheckError
from numpy.testing import assert_allclose, assert_array_max_ulp

from spaceodyssey.belief import Belief, count_pmf
from spaceodyssey.claim import AbstractConjunct, Claim, GasAbove
from spaceodyssey.information import log_score
from spaceodyssey.types import Pool

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
    """Role weights are normalized exp(log_weights + log_prior_ratio + log_like).

    Cloud weight (1, 3), refit prior density (1, 2). Role 0: prior (1, 1), so ratio
    (1, 1/2), likelihood (1, 1) gives (1, 3/2), normalized (0.4, 0.6). Role 1: prior
    (3, 1), ratio (3, 1/2), likelihood (1, 2) gives (3, 3 * 2 / 2) = (3, 3), normalized
    (0.5, 0.5).

    Fails if a term enters with the wrong sign, is dropped, or a row is not
    normalized.
    """
    atoms = toy.atoms([[1.0], [0.0]], [[1.0], [1.0]], [1.0, 3.0])
    b = toy.belief(atoms, n_roles=2)
    log_ratio = jnp.log(jnp.array([[1.0, 1.0], [3.0, 1.0]]) / jnp.array([1.0, 2.0]))
    log_like = jnp.log(jnp.array([[1.0, 1.0], [1.0, 2.0]]))
    b = Belief(atoms, log_ratio, log_like, b.pool)
    assert_allclose(toy.jit.weights(b), [[0.4, 0.6], [0.5, 0.5]], rtol=1e-12)


def test_chain_multiplies_to_q_and_given_splits_it(toy):
    """The chain multiplies to q; with `given`, the condition entries multiply to
    P(given).

    The expected P(given) is the weighted share of draws satisfying the first conjuncts,
    computed here directly from the draws; q is the joint over that share.

    A condition that no draw meets has zero mass; q and chain are then 0, not NaN.

    Fails if q and chain are computed from different masses, `given` does not split the
    chain, or a zero condition mass is divided by.
    """
    b, w = _random_belief(toy)
    conj = (toy.InHz(), toy.InSize(), GasAbove(0, 0.0), GasAbove(1, 0.0))
    plain = Claim(conj)
    probs = np.stack([np.asarray(c.prob(b.atoms, toy.star))[:, 0] for c in conj])
    q_plain = toy.jit.q(b, plain, toy.star, PLANET)
    assert_allclose(q_plain, [(w * probs.prod(0)).sum()], rtol=1e-12)
    assert_array_max_ulp(
        np.asarray(toy.jit.chain(b, plain, toy.star, PLANET).prod(axis=1)),
        np.asarray(q_plain),
        maxulp=4,
    )
    for given in (1, 2, 3):
        claim = Claim(conj, given=given)
        c = np.asarray(toy.jit.chain(b, claim, toy.star, PLANET))[0]
        p_given = (w * probs[:given].prod(0)).sum()
        assert_allclose(c[:given].prod(), p_given, rtol=1e-12)
        assert_allclose(
            c[given:].prod(), toy.jit.q(b, claim, toy.star, PLANET)[0], rtol=1e-12
        )
        assert_allclose(
            toy.jit.q(b, claim, toy.star, PLANET)[0], q_plain[0] / p_given, rtol=1e-12
        )
    never = toy.belief(toy.atoms([[0.0], [0.0]], [[1.0], [1.0]], [0.5, 0.5]))
    claim = Claim((toy.InHz(), toy.InSize()), given=1)
    assert_allclose(toy.jit.q(never, claim, toy.star, PLANET), [0.0], atol=0.0)
    assert_allclose(
        toy.jit.chain(never, claim, toy.star, PLANET), [[0.0, 0.0]], atol=0.0
    )


def test_pool_count_pmf_by_hand(toy):
    """A pool with no draws' planets: the count is Poisson with mean mass * pbar per
    role.

    Role 0 mark weights (0.5, 0.5), role 1 weights (0.2, 0.8), claim probabilities (1,
    0), mass 2. Then lambda_0 = 2 * 0.5 = 1.0 and lambda_1 = 2 * 0.2 = 0.4. For role 0
    with n_max = 3: P(0) = e^-1 = 0.3679, P(1) = 0.3679, P(2) = 0.1839, P(3 or more) = 1
    - 0.9197 = 0.0803. Role 1 has P(0) = e^-0.4 = 0.6703. Each row sums to one.

    Every term of the mark weights is live: cloud weight (2, 1) over reference density
    (1, 2) is (2, 0.5); prior (0.5, 2) in both roles makes it (1, 1); likelihood (1, 1)
    in role 0 and (0.2, 0.8) in role 1 gives the weights above.

    Fails if the pool uses the wrong role's mark weights, forgets the claim probability
    of a mark, drops the marks' cloud weight, reference density or likelihood, or does
    not lump the tail.
    """
    atoms = toy.atoms(
        np.zeros((1, 0)), np.zeros((1, 0)), [1.0]
    )  # K = 0: no draws' planets
    pool = toy.pool(
        [1.0, 0.0],
        [[0.5, 2.0], [0.5, 2.0]],
        np.log(2.0),
        n_roles=2,
        like=[[1.0, 1.0], [0.2, 0.8]],
        cloud=[2.0, 1.0],
        ref=[1.0, 2.0],
    )
    pmf = toy.jit.count_pmf(
        toy.belief(atoms, pool, n_roles=2), Claim((toy.InHz(),)), toy.star, 3
    )
    e1 = np.exp(-1.0)
    assert_allclose(pmf[0], [e1, e1, e1 / 2, 1 - 2.5 * e1], rtol=1e-12)
    assert_allclose(pmf[0, 3], 0.0803, atol=1e-4)
    e04 = np.exp(-0.4)
    assert_allclose(pmf[1], [e04, 0.4 * e04, 0.08 * e04, 1 - 1.48 * e04], rtol=1e-12)
    assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-12)


def test_pool_rate_now_and_at_time_zero(toy):
    """The rate now and at time zero, from the thinned pool and the pool at time zero.

    The pool at time zero is its own value. Mass 2, two marks of weights (0.5, 0.5)
    and claim probabilities (1, 0), detection probabilities (0.8, 0.2). Expected
    claim-true planets found by the observation: 2 * (0.5 * 1 * 0.8 + 0.5 * 0 * 0.2) =
    0.8. After a null the mass is 2 * (1 - 0.5) = 1.0 and the marks reweight by (0.2,
    0.8), to weights (0.2, 0.8), so the claim-true rate is 0.2, against 2 * 0.5 = 1.0
    at time zero; the difference is the 0.8 found. The count after the null is
    Poisson(0.2).

    Fails if the rate drops the thinned mass or the marks' null likelihood.
    """
    atoms = toy.atoms(np.zeros((1, 0)), np.zeros((1, 0)), [1.0])
    pool0 = toy.pool([1.0, 0.0], [[0.5, 0.5]], np.log(2.0))
    thinned = toy.pool([1.0, 0.0], [[0.5, 0.5]], np.log(1.0), like=[[0.2, 0.8]])
    claim = Claim((toy.InHz(),))
    now = toy.jit.pool_rate(thinned, claim, toy.star)
    then = toy.jit.pool_rate(pool0, claim, toy.star)
    assert_allclose(now, [0.2], rtol=1e-12)
    assert_allclose(then, [1.0], rtol=1e-12)
    assert_allclose(then - now, [0.8], rtol=1e-12)
    pmf = toy.jit.count_pmf(toy.belief(atoms, thinned), claim, toy.star, 1)
    assert_allclose(pmf, [[np.exp(-0.2), 1 - np.exp(-0.2)]], rtol=1e-12)


def test_count_pmf_convolves_draws_and_pool(toy):
    """One certain planet in every draw plus a pool of mean 1 shifts the pool's pmf up
    by one; more planets than bins stay in the last bin.

    The draws contribute exactly one claim-true planet, so the count is 1 + Poisson(1):
    with n_max = 3, P = (0, e^-1, e^-1, 1 - 2 e^-1). A draw of three planets, each
    claim-true with probability 0.5, and n_max = 1 (K > n_max) gives (0.125, 0.875).

    Fails if the draws' count and the pool's count are combined by addition of the pmfs
    instead of convolution, if the convolved tail is truncated instead of lumped, or if
    mass at the cap leaks while the draws' planets are counted.
    """
    atoms = toy.atoms([[1.0]], [[1.0]], [1.0])
    pool = toy.pool([1.0], [[1.0]], 0.0)
    pmf = toy.jit.count_pmf(toy.belief(atoms, pool), Claim((toy.InHz(),)), toy.star, 3)
    e1 = np.exp(-1.0)
    assert_allclose(pmf[0], [0.0, e1, e1, 1 - 2 * e1], atol=1e-12)
    three = toy.atoms([[0.5, 0.5, 0.5]], [[1.0, 1.0, 1.0]], [1.0])
    pmf = toy.jit.count_pmf(toy.belief(three), Claim((toy.InHz(),)), toy.star, 1)
    assert_allclose(pmf[0], [0.125, 0.875], atol=1e-12)


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


def test_a_nonexistent_slot_changes_nothing(toy):
    """Padding draws with planet slots that do not exist, and adding a zero-weight
    draw that holds no planet, leaves q, chain and count_pmf unchanged.

    The padded slot (planet 1) is claim-true by its bits, so counting it would show;
    its gas state and the padding draw's whole state are NaN, which the values must
    never read. The padded slot's own q is 0.

    Fails if a slot that does not exist is counted or read, in the draws' planets or
    in the marks, or a zero-weight padding draw passes a NaN into a probability.
    """
    b, _ = _random_belief(toy)
    pool = toy.pool([1.0, 0.0], [[0.5, 0.5]], np.log(2.0))
    b = toy.belief(b.atoms, pool)
    claim = Claim((toy.InHz(), toy.InSize(), GasAbove(0, 0.0)))
    plain = Claim((toy.InHz(), toy.InSize()))
    given = Claim((toy.InHz(), GasAbove(0, 0.0)), given=1)

    a = b.atoms
    n = a.cloud.log_weights.shape[0]

    def pad(x, fill):
        x = jnp.concatenate([x, jnp.full_like(x, fill)], axis=1)  # a second slot
        return jnp.concatenate([x, jnp.full_like(x[:1], jnp.nan)], axis=0)  # a draw

    samples = {k: pad(v, 1.0) for k, v in a.cloud.samples.items()}
    exists = jnp.concatenate([jnp.ones((n, 1), bool), jnp.zeros((n, 1), bool)], 1)
    exists = jnp.concatenate([exists, jnp.zeros((1, 2), bool)], 0)
    padded = toy.atoms(
        samples["hz"], samples["size"], [*np.exp(a.cloud.log_weights), 0]
    )
    padded = eqx.tree_at(
        lambda p: (p.x_mean, p.x_var, p.exists),
        padded,
        (pad(a.x_mean, jnp.nan), pad(a.x_var, 1.0), exists),
    )
    marks = eqx.tree_at(
        lambda m: (m.cloud.samples["hz"], m.exists),
        pool.marks,
        (jnp.ones((2, 1)), jnp.array([[True], [False]])),
    )
    b_pad = toy.belief(padded, eqx.tree_at(lambda p: p.marks, pool, marks))

    for c in (claim, given):
        assert_allclose(
            toy.jit.q(b_pad, c, toy.star, PLANET), toy.jit.q(b, c, toy.star, PLANET)
        )
        assert_allclose(
            toy.jit.chain(b_pad, c, toy.star, PLANET),
            toy.jit.chain(b, c, toy.star, PLANET),
        )
        assert_allclose(toy.jit.q(b_pad, c, toy.star, jnp.asarray(1)), [0.0], atol=0.0)
    assert_allclose(
        toy.jit.count_pmf(b_pad, plain, toy.star, 3),
        toy.jit.count_pmf(b, plain, toy.star, 3),
    )


def test_pool_and_belief_reject_mismatched_shapes(toy):
    """A pool whose marks hold two planets, or whose fields disagree on the number of
    roles, and a belief whose roles disagree with its pool's, are refused when built.

    Fails if a pool with K > 1 marks is accepted (its first slot would be read
    silently) or a role-count mismatch surfaces only later, inside a vmap.
    """
    pool = toy.pool([1.0, 0.0], [[0.5, 0.5]], 0.0)
    fields = [pool.log_prior_ratio, pool.log_like, pool.log_mass]
    two_planets = toy.atoms([[1.0, 1.0]] * 2, [[0.0, 0.0]] * 2, [1.0, 1.0])
    with pytest.raises(ValueError, match="K = 1"):
        Pool(two_planets, *fields)
    with pytest.raises(ValueError, match="disagree"):
        Pool(pool.marks, pool.log_prior_ratio, jnp.zeros((2, 2)), pool.log_mass)
    atoms = toy.atoms([[1.0]], [[1.0]], [1.0])
    with pytest.raises(ValueError, match="disagree"):
        Belief(atoms, jnp.zeros((2, 1)), jnp.zeros((2, 1)), pool)


def test_a_conjunct_without_prob_cannot_be_built():
    """`prob` is abstract, so a conjunct that does not define it fails when built.

    Fails if such a conjunct instantiates and fails only at its first use.
    """

    class Bare(AbstractConjunct):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Bare()


def test_annotations_are_checked_and_boolean_conjuncts_count(toy, typecheck):
    """The jaxtyping hook is on in the tests, and a conjunct that returns booleans is
    counted as 0 and 1.

    A float count passed to `log_score` (annotated as an integer) raises. A boolean
    `hz` conjunct over one claim-true planet gives the count distribution (0, 1).

    Fails if the hook is not installed (so annotations go unchecked), or a boolean
    conjunct breaks the count's scan with a dtype error.
    """
    if typecheck:
        with pytest.raises(TypeCheckError):
            log_score(jnp.ones(3) / 3, jnp.asarray(1.5))

    class HzBit(AbstractConjunct):
        def prob(self, atoms, star):
            return atoms.cloud.samples["hz"] > 0.5

    atoms = toy.atoms([[1.0]], [[1.0]], [1.0])
    pmf = toy.jit.count_pmf(toy.belief(atoms), Claim((HzBit(),)), toy.star, 1)
    assert pmf.dtype == atoms.x_mean.dtype
    assert_allclose(pmf, [[0.0, 1.0]], atol=0.0)


def test_count_pmf_rejects_a_conditional_claim(toy):
    """A conditional claim has no count distribution.

    Fails if count_pmf silently ignores `given`.
    """
    atoms = toy.atoms([[1.0]], [[1.0]], [1.0])
    with pytest.raises(ValueError, match="unconditional"):
        toy.jit.count_pmf(
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

    @jax.jit
    def pmf(log_mass, mark_weights, like):
        b = _pool_belief(toy, log_mass, mark_weights, like)
        return count_pmf(b, claim, toy.star, 3)

    @jax.jit
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

    @jax.jit
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

    @jax.jit
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
    pmf64 = toy.jit.count_pmf(b64, claim, toy.star, 3)
    pmf32 = toy.jit.count_pmf(b32, claim, toy.star, 3)
    assert pmf32.dtype == jnp.float32
    assert_allclose(pmf64[0, 3], 1.6654e-10, rtol=1e-4)
    assert_allclose(pmf32, pmf64, rtol=1e-5, atol=1e-7)
    assert_allclose(pmf32[0, 3], pmf64[0, 3], rtol=1e-5)
