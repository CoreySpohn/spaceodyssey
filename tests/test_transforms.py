"""Every public function of `belief` and `information` under `jax.jit`, `jax.vmap` over
a leading world axis, and `jax.grad` where it is differentiable.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from numpy.testing import assert_allclose

from spaceodyssey.belief import Belief, chain, count_pmf, q, role_log_weights, weights
from spaceodyssey.claim import Claim, GasAbove
from spaceodyssey.information import (
    certify,
    chain_gain,
    clip_q,
    diagnose,
    information,
    log_score,
    refit_trigger,
)
from spaceodyssey.roles import AbstractPrior, Roles

N_ROLES, N_DRAWS, N_PLANETS = 3, 6, 2


class Tilt(AbstractPrior):
    """log density `shift * hz` of the first planet: a prior with one traced leaf."""

    shift: jax.Array = eqx.field(converter=jnp.asarray)

    def log_density(self, atoms, star):
        return self.shift * atoms.cloud.samples["hz"][:, 0]


def _worlds(toy):
    """Two worlds of one structure: three roles over six draws of two planets."""
    rng = np.random.default_rng(0)
    worlds = []
    for _ in range(2):
        atoms = toy.atoms(
            rng.uniform(0.0, 1.0, (N_DRAWS, N_PLANETS)),
            rng.uniform(0.0, 1.0, (N_DRAWS, N_PLANETS)),
            rng.uniform(0.2, 1.0, N_DRAWS),
            mean=rng.normal(size=(N_DRAWS, 1)),
            var=rng.uniform(0.5, 2.0, (N_DRAWS, 1)),
            n_gas=1,
        )
        pool = toy.pool(
            rng.uniform(0.0, 1.0, 3),
            rng.uniform(0.2, 1.0, (N_ROLES, 3)),
            rng.normal(),
            n_roles=N_ROLES,
        )
        log_prior = rng.normal(size=(N_ROLES, N_DRAWS))
        log_like = rng.normal(size=(N_ROLES, N_DRAWS))
        worlds.append(Belief(atoms, log_prior, log_like, pool))
    return worlds


def _cases(toy):
    """Per public function: (call, args per world, the argument to differentiate,
    a scalar of the output).

    The differentiated argument is a selector into the args, or None where the output
    is not differentiable. The scalar weights the outputs unevenly, so a derivative
    that sums to zero (the weights of a role) is still checked entry by entry.
    """
    star = toy.star
    claim = Claim((toy.InHz(), GasAbove(0, 0.3)))
    conditional = Claim((GasAbove(0, 0.3), toy.InHz(), toy.InSize()), given=1)
    plain = Claim((toy.InHz(), toy.InSize()))
    b0, b1 = _worlds(toy)
    planets = (jnp.asarray(0), jnp.asarray(1))

    def uneven(x):
        x = jnp.asarray(x)
        return jnp.sum(x * jnp.cos(jnp.arange(x.size).reshape(x.shape)))

    def diag(b, planet):
        return diagnose(b, conditional, star, planet, jnp.full(N_ROLES, 0.01))

    d0, d1 = diag(b0, planets[0]), diag(b1, planets[1])
    pmf = [count_pmf(b, plain, star, 3)[0] for b in (b0, b1)]
    counts = (jnp.asarray(1), jnp.asarray(2))
    roles = Roles(Tilt("toy", 0.0), Tilt("toy", jnp.array([0.5, -0.4])))
    roles1 = Roles(Tilt("toy", 0.2), Tilt("toy", jnp.array([0.1, 0.7])))

    def reweight(roles, b):
        return role_log_weights(b.atoms, roles, b.log_like, star, b.pool)

    return {
        "weights": (weights, [(b0,), (b1,)], lambda a: a[0].log_prior, uneven),
        "q": (
            lambda b, k: q(b, claim, star, k),
            [(b0, planets[0]), (b1, planets[1])],
            lambda a: a[0].log_prior,
            uneven,
        ),
        "chain": (
            lambda b, k: chain(b, conditional, star, k),
            [(b0, planets[0]), (b1, planets[1])],
            lambda a: a[0].atoms.x_mean,
            uneven,
        ),
        "count_pmf": (
            lambda b: count_pmf(b, plain, star, 3),
            [(b0,), (b1,)],
            lambda a: a[0].pool.log_mass,
            uneven,
        ),
        "role_log_weights": (
            reweight,
            [(roles, b0), (roles1, b1)],
            lambda a: a[0].verdict.shift,
            lambda b: uneven(weights(b)),
        ),
        "log_score": (
            log_score,
            [(pmf[0], counts[0]), (pmf[1], counts[1])],
            lambda a: a[0],
            uneven,
        ),
        "information": (
            information,
            [(pmf[0], pmf[1], counts[0]), (pmf[1], pmf[0], counts[1])],
            lambda a: a[1],
            uneven,
        ),
        "chain_gain": (
            lambda before, after, k: chain_gain(before, after, plain, star, k, 1),
            [(b0, b1, planets[0]), (b1, b0, planets[1])],
            lambda a: a[1].log_prior,
            uneven,
        ),
        "diagnose": (
            diag,
            [(b0, planets[0]), (b1, planets[1])],
            lambda a: a[0].log_prior,
            lambda d: uneven(d.q) + uneven(d.mc_se) + uneven(d.n_eff),
        ),
        "clip_q": (
            clip_q,
            [(d0.q, jnp.asarray(0.01)), (d1.q, jnp.asarray(0.02))],
            lambda a: a[0],
            uneven,
        ),
        "certify": (
            lambda d: certify(d, 0.05),
            [(d0,), (d1,)],
            lambda a: a[0].q,
            lambda c: c.q_lo + 2.0 * c.q_hi,
        ),
        "refit_trigger": (lambda d: refit_trigger(d, 0.05), [(d0,), (d1,)], None, None),
    }


CASES = [
    "weights",
    "q",
    "chain",
    "count_pmf",
    "role_log_weights",
    "log_score",
    "information",
    "chain_gain",
    "diagnose",
    "clip_q",
    "certify",
    "refit_trigger",
]


@pytest.fixture(scope="module")
def cases(toy):
    return _cases(toy)


def _close(actual, expected):
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert_allclose(a, e, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", CASES)
def test_public_functions_jit_vmap_and_grad(cases, name):
    """Each public function gives its eager result under `jax.jit` and, for two
    worlds stacked on a leading axis, under `jax.vmap`; where differentiable, its
    derivative matches finite differences forward and reverse.

    The planet index is traced (and differs between the worlds), the count is traced,
    and the stacked verdict prior of `role_log_weights` is mapped inside the vmap.

    Fails if a public function branches in Python on a traced value, needs a concrete
    planet index or count, reads a shape from the wrong axis so that it cannot batch
    over worlds, or has a derivative that raises, is NaN or is wrong.
    """
    fn, (args0, args1), wrt, scalar = cases[name]
    out0, out1 = fn(*args0), fn(*args1)
    _close(jax.jit(fn)(*args0), out0)
    stacked = jax.tree.map(lambda *x: jnp.stack(x), args0, args1)
    expected = jax.tree.map(lambda *x: jnp.stack(x), out0, out1)
    _close(jax.jit(jax.vmap(fn))(*stacked), expected)
    if wrt is None:
        return

    @jax.jit
    def f(x):
        return scalar(fn(*eqx.tree_at(wrt, args0, x)))

    x0 = wrt(args0)
    assert np.all(np.isfinite(jax.grad(f)(x0)))
    check_grads(f, (x0,), order=1, modes=("fwd", "rev"), eps=1e-5, atol=1e-6, rtol=1e-6)
