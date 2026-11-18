"""Tests of the sealed world: row keys, grants, unsealing and drawing worlds."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_max_ulp

from spaceodyssey.belief import count_pmf, q
from spaceodyssey.claim import Claim, GasAbove
from spaceodyssey.toy import GAS_FLOOR, GaussianGasLaw
from spaceodyssey.types import Action, Grant, Row
from spaceodyssey.world import SealedWorld, draw_worlds, row_key

KEY = jax.random.key(0)


def _stack(trees):
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *trees)


def _truth(toy, abundances):
    """A truth of one planet per star; `abundances` has shape (S, G)."""
    stars = [
        toy.atoms([[1.0]], [[1.0]], [1.0], jnp.asarray(x)[None], 0.0, len(x))
        for x in abundances
    ]
    return _stack(stars)


def _observe(world, law, star, index):
    action = Action(kind=0, star=star, mode=0, exposure_d=1.0)
    return world.observe(law, action, row_key(KEY, star, index)).payload[0]


def test_row_noise_depends_only_on_star_index_and_key(toy):
    """The same row built in a different order draws the same noise.

    Two stars with identical truths. Rows (star, index) are built in one order, then in
    reverse with other rows interleaved; each row's payload agrees to ULPs. Rows at the
    same index on the two stars differ, and so do two rows of one star. World i of
    `draw_worlds` does not depend on how many worlds are drawn.

    Fails if a row's key comes from call order or a running counter, or if the star or
    the row index is not folded into it.
    """
    world = SealedWorld(_truth(toy, [[0.5, 1.0], [0.5, 1.0]]))
    law = GaussianGasLaw(snr_one_day=[5.0, 5.0])
    rows = [(0, 0), (1, 1), (0, 2), (1, 3)]
    first = {r: _observe(world, law, *r) for r in rows}
    second = {}
    for r in reversed(rows):
        _observe(world, law, 1, 7)  # an unrelated row in between
        second[r] = _observe(world, law, *r)
    for r in rows:
        assert_array_max_ulp(first[r], second[r], maxulp=4)
    assert not np.allclose(_observe(world, law, 0, 5), _observe(world, law, 1, 5))
    assert not np.allclose(first[(0, 0)], first[(0, 2)])

    def prior(key):
        return _truth(toy, jax.random.normal(key, (2, 2)))

    two = draw_worlds(prior, KEY, 2).unseal().atoms.x_mean
    three = draw_worlds(prior, KEY, 3).unseal().atoms.x_mean
    assert_array_max_ulp(two, three[:2], maxulp=4)


def test_observe_reads_the_right_star_and_unseal_gives_a_point_belief(toy):
    """`observe` draws from the action's star, and `unseal` returns the truth itself.

    Star 0 has gases (0.3, -0.4) and star 1 (-0.3, 0.4); at S/N 10^4 the readings are
    within 1e-3 dex of the true star's abundances. Unsealed, each star is one draw of
    weight one even when the truth was built with a zero weight: q of (water, oxygen)
    is the truth bit (0 and 0) and q of water alone is (1, 0); the count distribution
    is a one at the true count. A truth with a gas variance is refused.

    Fails if `observe` reads the wrong star, `unseal` keeps a weight or a variance, or
    the pool of an unsealed world adds expected planets.
    """
    truth = _truth(toy, [[0.3, -0.4], [-0.3, 0.4]])
    truth = eqx.tree_at(lambda a: a.cloud.log_weight, truth, jnp.full((2, 1), -jnp.inf))
    world = SealedWorld(truth)
    law = GaussianGasLaw(snr_one_day=[1.0e4, 1.0e4])
    for star, x in enumerate(([0.3, -0.4], [-0.3, 0.4])):
        assert_allclose(_observe(world, law, star, 0)[0, :, 1], x, atol=1e-3)

    belief = world.unseal()
    planet = jnp.asarray(0)
    water = Claim((GasAbove(0, GAS_FLOOR),))
    both = Claim((GasAbove(0, GAS_FLOOR), GasAbove(1, GAS_FLOOR)))
    per_star = jax.vmap(q, in_axes=(0, None, None, None))
    assert_allclose(per_star(belief, water, toy.star, planet), [[1.0], [0.0]])
    assert_allclose(per_star(belief, both, toy.star, planet), [[0.0], [0.0]])
    pmf = jax.vmap(count_pmf, in_axes=(0, None, None, None))(belief, water, (), 2)
    assert_allclose(pmf, [[[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]]])

    fuzzy = eqx.tree_at(lambda a: a.x_var, truth, jnp.full((2, 1, 1, 2), 0.1))
    with pytest.raises(eqx.EquinoxRuntimeError, match="point state"):
        SealedWorld(fuzzy)


def test_reveal_releases_grant_rows_at_time_zero(toy):
    """A grant's rows enter at time zero, charged nothing, with their payloads intact.

    Fails if a granted row is released late, charged against the budget, or altered.
    """
    payload = jnp.arange(4.0).reshape(1, 2, 2)
    rows = _stack(
        [Row(i, 0, 3.0, 2.0, 2.0, 5.0, 0, 0, (payload + i,)) for i in range(2)]
    )
    world = SealedWorld(_truth(toy, [[0.0, 0.0]]))
    revealed = world.reveal(Grant(rows), KEY)
    assert_allclose(revealed.release_d, [0.0, 0.0])
    assert_allclose(revealed.charged_d, [0.0, 0.0])
    assert_allclose(revealed.payload[0], rows.payload[0])
    assert_allclose(revealed.star, [0, 1])
