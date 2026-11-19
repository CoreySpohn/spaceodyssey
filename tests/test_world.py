"""Tests of the sealed world: row keys, grants, unsealing and drawing worlds."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import norm
from numpy.testing import assert_allclose, assert_array_max_ulp

from spaceodyssey.belief import count_pmf, q
from spaceodyssey.claim import Claim, GasAbove
from spaceodyssey.design import Mode
from spaceodyssey.log import grant_rows
from spaceodyssey.toy import GAS_FLOOR, GaussianGasModel, StepReportModel
from spaceodyssey.types import Action, Grant, ObservationLog, WeightedCloud
from spaceodyssey.world import SealedWorld, draw_worlds, row_key, run_keys

KEY = jax.random.key(0)
STARS = jnp.arange(2)  # two stars, described by their index alone


def _point_truth(toy, abundances):
    """A truth of one planet per star; `abundances` has shape (S, G)."""
    stars = [
        toy.atoms([[1.0]], [[1.0]], [1.0], jnp.asarray(x)[None], 0.0, len(x))
        for x in abundances
    ]
    return toy.stack(*stars)


def _differ(a, b):
    """Whether two payloads differ in some array."""
    return not all(jax.tree.leaves(jax.tree.map(np.allclose, a, b)))


def _observe(world, model, star, index):
    action = Action(star=star, mode=0, exposure_d=1.0)
    return world.observe(row_key(KEY, star, index), model, action)


def test_row_noise_depends_only_on_star_index_and_key(toy):
    """The same row built in a different order draws the same noise.

    Two stars with identical truths. Rows (star, index) are built in one order, then in
    reverse with other rows interleaved; each row's payload agrees to ULPs. Rows at the
    same index on the two stars differ, and so do two rows of one star. World i of
    `draw_worlds` does not depend on how many worlds are drawn.

    Fails if a row's key comes from call order or a running counter, or if the star or
    the row index is not folded into it.
    """
    world = SealedWorld(_point_truth(toy, [[0.5, 1.0], [0.5, 1.0]]))
    model = GaussianGasModel(snr_one_day=[5.0, 5.0])
    rows = [(0, 0), (1, 1), (0, 2), (1, 3)]
    first = {r: _observe(world, model, *r) for r in rows}
    second = {}
    for r in reversed(rows):
        _observe(world, model, 1, 7)  # an unrelated row in between
        second[r] = _observe(world, model, *r)
    for r in rows:
        toy.assert_ulp(first[r], second[r])
    assert _differ(_observe(world, model, 0, 5), _observe(world, model, 1, 5))
    assert _differ(first[(0, 0)], first[(0, 2)])

    two = draw_worlds(KEY, toy.GasPrior("toy", n_gas=2), STARS, 2).unseal().x_mean
    three = draw_worlds(KEY, toy.GasPrior("toy", n_gas=2), STARS, 3).unseal().x_mean
    assert_array_max_ulp(two, three[:2], maxulp=4)


def test_draw_worlds_from_a_role_prior(toy):
    """Worlds come from the truth's role prior, one point draw per star.

    4,000 worlds of two stars under iid N(0, 1) gases: the leading axes are (world,
    star), every truth is one draw of weight one, and the 16,000 abundances have mean
    within 0.05 of 0 and variance within 0.05 of 1 (about 4 standard errors). The
    prior record that drew them scores them: its `log_prob` of the unsealed truth is
    the N(0, 1) log density summed over the gases.

    Fails if `draw_worlds` ignores the prior's sampler, reuses one key across stars or
    worlds (the spread collapses), or the drawn truth is not on the prior's terms.
    """
    prior = toy.GasPrior("toy", n_gas=2)
    truth = draw_worlds(KEY, prior, STARS, 4000).unseal()
    assert truth.x_mean.shape == (4000, 2, 1, 1, 2)
    assert_allclose(truth.cloud.log_weights, 0.0)
    x = np.asarray(truth.x_mean)
    assert abs(x.mean()) < 0.05 and abs(x.var() - 1.0) < 0.05
    one = jax.tree.map(lambda leaf: leaf[0, 1], truth)
    expected = norm.logpdf(one.x_mean).sum()
    assert_allclose(prior.log_prob(one, STARS[1]), [expected], rtol=1e-12)


def test_world_keys_and_row_keys_are_distinct(toy):
    """One run key gives a world key and a noise key that share nothing.

    Without distinct tags, star s of world i and row (star i, index s) would fold the
    same numbers into one key. The prior draws its gases as N(0, 1) of its key and the
    gas model draws its noise as N(0, 1) of the row key, with the same shape, so a
    shared key shows as equal numbers. For every world i and star s in {0, 1}, the
    truth `draw_worlds` draws from the run key differs from the standard normal of
    row_key(noise, i, s) and of row_key(noise, s, i), with the noise key of the same
    run key; and the two keys of `run_keys` differ.

    Fails if `run_keys` folds no tag or one tag twice, so world i is correlated with
    star i's row noise.
    """
    world_key, noise = run_keys(KEY)
    assert not np.array_equal(
        jax.random.key_data(world_key), jax.random.key_data(noise)
    )
    truth = draw_worlds(KEY, toy.GasPrior("toy", n_gas=2), STARS, 2).unseal().x_mean
    for i in range(2):
        for s in range(2):
            for star, index in ((i, s), (s, i)):
                noise_draw = jax.random.normal(row_key(noise, star, index), (1, 1, 2))
                assert not np.allclose(truth[i, s], noise_draw)


def test_observe_reads_the_right_star_and_unseal_gives_the_truth(toy):
    """`observe` draws from the action's star, and `unseal` returns the truth itself.

    Star 0 has gases (0.3, -0.4) and star 1 (-0.3, 0.4); at S/N 10^4 the readings are
    within 1e-3 dex of the true star's abundances. Unsealed, each star is one draw of
    unit weight (log weight and reference density zero) even when the truth was built
    with a zero weight, and read as a belief of one role: q of (water, oxygen) is the
    truth bit (0 and 0) and q of water alone is (1, 0); the count distribution is a one
    at the true count. A truth of two planets unseals whole. A truth with a gas
    variance is refused, and so is a truth on no chart.

    Fails if `observe` reads the wrong star, `unseal` keeps a weight or a variance or
    drops a planet, or sealing skips the cloud's chart check.
    """
    truth = _point_truth(toy, [[0.3, -0.4], [-0.3, 0.4]])
    truth = eqx.tree_at(
        lambda a: a.cloud.log_weights, truth, jnp.full((2, 1), -jnp.inf)
    )
    world = SealedWorld(truth)
    model = GaussianGasModel(snr_one_day=[1.0e4, 1.0e4])
    for star, x in enumerate(([0.3, -0.4], [-0.3, 0.4])):
        assert_allclose(_observe(world, model, star, 0)["reading"][0], x, atol=1e-3)

    unsealed = world.unseal()
    assert_allclose(unsealed.cloud.log_weights, 0.0)
    assert_allclose(unsealed.cloud.log_ref_density, 0.0)
    belief = jax.vmap(toy.belief)(unsealed)
    planet = jnp.asarray(0)
    water = Claim((GasAbove(0, GAS_FLOOR),))
    both = Claim((GasAbove(0, GAS_FLOOR), GasAbove(1, GAS_FLOOR)))
    per_star = jax.vmap(q, in_axes=(0, None, None, None))
    assert_allclose(per_star(belief, water, toy.star, planet), [[1.0], [0.0]])
    assert_allclose(per_star(belief, both, toy.star, planet), [[0.0], [0.0]])
    pmf = jax.vmap(count_pmf, in_axes=(0, None, None, None))(belief, water, (), 2)
    assert_allclose(pmf, [[[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]]])

    two = toy.atoms([[1.0, 1.0]], [[1.0, 0.0]], [1.0], [[0.5, -0.5]], 0.0)
    assert SealedWorld(toy.stack(two)).unseal().x_mean.shape == (1, 1, 2, 2)

    fuzzy = eqx.tree_at(lambda a: a.x_var, truth, jnp.full((2, 1, 1, 2), 0.1))
    with pytest.raises(eqx.EquinoxRuntimeError, match="point state"):
        SealedWorld(fuzzy)
    no_chart = WeightedCloud(truth.cloud.samples, truth.cloud.log_weights, jnp.nan)
    with pytest.raises(ValueError, match="chart"):
        SealedWorld(eqx.tree_at(lambda a: a.cloud, truth, no_chart))


def _grant():
    """Three granted rows: star 1 in mode 2 (model slot 1), then stars 0 and 1 in mode
    0 (slot 0).
    """
    actions = Action(star=[1, 0, 1], mode=[2, 0, 0], exposure_d=[4.0, 1.0, 2.0])
    return Grant(actions)


MODELS = (
    GaussianGasModel(snr_one_day=[5.0, 5.0]),
    StepReportModel(snr_one_day=[3.0, 3.0]),
)
# Modes 0 and 1 read the gas model, mode 2 the step report.
MODES = (Mode(0, 0.0), Mode(0, 0.0), Mode(1, 0.0))
SLOTS = (1, 0, 0)  # the model slot of each granted row, read from MODES


def test_reveal_draws_grant_rows_from_the_truth_at_time_zero(toy):
    """Granted rows are drawn as `observe` draws them, released at time zero, free.

    Granted row i equals `observe` under the model in its slot with key row_key(key,
    star, i), to ULPs; its log row is released at 0 and charged 0, keeps its action's
    star, mode and exposure, carries its model slot, and its other model slot is zero.
    The step row (S/N 3 x 2 = 6) reports the true bits of star 1, (0, 1). The rows fill
    the head of a log of capacity 5 with count 3.

    Fails if a granted row is released late or charged, its noise is keyed by anything
    but (star, row index, key), its payload lands in the wrong model slot, or the grant
    reads a star other than its action's.
    """
    world = SealedWorld(_point_truth(toy, [[0.5, 1.0], [-0.5, 1.0]]))
    grant = _grant()
    payload = world.reveal(KEY, grant, MODELS, MODES)
    revealed = grant_rows(grant, payload, MODES)
    assert_allclose(revealed.release_d, 0.0)
    assert_allclose(revealed.charged_d, 0.0)
    assert_allclose(revealed.time_d, 0.0)
    assert_allclose(revealed.exposure_d, [4.0, 1.0, 2.0])
    assert_allclose(revealed.star, [1, 0, 1])
    assert_allclose(revealed.mode, [2, 0, 0])
    assert_allclose(revealed.model_slot, [1, 0, 0])
    for i, slot in enumerate(SLOTS):
        action = jax.tree.map(lambda leaf, i=i: leaf[i], grant.actions)
        key = row_key(KEY, action.star, i)
        expected = world.observe(key, MODELS[slot], action)
        toy.assert_ulp(
            jax.tree.map(lambda x, i=i: x[i], revealed.payload[slot]), expected
        )
        for other in jax.tree.leaves(revealed.payload[1 - slot]):
            assert_allclose(other[i], 0.0)
    assert_allclose(revealed.payload[1]["snr"][0, 0], [6.0, 6.0])
    assert_allclose(revealed.payload[1]["bit"][0, 0], [0.0, 1.0])

    pad = jax.tree.map(lambda x: jnp.zeros((2, *x.shape[1:]), x.dtype), revealed)
    log = ObservationLog(
        jax.tree.map(lambda x, z: jnp.concatenate([x, z]), revealed, pad), count=3
    )
    assert log.rows.star.shape == (5,) and int(log.count) == 3
    with pytest.raises(ValueError, match="capacity"):
        ObservationLog(eqx.tree_at(lambda r: r.star, log.rows, jnp.zeros(4, int)), 3)


def test_reveal_differs_between_worlds(toy):
    """Two worlds with different truths reveal different rows from one grant and key.

    An empty grant reveals no rows, and a grant whose model slot is not in the model
    table is refused.

    Fails if the grant's rows do not depend on the truth (rows carried by the grant, or
    drawn from a fixed state), or a row of a missing model is silently dropped.
    """
    grant = _grant()
    a = SealedWorld(_point_truth(toy, [[0.5, 1.0], [-0.5, 1.0]]))
    b = SealedWorld(_point_truth(toy, [[0.5, 1.0], [0.5, -1.0]]))
    payload_a, payload_b = (
        a.reveal(KEY, grant, MODELS, MODES),
        b.reveal(KEY, grant, MODELS, MODES),
    )
    for slot in (0, 1):
        assert _differ(payload_a[slot], payload_b[slot])
    no_int = jnp.zeros(0, int)
    empty = Grant(Action(no_int, no_int, jnp.zeros(0)))
    none = grant_rows(empty, a.reveal(KEY, empty, MODELS, MODES), MODES)
    assert none.star.shape == (0,)
    assert {k: v.shape for k, v in none.payload[0].items()} == {
        "snr": (0, 1, 2),
        "reading": (0, 1, 2),
    }
    with pytest.raises(ValueError, match="model table"):
        a.reveal(KEY, grant, MODELS[:1], MODES)
