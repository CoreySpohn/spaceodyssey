"""Replaying logged rows into a belief: `infer`, `replay`, `absorb`.

The toy state is one planet with one gas per draw. A Gaussian gas row of S/N 2.5 has
noise sd 2.5 / 2.5 = 1 dex, so the closed forms are those of a unit-noise reading.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import ndtri
from jax.scipy.stats import norm
from numpy.testing import assert_allclose, assert_array_max_ulp

from spaceodyssey.belief import absorb, infer, replay, replay_belief, weights
from spaceodyssey.roles import AbstractRolePrior, Roles
from spaceodyssey.toy import GaussianGasModel, StepReportModel
from spaceodyssey.types import Atoms, ObservationLog, Refit, Row, WeightedCloud

STAR = ()
CAPACITY = 8  # W
GAS = GaussianGasModel(snr_one_day=[2.5])
MODELS = (GAS, StepReportModel(snr_one_day=[5.0]))


class GasGaussian(AbstractRolePrior):
    """N(mean, sd^2) on the abundance of the one gas of the one planet."""

    mean: jax.Array = eqx.field(converter=jnp.asarray, default=0.0)
    sd: jax.Array = eqx.field(converter=jnp.asarray, default=1.0)

    def log_prob(self, atoms, star):
        return norm.logpdf(atoms.x_mean[:, 0, 0], self.mean, self.sd)

    def sample(self, key, star):
        x = self.mean + self.sd * jax.random.normal(key, (1,))
        return _atoms(x, 0.0, jnp.zeros(1))


# Forecast N(0, 1); verdict members N(0, 1) and N(1, 1).
ROLES = Roles(GasGaussian("toy"), GasGaussian("toy", mean=[0.0, 1.0], sd=[1.0, 1.0]))


def _atoms(x, var, log_ref_density):
    """Draws of one planet with one gas: abundance mean `x` and variance `var`."""
    x = jnp.asarray(x, float)
    n_draw = x.shape[0]
    block = jnp.broadcast_to(jnp.asarray(var, float), (n_draw,))[:, None, None]
    cloud = WeightedCloud({"x": x}, jnp.zeros(n_draw), log_ref_density, "toy")
    return Atoms(cloud, x[:, None, None], block, jnp.ones((n_draw, 1), bool))


def _refit(atoms, represented=(), star=0):
    """A refit of star `star` whose cloud represents the log rows `represented`."""
    mask = jnp.zeros(CAPACITY, bool).at[jnp.asarray(represented, int)].set(True)
    return Refit(atoms, mask, star)


def _payload(snr, value, name):
    """One payload buffer, `(K, G) = (1, 1)` arrays."""
    return {"snr": jnp.full((1, 1), snr), name: jnp.full((1, 1), value)}


def _gas(snr, reading):
    return _payload(snr, reading, "reading")


def _step(snr, bit):
    return _payload(snr, bit, "bit")


NO_GAS, NO_STEP = _gas(0.0, 0.0), _step(0.0, 0.0)


def _row(slot, *payload, star=0, refusal=0):
    """One row; `payload` holds one buffer per model slot, `slot` names the real one."""
    return Row(
        star=star,
        mode=0,
        time_d=0.0,
        exposure_d=1.0,
        charged_d=0.0,
        release_d=0.0,
        refusal=refusal,
        model_slot=slot,
        payload=payload,
    )


def _log(rows, count):
    """A log of capacity `CAPACITY` holding `rows` (padded with copies of the last)."""
    rows = rows + [rows[-1]] * (CAPACITY - len(rows))
    return ObservationLog(jax.tree.map(lambda *x: jnp.stack(x), *rows), count)


def _with_count(log, count):
    return eqx.tree_at(lambda x: x.count, log, jnp.asarray(count))


def _moments(b):
    """Weighted mean and variance of the abundance under every role."""
    w, x = weights(b), b.atoms.x_mean[:, 0, 0]
    mean = w @ x
    return mean, w @ x**2 - mean**2


def _assert_ulp(a, b, maxulp=4):
    """Equal to `maxulp` ULPs where finite, and -inf in the same places."""
    a, b = np.asarray(a), np.asarray(b)
    np.testing.assert_array_equal(np.isneginf(a), np.isneginf(b))
    finite = np.isfinite(a)
    assert_array_max_ulp(a[finite], b[finite], maxulp=maxulp)


INFER = eqx.filter_jit(infer)


def test_replay_applies_each_row_outside_the_mask_once(toy):
    """The conjugate replay: a represented row counts zero times, a new row once.

    x ~ N(0, 1) (the refit target p0 and the role prior of the forecast and the first
    member); rows y1 = y2 = 1 with noise sd 1. The draws are the 20,000 midpoint
    quantiles of the posterior given row 1, N(0.5, 0.5), with `row_mask` holding row
    1 and `log_weight` 0, and `log_ref_density` is log p0. Row 2 is outside. `infer`
    gives the posterior of precision 1 + 1 + 1 = 3: mean 2/3 and variance 1/3 under
    N(0, 1), and mean (1 + 1 + 1) / 3 = 1 under the member N(1, 1). Rows that must add
    nothing sit in the same log: another star's row, a refused row, a step row whose
    gas buffer holds a stray reading, a gas row whose step buffer holds a stray bit, and
    a row past `count`, each reading 5 (or reporting the gas absent). Counting row 1
    again gives precision 4 (mean 3/4, variance 1/4); dropping row 2 leaves the draws'
    own N(0.5, 0.5); both are computed here to show the test separates them.

    Fails if a row in `row_mask` is applied again, a released row is dropped, a row of
    another star, a refused row or a row past `count` is applied, a buffer is read in
    the wrong model slot, or `log_ref_density` enters with the wrong sign.
    """
    n = 20_000
    x = 0.5 + jnp.sqrt(0.5) * ndtri((jnp.arange(n) + 0.5) / n)
    atoms = _atoms(x, 0.0, norm.logpdf(x))
    refit = _refit(atoms, [0])
    rows = [
        _row(0, _gas(2.5, 1.0), NO_STEP),  # row 1, represented by the cloud
        _row(0, _gas(2.5, 1.0), NO_STEP),  # row 2, applied
        _row(0, _gas(2.5, 5.0), NO_STEP, star=1),
        _row(0, _gas(2.5, 5.0), NO_STEP, refusal=1),
        _row(1, _gas(2.5, 5.0), NO_STEP),
        _row(0, NO_GAS, _step(5.0, 0.0)),
        _row(0, _gas(2.5, 5.0), NO_STEP),  # past count
    ]
    log = _log(rows, 6)
    pool = toy.empty_pool(n_roles=3)

    mean, var = _moments(INFER(ROLES, refit, log, MODELS, STAR, pool))
    assert_allclose(mean, [2 / 3, 2 / 3, 1.0], rtol=1e-3)
    assert_allclose(var, [1 / 3, 1 / 3, 1 / 3], rtol=1e-3)

    twice = INFER(ROLES, _refit(atoms), log, MODELS, STAR, pool)
    assert_allclose(_moments(twice)[0][0], 0.75, rtol=1e-3)
    assert_allclose(_moments(twice)[1][0], 0.25, rtol=1e-3)
    dropped = INFER(ROLES, refit, _with_count(log, 1), MODELS, STAR, pool)
    assert_allclose(_moments(dropped)[0][0], 0.5, rtol=1e-3)
    assert_allclose(_moments(dropped)[1][0], 0.5, rtol=1e-3)


def test_replay_is_order_free_and_absorb_equals_infer(toy):
    """Row order does not matter, and rows absorbed one by one equal the batch.

    Nine point draws, a grant row in `row_mask` (row 0), then gas rows at S/N 5, 1 and
    3, a step row that reports the gas present (so draws below the floor drop out), a
    silent step row and a row of star 1. Permuting the observed rows changes the
    likelihood and weights by at most 4 ULPs. A replay of the grant alone, advanced by
    `absorb` after each appended row, gives `infer`'s belief on the whole log to 4
    ULPs; absorbing with no new row changes nothing. Under `eqx.filter_jit`, `absorb`
    traces once for every row and for a replay of the other star, which applies only
    its own row.

    Fails if a row is absorbed twice or skipped, `absorb` applies a different rule or
    arithmetic than `infer`, the replay depends on where a row sits in the log, or a
    second row or star retraces `absorb`.
    """
    x = jnp.linspace(-1.75, 2.25, 9)
    atoms = _atoms(x, 0.0, norm.logpdf(x))
    refit = _refit(atoms, [0])
    observed = [
        _row(0, _gas(5.0, 1.3), NO_STEP),
        _row(1, NO_GAS, _step(5.0, 1.0)),
        _row(0, _gas(2.5, 0.4), NO_STEP, star=1),
        _row(0, _gas(1.0, -0.2), NO_STEP),
        _row(1, NO_GAS, _step(2.0, 0.0)),
        _row(0, _gas(3.0, 0.9), NO_STEP),
    ]
    grant = _row(0, _gas(2.5, 0.8), NO_STEP)
    log = _log([grant, *observed], 7)
    pool = toy.empty_pool(n_roles=3)
    batch = INFER(ROLES, refit, log, MODELS, STAR, pool)

    shuffled = _log([grant, *(observed[i] for i in (4, 2, 0, 5, 3, 1))], 7)
    other = INFER(ROLES, refit, shuffled, MODELS, STAR, pool)
    _assert_ulp(other.log_like, batch.log_like)
    _assert_ulp(weights(other), weights(batch))

    traces = []

    @eqx.filter_jit
    def step(r, log):
        traces.append(None)
        return absorb(r, log, MODELS, STAR)

    r = replay(ROLES, refit, _with_count(log, 1), MODELS, STAR, pool)
    for count in range(2, 8):
        r = step(r, _with_count(log, count))
    held = step(r, log)
    assert int(held.count) == 7
    jax.tree.map(np.testing.assert_array_equal, held.stats, r.stats)
    online = replay_belief(r, MODELS, STAR)
    _assert_ulp(online.log_like, batch.log_like)
    _assert_ulp(weights(online), weights(batch))

    star_one = replay(
        ROLES, _refit(atoms, [0], star=1), _with_count(log, 3), MODELS, STAR, pool
    )
    star_one = step(star_one, log)
    assert len(traces) == 1
    only_its_row = _log([grant, _row(0, _gas(2.5, 0.4), NO_STEP)], 2)
    expected = INFER(ROLES, refit, only_its_row, MODELS, STAR, pool)
    _assert_ulp(replay_belief(star_one, MODELS, STAR).log_like, expected.log_like)


def test_infer_on_an_uncertain_gas_uses_the_joint_marginal(toy):
    """Rows reading one Gaussian gas enter through summed statistics and `finalize`.

    Two draws whose gas is N(0, 1) and N(1, 0.5); rows y1 = 1 at S/N 5 (noise variance
    0.25) and y2 = 0.4 at S/N 2.5 (variance 1), neither represented. The likelihood of
    a draw N(m, v) is the joint (y1, y2) ~ N((m, m), [[v + 0.25, v], [v, v + 1]]),
    computed here (-2.50728 for the first draw), not the sum of the two marginals. The
    blocks update to precision 1 + 4 + 1 = 6, mean 4.4 / 6, and precision 2 + 4 + 1 =
    7, mean 6.4 / 7. The same rows split over two model slots give the same belief to
    1e-12: the slots are finalized in turn, each against the draws the last returned.

    Fails if `infer` sums per-row likelihoods on an uncertain gas, finalizes each slot
    against the refit's draws (the prior counted once per slot), or drops a row's
    statistics.
    """
    refit = _refit(_atoms([0.0, 1.0], [1.0, 0.5], jnp.zeros(2)))
    pool = toy.empty_pool(n_roles=3)
    one_slot = _log([_row(0, _gas(5.0, 1.0)), _row(0, _gas(2.5, 0.4))], 2)
    b = INFER(ROLES, refit, one_slot, (GAS,), STAR, pool)

    y = np.array([1.0, 0.4])
    joint = []
    for m, v in ((0.0, 1.0), (1.0, 0.5)):
        cov = np.array([[v + 0.25, v], [v, v + 1.0]])
        r = y - m
        joint.append(
            -0.5
            * (r @ np.linalg.solve(cov, r) + np.log(np.linalg.det(2 * np.pi * cov)))
        )
    assert_allclose(joint[0], -2.50728, atol=1e-5)
    assert_allclose(b.log_like, np.broadcast_to(joint, (3, 2)), rtol=1e-12)
    assert_allclose(b.atoms.x_mean[:, 0, 0], [4.4 / 6, 6.4 / 7], rtol=1e-12)
    assert_allclose(b.atoms.x_var[:, 0, 0], [1 / 6, 1 / 7], rtol=1e-12)

    two_slots = _log(
        [_row(0, _gas(5.0, 1.0), NO_GAS), _row(1, NO_GAS, _gas(2.5, 0.4))], 2
    )
    split = INFER(ROLES, refit, two_slots, (GAS, GAS), STAR, pool)
    assert_allclose(split.log_like, b.log_like, rtol=1e-12)
    assert_allclose(split.atoms.x_mean, b.atoms.x_mean, rtol=1e-12)
    assert_allclose(split.atoms.x_var, b.atoms.x_var, rtol=1e-12)


def test_infer_refuses_what_it_cannot_replay(toy):
    """The measure check of the role weights holds through `infer`, and shapes are
    checked.

    A verdict prior on measure "y" against a cloud on "toy" raises, as do a model table
    whose length differs from the log's payload buffers, a `row_mask` whose capacity
    differs from the log's, and a belief's updated draws passed as a refit.

    Fails if `infer` builds role weights without the measure check (a missing Jacobian
    goes unnoticed), reads a payload buffer or a mask entry that does not exist, or
    replays draws that already carry the rows.
    """
    refit = _refit(_atoms([0.0, 1.0], 0.0, jnp.zeros(2)))
    log = _log([_row(0, _gas(2.5, 1.0), NO_STEP)], 1)
    pool = toy.empty_pool(n_roles=3)
    on_y = Roles(ROLES.forecast, GasGaussian("y", mean=[0.0, 1.0], sd=[1.0, 1.0]))
    with pytest.raises(ValueError, match="measure"):
        infer(on_y, refit, log, MODELS, STAR, pool)
    with pytest.raises(ValueError, match="payload buffers"):
        infer(ROLES, refit, log, (GAS,), STAR, pool)
    short = eqx.tree_at(lambda r: r.row_mask, refit, jnp.zeros(3, bool))
    with pytest.raises(ValueError, match="capacity"):
        infer(ROLES, short, log, MODELS, STAR, pool)
    updated = infer(ROLES, refit, log, MODELS, STAR, pool).atoms
    with pytest.raises((TypeError, AttributeError)):
        infer(ROLES, updated, log, MODELS, STAR, pool)
