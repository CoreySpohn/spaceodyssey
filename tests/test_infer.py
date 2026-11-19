"""Replaying logged rows into a belief: `replay`, `absorb`, `replay_belief`.

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
from numpy.testing import assert_allclose

from spaceodyssey.belief import absorb, replay, replay_belief, weights
from spaceodyssey.roles import Roles
from spaceodyssey.toy import GaussianGasModel, StepReportModel
from spaceodyssey.types import Atoms, ObservationLog, Refit, Row, WeightedCloud

STAR = ()
CAPACITY = 8  # W
GAS = GaussianGasModel(snr_one_day=[2.5])
MODELS = (GAS, StepReportModel(snr_one_day=[5.0]))


@pytest.fixture(scope="module")
def roles(toy):
    """Forecast N(0, 1) on the one gas; verdict members N(0, 1) and N(1, 1)."""
    return Roles(
        toy.GasPrior("toy"), toy.GasPrior("toy", mean=[0.0, 1.0], sd=[1.0, 1.0])
    )


def _atoms(x, var, log_ref_density):
    """Draws of one planet with one gas: abundance mean `x` and variance `var`."""
    x = jnp.asarray(x, float)
    n_draw = x.shape[0]
    block = jnp.broadcast_to(jnp.asarray(var, float), (n_draw,))[:, None, None]
    cloud = WeightedCloud(
        {"x": x}, jnp.zeros(n_draw), jnp.nan, log_ref_density, chart="toy"
    )
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


def _row(slot, *payload, star=0, refusal=0, release_d=0.0):
    """One row; `payload` holds one buffer per model slot, `slot` names the real one."""
    return Row(
        star=star,
        mode=0,
        time_d=0.0,
        exposure_d=1.0,
        charged_d=0.0,
        release_d=release_d,
        refusal=refusal,
        model_slot=slot,
        payload=payload,
    )


def _log(rows, count):
    """A log of capacity `CAPACITY` holding `rows` (padded with copies of the last)."""
    rows = rows + [rows[-1]] * (CAPACITY - len(rows))
    return ObservationLog(jax.tree.map(lambda *x: jnp.stack(x), *rows), count)


def _with_count(log, count):
    """The log with another count, built through its converter (a strong integer)."""
    return ObservationLog(log.rows, count)


def _moments(b):
    """Weighted mean and variance of the abundance under every role."""
    w, x = weights(b), b.atoms.x_mean[:, 0, 0]
    mean = w @ x
    return mean, w @ x**2 - mean**2


NOW = jnp.asarray(1.0)  # a query time after every row released at time zero


@eqx.filter_jit
def _infer(roles, refit, log, models, pool, time_d=NOW):
    """The belief at `time_d` under `roles`: `replay_belief` of `replay`."""
    return replay_belief(
        replay(refit, log, models, STAR, time_d), roles, models, STAR, pool
    )


def test_replay_applies_each_row_outside_the_mask_once(toy, roles):
    """The conjugate replay: a represented row counts zero times, a new row once.

    x ~ N(0, 1) (the refit target p0 and the role prior of the forecast and the first
    member); rows y1 = y2 = 1 with noise sd 1. The draws are the 20,000 midpoint
    quantiles of the posterior given row 1, N(0.5, 0.5), with `row_mask` holding row
    1 and `log_weights` 0, and `log_ref_density` is log p0. Row 2 is outside. The
    belief has the posterior of precision 1 + 1 + 1 = 3: mean 2/3 and variance 1/3
    under N(0, 1), and mean (1 + 1 + 1) / 3 = 1 under the member N(1, 1). Rows that
    must add nothing sit in the same log: another star's row, a refused row, a step
    row whose gas buffer holds a stray reading, a gas row whose step buffer holds a
    stray bit, a row in flight (released at day 10) and a row past `count`, each
    reading 5 (or reporting the gas absent) except the row in flight, which reads 2.
    Counting row 1 again gives precision 4 (mean 3/4, variance 1/4); dropping row 2
    leaves the draws' own N(0.5, 0.5); both are computed here to show the test
    separates them. At day 10 the row in flight is released: precision 4, mean
    (1 + 1 + 2) / 4 = 1 under N(0, 1) and (1 + 1 + 1 + 2) / 4 = 5/4 under N(1, 1),
    variance 1/4.

    Fails if a row in `row_mask` is applied again, a released row is dropped, a row of
    another star, a refused row, a row in flight or a row past `count` is applied, a
    row is not applied once released, a buffer is read in the wrong model slot, or
    `log_ref_density` enters with the wrong sign.
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
        _row(0, _gas(2.5, 2.0), NO_STEP, release_d=10.0),  # in flight until day 10
        _row(0, _gas(2.5, 5.0), NO_STEP),  # past count
    ]
    log = _log(rows, 7)
    pool = toy.empty_pool(n_roles=3)

    mean, var = _moments(_infer(roles, refit, log, MODELS, pool))
    assert_allclose(mean, [2 / 3, 2 / 3, 1.0], rtol=1e-3)
    assert_allclose(var, [1 / 3, 1 / 3, 1 / 3], rtol=1e-3)

    twice = _infer(roles, _refit(atoms), log, MODELS, pool)
    assert_allclose(_moments(twice)[0][0], 0.75, rtol=1e-3)
    assert_allclose(_moments(twice)[1][0], 0.25, rtol=1e-3)
    dropped = _infer(roles, refit, _with_count(log, 1), MODELS, pool)
    assert_allclose(_moments(dropped)[0][0], 0.5, rtol=1e-3)
    assert_allclose(_moments(dropped)[1][0], 0.5, rtol=1e-3)

    released = _infer(roles, refit, log, MODELS, pool, jnp.asarray(10.0))
    mean, var = _moments(released)
    assert_allclose(mean, [1.0, 1.0, 1.25], rtol=1e-3)
    assert_allclose(var, [0.25, 0.25, 0.25], rtol=1e-3)


def test_replay_is_order_free_and_absorb_equals_replay(toy, roles):
    """Row order does not matter, and rows absorbed as they fall due equal the batch.

    Nine point draws, a grant row in `row_mask` (row 0), then six rows appended on days
    1 to 6 and released half a day later, except row 4, in flight until day 9: gas rows
    at S/N 5, 1 and 3, a step row that reports the gas present (so draws below the
    floor drop out), a silent step row and a row of star 1. At day 10 every row is
    released. Permuting the observed rows changes the likelihood and weights by at most
    4 ULPs. A replay of the grant alone, brought by `absorb` to day k + 0.6 after row k
    is appended, leaves row 4 unapplied until day 10 and then gives the belief of
    `replay` on the whole log to 4 ULPs; absorbing again changes nothing. Under
    `eqx.filter_jit`, `absorb` traces once for every row and time and for a replay of
    the other star, which applies only its own row.

    Fails if a row is absorbed twice, skipped or applied before its release, `absorb`
    applies a different rule or arithmetic than `replay`, the replay depends on where
    a row sits in the log, or a second row, time or star retraces `absorb`.
    """
    x = jnp.linspace(-1.75, 2.25, 9)
    atoms = _atoms(x, 0.0, norm.logpdf(x))
    refit = _refit(atoms, [0])
    observed = [
        _row(0, _gas(5.0, 1.3), NO_STEP, release_d=1.5),
        _row(1, NO_GAS, _step(5.0, 1.0), release_d=2.5),
        _row(0, _gas(2.5, 0.4), NO_STEP, star=1, release_d=3.5),
        _row(0, _gas(1.0, -0.2), NO_STEP, release_d=9.0),
        _row(1, NO_GAS, _step(2.0, 0.0), release_d=5.5),
        _row(0, _gas(3.0, 0.9), NO_STEP, release_d=6.5),
    ]
    grant = _row(0, _gas(2.5, 0.8), NO_STEP)
    log = _log([grant, *observed], 7)
    pool = toy.empty_pool(n_roles=3)
    end = jnp.asarray(10.0)
    batch = _infer(roles, refit, log, MODELS, pool, end)

    shuffled = _log([grant, *(observed[i] for i in (4, 2, 0, 5, 3, 1))], 7)
    other = _infer(roles, refit, shuffled, MODELS, pool, end)
    toy.assert_ulp(other.log_like, batch.log_like)
    toy.assert_ulp(toy.jit.weights(other), toy.jit.weights(batch))

    traces = []

    @eqx.filter_jit
    def step(r, log, time_d):
        traces.append(None)
        return absorb(r, log, MODELS, STAR, time_d)

    r = toy.jit.replay(refit, _with_count(log, 1), MODELS, STAR, jnp.asarray(0.0))
    for count in range(2, 8):
        r = step(r, _with_count(log, count), jnp.asarray(count - 1 + 0.6))
    assert not r.applied[4]
    r = step(r, log, end)
    assert r.applied[4]
    held = step(r, log, end)
    jax.tree.map(np.testing.assert_array_equal, held, r)
    online = toy.jit.replay_belief(r, roles, MODELS, STAR, pool)
    toy.assert_ulp(online.log_like, batch.log_like)
    toy.assert_ulp(toy.jit.weights(online), toy.jit.weights(batch))

    star_one = toy.jit.replay(
        _refit(atoms, [0], star=1), _with_count(log, 3), MODELS, STAR, NOW
    )
    star_one = step(star_one, log, end)
    assert len(traces) == 1
    only_its_row = _log([grant, _row(0, _gas(2.5, 0.4), NO_STEP)], 2)
    expected = _infer(roles, refit, only_its_row, MODELS, pool)
    held = toy.jit.replay_belief(star_one, roles, MODELS, STAR, pool)
    toy.assert_ulp(held.log_like, expected.log_like)


def test_replay_on_an_uncertain_gas_uses_the_joint_marginal(toy, roles):
    """Rows reading one Gaussian gas enter through summed statistics and `finalize`.

    Two draws whose gas is N(0, 1) and N(1, 0.5); rows y1 = 1 at S/N 5 (noise variance
    0.25) and y2 = 0.4 at S/N 2.5 (variance 1), neither represented. The likelihood of
    a draw N(m, v) is the joint (y1, y2) ~ N((m, m), [[v + 0.25, v], [v, v + 1]]),
    computed here (-2.50728 for the first draw), not the sum of the two marginals. The
    blocks update to precision 1 + 4 + 1 = 6, mean 4.4 / 6, and precision 2 + 4 + 1 =
    7, mean 6.4 / 7. The same rows split over two model slots give the same belief to
    1e-12: the slots are finalized in turn, each against the draws the last returned.
    With the step report in the table and only a row of another star in its slot, the
    belief is the same again: the step report reads only point draws, but no row of
    this star reports through it.

    Fails if the replay sums per-row likelihoods on an uncertain gas, finalizes each
    slot against the refit's draws (the prior counted once per slot), drops a row's
    statistics, or the step report refuses Gaussian draws it has no row about.
    """
    refit = _refit(_atoms([0.0, 1.0], [1.0, 0.5], jnp.zeros(2)))
    pool = toy.empty_pool(n_roles=3)
    one_slot = _log([_row(0, _gas(5.0, 1.0)), _row(0, _gas(2.5, 0.4))], 2)
    b = _infer(roles, refit, one_slot, (GAS,), pool)

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
    split = _infer(roles, refit, two_slots, (GAS, GAS), pool)
    assert_allclose(split.log_like, b.log_like, rtol=1e-12)
    assert_allclose(split.atoms.x_mean, b.atoms.x_mean, rtol=1e-12)
    assert_allclose(split.atoms.x_var, b.atoms.x_var, rtol=1e-12)

    with_step = _log(
        [
            _row(0, _gas(5.0, 1.0), NO_STEP),
            _row(0, _gas(2.5, 0.4), NO_STEP),
            _row(1, NO_GAS, _step(5.0, 1.0), star=1),
        ],
        3,
    )
    mixed = _infer(roles, refit, with_step, MODELS, pool)
    assert_allclose(mixed.log_like, b.log_like, rtol=1e-12)
    assert_allclose(mixed.atoms.x_mean, b.atoms.x_mean, rtol=1e-12)


def test_replay_refuses_what_it_cannot_replay(toy, roles):
    """The chart check of the role weights holds through a replay, and shapes are
    checked.

    A verdict prior on chart "y" against a cloud on "toy" raises when the belief is
    read, as do a model table whose length differs from the log's payload buffers, a
    `row_mask` whose capacity differs from the log's, a belief's updated draws passed
    as a refit, and a step report applied to a Gaussian gas, under `jit` too.

    Fails if the belief is read without the chart check (a missing Jacobian goes
    unnoticed), the replay reads a payload buffer or a mask entry that does not exist,
    it replays draws that already carry the rows, or a step report on a Gaussian gas
    is summed as if the rows were independent.
    """
    refit = _refit(_atoms([0.0, 1.0], 0.0, jnp.zeros(2)))
    log = _log([_row(0, _gas(2.5, 1.0), NO_STEP)], 1)
    pool = toy.empty_pool(n_roles=3)
    r = replay(refit, log, MODELS, STAR, NOW)
    on_y = Roles(roles.forecast, toy.GasPrior("y", mean=[0.0, 1.0], sd=[1.0, 1.0]))
    with pytest.raises(ValueError, match="chart"):
        replay_belief(r, on_y, MODELS, STAR, pool)
    with pytest.raises(ValueError, match="payload buffers"):
        replay(refit, log, (GAS,), STAR, NOW)
    short = eqx.tree_at(lambda r: r.row_mask, refit, jnp.zeros(3, bool))
    with pytest.raises(ValueError, match="capacity"):
        replay(short, log, MODELS, STAR, NOW)
    updated = replay_belief(r, roles, MODELS, STAR, pool).atoms
    with pytest.raises((TypeError, AttributeError)):
        replay(updated, log, MODELS, STAR, NOW)
    fuzzy = _refit(_atoms([0.0, 1.0], 1.0, jnp.zeros(2)))
    reported = _log([_row(1, NO_GAS, _step(5.0, 1.0))], 1)
    with pytest.raises(RuntimeError, match="point gas"):
        _infer(roles, fuzzy, reported, MODELS, pool)
