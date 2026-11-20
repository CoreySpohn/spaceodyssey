"""Closed-form tests of the toy models: cells, the step report, the gas limit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import ndtri
from numpy.testing import assert_allclose, assert_array_max_ulp

from spaceodyssey.belief import weights
from spaceodyssey.claim import GasAbove
from spaceodyssey.information import cell_probs, update_by_cell
from spaceodyssey.toy import (
    GAS_FLOOR,
    GAS_PRIOR_SD,
    DetectionModel,
    GaussianGasModel,
    StepReportModel,
    ToyPrior,
    point_atoms,
)
from spaceodyssey.types import (
    CELL_SAMPLES,
    POISSON,
    Action,
    binned_cells,
    planet_tracks,
    row_log_like,
    row_update,
)


def _action(exposure_d=1.0):
    return Action(star=0, mode=0, exposure_d=exposure_d)


def _expected_row(snr, u, name="reading"):
    """Payload of one planet whose readings sit at their expected value."""
    snr = jnp.asarray(snr, float)
    return {"snr": snr[None], name: jnp.where(snr > 0, u, 0.0)[None]}


def _first(payload):
    """The payload of the first draw."""
    return jax.tree.map(lambda x: x[0], payload)


def _prior(toy, n_gas=2):
    """One draw whose gases carry the verdict's gas prior N(floor, 1)."""
    return toy.atoms([[1.0]], [[1.0]], [1.0], GAS_FLOOR, GAS_PRIOR_SD**2, n_gas)


@pytest.mark.parametrize(
    ("model", "edges_b", "var"),
    [
        (
            GaussianGasModel(snr_one_day=[0.0, 2.5], edges=[-0.5, 0.0, 0.8]),
            4,
            [[0.0, 0.0], [1.0, 0.5], [0.2, 0.0], [0.0, 2.0]],
        ),
        (StepReportModel(snr_one_day=[4.0, 6.0]), 3, 0.0),
    ],
    ids=["gaussian", "step"],
)
def test_cells_normalize_and_match_the_binned_default(model, edges_b, var, toy):
    """The analytic cells sum to one and match the binned-sample default.

    Grid: four draws (below, at and above the floor; point and Gaussian gases for the
    Gaussian model, point gases for the step model, which applies to those only) and two
    exposures (1 and 4 days, so S/N doubles). The Gaussian model has one uncovered gas
    (S/N 0); the step model has one gas below its gate at 1 day and above it at 4, and
    its four draws report four different bit pairs. Each analytic cell is within 3
    Monte Carlo standard errors, sqrt(p (1 - p) / n) with n = CELL_SAMPLES, of the
    default; a cell of probability 0 must match exactly.

    Fails if a cell is dropped or double counted (normalization), the joint index of the
    analytic form and of `cell` disagree, or the closed form misreads a variance.
    """
    mean = [[-0.4, 1.0], [0.0, 0.0], [0.3, -0.2], [1.5, 0.1]]
    atoms = toy.atoms([[1.0]] * 4, [[1.0]] * 4, [0.25] * 4, mean, var)
    assert model.n_cells(atoms) == edges_b**2
    for exposure in (1.0, 4.0):
        exact = model.cells(atoms, _action(exposure))
        binned = binned_cells(
            model, atoms, _action(exposure), model.cell, model.n_cells(atoms)
        )
        for p in (exact, binned):
            assert_array_max_ulp(p.sum(axis=1), np.ones(4), maxulp=4)
        se = np.sqrt(exact * (1.0 - exact) / CELL_SAMPLES)
        assert np.all(np.abs(binned - exact) <= 3.0 * se)


def _cell_likelihood(model, y, codes, atoms, star):
    """Each draw's likelihood of a row in cell `y`, from the replay (`row_log_like`).

    The search and the step report have one payload per cell; the Gaussian reading of
    slot 0 is integrated over its bin (one edge at the floor) by Gauss-Legendre on 80
    nodes out to 12 dex, its other slot unread.
    """
    if isinstance(model, DetectionModel):
        payload = {
            "exposure_d": jnp.asarray(1.0),
            "time_d": jnp.asarray(0.0),
            "p_hz_one_day": model.p_hz_one_day,
            "p_other_one_day": model.p_other_one_day,
            "detected": jnp.asarray([y // 2, y % 2], float),
        }
        assert int(model.cell(payload)) == y
        return np.exp(row_log_like(model, payload, atoms, star))
    if isinstance(model, StepReportModel):
        snr = jnp.where(codes > 0, model.snr_one_day, 0.0)
        payload = {"snr": snr, "bit": (codes == 2).astype(float)}
        assert int(model.cell(payload)) == y
        return np.exp(row_log_like(model, payload, atoms, star))
    snr = jnp.broadcast_to(model.snr_one_day, codes.shape).at[1].set(0.0)
    lo, hi = (-12.0, 0.0) if int(codes[0, 0]) == 0 else (0.0, 12.0)
    nodes, w = np.polynomial.legendre.leggauss(80)
    readings = lo + (hi - lo) * (nodes + 1.0) / 2.0

    def like(r):
        payload = {"snr": snr, "reading": jnp.zeros_like(snr).at[0, 0].set(r)}
        return jnp.exp(row_log_like(model, payload, atoms, star))

    middle = {"snr": snr, "reading": jnp.zeros_like(snr).at[0, 0].set(readings[40])}
    assert int(model.cell(middle)) == y
    return (hi - lo) / 2.0 * np.asarray(w @ jax.jit(jax.vmap(like))(readings))


@pytest.mark.parametrize(
    "model",
    [
        DetectionModel(0.6, 0.2),
        StepReportModel(snr_one_day=[6.0]),
        GaussianGasModel(snr_one_day=[2.5]),
    ],
    ids=["search", "step", "gaussian"],
)
def test_a_cell_updates_the_draws_as_the_replay_of_its_row(model, toy):
    """For every cell the draws can give, the update by the cell (`update_by_cell`)
    is the replay's update by a row in that cell (`row_log_like`), on draws that differ
    in whether a planet exists.

    Four point draws weighted (0.1, 0.2, 0.3, 0.4), two slots: slot 0 holds a planet
    on draws 0 and 1 only, in the HZ on draws 0 and 2; slot 1 holds none. One gas, 1
    dex above the floor on draws 0 and 2 and below it on 1 and 3. One day: the search
    detects with 0.6 in the HZ and 0.2 out of it, the step report reads at S/N 6, the
    Gaussian reading at S/N 2.5 (sigma 1 dex, one bin edge at the floor). The
    likelihood of a cell is the replay's: one payload per cell for the search and the
    report, the reading integrated over its bin for the Gaussian.

    Fails if a gas model's cells give a draw without the planet one outcome in a slot
    another draw holds (they learn whether it exists, and the likelihood does not
    score it), if the search's cells and likelihood disagree on an empty slot, or if
    `cells` and `cell` code a row differently.
    """
    exists = [[True, False], [True, False], [False, False], [False, False]]
    hz = [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
    water = [[1.0], [-1.0], [1.0], [-1.0]]
    w = np.array([0.1, 0.2, 0.3, 0.4])
    atoms = toy.atoms(hz, hz, w, water, 0.0, n_gas=1, exists=exists)
    b = toy.belief(atoms)
    cells = toy.jit.outcome_cells(b, model, _action())
    p_cell = w @ np.asarray(cell_probs(b, cells))
    possible = np.flatnonzero(p_cell > 0)
    assert len(possible) >= 2
    for y in possible:
        updated = weights(update_by_cell(b, cells, jnp.asarray(y)))[0]
        like = _cell_likelihood(model, int(y), cells.codes[y], atoms, toy.star)
        assert_allclose(updated, w * like / (w @ like), rtol=1e-9, atol=1e-14)


def test_a_search_detects_with_its_closed_form_probability(toy):
    """`DetectionModel`: slot k of a draw is seen with d_k = 1 - (1 - p)^t, p the
    one-day probability of its planet (0.6 in the HZ, 0.2 out of it), 0 for an empty
    slot or no exposure.

    Draws of two slots: (HZ, other), (other, empty) and (empty, empty). Over two days
    d = (0.84, 0.36), (0.36, 0) and (0, 0). The four cells index the slots' bits as a
    base-2 number, slot 0 the most significant: draw 0's cells are (0.16 x 0.64,
    0.16 x 0.36, 0.84 x 0.64, 0.84 x 0.36), and they match the binned-sample default
    within 3 Monte Carlo standard errors (exact for a zero cell). A payload's `cell`
    reads its bits the same way, a zero payload in cell 0. A row's log likelihood is
    the sum of log d_k over the slots seen and log (1 - d_k) over the rest: a
    detection of slot 0 alone is log (0.84 x 0.64), log 0.36 and impossible.

    Fails if a slot's probability ignores its HZ bit, an empty slot or the exposure,
    if `cell` and `cells` order the joint index differently, or the likelihood
    misreads a slot not seen.
    """
    exists = [[True, True], [True, False], [False, False]]
    atoms = toy.atoms(
        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0] * 2] * 3, [1 / 3] * 3,
        exists=exists,
    )  # fmt: skip
    model = DetectionModel(0.6, 0.2)
    exact = model.cells(atoms, _action(2.0))
    d0, d1 = 0.84, 0.36
    assert_allclose(
        exact[0], [(1 - d0) * (1 - d1), (1 - d0) * d1, d0 * (1 - d1), d0 * d1],
        rtol=1e-12,
    )  # fmt: skip
    assert_allclose(exact[1], [1 - d1, 0.0, d1, 0.0], rtol=1e-12, atol=1e-15)
    assert_allclose(exact[2], [1.0, 0.0, 0.0, 0.0], atol=1e-15)
    assert_allclose(model.cells(atoms, _action(0.0))[:, 0], 1.0)
    binned = binned_cells(model, atoms, _action(2.0), model.cell, 4)
    se = np.sqrt(exact * (1.0 - exact) / CELL_SAMPLES)
    assert np.all(np.abs(binned - exact) <= 3.0 * se)

    def payload(bits):
        return {
            "exposure_d": jnp.asarray(2.0),
            "p_hz_one_day": jnp.asarray(0.6),
            "p_other_one_day": jnp.asarray(0.2),
            "detected": jnp.asarray(bits, float),
        }

    cells = [int(model.cell(payload(b))) for b in ([0, 0], [0, 1], [1, 0], [1, 1])]
    assert cells == [0, 1, 2, 3]
    assert int(model.cell(jax.tree.map(jnp.zeros_like, payload([1, 1])))) == 0
    like = row_log_like(model, payload([1, 0]), atoms, toy.star)
    assert_allclose(like, [np.log(d0 * (1 - d1)), np.log(d1), -np.inf], rtol=1e-12)


def test_a_detection_no_draw_tracks_is_not_scored(toy):
    """A detection in a slot no draw holds is a new planet: `stats` leaves it
    unscored, and `detections` reports it.

    Draws (HZ, empty) and (other, empty), a two-day search (d = 0.84 and 0.36 in slot
    0): slot 1 is held by no draw (`planet_tracks` is (True, False)). A row that
    detects both slots scores slot 0 alone: log 0.84 and log 0.36.

    Fails if a new detection is scored against the draws (each would read it as
    impossible), if `planet_tracks` misreads a slot, or if `detections` misreads the
    detected bits.
    """
    exists = [[True, False], [True, False]]
    atoms = toy.atoms(
        [[1.0, 0.0], [0.0, 0.0]], [[0.0] * 2] * 2, [0.5] * 2, exists=exists
    )
    model = DetectionModel(0.6, 0.2)
    payload = {
        "exposure_d": jnp.asarray(2.0),
        "time_d": jnp.asarray(0.0),
        "p_hz_one_day": jnp.asarray(0.6),
        "p_other_one_day": jnp.asarray(0.2),
        "detected": jnp.asarray([1.0, 1.0]),
    }
    assert list(np.asarray(planet_tracks(atoms))) == [True, False]
    like = row_log_like(model, payload, atoms, toy.star)
    assert_allclose(like, np.log([0.84, 0.36]), rtol=1e-12)
    assert list(np.asarray(model.detections(payload))) == [True, True]


def test_step_report(toy):
    """Nothing below S/N 5; at or above it, the bit of each gas above its floor.

    Draw 0 has one gas above its floor (0.3) and one below (-0.3); draw 1 has two
    positive abundances (0.2, 1.5). At S/N 4.9 no bit is reported and the log likelihood
    is zero for every draw; at S/N 5 the bits are (1, 0) and (1, 1). A gas block with a
    concrete nonzero variance is refused by every method, since two bits of one
    Gaussian gas are not independent.

    Fails if the gate is applied on the wrong side of 5, a bit ignores its floor or
    reads magnitude, a gas below the gate still informs the likelihood, or the model
    accepts a Gaussian gas block.
    """
    atoms = toy.atoms([[1.0]] * 2, [[1.0]] * 2, [0.5] * 2, [[0.3, -0.3], [0.2, 1.5]], 0)
    key = jax.random.key(0)
    below = StepReportModel(snr_one_day=[4.9, 4.9])
    at = StepReportModel(snr_one_day=[5.0, 5.0])

    silent = below.sample(key, atoms, _action())
    assert_allclose(silent["bit"], 0.0)
    assert_allclose(row_log_like(below, _first(silent), atoms, toy.star), [0.0, 0.0])
    gated = _expected_row([4.9, 4.9], 1.0, "bit")
    assert_allclose(row_log_like(below, gated, atoms, ()), 0.0)

    bits = at.sample(key, atoms, _action())["bit"][:, 0, :]
    assert_allclose(bits, [[1.0, 0.0], [1.0, 1.0]])
    # The report of draw 0 is certain under draw 0 and impossible under draw 1.
    payload = _first(at.sample(key, atoms, _action()))
    assert_allclose(row_log_like(at, payload, atoms, toy.star), [0.0, -np.inf])

    fuzzy = toy.atoms([[1.0]], [[1.0]], [1.0], 0.4, 1.0, n_gas=1)
    with pytest.raises(RuntimeError, match="point gas"):
        row_log_like(at, _expected_row([5.0], 1.0, "bit"), fuzzy, toy.star)
    with pytest.raises(ValueError, match="point gas"):
        at.sample(key, fuzzy, _action())
    with pytest.raises(ValueError, match="point gas"):
        at.cells(fuzzy, _action())


def test_gas_model_tends_to_the_step_rule(toy):
    """At high S/N the Gaussian reading's P(gas) tends to the step rule's bit.

    The datum sits at its expected value y = u for heights u = 0.05, 0.2 and 1 dex above
    the floor and -0.2 below it, under the verdict's prior N(0, 1). P(gas) never moves
    away from the step bit as S/N grows through 5, 50, 500 and 10^4, and at 10^4 it is
    within 1e-6 of the bit the step report gives.

    Fails if the posterior moves away from the reading (a sign error in the gain), or
    the gas model and the step report disagree on which side of the floor a gas is.
    """
    heights = [0.05, 0.2, 1.0, -0.2]
    prior = _prior(toy, n_gas=1)
    above = GasAbove(0, GAS_FLOOR)
    model = GaussianGasModel(snr_one_day=[1.0])
    step = StepReportModel(snr_one_day=[1.0e4])
    for u in heights:
        p = [
            above.prob(
                row_update(model, _expected_row([s], u), prior, toy.star), toy.star
            )[0, 0]
            for s in (5.0, 50.0, 500.0, 1.0e4)
        ]
        truth = toy.atoms([[1.0]], [[1.0]], [1.0], u, 0.0, n_gas=1)
        bit = step.sample(jax.random.key(0), truth, _action())["bit"][0, 0, 0]
        distance = np.abs(np.asarray(p) - float(bit))
        assert np.all(np.diff(distance) <= 0) and distance[0] > 1e-3
        assert distance[-1] < 1e-6


def test_gaussian_likelihood_and_update_describe_one_model(toy):
    """Weighting prior draws by `row_log_like` gives the conjugate `row_update`.

    Point draws at the 20,000 midpoint quantiles of N(0, 1), one reading y = 1 at S/N 5
    (sigma = 0.5): the weighted mean and variance equal the update's 0.8 and 0.2. A
    Gaussian draw scored by `row_log_like` gives the marginal N(1 | 0, 1.25).

    Fails if the likelihood uses sigma for the variance, drops the draw's own variance,
    or the update's gain disagrees with the likelihood it claims to integrate.
    """
    n = 20_000
    x = ndtri((jnp.arange(n) + 0.5) / n)
    ones = jnp.ones((n, 1))
    points = toy.atoms(ones, ones, jnp.ones(n), x[:, None], 0.0, n_gas=1)
    model = GaussianGasModel(snr_one_day=[5.0])
    row = _expected_row([5.0], 1.0)
    w = jax.nn.softmax(row_log_like(model, row, points, toy.star))
    mean = jnp.sum(w * x)
    var = jnp.sum(w * (x - mean) ** 2)
    posterior = row_update(model, row, _prior(toy, n_gas=1), toy.star)
    assert_allclose(posterior.x_mean[0, 0, 0], 0.8, rtol=1e-12)
    assert_allclose(posterior.x_var[0, 0, 0], 0.2, rtol=1e-12)
    assert_allclose([mean, var], [0.8, 0.2], rtol=1e-3)
    marginal = -0.5 * (1.0 / 1.25 + np.log(2 * np.pi * 1.25))
    assert_allclose(row_log_like(model, row, _prior(toy, n_gas=1), ()), [marginal])


def test_two_readings_of_an_uncertain_gas_are_exact(toy):
    """Two rows reading one Gaussian gas give the exact joint marginal and posterior.

    Prior x ~ N(0, 1); readings y1 = 1 at S/N 5 (sigma 0.5, precision 4) and y2 = 0.4
    at S/N 2.5 (sigma 1, precision 1). Jointly (y1, y2) ~ N(0, [[1.25, 1], [1, 2]]),
    log density -2.50728; the posterior has precision 1 + 4 + 1 = 6, mean
    (4 + 0.4) / 6 = 0.7333 and variance 1 / 6. The sum of the two rows' marginals,
    log N(1 | 0, 1.25) + log N(0.4 | 0, 2) = -2.73602, counts the shared abundance
    twice. Summed statistics in either order give the same result.

    Fails if a model sums per-row log likelihoods on an uncertain gas, the conjugate
    statistics drop a row's precision or reading, or the result depends on row order.
    """
    model = GaussianGasModel(snr_one_day=[1.0])
    prior = _prior(toy, n_gas=1)
    rows = [_expected_row([5.0], 1.0), _expected_row([2.5], 0.4)]
    stats = [model.stats(r, prior, toy.star) for r in rows]
    both = jax.tree.map(jnp.add, *stats)
    log_like, posterior = model.finalize(both, prior, toy.star)

    cov = np.array([[1.25, 1.0], [1.0, 2.0]])
    y = np.array([1.0, 0.4])
    joint = -0.5 * (
        y @ np.linalg.solve(cov, y) + np.log(np.linalg.det(2 * np.pi * cov))
    )
    assert_allclose(joint, -2.50728, atol=1e-5)
    assert_allclose(log_like, [joint], rtol=1e-12)
    assert_allclose(posterior.x_mean[0, 0, 0], 4.4 / 6.0, rtol=1e-12)
    assert_allclose(posterior.x_var[0, 0, 0], 1.0 / 6.0, rtol=1e-12)

    swapped = model.finalize(jax.tree.map(jnp.add, *stats[::-1]), prior, toy.star)
    assert_array_max_ulp(swapped[0], log_like, maxulp=4)
    per_row = sum(row_log_like(model, r, prior, toy.star) for r in rows)
    assert_allclose(per_row, [-2.73602], atol=1e-5)
    assert np.abs(per_row - log_like)[0] > 0.2


def test_a_poisson_toy_prior_draws_its_own_density():
    """`ToyPrior` under `POISSON` draws each slot-existence pattern with the
    probability its `log_prob` gives it.

    Three slots: the track with probability 0.8, and two free slots holding a Poisson
    count of mean 0.5 lumped at 2, in a uniform subset. A pattern's probability is
    `exp(log_prob)` of a draw with that pattern, less each planet's own term (read off
    a one-slot prior). 4,000 draws: each of the 8 patterns' frequency is within 4
    standard errors of it (the largest, 0.485, has an error of 0.008), and the 8
    probabilities sum to one.

    Fails if the draws and the density disagree on the count's law (the tail not
    lumped at the free slots), on the subsets (free slots filled in order, which the
    density gives zero), or on the track's own probability.
    """
    star = {"p_hz": jnp.asarray(0.4)}
    args = ("toy", [0.8, 0.25, 0.25], 1.0, 0.5, GAS_FLOOR, GAS_PRIOR_SD)
    prior = ToyPrior(*args, n_gas=1, pool_law=POISSON)
    one_slot = ToyPrior("toy", [1.0], *args[2:], n_gas=1)
    exists = np.array([[(i >> (2 - k)) & 1 for k in range(3)] for i in range(8)], bool)
    zeros = jnp.zeros((8, 3))
    patterns = point_atoms(zeros, zeros, jnp.full((8, 3, 1), GAS_FLOOR), exists, prior)
    lone = point_atoms(zeros[:1, :1], zeros[:1, :1], jnp.full((1, 1, 1), GAS_FLOOR),
                       np.ones((1, 1), bool), one_slot)  # fmt: skip
    planet = one_slot.log_prob(lone, star)[0]
    expected = np.exp(prior.log_prob(patterns, star) - exists.sum(1) * planet)
    assert_allclose(expected.sum(), 1.0, rtol=1e-12)

    n = 4000
    drawn = np.asarray(prior.draws(jax.random.key(0), star, n).exists)
    index = drawn @ np.array([4, 2, 1])
    frequency = np.bincount(index, minlength=8) / n
    se = np.sqrt(expected * (1 - expected) / n)
    assert np.all(np.abs(frequency - expected) <= 4 * se + 1e-12)
