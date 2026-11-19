"""Closed-form tests of the toy models: cells, the step report, the gas limit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import ndtri
from numpy.testing import assert_allclose, assert_array_max_ulp

from spaceodyssey.claim import GasAbove
from spaceodyssey.toy import (
    GAS_FLOOR,
    GAS_PRIOR_SD,
    GaussianGasModel,
    StepReportModel,
)
from spaceodyssey.types import CELL_SAMPLES, AbstractObservationModel, Action


def _action(exposure_d=1.0):
    return Action(kind=0, star=0, mode=0, exposure_d=exposure_d)


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
        binned = AbstractObservationModel.cells(model, atoms, _action(exposure))
        for p in (exact, binned):
            assert_array_max_ulp(p.sum(axis=1), np.ones(4), maxulp=4)
        se = np.sqrt(exact * (1.0 - exact) / CELL_SAMPLES)
        assert np.all(np.abs(binned - exact) <= 3.0 * se)


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

    silent = below.sample(atoms, _action(), key)
    assert_allclose(silent["bit"], 0.0)
    assert_allclose(below.log_like(_first(silent), atoms, toy.star), [0.0, 0.0])
    gated = _expected_row([4.9, 4.9], 1.0, "bit")
    assert_allclose(below.log_like(gated, atoms, ()), 0.0)

    bits = at.sample(atoms, _action(), key)["bit"][:, 0, :]
    assert_allclose(bits, [[1.0, 0.0], [1.0, 1.0]])
    # The report of draw 0 is certain under draw 0 and impossible under draw 1.
    payload = _first(at.sample(atoms, _action(), key))
    assert_allclose(at.log_like(payload, atoms, toy.star), [0.0, -np.inf])

    fuzzy = toy.atoms([[1.0]], [[1.0]], [1.0], 0.4, 1.0, n_gas=1)
    with pytest.raises(ValueError, match="point gas"):
        at.log_like(_expected_row([5.0], 1.0, "bit"), fuzzy, toy.star)
    with pytest.raises(ValueError, match="point gas"):
        at.sample(fuzzy, _action(), key)
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
            above.prob(model.update(_expected_row([s], u), prior), toy.star)[0, 0]
            for s in (5.0, 50.0, 500.0, 1.0e4)
        ]
        truth = toy.atoms([[1.0]], [[1.0]], [1.0], u, 0.0, n_gas=1)
        bit = step.sample(truth, _action(), jax.random.key(0))["bit"][0, 0, 0]
        distance = np.abs(np.asarray(p) - float(bit))
        assert np.all(np.diff(distance) <= 0) and distance[0] > 1e-3
        assert distance[-1] < 1e-6


def test_gaussian_likelihood_and_update_describe_one_model(toy):
    """Weighting prior draws by `log_like` gives the conjugate `update`.

    Point draws at the 20,000 midpoint quantiles of N(0, 1), one reading y = 1 at S/N 5
    (sigma = 0.5): the weighted mean and variance equal the update's 0.8 and 0.2. A
    Gaussian draw scored by `log_like` gives the marginal N(1 | 0, 1.25).

    Fails if `log_like` uses sigma for the variance, drops the draw's own variance, or
    the update's gain disagrees with the likelihood it claims to integrate.
    """
    n = 20_000
    x = ndtri((jnp.arange(n) + 0.5) / n)
    ones = jnp.ones((n, 1))
    points = toy.atoms(ones, ones, jnp.ones(n), x[:, None], 0.0, n_gas=1)
    model = GaussianGasModel(snr_one_day=[5.0])
    row = _expected_row([5.0], 1.0)
    w = jax.nn.softmax(model.log_like(row, points, toy.star))
    mean = jnp.sum(w * x)
    var = jnp.sum(w * (x - mean) ** 2)
    posterior = model.update(row, _prior(toy, n_gas=1))
    assert_allclose(posterior.x_mean[0, 0, 0], 0.8, rtol=1e-12)
    assert_allclose(posterior.x_var[0, 0, 0], 0.2, rtol=1e-12)
    assert_allclose([mean, var], [0.8, 0.2], rtol=1e-3)
    marginal = -0.5 * (1.0 / 1.25 + np.log(2 * np.pi * 1.25))
    assert_allclose(model.log_like(row, _prior(toy, n_gas=1), ()), [marginal])
