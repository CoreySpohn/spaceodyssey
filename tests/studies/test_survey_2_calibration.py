"""Survey 2, calibration of the conjugate posterior under the adaptive design."""

import dataclasses
import math

import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import CompiledExecutor
from spaceodyssey.core.conformance import check_calibration, make_worlds
from spaceodyssey.studies.survey_2 import (
    GaussianMean,
    GaussianSignals,
    gaussian_pit,
    survey_2,
)

pytestmark = pytest.mark.medium(
    reason="2000 worlds on one adaptive run shared by every test: 14 s on a laptop"
)

N_WORLDS = 2000
N_STEPS = 40
MARGIN = 0.075
PRIOR = {"prior_mean": 1.0, "prior_std": 0.3, "noise": 0.5}


@pytest.fixture(scope="module")
def sample():
    """Two thousand worlds drawn from the prior, run under the adaptive policy."""
    observer = GaussianMean(**PRIOR)
    study = dataclasses.replace(
        survey_2(observer=observer), world_source=GaussianSignals(1.0, 0.3)
    )
    truths, roots = make_worlds(study, N_WORLDS)
    run = CompiledExecutor().run(study, truths, roots, N_STEPS)
    return observer, truths, run


def per_subject(run, truths, pit):
    """Apply the calibration guard separately to each surveyed subject."""
    results = []
    for subject in range(2):

        def one(belief, truth, subject=subject):
            values, counted = pit(belief, truth)
            return values[subject : subject + 1], counted[subject : subject + 1]

        results.append(check_calibration(run, truths, one, margin=MARGIN))
    return results


def test_the_adaptive_design_took_data_dependent_samples(sample):
    """Fails if every world got the same samples, which would make the calibration
    claim a statement about a fixed design only.
    """
    _, _, run = sample
    assert bool(np.all(np.asarray(run.complete)))
    n = np.asarray(run.carry.belief.n)[:, :2]
    assert len(np.unique(n, axis=0)) > 3


def test_the_posterior_is_calibrated_under_the_adaptive_design(sample):
    """Fails if the posterior CDF at the true signal is not uniform within the kit's
    margin, over every subject that was sampled.
    """
    observer, truths, run = sample
    result = check_calibration(run, truths, gaussian_pit(observer), margin=MARGIN)
    assert result, result.diagnostic
    assert result.detail["n_pits"] > 3000
    for subject in per_subject(run, truths, gaussian_pit(observer)):
        assert subject, subject.diagnostic
        assert subject.detail["n_pits"] == N_WORLDS


def test_opposite_subject_errors_cannot_cancel_in_the_calibration_guard(sample):
    """Fails if pooling conceals opposite subject-level PIT errors.

    The increasing CDF transforms sqrt(F) and 1 - sqrt(1 - F) have PIT CDFs
    x^2 and 2x - x^2. Their equal mixture is uniform despite distance 0.25 per subject.
    """
    observer, truths, run = sample
    pit = gaussian_pit(observer)

    def canceling(belief, truth):
        values, counted = pit(belief, truth)
        return jnp.where(
            jnp.arange(values.size) == 0, jnp.sqrt(values), 1 - jnp.sqrt(1 - values)
        ), counted

    pooled = check_calibration(run, truths, canceling, margin=MARGIN)
    assert pooled, pooled.diagnostic
    subjects = per_subject(run, truths, canceling)
    assert all(not result for result in subjects)
    assert all(result.detail["distance"] > 0.2 for result in subjects)


@pytest.mark.parametrize(
    "wrong",
    [
        {"noise": 1.0},
        {"noise": 0.25},
        {"prior_std": 0.05},
        {"prior_std": 1.0},
    ],
    ids=["doubled-noise", "halved-noise", "narrow-prior", "wide-prior"],
)
def test_a_mistaken_posterior_is_not_calibrated(sample, wrong):
    """Fails if the check accepts a posterior with the wrong spread or centre."""
    _, truths, run = sample
    mistaken = GaussianMean(**{**PRIOR, **wrong})
    result = check_calibration(run, truths, gaussian_pit(mistaken), margin=MARGIN)
    assert not result


def test_a_posterior_centre_that_is_shifted_is_not_calibrated(sample):
    """Fails if the check accepts a posterior whose centre is off by half a prior
    standard deviation.
    """
    observer, truths, run = sample
    pit = gaussian_pit(observer)

    def shifted(belief, truth):
        return pit(belief, dataclasses.replace(truth, signal=truth.signal + 0.15))

    assert not check_calibration(run, truths, shifted, margin=MARGIN)


def test_a_posterior_that_ignores_the_prior_is_not_calibrated(sample):
    """Fails if the check accepts a near-flat prior spread, which is over-wide."""
    _, truths, run = sample
    flat = GaussianMean(**{**PRIOR, "prior_std": 1.0e6})
    result = check_calibration(run, truths, gaussian_pit(flat), margin=MARGIN)
    assert not result


@pytest.mark.parametrize("margin", [math.nan, math.inf, -0.1, 0.0])
def test_a_margin_that_is_not_positive_and_finite_is_refused(sample, margin):
    """Fails if the check runs with a NaN, infinite, negative or zero margin."""
    observer, truths, run = sample
    with pytest.raises(ValueError, match="margin"):
        check_calibration(run, truths, gaussian_pit(observer), margin=margin)


def test_a_nan_signal_is_not_calibrated(sample):
    """Fails if a NaN true signal passes as calibrated."""
    observer, truths, run = sample
    bad = dataclasses.replace(truths, signal=truths.signal.at[:, 0].set(np.nan))
    result = check_calibration(run, bad, gaussian_pit(observer), margin=MARGIN)
    assert not result
