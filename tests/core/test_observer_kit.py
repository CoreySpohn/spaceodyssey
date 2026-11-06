"""The observer kit's declarations and refusals, which need no run.

Each refusal comes before the kit reads a run, so these pass `None` for it. The
checks on runs are in `test_observer_kit_runs.py` (four host worlds) and
`test_observer_kit_statistics.py` (2,000 compiled worlds), both slow: one executor
compile alone is over the unit budget.
"""

import math

import numpy as np
import pytest

from spaceodyssey.core.conformance import (
    ParityMargin,
    calibration_size,
    check_calibration,
    check_declared_margins,
    check_promotion_parity,
    check_refit_calibration,
    declared_margin,
    t_quantile,
)
from spaceodyssey.core.conformance.observer_kit import _stratum_failures
from tests.core.kit_helpers import (
    NOISE,
    OBSERVERS,
    RefitMean,
    RefitSample,
    ResampleSample,
    kit_study,
    pit_of_mean,
)

STUDY = kit_study(OBSERVERS["mean"])


class Unowned(RefitMean):
    """A refit-capable observer that does not say which leaves a subject owns."""

    subject_leaves = None


class MisOwned(RefitMean):
    """An observer that claims a leaf its belief does not have."""

    subject_leaves = ("mean", "spread")


class ExactStochastic(RefitMean):
    """An exact observer that declares a stochastic leaf."""

    stochastic_leaves = ("mean",)


class IntStochastic(RefitSample):
    """A stochastic leaf that is not floating point."""

    stochastic_leaves = ("n",)


@pytest.mark.parametrize(
    ("observer", "message"),
    [
        (Unowned(), "subject_leaves"),
        (MisOwned(), "'spread'"),
        (ExactStochastic(), "stochastic leaves"),
        (IntStochastic(noise=NOISE), "'n'"),
    ],
    ids=["unowned", "mis_owned", "exact_stochastic", "int_stochastic"],
)
def test_promotion_parity_refuses_undeclared_leaf_roles(observer, message):
    """Fails if parity judges an observer that does not name its subject-owned
    leaves, names one its belief lacks, or declares stochastic leaves while
    exact or on an integer leaf.
    """
    with pytest.raises(ValueError, match=message):
        check_promotion_parity(STUDY, None, None, observer=observer)


def test_declared_margins():
    """Fails if an exact observer is asked for a margin, a declared statistical or
    order-sensitive observer is refused, or the t quantile is off.
    """
    result = check_declared_margins(RefitMean())
    assert result and not result.detail["judged_in_margin"]
    assert check_declared_margins(RefitSample()).detail["margin"] == 0.1
    assert check_declared_margins(ResampleSample()).detail["seeds"] == (0, 1)
    assert t_quantile(9) == pytest.approx(2.2622, abs=1e-3)
    assert t_quantile(1999) == pytest.approx(1.9612, abs=1e-4)


class Undeclared(RefitSample):
    """A statistical observer that declares nothing."""

    statistical = True
    margin = None
    seeds = ()
    standard_error = None
    stochastic_leaves = ()


class Undersized(ResampleSample):
    """A margin narrower than four declared standard errors."""

    margin = 0.05


class ExactWithMargin(RefitMean):
    """An exact observer that declares a margin anyway."""

    margin = 0.1


class NoStochastic(ResampleSample):
    """A statistical observer that names no stochastic leaf."""

    stochastic_leaves = ()


class FractionalSeeds(ResampleSample):
    """Seeds that truncate to distinct integers but are not integers."""

    seeds = (0.5, 1.5)


class WrappedSeeds(ResampleSample):
    """Seeds whose uint32 fold-in keys coincide."""

    seeds = (0, 2**32)


class BoolSeeds(ResampleSample):
    """Booleans are not seeds."""

    seeds = (False, True)


class NanMargin(ResampleSample):
    """A margin every interval passes."""

    margin = float("nan")


MARGIN_CONTROLS = (
    Undeclared,
    Undersized,
    ExactWithMargin,
    NoStochastic,
    FractionalSeeds,
    WrappedSeeds,
    BoolSeeds,
    NanMargin,
)


@pytest.mark.parametrize("cls", MARGIN_CONTROLS, ids=lambda c: c.__name__)
def test_declared_margins_controls(cls):
    """Fails if the kit judges an observer without a margin, seeds, standard error
    and stochastic leaves; one sized below four standard errors; an exact one that
    declares a margin; seeds that are not integers in [0, 2^32) (0.5 and 1.5, or 0
    and 2^32, which fold into the same key); or a NaN margin.
    """
    observer = cls(noise=NOISE) if issubclass(cls, RefitSample) else cls()
    assert not check_declared_margins(observer)
    with pytest.raises(ValueError):
        declared_margin(observer)
    with pytest.raises(ValueError):
        check_promotion_parity(STUDY, None, None, observer=observer)


def test_margin_interval_at_its_boundary():
    """Fails if a stratum's interval uses the normal 1.96 instead of the t quantile
    on its worlds' degrees of freedom, or takes the degrees of freedom from every
    seed of every world: ten worlds of twenty seeds, a mean difference of 0.055 and
    a declared standard error of 0.02 give 0.055 + 2.262 x 0.02 = 0.1002, just
    outside a margin of 0.1 (1.96 or 199 degrees of freedom would pass).
    """
    n_worlds, n_seeds = 10, 20
    spread = np.where(np.arange(n_seeds) % 2 == 0, -0.01, 0.01)
    promoted = np.zeros((n_worlds, 1, n_seeds, 4))
    promoted[:, 0, :, 0] = 0.055 + spread
    incremental = np.zeros((n_worlds, 1, 4))
    subjects = np.zeros((n_worlds, 1), np.int64)
    valid = np.ones((n_worlds, 1), bool)
    declared = ParityMargin(0.1, tuple(range(n_seeds)), 0.02)
    failures, worst = _stratum_failures(
        "path", promoted, incremental, subjects, valid, declared
    )
    assert worst == pytest.approx(0.055 + t_quantile(9) * 0.02, rel=1e-9)
    assert failures and "not inside" in failures[0]
    assert 0.055 + 1.96 * 0.02 < 0.1 and 0.055 + t_quantile(199) * 0.02 < 0.1


@pytest.mark.parametrize("margin", [math.nan, 0.0, -0.1, math.inf])
def test_calibration_margin_must_be_positive_and_finite(margin):
    """Fails if calibration (or its sizing, or refit calibration) accepts a NaN,
    zero, negative or infinite margin, which would pass arbitrary PITs.
    """
    with pytest.raises(ValueError, match="positive and finite"):
        check_calibration(None, None, pit_of_mean, margin=margin)
    with pytest.raises(ValueError, match="positive and finite"):
        calibration_size(margin, 4)
    with pytest.raises(ValueError, match="positive and finite"):
        check_refit_calibration(STUDY, None, None, None, pit_of_mean, margin=margin)
