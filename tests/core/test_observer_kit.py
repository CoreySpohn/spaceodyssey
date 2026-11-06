"""The observer kit's four checks on the toy observers, with their negative controls."""

import dataclasses
import inspect

import equinox as eqx
import jax.numpy as jnp
import pytest

from spaceodyssey.core import handlers
from spaceodyssey.core.conformance import (
    calibration_size,
    check_calibration,
    check_declared_margins,
    check_folded_once,
    check_promotion_parity,
)
from tests.core.kit_helpers import (
    NOISE,
    LateTwice,
    LateTwiceSample,
    RefitMean,
    RefitSample,
    ResampleSample,
    first_worlds,
    kit_run,
    pit,
)

TOY = ("mean", "sample")
# Calibration margin on the PIT statistics: N = (4 sigma / 0.07)^2 = 817 PITs with
# sigma = 1/2; the 400 worlds give 1,600, so a calibrated observer fails with
# probability below 1e-3 and the control (central half near 0.36) cannot pass.
MARGIN = 0.07
PARITY_WORLDS = 8


@pytest.mark.parametrize("name", TOY)
def test_calibration_on_a_toy_prior(name):
    """Fails if the observer's posterior is biased, too narrow or too wide, or if the
    kit drops worlds of the batched run (all 1,600 PITs must count).
    """
    _, run, truths, _ = kit_run(name)
    result = check_calibration(run, truths, pit, margin=MARGIN)
    assert result, result.diagnostic
    assert result.detail["n_pits"] == 1600 >= calibration_size(MARGIN)


def test_calibration_control_an_observer_ignoring_half_the_records():
    """Fails if calibration passes an observer whose posterior claims records it
    ignored, or accepts fewer PITs than its margin needs.
    """
    _, run, truths, _ = kit_run("ignores_half")
    result = check_calibration(run, truths, pit, margin=MARGIN)
    assert not result
    assert "central half" in result.diagnostic
    assert result.detail["central half"] < 0.5 - MARGIN
    few, few_truths = first_worlds((run, truths), 100)
    with pytest.raises(ValueError, match="cannot resolve"):
        check_calibration(few, few_truths, pit, margin=MARGIN)


@pytest.mark.parametrize("name", TOY)
def test_no_record_is_folded_twice(name):
    """Fails if any world sets `dup_fold`, or a release delivered twice changes a
    belief or goes unflagged.
    """
    study, run, truths, roots = kit_run(name)
    result = check_folded_once(study, run, truths, roots)
    assert result, result.diagnostic


def _refolding_release(monkeypatch):
    """The release handler without its first-acceptance guard on the observer fold."""
    original = handlers.h_release
    source = inspect.getsource(original)
    guard = "to_observer & ~already & ~accepted.any()"
    old = f"belief = where_tree({guard}, folded, c.belief)"
    assert old in source
    namespace = dict(original.__globals__)
    new = source.replace(old, "belief = where_tree(to_observer, folded, c.belief)")
    exec(compile(new, "<refolding_release>", "exec"), namespace)
    monkeypatch.setattr(
        handlers,
        "HANDLERS",
        tuple(
            namespace["h_release"] if h is original else h for h in handlers.HANDLERS
        ),
    )


class Refolded(RefitMean):
    """A distinct observer type, so the patched handler is traced afresh."""


def test_folded_twice_control_the_dup_fold_mutants(monkeypatch):
    """Fails if the check passes a run whose `dup_fold` is set (the Slice 1 planted
    mutant) or a release handler that folds a second release again.
    """
    study, run, truths, roots = kit_run("mean")
    planted = dataclasses.replace(
        run,
        carry=eqx.tree_at(
            lambda c: c.dup_fold, run.carry, jnp.ones_like(run.carry.dup_fold)
        ),
    )
    result = check_folded_once(study, planted, truths, roots)
    assert not result and "dup_fold is set" in result.diagnostic
    _refolding_release(monkeypatch)
    mutant = dataclasses.replace(study, observer=Refolded())
    result = check_folded_once(mutant, run, truths, roots)
    assert not result and "changed the belief" in result.diagnostic


@pytest.mark.parametrize("name", TOY)
def test_promotion_parity_in_release_order(name):
    """Fails if a refit installed after any prefix of the folded rows differs from the
    incremental belief beyond the parity tolerances; or if the
    check stops reading rows in release order. The sampling observer keeps the draw
    of its latest fold, so only release order reproduces it: its canonical-order
    refit must differ (subjects 0 and 2 are released out of canonical order), while
    the running mean is order-insensitive.
    """
    study, run, _, roots = kit_run(name)
    result = check_promotion_parity(study, first_worlds(run, PARITY_WORLDS), roots)
    assert result, result.diagnostic
    assert result.detail["n_checked"] == 8 * PARITY_WORLDS
    assert result.detail["canonical_equal"] is (name == "mean")


@pytest.mark.parametrize(
    ("name", "observer", "message"),
    [
        ("mean", LateTwice(), "after 1 row,"),
        ("sample", LateTwiceSample(noise=NOISE), "after 1 row,"),
        ("mean", RefitMean(ignore_odd=True), "do not reproduce the run"),
    ],
    ids=["exact", "statistical", "unanchored"],
)
def test_promotion_parity_control_a_late_record_weighted_twice(name, observer, message):
    """Fails if promotion parity passes a refit that weights its latest record twice,
    for an exact observer (parity tolerances) or a statistical one (integer leaves),
    or judges an observer whose folds do not reproduce the run it is given.
    """
    study, run, _, roots = kit_run(name)
    run = first_worlds(run, PARITY_WORLDS)
    result = check_promotion_parity(study, run, roots, observer=observer)
    assert not result
    assert message in result.diagnostic


def test_declared_margins():
    """Fails if an exact observer is asked for a margin, or a statistical observer
    with complete declarations is refused or judged outside its margin.
    """
    for observer in (RefitMean(), RefitSample()):
        result = check_declared_margins(observer)
        assert result and not result.detail["statistical"]
    statistical = ResampleSample(noise=NOISE)
    assert check_declared_margins(statistical).detail["margin"] == 0.08
    study, run, _, roots = kit_run("sample")
    result = check_promotion_parity(study, run, roots, observer=statistical)
    assert result, result.diagnostic
    assert result.detail["statistical"]
    assert result.detail["observed_se"] <= ResampleSample.standard_error


class Undeclared(RefitSample):
    """A statistical observer that declares nothing."""

    statistical = True


class Undersized(ResampleSample):
    """A margin narrower than four declared standard errors."""

    margin = 0.05


class ExactWithMargin(RefitMean):
    """An exact observer that declares a margin anyway."""

    margin = 0.1


def test_declared_margins_control_missing_or_inconsistent():
    """Fails if the kit judges an observer without a margin, seeds and standard error,
    one sized below four standard errors, or an exact one that declares a margin.
    """
    study, run, _, roots = kit_run("sample")
    for observer in (Undeclared(), Undersized(), ExactWithMargin()):
        assert not check_declared_margins(observer)
        with pytest.raises(ValueError):
            check_promotion_parity(
                study, first_worlds(run, 2), roots, observer=observer
            )
