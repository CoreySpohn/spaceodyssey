"""The observer kit's four checks on the toy observers, with their negative controls."""

import dataclasses
import inspect
import math

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
    t_quantile,
)
from tests.core.kit_helpers import (
    N_WORLDS,
    NOISE,
    MeanMutant,
    RefitMean,
    RefitSample,
    ResampleSample,
    SampleMutant,
    first_worlds,
    kit_batch,
    kit_run,
    pit_of_mean,
    pit_of_sample,
)

TOY = ("mean", "sample")
# Calibration margin on the Kolmogorov distance of the PITs from uniform. Four PIT
# slots need calibration_size(0.075, 4) = 1,805 worlds for a slack of half the
# margin; the batch has 2,000 (slack 0.036). A calibrated observer fails only if the
# empirical distance of its 8,000 PITs exceeds 0.039 (probability near 1e-10); the
# control that ignores half its records sits at 0.083, outside the margin itself.
MARGIN = 0.075
PITS = {"mean": pit_of_mean, "sample": pit_of_sample(NOISE)}


@pytest.fixture(scope="module", autouse=True)
def batch():
    """The shared 2,000-world batch, built (and replay-checked) before any test."""
    return kit_batch()


@pytest.mark.parametrize("name", TOY)
def test_calibration_on_a_toy_prior(name):
    """Fails if the observer's posterior (its mean and count, or its posterior draws
    under its declared noise) is biased, too narrow or too wide, or if the kit drops
    worlds of the batched run.
    """
    _, run, truths, _ = kit_run(name)
    result = check_calibration(run, truths, PITS[name], margin=MARGIN)
    assert result, result.diagnostic
    assert result.detail["n_pits"] == 4 * N_WORLDS


def _quantized(belief, truth):
    """PITs moved to the nearest of 0.1, 0.4, 0.6, 0.9: equal mass, not uniform."""
    pits, counted = pit_of_mean(belief, truth)
    levels = jnp.asarray([0.1, 0.4, 0.6, 0.9])
    return levels[jnp.clip((pits * 4).astype(int), 0, 3)], counted


@pytest.mark.parametrize(
    ("name", "pit", "message"),
    [
        ("ignores_half", pit_of_mean, "from uniform"),
        ("nan_sample", pit_of_sample(NOISE), "not finite"),
        ("wrong_noise", pit_of_sample(0.2), "from uniform"),
        ("mean", _quantized, "from uniform"),
    ],
    ids=["ignores_half", "nan_samples", "wrong_noise", "four_point_mass"],
)
def test_calibration_controls(name, pit, message):
    """Fails if calibration passes an observer that counts records it ignored, one
    whose posterior draws are NaN or drawn with a wrong declared noise, or PITs with
    equal mass at four points (their mean and central-half fraction are uniform's).
    """
    _, run, truths, _ = kit_run(name)
    result = check_calibration(run, truths, pit, margin=MARGIN)
    assert not result
    assert message in result.diagnostic


def _shared(belief, truth):
    """Subject 0's PIT copied to all four slots: calibrated but fully correlated."""
    pits, counted = pit_of_mean(belief, truth)
    return jnp.broadcast_to(pits[0], pits.shape), counted


def test_calibration_slack_counts_worlds_not_pits():
    """Fails if the slack treats the PITs of one world as independent (it must be
    built from per-slot world counts), or accepts too few worlds for the margin.
    """
    _, run, truths, _ = kit_run("mean")
    result = check_calibration(run, truths, _shared, margin=MARGIN)
    assert result, result.diagnostic
    want = math.sqrt(math.log(2 * 4 / 0.05) / (2 * N_WORLDS))
    assert result.detail["slack"] == pytest.approx(want, rel=1e-12)
    assert calibration_size(MARGIN, 4) <= N_WORLDS
    few, few_truths = first_worlds((run, truths), 1000)
    with pytest.raises(ValueError, match="cannot resolve"):
        check_calibration(few, few_truths, pit_of_mean, margin=MARGIN)


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
def test_promotion_parity(name):
    """Fails if a refit installed after any prefix of the folded rows differs from the
    incremental belief, in release order or as `refit_rows` selects the rows, or if
    `install` misses its subject or touches another. The sampling observer keeps
    the draw of its latest fold: its release-order refit must match exactly and its
    canonical refit must not (subjects 0 and 2 are released out of canonical
    order), so a change of order in either path fails here.
    """
    study, run, _, roots = kit_run(name)
    result = check_promotion_parity(study, run, roots)
    assert result, result.diagnostic
    assert result.detail["n_checked"] == 8 * N_WORLDS
    assert result.detail["canonical_equal"] is (name == "mean")


PARITY_CONTROLS = [
    ("mean", MeanMutant(late_twice=True), "release order"),
    ("mean", MeanMutant(noop_install=True), "did not take"),
    ("mean", MeanMutant(noop_install=True, late_twice=True), "did not take"),
    ("mean", MeanMutant(canonical_bug=True), "refit_rows"),
    ("mean", MeanMutant(ignore_odd=True), "do not reproduce the run"),
    ("sample", SampleMutant(noise=NOISE, late_twice=True), "n differs"),
    ("sample", SampleMutant(noise=NOISE, nan_refit=True), "sample differs"),
    ("sample", SampleMutant(noise=NOISE, cancelling=True), "not inside"),
    ("sample", SampleMutant(noise=NOISE, corrupt_install=True), "changed the"),
]


@pytest.mark.parametrize(
    ("name", "observer", "message"),
    PARITY_CONTROLS,
    ids=[
        "late_twice",
        "noop_install",
        "noop_install_late_twice",
        "canonical_only_bug",
        "unanchored",
        "statistical_late_twice",
        "statistical_nan",
        "statistical_cancelling",
        "statistical_corrupt_install",
    ],
)
def test_promotion_parity_controls(name, observer, message):
    """Fails if promotion parity passes a refit that folds its latest record twice
    (exact, or statistical through its counts), an install that does nothing (even
    beside a doubled record) or corrupts another subject, a refit wrong only in
    canonical order, NaN draws, errors that cancel across subjects, or an observer
    whose folds do not reproduce the run.
    """
    study, run, _, roots = kit_run(name)
    result = check_promotion_parity(study, run, roots, observer=observer)
    assert not result
    assert message in result.diagnostic


def test_declared_margins():
    """Fails if an exact observer is asked for a margin, a declared statistical or
    order-sensitive observer is refused or judged outside its margin, or the t
    quantile is off.
    """
    result = check_declared_margins(RefitMean())
    assert result and not result.detail["judged_in_margin"]
    assert check_declared_margins(RefitSample()).detail["margin"] == 0.1
    statistical = ResampleSample(noise=NOISE)
    assert check_declared_margins(statistical).detail["seeds"] == (0, 1)
    study, run, _, roots = kit_run("sample")
    result = check_promotion_parity(study, run, roots, observer=statistical)
    assert result, result.diagnostic
    assert result.detail["release order worst"] < statistical.margin
    assert t_quantile(9) == pytest.approx(2.2622, abs=1e-3)
    assert t_quantile(1999) == pytest.approx(1.9612, abs=1e-4)


class Undeclared(RefitSample):
    """A statistical observer that declares nothing."""

    statistical = True
    margin = None
    seeds = ()
    standard_error = None


class Undersized(ResampleSample):
    """A margin narrower than four declared standard errors."""

    margin = 0.05


class ExactWithMargin(RefitMean):
    """An exact observer that declares a margin anyway."""

    margin = 0.1


def test_declared_margins_control_missing_inconsistent_or_too_few_worlds():
    """Fails if the kit judges an observer without a margin, seeds and standard error,
    one sized below four standard errors, or an exact one that declares a margin; or
    judges a margin over fewer than ten worlds.
    """
    study, run, _, roots = kit_run("sample")
    for observer in (Undeclared(), Undersized(), ExactWithMargin()):
        assert not check_declared_margins(observer)
        with pytest.raises(ValueError):
            check_promotion_parity(study, run, roots, observer=observer)
    few, few_roots = first_worlds((run, roots), 5)
    result = check_promotion_parity(
        study, few, few_roots, observer=ResampleSample(noise=NOISE)
    )
    assert not result and "10 needed" in result.diagnostic
