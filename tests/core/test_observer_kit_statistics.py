"""The observer kit's judged checks on the 2,000-world compiled batch (slow tier).

Calibration and margins need thousands of worlds; the compiled executor's first
compile for that batch alone is over the unit budget, so this file is slow. The
exact checks run in the quick tier on a host batch (`test_observer_kit.py`).
"""

import math

import jax.numpy as jnp
import pytest

from spaceodyssey.core.conformance import (
    calibration_size,
    check_calibration,
    check_promotion_parity,
    check_refit_calibration,
)
from tests.core.kit_helpers import (
    N_WORLDS,
    NOISE,
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
    return kit_batch("compiled")


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


def test_promotion_parity_order_sensitive():
    """Fails if the sampling observer's refit, installed after any prefix, differs
    from the incremental belief: exactly in release order, and in canonical order
    exactly in every deterministic leaf and within its margin in its draw (subjects
    0 and 2 are released out of canonical order, so the draws differ there).
    """
    study, run, _, roots = kit_run("sample")
    result = check_promotion_parity(study, run, roots)
    assert result, result.diagnostic
    assert result.detail["n_checked"] == 8 * N_WORLDS
    assert not result.detail["canonical_equal"]
    assert result.detail["refit_rows worst"] < study.observer.margin


def test_promotion_parity_statistical():
    """Fails if a refit that redraws its subject's sample from the refit key is
    judged outside its margin, averaged over its two declared seeds.
    """
    statistical = ResampleSample(noise=NOISE)
    study, run, _, roots = kit_run("sample")
    result = check_promotion_parity(study, run, roots, observer=statistical)
    assert result, result.diagnostic
    assert result.detail["release order worst"] < statistical.margin


def _mutant(**faults):
    return SampleMutant(noise=NOISE, **faults)


PARITY_CONTROLS = {
    "late_twice": (_mutant(late_twice=True), "n differs"),
    "nan_refit": (_mutant(nan_refit=True), "sample differs"),
    "cancelling": (_mutant(cancelling=True), "not inside"),
    "corrupt_install": (_mutant(corrupt_install=True), "changed another"),
    "canonical_mean": (_mutant(canonical_mean=True), "refit_rows"),
    "stale_install": (_mutant(stale_install=True), "did not take the refit's mean"),
    "leak_install": (_mutant(leak_install=True), "changed another subject's mean"),
}


@pytest.mark.parametrize("control", PARITY_CONTROLS)
def test_promotion_parity_controls(control):
    """Fails if promotion parity passes, for a statistical refit: a doubled record
    (seen in its counts), NaN draws, errors that cancel across subjects, an install
    that corrupts another subject's draw, a deterministic mean wrong by 100 in
    canonical order only, an install that leaves the subject's mean stale, or one
    that adds 100 to another subject's mean.
    """
    observer, message = PARITY_CONTROLS[control]
    study, run, _, roots = kit_run("sample")
    result = check_promotion_parity(study, run, roots, observer=observer)
    assert not result
    assert message in result.diagnostic


@pytest.mark.parametrize("name", ["statistical", "order_sensitive"])
def test_refit_calibration(name):
    """Fails if correct refits are refused at either row depth or if prefix PITs
    lose worlds or treat subjects of one world as independent, under each seed:
    a fresh posterior draw, or the canonical-order draw of the latest fold.
    """
    study, run, truths, roots = kit_run("sample")
    observer = ResampleSample(noise=NOISE) if name == "statistical" else None
    result = check_refit_calibration(
        study,
        run,
        truths,
        roots,
        pit_of_sample(NOISE),
        margin=MARGIN,
        observer=observer,
    )
    assert result, result.diagnostic
    assert len(result.detail) == (2 if name == "statistical" else 1)
    for depths in result.detail.values():
        assert set(depths) == {"after 1 of its rows", "after 2 of its rows"}
        for detail in depths.values():
            assert detail["n_pits"] == 4 * N_WORLDS
            assert detail["slack"] == pytest.approx(
                math.sqrt(math.log(2 * 4 / 0.05) / (2 * N_WORLDS)), rel=1e-12
            )


@pytest.mark.parametrize("fault", ["wide_refit", "one_row_wide_refit"])
def test_refit_calibration_control_a_refit_twice_as_wide(fault):
    """Fails if refit calibration passes draws of twice the posterior's standard
    deviation: their mean matches the incremental draw, so parity alone can pass
    them (its worst bound near 0.065 is inside the 0.1 margin), even when the
    refit is correct after two rows and wrong only after one.
    """
    study, run, truths, roots = kit_run("sample")
    wide = _mutant(**{fault: True})
    parity = check_promotion_parity(study, run, roots, observer=wide)
    assert parity, parity.diagnostic
    result = check_refit_calibration(
        study, run, truths, roots, pit_of_sample(NOISE), margin=MARGIN, observer=wide
    )
    assert not result and "from uniform" in result.diagnostic
    if fault == "one_row_wide_refit":
        assert "after 1 of its rows" in result.diagnostic
        assert "after 2 of its rows" not in result.diagnostic
