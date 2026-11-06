"""The observer kit's exact checks on four host-executor worlds (slow tier).

The checks need a real run, and one executor compile alone (near 5 s cold) is over
the unit budget, so this file is slow. Declarations and refusals that need no run
are in `test_observer_kit.py`; calibration and margins, which need 2,000 worlds,
are in `test_observer_kit_statistics.py`.
"""

import dataclasses
import inspect

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import HostExecutor, handlers
from spaceodyssey.core.conformance import (
    check_folded_once,
    check_promotion_parity,
    replay_beliefs,
)
from tests.core.kit_helpers import (
    HOST_WORLDS,
    N_STEPS,
    NOISE,
    MeanMutant,
    ReadyMean,
    RefitMean,
    ResampleSample,
    first_worlds,
    kit_batch,
    kit_run,
    kit_study,
)

TOY = ("mean", "sample")


@pytest.fixture(scope="module", autouse=True)
def batch():
    """The shared four-world host batch, built (and replay-checked) before any test."""
    return kit_batch("host")


@pytest.mark.parametrize("name", TOY)
def test_no_record_is_folded_twice(name):
    """Fails if any world sets `dup_fold`, or a release delivered twice changes a
    belief or goes unflagged.
    """
    study, run, truths, roots = kit_run(name, "host")
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
    study, run, truths, roots = kit_run("mean", "host")
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


def test_replay_presents_records_as_the_handler_did():
    """Fails if `replay_beliefs` folds the final log rows (stamped after their
    fold) instead of the rows as the release handler presented them, for an
    observer that reads `accepted_tick`; or replays a log with a revised
    acquisition, whose earlier states the final log does not hold.
    """
    _, batch, truths, roots = kit_batch("host")
    study = kit_study(MeanMutant(tick_aux=True))  # shares the batch's programs
    one = first_worlds((truths, roots), 1)
    run = HostExecutor().run_worlds(study, *one, N_STEPS)
    replayed, _ = replay_beliefs(study, batch, roots)
    assert float(run.carry.belief.aux[0, 1]) == -8.0
    np.testing.assert_array_equal(replayed.aux[:1], run.carry.belief.aux)
    revised = eqx.tree_at(
        lambda c: c.log.revision,
        batch.carry,
        batch.carry.log.revision.at[:, 0].set(1),
    )
    with pytest.raises(ValueError, match="revised"):
        replay_beliefs(study, dataclasses.replace(batch, carry=revised), roots)


def test_promotion_parity_exact():
    """Fails if a refit installed after any prefix of the folded rows differs from the
    incremental belief, in release order or as `refit_rows` selects the rows, or if
    `install` misses its subject's owned state or touches another's.
    """
    study, run, _, roots = kit_run("mean", "host")
    result = check_promotion_parity(study, run, roots)
    assert result, result.diagnostic
    assert result.detail["n_checked"] == 8 * HOST_WORLDS
    assert result.detail["canonical_equal"]


def test_promotion_parity_refuses_stale_boolean_install():
    """Fails if install can leave a subject-owned boolean stale."""
    _, batch, _, roots = kit_run("mean", "host")
    observer = ReadyMean()
    study = kit_study(observer)
    belief, summary = replay_beliefs(study, batch, roots)
    run = dataclasses.replace(
        batch, carry=dataclasses.replace(batch.carry, belief=belief, pbelief=summary)
    )
    correct = check_promotion_parity(study, run, roots)
    assert correct, correct.diagnostic
    result = check_promotion_parity(
        study, run, roots, observer=ReadyMean(stale_ready=True)
    )
    assert not result
    assert "install did not take the refit's ready" in result.diagnostic


PARITY_CONTROLS = {
    "late_twice": (MeanMutant(late_twice=True), "release order"),
    "noop_install": (MeanMutant(noop_install=True), "did not take"),
    "noop_install_late_twice": (
        MeanMutant(noop_install=True, late_twice=True),
        "did not take",
    ),
    "canonical_only_bug": (MeanMutant(canonical_bug=True), "refit_rows"),
    "unanchored": (MeanMutant(ignore_odd=True), "do not reproduce the run"),
}


@pytest.mark.parametrize("control", PARITY_CONTROLS)
def test_promotion_parity_controls(control):
    """Fails if promotion parity passes a refit that folds its latest record twice, an
    install that does nothing (even beside a doubled record), a refit wrong only in
    canonical order, or an observer whose folds do not reproduce the run.
    """
    observer, message = PARITY_CONTROLS[control]
    study, run, _, roots = kit_run("mean", "host")
    result = check_promotion_parity(study, run, roots, observer=observer)
    assert not result
    assert message in result.diagnostic


def test_declared_margins_need_ten_worlds():
    """Fails if a margin is judged over fewer than ten worlds."""
    study, run, _, roots = kit_run("sample", "host")
    result = check_promotion_parity(
        study, run, roots, observer=ResampleSample(noise=NOISE)
    )
    assert not result and "10 needed" in result.diagnostic
