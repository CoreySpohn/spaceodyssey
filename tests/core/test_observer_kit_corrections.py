"""The observer kit's correction checks on two scripted host worlds (slow tier).

Each study runs the refresh protocol on the host executor, and one executor compile
alone (near 5 s cold) is over the unit budget, so the file is slow. Refusals and the
kit's own reading of the replacement rule need no run: `test_observer_kit.py`.
"""

import numpy as np
import pytest

from spaceodyssey.core.conformance import (
    check_delayed_correction,
    check_derived_summary,
)
from spaceodyssey.core.enums import Capacity
from tests.core.kit_helpers import SUMMARY_KIND, refresh_run


def _belief(run, subject=0):
    """`(mean, n)` of a subject in every world of a batched run."""
    belief = run.carry.belief
    return np.asarray(belief.mean)[:, subject], np.asarray(belief.n)[:, subject]


def test_a_delayed_correction_replaces_the_superseded_row():
    """Fails if a revision released after acceptance leaves the refit-capable observer
    unequal to a flat-prior refit of the corrected rows, or if the run is not the one
    the check names (a correction accepted after its original: belief 6.5, two rows).
    """
    study, run, roots = refresh_run("correction")
    log = run.carry.log
    revision = np.asarray(log.revision)
    assert (np.asarray(log.valid) & (revision == 1)).any(axis=1).all()  # a correction
    result = check_delayed_correction(study, run, roots)
    assert result, result.diagnostic
    mean, n = _belief(run)
    np.testing.assert_allclose(mean, 6.5, rtol=0, atol=1e-12)  # (9 + 4) / 2, not 5
    assert n.tolist() == [2, 2]


def test_delayed_correction_control_an_observer_that_keeps_the_superseded_row():
    """Fails if an observer whose install merges into what the subject holds, so that
    the superseded row stays in its belief, passes the delayed correction check.
    """
    study, run, roots = refresh_run("keeps_superseded")
    mean, n = _belief(run)
    assert not np.allclose(mean, 6.5) and n.min() > 2  # control: the row stayed
    result = check_delayed_correction(study, run, roots)
    assert not result
    assert "subject 0" in result.diagnostic and "6.5" in result.diagnostic


@pytest.mark.parametrize("name", ["summary", "summary_twice"])
def test_a_derived_summary_adds_nothing_to_the_belief(name):
    """Fails if a derived summary of folded rows (alone, or a second one under its own
    acquisition) leaves the belief unequal to a refit that counts each datum once (mean
    3 over two rows; counted as a datum it would be 10/3 over three).
    """
    study, run, roots = refresh_run(name)
    assert SUMMARY_KIND in study.caps.derived_kinds
    log = run.carry.log
    assert (np.asarray(log.valid) & (np.asarray(log.kind) == SUMMARY_KIND)).any()
    result = check_derived_summary(study, run, roots, summary_kind="summary")
    assert result, result.diagnostic
    mean, n = _belief(run)
    np.testing.assert_allclose(mean, 3.0, rtol=0, atol=1e-12)
    assert n.tolist() == [2, 2]


def test_derived_summary_control_the_same_summary_registered_as_non_derived():
    """Fails if a summary registered as an ordinary kind, listed for refit and counted
    by the observer, passes: its belief counts the summary as a third datum.
    """
    study, run, roots = refresh_run("summary_not_derived")
    assert SUMMARY_KIND not in study.caps.derived_kinds
    mean, n = _belief(run)
    assert n.tolist() == [3, 3]  # control: the summary was counted
    np.testing.assert_allclose(mean, 10.0 / 3.0, rtol=0, atol=1e-12)
    result = check_derived_summary(study, run, roots, summary_kind="summary")
    assert not result and "subject 0" in result.diagnostic


def test_a_summary_reissued_under_its_own_id_is_refused_as_a_correction():
    """Fails if the check judges a run in which the core refused a correction that
    reuses a derived acquisition id (`Capacity.INVALID_CORRECTION`), whose log is not
    the one the study sent.
    """
    study, run, roots = refresh_run("summary_reissued")
    assert (
        np.asarray(run.carry.overflow_mask) & int(Capacity.INVALID_CORRECTION)
    ).all()
    with pytest.raises(ValueError, match="refused as a correction"):
        check_derived_summary(study, run, roots, summary_kind="summary")
