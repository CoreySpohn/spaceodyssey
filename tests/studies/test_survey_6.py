"""Survey 6, aging ephemerides: the epoch origin changes nothing, the age moves value.

The relation test is the merge-time guard (medium). The control shares its compiled
program (the shift and the `absolute` flag are arrays); the age test runs the plain
study, one more program per executor, for four ages and two decay rates. Both are slow
(the Slice 2b quick-tier budget, K6).
"""

import numpy as np
import pytest

from spaceodyssey.core import CompiledExecutor, HostExecutor, battery
from spaceodyssey.core.preflight import preflight
from tests.core.helpers import worlds
from tests.studies.scenarios.survey_6 import (
    AGE,
    AgingValuePolicy,
    survey_6,
    survey_6_table,
)
from tests.studies.survey_helpers import run_on, same_histories

N_STEPS = 40
OLDEST = 2 * AGE  # the prior starts 2 AGE before the run
SHIFTS = (3, OLDEST - 1, 1000)  # the last is larger than the oldest age
N_IMPORTED = 3  # rows the import writes per world (the public prior twice)
N_DECISIONS = 4  # the wait, two observations and the drain
AGES = tuple(AGE * 2**k for k in range(4))  # each one twice the one before


@pytest.mark.slow(reason="one cold program per executor")
def test_the_epoch_relation_holds_for_three_shifts_on_both_executors():
    """B7 row 6 (a), and B6 on survey 6. Fails if a shift of the epoch origin, one of
    them larger than the oldest prior age, moves any decision, record, belief or
    balance other than by the shift on either executor; for instance if the value or
    the wait reads an absolute tick, or a comparator leaves a negative time unshifted.
    """
    assert max(SHIFTS) > OLDEST
    truths, roots = worlds(2, survey_6().caps)
    study = survey_6()
    for executor in (CompiledExecutor(), HostExecutor()):
        for delta in SHIFTS:
            result = battery.shift_epoch(
                study, truths, roots, N_STEPS, delta, executor=executor
            )
            where = f"{type(executor).__name__}, shift {delta}"
            assert result, f"{where}: {result.diagnostic}"
            # evidence: every imported row is before the start, every decision compared
            assert result.detail["n_before_start"] == 2 * N_IMPORTED, where
            assert result.detail["n_views"] == 2 * N_DECISIONS, where


@pytest.mark.slow(reason="a control; shares the relation's program")
def test_a_value_reading_an_absolute_end_tick_fails_the_epoch_relation():
    """B7 row 6 (c), control. A policy that takes the prior's age as minus its end
    tick (as if the run started at tick 0) waits for another tick once the origin
    moves. Fails if the relation lets that through, or reports it anywhere but in the
    ticks.
    """
    truths, roots = worlds(2, survey_6().caps)
    study = survey_6(policy=AgingValuePolicy(absolute=True))
    result = battery.shift_epoch(study, truths, roots, N_STEPS, SHIFTS[0])
    assert not result
    failures = result.diagnostic.split("; ")
    assert any(f.endswith("trace field tick changed under the shift") for f in failures)
    assert all("tick" in f for f in failures), failures


def _first_values(executor, decay, truths, roots):
    """The value of the first decision, per age and world, on one executor."""
    out = []
    for age in AGES:
        study = survey_6(survey_6_table(age), policy=AgingValuePolicy(decay=decay))
        run = run_on(executor, study, truths, roots, N_STEPS)
        assert np.asarray(run.complete).all()
        out.append(np.asarray(run.carry.pstate)[:, 0])
    return np.asarray(out)


@pytest.mark.slow(reason="the plain study, one more program per executor")
def test_doubling_the_prior_age_moves_the_value_monotonically():
    """B7 row 6 (b). Fails if, over four ages each twice the one before, the value
    `base * decay ** age` of the first decision does not strictly decrease for a decay
    below one and strictly increase for one above, if it is not the value of the
    prior's relative age, or if the executors disagree on the value or the run.
    """
    study = survey_6()
    assert not preflight(study, executor="compiled")
    truths, roots = worlds(2, study.caps)
    for decay in (0.9, 1.1):
        values = {}
        for executor in ("compiled", "host"):
            values[executor] = _first_values(executor, decay, truths, roots)
        assert np.array_equal(values["host"], values["compiled"])
        steps = np.diff(values["compiled"], axis=0)
        assert (steps < 0).all() if decay < 1 else (steps > 0).all(), values
        # the age read at the first decision (tick 0) is the prior's own age
        want = np.asarray([[decay**age] * 2 for age in AGES])
        np.testing.assert_allclose(values["compiled"], want, rtol=1e-12, atol=0)
    host = run_on("host", study, truths, roots, N_STEPS)
    assert same_histories(host, run_on("compiled", study, truths, roots, N_STEPS), 2)
