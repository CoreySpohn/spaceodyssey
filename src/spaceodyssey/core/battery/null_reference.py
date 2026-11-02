"""Null reference: a world with nothing in it must credit nothing."""

import math

import numpy as np

from spaceodyssey.core.battery.result import (
    default_executor,
    make_result,
    worlds_of,
)
from spaceodyssey.core.enums import RunStatus
from spaceodyssey.core.executors.base import run_status


def false_alarm_interval(rate, n_trials, n_worlds, z=4.0):
    """Interval for the mean credit per world when each trial can false-alarm.

    `rate` is the declared false-alarm probability of one trial and `n_trials` the
    number of trials over all worlds. The upper end is the binomial mean plus `z`
    standard deviations, divided by the number of worlds.
    """
    mean = rate * n_trials
    spread = math.sqrt(n_trials * rate * (1.0 - rate))
    return 0.0, (mean + z * spread) / n_worlds


def null_reference(
    study, null_truths, roots, n_steps, *, expected=None, interval=None, executor=None
):
    """Run `study` on worlds that hold nothing and compare the credit with the null.

    Assumptions: `null_truths` (a batch with a leading world axis) really holds nothing
    the objective counts, and the study observes at least once in every world. Pass
    `expected` where the objective has a closed form on the null: every world must
    then credit exactly that value, with no tolerance. Otherwise pass `interval`, the
    (low, high) range the mean credit per world must fall inside (see
    `false_alarm_interval`). A belief that moves on noise is not a failure; an
    informative prior shows up as credit that no null world should earn.
    """
    if (expected is None) == (interval is None):
        raise ValueError("pass exactly one of expected and interval")
    run = default_executor(executor).run_worlds(study, null_truths, roots, n_steps)
    failures = []
    pairs = worlds_of(run)
    credits = np.asarray(
        [float(study.objective.evaluate(c, study)) for c, _ in pairs], dtype=np.float64
    )
    n_records = [int(np.asarray(c.log.valid).sum()) for c, _ in pairs]
    if min(n_records) < 1:
        failures.append("a null world logged no record, so the check is vacuous")
    if bool(np.any(np.asarray(run_status(run.carry)) == RunStatus.OVERFLOW)):
        failures.append("a null run overflowed its capacities")
    if expected is not None:
        wrong = np.flatnonzero(credits != expected)
        if wrong.size:
            failures.append(
                f"credit differs from the exact null {expected} in worlds "
                f"{wrong.tolist()} (first value {credits[wrong[0]]})"
            )
    else:
        low, high = interval
        mean = float(credits.mean())
        if not low <= mean <= high:
            failures.append(f"mean credit {mean} outside the interval [{low}, {high}]")
    return make_result(
        "null_reference",
        failures,
        credits=credits.tolist(),
        n_records=n_records,
    )
