"""The epoch relation with observation times before the run starts (negative ticks).

The run-based checks use the synthetic study of `tests/core/battery_helpers.py` and
are in the slow tier, where the control that needs a second compiled policy also is
(`test_metamorphic.py`). In the quick tier the comparators are checked here, and the
relation runs with negative times in the survey 6 guard.
"""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from spaceodyssey.core import CompiledExecutor, HostExecutor, battery
from spaceodyssey.core.battery import metamorphic, views
from spaceodyssey.core.battery.views import moved_ticks
from spaceodyssey.core.enums import Audience, EventKind
from spaceodyssey.core.records import empty_log, log_columns
from spaceodyssey.core.snapshot import MeanBelief, Snapshot, mean_layout
from tests.core.battery_helpers import (
    EPOCH_CAPS,
    EPOCH_SHIFTS,
    EPOCH_STEPS,
    OLDEST,
    epoch_study,
    epoch_table,
)
from tests.core.helpers import worlds

N_IMPORTED = 5  # log rows the import writes per world (the public prior twice)


def test_tick_columns_move_by_name_whatever_their_sign():
    """Fails if a tick column is recognized by sign (a negative time left in place),
    if -1 is taken as unset in a start or end tick, if an unset release or acceptance
    stamp moves, if an invalid row or a column that is not a tick moves, or if a
    time is not moved onto -1 when the shift takes it there.
    """
    valid = np.array([True, True, True, False])
    times = np.array([-12, -1, 5, -3])
    assert moved_ticks("start_tick", times, valid, 7).tolist() == [-5, 6, 12, -3]
    assert moved_ticks("end_tick", times, valid, 7).tolist() == [-5, 6, 12, -3]
    assert moved_ticks("available_tick", times, valid, 7).tolist() == [-5, 6, 12, -3]
    stamps = np.array([-1, 0, 4, 9])
    for name in ("released_tick", "accepted_tick"):
        assert moved_ticks(name, stamps, valid, 7).tolist() == [-1, 7, 11, 9]
        assert moved_ticks(name, stamps, valid, -4).tolist() == [-1, -4, 0, 9]
    assert moved_ticks("subject", times, valid, 7).tolist() == times.tolist()
    # the log comparator moves the shifted run back: a time may land on -1
    assert moved_ticks("end_tick", np.array([4]), np.array([True]), -5).tolist() == [-1]


def test_every_log_tick_column_is_compared_and_moved():
    """Fails if a log column named `*_tick` is not in the moved set (an importer that
    writes it at absolute ticks would pass the relation), if a tick column is dropped
    from the columns the log or decision-view comparator reads, or if either
    comparator reads its columns from anywhere but the shared `log_columns`, rejects
    a correct shift, or accepts a start tick left absolute in the shifted log.
    """
    log = empty_log(EPOCH_CAPS)
    cols = log_columns(log)
    ticks = {name for name in cols if name.endswith("_tick")}
    assert ticks == set(views.TICKS)
    assert metamorphic.log_columns is log_columns
    assert views.log_columns is log_columns
    valid = np.array([True])
    for name in sorted(ticks):
        assert moved_ticks(name, np.array([3]), valid, 7).tolist() == [10], name

    log = dataclasses.replace(
        log,
        **{name: np.asarray(col)[:1].copy() for name, col in cols.items()},
    )
    log = dataclasses.replace(
        log,
        valid=valid,
        kind=np.array([1]),
        audience=np.array([int(Audience.OBSERVER | Audience.POLICY)]),
        start_tick=np.array([-3]),
        end_tick=np.array([-1]),
        superseded_by=np.array([-1]),
    )
    shifted = dataclasses.replace(
        log,
        start_tick=np.array([4]),
        end_tick=np.array([6]),
        available_tick=np.array([7]),
        released_tick=np.array([7]),
        accepted_tick=np.array([7]),
    )

    @dataclasses.dataclass
    class DecisionTrace:
        handled: np.ndarray
        wake: np.ndarray
        kind: np.ndarray
        tick: np.ndarray
        n_dec: np.ndarray

    trace = DecisionTrace(
        valid, valid, np.array([EventKind.RELEASE]), np.array([0]), np.array([1])
    )
    late_trace = dataclasses.replace(trace, tick=trace.tick + 7)
    belief = MeanBelief(mean=np.array([0.0]), n=np.array([1]), aux=np.zeros(0))
    layout = mean_layout(belief, belief)
    carry = SimpleNamespace(
        log=log,
        belief=belief,
        pbelief=belief,
        acquisition_counts=np.array([1]),
        ledger=SimpleNamespace(balance=np.array([0.0])),
    )
    view = Snapshot(
        log=log,
        n_rows=np.array(1),
        belief=belief,
        tick=np.array(0),
        n_dec=np.array(0),
        release_version=np.array(1),
    )
    for absolute in (False, True):
        late_log = (
            dataclasses.replace(shifted, start_tick=log.start_tick)
            if absolute
            else shifted
        )
        late_carry = SimpleNamespace(**{**vars(carry), "log": late_log})
        late_view = dataclasses.replace(view, log=late_log, tick=view.tick + 7)
        assert metamorphic._epoch_world(
            0, carry, trace, late_carry, late_trace, 7, layout
        ) == (
            ["world 0: log column start_tick changed under the shift"]
            if absolute
            else []
        )
        assert views.compare_views(
            [(view, None)],
            [(late_view, None)],
            carry,
            late_carry,
            EPOCH_CAPS,
            trace,
            layout=layout,
            epoch=7,
        ) == (["decision 0: policy view log start_tick changed"] if absolute else [])


def test_shift_epoch_refuses_a_shift_that_is_not_positive():
    """Fails if a zero shift (which checks nothing) or a negative one (which could
    move a stamp onto the unset value) is run instead of refused.
    """
    truths, roots = worlds(1, EPOCH_CAPS)
    for delta in (0, -5):
        with pytest.raises(ValueError, match="positive"):
            battery.shift_epoch(epoch_study(), truths, roots, EPOCH_STEPS, delta)


def test_the_fixture_covers_negative_times_minus_one_and_a_shift_past_the_oldest():
    """Fails if the fixture loses what the run checks rely on: times below zero, a
    time at -1, shifts that land a time on -1, and a shift past the oldest time.
    """
    t = epoch_table()
    times = [int(x) for x in (*t.start_offset, *t.end_offset)]
    assert min(times) == -OLDEST and -1 in times
    landed = {d for d in EPOCH_SHIFTS for x in times if x + d == -1}
    assert landed == {5, 19}
    assert max(EPOCH_SHIFTS) > OLDEST


# the survey 6 guard runs the relation with negative times at merge time (K6)
@pytest.mark.slow(
    reason="one compiled program per executor, shared by the three shifts"
)
def test_the_epoch_relation_holds_with_negative_times_on_both_executors():
    """B6 on the synthetic study. Fails if either comparator (the log comparator or
    the decision-view comparator) ignores the shift, compares absolute ticks, takes
    tick columns by sign (a negative time left unshifted), takes -1 as unset in a
    start or end tick, or moves an unset release stamp; or if a shift smaller than,
    or one larger than, the oldest imported time breaks the run on either executor.
    """
    truths, roots = worlds(2, EPOCH_CAPS)
    study = epoch_study()
    for executor in (CompiledExecutor(), HostExecutor()):
        for delta in EPOCH_SHIFTS:
            result = battery.shift_epoch(
                study, truths, roots, EPOCH_STEPS, delta, executor=executor
            )
            where = f"{type(executor).__name__}, shift {delta}"
            assert result, f"{where}: {result.diagnostic}"
            # every imported row is before the start, in both worlds
            assert result.detail["n_before_start"] == 2 * N_IMPORTED, where
            # the first decision, the window's and three more, in both worlds
            assert result.detail["n_views"] == 2 * 5, where


@pytest.mark.slow(reason="a control; the same program as the relation test")
def test_the_epoch_relation_catches_rows_written_at_absolute_ticks():
    """Control: an importer that writes start ticks as if the run started at tick 0.
    Fails if either comparator lets the unshifted start ticks through (a comparator
    that skips tick columns, or one that moves the shifted run's ticks wrongly), or
    if the failure is reported anywhere but in the start ticks.
    """
    truths, roots = worlds(2, EPOCH_CAPS)
    result = battery.shift_epoch(
        epoch_study(absolute=True), truths, roots, EPOCH_STEPS, EPOCH_SHIFTS[0]
    )
    assert not result
    failures = result.diagnostic.split("; ")
    assert all("start_tick" in f for f in failures), failures
    assert any(
        f.endswith("log column start_tick changed under the shift") for f in failures
    )
    assert any(f.endswith("policy view log start_tick changed") for f in failures)
