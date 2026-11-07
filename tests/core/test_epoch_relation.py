"""The epoch relation with observation times before the run starts (negative ticks)."""

import numpy as np

from spaceodyssey.core.battery.views import moved_ticks


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
