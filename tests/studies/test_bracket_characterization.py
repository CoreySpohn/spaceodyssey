"""Capacity bounds for imaging visits and forecast renewal."""

import math

import pytest

from spaceodyssey.studies import bracket as br


def test_default_capacities_scale_with_stars_and_horizon():
    """Fails if capacity scaling loses the horizon or fixed small-study floor.

    The 150-star values are at least those the earlier 150-star runs needed to finish
    (4096 log rows, 96 counting rows would have been generous, 16384 ids).
    """
    days = br.TICKS_PER_DAY
    big = br.default_capacities(150, 4, n_channels=25, horizon_d=1825.0)
    assert big.n_log >= 4096 and big.rows_per_subject >= 1 + 24
    assert big.max_sid >= 16384
    assert big.max_tick > 1825 * days
    assert big.n_pstate == br.pstate_size(150)
    small = br.default_capacities(2, 6, n_channels=10, horizon_d=120.0)
    assert (small.n_log, small.max_sid) == (1024, 4096)
    assert small.rows_per_subject >= 1 + 24 + 3
    legacy = br.default_capacities(2, 6, n_channels=10)
    assert (legacy.n_log, legacy.rows_per_subject, legacy.max_sid) == (1024, 32, 4096)
    with pytest.raises(ValueError, match="max_tick"):
        br.default_capacities(150, 4, n_channels=25, horizon_d=60_000.0)
    # A full 1059-star list over five years fits only on a longer declared tick.
    with pytest.raises(ValueError, match="tick_length_s"):
        br.default_capacities(1059, 4, n_channels=25, horizon_d=1826.0)
    second = br.default_capacities(
        1059, 4, n_channels=25, horizon_d=1826.0, ticks_per_day=86_400
    )
    assert second.max_tick > 1826 * 86_400


def test_capacities_cover_every_visit_block_and_renewal_a_star_can_add():
    """Each star holds a catalog row, visits and forecast-renewal products.

    Each visit and forecast renewal adds `REFIT_ROWS` refit products. Fails if the
    log is sized from a typical visit count instead of the policy's cap.
    """
    n, slots, horizon = 150, 4, 1825.0
    for max_visits in (24, 40, 8):
        caps = br.default_capacities(
            n,
            slots,
            n_channels=25,
            horizon_d=horizon,
            max_visits=max_visits,
        )
        renewals = math.ceil(horizon / br.RENEWAL_D) + 1
        per_star = 1 + max_visits
        refits = br.REFIT_ROWS * (max_visits + renewals)
        assert caps.n_log >= n * (per_star + refits)
        assert caps.rows_per_subject >= per_star
        assert caps.max_sid >= 2 * n * (per_star + refits)
        assert caps.max_tick > horizon * br.TICKS_PER_DAY
