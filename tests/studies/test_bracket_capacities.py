"""Declared row capacities for imaging, spectra and forecast renewal."""

import pytest

from spaceodyssey.studies import bracket as br


@pytest.mark.parametrize(
    "n, horizon_d, n_log, rows, max_sid",
    [
        (
            2,
            120.0,
            1024,
            40,
            4096,
        ),  # need 2 * (37 + 2 * (2 + 24)) = 178: the floors hold
        (
            150,
            730.0,
            17920,
            40,
            32768,
        ),  # need 14250: n_log 256 * ceil(1.25 * 14250 / 256)
        (150, 1826.0, 19712, 40, 65536),  # need 15750; ids 2 * 15750 + 2 * 1826 + 1200
    ],
)
def test_capacities_hold_every_row_a_star_can_add(n, horizon_d, n_log, rows, max_sid):
    """Fails if a capacity misses the catalog row, a visit, a spectrum, or a refit
    product.

    Hand values at MAX_VISITS 24, MAX_SPECTRA 12, RENEWAL_D 225 d, REFIT_ROWS 2.
    """
    caps = br.default_capacities(
        n, 6, n_pstate=1, n_channels=37, horizon_d=horizon_d, ticks_per_day=86_400
    )
    assert (caps.n_log, caps.rows_per_subject, caps.max_sid) == (n_log, rows, max_sid)
    assert caps.n_pstate == 1
    assert caps.max_tick >= horizon_d * 86_400


def test_more_spectra_need_more_rows_and_the_horizon_is_required():
    """Fails if spectrograph rows are left out of the bound, or a horizon-free
    branch remains.
    """
    base = br.default_capacities(150, 4, n_pstate=1, n_channels=25, horizon_d=1825.0)
    more = br.default_capacities(
        150, 4, n_pstate=1, n_channels=25, horizon_d=1825.0, max_spectra=24
    )
    assert (
        more.rows_per_subject == base.rows_per_subject + 12 and more.n_log > base.n_log
    )
    with pytest.raises(TypeError, match="n_pstate"):
        br.default_capacities(2, 6, n_channels=10, horizon_d=120.0)
    with pytest.raises(TypeError, match="horizon_d"):
        br.default_capacities(2, 6, n_pstate=1, n_channels=10)


def test_an_unreachable_horizon_is_refused():
    """Fails if a study gets a clock that ends before its horizon."""
    with pytest.raises(ValueError, match="max_tick"):
        br.default_capacities(150, 4, n_pstate=1, n_channels=25, horizon_d=60_000.0)
    with pytest.raises(ValueError, match="tick_length_s"):
        br.default_capacities(1059, 4, n_pstate=1, n_channels=25, horizon_d=1826.0)
    assert (
        br.default_capacities(
            1059, 4, n_pstate=1, n_channels=25, horizon_d=1826.0, ticks_per_day=86_400
        ).max_tick
        > 1826 * 86_400
    )
