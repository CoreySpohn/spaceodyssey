"""A bracket study on the survey profile: the policy runs on the JATIS grids."""

import numpy as np
import pytest

from spaceodyssey.adapters.jaxedith import ObservingModeChannel
from spaceodyssey.adapters.physical_profile import JATIS_EAC1_V1
from spaceodyssey.adapters.skyscapes import star_spec_from_model
from spaceodyssey.studies import bracket as br

from .bracket_helpers import N_SLOTS, fixture_source
from .bracket_helpers import build as build_fixture

pytestmark = pytest.mark.slow(reason="compiles the chunked refresh driver once")

N_GRID_POINTS = 3


def _survey_channel():
    """The fixture's two hosts as PHOENIX stars carrying both survey bands."""
    source = fixture_source()
    stars = [
        star_spec_from_model(
            star.name,
            mass_msun=star.mass_msun,
            dist_pc=star.dist_pc,
            luminosity_lsun=star.luminosity_lsun,
            teff_k=teff,
            radius_rsun=radius,
            ra_deg=star.ra_deg,
            dec_deg=star.dec_deg,
            wavelength_nm=500.0,
            bands_nm=((500.0, 100.0), (910.0, 6.5)),
        )
        for star, teff, radius in zip(
            source.stars, (5900.0, 5300.0), (1.1, 0.85), strict=True
        )
    ]
    return ObservingModeChannel.from_stars(
        stars,
        n_slots=N_SLOTS,
        epoch_jd=source.epoch_jd,
        profile=JATIS_EAC1_V1,
        n_grid=N_GRID_POINTS,
    )


@pytest.fixture(scope="module")
def built():
    return build_fixture(channel=_survey_channel(), budget_d=6.0, horizon_d=90.0)


def test_the_survey_channel_sets_the_policy_grids(built):
    """The summary's mode counts are the channel's grids, not six and three."""
    _, study, _, _, _, _ = built
    channel = study.channel_model
    assert channel.n_modes == channel.n_block_modes == N_GRID_POINTS
    assert study.caps.n_channels == 2 * N_GRID_POINTS + 1


def test_the_survey_run_completes_and_visits_the_stars(built):
    """No stall, overflow or violation; the shared budget is spent down."""
    run = built[-1]
    carry = run.carry
    assert bool(np.all(np.asarray(carry.done)))
    assert not bool(np.any(np.asarray(carry.overflow)))
    assert not bool(np.any(np.asarray(carry.stalled)))
    assert not bool(np.any(np.asarray(carry.violation)))
    balance = np.asarray(carry.ledger.balance)[:, -1, br.SHARED_RESOURCE]
    assert np.all(balance < 6.0)
