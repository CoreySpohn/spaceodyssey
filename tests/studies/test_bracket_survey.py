"""A bracket study on the survey profile: the policy runs on the JATIS grids.

The fixture files do not reach the 500 nm detection band, so the worlds are a draw of
the occurrence-rate generator on the fixture's two hosts as PHOENIX stars.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.adapters.jaxedith import (
    ObservingModeChannel,
    survey_grid_d,
    survey_table,
)
from spaceodyssey.adapters.modes import profile_modes
from spaceodyssey.adapters.physical_profile import JATIS_EAC1_V1
from spaceodyssey.adapters.skyscapes import (
    ExoVistaWorldSource,
    PopulationModel,
    star_spec_from_model,
)
from spaceodyssey.criteria.claims import EarthCandidateClaim, claims_on_truth
from spaceodyssey.studies import bracket as br

from .bracket_helpers import (
    N_SLOTS,
    N_WORLDS,
    PERIOD_SCALE,
    SURVEY_BANDS,
    fixture_source,
)
from .bracket_helpers import build as build_fixture

pytestmark = pytest.mark.slow(reason="compiles the chunked refresh driver once")

N_GRID_POINTS = 3
# The world key: the first of `split(PRNGKey(0), 64)` whose draw on these hosts holds
# an exoEarth candidate in a slot (keys 0 to 3 hold none, so the oracle never visits).
WORLD_KEY = 4


def _hosts():
    """The fixture's two hosts as PHOENIX stars carrying both survey bands."""
    source = fixture_source(1)
    return [
        star_spec_from_model(
            star.name,
            mass_msun=star.mass_msun,
            dist_pc=star.dist_pc,
            luminosity_lsun=star.luminosity_lsun,
            teff_k=teff,
            radius_rsun=radius,
            ra_deg=star.ra_deg,
            dec_deg=star.dec_deg,
            bands=SURVEY_BANDS,
            epoch_jd=source.epoch_jd,
        )
        for star, teff, radius in zip(
            source.stars, (5900.0, 5300.0), (1.1, 0.85), strict=True
        )
    ]


def _survey_source(hosts, epoch_jd):
    """One drawn world in `SURVEY_BANDS`; later worlds scale every period."""
    key = jax.random.split(jax.random.PRNGKey(0), 64)[WORLD_KEY]
    source = ExoVistaWorldSource.from_draws(
        jnp.stack([key] * N_WORLDS),
        hosts,
        n_slots=N_SLOTS,
        bands=SURVEY_BANDS,
        generator=PopulationModel.exovista(),
        epoch_jd=epoch_jd,
    )
    scale = jnp.asarray([PERIOD_SCALE**w for w in range(N_WORLDS)])[:, None, None]
    return eqx.tree_at(
        lambda s: s.population.period_d, source, source.population.period_d * scale
    )


@pytest.fixture(scope="module")
def built():
    hosts = _hosts()
    epoch_jd = fixture_source(1).epoch_jd
    source = _survey_source(hosts, epoch_jd)
    assert bool(
        claims_on_truth(
            source.population,
            (EarthCandidateClaim(),),
            jnp.asarray([s.mass_msun for s in hosts]),
            jnp.asarray([s.luminosity_lsun for s in hosts]),
            epoch_jd,
        ).any()
    ), "no exoEarth candidate in a slot"
    channel = ObservingModeChannel.from_stars(
        hosts,
        modes=profile_modes(JATIS_EAC1_V1),
        n_slots=N_SLOTS,
        epoch_jd=epoch_jd,
        profile=JATIS_EAC1_V1,
        durations_d=survey_grid_d(survey_table(JATIS_EAC1_V1), N_GRID_POINTS),
    )
    return build_fixture(channel=channel, source=source, budget_d=6.0, horizon_d=90.0)


def test_the_survey_modes_keep_the_declared_bands(built):
    """Fails if the survey hosts observe fixture bands instead of the survey bands."""
    table = built[1].channel_model.mode_table
    assert table.bands == SURVEY_BANDS
    assert [(b.center_nm, b.width_nm) for b in table.bands] == [
        (500.0, 100.0),
        (910.0, 6.5),
    ]


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
