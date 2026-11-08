"""Small physical surveys for the scene and the observing context (slow tier).

The working angles come from the pinned response package through yippy. Each
survey asserts the physical bound it is about; the measured numbers are quoted
in the change records.
"""

import jax
import numpy as np
import pytest
from hwoutils.conversions import lambda_d_to_arcsec

from spaceodyssey.adapters.physical_context import context_from_profile
from spaceodyssey.adapters.physical_profile import load_profile
from spaceodyssey.adapters.skyscapes import (
    PhysicalWorldSource,
    SceneSpec,
    SeparationSupport,
    build_scene,
    geometry,
    truth_scene,
)

# Loads the 2D response package and samples years of keepout.
pytestmark = pytest.mark.slow(
    reason="small physical surveys on the real response package"
)

PROFILE = load_profile()
UMA = SceneSpec.from_profile(PROFILE, 0)


@pytest.fixture(scope="module")
def working_angles_arcsec():
    """Inner and outer working angles of the pinned response, arcsec at 1000 nm."""
    from yippy import Coronagraph
    from yippy.datasets import fetch_yip

    response = PROFILE["response"]
    coronagraph = Coronagraph(fetch_yip(response["package"]))

    def arcsec(angle):
        return float(
            lambda_d_to_arcsec(
                angle.value, response["central_wavelength_nm"], response["diameter_m"]
            )
        )

    return arcsec(coronagraph.IWA), arcsec(coronagraph.OWA)


def test_scene_survey_over_one_orbit_and_world_draws(working_angles_arcsec):
    """Fails if 47 UMa b or its prior worlds leave the field or never clear the IWA.

    Over one period the nominal planet stays inside the outer working angle and
    spends part of the orbit outside the inner working angle, with contrast
    between 1e-10 and 1e-7. Sixty-four worlds drawn from the prior also stay in
    the field and clear the inner working angle for part of their orbits.
    """
    iwa, owa = working_angles_arcsec
    t = UMA.tp_jd + UMA.period_d * np.arange(256) / 256
    g = geometry(build_scene(UMA), t, UMA.wavelength_nm)
    sep = np.asarray(g.separation_arcsec)
    assert sep.max() < owa
    assert 0.0 < np.mean(sep >= iwa) < 1.0
    assert 1e-10 < float(np.min(g.contrast)) and float(np.max(g.contrast)) < 1e-7

    source = PhysicalWorldSource.from_profile(PROFILE, targets=[0])
    caps = type("Caps", (), {"n_subjects": 1})()
    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    worlds = jax.vmap(lambda k: source.draw(k, caps))(keys)
    cleared = []
    for w in range(64):
        truth = jax.tree_util.tree_map(lambda x, w=w: x[w], worlds)
        epochs = source.epoch_jd + float(truth.period_d[0]) * np.arange(64) / 64
        s = np.asarray(
            geometry(truth_scene(source, truth, 0), epochs, 1000.0).separation_arcsec
        )
        assert s.max() < owa
        cleared.append(np.mean(s >= iwa))
    assert np.mean(np.asarray(cleared) > 0.0) > 0.5


def test_context_survey_two_year_calendar(working_angles_arcsec):
    """Fails if the keepout calendar or the observable orbit share is implausible.

    Over two years 47 UMa is observable about 80 percent of the time (70 days of
    keepout a year) and the ecliptic twin about 75 percent (91 days). Over one
    orbital period of 47 UMa b, the share of days that are both outside keepout
    and inside the working angles is positive and below the keepout share.
    """
    iwa, owa = working_angles_arcsec
    context = context_from_profile(PROFILE)
    days = np.arange(0.0, 730.0, 1.0)
    jd = np.asarray(context.clock.jd(0)) + days
    share = [float(np.mean(np.asarray(context.observable(t, jd)))) for t in (0, 1)]
    assert share[0] == pytest.approx(1.0 - 70.0 / 365.25, abs=0.01)
    assert share[1] == pytest.approx(1.0 - 91.0 / 365.25, abs=0.01)

    orbit = np.asarray(context.clock.jd(0)) + np.arange(0.0, UMA.period_d, 1.0)
    support = SeparationSupport((build_scene(UMA),), (UMA.wavelength_nm,), iwa, owa)
    visible = np.asarray(context.observable(0, orbit))
    supported = np.asarray(support(0, orbit))
    both = float(np.mean(visible & supported))
    assert 0.0 < both < float(np.mean(visible))
