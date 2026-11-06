"""Hand-built scene: signed conjunctions, sky basis, contrast, cos(delta), world source.

Expected values come from tests/anchors/reference_geometry.py (mpmath, astropy
constants, no library under test) or are written by hand. Each anchor names the
mutant it rejects.
"""

import dataclasses
import types

import astropy.units as u
import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from numpyro import handlers
from photomancy.orbit.data import OrbitData
from photomancy.orbit.forward import predict_relative_astrometry
from photomancy.orbit.model import build_model

from spaceodyssey.adapters.physical_context import context_from_profile
from spaceodyssey.adapters.physical_profile import load_profile
from spaceodyssey.adapters.skyscapes import (
    PhysicalWorldSource,
    SceneSpec,
    build_scene,
    diagnose,
    geometry,
    truth_scene,
)
from tests.anchors import reference_geometry as ref

PROFILE = load_profile()
UMA = SceneSpec.from_profile(PROFILE, 0)
DAY_TICKS = 86_400_000
# 47 UMa b: M sin i = 2.53 Jupiter masses (Gregory and Fischer 2010) at i = 60 deg.
REF_MP_KG = mp.mpf("2.53") * ref.M_JUP_KG / mp.sin(mp.radians(60))


def circular(inc_deg, node_deg):
    """A circular orbit with the periastron at the ascending node (planet w = 0)."""
    return dataclasses.replace(
        UMA, ecc=0.0, omega_star_deg=180.0, inclination_deg=inc_deg, node_deg=node_deg
    )


def quarter_epochs(spec):
    """Node, superior conjunction, descending node, inferior conjunction."""
    return spec.tp_jd + spec.period_d * np.array([0.0, 0.25, 0.5, 0.75])


def reference_scene(spec, t_jd):
    """(east, north, away) in AU from the independent reference, per epoch."""
    central = mp.mpf(spec.mass_msun) * ref.M_SUN_KG + REF_MP_KG
    a_au = ref.semimajor_axis_au(spec.period_d, central)
    return [
        ref.relative_position(
            t,
            period_d=spec.period_d,
            ecc=spec.ecc,
            inc_deg=spec.inclination_deg,
            node_deg=spec.node_deg,
            omega_deg=spec.omega_planet_deg,
            tp_jd=spec.tp_jd,
            a_au=a_au,
        )
        for t in t_jd
    ]


def away_au(spec, t_jd):
    system = build_scene(spec)
    r_au, _, _ = system.planets[0].propagate(
        system.trig_solver, jnp.asarray(t_jd), star=system.star
    )
    return np.asarray(r_au[0, 2])


def test_signed_conjunctions_and_node():
    """Fails on a phase supplement, a flipped away axis or an approaching node.

    Edge-on circular orbit, node at north, periastron at the node: by hand the
    planet is at the node (away = 0, receding) at tp, behind the star (away > 0,
    phase angle 0, Lambert phase 1) a quarter period later, and in front (away < 0,
    phase angle pi, Lambert phase 0) at three quarters.
    """
    spec = circular(90.0, 0.0)
    t = quarter_epochs(spec)
    g = geometry(build_scene(spec), t, spec.wavelength_nm)
    z = away_au(spec, t)
    a = float(jnp.max(g.distance_au))
    np.testing.assert_allclose(z / a, [0.0, 1.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(g.phase_angle_rad)[[1, 3]], [0.0, np.pi], atol=1e-9
    )
    dz = away_au(spec, t[[0]] + 1e-3) - away_au(spec, t[[0]] - 1e-3)
    assert dz[0] > 0.0  # the ascending node recedes
    full = spec.geometric_albedo * (spec.planet_radius_m / (a * ref.AU_M)) ** 2
    np.testing.assert_allclose(float(g.contrast[1]), float(full), rtol=1e-9)
    assert float(g.contrast[3]) < 1e-12 * float(full)


@pytest.mark.parametrize(
    ("inc_deg", "east_signs"),
    [(60.0, [0, 1, 0, -1]), (120.0, [0, -1, 0, 1])],
    ids=["prograde", "retrograde"],
)
def test_sky_signs_at_four_phases(inc_deg, east_signs):
    """Fails on an east-north swap, a flipped east axis or a reversed sense of motion.

    Circular orbit, node at north, periastron at the node. By hand: north is +r,
    0, -r, 0 at the four quarter epochs; east is +-r cos(i) sin(f), positive at
    the superior conjunction for i < 90 degrees (position angle increasing) and
    negative for i > 90 degrees; away is r sin(i) sin(f).
    """
    spec = circular(inc_deg, 0.0)
    t = quarter_epochs(spec)
    g = geometry(build_scene(spec), t, spec.wavelength_nm)
    east = np.asarray(g.east_arcsec)
    north = np.asarray(g.north_arcsec)
    scale = float(jnp.max(g.separation_arcsec))
    tiny = 1e-9 * scale
    assert np.array_equal(np.sign(np.where(abs(east) < tiny, 0, east)), east_signs)
    assert np.array_equal(np.sign(np.where(abs(north) < tiny, 0, north)), [1, 0, -1, 0])
    assert away_au(spec, t[[1]])[0] > 0.0  # behind the star at the superior conjunction
    later = geometry(build_scene(spec), t[:1] + 1.0, spec.wavelength_nm)
    pa = np.arctan2(east[0], north[0])
    pa_later = float(np.arctan2(later.east_arcsec[0], later.north_arcsec[0]))
    assert np.sign(pa_later - pa) == (1 if inc_deg < 90 else -1)


@pytest.mark.parametrize("ecc", [0.0, 0.5, 0.9])
def test_positions_against_30_digit_kepler(ecc):
    """Fails if the Kepler solve, node, argument of periastron or scale drifts.

    The 47 UMa scene at three eccentricities over 64 epochs of one period,
    against the 30-digit reference: sky offsets within 0.01 mas (the Kepler-solve
    allocation) and the phase angle within 1e-9 rad.
    """
    spec = dataclasses.replace(UMA, ecc=ecc)
    t = spec.tp_jd + spec.period_d * (np.arange(64) + 0.37) / 64
    g = geometry(build_scene(spec), t, spec.wavelength_nm)
    expected = reference_scene(spec, t)
    east = np.array([float(ref.arcsec(e, spec.dist_pc)) for e, _, _ in expected])
    north = np.array([float(ref.arcsec(n, spec.dist_pc)) for _, n, _ in expected])
    beta = np.array([float(ref.phase_angle(*p)) for p in expected])
    np.testing.assert_allclose(g.east_arcsec, east, atol=1e-5, rtol=0)
    np.testing.assert_allclose(g.north_arcsec, north, atol=1e-5, rtol=0)
    np.testing.assert_allclose(g.phase_angle_rad, beta, atol=1e-9, rtol=0)


def test_contrast_at_three_phases():
    """Fails on a phase supplement, a radius unit slip or a squared-distance error.

    47 UMa b at superior conjunction (w + f = 90 deg), quadrature (the node,
    w + f = 0) and near inferior conjunction (w + f = 260 deg), against Ag
    Phi(beta) (R / r)^2 by hand within 0.1 percent.
    """
    w = UMA.omega_planet_deg
    t = [
        float(
            ref.time_of_true_anomaly(
                x - w, period_d=UMA.period_d, ecc=UMA.ecc, tp_jd=UMA.tp_jd
            )
        )
        for x in (90.0, 0.0, 260.0)
    ]
    g = geometry(build_scene(UMA), t, UMA.wavelength_nm)
    expected = [
        float(
            ref.contrast(*p, radius_m=UMA.planet_radius_m, albedo=UMA.geometric_albedo)
        )
        for p in reference_scene(UMA, t)
    ]
    np.testing.assert_allclose(g.contrast, expected, rtol=1e-3)
    np.testing.assert_allclose(g.phase_angle_rad[1], np.pi / 2, atol=1e-9)


def _east_from_coordinates(star, planet):
    """Delta alpha cos(delta) from absolute coordinates, arcsec."""
    dra = (planet.ra - star.ra).wrap_at(180 * u.deg)
    return (dra * np.cos(star.dec.radian)).to_value(u.arcsec)


@pytest.mark.parametrize(
    ("factor", "passes"),
    [(lambda c: 1.0, True), (lambda c: 1.0 / c, False), (lambda c: c, False)],
    ids=["as-published", "cos-dropped", "cos-twice"],
)
def test_east_offset_carries_cos_delta_once(factor, passes):
    """Fails if the east offset drops cos(delta) or applies it twice.

    At a stellar declination of 60 degrees the RA difference is twice the east
    offset. The scene's east offset is unchanged by the declination, and placing
    the planet with astropy's spherical offsets recovers it as (RA_p - RA_s)
    cos(delta_s); the dropped and doubled forms miss by a factor of two.
    """
    spec = dataclasses.replace(UMA, dec_deg=60.0)
    t = [UMA.tp_jd + 100.0]
    east = float(geometry(build_scene(spec), t, spec.wavelength_nm).east_arcsec[0])
    north = float(geometry(build_scene(spec), t, spec.wavelength_nm).north_arcsec[0])
    same = float(geometry(build_scene(UMA), t, UMA.wavelength_nm).east_arcsec[0])
    assert east == same
    star = SkyCoord(spec.ra_deg * u.deg, spec.dec_deg * u.deg)
    planet = star.spherical_offsets_by(east * u.arcsec, north * u.arcsec)
    reported = east * factor(np.cos(np.radians(spec.dec_deg)))
    ok = np.isclose(reported, _east_from_coordinates(star, planet), rtol=1e-6)
    assert bool(ok) is passes


def test_scene_reproduces_published_stellar_velocity():
    """Fails if the radial-velocity argument of periastron is used for the planet.

    The star's velocity is -Mp / (Ms + Mp) times the planet's relative dz/dt.
    With the planet's argument of periastron 180 degrees from the published
    stellar one, it matches K [cos(f + w_star) + e cos(w_star)] with the published
    K = 48.4 m/s (Gregory and Fischer 2010, Table 6) within 1.5 m/s; using the
    stellar value for the planet flips the sign and misses by up to 2K.
    """
    t = UMA.tp_jd + UMA.period_d * np.arange(8) / 8
    h = 1e-3
    dz = (away_au(UMA, t + h) - away_au(UMA, t - h)) / (2 * h)
    ms = UMA.mass_msun * float(ref.M_SUN_KG)
    mp_kg = float(REF_MP_KG)
    v_star = -mp_kg / (ms + mp_kg) * dz * float(ref.AU_M) / ref.DAY_S
    w_s = np.radians(UMA.omega_star_deg)
    e = UMA.ecc
    f = []
    for ti in t:
        M = 2 * mp.pi * (ti - UMA.tp_jd) / UMA.period_d
        E = ref.kepler_eccentric_anomaly(M, e)
        f.append(
            float(
                2
                * mp.atan2(
                    mp.sqrt(1 + e) * mp.sin(E / 2), mp.sqrt(1 - e) * mp.cos(E / 2)
                )
            )
        )
    v_published = 48.4 * (np.cos(np.array(f) + w_s) + e * np.cos(w_s))
    np.testing.assert_allclose(v_star, v_published, atol=1.5)


def _caps(n):
    return types.SimpleNamespace(n_subjects=n)


@pytest.fixture(scope="module")
def world():
    source = PhysicalWorldSource.from_profile(PROFILE)
    return source, source.draw(jax.random.PRNGKey(0), _caps(2))


def test_world_draws_are_keyed_by_subject(world):
    """Fails if a subject's draw depends on how many subjects the world holds.

    Subject 0 of the two-target world equals the draw of a one-target world.
    """
    source, truth = world
    alone = PhysicalWorldSource.from_profile(PROFILE, targets=[0])
    one = alone.draw(jax.random.PRNGKey(0), _caps(1))
    for name in ("period_d", "e", "cos_i", "tp_jd", "Rp_AU", "Ag"):
        assert float(getattr(one, name)[0]) == float(getattr(truth, name)[0]), name
    with pytest.raises(ValueError, match="subjects"):
        source.draw(jax.random.PRNGKey(0), _caps(3))


def test_truth_scene_matches_photomancy_forward_model(world):
    """Fails if the world source misplaces photomancy's time origin, mass or angles.

    Subject 0 draws photomancy's prior model with ``fold_in(key, 0)``. The scene
    built from the truth gives the same sky offsets as photomancy's own forward
    model of that draw's sites, with photomancy's times measured from the
    source's epoch.
    """
    source, truth = world
    scene = source.scenes[0]
    prior = source.priors[0]
    model = build_model(
        sample_planet_scale=True,
        log_P_range=prior.log_P_range,
        log_Mp_range=prior.log_Mp_range,
        log_Rp_range=prior.log_Rp_range,
        log_Ag_range=prior.log_Ag_range,
        ecc_prior=prior.ecc_prior,
    )
    ms = scene.mass_msun * float(ref.M_SUN_KG)
    key = jax.random.fold_in(jax.random.PRNGKey(0), 0)
    trace = handlers.trace(handlers.seed(model, key)).get_trace(
        ms, scene.dist_pc, OrbitData()
    )
    site = {name: jnp.squeeze(v["value"]) for name, v in trace.items()}
    t = source.epoch_jd + np.linspace(0.0, 1500.0, 7)
    g = geometry(truth_scene(source, truth, 0), t, scene.wavelength_nm)
    ra, dec = predict_relative_astrometry(
        jnp.asarray(t - source.epoch_jd),
        site["a"],
        site["e"],
        site["cos_i"],
        site["W"],
        site["cos_w"],
        site["sin_w"],
        site["tp"],
        ms,
        scene.dist_pc,
        Mp=site["Mp"],
    )
    np.testing.assert_allclose(g.east_arcsec, ra, atol=1e-9)
    np.testing.assert_allclose(g.north_arcsec, dec, atol=1e-9)


def test_diagnose_recomputes_truth_geometry_from_record_identity(world):
    """Fails if diagnose reads the model scene, the occupancy midpoint or the payload.

    A record's truth-side geometry is the drawn world's geometry at the
    science-interval midpoint (after settling), not the nominal scene's and not
    the occupancy midpoint. A record that ends during settling has no science.
    """
    source, truth = world
    context = context_from_profile(PROFILE)
    start, end = 10 * DAY_TICKS, 13 * DAY_TICKS
    record = types.SimpleNamespace(
        subject=0, start_tick=start, end_tick=end, payload=None
    )
    got = diagnose(truth, record, source=source, context=context)
    epoch = float(context.clock.jd(start + DAY_TICKS, DAY_TICKS))
    want = geometry(truth_scene(source, truth, 0), [epoch], UMA.wavelength_nm)
    assert float(got.geometry.epoch_jd[0]) == epoch
    assert bool(got.has_science)
    np.testing.assert_allclose(got.geometry.east_arcsec, want.east_arcsec, rtol=1e-12)
    nominal = geometry(build_scene(UMA), [epoch], UMA.wavelength_nm)
    assert abs(float(nominal.east_arcsec[0] - got.geometry.east_arcsec[0])) > 1e-6
    short = types.SimpleNamespace(
        subject=0, start_tick=start, end_tick=start + DAY_TICKS // 2
    )
    assert not bool(diagnose(truth, short, source=source, context=context).has_science)
