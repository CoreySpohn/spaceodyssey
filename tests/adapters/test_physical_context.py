"""Absolute time, science intervals and keepout certification on the context alone.

The time anchors are integer arithmetic; the keepout window is checked against
spherical trigonometry (astropy for the ecliptic latitude, no orbix); the
certification outcomes are named codes, so a visit with no visibility gives an
explicit outcome and an uncertified plan is refused at load.
"""

import dataclasses

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord
from orbix.observatory import body_angle
from orbix.observatory.solar_system import earth_position_ecliptic

from spaceodyssey.adapters.physical_context import (
    MJD_ZERO_JD,
    Certification,
    PhysicalClock,
    PlannedVisit,
    UncertifiedPlan,
    context_from_profile,
    require_certified,
)
from spaceodyssey.adapters.physical_profile import load_profile
from spaceodyssey.adapters.skyscapes import (
    SceneSpec,
    SeparationSupport,
    build_scene,
    geometry,
)
from spaceodyssey.core.study import Clock

PROFILE = load_profile()
DAY = 86_400_000  # ticks per day at 1 ms
UMA, TWIN = 0, 1


@pytest.fixture(scope="module")
def context():
    return context_from_profile(PROFILE)


@pytest.fixture(scope="module")
def edge(context):
    """First keepout entry of 47 UMa after the study epoch, to one minute (ticks)."""
    days = np.arange(150.0, 250.0, 0.25)
    ok = np.asarray(context.observable(UMA, context.clock.jd(0) + days))
    first = int(np.argmax(~ok))
    lo, hi = int(days[first - 1] * DAY), int(days[first] * DAY)
    while hi - lo > 60_000:
        mid = (lo + hi) // 2
        inside = bool(context.observable(UMA, context.clock.jd(mid))[0])
        lo, hi = (mid, hi) if inside else (lo, mid)
    return hi


def test_tick_jd_round_trip_and_relabelling():
    """Fails if a tick does not survive tick to JD to tick, or epochs depend on labels.

    One million random ticks over +-16 years return exactly; moving the origin by
    k ticks and every tick by -k gives bit-identical Julian Dates; tick 0 at the
    study epoch JD 2462502.5 is MJD 62502 exactly.
    """
    clock = PhysicalClock.from_jd(PROFILE["time"]["study_epoch_jd"])
    rng = np.random.default_rng(0)
    ticks = rng.integers(-5 * 10**11, 5 * 10**11, size=1_000_000)
    assert np.array_equal(np.asarray(clock.tick(clock.jd(ticks))), ticks)
    k = int(rng.integers(-(10**12), 10**12))
    moved = PhysicalClock(clock.clock, clock.origin_ticks + k)
    assert np.array_equal(np.asarray(moved.jd(ticks - k)), np.asarray(clock.jd(ticks)))
    assert float(clock.mjd(0)) == 62502.0
    assert float(clock.jd(0)) - MJD_ZERO_JD == float(clock.mjd(0))
    assert int(PhysicalClock.from_jd(2451545.0).origin_ticks) == 0
    with pytest.raises(ValueError, match="whole number of ticks"):
        PhysicalClock(Clock(tick_length_s=7e-3), 0)


def test_science_interval_is_the_last_exposure_of_the_occupancy(context):
    """Fails if the epoch is the occupancy midpoint or partial science scales occupancy.

    Complete visit: science is the last exposure_ticks of settle + exposure and
    the epoch is start + settle + exposure / 2 (within half a tick). Cancelled
    during settling: no science. Cancelled during science: the realized duration
    is end - start - settle, not the completed fraction of the occupancy times
    the planned exposure.
    """
    settle = context.settle_ticks
    assert settle == DAY  # orbix Observatory settling time, 1 day
    start, exposure = 5 * DAY + 123, DAY // 2
    end = start + int(context.occupancy_ticks(exposure))
    s0, s1, has = context.science_interval(start, end)
    assert (int(s0), int(s1), bool(has)) == (end - exposure, end, True)
    epoch = float(context.science_midpoint_jd(start, end))
    want = float(context.clock.jd(start + settle + exposure // 2))
    assert abs(epoch - want) * DAY <= 0.5
    occupancy_mid = float(context.clock.jd(start + (end - start) // 2))
    assert abs(epoch - occupancy_mid) * DAY > settle // 4
    _, _, has = context.science_interval(start, start + settle // 2)
    assert not bool(has)
    s0, s1, _ = context.science_interval(start, start + settle + DAY // 8)
    assert int(s1 - s0) == DAY // 8


@pytest.mark.parametrize("target", [UMA, TWIN], ids=["47uma", "twin"])
def test_keepout_window_length_against_spherical_trigonometry(context, target):
    """Fails if the Sun keepout limit, the halo orbit or the target coordinates slip.

    Days per year inside a 45 degree Sun keepout, sampled every 6 hours over two
    years, equal 2 arccos(cos 45 deg / cos beta) / 360 deg of a year at the
    target's ecliptic latitude beta, within 2 days per year (about 70 days for
    47 UMa at beta = 31 deg, about 91 days for the twin on the ecliptic).
    """
    entry = PROFILE["targets"][target]
    star = SkyCoord(entry["ra_deg"] * u.deg, entry["dec_deg"] * u.deg)
    beta = star.transform_to(BarycentricMeanEcliptic()).lat.radian
    half_width = np.arccos(np.cos(np.radians(45.0)) / np.cos(beta))
    expected = 2.0 * half_width / (2.0 * np.pi) * 365.25
    days = np.arange(0.0, 730.0, 0.25)
    ok = np.asarray(context.observable(target, context.clock.jd(0) + days))
    blocked = (~ok).sum() * 0.25 / 2.0
    assert abs(blocked - expected) <= 2.0, (blocked, expected)


def test_certification_outcomes_are_explicit(context, edge):
    """Fails if a straddling or invisible visit certifies, or a refusal is silent.

    Inside the window: certified. Occupancy across the keepout edge: KEEPOUT_EDGE.
    Planned start inside but the plan's slack reaching across the edge:
    KEEPOUT_EDGE (a check of the planned start alone would certify it). Wholly in
    keepout: NO_VISIBILITY. Zero exposure: NO_SCIENCE. A support predicate that
    fails somewhere: OUTSIDE_SUPPORT.
    """
    half = DAY // 2

    def outcome(start, exposure=half, delay=0, support=None):
        visit = PlannedVisit(UMA, start, exposure, delay)
        return Certification(int(context.certify(visit, support).outcome))

    assert outcome(edge - 10 * DAY) is Certification.CERTIFIED
    assert outcome(edge - DAY) is Certification.KEEPOUT_EDGE
    assert outcome(edge - 5 * DAY, delay=0) is Certification.CERTIFIED
    assert outcome(edge - 5 * DAY, delay=5 * DAY) is Certification.KEEPOUT_EDGE
    assert outcome(edge + 20 * DAY) is Certification.NO_VISIBILITY
    assert outcome(edge - 10 * DAY, exposure=0) is Certification.NO_SCIENCE
    cutoff = float(context.clock.jd(edge - 10 * DAY + DAY))

    def support(target, jd, margin_d):
        return jnp.asarray(jd) + margin_d < cutoff

    assert outcome(edge - 10 * DAY, support=support) is Certification.OUTSIDE_SUPPORT


def test_certification_is_conservative_at_the_edge(context, edge):
    """Fails if the keepout margin is dropped or smaller than the sampling needs.

    An occupancy ending one hour before the edge is refused (the margin covers
    the angle change over half a grid step); one ending two days before is
    certified. The margin's rate bound covers the fastest Sun and Earth angle
    change seen from the halo orbit over two years for both targets.
    """
    hour = DAY // 24
    occupancy = int(context.occupancy_ticks(DAY // 2))
    near = PlannedVisit(UMA, edge - hour - occupancy, DAY // 2)
    far = PlannedVisit(UMA, edge - 2 * DAY - occupancy, DAY // 2)
    assert int(context.certify(near).outcome) == Certification.KEEPOUT_EDGE
    assert int(context.certify(far).outcome) == Certification.CERTIFIED
    mjd = jnp.asarray(context.clock.mjd(0)) + jnp.arange(0.0, 730.0, 0.1)
    orbit = context.observatory.orbit
    rates = []
    for target in (UMA, TWIN):
        ra = jnp.radians(context.target_ra_deg[target])
        dec = jnp.radians(context.target_dec_deg[target])
        for body in (lambda m: jnp.zeros(3), earth_position_ecliptic):

            def angle(m, body=body, ra=ra, dec=dec):
                return body_angle(orbit.position_ecliptic(m), body(m), ra, dec, m)

            degrees = np.degrees(np.asarray(jax.vmap(angle)(mjd)))
            rates.append(np.max(np.abs(np.diff(degrees))) / 0.1)
    assert max(rates) < context.angle_rate_bound_deg_per_day, rates


def test_uncertified_plan_is_refused_at_load(context, edge):
    """Fails if a plan with an invisible visit loads, or the refusal hides which visit.

    B5 on the context alone: the visit in keepout is named with NO_VISIBILITY and
    the certified visit is not listed.
    """
    good = PlannedVisit(UMA, edge - 10 * DAY, DAY // 2)
    bad = PlannedVisit(UMA, edge + 20 * DAY, DAY // 2)
    assert len(require_certified(context, [good])) == 1
    with pytest.raises(UncertifiedPlan) as info:
        require_certified(context, [good, bad])
    assert len(info.value.findings) == 1
    assert "visit 1" in info.value.findings[0]
    assert "NO_VISIBILITY" in info.value.findings[0]


def test_realized_science_rechecks_the_certified_window(context, edge):
    """Fails if a visit delayed across the keepout edge yields science or no charge.

    Certified with two days of slack: a start inside the slack gives valid
    science of the full exposure; an admission delay beyond the slack that moves
    the occupancy across the edge gives valid_science = 0 with the occupied time
    still reported; a cancellation during settling gives valid_science = 0.
    """
    exposure = DAY
    start = edge - 6 * DAY
    certificate = context.certify(PlannedVisit(UMA, start, exposure, 2 * DAY))
    assert int(certificate.outcome) == Certification.CERTIFIED
    occupancy = int(context.occupancy_ticks(exposure))
    inside = context.realized_science(certificate, start + DAY, start + DAY + occupancy)
    assert (int(inside.valid_science), int(inside.science_ticks)) == (1, exposure)
    late = edge - DAY
    crossed = context.realized_science(certificate, late, late + occupancy)
    assert (int(crossed.valid_science), int(crossed.science_ticks)) == (0, 0)
    assert int(crossed.occupied_ticks) == occupancy
    cancelled = context.realized_science(certificate, start, start + DAY // 2)
    assert (int(cancelled.valid_science), int(cancelled.occupied_ticks)) == (
        0,
        DAY // 2,
    )


@pytest.mark.parametrize(
    ("visit", "message"),
    [
        (PlannedVisit(2, 0, DAY), "not a known target"),
        (PlannedVisit(UMA, 0, -1), "nonnegative"),
        (PlannedVisit(UMA, 0, DAY, -1), "nonnegative"),
    ],
    ids=["unknown-target", "negative-exposure", "negative-delay"],
)
def test_malformed_visits_are_refused(context, visit, message):
    """Fails if a malformed visit is certified or fails with an unnamed error.

    An empty epoch list is answered with an empty array, not an index error.
    """
    with pytest.raises(ValueError, match=message):
        context.certify(visit)
    assert context.observable(UMA, []).shape == (0,)


def test_support_certification_covers_epochs_between_samples(context, edge):
    """Fails if the support is checked only at the sampled epochs.

    An edge-on circular orbit with a 0.5 day period, phased so that every
    quarter-day sample falls at greatest elongation, passes every sample of an
    inner working angle at half its elongation, yet reaches the star at each
    conjunction between samples. Certification must refuse it; a slow orbit with
    the same band stays certified.
    """
    start = edge - 10 * DAY
    fast = dataclasses.replace(
        SceneSpec.from_profile(PROFILE, UMA),
        period_d=0.5,
        ecc=0.0,
        omega_star_deg=180.0,
        inclination_deg=90.0,
        tp_jd=float(context.clock.jd(start)),
    )
    system = build_scene(fast)
    elongation = float(
        geometry(system, [fast.tp_jd], fast.wavelength_nm).separation_arcsec[0]
    )
    support = SeparationSupport((system,), (fast.wavelength_nm,), elongation / 2, 1.0)
    visit = PlannedVisit(UMA, start, DAY // 2)
    samples = context.clock.jd(start + np.arange(0, 7) * (DAY // 4))
    assert bool(jnp.all(support(UMA, samples)))  # every sample passes
    midway = context.clock.jd(start + DAY // 8)
    assert not bool(support(UMA, midway)[0])  # but not between samples
    outcome = Certification(int(context.certify(visit, support).outcome))
    assert outcome is Certification.OUTSIDE_SUPPORT
    slow = SeparationSupport(
        (build_scene(SceneSpec.from_profile(PROFILE, UMA)),), (1000.0,), 0.0, 1.0
    )
    assert int(context.certify(visit, slow).outcome) == Certification.CERTIFIED
