"""Time refinement of the summary likelihood quantities on a fast-moving planet."""

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Rearth2AU
from hwoutils.conversions import Msun_to_kg
from jax.scipy.stats import norm

from spaceodyssey.adapters.jaxedith import (
    ScienceInterval,
    SummaryScene,
    nominal_planet,
)
from tests.adapters.conftest import EPOCH_JD

TAU = 5.0  # detection threshold in planet-free standard deviations


def likelihood_quantities(budget, record):
    """Counts, log p(f, pos | theta) and P(null | theta) under the joint law.

    `s` is the standard deviation the candidate predicts for the measured counts. The
    record is the observation and stays fixed: its counts, position, lambda/D and
    `limit_counts = TAU * s0`, the planet-free reference published with it. The
    position error per axis is (lambda/D) / snr with the snr from the record's
    observables only.
    """
    s = jnp.sqrt(budget.variance)
    f, ra, dec, lod, limit_counts = record
    s0 = limit_counts / TAU
    snr = f / jnp.sqrt(s0**2 + f)
    sigma = lod / snr
    log_p = (
        norm.logpdf(f, budget.planet, s)
        + norm.logpdf(ra, budget.ra_arcsec, sigma)
        + norm.logpdf(dec, budget.dec_arcsec, sigma)
    )
    p_null = norm.cdf((limit_counts - budget.planet) / s)
    return np.array([float(budget.planet), float(log_p), float(p_null)])


@pytest.mark.slow  # Compiles the budget at five node counts on the real response.
def test_b4_time_refinement_of_the_likelihood_quantities(yip_report):
    """Fails if counts, log p(f, pos | theta) or P(null | theta) do not converge.

    A fast mover (a 0.6 AU edge-on orbit at 4 pc, 0.46 lambda/D of motion in a
    five-day exposure, along the throughput rise just outside the inner working
    angle, with the planet near the detection threshold) is integrated with 4, 8
    and 16 Gauss-Legendre nodes against a 64-node reference, all scored against one
    fixed observation (counts, position and limit counts). Each quantity either
    sits at roundoff from the first level (the counts: the throughput is one smooth
    table segment over the arc) or falls at every level with an observed order over
    the ladder of at least 2 (the midpoint rule's order; the leakage table is
    piecewise cubic, so no spectral order is claimed). The 8-node default sits
    inside the allocation: 1e-4 relative on counts, 0.01 in log p and 1e-3 in
    P(null). The single-node (midpoint) evaluation misses the allocation, so the
    fixture resolves the motion.
    """
    scene = SummaryScene(
        star_flux_density_jy=42.0,
        dist_pc=4.0,
        star_mass_kg=Msun_to_kg(1.0),
        luminosity_lsun=1.0,
        ra_deg=164.867,
        dec_deg=40.43,
        star_diameter_mas=0.0,
        n_exozodi=3.0,
    )
    base = eqx.tree_at(lambda r: r.nominal, yip_report, yip_report.prepare(scene))
    theta = nominal_planet(
        a_au=0.6,
        e=0.0,
        inclination_rad=0.5 * math.pi,
        W_rad=0.4,
        w_rad=0.0,
        tp_d=0.0,
        Rp_au=0.19 * Rearth2AU,
        Ag=0.3,
    )
    # From 4.7 to 4.3 lambda/D, near the threshold: N is about 5 s0.
    period_d = 365.25 * 0.6**1.5
    interval = ScienceInterval.from_start(EPOCH_JD + 0.069 * period_d, 5 * 86400.0)

    def at(n):
        x, w = np.polynomial.legendre.leggauss(n)
        report = eqx.tree_at(
            lambda r: (r.nodes, r.weights), base, (jnp.asarray(x), jnp.asarray(w))
        )
        return report.predict(theta, interval)

    reference = at(64)
    # One fixed observation for every level: the threshold and the astrometric
    # covariance it implies do not move along the ladder.
    s0 = math.sqrt(float(reference.var_background + reference.var_detector))
    record = (
        0.9 * float(reference.planet),
        float(reference.ra_arcsec) + 1e-3,
        float(reference.dec_arcsec) - 1e-3,
        float(base.lod_arcsec),
        TAU * s0,
    )
    truth = likelihood_quantities(reference, record)
    assert 0.05 < truth[2] < 0.95  # the null probability is informative
    errors = np.array(
        [np.abs(likelihood_quantities(at(n), record) - truth) for n in (1, 4, 8, 16)]
    )
    errors[:, 0] /= truth[0]
    allocation = np.array([1e-4, 1e-2, 1e-3])
    assert np.any(errors[0] > allocation)
    assert np.all(errors[2] < allocation)
    for q in range(3):
        e4, e8, e16 = errors[1:, q]
        if e4 < 1e-10:
            continue  # at roundoff from the first level on
        assert e4 > e8 > e16
        assert math.log2(e4 / e16) / 2 >= 2.0
