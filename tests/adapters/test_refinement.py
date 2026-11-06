"""Time refinement of the summary likelihood quantities on a fast-moving planet."""

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Rearth2AU
from hwoutils.conversions import Msun_to_kg

from spaceodyssey.adapters.jaxedith import (
    ReportingLaw,
    ScienceInterval,
    SummaryRow,
    SummaryScene,
    loglike_report,
    nominal_planet,
    observed_snr,
)
from tests.adapters.conftest import EPOCH_JD

LAW = ReportingLaw()  # the declared law: threshold 5, kappa 1, corr 0


def likelihood_quantities(budget, record, lod_arcsec):
    """Counts, log p(f, pos | theta) and P(null | theta) under the reporting law.

    The record is the observation and stays fixed: a detection with its counts,
    position and the planet-free `limit_counts` published with it; its snr and
    position errors come from those observables. The null probability is the one the
    same record's limit implies.
    """
    f, ra, dec, limit_counts = record
    snr = observed_snr(jnp.asarray(f), limit_counts)
    sigma = LAW.position_error(snr, lod_arcsec)
    row = SummaryRow(
        time_d=jnp.asarray(0.0),
        science_s=budget.science_s,
        occupied_s=budget.science_s,
        detected=jnp.asarray(1.0),
        counts=jnp.asarray(f),
        limit_counts=jnp.asarray(limit_counts),
        snr=snr,
        ra=jnp.asarray(ra),
        dec=jnp.asarray(dec),
        ra_err=sigma,
        dec_err=sigma,
        corr=jnp.asarray(0.0),
        cadence=jnp.asarray(1.0),
        calibration_id=jnp.asarray(1.0),
        valid_science=jnp.asarray(1.0),
    )
    log_p = loglike_report(LAW, row, budget)
    p_null = jnp.exp(LAW.log_null_probability(limit_counts, budget))
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
    # covariance it implies do not move along the ladder. The planet-free limit is
    # the reference's own background and detector variance.
    s0 = float(LAW.limit_counts(reference))
    record = (
        1.1 * float(reference.planet),
        float(reference.ra_arcsec) + 1e-3,
        float(reference.dec_arcsec) - 1e-3,
        s0,
    )
    assert record[0] >= LAW.threshold * s0  # a detection
    lod = float(base.lod_arcsec)
    truth = likelihood_quantities(reference, record, lod)
    assert 0.05 < truth[2] < 0.95  # the null probability is informative
    errors = np.array(
        [
            np.abs(likelihood_quantities(at(n), record, lod) - truth)
            for n in (1, 4, 8, 16)
        ]
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
