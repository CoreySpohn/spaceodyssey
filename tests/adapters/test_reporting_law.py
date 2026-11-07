"""The summary reporting law: sampling against likelihood, normalization, schema.

Unit tests drive the law with hand-built count budgets, so every expected value is a
closed form in `scipy.stats`. The survey (slow tier) runs the law over planned visits
of 47 UMa b on the pinned public response for 200 worlds drawn from the prior.
"""

import dataclasses
import itertools
import math
import re

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from photomancy.orbit.data import ImagingData, NullData
from photomancy.orbit.likelihoods import loglike_imaging, loglike_null
from scipy import stats

from spaceodyssey.adapters.jaxedith import (
    CADENCE_CODES,
    SUMMARY_COLUMNS,
    CountBudget,
    ReportingLaw,
    SummaryRow,
    calibration_id,
    loglike_report,
    observed_snr,
    sample_report,
    summary_kind,
)
from spaceodyssey.adapters.physical_profile import audience_flags, load_profile
from spaceodyssey.core.records import KindRegistry, RecordKind

PROFILE = load_profile()
LOD = 0.0286  # lambda/D at 1 micron on a 7.2 m aperture [arcsec]
S0 = 10.0  # planet-free noise standard deviation of the hand budgets [e-]
MU = (0.12, -0.05)  # expected centroid of the hand budgets [arcsec]
N_DRAWS = 20_000
# Expected counts at three levels: below, at and above the threshold of 5 s0.
LEVELS = (3.0 * S0, 5.0 * S0, 7.0 * S0)


def hand_budget(planet, var_background=S0**2, *, science_s=3600.0, ra=MU[0]):
    """A count budget with planet shot noise, so s(theta) = sqrt(N + B) != s0."""
    z = jnp.asarray(0.0)
    return CountBudget(
        planet=jnp.asarray(planet, jnp.float64),
        leakage=jnp.asarray(var_background, jnp.float64),
        zodi=z,
        exozodi=z,
        thermal=z,
        var_planet=jnp.asarray(planet, jnp.float64),
        var_background=jnp.asarray(var_background, jnp.float64),
        var_detector=z,
        var_systematic=z,
        ra_arcsec=jnp.asarray(ra, jnp.float64),
        dec_arcsec=jnp.asarray(MU[1]),
        science_s=jnp.asarray(science_s, jnp.float64),
        in_field=jnp.asarray(True),
        cadence=jnp.asarray(CADENCE_CODES["photon_counting_legacy"]),
    )


def draw(law, budget, n=N_DRAWS, seed=0, limit=S0, **kwargs):
    """`n` rows of `budget` under `law`."""
    options = dict(lod_arcsec=LOD, time_d=2462600.5, occupied_s=90000.0)
    options.update(calibration_id=4242, **kwargs)
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    return eqx.filter_jit(
        jax.vmap(lambda k: sample_report(k, law, budget, limit, **options))
    )(keys)


def whitened_residuals(rows):
    """The detections' position residuals whitened by each row's own covariance."""
    keep = np.asarray(rows.detected) == 1
    rho = np.asarray(rows.corr)[keep]
    u1 = (np.asarray(rows.ra)[keep] - MU[0]) / np.asarray(rows.ra_err)[keep]
    v = (np.asarray(rows.dec)[keep] - MU[1]) / np.asarray(rows.dec_err)[keep]
    return u1, (v - rho * u1) / np.sqrt(1 - rho**2)


def whitened_chi2(law, rows, budget):
    """The Mahalanobis distance of each detection's position, read off the likelihood.

    The position density is ``-log(2 pi sx sy sqrt(1 - rho^2)) - chi2 / 2``, so the
    likelihood itself gives chi2, which is chi-square with two degrees of freedom when
    sampling and likelihood agree.
    """
    keep = np.asarray(rows.detected) == 1
    lp = jax.vmap(lambda r: law.log_position_density(r, budget, jnp.asarray(True)))(
        rows
    )
    norm = jnp.log(2 * jnp.pi * rows.ra_err * rows.dec_err * jnp.sqrt(1 - rows.corr**2))
    return np.asarray(-2.0 * (lp + norm))[keep]


# -- Declarations ----------------------------------------------------------------


def test_law_refuses_undeclared_values():
    """Fails if a nonpositive threshold or kappa, a bad corr or a flag is accepted."""
    for bad in (
        dict(threshold=0.0),
        dict(threshold=-5.0),
        dict(threshold=True),
        dict(kappa=0.0),
        dict(kappa=math.nan),
        dict(corr=1.0),
        dict(corr=-1.0),
        dict(corr=math.inf),
        dict(corr=None),
    ):
        with pytest.raises(ValueError):
            ReportingLaw(**bad)
    assert ReportingLaw() == ReportingLaw(threshold=5.0, kappa=1.0, corr=0.0)
    with pytest.raises(ValueError, match="15 columns"):
        SummaryRow.from_payload(jnp.zeros(14))


def test_record_kind_passes_kp5_against_the_profile_table():
    """Fails if the record kind and the profile table disagree on any column.

    The kind's columns are the profile's payload columns in order, each with the
    profile's audience; the row type lists them in the same order; the cadence
    codes are the ones the profile's `cadence` column states; and the kind
    registers and lowers to fifteen readable columns. An incomplete or reordered
    profile is refused.
    """
    kind = summary_kind()
    columns = PROFILE["payload_columns"]
    assert tuple(c.name for c in kind.columns) == SUMMARY_COLUMNS
    assert tuple(c["name"] for c in columns) == SUMMARY_COLUMNS
    assert [c.audience for c in kind.columns] == [
        int(audience_flags(c["audience"])) for c in columns
    ]
    assert tuple(f.name for f in dataclasses.fields(SummaryRow)) == SUMMARY_COLUMNS
    sign = next(c["sign"] for c in columns if c["name"] == "cadence")
    stated = {name: int(code) for code, name in re.findall(r"(\d+) (\w+)", sign)}
    assert stated == CADENCE_CODES
    caps = KindRegistry((RecordKind("none", ()), kind)).lower()
    assert caps.payload_width == len(SUMMARY_COLUMNS)
    assert all(a != 0 for a in caps.payload_audience[1])

    unitless = dict(PROFILE, payload_columns=[dict(columns[0], unit=" "), *columns[1:]])
    with pytest.raises(ValueError, match=r"time_d\.unit"):
        summary_kind(unitless)
    swapped = dict(PROFILE, payload_columns=[columns[1], columns[0], *columns[2:]])
    with pytest.raises(ValueError, match="not the summary columns"):
        summary_kind(swapped)


# -- Sampling against likelihood (B2) --------------------------------------------


@pytest.mark.slow  # A 20,000-draw Monte Carlo panel (the tier the spec assigns).
@pytest.mark.parametrize("corr", [0.0, 0.5])
def test_sampling_and_likelihood_agree(corr):
    """Fails if draws and the likelihood disagree beyond Monte Carlo error.

    At three signal levels (3, 5 and 7 s0, with s(theta) = sqrt(N + s0^2) > s0),
    20,000 draws each: the detection frequency equals 1 - P(null | theta) from the
    likelihood within 3 standard errors; retained counts follow the normal truncated
    at 5 s0 (Kolmogorov-Smirnov p above 0.01); and the position's Mahalanobis
    distance read off the likelihood is chi-square with two degrees of freedom
    (p above 0.01), including the correlated case; the whitened residuals have zero
    mean and no cross-correlation within 4 standard errors (nine moments per test,
    so 4 rather than 3 keeps the family-wise false alarm below 1e-3). Controls that
    fail the same check: a model-side threshold of 5.5, the fixed-denominator null
    form with s0 in place of s(theta), and a doubled variance.
    """
    law = ReportingLaw(corr=corr)
    for level in LEVELS:
        budget = hand_budget(level)
        rows = draw(law, budget, seed=int(level))
        detected = np.asarray(rows.detected)
        p_det = 1.0 - math.exp(float(law.log_null_probability(S0, budget)))
        se = math.sqrt(p_det * (1 - p_det) / N_DRAWS)
        assert abs(detected.mean() - p_det) < 3 * se

        s = math.sqrt(level + S0**2)
        assert s > 1.1 * S0
        counts = np.asarray(rows.counts)[detected == 1]
        a = (5.0 * S0 - level) / s
        truncated = stats.truncnorm(a, np.inf, loc=level, scale=s)
        assert stats.kstest(counts, truncated.cdf).pvalue > 0.01
        chi2 = whitened_chi2(law, rows, budget)
        assert stats.kstest(chi2, stats.chi2(2).cdf).pvalue > 0.01
        u1, u2 = whitened_residuals(rows)
        for moment in (u1, u2, u1 * u2):
            assert abs(moment.mean()) < 4.0 / math.sqrt(len(u1))

        wrong_law = ReportingLaw(threshold=5.5)
        wrong = 1.0 - math.exp(float(wrong_law.log_null_probability(S0, budget)))
        fixed = 1.0 - stats.norm.cdf((5.0 * S0 - level) / S0)
        doubled = 1.0 - stats.norm.cdf((5.0 * S0 - level) / (math.sqrt(2) * s))
        assert abs(detected.mean() - wrong) > 3 * se
        if level != 5.0 * S0:  # every form gives one half at the threshold
            assert abs(detected.mean() - fixed) > 3 * se
            assert abs(detected.mean() - doubled) > 3 * se


@pytest.mark.slow  # The correlated-error control of the 20,000-draw panel.
def test_position_whitening_control_fails_on_the_correlated_case():
    """Fails if the chi-square check cannot see a correlation sign error.

    Positions drawn with corr = 0.5 and scored by a likelihood whose rows claim
    corr = -0.5 fail the two-degree chi-square check.
    """
    law = ReportingLaw(corr=0.5)
    budget = hand_budget(7.0 * S0)
    rows = draw(law, budget, seed=7)
    flipped = eqx.tree_at(lambda r: r.corr, rows, -rows.corr)
    chi2 = whitened_chi2(law, flipped, budget)
    assert stats.kstest(chi2, stats.chi2(2).cdf).pvalue < 1e-6


@pytest.mark.parametrize("corr", [0.0, 0.5])
def test_joint_detection_and_null_probability_normalizes(corr):
    """Fails if detection density plus null probability does not integrate to one.

    The detection density of `loglike_report` is integrated over counts above the
    threshold and over the position (Gauss-Legendre in each axis, the position in
    the row's own whitened coordinates) and the null row's probability is added, at
    three signal levels with s(theta) != s0: the total is one to 1e-9. A row
    flagged detected below the threshold lies outside the support. Control:
    the fixed-denominator null of photomancy's form, with s0 in place of s(theta),
    misses one by more than 1e-3 below the threshold.
    """
    law = ReportingLaw(corr=corr)
    xf, wf = np.polynomial.legendre.leggauss(240)
    xu, wu = np.polynomial.legendre.leggauss(48)
    u1, u2 = (g.ravel() * 10.0 for g in np.meshgrid(xu, xu, indexing="ij"))
    wpos = np.outer(wu, wu).ravel() * 100.0

    def row(counts, ra, dec, detected):
        snr = counts / jnp.sqrt(S0**2 + counts) * detected
        err = jnp.where(detected > 0, LOD / jnp.where(snr > 0, snr, 1.0), 0.0)
        return SummaryRow(
            time_d=jnp.asarray(0.0),
            science_s=jnp.asarray(3600.0),
            occupied_s=jnp.asarray(90000.0),
            detected=jnp.asarray(detected, jnp.float64),
            counts=counts,
            limit_counts=jnp.asarray(S0),
            snr=snr,
            ra=ra,
            dec=dec,
            ra_err=err,
            dec_err=err,
            corr=jnp.asarray(corr),
            cadence=jnp.asarray(1.0),
            calibration_id=jnp.asarray(1.0),
            valid_science=jnp.asarray(1.0),
        )

    @eqx.filter_jit
    def detection_mass(budget, f):
        snr = f / jnp.sqrt(S0**2 + f)
        sigma = LOD / snr
        ra = MU[0] + sigma * u1
        dec = MU[1] + sigma * (corr * u1 + math.sqrt(1 - corr**2) * u2)
        jac = sigma**2 * math.sqrt(1 - corr**2)
        ll = jax.vmap(lambda a, b: loglike_report(law, row(f, a, b, 1.0), budget))(
            ra, dec
        )
        return jnp.sum(wpos * jnp.exp(ll)) * jac

    for level in LEVELS:
        budget = hand_budget(level)
        s = math.sqrt(level + S0**2)
        lo, hi = 5.0 * S0, max(level, 5.0 * S0) + 14.0 * s
        f = 0.5 * (hi - lo) * xf + 0.5 * (hi + lo)
        masses = jax.vmap(lambda x, b=budget: detection_mass(b, x))(jnp.asarray(f))
        detection = 0.5 * (hi - lo) * float(jnp.sum(wf * masses))
        below = row(jnp.asarray(4.9 * S0), jnp.asarray(MU[0]), jnp.asarray(MU[1]), 1.0)
        assert float(loglike_report(law, below, budget)) == -math.inf
        null_row = row(jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0), 0.0)
        null = math.exp(float(loglike_report(law, null_row, budget)))
        assert detection + null == pytest.approx(1.0, abs=1e-9)
        if level < 5.0 * S0:
            fixed = stats.norm.cdf((5.0 * S0 - level) / S0)
            assert abs(detection + fixed - 1.0) > 1e-3


# -- Observables only --------------------------------------------------------------


def test_payload_carries_observables_only():
    """Fails if any payload column depends on an expected-truth quantity.

    For each of 64 keys, a second truth budget with twice the noise and the
    expected counts moved so that the realized counts are the same draw gives the
    identical row in every column: no column can carry the expected counts, the
    expected noise or the expected signal-to-noise. `snr`, `ra_err` and `dec_err`
    are recomputed from the row's own `counts` and `limit_counts`.
    """
    law = ReportingLaw(corr=0.5)
    first = hand_budget(6.0 * S0)
    s1 = math.sqrt(6.0 * S0 + S0**2)
    rows = draw(law, first, n=64, seed=3)
    n_detected = 0
    for i in range(64):
        one = jax.tree_util.tree_map(lambda x, i=i: x[i], rows)
        key = jax.random.split(jax.random.PRNGKey(3), 64)[i]
        eps0 = float(jax.random.normal(key, (3,), dtype=jnp.float64)[0])
        s2 = 2.0 * s1
        planet2 = 6.0 * S0 + (s1 - s2) * eps0
        second = hand_budget(planet2, s2**2 - planet2)
        other = sample_report(
            key,
            law,
            second,
            S0,
            lod_arcsec=LOD,
            time_d=2462600.5,
            occupied_s=90000.0,
            calibration_id=4242,
        )
        assert float(second.planet / jnp.sqrt(second.variance)) != pytest.approx(
            float(first.planet / jnp.sqrt(first.variance)), rel=1e-3
        )
        np.testing.assert_allclose(other.payload(), one.payload(), rtol=1e-12)
        n_detected += int(one.detected)
        snr = float(observed_snr(one.counts, one.limit_counts))
        hand = one.counts / math.sqrt(one.limit_counts**2 + max(one.counts, 0.0))
        assert snr == pytest.approx(float(hand), rel=1e-14)
        assert float(one.snr) == pytest.approx(snr, rel=1e-14)
        if one.detected:
            assert float(one.ra_err) == pytest.approx(LOD / snr, rel=1e-14)
            assert float(one.dec_err) == float(one.ra_err)
    assert 0 < n_detected < 64


def test_null_and_no_science_rows():
    """Fails if a null or an empty interval publishes position or counts.

    A null row carries its limit, epoch, durations and declarations and zero
    counts, signal-to-noise, position and errors. A row with no science time (or a
    visit flagged invalid) carries no limit and contributes zero log likelihood;
    a non-finite truth gives NaN, never a silent null.
    """
    law = ReportingLaw()
    rows = draw(law, hand_budget(2.0 * S0), n=256)
    null = np.asarray(rows.detected) == 0
    assert null.sum() > 200
    payload = np.asarray(rows.payload())[null]
    for name in ("counts", "snr", "ra", "dec", "ra_err", "dec_err"):
        assert np.all(payload[:, SUMMARY_COLUMNS.index(name)] == 0.0)
    assert np.all(payload[:, SUMMARY_COLUMNS.index("limit_counts")] == S0)
    assert np.all(payload[:, SUMMARY_COLUMNS.index("valid_science")] == 1.0)

    for empty in (
        draw(law, hand_budget(9.0 * S0, science_s=0.0), n=8),
        draw(law, hand_budget(9.0 * S0), n=8, valid_science=False),
        draw(law, hand_budget(9.0 * S0), n=8, limit=0.0),
    ):
        assert np.all(np.asarray(empty.valid_science) == 0.0)
        assert np.all(np.asarray(empty.limit_counts) == 0.0)
        assert np.all(np.asarray(empty.detected) == 0.0)
        ll = jax.vmap(lambda r: loglike_report(law, r, hand_budget(9.0 * S0)))(empty)
        assert np.all(np.asarray(ll) == 0.0)

    for bad in (hand_budget(math.nan), hand_budget(9.0 * S0, math.inf)):
        row = draw(law, bad, n=1)
        assert np.isnan(row.detected).all() and np.isnan(row.counts).all()
        one = jax.tree_util.tree_map(lambda x: x[0], row)
        assert math.isnan(float(loglike_report(law, one, hand_budget(9.0 * S0))))
    payload = rows.payload()
    back = SummaryRow.from_payload(payload)
    np.testing.assert_array_equal(back.payload(), payload)


def test_kappa_scales_only_the_position_errors():
    """Fails if kappa changes anything but the position errors and offsets.

    With the same draws, kappa = 2 leaves detection, counts and snr unchanged and
    doubles `ra_err`, `dec_err` and each position's offset from the centroid.
    """
    one = draw(ReportingLaw(kappa=1.0, corr=0.5), hand_budget(6.0 * S0), n=512)
    two = draw(ReportingLaw(kappa=2.0, corr=0.5), hand_budget(6.0 * S0), n=512)
    for name in ("detected", "counts", "snr", "limit_counts"):
        np.testing.assert_array_equal(getattr(two, name), getattr(one, name))
    np.testing.assert_allclose(two.ra_err, 2.0 * one.ra_err, rtol=1e-14)
    np.testing.assert_allclose(two.dec_err, 2.0 * one.dec_err, rtol=1e-14)
    keep = np.asarray(one.detected) == 1
    for name, mu in (("ra", MU[0]), ("dec", MU[1])):
        a = np.asarray(getattr(one, name))[keep] - mu
        b = np.asarray(getattr(two, name))[keep] - mu
        np.testing.assert_allclose(b, 2.0 * a, rtol=1e-9, atol=1e-15)


def test_always_null_scene_gives_no_information():
    """Fails if nulls of a planet that never shows carry information about theta.

    A proposal with no planet counts whose background differs from the reference by
    up to 10 percent gives a null log likelihood within 1e-5 of zero, so a run of
    nulls cannot move a posterior between such proposals.
    """
    law = ReportingLaw()
    null = jax.tree_util.tree_map(lambda x: x[0], draw(law, hand_budget(0.0), n=1))
    assert float(null.detected) == 0.0
    for scale in (0.9, 1.0, 1.1):
        for ra in (-0.3, 0.0, 0.4):
            proposal = hand_budget(0.0, scale * S0**2, ra=ra)
            assert -1e-5 < float(loglike_report(law, null, proposal)) <= 0.0


# -- photomancy cross-check ---------------------------------------------------------


def photomancy_null(level, limit, threshold=5.0):
    """photomancy's `loglike_null` at one epoch with a flat `threshold * limit` limit.

    Counts are written as magnitudes against a reference of one count, so the flux
    ratio photomancy forms is ``level / (threshold * limit)``.
    """
    dmag0 = -2.5 * math.log10(threshold * limit)
    data = NullData(
        epochs=jnp.zeros(1),
        sep_grid=jnp.asarray([[0.0, 1.0]]),
        dmag0_grid=jnp.full((1, 2), dmag0),
        is_valid=jnp.ones(1, bool),
        snr_thresh=threshold,
    )
    dmag = -2.5 * math.log10(max(level, 1e-300))
    return float(loglike_null(jnp.asarray([0.2]), jnp.asarray([dmag]), data))


def test_photomancy_null_coincides_only_when_s_equals_s0():
    """Fails if the law and photomancy's null disagree where they must coincide.

    With the planet's shot noise removed (s(theta) = s0) the law's log P(null)
    equals photomancy's `loglike_null` and the null branch of `loglike_imaging` to
    1e-12 at five levels. With shot noise the law penalizes a planet below the
    threshold more and one above it less than photomancy's fixed-denominator form,
    and both give one half at the threshold.
    """
    law = ReportingLaw()
    for level in (0.0, 2.0 * S0, 4.0 * S0, 5.0 * S0, 8.0 * S0):
        flat = eqx.tree_at(lambda b: b.var_planet, hand_budget(level), jnp.asarray(0.0))
        ours = float(law.log_null_probability(S0, flat))
        theirs = photomancy_null(level, S0)
        assert ours == pytest.approx(theirs, abs=1e-12)
        imaging = ImagingData(
            epochs=jnp.zeros(1),
            sep_grid=jnp.asarray([[0.0, 1.0]]),
            dmag0_grid=jnp.full((1, 2), -2.5 * math.log10(5.0 * S0)),
            is_detected=jnp.zeros(1, bool),
            dmag_obs=jnp.zeros(1),
            dmag_err=jnp.ones(1),
            is_valid=jnp.ones(1, bool),
        )
        dmag = -2.5 * math.log10(max(level, 1e-300))
        assert float(
            loglike_imaging(jnp.asarray([0.2]), jnp.asarray([dmag]), imaging)
        ) == pytest.approx(theirs, abs=1e-12)
        shot = float(law.log_null_probability(S0, hand_budget(level)))
        if 0.0 < level < 5.0 * S0:
            assert shot < theirs
        elif level > 5.0 * S0:
            assert shot > theirs
        elif level == 5.0 * S0:
            assert shot == pytest.approx(math.log(0.5), abs=1e-12)


# -- Calibration id ------------------------------------------------------------------


def _renominal(report, where, value):
    """`report` with one field replaced and its nominal scene prepared again."""
    out = eqx.tree_at(where, report, value)
    return eqx.tree_at(lambda r: r.nominal, out, out.prepare(out.nominal.scene))


def _mutations(observatory):
    """One declaration change per entry: (name, mutated report or law)."""
    from tests.adapters.test_jaxedith_summary import (
        ToyCoronagraph,
        toy_report,
        toy_scene,
    )

    base = toy_report(observatory)
    spec = type(base.detector_spec)
    halo = observatory.orbit._interp_x
    return [
        ("threshold", ReportingLaw(threshold=7.0)),
        ("kappa", ReportingLaw(kappa=2.0)),
        ("corr", ReportingLaw(corr=0.5)),
        (
            "dark_current",
            toy_report(observatory, spec=spec.fixture(dark_current_rate_e_per_s=1e-4)),
        ),
        (
            "quantum_efficiency",
            toy_report(observatory, spec=spec.fixture(quantum_efficiency=0.8)),
        ),
        (
            "pixel_scale",
            toy_report(observatory, spec=spec.fixture(pixel_scale_arcsec=0.02)),
        ),
        ("n_exozodi", toy_report(observatory, scene=toy_scene(n_exozodi=1.0))),
        (
            "star_flux",
            toy_report(observatory, scene=toy_scene(star_flux_density_jy=40.0)),
        ),
        ("ppfact", toy_report(observatory, ppfact=30.0)),
        ("bg_multiplier", toy_report(observatory, bg_multiplier=2.0)),
        ("eps_warm_T_cold", toy_report(observatory, eps_warm_T_cold=0.1)),
        ("bin_width", toy_report(observatory, bin_width_nm=50.0)),
        ("n_time_nodes", toy_report(observatory, n_time_nodes=4)),
        ("orbit_epoch", toy_report(observatory, orbit_epoch_jd=2461001.5)),
        (
            "occulter_transmission",
            toy_report(observatory, coronagraph=ToyCoronagraph(sky=0.1)),
        ),
        (
            "throughput_curve",
            toy_report(observatory, coronagraph=ToyCoronagraph(peak=0.2)),
        ),
        ("n_channels", _renominal(base, lambda r: r.optical_path.n_channels, 2.0)),
        (
            "npix_multiplier",
            _renominal(base, lambda r: r.optical_path.npix_multiplier, 2.0),
        ),
        (
            "optics_throughput",
            _renominal(
                base, lambda r: r.optical_path.attenuating_elements[0].throughput, 0.4
            ),
        ),
        (
            "obscuration",
            _renominal(base, lambda r: r.optical_path.primary.obscuration, 0.3),
        ),
        (
            "observatory_orbit",
            eqx.tree_at(
                lambda r: r.observatory.orbit._interp_x,
                base,
                eqx.tree_at(lambda i: i.f, halo, halo.f + 0.01),
            ),
        ),
    ]


def test_calibration_id_digests_every_declaration(observatory):
    """Fails if any declaration a predicted budget depends on leaves the id unchanged.

    The id is stable across two builds and an integer below 2^48 that a float64
    payload column holds exactly. Each of 21 single-field mutations (the law's
    threshold, kappa and corr; the detector's dark current, QE and pixel scale; the
    nominal exozodi and stellar flux; the floor, background multiplier, thermal
    emissivity, band width, time nodes and orbit epoch; the occulter transmission
    and throughput curves; `n_channels`, `npix_multiplier`, the optics throughput and
    the obscuration of the optical path; the observatory orbit) gives a distinct id.
    """
    from tests.adapters.test_jaxedith_summary import toy_report

    law = ReportingLaw()
    base_report = toy_report(observatory)
    base = calibration_id(base_report, law)
    assert base == calibration_id(toy_report(observatory), law)
    assert 0 < base < 2**48 and int(float(base)) == base
    ids = {}
    for name, mutated in _mutations(observatory):
        if isinstance(mutated, ReportingLaw):
            ids[name] = calibration_id(base_report, mutated)
        else:
            ids[name] = calibration_id(mutated, law)
    assert base not in ids.values(), [k for k, v in ids.items() if v == base]
    assert len(set(ids.values())) == len(ids)


@pytest.mark.medium  # One predicted budget per mutated report (21 evaluations).
def test_every_digested_report_field_moves_the_predicted_budget(observatory):
    """Fails if a report field the id digests does not change a predicted budget.

    Each report mutation of the calibration test, evaluated on a face-on planet over
    thirty days (long enough that 4 and 8 time nodes differ), changes the predicted
    counts, a variance component or the centroid: the id covers fields that matter.
    """
    from spaceodyssey.adapters.jaxedith import ScienceInterval
    from tests.adapters.test_jaxedith_summary import face_on, toy_report

    planet = face_on(1.0)
    interval = ScienceInterval.from_start(2461000.5, 30 * 86400.0)

    def summary(report):
        b = report.predict(planet, interval)
        return np.array(
            [
                float(x)
                for x in (
                    b.planet,
                    b.background,
                    b.var_background,
                    b.var_detector,
                    b.var_systematic,
                    b.thermal,
                    b.ra_arcsec,
                    b.dec_arcsec,
                )
            ]
        )

    base = summary(toy_report(observatory))
    for name, mutated in _mutations(observatory):
        if isinstance(mutated, ReportingLaw):
            continue
        moved = summary(mutated)
        assert not np.allclose(moved, base, rtol=1e-12, atol=0.0), name


# -- Survey: detection and null sequence over planned visits -----------------------

N_WORLDS = 200
N_VISITS = 8  # at most; the plan keeps the visits it can certify
VISIT_SPACING_D = 120.0
# Laplace widths: a Gaussian stand-in for each prior (its standard deviation), so
# parameters the detections do not constrain stay finite. Order of `FISHER_NAMES`.
FISHER_NAMES = ("a", "e", "cos_i", "W", "w", "tp", "log10_Lambda")


def _planet_from_vector(p, mp_kg):
    a, e, cos_i, w_node, w, tp, log_lambda = p
    return {
        "a": a,
        "e": e,
        "cos_i": cos_i,
        "W": w_node,
        "cos_w": jnp.cos(w),
        "sin_w": jnp.sin(w),
        "tp": tp,
        "Lambda": 10.0**log_lambda,
        "Mp": mp_kg,
    }


def run_survey(n_worlds=N_WORLDS, kappas=(1.0, 2.0)):
    """The detection and null sequence of 47 UMa b worlds over planned visits.

    Visits are planned from the profile's nominal planet only: the first day of
    each 120-day step on which the target is outside keepout for the whole
    occupancy and the nominal planet lies between the inner working angle plus
    0.5 lambda/D and the field edge; the exposure is jaxedith's `exptime_ayo` for
    signal-to-noise 10 at the nominal scene in whole fixed frames, after one day of
    settling. Worlds are
    drawn from photomancy's prior through the profile's world source; the true
    scene is the nominal one. Returns the rows and the measured statistics.
    """
    from hwoutils.constants import AU2m, Mearth2kg, Msun2kg, d2s
    from hwoutils.conversions import photons_per_nm_per_m2_to_jy
    from jaxedith import ETCScene, exptime_ayo, zodi_fn_leinert
    from orbix.equations.orbit import period_to_sma

    from spaceodyssey.adapters.jaxedith import (
        DetectorSpec,
        ScienceInterval,
        SummaryReport,
        SummaryScene,
        nominal_planet,
        planet_free,
    )
    from spaceodyssey.adapters.physical_context import context_from_profile
    from spaceodyssey.adapters.skyscapes import (
        PhysicalWorldSource,
        SceneSpec,
        build_star,
    )

    spec = SceneSpec.from_profile(PROFILE, 0)
    context = context_from_profile(PROFILE, targets=[0])
    observatory = context.observatory
    epoch = PROFILE["time"]["study_epoch_jd"]
    photons = build_star(spec).spec_flux_density(spec.wavelength_nm, epoch)
    scene = SummaryScene(
        star_flux_density_jy=float(
            photons_per_nm_per_m2_to_jy(photons, spec.wavelength_nm)
        ),
        dist_pc=spec.dist_pc,
        star_mass_kg=spec.mass_msun * Msun2kg,
        luminosity_lsun=spec.luminosity_lsun,
        ra_deg=spec.ra_deg,
        dec_deg=spec.dec_deg,
        star_diameter_mas=spec.angular_diameter_mas,
        n_exozodi=3.0,
    )
    report = SummaryReport.from_yip(
        DetectorSpec.fixture(), scene, orbit_epoch_jd=epoch, observatory=observatory
    )
    lod = report.lod_arcsec
    settle_d = float(observatory.settling_time_d)
    mp_nominal = spec.planet_mass_mearth * Mearth2kg
    nominal = nominal_planet(
        a_au=float(period_to_sma(spec.period_d, spec.mass_msun * Msun2kg + mp_nominal)),
        e=spec.ecc,
        inclination_rad=math.radians(spec.inclination_deg),
        W_rad=math.radians(spec.node_deg),
        w_rad=math.radians(spec.omega_planet_deg),
        tp_d=spec.tp_jd - epoch,
        Rp_au=spec.planet_radius_m / AU2m,
        Ag=spec.geometric_albedo,
    )
    nominal["Mp"] = jnp.asarray(mp_nominal)

    # -- Plan from nominal quantities only.
    days = epoch + np.arange(0.0, 4 * 365.25)
    visible = np.asarray(context.observable(0, days))
    occupied = np.convolve(visible, np.ones(3), mode="same") == 3
    rates = report.rates(report.nominal, nominal, days + settle_d)
    sep = np.asarray(rates.separation_lod)
    iwa = float(report.optical_path.coronagraph.IWA)
    usable = occupied & (sep > iwa + 0.5) & (sep < report.field_radius_lod - 1.0)
    starts = []
    for d, ok in zip(days, usable, strict=True):
        if ok and (not starts or d >= starts[-1] + VISIT_SPACING_D):
            starts.append(d)
        if len(starts) == N_VISITS:
            break
    intervals, limits = [], []
    law = ReportingLaw()
    for d in starts:
        start = d + settle_d
        r = report.rates(report.nominal, nominal, start)
        etc = ETCScene(
            F0=report.nominal.zero_point,
            Fs_over_F0=report.nominal.star_ratio,
            Fp_over_Fs=r.contrast[0],
            Fzodi=zodi_fn_leinert(observatory, report._exposure(start), scene),
            Fexozodi=report.nominal.exozodi_1au,
            dist_pc=scene.dist_pc,
            sep_arcsec=float(r.separation_lod[0]) * lod,
            Fbinary=0.0,
            stellar_diam_mas=scene.star_diameter_mas,
        )
        t10 = float(
            exptime_ayo(
                report.optical_path,
                etc,
                report.wavelength_nm,
                float(r.separation_lod[0]),
                report.bin_width_nm,
                10.0,
                temp_K=observatory.temperature_K,
                ez_ppf=jnp.inf,
                ppfact=math.inf,
                bg_multiplier=1.0,
                eps_warm_T_cold=0.0,
                cadence="fixed_frame",
            )
        )
        interval = ScienceInterval.from_start(start, t10)
        intervals.append(interval)
        limits.append(law.limit_counts(report.predict(planet_free(nominal), interval)))
    intervals = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *intervals)
    limits = jnp.stack(limits)
    occupied_s = settle_d * d2s + intervals.duration_s

    # -- Worlds from the prior; the true scene is the nominal one.
    source = PhysicalWorldSource.from_profile(PROFILE, targets=[0])
    caps = type("Caps", (), {"n_subjects": 1})()
    keys = jax.random.split(jax.random.PRNGKey(2026), n_worlds)
    truth = jax.vmap(lambda k: source.draw(k, caps))(keys)
    truth = jax.tree_util.tree_map(lambda x: x[:, 0], truth)
    params = jnp.stack(
        [
            period_to_sma(truth.period_d, scene.star_mass_kg + truth.Mp_kg),
            truth.e,
            truth.cos_i,
            truth.W_rad,
            jnp.arctan2(truth.sin_w, truth.cos_w),
            truth.tp_jd - epoch,
            jnp.log10(truth.Ag * truth.Rp_AU**2),
        ],
        axis=1,
    )
    true_scene = report.prepare(scene)
    calibration = calibration_id(report, law)

    def visits(p, mp, prepared):
        planet = _planet_from_vector(p, mp)
        return jax.vmap(lambda iv: report.budget(prepared, planet, iv))(intervals)

    # `SummaryReport.expected` with the true scene prepared once for all worlds.
    expected = eqx.filter_jit(jax.vmap(lambda p, m: visits(p, m, true_scene)))(
        params, truth.Mp_kg
    )

    def sample_all(law_k):
        # Each law is its own declaration: its rows carry its own calibration id.
        calibration_k = calibration_id(report, law_k)

        def one(w, v):
            budget = jax.tree_util.tree_map(lambda x: x[w, v], expected)
            iv = jax.tree_util.tree_map(lambda x: x[v], intervals)
            key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(7), w), v)
            return sample_report(
                key,
                law_k,
                budget,
                limits[v],
                lod_arcsec=lod,
                time_d=iv.midpoint_jd,
                occupied_s=occupied_s[v],
                calibration_id=calibration_k,
            )

        w, v = jnp.meshgrid(
            jnp.arange(n_worlds), jnp.arange(len(starts)), indexing="ij"
        )
        return eqx.filter_jit(jax.vmap(jax.vmap(one)))(w, v)

    rows = {k: sample_all(ReportingLaw(kappa=k)) for k in kappas}
    base = rows[kappas[0]]
    # The likelihood at the truth: the predicted budget is the expected one here.
    ll = eqx.filter_jit(jax.vmap(jax.vmap(lambda r, b: loglike_report(law, r, b))))(
        base, expected
    )
    detected = np.asarray(base.detected) == 1
    level = np.asarray(expected.planet) / np.asarray(limits)[None, :]
    s_ratio = np.sqrt(np.asarray(expected.variance)) / np.asarray(limits)[None, :]

    # photomancy's fixed-denominator null against the law's, over every row.
    ours_null = np.asarray(
        jax.vmap(jax.vmap(law.log_null_probability))(
            jnp.broadcast_to(limits, level.shape), expected
        )
    )
    z = 5.0 * (1.0 - level / 5.0)
    theirs_null = np.asarray(jax.scipy.special.log_ndtr(z))
    delta = ours_null - theirs_null

    # photomancy's imaging detection term (Gaussian in magnitude) against the law's
    # counts term, both as densities of the counts (the magnitude Jacobian added).
    f = np.asarray(base.counts)[detected]
    snr = np.asarray(base.snr)[detected]
    n_true = np.asarray(expected.planet)[detected]
    s_true = np.sqrt(np.asarray(expected.variance))[detected]
    ours_det = stats.norm.logpdf(f, n_true, s_true)
    dmag_err = 2.5 / math.log(10.0) / snr
    theirs_det = stats.norm.logpdf(
        -2.5 * np.log10(n_true), -2.5 * np.log10(f), dmag_err
    ) + np.log(2.5 / math.log(10.0) / f)

    # Laplace widths at the truth for worlds with at least two detections.
    def jac(p, mp):
        def obs(q):
            b = visits(q, mp, report.nominal)
            return jnp.stack(
                [b.planet, jnp.sqrt(b.variance), b.ra_arcsec, b.dec_arcsec], axis=-1
            )

        return jax.jacrev(obs)(p)

    n_det = detected.sum(axis=1)
    chosen = np.flatnonzero(n_det >= 2)
    jacobians = np.asarray(
        eqx.filter_jit(jax.vmap(jac))(params[chosen], truth.Mp_kg[chosen])
    )
    period = np.asarray(truth.period_d)[chosen]
    prior_sd = np.stack(
        [
            np.asarray(params[chosen, 0]) * (2 / 3) * (12 / math.sqrt(12)) / period,
            np.full(len(chosen), 0.19),
            np.full(len(chosen), 2 / math.sqrt(12)),
            np.full(len(chosen), 2 * math.pi / math.sqrt(12)),
            np.full(len(chosen), 2 * math.pi / math.sqrt(12)),
            period / math.sqrt(12),
            np.full(len(chosen), math.sqrt(4 * 2.5**2 / 12 + 2.0**2 / 12)),
        ],
        axis=1,
    )
    widths = {}
    for k in kappas:
        rk = rows[k]
        out = []
        for j, w in enumerate(chosen):
            fisher = np.diag(1.0 / prior_sd[j] ** 2)
            for v in np.flatnonzero(detected[w]):
                dn, ds, dra, ddec = jacobians[j, v]
                s = float(np.sqrt(expected.variance[w, v]))
                sx = float(rk.ra_err[w, v])
                rho = float(rk.corr[w, v])
                cov = sx**2 * np.array([[1.0, rho], [rho, 1.0]])
                dmu = np.stack([dra, ddec])
                fisher += np.outer(dn, dn) / s**2 + 2 * np.outer(ds, ds) / s**2
                fisher += dmu.T @ np.linalg.solve(cov, dmu)
            out.append(np.sqrt(np.diag(np.linalg.inv(fisher))))
        widths[k] = np.asarray(out)
    ratio = widths[kappas[1]] / widths[kappas[0]]

    # Worlds whose planet never reaches half the limit: nulls carry no information.
    faint = np.flatnonzero(level.max(axis=1) < 0.5)
    info = []
    for w in faint:
        planet = _planet_from_vector(params[w], truth.Mp_kg[w])
        dark = jax.vmap(lambda iv, p=planet: report.predict(planet_free(p), iv))(
            intervals
        )
        ll_dark = jax.vmap(lambda r, b: loglike_report(law, r, b))(
            jax.tree_util.tree_map(lambda x, w=w: x[w], base), dark
        )
        info.append(float(np.sum(ll[w]) - np.sum(ll_dark)))

    bins = [0.0, 1.0, 3.0, 5.0, 7.0, np.inf]
    table = []
    for lo, hi in itertools.pairwise(bins):
        sel = (level >= lo) & (level < hi)
        if sel.any():
            table.append(
                (
                    lo,
                    hi,
                    int(sel.sum()),
                    float(np.median(delta[sel])),
                    float(np.max(np.abs(delta[sel]))),
                )
            )
    return dict(
        report=report,
        starts=np.asarray(starts),
        t_s=np.asarray(intervals.duration_s),
        limits=np.asarray(limits),
        rows=rows,
        ll=np.asarray(ll),
        detected=detected,
        n_det=n_det,
        level=level,
        s_ratio=s_ratio,
        delta=delta,
        delta_table=table,
        det_term_diff=ours_det - theirs_det,
        widths=widths,
        ratio=ratio,
        faint_info=np.asarray(info),
        n_all_null=int(np.sum(n_det == 0)),
        p_detect=1.0 - np.exp(ours_null),
        s_over_shat=s_true / np.sqrt(np.asarray(base.limit_counts)[detected] ** 2 + f),
        calibration=calibration,
        row_ids={k: np.unique(np.asarray(r.calibration_id)) for k, r in rows.items()},
        law_ids={k: calibration_id(report, ReportingLaw(kappa=k)) for k in kappas},
    )


@pytest.mark.slow  # 200 worlds over the planned visits on the real response (~40 s).
def test_survey_detection_and_null_sequence_over_planned_visits():
    """Fails if the survey's detections, nulls or sensitivities break the law.

    Over the planned visits of 47 UMa b, 200 worlds from the prior: detections and
    nulls both occur; the detection count equals the sum of 1 - P(null | truth)
    within 3 standard errors; every row's log likelihood at the truth is finite.
    The photomancy-form null is measured against the law's: below the threshold
    the law penalizes more, well above it far less (photomancy's fixed denominator
    ignores the planet's shot noise). photomancy's magnitude-space detection term
    agrees with the law's counts term to 0.1 in median where the observable noise
    estimate matches the true noise. kappa = 2 widens every Laplace width by a
    factor in [1, 2], the orbital angles and epoch by more than 1.4 in median. A
    planet that never reaches half the limit gives nulls with under 1e-3 of
    information.
    """
    s = run_survey()
    for k, ids in s["row_ids"].items():
        assert ids.tolist() == [float(s["law_ids"][k])]
    assert len(set(s["law_ids"].values())) == len(s["law_ids"])
    detected, p = s["detected"], s["p_detect"]
    assert 0 < detected.sum() < detected.size
    se = math.sqrt(float(np.sum(p * (1 - p))))
    assert abs(detected.sum() - p.sum()) < 3 * se
    assert np.isfinite(s["ll"]).all()
    assert np.sum(s["n_det"] >= 2) >= 20 and s["n_all_null"] >= 20

    medians = {(lo, hi): med for lo, hi, _, med, _ in s["delta_table"]}
    assert medians[(1.0, 3.0)] < 0 and medians[(3.0, 5.0)] < 0
    assert medians[(7.0, np.inf)] > 1.0
    matched = s["s_over_shat"] < 1.1
    assert np.median(np.abs(s["det_term_diff"][matched])) < 0.1

    ratio = s["ratio"]
    assert np.all(ratio >= 1.0 - 1e-9) and np.all(ratio <= 2.0 + 1e-9)
    median = dict(zip(FISHER_NAMES, np.median(ratio, axis=0), strict=True))
    assert median["W"] > 1.4 and median["tp"] > 1.4
    assert len(s["faint_info"]) >= 20
    assert np.max(np.abs(s["faint_info"])) < 1e-3
