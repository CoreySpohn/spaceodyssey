"""The OFTI stand-in refit on physical summary rows.

Anchors (unit): noiseless physical rows of a known orbit hand OFTI exactly the
science-interval midpoints, in days from the time origin, and the true positions;
null rows and rows without a certified science interval never count as astrometry;
an audience blind to a flag column reads no row; the scope is declared in the model
and in the run-document configuration. One real fit (medium). Survey and done
criterion (slow): posterior contraction over the five scripted visits of the fixed
plan, and five refreshes that install particles on both executors.
"""

import copy
import dataclasses
import itertools
import math
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Msun2kg
from photomancy.orbit.forward import predict_relative_astrometry
from photomancy.orbit.ofti import PARAM_NAMES

from spaceodyssey.adapters.jaxedith import SUMMARY_COLUMNS, SummaryRow
from spaceodyssey.adapters.photomancy import OftiRefit, ofti_model
from spaceodyssey.adapters.physical_context import (
    PhysicalClock,
    PlannedVisit,
    context_from_profile,
    require_certified,
)
from spaceodyssey.adapters.physical_profile import load_profile
from spaceodyssey.core import CompiledExecutor, HostExecutor
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import RefitStatus
from spaceodyssey.core.executors.base import index_world
from spaceodyssey.core.records import record_none
from spaceodyssey.core.refresh import RefitEntry
from spaceodyssey.studies import physical as ph
from tests.core.helpers import worlds

PROFILE = load_profile()
ORIGIN_JD = PROFILE["time"]["study_epoch_jd"]
COL = {name: i for i, name in enumerate(SUMMARY_COLUMNS)}
N_ROWS = 8
KEY = np.asarray(jax.random.PRNGKey(0))

# The T4d fixture's orbit and epochs (days from the time origin), on the profile's
# second target: a solar-mass host at 10 pc.
T4D_TARGET = 1
T4D_EPOCHS_D = (0, 12, 30)
T4D_ORBIT = {
    "a": 1.2,
    "e": 0.1,
    "cos_i": 0.6,
    "W": 0.8,
    "cos_w": math.cos(0.5),
    "sin_w": math.sin(0.5),
    "tp": 3.0,
}
EXPOSURE_TICKS = 1_000_000  # 1000 s of 1 ms ticks


def t4d_context():
    """The second target's context on a clock whose tick 0 is two days early."""
    clock = PhysicalClock.from_jd(ORIGIN_JD - 2.0, PROFILE["time"]["tick_length_s"])
    return context_from_profile(PROFILE, clock=clock, targets=[T4D_TARGET])


def t4d_visits(context):
    """Visits whose science-interval midpoints fall on the T4d epochs exactly.

    The midpoint of a visit starting at `s` is ``s + settle + exposure / 2``, in
    integer ticks; the clock's tick 0 is two days before the time origin.
    """
    per_day = context.clock.ticks_per_day
    lead = context.settle_ticks + EXPOSURE_TICKS // 2
    return tuple(
        PlannedVisit(0, (2 + t) * per_day - lead, EXPOSURE_TICKS) for t in T4D_EPOCHS_D
    )


def t4d_rows():
    """Noiseless physical rows of the T4d orbit, one per T4d epoch."""
    context = t4d_context()
    return ph.scripted_rows(
        context, t4d_visits(context), T4D_ORBIT, target=T4D_TARGET
    ).payloads


def t4d_truth():
    """The T4d orbit's (east, north) offsets at the T4d epochs, arcsec."""
    target = PROFILE["targets"][T4D_TARGET]
    return predict_relative_astrometry(
        jnp.asarray(T4D_EPOCHS_D, jnp.float64),
        *(T4D_ORBIT[k] for k in PARAM_NAMES),
        target["mass_msun"] * Msun2kg,
        target["dist_pc"],
    )


def extra_rows(payloads):
    """A null row and a no-science detection row, each with a usable position.

    Both copy a real detection row's position, errors and epoch, so either would
    fit as astrometry if it counted.
    """
    null = payloads[1].copy()
    null[COL["detected"]] = 0.0
    no_science = payloads[2].copy()
    no_science[COL["valid_science"]] = 0.0
    return np.stack([null, no_science])


def entry_of(payloads, key=KEY):
    """A `RefitEntry` whose counting rows are `payloads`, padded to `N_ROWS`."""
    caps = Capacities(payload_width=len(SUMMARY_COLUMNS))
    blank = record_none(caps)
    rows = jax.tree_util.tree_map(
        lambda x: np.broadcast_to(np.asarray(x), (N_ROWS, *np.shape(x))).copy(), blank
    )
    payload = np.zeros((N_ROWS, len(SUMMARY_COLUMNS)))
    payload[: len(payloads)] = payloads
    rows = dataclasses.replace(
        rows, valid=np.arange(N_ROWS) < len(payloads), payload=payload
    )
    return RefitEntry(
        subject=np.asarray(0),
        rows=rows,
        key=np.asarray(key),
        row_keys=np.zeros((N_ROWS, 2), np.uint32),
        n_rows=np.asarray(len(payloads)),
    )


def stub_study(n_particles=ph.N_PARTICLES):
    return types.SimpleNamespace(
        observer=ph.ParticleObserver(n_particles=n_particles), caps=Capacities()
    )


def standin(**kw):
    """The stand-in for the T4d target, with T4d's loose period support."""
    base = {"log_P_range": (2.0, 5.0), "batch": 4096}
    return ph.StandInOftiRefit.from_profile(PROFILE, T4D_TARGET, **{**base, **kw})


def capture_ofti(monkeypatch):
    """Replace OFTI with a recorder that returns ones; returns the call list."""
    calls = []

    def fake(data, **kw):
        calls.append((data, kw))
        return types.SimpleNamespace(samples=np.ones((kw["n_accept"], 7)))

    monkeypatch.setattr("spaceodyssey.adapters.photomancy.ofti", fake)
    return calls


def test_noiseless_physical_rows_hand_ofti_the_midpoints_and_the_true_orbit(
    monkeypatch,
):
    """The table OFTI receives is the T4d table: epochs, true positions, errors.

    Each epoch is the science-interval midpoint in days from the time origin, built
    by hand in integer ticks (0, 12 and 30 days exactly); each position is the T4d
    orbit's at that epoch. Fails if the record epoch is not the science midpoint
    (a start or an occupancy midpoint), if the time origin is not subtracted, or if
    a position is not the true one.
    """
    calls = capture_ofti(monkeypatch)
    rows = t4d_rows()
    slices, status, diag = standin()._fit(stub_study(), entry_of(rows))
    assert status == RefitStatus.OK and diag == [3.0, float(ph.N_PARTICLES)]
    assert int(slices[1]) == 3
    ((data, kw),) = calls
    np.testing.assert_array_equal(np.asarray(data.times)[:3], T4D_EPOCHS_D)
    ra, dec = t4d_truth()
    np.testing.assert_allclose(np.asarray(data.ra)[:3], ra, rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(data.dec)[:3], dec, rtol=0, atol=1e-15)
    assert np.asarray(data.is_valid).tolist() == [True] * 3 + [False] * 5
    # the law's error at the observed signal-to-noise: kappa (lambda/D) / snr
    lod_arcsec = 1e-6 / PROFILE["response"]["diameter_m"] * 180 / math.pi * 3600
    snr = rows[:, COL["snr"]]
    np.testing.assert_allclose(snr, ph.SCRIPTED_SNR, rtol=1e-12)
    expected = ph.STANDIN_KAPPA * lod_arcsec / snr
    np.testing.assert_allclose(np.asarray(data.ra_err)[:3], expected, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(data.dec_err)[:3], expected, rtol=1e-12)
    target = PROFILE["targets"][T4D_TARGET]
    assert kw["Ms"] == target["mass_msun"] * Msun2kg
    assert kw["dist_pc"] == target["dist_pc"]


def test_null_and_no_science_rows_never_count_as_astrometry(monkeypatch):
    """Only detections with a certified science interval reach OFTI.

    A null row and a detection row with `valid_science = 0`, each carrying a usable
    position, are left out of the fit and the epoch count; an entry holding only
    such rows is `NO_SUPPORT`, never reaches OFTI and installs nothing. Fails if a
    null or a no-science row counts as an astrometric row.
    """
    calls = capture_ofti(monkeypatch)
    rows = t4d_rows()
    extra = extra_rows(rows)
    slices, status, diag = standin()._fit(
        stub_study(), entry_of(np.concatenate([rows, extra]))
    )
    assert status == RefitStatus.OK and int(slices[1]) == 3 and diag[0] == 3.0
    ((data, _),) = calls
    assert np.asarray(data.is_valid).tolist() == [True] * 3 + [False] * 5
    np.testing.assert_array_equal(
        ph.astrometric_rows(np.ones(5, bool), np.concatenate([rows, extra])),
        [True, True, True, False, False],
    )
    calls.clear()
    slices, status, diag = standin()._fit(stub_study(), entry_of(extra))
    assert (slices, status, diag) == (None, RefitStatus.NO_SUPPORT, [0.0, 0.0])
    assert calls == []


def blind_caps(column):
    """Capacities whose summary kind hides `column` from the policy."""
    profile = copy.deepcopy(PROFILE)
    for entry in profile["payload_columns"]:
        if entry["name"] == column:
            entry["audience"] = ["observer", "scorer"]
    caps = Capacities(rows_per_subject=N_ROWS, refit_lanes=2)
    return ph.physical_registry(profile).lower(caps, ph.ParticleObserver())


@pytest.mark.parametrize("column", ["detected", "valid_science", "dec_err"])
def test_an_audience_blind_to_a_column_the_standin_reads_gets_no_rows(column):
    """A policy that cannot read a flag or a position column is refit from nothing.

    A masked column reads as zero: a masked `detected` would silently drop every
    row and a masked error would fail the fit, so the audience is marked blind and
    its entry is `NO_SUPPORT`. Fails if the flag columns are left out of the check.
    """
    study = types.SimpleNamespace(caps=blind_caps(column))
    assert standin()._readable(study).tolist() == [True, False]
    study = types.SimpleNamespace(caps=blind_caps("snr"))
    assert standin()._readable(study).tolist() == [True, True]


def test_the_scope_is_declared_in_the_model_hash_and_the_document_configuration():
    """The declared target names the approximation, its domain and the time origin.

    Fails if the model hash equals plain OFTI's (the scope would not reach a run
    document), if the configuration drops the time origin or does not rebuild the
    same model, or if the stand-in accepts other payload columns.
    """
    refit = standin()
    declaration = refit.model.declaration()
    assert declaration["support"]["scope"] == ph.STANDIN_SCOPE
    assert declaration["likelihood"]["time_origin_jd"] == ORIGIN_JD
    assert declaration["name"] == "ofti_standin_conditional_on_detection"
    for word in ("detection", "induced prior", "midpoint", "ignored", "loose"):
        assert word in " ".join(ph.STANDIN_SCOPE.values())
    target = PROFILE["targets"][T4D_TARGET]
    plain = ofti_model(
        stellar_mass_kg=target["mass_msun"] * Msun2kg,
        dist_pc=target["dist_pc"],
        log_P_range=(2.0, 5.0),
    )
    assert refit.model_hash == refit.model.digest() != plain.digest()
    config = refit.configuration()
    assert "columns" not in config and config["time_origin_jd"] == ORIGIN_JD
    binding = ph.standin_binding()
    rebuilt = binding.build(binding.describe(refit, None), None)
    assert rebuilt.model_hash == refit.model_hash
    assert rebuilt.binding()["kind"] == "spaceodyssey.ofti_standin"
    moved = ph.StandInOftiRefit(**{**config, "time_origin_jd": ORIGIN_JD + 1.0})
    assert moved.model_hash != refit.model_hash
    with pytest.raises(TypeError, match="summary columns"):
        ph.StandInOftiRefit(**config, columns=(0, 1, 2, 3, 4, 5))


@pytest.mark.medium
def test_one_real_fit_reproduces_the_orbit_and_equals_ofti_on_the_t4d_table():
    """A real OFTI fit of noiseless physical rows, with a null and a no-science row.

    Medium: two OFTI scorers compile (about 4 s cold). The particles are
    bit-identical to plain `OftiRefit` on the T4d-style table of the three
    detections (days, offsets, errors) under the same key, and every particle
    predicts the rows within 5 sigma at every epoch. Fails if the extra rows or the
    Julian Dates reach the sampler, or the particles do not describe the orbit.
    """
    rows = t4d_rows()
    entry = entry_of(np.concatenate([rows, extra_rows(rows)]))
    slices, status, _ = standin()._fit(stub_study(), entry)
    assert status == RefitStatus.OK and int(slices[1]) == 3
    samples = slices[0]
    assert samples.shape == (ph.N_PARTICLES, 7) and np.isfinite(samples).all()
    assert len({tuple(r) for r in samples}) > 1

    target = PROFILE["targets"][T4D_TARGET]
    plain = OftiRefit(
        stellar_mass_kg=target["mass_msun"] * Msun2kg,
        dist_pc=target["dist_pc"],
        log_P_range=(2.0, 5.0),
        batch=4096,
        count="n",
    )
    names = ("time_d", "ra", "dec", "ra_err", "dec_err", "corr")
    table = rows[:, [COL[n] for n in names]]
    table[:, 0] -= ORIGIN_JD
    plain_entry = entry_of(np.zeros((3, len(SUMMARY_COLUMNS))))
    payload = np.zeros((N_ROWS, len(SUMMARY_COLUMNS)))
    payload[:3, :6] = table
    plain_entry = dataclasses.replace(
        plain_entry, rows=dataclasses.replace(plain_entry.rows, payload=payload)
    )
    reference, status, _ = plain._fit(stub_study(), plain_entry)
    assert status == RefitStatus.OK
    assert samples.tobytes() == reference[0].tobytes()

    times = jnp.asarray(T4D_EPOCHS_D, jnp.float64)
    ra, dec = jax.vmap(
        lambda p: predict_relative_astrometry(
            times, *p, target["mass_msun"] * Msun2kg, target["dist_pc"]
        )
    )(jnp.asarray(samples))
    sigma = rows[:, COL["ra_err"]]
    assert np.all(np.abs(np.asarray(ra) - rows[:, COL["ra"]]) < 5 * sigma)
    assert np.all(np.abs(np.asarray(dec) - rows[:, COL["dec"]]) < 5 * sigma)


# -- Survey and done criterion (slow) ----------------------------------------------

SURVEY_KEY = jax.random.PRNGKey(2026)
SURVEY_PARTICLES = 64


def plan_rows():
    """The certified fixed plan of 47 UMa b and its scripted rows under the law."""
    context = context_from_profile(PROFILE, targets=[0])
    plan = ph.fixed_plan(context)
    require_certified(context, plan)
    rows = ph.scripted_rows(context, plan, ph.nominal_orbit(PROFILE), key=SURVEY_KEY)
    return context, rows


@pytest.mark.slow  # five OFTI fits of up to five epochs, 64 particles (about 100 s)
def test_survey_posterior_contraction_over_five_scripted_visits():
    """The predictive sky position contracts as the five visits arrive.

    Rows: the certified fixed plan, drawn under the law at signal-to-noise 10 with
    the stand-in's kappa. After each visit the stand-in refits every row so far
    (64 particles, one key); the spread is the root mean square over one period
    after the last visit of the particles' predicted (east, north) scatter. Fails
    if every visit is not a detection, if the spread after five visits is not below
    0.4 of the spread after one, if a visit widens it by more than 15 percent, or if
    the true track leaves 4 predictive standard deviations.
    """
    _, rows = plan_rows()
    payloads = rows.payloads
    assert payloads[:, COL["detected"]].tolist() == [1.0] * 5
    target = PROFILE["targets"][0]
    ms, dist = target["mass_msun"] * Msun2kg, target["dist_pc"]
    orbit = ph.nominal_orbit(PROFILE)
    last = payloads[-1, COL["time_d"]] - ORIGIN_JD
    grid = jnp.asarray(last + np.linspace(0.0, target["planet"]["period_d"], 33)[:-1])
    truth = np.stack(
        predict_relative_astrometry(grid, *(orbit[k] for k in PARAM_NAMES), ms, dist)
    )
    refit = ph.StandInOftiRefit.from_profile(PROFILE, 0)
    spreads, widths = [], []
    for k in range(1, 6):
        slices, status, _ = refit._fit(
            stub_study(SURVEY_PARTICLES), entry_of(payloads[:k], SURVEY_KEY)
        )
        assert status == RefitStatus.OK and int(slices[1]) == k
        samples = slices[0]
        xy = np.asarray(
            jax.vmap(
                lambda p: jnp.stack(predict_relative_astrometry(grid, *p, ms, dist))
            )(jnp.asarray(samples))
        )
        spreads.append(float(np.sqrt(np.mean(xy.var(axis=0).sum(axis=0)))))
        widths.append((float(samples[:, 1].std()), float(samples[:, 2].std())))
        z = np.abs(xy.mean(axis=0) - truth) / np.sqrt(xy.var(axis=0))
        assert z.max() < 4.0, (k, z.max())
    print("spread mas", [round(1e3 * s, 1) for s in spreads])
    print("std e, cos i", [tuple(round(w, 3) for w in pair) for pair in widths])
    assert spreads[-1] < 0.4 * spreads[0]
    assert all(b < 1.15 * a for a, b in itertools.pairwise(spreads))


@pytest.mark.slow  # ten OFTI fits and two cold study compiles (about 90 s)
def test_five_refreshes_install_particles_on_both_executors():
    """Each visit's refresh installs particles, and the executors agree.

    The fixed plan runs as scripted summary records on a one-day checkpoint grid.
    Fails if a refresh does not install (five `OK` product rows per audience with
    one to five rows), if a record's epoch is not its science-interval midpoint
    within half a tick, or if the compiled executor installs other particles than
    the host executor.
    """
    context, rows = plan_rows()
    study = ph.physical_study(rows, context)
    refit = ph.StandInOftiRefit.from_profile(PROFILE, 0)
    truths, roots = worlds(1, study.caps)
    host = HostExecutor().run(study, index_world(truths, 0), roots[0], 25, refit=refit)
    log = host.carry.log
    valid = np.asarray(log.valid)
    kind = np.asarray(log.kind)
    products = np.asarray(log.payload)[valid & (kind == 3)][:, :2]
    assert products[:, 0].tolist() == [RefitStatus.OK] * 10
    assert products[:, 1].tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert int(host.carry.belief.n[0]) == 5
    summary = valid & (kind == ph.SUMMARY_KIND_INDEX)
    epochs = np.asarray(log.payload)[summary][:, COL["time_d"]]
    starts, ends = (
        np.asarray(log.start_tick)[summary],
        np.asarray(log.end_tick)[summary],
    )
    midpoints = np.asarray(context.science_midpoint_jd(starts, ends))
    half_tick_d = 0.5 / context.clock.ticks_per_day
    assert np.all(np.abs(epochs - midpoints) <= half_tick_d)

    compiled = CompiledExecutor().run(study, truths, roots, 25, refit=refit)
    assert int(np.asarray(compiled.carry.belief.n)[0, 0]) == 5
    assert np.asarray(compiled.carry.belief.samples)[0].tobytes() == (
        np.asarray(host.carry.belief.samples).tobytes()
    )
    row = SummaryRow.from_payload(np.asarray(log.payload)[summary][0, :15])
    assert float(row.valid_science) == 1.0 and float(row.detected) == 1.0
