"""The summary channel: executed occupancy to a summary record, on both executors.

Every expected value comes from outside the channel: the offline reference builds its
own report from the profile and calls `sample_report` at the executed science
interval, tick arithmetic is plain integers, and the keepout window is located with
the observing context's own visibility test. The scenarios share one study structure
(the policy's start tick, cancellation and channel are array leaves), so they share
one compiled program per executor.
"""

import dataclasses
import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import AU2m, Mearth2kg, Msun2kg
from hwoutils.conversions import photons_per_nm_per_m2_to_jy
from orbix.equations.orbit import period_to_sma

from spaceodyssey.adapters.jaxedith import (
    SUMMARY_COLUMNS,
    DetectorSpec,
    JaxedithSummaryChannel,
    ReportingLaw,
    ScienceInterval,
    SummaryReport,
    SummaryScene,
    SummaryTruth,
    calibration_id,
    nominal_planet,
    planet_free,
    sample_report,
)
from spaceodyssey.adapters.physical_context import (
    PlannedVisit,
    UncertifiedPlan,
    context_from_profile,
)
from spaceodyssey.adapters.physical_profile import load_profile
from spaceodyssey.adapters.skyscapes import (
    PhysicalTruth,
    PhysicalWorldSource,
    PriorSpec,
    SceneSpec,
    build_star,
)
from spaceodyssey.codec import Bindings, dump_run, load_run
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import (
    F64,
    I64,
    PUBLIC,
    CoincidentOrder,
    Disposition,
    EffectKind,
    Purpose,
    RecordStatus,
    Scope,
)
from spaceodyssey.core.executors.base import histories_equal
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.core.keys import asset_coordinate, draw_key
from spaceodyssey.core.kinds import AbstractPolicy
from spaceodyssey.core.ledger import ChannelTable
from spaceodyssey.core.study import Clock
from spaceodyssey.studies import toy
from spaceodyssey.studies.physical import (
    FIXED_EXPOSURE_S,
    FIXED_PLAN_DAYS,
    SUMMARY_KIND_INDEX,
    physical_registry,
)
from tests.codec.test_document import TOY_KINDS
from tests.core.battery_helpers import _table

PROFILE = load_profile()
COLUMN = {name: i for i, name in enumerate(SUMMARY_COLUMNS)}
N_STEPS = 30
SLACK_DAYS = 3
HOURS_PER_TICK = 0.001 / 3600.0


class VisitPolicy(AbstractPolicy):
    """Wait to `start`, observe `channel`, and cancel `cancel_after` ticks later.

    A nonpositive `cancel_after` lets the occupancy finish; the campaign then drains.
    """

    start: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, I64))
    cancel_after: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, I64))
    channel: jax.Array = eqx.field(default=0, converter=lambda x: jnp.asarray(x, I64))

    def declare(self):
        """The campaign drain this policy ends with."""
        return Declaration(stops=((Scope.CAMPAIGN, Disposition.DRAIN),))

    def decide(self, pstate, view, study, key):
        """Decision 0 waits, 1 observes (and wakes to cancel), 2 cancels or stops."""
        n = view.n_dec
        cancel = self.cancel_after > 0
        kind0 = jnp.select(
            [n == 0, n == 1, (n == 2) & cancel],
            [EffectKind.WAIT, EffectKind.OBSERVE, EffectKind.CANCEL],
            EffectKind.STOP,
        )
        kind1 = jnp.where((n == 1) & cancel, EffectKind.WAIT, EffectKind.NONE)
        eff = _table(
            study.caps,
            kind0,
            kind1,
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
            channel_index=self.channel,
            subject=0,
        )
        tick = eff.tick.at[0].set(jnp.where(n == 0, self.start, 0))
        tick = tick.at[1].set(view.tick + self.cancel_after)
        return dataclasses.replace(eff, tick=tick), pstate, jnp.asarray(0.0, F64)


KINDS = {
    **TOY_KINDS,
    ("policy", "test.visit", 1): VisitPolicy,
    ("channel_model", "jaxedith.summary_channel", 2): JaxedithSummaryChannel,
    ("world_source", "physical.prior", 1): PhysicalWorldSource,
    ("scene", "physical.scene_spec", 1): SceneSpec,
    ("scene", "physical.prior_spec", 1): PriorSpec,
}


def _days(context, days):
    return round(days * context.clock.ticks_per_day)


def keepout_edge_day(context, first_day=200, last_day=900):
    """The last whole day (from the study epoch) the target is observable before the
    first keepout day after `first_day`, found with the context's visibility test."""
    jd = PROFILE["time"]["study_epoch_jd"] + np.arange(first_day, last_day, dtype=float)
    seen = np.asarray(context.observable(0, jd))
    drop = np.flatnonzero(seen[:-1] & ~seen[1:])
    assert drop.size, "the calendar has no keepout edge in the search range"
    return first_day + int(drop[0])


def plan_for(context):
    """Visit 0: the fixed plan's first visit; visit 1 ends four days before the edge."""
    edge = keepout_edge_day(context)
    exposure = round(FIXED_EXPOSURE_S / float(context.clock.clock.tick_length_s))
    return (
        PlannedVisit(
            0, _days(context, FIXED_PLAN_DAYS[0]), exposure, _days(context, 3)
        ),
        PlannedVisit(0, _days(context, edge - 4), exposure, _days(context, 1)),
    )


def world():
    """A bright 47 UMa b: the profile's orbit at 1.5 times the declared radius."""
    spec = SceneSpec.from_profile(PROFILE, 0)

    def one(x):
        return jnp.asarray([x], F64)

    w = math.radians(spec.omega_planet_deg)
    truth = PhysicalTruth(
        period_d=one(spec.period_d),
        e=one(spec.ecc),
        cos_i=one(math.cos(math.radians(spec.inclination_deg))),
        W_rad=one(math.radians(spec.node_deg)),
        cos_w=one(math.cos(w)),
        sin_w=one(math.sin(w)),
        tp_jd=one(spec.tp_jd),
        Mp_kg=one(spec.planet_mass_mearth * Mearth2kg),
        Rp_AU=one(1.5 * spec.planet_radius_m / AU2m),
        Ag=one(spec.geometric_albedo),
    )
    return truth, jax.random.PRNGKey(0)


def build_study(channel, context, plan):
    """One study, one channel per planned visit, run by `VisitPolicy`."""
    n = len(plan)
    caps = physical_registry().lower(Capacities(n_subjects=1, n_an=1, n_channels=n))
    occupancy = context.settle_ticks + plan[0].exposure_ticks
    hours = occupancy * HOURS_PER_TICK
    price = jnp.asarray([[hours, 0.0, 0.0]] * n)
    channels = ChannelTable(
        id=jnp.arange(10, 10 + n, dtype=I64),
        lane=jnp.zeros(n, I64),
        occupancy_ticks=jnp.full(n, occupancy, I64),
        latency_ticks=jnp.zeros(n, I64),
        kind=jnp.full(n, SUMMARY_KIND_INDEX, I64),
        audience=jnp.full(n, PUBLIC, I64),
        cost=price,
        charge=price,
        coincident_order=jnp.full(n, CoincidentOrder.FAULT_FIRST, I64),
    )
    budget = jnp.full((caps.n_balance_rows, caps.n_resources), 1000.0)
    horizon = _days(context, 900)
    study = toy.survey_1(
        caps,
        channels=channels,
        channel_model=channel,
        budget=budget,
        horizon_tick=horizon,
        phase_until=(horizon,),
        policies=(VisitPolicy(plan[0].start_tick, -1),),
    )
    source = PhysicalWorldSource.from_profile(PROFILE, targets=[0])
    return dataclasses.replace(study, clock=Clock(0.001), world_source=source)


def with_policy(study, start, cancel_after=-1, channel=0):
    """The same study structure with another visit script."""
    return eqx.tree_at(
        lambda s: s.agenda.policies, study, (VisitPolicy(start, cancel_after, channel),)
    )


def summary_rows(log):
    """Log slot, status, start and end tick and payload of every written row."""
    keep = np.flatnonzero(np.asarray(log.valid))
    return [
        types.SimpleNamespace(
            slot=int(i),
            status=int(log.status[i]),
            start=int(log.start_tick[i]),
            end=int(log.end_tick[i]),
            payload=np.asarray(log.payload[i])[: len(SUMMARY_COLUMNS)],
        )
        for i in keep
    ]


def measurement_key(study, root, log, slot):
    """The key the core drew the record's measurement with (completed coordinate)."""
    rec = types.SimpleNamespace(
        subject=log.subject[slot],
        channel=log.channel[slot],
        acquisition_index=log.acquisition_index[slot],
        phase=log.phase[slot],
        revision=log.revision[slot],
        status=jnp.asarray(RecordStatus.COMPLETED, I64),
    )
    return draw_key(
        root,
        rec,
        Purpose.MEASUREMENT,
        asset=asset_coordinate(study, log.channel_index[slot]),
        branch=study.fork.branch_of(rec.acquisition_index),
        coupling=study.coupling,
    )


class Offline:
    """PT4's offline path: its own report, `sample_report` at a tick interval."""

    def __init__(self):
        spec = SceneSpec.from_profile(PROFILE, 0)
        self.context = context_from_profile(PROFILE)
        epoch = PROFILE["time"]["study_epoch_jd"]
        photons = build_star(spec).spec_flux_density(spec.wavelength_nm, epoch)
        self.scene = SummaryScene(
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
        self.report = SummaryReport.from_yip(
            DetectorSpec.fixture(),
            self.scene,
            orbit_epoch_jd=epoch,
            observatory=self.context.observatory,
        )
        mp = spec.planet_mass_mearth * Mearth2kg
        self.nominal = nominal_planet(
            a_au=float(period_to_sma(spec.period_d, self.scene.star_mass_kg + mp)),
            e=spec.ecc,
            inclination_rad=math.radians(spec.inclination_deg),
            W_rad=math.radians(spec.node_deg),
            w_rad=math.radians(spec.omega_planet_deg),
            tp_d=spec.tp_jd - epoch,
            Rp_au=spec.planet_radius_m / AU2m,
            Ag=spec.geometric_albedo,
        )
        self.nominal["Mp"] = jnp.asarray(mp)
        self.law = ReportingLaw()
        self.epoch = epoch
        self.calibration = calibration_id(self.report, self.law)

    def row(self, truth, start, end, key):
        """`sample_report` at the science interval of the occupancy `[start, end]`."""
        return _offline_row(
            self.report,
            self.scene,
            self.law,
            self.nominal,
            self.context,
            self.epoch,
            self.calibration,
            truth,
            start,
            end,
            key,
        )


@eqx.filter_jit
def _offline_row(
    report, scene, law, nominal, context, epoch, calibration, truth, start, end, key
):
    """The offline row, compiled like the channel's (eager and fused arithmetic
    differ in the last bit, so a bit-for-bit anchor compares like with like)."""
    clock = context.clock
    s0, s1, _ = context.science_interval(start, end)
    interval = ScienceInterval(
        clock.jd(s0, (s1 - s0).astype(F64) / 2.0), clock.seconds(s1 - s0)
    )
    mp = truth.Mp_kg[0]
    planet = {
        "a": period_to_sma(truth.period_d[0], scene.star_mass_kg + mp),
        "e": truth.e[0],
        "cos_i": truth.cos_i[0],
        "W": truth.W_rad[0],
        "cos_w": truth.cos_w[0],
        "sin_w": truth.sin_w[0],
        "tp": truth.tp_jd[0] - epoch,
        "Lambda": truth.Ag[0] * truth.Rp_AU[0] ** 2,
        "Mp": mp,
    }
    expected = report.expected(SummaryTruth(scene, planet), interval)
    limit = law.limit_counts(report.predict(planet_free(nominal), interval))
    return sample_report(
        key,
        law,
        expected,
        limit,
        lod_arcsec=report.lod_arcsec,
        time_d=interval.midpoint_jd,
        occupied_s=clock.seconds(end - start),
        calibration_id=calibration,
    ).payload()


def check_time_boundary(context, windows, rows):
    """KP9: epoch at the science midpoint; valid science inside the keepout window.

    Independent arithmetic: the midpoint tick in integers, converted with the study
    epoch and the day length; the keepout test on a fine grid over the interval.
    """
    per_day = context.clock.ticks_per_day
    epoch = PROFILE["time"]["study_epoch_jd"]
    for row in rows:
        s0 = row.start + context.settle_ticks
        s1 = max(row.end, s0)
        midpoint = epoch + (s0 + (s1 - s0) / 2) / per_day
        epoch_row = row.payload[COLUMN["time_d"]]
        assert abs(epoch_row - midpoint) * per_day <= 0.5, "epoch is not the midpoint"
        if row.payload[COLUMN["valid_science"]] == 1:
            lo, hi = windows[0]
            assert lo <= row.start and row.end <= hi, "science outside the window"
            grid = np.linspace(epoch + s0 / per_day, epoch + s1 / per_day, 200)
            assert np.all(np.asarray(context.observable(0, grid))), "science in keepout"


@pytest.fixture(scope="module")
def offline():
    """The independent report, built once (about 4 s cold)."""
    return Offline()


@pytest.fixture(scope="module")
def setup():
    """The two-visit plan, its channel and study, and the bright world."""
    context = context_from_profile(PROFILE)
    plan = plan_for(context)
    channel = JaxedithSummaryChannel.from_plan(plan)
    truth, root = world()
    return types.SimpleNamespace(
        context=context,
        plan=plan,
        channel=channel,
        study=build_study(channel, context, plan),
        truth=truth,
        root=root,
    )


def run_host(setup, start, cancel_after=-1, channel=0):
    """One host run of the visit script; the study and the carry."""
    study = with_policy(setup.study, start, cancel_after, channel)
    run = HostExecutor().run(study, setup.truth, setup.root, N_STEPS)
    return study, run


def executed(setup, study, run, offline):
    """The one written row, its offline reference and its measurement key."""
    (row,) = summary_rows(run.carry.log)
    key = measurement_key(study, setup.root, run.carry.log, row.slot)
    return row, offline.row(setup.truth, row.start, row.end, key)


def test_declarations_are_refused_before_anything_is_built():
    """Fails if a malformed channel declaration builds a channel: a window that ends
    before it starts, a zero calibration, an unknown profile, a bad detector, a law
    outside its range or no visits."""
    good = dict(
        profile="first-summary-v1",
        target=0,
        detector=dataclasses.astuple(DetectorSpec.fixture()),
        n_time_nodes=8,
        n_exozodi=3.0,
        law=(5.0, 1.0, 0.0),
        windows=((0, 10, 0),),
        calibration=1,
    )
    JaxedithSummaryChannel(**good)
    bad = {
        "windows": ((10, 0, 0),),
        "calibration": 0,
        "profile": "no-such-profile",
        "detector": (*good["detector"][:-1], "no-such-cadence"),
        "law": (5.0, 1.0, 1.0),
        "target": -1,
        "n_time_nodes": 0,
    }
    for name, value in bad.items():
        with pytest.raises((ValueError, KeyError)):
            JaxedithSummaryChannel(**{**good, name: value})
    with pytest.raises(ValueError, match="at least one planned visit"):
        JaxedithSummaryChannel.from_plan(())
    two = (PlannedVisit(0, 0, 1000), PlannedVisit(1, 0, 1000))
    with pytest.raises(ValueError, match="same target"):
        JaxedithSummaryChannel.from_plan(two)


@pytest.mark.medium(reason="one study through the host executor, about 20 s cold")
def test_the_executed_record_is_the_offline_report(setup, offline):
    """Fails if the channel's row differs by a bit from `sample_report` at the
    executed science interval and key, if the calibration id is not the independent
    report's, or if the epoch leaves the science midpoint."""
    study = setup.study
    host = HostExecutor().run(study, setup.truth, setup.root, N_STEPS)
    row, reference = executed(setup, study, host, offline)
    assert row.status == RecordStatus.COMPLETED
    np.testing.assert_array_equal(row.payload, np.asarray(reference))
    payload = row.payload
    assert payload[COLUMN["detected"]] == 1 and payload[COLUMN["counts"]] > 0
    assert payload[COLUMN["calibration_id"]] == offline.calibration
    assert payload[COLUMN["calibration_id"]] == setup.channel.calibration
    occupied = setup.context.settle_ticks + setup.plan[0].exposure_ticks
    assert payload[COLUMN["occupied_s"]] == occupied / 1000
    assert payload[COLUMN["science_s"]] == FIXED_EXPOSURE_S
    windows = [w[:2] for w in setup.channel.windows]
    check_time_boundary(setup.context, windows, [row])
    late = types.SimpleNamespace(**{**vars(row), "end": row.end + 10_000})
    with pytest.raises(AssertionError, match="midpoint"):
        check_time_boundary(setup.context, windows, [late])


@pytest.mark.medium(reason="five host runs on one compiled structure, cold")
def test_delay_edge_and_cancellation_set_the_science_interval(setup, offline):
    """Fails if a delayed admission does not move the epoch by the delay, if an
    admission shifted across the keepout edge still reports science, if it is charged
    twice, if a cancellation inside settling yields science, or if a cancellation
    during the exposure is not scored over the time actually observed."""
    context, study = setup.context, setup.study
    per_day = context.clock.ticks_per_day
    plan0, plan1 = setup.plan
    windows = [w[:2] for w in setup.channel.windows]

    # Delaying admission by two days moves the record's epoch by two days.
    study0, run0 = run_host(setup, plan0.start_tick)
    study2, run2 = run_host(setup, plan0.start_tick + 2 * per_day)
    (row0,), (row2,) = summary_rows(run0.carry.log), summary_rows(run2.carry.log)
    assert row2.start - row0.start == 2 * per_day
    shift = row2.payload[COLUMN["time_d"]] - row0.payload[COLUMN["time_d"]]
    assert abs(shift * per_day - 2 * per_day) < 0.5
    for study_i, run_i in ((study0, run0), (study2, run2)):
        row, reference = executed(setup, study_i, run_i, offline)
        np.testing.assert_array_equal(row.payload, np.asarray(reference))
        assert row.payload[COLUMN["valid_science"]] == 1

    # The edge visit: inside its slack the science is valid; shifted across the
    # keepout edge it is a no-science row, charged once, with the occupancy kept.
    inside = run_host(setup, plan1.start_tick, channel=1)
    across = run_host(setup, plan1.start_tick + _days(context, 4.5), channel=1)
    (kept,), (lost,) = (summary_rows(r.carry.log) for _, r in (inside, across))
    assert kept.payload[COLUMN["valid_science"]] == 1
    p = lost.payload
    assert p[COLUMN["valid_science"]] == 0
    for name in ("science_s", "counts", "limit_counts", "snr", "ra", "dec", "detected"):
        assert p[COLUMN[name]] == 0, name
    assert p[COLUMN["occupied_s"]] == kept.payload[COLUMN["occupied_s"]] > 0
    assert p[COLUMN["calibration_id"]] == setup.channel.calibration
    debited = np.asarray(across[1].carry.ledger.debited)
    hours = (context.settle_ticks + plan0.exposure_ticks) * HOURS_PER_TICK
    assert debited[0, 0] == pytest.approx(hours, rel=1e-12)
    assert np.count_nonzero(debited) == 1
    check_time_boundary(context, windows, [row0, row2])
    check_time_boundary(context, [setup.channel.windows[1][:2]] * 2, [kept, lost])

    # Cancelled half a day in (settling): no science. Cancelled 400 s into the
    # exposure: the science is the 400 s observed, scored like any other interval.
    settle_cancel = run_host(setup, plan0.start_tick, _days(context, 0.5))
    (early,) = summary_rows(settle_cancel[1].carry.log)
    assert early.status != RecordStatus.COMPLETED
    assert early.payload[COLUMN["valid_science"]] == 0
    assert early.payload[COLUMN["science_s"]] == 0
    assert early.payload[COLUMN["counts"]] == 0
    cut = context.settle_ticks + 400_000
    mid = run_host(setup, plan0.start_tick, cut)
    row, reference = executed(setup, *mid, offline)
    assert row.status != RecordStatus.COMPLETED
    observed = (row.end - row.start - context.settle_ticks) / 1000
    assert 399.0 < observed <= 400.0
    assert row.payload[COLUMN["science_s"]] == observed
    assert row.end == plan0.start_tick + cut
    np.testing.assert_array_equal(row.payload, np.asarray(reference))

    # A plan with a visit inside keepout is refused; a build under another
    # calibration is refused instead of writing rows under the wrong id.
    edge = keepout_edge_day(context)
    exposure = plan0.exposure_ticks
    dark = PlannedVisit(0, _days(context, edge + 1), exposure)
    with pytest.raises(UncertifiedPlan):
        JaxedithSummaryChannel.from_plan((dark,))
    stale = dataclasses.replace(
        setup.channel, calibration=setup.channel.calibration + 1
    )
    with pytest.raises(ValueError, match="calibration"):
        stale.assets()
    assert study.channel_model is setup.channel


@pytest.mark.slow(
    reason="a compiled run (its own program) and a child process that loads the "
    "run document, rebuilds the channel and runs it"
)
def test_both_executors_and_a_new_process_write_the_same_row(setup, offline, tmp_path):
    """Fails if the executors write different histories or rows, if the document drops
    a channel declaration, if a reloaded channel builds another calibration, or if a
    new process writes a different row from the same study, truth and root."""
    study, run = run_host(setup, setup.plan[0].start_tick)
    batched = jax.tree_util.tree_map(lambda x: x[None], setup.truth)
    compiled = CompiledExecutor().run(study, batched, setup.root[None], N_STEPS)
    assert histories_equal(run.history(), compiled.history(0)).ok
    first_world = jax.tree_util.tree_map(lambda x: x[0], compiled.carry.log)
    (twin,) = summary_rows(first_world)
    row, reference = executed(setup, study, run, offline)
    np.testing.assert_array_equal(np.asarray(twin.payload), np.asarray(reference))
    np.testing.assert_array_equal(row.payload, np.asarray(reference))
    doc = dump_run(
        study,
        physical_registry(),
        executor="host",
        bindings=Bindings(),
        kinds=KINDS,
    )
    path = tmp_path / "run.json"
    path.write_text(doc.to_json())
    child = (
        "import json, sys\n"
        "import jax\n"
        "jax.config.update('jax_enable_x64', True)\n"
        "from tests.adapters.test_summary_channel import KINDS, world, summary_rows\n"
        "from spaceodyssey.codec import Bindings, load_run\n"
        "from spaceodyssey.core.executors.host import HostExecutor\n"
        "run = load_run(open(sys.argv[1]).read(), Bindings(), KINDS)\n"
        "truth, root = world()\n"
        "out = HostExecutor().run(run.study, truth, root, 30)\n"
        "rows = summary_rows(out.carry.log)\n"
        "print('ROW' + json.dumps([float(x) for x in rows[0].payload]))\n"
    )
    root = Path(__file__).parents[2]
    done = subprocess.run(
        [sys.executable, "-c", child, str(path)],
        capture_output=True,
        text=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": f"{root / 'src'}:{root}"},
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    line = next(x for x in done.stdout.splitlines() if x.startswith("ROW"))
    (mine,) = summary_rows(run.carry.log)
    np.testing.assert_array_equal(np.asarray(json.loads(line[3:])), mine.payload)
    assert load_run(doc.to_json(), Bindings(), KINDS).study.channel_model == (
        study.channel_model
    )
