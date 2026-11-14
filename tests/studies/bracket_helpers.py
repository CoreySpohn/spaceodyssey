"""Shared build of the bracket tests: two stored stars, two worlds, the oracle arm.

The fixture is one ExoVista universe of two stars (`tests/fixtures/exovista`); the
second world is the first with every period scaled, so the two worlds differ in
what the records show. The files tabulate 900 to 1094 nm, so the fixture declares
covered bands at 1000 nm and 940 nm; the survey
bands (`SURVEY_BANDS`, 500 and 910 nm) need a drawn or hand-built world.
"""

import dataclasses
import pathlib
from collections import namedtuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.adapters.jaxedith import (
    ObservingModeChannel,
    survey_grid_d,
    survey_table,
)
from spaceodyssey.adapters.measurement import measurement_template
from spaceodyssey.adapters.modes import profile_modes
from spaceodyssey.adapters.photomancy import GRID_SPACING_D, N_GRID, DecisionSummary
from spaceodyssey.adapters.physical_profile import JATIS_EAC1_V1
from spaceodyssey.adapters.skyscapes import Band, ExoVistaWorldSource, PopulationModel
from spaceodyssey.core.carry import init_carry
from spaceodyssey.core.enums import (
    EVERYONE,
    F64,
    I64,
    Audience,
    EffectKind,
    RecordStatus,
    RefitStatus,
)
from spaceodyssey.core.events import empty_event_row
from spaceodyssey.core.handlers import h_release
from spaceodyssey.core.records import pack, record_none, unpack, write_log
from spaceodyssey.core.snapshot import snapshot_view
from spaceodyssey.criteria.claims import EarthCandidateClaim
from spaceodyssey.criteria.transition import TransitionTable
from spaceodyssey.studies import bracket as br

FIXTURE = pathlib.Path(__file__).parents[1] / "fixtures" / "exovista"
N_SLOTS, N_WORLDS = 6, 2
N_TRACKS, N_BANDS, BAND = 3, 2, 1
HORIZON_D, BUDGET_D = 730.0, 200.0
N_STEPS = 3000
PERIOD_SCALE = 1.07
FIXTURE_BANDS = (Band(1000.0, 100.0), Band(940.0, 6.7))
SURVEY_BANDS = (Band(500.0, 100.0), Band(910.0, 6.5))  # detection, characterization


def fixture_files():
    """The two stars' files of the one fixture universe, nearest star first."""
    return sorted(FIXTURE.glob("*-HIP_*.fits"))


def fixture_source(n_worlds=N_WORLDS):
    """The fixture as a world source in `FIXTURE_BANDS`; later worlds scale periods."""
    files = fixture_files()
    source = ExoVistaWorldSource.from_files(
        [files] * n_worlds,
        n_slots=N_SLOTS,
        bands=FIXTURE_BANDS,
        generator=PopulationModel.exovista(),
    )
    scale = jnp.asarray([PERIOD_SCALE**w for w in range(n_worlds)])[:, None, None]
    return eqx.tree_at(
        lambda s: s.population.period_d,
        source,
        source.population.period_d * scale,
    )


def fixture_modes():
    """Survey modes in covered bands: stored spectra cover 900 to 1094 nm."""
    return tuple(
        dataclasses.replace(mode, band=band)
        for mode, band in zip(profile_modes(JATIS_EAC1_V1), FIXTURE_BANDS, strict=True)
    )


def fixture_channel(source, **kw):
    """The stored spectra cover 900 to 1094 nm, so the fixture observes the survey's
    modes in covered bands.
    """
    return ObservingModeChannel.from_stars(
        source.stars,
        n_slots=N_SLOTS,
        epoch_jd=source.epoch_jd,
        modes=fixture_modes(),
        durations_d=survey_grid_d(survey_table(JATIS_EAC1_V1), 3),
        profile=JATIS_EAC1_V1,
        **kw,
    )


def prior_summary(channel, n_stars, p_exist=0.0):
    """The first-tier summary the observer starts from (`p_exist` everywhere)."""
    summary = DecisionSummary.zeros(
        n_stars,
        N_SLOTS,
        N_GRID,
        channel.n_modes,
        n_an=1,
        n_claims=1,
        n_variants=channel.mode_table.catalog_channel,
    )
    return eqx.tree_at(
        lambda x: (x.p_exist, x.p_det, x.p_claim),
        summary,
        (summary.p_exist + p_exist, summary.p_det + 1.0, summary.p_claim + 1.0),
    )


def fixture_criterion():
    """One declared spectral requirement after the candidate imaging gate."""
    return TransitionTable(
        step_names=("spectrum",),
        labels=("met",),
        mode_names=("characterization",),
        claims=(EarthCandidateClaim(),),
        target_claim="earth_candidate",
        step_modes=[[True]],
        requirement_snr=[5.0],
        margin=[0.0],
        repeat=[1],
        on_pass=[1],
        on_fail=[-1],
        label_values=[1.0],
        start=0,
        step_labels=[-1],
    )


def needs_mode(name, criterion=None):
    """A one-step requirement on an explicitly named observing mode."""
    criterion = fixture_criterion() if criterion is None else criterion
    return dataclasses.replace(criterion, mode_names=(name,))


BracketBuild = namedtuple("BracketBuild", "source study executor truths roots run")


def build(
    arm="oracle",
    *,
    run=True,
    horizon_d=HORIZON_D,
    budget_d=BUDGET_D,
    channel=None,
    policy_kw=None,
    source=None,
    **settings,
):
    """The source, study, executor, truths, roots and run of the named grant preset.

    `settings` go to the arm's
    boundary, which reads the belief population. `channel` replaces
    the fixture's own channel (it must observe the fixture's stars); `policy_kw` are
    extra `PackagePolicy` arguments; `source` replaces the fixture's world source (it
    must hold `N_WORLDS` worlds of the same stars).
    """
    source = fixture_source() if source is None else source
    channel = channel or fixture_channel(source)
    n_stars = len(source.stars)
    study = br.bracket_study(
        source,
        channel,
        horizon_d=horizon_d,
        budget_d=budget_d,
        observer=br.SummaryObserver(
            grants=br.GRANT_PRESETS[arm],
            prior=prior_summary(channel, n_stars),
            criterion=fixture_criterion(),
        ),
        policy=br.PackagePolicy(**(policy_kw or {})),
    )
    executor = br.BoundExecutor(boundary=br.bracket_boundary(study, **settings))
    roots = jnp.stack(
        [jax.random.fold_in(jax.random.PRNGKey(0), i) for i in range(N_WORLDS)]
    )
    truths = source.take(jnp.arange(N_WORLDS))
    out = executor.run_worlds(study, truths, roots, N_STEPS) if run else None
    return BracketBuild(source, study, executor, truths, roots, out)


def spec_row(slot, counts, variance, true_counts=1000.0, true_variance=1000.0):
    row = jax.tree.map(
        lambda x: jnp.zeros(x.shape, x.dtype), measurement_template(N_TRACKS, N_BANDS)
    )
    return dataclasses.replace(
        row,
        mode=jnp.asarray(1, I64),
        valid_science=jnp.asarray(1.0, F64),
        assignment=jnp.asarray([slot, -1, -1], I64),
        counts=row.counts.at[0, BAND].set(counts),
        variance=row.variance.at[0, BAND].set(variance),
        true_counts=row.true_counts.at[0, BAND].set(true_counts),
        true_variance=row.true_variance.at[0, BAND].set(true_variance),
    )


_release = eqx.filter_jit(h_release)


def fold_rows(study, rows):
    carry = init_carry(study, None)
    for i, row in enumerate(rows):
        rec = dataclasses.replace(
            record_none(study.caps),
            kind=jnp.asarray(br.KIND_MEASUREMENT, I64),
            valid=jnp.asarray(True),
            acquisition_id=jnp.asarray(i + 1, I64),
            audience=jnp.asarray(EVERYONE, I64),
            status=jnp.asarray(RecordStatus.COMPLETED, I64),
            released_tick=jnp.asarray(-1, I64),
            accepted_tick=jnp.asarray(-1, I64),
            superseded_by=jnp.asarray(-1, I64),
            payload=jnp.pad(pack(row), (0, study.caps.payload_width - pack(row).size)),
            channel_index=jnp.asarray(1, I64),
        )
        carry, _ = write_log(carry, rec, study.caps)
        event = dataclasses.replace(
            empty_event_row(),
            slot=jnp.asarray(i, I64),
            audience=jnp.asarray(EVERYONE, I64),
            tick=jnp.asarray(i + 1, I64),
        )
        for _ in range(2):
            carry = _release(carry, event, study, None, jax.random.key(0))[0]
    np.testing.assert_array_equal(carry.belief.state, carry.pbelief.state)
    return carry.pbelief


def unpack_state(study, summary):
    return unpack(
        study.observer.criterion.state_template(N_TRACKS, N_BANDS), summary.state
    )


def columns_of(registry, kind_name):
    """Column audiences of one registered kind."""
    kind = next(k for k in registry if k.name == kind_name)
    return {c.name: c.audience for c in kind.columns}


def with_grants(study, grants):
    """Replace observer grants and lower the matching audience table."""
    observer = dataclasses.replace(study.observer, grants=grants)
    caps = br.bracket_registry(
        grants, observer.prior.p_exist.shape[1], observer.mode_table.n_bands
    ).lower(study.caps, observer)
    return dataclasses.replace(study, observer=observer, caps=caps)


def charged_days(run, study):
    """Telescope days charged to the shared ledger across worlds."""
    resource = study.channel_model.mode_table.resource("telescope_days")
    return float(
        np.asarray(run.carry.ledger.debited - run.carry.ledger.refunded)[
            ..., resource
        ].sum()
    )


def telescope_rows(run):
    """Completed measurement rows across worlds."""
    log = run.carry.log
    return int(
        np.count_nonzero(
            np.asarray(log.valid) & (np.asarray(log.kind) == br.KIND_MEASUREMENT)
        )
    )


__all__ = ["GRID_SPACING_D", "build", "fixture_source"]


Decision = namedtuple("Decision", "effects state trace_row selected_d")


@eqx.filter_jit
def _decide(study, pstate, view):
    return study.agenda.policies[0].decide(
        pstate, view, study.decision_view(), jax.random.key(0)
    )


def decide_once(study, *, summary=None, state=None, tick_d=0.0, balance_d=None):
    """One decision on the same policy snapshot and decision view as the executor."""
    carry = init_carry(study, None)
    policy = study.agenda.policies[0]
    carry = dataclasses.replace(
        carry, tick=jnp.asarray(round(tick_d * study.observer.ticks_per_day), I64)
    )
    if summary is not None:
        carry = dataclasses.replace(carry, pbelief=summary)
    if balance_d is not None:
        resource = study.channel_model.mode_table.resource("telescope_days")
        carry = dataclasses.replace(
            carry,
            ledger=dataclasses.replace(
                carry.ledger,
                balance=carry.ledger.balance.at[-1, resource].set(balance_d),
            ),
        )
    pstate = (
        carry.pstate
        if state is None
        else jnp.pad(pack(state), (0, study.caps.n_pstate - pack(state).size))
    )
    effects, packed, _ = _decide(
        study, pstate, snapshot_view(carry, Audience.POLICY, study.caps)
    )
    new = unpack(policy.state_layout(study.caps), packed)
    row = new.trace[max(int(new.n_trace) - 1, 0)]
    return Decision(effects, new, row, float(row[2]))


def state_with(study, **leaves):
    """Explicit fresh post-kickoff state, with broadcast overrides of named leaves."""
    state = study.agenda.policies[0].state_layout(study.caps)
    leaves = {"kick": study.caps.n_subjects, **leaves}
    return dataclasses.replace(
        state,
        **{
            name: jnp.full_like(getattr(state, name), value)
            for name, value in leaves.items()
        },
    )


def summary_selecting_node(study, node, *, detected=True):
    """A certain package whose only available forecast is the requested node."""
    summary = study.observer.init(study.caps)
    return dataclasses.replace(
        summary,
        p_exist=jnp.ones_like(summary.p_exist),
        p_claim=jnp.ones_like(summary.p_claim),
        p_det=jnp.ones_like(summary.p_det),
        n_det=jnp.full_like(summary.n_det, 3 if detected else 0),
        status=jnp.full_like(summary.status, br.SLOT_OK),
        scatter=jnp.zeros_like(summary.scatter),
        fc_counts=jnp.full_like(summary.fc_counts, 1000.0),
        fc_variance=jnp.ones_like(summary.fc_variance),
        available=jnp.zeros_like(summary.available).at[:, node, :].set(True),
    )


def effect_kind(decision):
    return int(decision.effects.kind[0])


def starts_spectrograph(decision, study):
    return (
        effect_kind(decision) == EffectKind.OBSERVE
        and int(decision.effects.channel_index[0])
        in study.channel_model.mode_table.spectrograph_channels
    )


def policy_trace(built, world=0):
    policy = built.study.agenda.policies[0]
    state = unpack(policy.state_layout(built.study.caps), built.run.carry.pstate[world])
    return np.asarray(state.trace)[: int(state.n_trace)]


def n_refreshes(built, world=0):
    """Count fresh policy decisions independently from admissions and refit products."""
    study, trace, log = built.study, built.run.trace, built.run.carry.log
    modes = study.channel_model.mode_table
    host = {
        name: np.asarray(getattr(log, name)[world])
        for name in (
            "valid",
            "kind",
            "subject",
            "channel",
            "channel_index",
            "start_tick",
            "released_tick",
            "end_tick",
            "payload",
        )
    }
    imaging = (
        host["valid"]
        & (host["kind"] == br.KIND_MEASUREMENT)
        & np.isin(host["channel_index"], modes.imaging_channels)
    )
    admitted = np.zeros(study.caps.n_subjects, int)
    read = admitted.copy()
    kick, count = 0, 0
    for i, fp in enumerate(np.asarray(trace.snapshot_fp[world])):
        if not fp:
            continue
        tick = int(trace.tick[world, i])
        products = np.flatnonzero(
            host["valid"]
            & (host["channel"] == study.refresh.policy_channel)
            & (np.arange(host["valid"].size) < int(trace.n_log[world, i]))
        )
        for row in products:
            if host["payload"][row, 0] in (RefitStatus.OK, RefitStatus.DEGRADED):
                subject = host["subject"][row]
                read[subject] = np.count_nonzero(
                    imaging
                    & (host["subject"] == subject)
                    & (host["released_tick"] <= host["end_tick"][row])
                )
        kind = int(trace.eff_kind[world, i, 0])
        if kick < study.caps.n_subjects:
            kick += kind == EffectKind.OBSERVE
            continue
        if kind == EffectKind.STOP:
            continue
        if not study.observer.grants.policy_reads_measurements or np.all(
            admitted <= read
        ):
            count += 1
        if kind == EffectKind.OBSERVE:
            subject = int(trace.eff_subject[world, i, 0])
            if np.any(
                imaging & (host["subject"] == subject) & (host["start_tick"] == tick)
            ):
                admitted[subject] += 1
    return count


def random_state(template, seed):
    leaves, structure = jax.tree.flatten(template)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    return jax.tree.unflatten(
        structure,
        [
            jax.random.bernoulli(k, shape=x.shape)
            if x.dtype == jnp.bool_
            else jax.random.normal(k, x.shape, x.dtype)
            if jnp.issubdtype(x.dtype, jnp.floating)
            else jax.random.randint(k, x.shape, 0, 3, x.dtype)
            for k, x in zip(keys, leaves, strict=True)
        ],
    )


def assert_trees_equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)
