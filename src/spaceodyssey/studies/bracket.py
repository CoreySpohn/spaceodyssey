"""The bracket study: one study of many stars on one telescope and one budget.

Stars are subjects of a single study; each has `n_slots` planet slots. Slots are
sorted by descending peak flux ratio, so the slot index carries the existence
ordering of the truth; to the belief the slots are exchangeable and it must not read
meaning into the index. One lane is
the telescope (every detection and block mode runs there and charges the shared
budget row) and a second lane carries the catalog mode, which occupies one tick and
costs nothing. Grants on the observer declare measurement delivery, catalog access,
policy access and the refit kernel (`GRANT_PRESETS`). The registry lowers the
grants into audience flags for each payload column of the record kinds.

- `bracket_registry(grants, n_slots, n_bands)` declares the kind registry.
- `bracket_study(...)` builds the `Study` of an arm.
- `PackagePolicy` selects conditional packages against the visible ledger.
- `bracket_boundary(study)` reads the refit kernel from the observer's grants.
- `BoundExecutor` is the compiled executor with a boundary attached, so the batteries
  that call `run_worlds` without a `refit` argument run the arm as it is.

Declared conventions:

- The two product channels of the refresh schedule are not admission channels.
- Kickoff is the policy's: one catalog observation per star, chained by waits so that
  each completion lands before the next admission (a new admission on a busy lane
  makes the earlier completion stale and its row is lost).
- Zero transition ticks and a 1 ms clock; the refresh grid is one checkpoint a day.
- A detection or block mode charges its occupancy in days to the shared row; the
  asset and program rows are not binding.
"""

import csv
import dataclasses
import math
import pathlib
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import d2s, s2d
from jaxtyping import Float, Int
from skyscapes.population import OccurrenceTable, PlanetTypes, PopulationConfig

from spaceodyssey.adapters.jaxedith import ObservingModeChannel, SummaryScene
from spaceodyssey.adapters.measurement import (
    MEASUREMENT_KIND_NAME,
    measurement_kind,
    measurement_template,
)
from spaceodyssey.adapters.modes import ModeSpec, ModeTable
from spaceodyssey.adapters.parameters import (
    CATALOG_GROUPS,
    CATALOG_KIND_NAME,
    catalog_group_of,
    catalog_kind,
)
from spaceodyssey.adapters.photomancy import (
    GRID_SPACING_D,
    N_GRID,
    SLOT_DEGRADED,
    SLOT_KEPT,
    SLOT_OK,
    SUMMARY_LEAVES,
    DecisionSummary,
    ReportRefit,
)
from spaceodyssey.adapters.skyscapes import (
    Band,
    ExoVistaWorldSource,
    PopulationModel,
    PopulationTruth,
    StarSpec,
    SupportRange,
)
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import (
    F64,
    I64,
    PUBLIC,
    AdmissionCode,
    Audience,
    CoincidentOrder,
    Disposition,
    EffectKind,
    Purpose,
    RecordStatus,
    Scope,
)
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.keys import Coupling
from spaceodyssey.core.kinds import (
    AbstractObserver,
    AbstractPolicy,
    ObserverCapabilities,
)
from spaceodyssey.core.ledger import ChannelTable
from spaceodyssey.core.records import (
    KindRegistry,
    RecordKind,
    layout_size,
    pack,
    unpack,
)
from spaceodyssey.core.refresh import REFRESH_PRODUCT_KIND, RefreshSchedule
from spaceodyssey.core.snapshot import BeliefLayout
from spaceodyssey.core.study import Study
from spaceodyssey.criteria.base import AbstractCriterion
from spaceodyssey.criteria.claims import EarthCandidateClaim
from spaceodyssey.criteria.transition import TransitionSettings, TransitionTable
from spaceodyssey.studies import builder, effect_rows
from spaceodyssey.studies.scoring import CriterionObjective, row_log, verdict
from spaceodyssey.studies.selection import act, capacity_of, select, trace_scalars

# Candidates need repeated detections, a supported fit and a tight sky position.
CANDIDATE_DETECTIONS, CANDIDATE_SCATTER_LOD = 3, 1.0
# Kind indices of `bracket_registry`.
KIND_NONE, KIND_PRODUCT, KIND_MEASUREMENT, KIND_CATALOG = range(4)
# Channel ids of the modes start here; the refresh product channels are 90 and 91.
FIRST_CHANNEL_ID = 10
# Seconds in a day, and the ticks in a day of the default 1 ms clock (the clock a
# bracket study runs on is the profile's declared `tick_length_s`).
SECONDS_PER_DAY = d2s
TICKS_PER_DAY = 86_400_000


# Occupancy of the catalog mode, in ticks.
CATALOG_TICKS = 1


class Grants(eqx.Module):
    """Declared measurement delivery, catalog access and refit kernel."""

    delivery: str = eqx.field(default="measured", static=True)
    catalog_groups: tuple[str, ...] = eqx.field(default=(), static=True)
    policy_reads_measurements: bool = eqx.field(default=True, static=True)
    kernel: str = eqx.field(default="posterior", static=True)

    def __check_init__(self):
        """Reject unknown delivery, catalog groups and kernels."""
        if self.delivery not in ("measured", "true_snr"):
            raise ValueError(f"unknown delivery {self.delivery!r}")
        unknown = set(self.catalog_groups) - set(CATALOG_GROUPS)
        if unknown:
            raise ValueError(f"unknown catalog groups {sorted(unknown)}")
        if self.kernel not in ("posterior", "truth", "shrinkage"):
            raise ValueError(f"unknown kernel {self.kernel!r}")


GRANT_PRESETS = {
    "replan": Grants(),
    "oracle": Grants(catalog_groups=CATALOG_GROUPS, kernel="truth"),
    "prior_planned": Grants(policy_reads_measurements=False),
    "shrinkage": Grants(catalog_groups=("ephemeris", "class"), kernel="shrinkage"),
}


def bracket_registry(grants, n_slots, n_bands):
    """The kinds and column audiences declared by the observer's grants."""
    measured = Audience.OBSERVER | Audience.SCORER
    if grants.policy_reads_measurements:
        measured |= Audience.POLICY
    measurement = measurement_kind(n_slots, n_bands, measured)
    if grants.delivery == "true_snr":
        measurement = dataclasses.replace(
            measurement,
            columns=tuple(
                dataclasses.replace(c, audience=measured)
                if c.name.startswith("true_")
                else c
                for c in measurement.columns
            ),
        )
    return KindRegistry(
        (
            RecordKind("none", ()),
            REFRESH_PRODUCT_KIND,
            measurement,
            catalog_kind(n_slots, n_bands, grants.catalog_groups),
        )
    )


# Days between forecast renewals of a summary, and the per-star rows a long study holds.
RENEWAL_D = GRID_SPACING_D * (N_GRID - 1)
# Refit products one renewal or visit adds to the log.
REFIT_ROWS = 2
MAX_VISITS = 24
MAX_SPECTRA = 12


def default_capacities(
    n_subjects,
    n_slots,
    *,
    n_channels,
    n_pstate: int,
    horizon_d,
    max_visits=MAX_VISITS,
    max_spectra=MAX_SPECTRA,
    ticks_per_day=None,
):
    """Capacities of the bracket study of `n_subjects` stars.

    `n_pstate` is the width declared by the policy state layout. The required
    `horizon_d` bounds counts whatever the observing times, not typical counts.
    A star holds one catalog row, at most `max_visits` imaging rows and at most
    `max_spectra` spectrograph rows. Each imaging row and forecast renewal adds
    `REFIT_ROWS` refit products; spectrograph rows add none. The renewal bound is
    `ceil(horizon_d / RENEWAL_D) + 1` per star.

    - `n_log` holds those rows of every star, with a quarter spare, in steps of 256.
    - `rows_per_subject` holds the catalog, imaging and spectrograph rows of a
      star, plus three spare.
    - `max_sid` is the next power of two above two ids per row, two per day for the
      policy's waits and eight per star; it must leave `max_tick` above the horizon,
      so a bound that needs more ids than that allows is refused. The horizon is
      counted in ticks of `ticks_per_day` (the profile's declared tick length;
      `TICKS_PER_DAY` when not given), so a longer tick admits more stars.

    The floors are `n_log` 1024, `rows_per_subject` 32 and `max_sid` 4096.
    """
    renewals = math.ceil(horizon_d / RENEWAL_D) + 1
    per_star = 1 + max_visits + max_spectra
    refits = REFIT_ROWS * (renewals + max_visits)
    need = n_subjects * (per_star + refits)
    n_log = max(1024, 256 * math.ceil(1.25 * need / 256))
    rows = max(32, per_star + 3)
    ids = 2 * need + 2 * math.ceil(horizon_d) + 8 * n_subjects
    max_sid = max(4096, 1 << (ids - 1).bit_length())
    caps = Capacities(
        n_subjects=n_subjects,
        n_assets=2,
        n_lanes=2,
        n_programs=1,
        n_resources=3,
        n_channels=n_channels,
        n_log=n_log,
        rows_per_subject=rows,
        refit_lanes=2,
        n_pstate=n_pstate,
        n_an=1,
        max_sid=max_sid,
    )
    per_day = TICKS_PER_DAY if ticks_per_day is None else int(ticks_per_day)
    if caps.max_tick < horizon_d * per_day:
        raise ValueError(
            f"{n_subjects} stars over {horizon_d} d need max_sid {max_sid}, which"
            f" leaves max_tick {caps.max_tick} below the horizon of"
            f" {horizon_d * per_day} ticks; declare a longer tick_length_s"
        )
    return caps


def channel_table(channel, caps, *, budget_unit_s=None):
    """Charge each observation's science plus overhead to telescope_days.

    The catalog occupies one tick and charges nothing.
    """
    built = channel.assets()
    clock = built.context.clock
    settle = int(built.context.settle_ticks)
    per_day = int(clock.ticks_per_day)
    tick_s = float(clock.clock.tick_length_s)
    occupancy, lane, kind = [], [], []
    table = channel.mode_table
    for c in range(table.catalog_channel):
        occupancy.append(settle + round(table.duration_s(c) / tick_s))
        lane.append(0)
        kind.append(KIND_MEASUREMENT)
    occupancy.append(CATALOG_TICKS)
    lane.append(1)
    kind.append(KIND_CATALOG)
    n = len(occupancy)
    if n != caps.n_channels:
        raise ValueError(f"caps.n_channels is {caps.n_channels}, the channel has {n}")
    occupancy = jnp.asarray(occupancy, I64)
    days = occupancy.astype(F64) * (1.0 / per_day)
    on_telescope = jnp.asarray(lane, I64) == 0
    cost = jnp.zeros((n, caps.n_resources), F64)
    cost = cost.at[:, table.resource("telescope_days")].set(
        jnp.where(on_telescope, days, 0.0)
    )
    return ChannelTable(
        id=jnp.arange(FIRST_CHANNEL_ID, FIRST_CHANNEL_ID + n, dtype=I64),
        lane=jnp.asarray(lane, I64),
        occupancy_ticks=occupancy,
        latency_ticks=jnp.zeros(n, I64),
        kind=jnp.asarray(kind, I64),
        audience=jnp.full(n, PUBLIC, I64),
        cost=cost,
        charge=cost,
        coincident_order=jnp.full(n, CoincidentOrder.FAULT_FIRST, I64),
        key_groups=jnp.asarray(
            [
                [
                    b == table.band_index(table.modes[table.mode_of(c)].name)
                    for b in range(table.n_bands)
                ]
                for c in range(table.catalog_channel)
            ]
            + [[False] * table.n_bands],
            bool,
        )
        if caps.n_key_groups
        else None,
    )


def bracket_study(
    source,
    channel,
    *,
    horizon_d,
    budget_d,
    observer,
    policy,
    planning_model=None,
    belief_population=None,
    caps=None,
):
    """The stars of `source` on one telescope, with grants on the observer.

    Args:
        source: The world source (stored universes), whose stars are the subjects.
        channel: The `ObservingModeChannel` of those stars.
        horizon_d: Study length in days.
        budget_d: Telescope days on the shared row.
        caps: Capacities; defaults to `default_capacities` for this horizon, the
            policy's visit and spectrum limits.
        observer: The summary observer. It folds criterion state incrementally
            and refits the measurement and catalog kinds.
        policy: The decision policy.
        belief_population: The population decisions and refits start from; `None`
            is the world source's declared one (`Study.decision_view`).
        planning_model: The channel model decisions and refits plan with; `None` is
            `channel`.
    """
    if channel.mode_table.bands != source.population.bands:
        raise ValueError("the channel and source disagree on the declared bands")
    built = channel.assets()
    if isinstance(observer, SummaryObserver):
        observer = dataclasses.replace(
            observer,
            mode_table=channel.mode_table,
            ticks_per_day=built.context.clock.ticks_per_day,
        )
    n_stars = int(channel.catalog_mass_kg.shape[0])
    if len(source.stars) != n_stars:
        raise ValueError("the world source and the channel disagree on the stars")
    if isinstance(policy, PackagePolicy):
        renewals = math.ceil(horizon_d / (GRID_SPACING_D * (N_GRID - 1.5))) + 1
        # Row completions and refit products can interrupt daily waits.
        trace_capacity = (
            math.ceil(horizon_d)
            + 1
            + n_stars
            * (4 * (1 + policy.max_visits + policy.max_spectra) + REFIT_ROWS * renewals)
        )
        policy = dataclasses.replace(policy, trace_capacity=trace_capacity)
    caps = caps or default_capacities(
        n_stars,
        channel.n_slots,
        n_channels=channel.n_channels,
        n_pstate=policy.state_size(n_stars),
        horizon_d=horizon_d,
        max_visits=getattr(policy, "max_visits", MAX_VISITS),
        max_spectra=getattr(policy, "max_spectra", MAX_SPECTRA),
        ticks_per_day=built.context.clock.ticks_per_day,
    )
    per_day = int(built.context.clock.ticks_per_day)
    caps = dataclasses.replace(
        caps,
        n_key_groups=channel.n_bands,
        snapshot_balances=True,
        balance_quantum=1.0 / per_day,
        balance_round_down=True,
    )
    caps = bracket_registry(
        observer.grants, channel.n_slots, n_bands=channel.n_bands
    ).lower(caps, observer)
    horizon = round(horizon_d * per_day)
    # These rows cannot bind before the shared budget and fit the tick quantum.
    nonbinding_d = max(float(horizon_d), float(budget_d), 1.0)
    budget = jnp.full((caps.n_balance_rows, caps.n_resources), nonbinding_d, F64)
    resource = channel.mode_table.resource("telescope_days")
    budget = budget.at[-1, resource].set(float(budget_d))
    schedule = RefreshSchedule(per_day, 0, *builder.PRODUCT_CHANNELS, KIND_PRODUCT)
    channels = channel_table(channel, caps)
    if isinstance(policy, PackagePolicy):
        assumed = built if planning_model is None else planning_model.assets()
        band_index = (
            channel.imaging_band_index
            if planning_model is None
            else planning_model.imaging_band_index
        )
        policy = dataclasses.replace(
            policy,
            effect_caps=caps,
            ticks_per_day=per_day,
            settings=TransitionSettings(
                candidate_detections=policy.candidate_detections,
                candidate_scatter_lod=policy.candidate_scatter_lod,
                imaging_lod_arcsec=assumed.reports[band_index].lod_arcsec,
                channel_cost=channels.cost,
                resources=channel.mode_table.resources,
            ),
        )
    study = builder.survey_1(
        caps,
        channels=channels,
        channel_model=channel,
        budget=budget,
        observer=observer,
        transition_ticks=jnp.zeros(caps.n_lanes, I64),
        lane_asset=jnp.asarray([0, 1], I64),
        refresh=schedule,
        policies=(policy,),
        horizon_tick=horizon,
        phase_until=(horizon,),
    )
    study = dataclasses.replace(
        study,
        objective=CriterionObjective(),
        clock=built.context.clock.clock,
        world_source=source,
        belief_population=belief_population,
        planning_model=planning_model,
        coupling=Coupling(
            shared=((Purpose.MEASUREMENT, ("channel", "acquisition_index")),)
        ),
    )

    if isinstance(observer, SummaryObserver):
        prior = ReportRefit(kernel="posterior").initial_summary(study)
        study = dataclasses.replace(
            study, observer=dataclasses.replace(observer, prior=prior)
        )
    return study


def ladder_studies(
    base: Study, criteria: Sequence[AbstractCriterion]
) -> tuple[Study, ...]:
    """Replace only the criterion on studies sharing the base channel table.

    Build the base channel with ``union_modes`` of all entries' declarations.
    Entries must share the base criterion's concrete family, packed state width
    and ordered claim definitions, since its prior and claim probabilities are
    retained. The observer initializes packed state from each entry at runtime.
    """
    observer = base.observer
    modes = base.channel_model.mode_table
    names = {mode.name for mode in modes.modes}
    n_tracks = observer.prior.p_exist.shape[1]
    criterion = observer.criterion
    width = layout_size(criterion.state_template(n_tracks, modes.n_bands))
    studies = []
    for entry in criteria:
        for name in entry.mode_names:
            if name not in names:
                raise ValueError(
                    f"criterion mode {name!r} is absent from the base table"
                )
        if type(entry) is not type(criterion):
            raise ValueError("ladder entries must share the base criterion family")
        if layout_size(entry.state_template(n_tracks, modes.n_bands)) != width:
            raise ValueError("ladder entries must share the base criterion state width")
        if not eqx.tree_equal(
            entry.claims, criterion.claims
        ) or observer.prior.p_claim.shape[-1] != len(entry.claims):
            raise ValueError(
                "ladder entries must share the base criterion claim schema"
            )
        studies.append(
            dataclasses.replace(
                base, observer=dataclasses.replace(observer, criterion=entry)
            )
        )
    return tuple(studies)


def bracket_boundary(study, **settings):
    """The refit boundary declared by the study observer's grants.

    Construction refuses a registry that hides a catalog group the kernel needs.
    `settings` are the kernel's own. The world source is not read: the kernels
    take their prior
    bounds from the study's belief population at refit time.
    """
    return ReportRefit(
        kernel=study.observer.grants.kernel,
        truth_groups=study.observer.grants.catalog_groups,
        caps=study.caps,
        **settings,
    )


def grants_in_force(study) -> dict[str, str]:
    """Decode effective access from the lowered study, rather than a preset label."""
    caps = study.caps
    observer = study.observer
    registry = bracket_registry(
        Grants(), observer.prior.p_exist.shape[1], observer.mode_table.n_bands
    )

    def columns(kind):
        layout = registry[registry.index(kind)]
        audience = caps.payload_audience[caps.kind_names.index(kind)]
        return {c.name: a for c, a in zip(layout.columns, audience, strict=False)}

    measured = columns(MEASUREMENT_KIND_NAME)
    catalog = columns(CATALOG_KIND_NAME)
    groups = [
        g
        for g in CATALOG_GROUPS
        if all(
            a & Audience.OBSERVER
            for c, a in catalog.items()
            if catalog_group_of(c) == g
        )
    ]
    true_signal = all(
        a & Audience.OBSERVER for c, a in measured.items() if c.startswith("true_")
    )
    policy_reads = all(
        a & Audience.POLICY
        for c, a in measured.items()
        if not c.startswith(("identity", "true_"))
    )
    return {
        "association": study.channel_model.association,
        "model": "truth" if study.planning_model is None else "declared",
        "population": "truth" if study.belief_population is None else "declared",
        "delivery": "true_snr" if true_signal else "measured",
        "catalog": ",".join(groups) or "none",
        "policy_reads": "yes" if policy_reads else "no",
        "kernel": study.observer.grants.kernel,
    }


def grants_line(study) -> str:
    """A printable line of the grants recorded in the study document."""
    return " ".join(f"{key}={value}" for key, value in grants_in_force(study).items())


class BoundExecutor(CompiledExecutor):
    """The compiled executor with an arm's refit boundary attached.

    `truth_swap` and the other batteries call `run_worlds` with no `refit`, so a
    study that refits needs the boundary to travel with the executor.
    """

    boundary: object = eqx.field(default=None, static=True)

    def run_worlds(self, study, truths, roots, n_steps, **kw):
        """`run` with the bound boundary unless the call passes its own."""
        if self.boundary is not None:
            kw.setdefault("refit", self.boundary)
        return self.run(study, truths, roots, n_steps, **kw)


# Future epochs of the summary and their spacing [d].
N_NODES = N_GRID
NODE_SPACING_D = GRID_SPACING_D


class PolicyState(eqx.Module):
    """Per-star row counts and last ticks, kickoff, stop and price trace."""

    last_tick: jax.Array
    n_imaging: jax.Array
    n_spectra: jax.Array
    kick: jax.Array
    stopped: jax.Array
    trace: jax.Array
    n_trace: jax.Array


class PackagePolicy(AbstractPolicy):
    """Select conditional packages against the ledger after catalog kickoff.

    The builder binds public operational settings once. Decisions read only the
    snapshot, criterion, grants, mode table and horizon of the decision view.
    Each fresh post-kickoff decision appends one price trace row.
    """

    trace_capacity: int = eqx.field(default=1, static=True)
    max_visits: int = eqx.field(default=MAX_VISITS, static=True)
    max_spectra: int = eqx.field(default=MAX_SPECTRA, static=True)
    candidate_detections: int = eqx.field(default=3, static=True)
    candidate_scatter_lod: jax.Array = eqx.field(
        default=1.0, converter=lambda x: jnp.asarray(x, F64)
    )
    n_bisect: int = eqx.field(default=48, static=True)
    effect_caps: Capacities | None = eqx.field(default=None, static=True, kw_only=True)
    ticks_per_day: int = eqx.field(default=TICKS_PER_DAY, static=True, kw_only=True)
    settings: TransitionSettings | None = eqx.field(default=None, kw_only=True)

    def __check_init__(self):
        """Refuse unusable trace shapes and row limits."""
        if self.trace_capacity < 1 or min(self.max_visits, self.max_spectra) < 0:
            raise ValueError("trace capacity must be positive and row caps nonnegative")

    def _layout(self, n_subjects):
        return PolicyState(
            last_tick=jnp.zeros(n_subjects, I64),
            n_imaging=jnp.zeros(n_subjects, I64),
            n_spectra=jnp.zeros(n_subjects, I64),
            kick=jnp.asarray(0, I64),
            stopped=jnp.asarray(False),
            trace=jnp.zeros((self.trace_capacity, 6), F64),
            n_trace=jnp.asarray(0, I64),
        )

    def state_layout(self, caps) -> PolicyState:
        """The codec template for this study's subject count."""
        return self._layout(caps.n_subjects)

    def state_size(self, n_subjects) -> int:
        """Width of the codec-packed named state."""
        return layout_size(self._layout(n_subjects))

    def declare(self):
        """Permit the campaign drain at the horizon."""
        return Declaration(stops=((Scope.CAMPAIGN, Disposition.DRAIN),))

    def decide(self, pstate, view, study, key):
        """Return a catalog action, a refit wait, or one selected package step."""
        caps, settings = self.effect_caps, self.settings
        if caps is None or settings is None:
            raise ValueError("bind the package policy with bracket_study")
        settings = dataclasses.replace(
            settings,
            candidate_detections=self.candidate_detections,
            candidate_scatter_lod=self.candidate_scatter_lod,
        )
        summary = view.belief
        n = summary.p_exist.shape[0]
        state = unpack(self._layout(n), pstate)
        modes = study.channel_model.mode_table
        criterion = study.observer.criterion
        tick, per_day = view.tick, self.ticks_per_day
        done = state.stopped | (tick >= study.horizon_tick)
        kicking = state.kick < n
        stale = jnp.any(state.n_imaging > summary.n_obs)
        if not study.observer.grants.policy_reads_measurements:
            stale = jnp.asarray(False)
        fresh = ~done & ~kicking & ~stale

        def refresh(state):
            template = criterion.state_template(summary.p_exist.shape[1], modes.n_bands)

            def options(star):
                one = dataclasses.replace(
                    summary, aux=jnp.broadcast_to(summary.aux, (n, summary.aux.size))
                )
                one = jax.tree.map(lambda x: x[star], one)
                imaging = jnp.isin(
                    jnp.arange(modes.catalog_channel),
                    jnp.asarray(modes.imaging_channels, I64),
                )
                allowed = jnp.where(
                    imaging,
                    state.n_imaging[star] < self.max_visits,
                    state.n_spectra[star] < self.max_spectra,
                )
                node_ticks = one.ref_tick + jnp.round(
                    jnp.arange(one.available.shape[0]) * NODE_SPACING_D * per_day
                ).astype(I64)
                one = dataclasses.replace(
                    one,
                    available=one.available
                    & allowed[None, :]
                    & (node_ticks <= study.horizon_tick)[:, None],
                )
                return criterion.options(
                    unpack(template, one.state), one, tick, modes, settings
                )

            options = jax.lax.map(options, jnp.arange(n))
            n_grid = summary.available.shape[1]
            current = jnp.clip(
                jnp.round((tick - summary.ref_tick) / (NODE_SPACING_D * per_day)),
                0,
                n_grid - 1,
            ).astype(I64)
            available = jnp.arange(n_grid)[None, :, None] >= current[:, None, None]
            first_imaging = jnp.isin(
                options.first_channel, jnp.asarray(modes.imaging_channels, I64)
            )
            under_cap = jnp.where(
                first_imaging,
                state.n_imaging[:, None, None] < self.max_visits,
                state.n_spectra[:, None, None] < self.max_spectra,
            )
            gain = jnp.where(available & under_cap, options.gain, 0.0)
            resource = modes.resource("telescope_days")
            days_left = jnp.maximum(
                (study.horizon_tick - tick).astype(F64) / per_day, 0.0
            )
            selection = select(
                gain,
                options.time,
                capacity_of(
                    jnp.maximum(view.balance[-1], 0.0),
                    days_left,
                    resource,
                ),
                self.n_bisect,
            )
            star, option, starts = act(
                selection, gain, options.time, current, ranking_resource=resource
            )
            safe_star, safe_option = jnp.maximum(star, 0), jnp.maximum(option, 0)
            channel = options.first_channel.reshape(n, -1)[safe_star, safe_option]
            # A queued wait can wake before the last exposure completes.
            acted = (jnp.arange(self.trace_capacity) < state.n_trace) & (
                state.trace[:, 4] > 0
            )
            last = jnp.max(jnp.where(acted, jnp.arange(self.trace_capacity), 0))
            last_d = state.trace[last, 1] - jnp.where(
                last > 0, state.trace[jnp.maximum(last - 1, 0), 1], 0.0
            )
            busy_until = jnp.max(state.last_tick) + jnp.round(last_d * per_day).astype(
                I64
            )
            starts &= ~jnp.any(acted) | (tick >= busy_until)
            imaging = jnp.isin(channel, jnp.asarray(modes.imaging_channels, I64))
            spent_d = jnp.where(starts, settings.channel_cost[channel, resource], 0.0)
            value = jnp.where(starts, gain.reshape(n, -1)[safe_star, safe_option], 0.0)
            previous = jnp.where(
                state.n_trace > 0,
                state.trace[jnp.maximum(state.n_trace - 1, 0)],
                jnp.zeros(6, F64),
            )
            finite = (gain > 0) & jnp.all(jnp.isfinite(options.time), axis=-1)
            window_d = jnp.max(jnp.where(finite, options.time[..., resource], 0.0))
            row = jnp.stack(
                [
                    selection.price[resource],
                    previous[1] + spent_d,
                    selection.selected_time[resource],
                    window_d,
                    starts.astype(F64),
                    previous[5] + value,
                ]
            )
            index = eqx.error_if(
                state.n_trace,
                fresh & (state.n_trace >= self.trace_capacity),
                "package policy trace capacity exhausted",
            )
            state = dataclasses.replace(
                state,
                trace=state.trace.at[index].set(row),
                n_trace=index + 1,
                last_tick=state.last_tick.at[safe_star].set(
                    jnp.where(starts, tick, state.last_tick[safe_star])
                ),
                n_imaging=state.n_imaging.at[safe_star].add(
                    (starts & imaging).astype(I64)
                ),
                n_spectra=state.n_spectra.at[safe_star].add(
                    (starts & ~imaging).astype(I64)
                ),
            )
            return state, safe_star, channel, starts

        state, star, channel, starts = jax.lax.cond(
            fresh,
            refresh,
            lambda state: (
                state,
                jnp.asarray(0, I64),
                jnp.asarray(0, I64),
                jnp.asarray(False),
            ),
            state,
        )
        kind = jnp.select(
            [done, kicking, starts],
            [EffectKind.STOP, EffectKind.OBSERVE, EffectKind.OBSERVE],
            EffectKind.WAIT,
        )
        star = jnp.where(kicking, jnp.minimum(state.kick, n - 1), star)
        channel = jnp.where(kicking, modes.catalog_channel, channel)
        # Daily waits share a tick so interruptions cannot create extra wait chains.
        wake = ((tick // per_day) + 1) * per_day + 1
        eff = effect_rows.effects(
            caps,
            kind=jnp.stack([kind]),
            subject=star,
            channel_index=channel,
            group=view.n_dec,
            tick=jnp.where(kind == EffectKind.WAIT, wake, 0),
            lane=0,
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
        )
        state = dataclasses.replace(
            state, kick=state.kick + (kicking & ~done).astype(I64), stopped=done
        )
        packed = pack(state)
        return (
            eff,
            jnp.pad(packed, (0, pstate.size - packed.size)),
            jnp.asarray(0.0, F64),
        )


def project_detections(belief):
    """The trace projection: detections per star, and a zero auxiliary vector."""
    return belief.n_det.sum(axis=-1).astype(F64), jnp.zeros_like(belief.aux)


# Slot leaves of a summary that a failed slot keeps (`status` takes the new code).
_SLOT_LEAVES = tuple(
    n
    for n in SUMMARY_LEAVES
    if n
    not in (
        "ref_tick",
        "n_obs",
        "available",
        "status",
    )
)


class SummaryObserver(AbstractObserver):
    """A `DecisionSummary` per star with incremental criterion state.

    `prior` is the first-tier summary of each star with no data; it is what the
    observer and the policy hold until the first refit, and it is built from the
    belief's own prior, never from the truth. Refits replace only SUMMARY_LEAVES;
    the packed criterion state survives every install. `bracket_study` binds
    `mode_table` from its channel before initialization or row processing.
    """

    subject_leaves = SUMMARY_LEAVES
    prior: DecisionSummary
    criterion: AbstractCriterion
    grants: Grants = Grants()
    mode_table: ModeTable | None = eqx.field(static=True, default=None, kw_only=True)
    catalog_kind: str = eqx.field(default=CATALOG_KIND_NAME, static=True)
    measurement_kind: str = eqx.field(default=MEASUREMENT_KIND_NAME, static=True)
    ticks_per_day: int = eqx.field(default=TICKS_PER_DAY, static=True)

    def capabilities(self):
        """Refit from imaging measurements and the truth catalog."""
        return ObserverCapabilities(
            incremental=True,
            refit=True,
            refit_kinds=(self.measurement_kind, self.catalog_kind),
        )

    def refits_on(self, rec) -> jax.Array:
        """Imaging rows, including nulls, and catalog rows mark a refit."""
        if self.mode_table is None:
            raise ValueError("bind the observer's mode table with bracket_study")
        return ~jnp.isin(
            rec.channel_index, jnp.asarray(self.mode_table.spectrograph_channels, I64)
        )

    def init(self, caps):
        """The prior summary and the criterion's initial packed state per star."""
        if self.mode_table is None:
            raise ValueError("bind the observer's mode table with bracket_study")
        n_tracks, n_bands = self.prior.p_exist.shape[1], self.mode_table.n_bands
        width = layout_size(self.criterion.state_template(n_tracks, n_bands))
        state = pack(self.criterion.init_state(n_tracks, n_bands))
        return dataclasses.replace(
            self.prior, state=jnp.broadcast_to(state, (caps.n_subjects, width))
        )

    def fold(self, belief, rec, caps, key=None):
        """Fold a measured row into its star's packed criterion state."""
        if self.mode_table is None:
            raise ValueError("bind the observer's mode table with bracket_study")
        n_tracks, n_bands = belief.p_exist.shape[1], self.mode_table.n_bands
        template = self.criterion.state_template(n_tracks, n_bands)

        def updated(state):
            row = unpack(measurement_template(n_tracks, n_bands), rec.payload)
            if self.grants.delivery == "true_snr":
                row = dataclasses.replace(
                    row, counts=row.true_counts, variance=row.true_variance
                )
            value = self.criterion.update(
                unpack(template, state[rec.subject]), row, self.mode_table
            )
            return state.at[rec.subject].set(pack(value))

        take = rec.valid & (rec.kind == caps.kind_names.index(self.measurement_kind))
        take &= rec.status == RecordStatus.COMPLETED
        state = jax.lax.cond(take, updated, lambda state: state, belief.state)
        return dataclasses.replace(belief, state=state)

    def fold_summary(self, summary, rec, caps, key=None):
        """Fold the measured columns released to the policy audience."""
        return self.fold(summary, rec, caps, key=key)

    def expired(self, summary, tick, caps):
        """The stars whose forecast window the clock has reached its last node of.

        A summary forecasts `N_NODES` epochs `NODE_SPACING_D` days apart from its
        refit tick; once the policy's current node is the last one (`g_cur` in
        `PackagePolicy.decide`) the forecast no longer moves with time, so the star
        is renewed by a refit from the same rows.
        """
        elapsed = (tick - summary.ref_tick) / (NODE_SPACING_D * self.ticks_per_day)
        return jnp.round(elapsed) >= N_NODES - 1

    def init_summary(self, caps):
        """The policy starts from the same prior."""
        return self.init(caps)

    def layout(self, caps):
        """One type for belief and summary; `n_obs` counts the visits read."""
        shapes = self.init(caps)
        return BeliefLayout(
            belief=shapes,
            summary=shapes,
            count_leaf="n_obs",
            project=project_detections,
        )

    def install(self, belief, fitted, subject):
        """Replace one star's leaves with those of the refit belief, slot by slot.

        A slot whose `status` is `SLOT_NO_SUPPORT` or `SLOT_FAILED` in `fitted` keeps
        its previous leaves and takes the code, so it cannot support a claim while
        the slots that fit are installed. A failed dark slot keeps its previous
        existence probability along with its other conditional leaves.
        """
        status = fitted.status[subject]
        kept = jnp.isin(status, jnp.asarray(SLOT_KEPT, I64))

        def merged(name):
            new, old = getattr(fitted, name)[subject], getattr(belief, name)[subject]
            if name == "p_exist":
                keep = kept & (fitted.n_det[subject] == 0)
            elif name in _SLOT_LEAVES:
                keep = kept.reshape(kept.shape + (1,) * (new.ndim - 1))
            else:
                return new
            return jnp.where(keep, old, new)

        return eqx.tree_at(
            lambda b: tuple(getattr(b, n) for n in SUMMARY_LEAVES),
            belief,
            tuple(
                getattr(belief, n).at[subject].set(merged(n)) for n in SUMMARY_LEAVES
            ),
        )


# -- Activity and paired grants ----------------------------------------------------


def price_trace(
    result, study
) -> tuple[Float[np.ndarray, "W T 6"], Int[np.ndarray, " W"]]:  # noqa: F722
    """Return padded host price rows [W, T, 6] and written lengths [W].

    Only ``rows[w, :lengths[w]]`` are valid. Unwritten rows are zeroed rather
    than exposed from the policy buffer. Columns are price, cumulative charged
    days, selected days, window days, acted, and cumulative value.
    """
    policy = study.agenda.policies[0]
    state = unpack(policy.state_layout(study.caps), result.carry.pstate)
    lengths = np.asarray(state.n_trace, np.int64)
    rows = np.zeros(state.trace.shape, np.float64)
    for w, n in enumerate(lengths):
        rows[w, :n] = np.asarray(state.trace[w, :n])
    return rows, lengths


def candidate_slots(belief, lod_arcsec):
    """The imaging candidates a summary nominates: ``bool[n_stars, n_slots]``.

    A slot is a candidate when it holds at least `CANDIDATE_DETECTIONS` detections,
    a slot status of `SLOT_OK` (a fit
    above the size floor, never a degraded, unsupported or failed one) and a
    predicted-position scatter under one lambda over D (`lod_arcsec`, in arcsec like
    the summary's `scatter`). The policy reads its own summary through this rule
    and the observer keeps its own summary.
    """
    return (
        (belief.n_det >= CANDIDATE_DETECTIONS)
        & (belief.status == SLOT_OK)
        & (belief.scatter < CANDIDATE_SCATTER_LOD * lod_arcsec)
    )


@dataclasses.dataclass(frozen=True)
class ActivityTable:
    """Per-star activity [W, n] and per-world accounting [W].

    Star leaves are n_visits, t_imaging and t_spectrograph (days), dropped and
    no_funds. World leaves are charged (days) and low_ess (degraded slots).
    """

    star: dict
    world: dict
    grants: str = ""
    price_trace: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def n_worlds(self):
        """Worlds in the table."""
        return int(next(iter(self.world.values())).shape[0])

    def to_csv(self, directory, name):
        """Write star and world CSVs and the study's grants."""
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        shape = next(iter(self.star.values())).shape
        world, star = np.indices(shape)
        frame = {"world": world.ravel(), "star": star.ravel()}
        frame.update({k: v.ravel() for k, v in self.star.items()})
        _write_columns(directory / f"{name}_star.csv", frame)
        _write_columns(
            directory / f"{name}_world.csv",
            {"world": np.arange(self.n_worlds), **self.world},
        )
        (directory / "grants.txt").write_text(f"{name}: {self.grants}\n")


def _write_columns(path, columns):
    """Write equal-length 1-D columns as a CSV with a header row."""
    names = list(columns)
    data = [np.asarray(columns[k]).reshape(-1) for k in names]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        writer.writerows(zip(*(d.tolist() for d in data), strict=True))


def _rows_of(log, w, kind):
    """The valid rows of world `w` of one kind, in completion order."""
    rows = np.flatnonzero((log["kind"][w] == kind) & log["valid"][w])
    return rows[np.argsort(log["end_tick"][w][rows], kind="stable")]


def activity_table(result, study) -> ActivityTable:
    """Activity and ledger accounting for all source worlds in source order.

    The result must cover study.world_source's complete world batch in source
    order. Every world's completed catalog must equal that source world's truth
    catalog. Subsets, permutations and missing catalogs raise ValueError; dropped
    counts are then read by source index.

    The ``trace_scalars`` late mask includes every measurement admission.
    The package policy's first ``n_subjects`` decisions request the
    kickoff catalogs; only later OBSERVE effects on woken steps are measurements.
    Kickoff, waits and padded steps are excluded, with no mission-time cutoff.
    """
    carry, trace = result.carry, result.trace
    channel = study.channel_model
    n_worlds = carry.log.valid.shape[0]
    restriction = (
        "activity_table requires all study.world_source worlds in source order "
        "with matching truth catalogs"
    )
    if n_worlds != len(study.world_source.dropped):
        raise ValueError(restriction)
    truths = study.world_source.take(jnp.arange(n_worlds))
    log = {
        name: np.asarray(getattr(carry.log, name))
        for name in ("kind", "subject", "valid", "payload", "end_tick")
    }
    for w in range(n_worlds):
        truth = jax.tree.map(lambda x, w=w: x[w], truths)
        for star in range(study.caps.n_subjects):
            indices = _rows_of(log, w, KIND_CATALOG)
            indices = indices[log["subject"][w, indices] == star]
            expected = np.asarray(channel._catalog_row(truth, star, 0, None, None))
            if not indices.size or not np.array_equal(
                log["payload"][w, indices[-1], : expected.size], expected
            ):
                raise ValueError(restriction)
    shape = (n_worlds, study.caps.n_subjects)
    star = {
        name: np.zeros(shape, np.float64)
        for name in ("n_visits", "t_imaging", "t_spectrograph", "no_funds")
    }
    rows = row_log(result, study)
    for i, w in enumerate(rows["world"]):
        subject = rows["star"][i]
        kind = channel.mode_table.modes[rows["mode"][i]].kind
        star["n_visits"][w, subject] += 1
        star[f"t_{kind}"][w, subject] += rows["occupied_s"][i] * s2d
    codes, subjects = np.asarray(trace.admit_code), np.asarray(trace.eff_subject)
    for w in range(n_worlds):
        refused = subjects[w][codes[w] == AdmissionCode.NO_FUNDS]
        for subject in refused:
            star["no_funds"][w, subject] += 1
    star["dropped"] = np.asarray(study.world_source.dropped, np.float64)
    resource = channel.mode_table.resource("telescope_days")
    budget = float(np.asarray(study.budget)[-1, resource])
    balance_d = np.asarray(carry.ledger.balance)[:, -1, resource]
    price_rows, price_lengths = price_trace(result, study)
    measurement = (
        np.asarray(trace.wake)[:, :, None]
        & (np.asarray(trace.n_dec)[:, :, None] > study.caps.n_subjects)
        & (np.asarray(trace.eff_kind) == EffectKind.OBSERVE)
    )
    world = {
        "charged": budget - balance_d,
        "low_ess": (np.asarray(carry.belief.status) == SLOT_DEGRADED)
        .sum(axis=(1, 2))
        .astype(np.float64),
        "early_overspend": np.zeros(n_worlds, np.int64),
        "stranded_d": np.zeros(n_worlds, np.float64),
    }
    for w, n in enumerate(price_lengths):
        world["early_overspend"][w], world["stranded_d"][w] = trace_scalars(
            jnp.asarray(price_rows[w, :n]),
            final_balance_d=jnp.asarray(balance_d[w]),
            admission_code=jnp.asarray(codes[w].ravel()),
            admission_late=jnp.asarray(measurement[w].ravel()),
        )
    return ActivityTable(
        star=star,
        world=world,
        grants=grants_line(study),
        price_trace=(price_rows, price_lengths),
    )


@dataclasses.dataclass(frozen=True)
class ArmsResult:
    """The outcome tables of the arms of one paired run, and their paired contrasts.

    Every arm ran the same truths and roots, so a contrast is a paired difference per
    world and its standard error is that of the mean of the differences.
    """

    activity: dict
    verdict: dict = dataclasses.field(default_factory=dict)

    def _endpoint(self, arm, endpoint):
        if endpoint in self.activity[arm].world:
            value = self.activity[arm].world[endpoint]
        elif endpoint in ("yield_", "wrong"):
            value = getattr(self.verdict[arm], endpoint)
        else:
            value = self.verdict[arm].counts[endpoint]
        return np.asarray(value, np.float64)

    def gap(self, a, b, endpoint):
        """``(mean, se, correlation, n)`` of `endpoint` of arm `a` minus arm `b`.

        The correlation is the Pearson correlation of the two arms' endpoints across
        worlds (NaN when either is constant); the standard error is the sample
        standard deviation of the differences over the square root of `n`.
        """
        xa, xb = self._endpoint(a, endpoint), self._endpoint(b, endpoint)
        n = xa.shape[0]
        d = xa - xb
        se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        if xa.std() == 0 or xb.std() == 0:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(xa, xb)[0, 1])
        return float(d.mean()), se, corr, n

    def log_ratio(self, a, b, endpoint):
        """``(value, se)`` of the log of the mean of `endpoint` in `a` over `b`.

        The standard error is the paired delta-method one: the standard deviation over
        worlds of ``x_a / mean_a - x_b / mean_b``, over the square root of `n`. When
        either mean is zero the log ratio is undefined and both values are NaN.
        """
        xa, xb = self._endpoint(a, endpoint), self._endpoint(b, endpoint)
        ma, mb = xa.mean(), xb.mean()
        if ma <= 0 or mb <= 0:
            return float("nan"), float("nan")
        n = xa.shape[0]
        se = float((xa / ma - xb / mb).std(ddof=1) / np.sqrt(n))
        return float(np.log(ma / mb)), se

    def to_csv(self, directory):
        """Write arm tables, valid price rows and each arm's grants."""
        for name, table in self.activity.items():
            table.to_csv(directory, name.replace("+", "_"))
        pathlib.Path(directory, "grants.txt").write_text(
            "".join(
                f"{name}: {table.grants}\n" for name, table in self.activity.items()
            )
        )
        names = (
            "price",
            "cum_charged_d",
            "selected_d",
            "window_d",
            "acted",
            "cum_value",
        )
        columns = {name: [] for name in ("arm", "world", "row", *names)}
        for arm, table in self.activity.items():
            if table.price_trace is None:
                continue
            rows, lengths = table.price_trace
            for w, n in enumerate(lengths):
                columns["arm"].extend([arm] * n)
                columns["world"].extend([w] * n)
                columns["row"].extend(range(n))
                for i, name in enumerate(names):
                    columns[name].extend(rows[w, :n, i])
        _write_columns(pathlib.Path(directory, "price_trace.csv"), columns)


def run_grants(presets, truths, roots, n_steps, *, executor=None) -> ArmsResult:
    """Run presets on shared truths and roots, retaining activity and verdicts.

    presets maps each name to (Study, boundary). The truth batch must cover each
    study's source worlds in source order, as required by activity_table.
    """
    activity, verdicts = {}, {}
    for name, (study, boundary) in presets.items():
        base = BoundExecutor() if executor is None else executor
        run = base.run(study, truths, roots, n_steps, refit=boundary)
        activity[name] = activity_table(run, study)
        verdicts[name] = verdict(run, study)
    return ArmsResult(activity=activity, verdict=verdicts)


BRACKET_KINDS = {
    ("observer", "bracket.summary", 1): SummaryObserver,
    ("policy", "bracket.package", 1): PackagePolicy,
    ("channel_model", "jaxedith.observing_mode", 2): ObservingModeChannel,
    ("world_source", "skyscapes.exovista", 1): ExoVistaWorldSource,
    ("objective", "criterion.verdict", 1): CriterionObjective,
    ("criterion", "criterion.transition_table", 1): TransitionTable,
    ("component", "criterion.transition_settings", 1): TransitionSettings,
    ("component", "skyscapes.band", 1): Band,
    ("component", "skyscapes.star_spec", 1): StarSpec,
    ("component", "jaxedith.summary_scene", 1): SummaryScene,
    ("component", "bracket.mode_spec", 1): ModeSpec,
    ("component", "bracket.mode_table", 1): ModeTable,
    ("component", "skyscapes.support_range", 1): SupportRange,
    ("component", "skyscapes.population_model", 1): PopulationModel,
    ("component", "skyscapes.population_truth", 1): PopulationTruth,
    ("component", "bracket.decision_summary", 1): DecisionSummary,
    ("component", "bracket.policy_state", 1): PolicyState,
    ("component", "bracket.grants", 1): Grants,
    ("component", "criterion.earth_candidate", 1): EarthCandidateClaim,
    ("component", "skyscapes.occurrence_table", 1): OccurrenceTable,
    ("component", "skyscapes.planet_types", 1): PlanetTypes,
    ("component", "skyscapes.population_config", 1): PopulationConfig,
}
"""Explicit kinds for bracket study documents; merge with `installed_kinds()`."""
