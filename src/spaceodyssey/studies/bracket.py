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
- `BracketPolicy` is the imaging decision rule every arm runs: follow-up, search.
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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import d2s, s2d

from spaceodyssey.adapters.measurement import (
    MEASUREMENT_KIND_NAME,
    measurement_kind,
    measurement_template,
)
from spaceodyssey.adapters.modes import ModeTable
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
from spaceodyssey.criteria.base import AbstractCriterion
from spaceodyssey.studies import builder, effect_rows
from spaceodyssey.studies.scoring import row_log, verdict

# Candidates need repeated detections, a supported fit and a tight sky position.
CANDIDATE_DETECTIONS, CANDIDATE_SCATTER_LOD = 3, 1.0
# Kind indices of `bracket_registry`.
KIND_NONE, KIND_PRODUCT, KIND_MEASUREMENT, KIND_CATALOG = range(4)
# The telescope and the catalog lanes, and the shared budget resource column.
LANE_TELESCOPE, LANE_CATALOG = 0, 1
SHARED_RESOURCE = 2
# Channel ids of the modes start here; the refresh product channels are 90 and 91.
FIRST_CHANNEL_ID = 10
# Seconds in a day, and the ticks in a day of the default 1 ms clock (the clock a
# bracket study runs on is the profile's declared `tick_length_s`).
SECONDS_PER_DAY = d2s
TICKS_PER_DAY = 86_400_000


# Occupancy of the catalog mode, in ticks.
CATALOG_TICKS = 1
# Asset and program rows hold this much; no channel charges them.
UNBOUNDED = 1.0e9


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


def pstate_size(n_subjects):
    """Per star: last tick, visits, searches; then charged days, kickoff and stop."""
    return n_subjects * 3 + 3


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
    horizon_d,
    max_visits=MAX_VISITS,
    max_spectra=MAX_SPECTRA,
    ticks_per_day=None,
):
    """Capacities of the bracket study of `n_subjects` stars.

    `n_pstate` holds the policy state (`pstate_size`). The required `horizon_d`
    sets upper bounds that hold whatever the observing times, not typical counts.
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
    n_pstate = pstate_size(n_subjects)
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
        lane.append(LANE_TELESCOPE)
        kind.append(KIND_MEASUREMENT)
    occupancy.append(CATALOG_TICKS)
    lane.append(LANE_CATALOG)
    kind.append(KIND_CATALOG)
    n = len(occupancy)
    if n != caps.n_channels:
        raise ValueError(f"caps.n_channels is {caps.n_channels}, the channel has {n}")
    occupancy = jnp.asarray(occupancy, I64)
    days = occupancy.astype(F64) / per_day
    on_telescope = jnp.asarray(lane, I64) == LANE_TELESCOPE
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
    caps = caps or default_capacities(
        n_stars,
        channel.n_slots,
        n_channels=channel.n_channels,
        horizon_d=horizon_d,
        max_visits=getattr(policy, "max_visits", MAX_VISITS),
        max_spectra=getattr(policy, "max_spectra", MAX_SPECTRA),
        ticks_per_day=built.context.clock.ticks_per_day,
    )
    if isinstance(policy, BracketPolicy):
        need = pstate_size(n_stars)
        if caps.n_pstate < need:
            raise ValueError(
                f"caps.n_pstate is {caps.n_pstate}; the policy needs {need}"
            )
    caps = dataclasses.replace(caps, n_key_groups=channel.n_bands)
    caps = bracket_registry(
        observer.grants, channel.n_slots, n_bands=channel.n_bands
    ).lower(caps, observer)
    per_day = int(built.context.clock.ticks_per_day)
    horizon = round(horizon_d * per_day)
    budget = jnp.full((caps.n_balance_rows, caps.n_resources), UNBOUNDED, F64)
    budget = budget.at[-1, SHARED_RESOURCE].set(float(budget_d))
    schedule = RefreshSchedule(per_day, 0, *builder.PRODUCT_CHANNELS, KIND_PRODUCT)
    study = builder.survey_1(
        caps,
        channels=channel_table(channel, caps),
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
    return dataclasses.replace(
        study,
        clock=built.context.clock.clock,
        world_source=source,
        belief_population=belief_population,
        planning_model=planning_model,
        coupling=Coupling(
            shared=((Purpose.MEASUREMENT, ("channel", "acquisition_index")),)
        ),
    )


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


# Future epochs of the summary, their spacing [d], detection and block modes.
N_NODES = N_GRID
NODE_SPACING_D = GRID_SPACING_D
N_DETECT, N_BLOCK = 6, 3


class BracketPolicy(AbstractPolicy):
    """Image known slots and search for new ones through each arm's summary.

    Kickoff observes one catalog row per star. Subsequent decisions wait for report
    refits, then favor follow-up of a known noncandidate slot at its best forecast
    node and mode, followed by search. Visits are at least 15 days apart and bounded
    by `max_visits` and `max_search`. A visit that exceeds the remaining budget
    falls back to the cheapest mode; one refused visit stops the policy.

    State holds each star's last visit tick, visit count and search count, followed
    by charged days, the kickoff count and a stop flag. The observer's grants
    determine whether decisions wait for measurement refits.
    """

    budget_d: float = eqx.field(static=True)
    n_slots: int = eqx.field(static=True)
    max_visits: int = eqx.field(default=24, static=True)
    max_search: int = eqx.field(default=8, static=True)
    followup_p: float = eqx.field(default=0.5, static=True)

    def declare(self):
        """The campaign drain the policy ends with."""
        return Declaration(stops=((Scope.CAMPAIGN, Disposition.DRAIN),))

    def decide(self, pstate, view, study, key):
        """One decision: kickoff, wait for a fresh summary, or act on it."""
        caps, ch = study.caps, study.channels
        model = study.channel_model
        n = caps.n_subjects
        stride = 3
        per_day = jnp.round(SECONDS_PER_DAY / study.clock.tick_length_s).astype(I64)
        stars = pstate[: n * stride].reshape(n, stride)
        last_tick, n_vis, n_search = stars[:, 0], stars[:, 1], stars[:, 2]
        charged, kick, stopped = pstate[n * stride : n * stride + 3]
        belief = view.belief
        n_detect = belief.p_det.shape[-1]
        cost = ch.cost[:, SHARED_RESOURCE]
        remaining = self.budget_d - charged
        tick = view.tick
        horizon = study.horizon_tick

        # Kickoff: the catalog row of star `kick`.
        kicking = kick < n
        # Fresh: every report visit made has been read by a refit.
        stale = jnp.any(n_vis > belief.n_obs.astype(F64))
        if not study.observer.grants.policy_reads_measurements:
            stale = jnp.asarray(False)
        next_day = ((tick // per_day) + 1) * per_day + 1

        node_ticks = belief.ref_tick[:, None] + jnp.round(
            jnp.arange(N_NODES) * NODE_SPACING_D * per_day
        ).astype(I64)
        g_cur = jnp.clip(
            jnp.round((tick - belief.ref_tick) / (NODE_SPACING_D * per_day)),
            0,
            N_NODES - 1,
        ).astype(I64)
        s_ix = jnp.arange(n)
        imaging_channels = jnp.asarray(model.mode_table.imaging_channels, I64)
        weight = jnp.maximum(cost[imaging_channels], 1e-9)

        lod = model.assets().reports[model.imaging_band_index].lod_arcsec
        candidate = candidate_slots(belief, lod)
        p_claim = (
            belief.p_claim.max(-1)
            if belief.p_claim.shape[-1]
            else jnp.ones_like(belief.p_exist)
        )
        known = (belief.n_det >= 1) | (p_claim >= 0.5)
        eligible = known & ~candidate
        eligible &= p_claim >= 0.05
        late = node_ticks >= (last_tick + 15 * per_day)[:, None]
        pd_val = (
            belief.p_exist[:, :, None, None]
            * belief.p_det
            / weight[None, None, None, :]
        )
        pd_val = jnp.where(
            late[:, None, :, None] & eligible[:, :, None, None], pd_val, -1.0
        )
        flat = pd_val.reshape(n, -1)
        fu_best = flat.argmax(axis=1)
        fu_score = flat.max(axis=1)
        fu_slot = fu_best // (N_NODES * n_detect)
        fu_node = (fu_best // n_detect) % N_NODES
        fu_mode = fu_best % n_detect
        fu_p = (
            belief.p_exist[s_ix, fu_slot]
            * belief.p_det[s_ix, fu_slot, fu_node, fu_mode]
        )
        fu_ok = (
            (fu_score > 0)
            & (fu_node == g_cur)
            & (fu_p >= self.followup_p)
            & (n_vis < self.max_visits)
        )
        fs = jnp.argmax(jnp.where(fu_ok, fu_score, -1.0))
        do_follow = fu_ok[fs]

        rested = (tick - last_tick) >= 15 * per_day
        slot_p = belief.p_exist[:, :, None] * belief.p_det[s_ix, :, g_cur, :]
        slot_p *= p_claim[:, :, None]
        slot_p = jnp.where(belief.n_det[:, :, None] == 0, slot_p, 0.0)
        sp = 1.0 - jnp.prod(1.0 - slot_p, axis=1)
        sp_val = sp / weight[None, :]
        sp_ok = (
            (rested | (n_vis == 0))[:, None]
            & (n_search < self.max_search)[:, None]
            & (n_vis < self.max_visits)[:, None]
            & (sp >= 0.02)
        )
        sp_flat = jnp.where(sp_ok, sp_val, -1.0).reshape(-1)
        sb = jnp.argmax(sp_flat)
        do_search = sp_flat[sb] > 0
        search_star, search_mode = sb // n_detect, sb % n_detect

        star = jnp.where(do_follow, fs, search_star)
        channel = imaging_channels[jnp.where(do_follow, fu_mode[fs], search_mode)]
        act = do_follow | do_search
        # Budget: a choice that does not fit falls back to the cheapest visit; when not
        # even that fits, one attempt is made and the policy stops.
        fits = cost[channel] <= remaining
        cheapest = imaging_channels[jnp.argmin(weight)]
        cheapest_fits = cost[cheapest] <= remaining
        channel = jnp.where(fits, channel, cheapest)
        attempt_last = act & ~fits & ~cheapest_fits
        use = act & ~jnp.asarray(stopped > 0.5)
        done = (tick >= horizon) | (stopped > 0.5)

        observe_cat = kicking
        kind = jnp.select(
            [done, observe_cat, stale & ~kicking, use],
            [EffectKind.STOP, EffectKind.OBSERVE, EffectKind.WAIT, EffectKind.OBSERVE],
            EffectKind.WAIT,
        )
        cat_channel = model.mode_table.catalog_channel
        eff_star = jnp.where(observe_cat, jnp.minimum(kick.astype(I64), n - 1), star)
        eff_channel = jnp.where(observe_cat, cat_channel, channel)
        wake = jnp.where(stale & ~kicking, next_day, tick + per_day)
        eff = effect_rows.effects(
            caps,
            kind=jnp.stack([kind, jnp.asarray(0, I64)]),
            subject=eff_star,
            channel_index=eff_channel,
            group=view.n_dec,
            tick=jnp.stack(
                [jnp.where(kind == EffectKind.WAIT, wake, 0), jnp.asarray(0, I64)]
            ),
            lane=0,
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
        )
        observing = (kind == EffectKind.OBSERVE) & ~observe_cat
        row = star
        is_search = observing & ~do_follow
        new = pstate
        new = new.at[row * stride].set(
            jnp.where(observing, tick.astype(F64), pstate[row * stride])
        )
        new = new.at[row * stride + 1].add(observing.astype(F64))
        new = new.at[row * stride + 2].add(is_search.astype(F64))
        base = n * stride
        spent = jnp.where(observing & (cost[channel] <= remaining), cost[channel], 0.0)
        new = new.at[base].add(spent)
        new = new.at[base + 1].add(observe_cat.astype(F64))
        new = new.at[base + 2].set(
            jnp.maximum(pstate[base + 2], (observing & attempt_last).astype(F64))
        )
        return eff, new, jnp.asarray(0.0, F64)


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
        `BracketPolicy.decide`) the forecast no longer moves with time, so the star
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
    budget = float(np.asarray(study.budget)[-1, SHARED_RESOURCE])
    world = {
        "charged": budget - np.asarray(carry.ledger.balance)[:, -1, SHARED_RESOURCE],
        "low_ess": (np.asarray(carry.belief.status) == SLOT_DEGRADED)
        .sum(axis=(1, 2))
        .astype(np.float64),
    }
    return ActivityTable(star=star, world=world, grants=grants_line(study))


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
        """Write each arm's tables and a `grants.txt` line for every arm."""
        for name, table in self.activity.items():
            table.to_csv(directory, name.replace("+", "_"))
        pathlib.Path(directory, "grants.txt").write_text(
            "".join(
                f"{name}: {table.grants}\n" for name, table in self.activity.items()
            )
        )


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
