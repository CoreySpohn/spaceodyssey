"""The bracket study: one study of many stars on one telescope and one budget.

Stars are subjects of a single study; each has `n_slots` planet slots. Slots are
sorted by descending peak flux ratio, so the slot index carries the existence
ordering of the truth; to the belief the slots are exchangeable and it must not read
meaning into the index. One lane is
the telescope (every detection and block mode runs there and charges the shared
budget row) and a second lane carries the catalog mode, which occupies one tick and
costs nothing. The arms of the bracket are registries: the same policy, boundary and
channel run in every arm, and an arm differs only in which audience may read which
payload columns of the three record kinds (`ARMS`).

- `bracket_registry(arm)` is the kind registry of an arm.
- `bracket_study(...)` builds the `Study` of an arm.
- `BracketPolicy` is the imaging decision rule every arm runs: follow-up, search.
- `bracket_boundary(study, arm)` is the refit boundary an arm runs under.
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
from hwoutils.constants import Msun2kg

from spaceodyssey.adapters.jaxedith import (
    block_outcome_kind,
    column_index,
    visit_columns,
    visit_report_kind,
)
from spaceodyssey.adapters.parameters import CATALOG_KIND_NAME, catalog_kind
from spaceodyssey.adapters.photomancy import (
    BLOCK_KIND_NAME,
    GRID_SPACING_D,
    N_GRID,
    REPORT_KIND_NAME,
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
    EVERYONE,
    F64,
    I64,
    PUBLIC,
    AdmissionCode,
    Audience,
    CoincidentOrder,
    Disposition,
    EffectKind,
    Scope,
)
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.kinds import (
    AbstractObserver,
    AbstractPolicy,
    ObserverCapabilities,
)
from spaceodyssey.core.ledger import ChannelTable
from spaceodyssey.core.records import KindRegistry, RecordKind
from spaceodyssey.core.refresh import REFRESH_PRODUCT_KIND, RefreshSchedule
from spaceodyssey.core.snapshot import BeliefLayout
from spaceodyssey.criteria.claims import AbstractClaim, claims_on_truth
from spaceodyssey.studies import builder, effect_rows

# Candidates need repeated detections, a supported fit and a tight sky position.
CANDIDATE_DETECTIONS, CANDIDATE_SCATTER_LOD = 3, 1.0
# Kind indices of `bracket_registry`.
KIND_NONE, KIND_PRODUCT, KIND_VISIT, KIND_BLOCK, KIND_CATALOG = range(5)
# The telescope and the catalog lanes, and the shared budget resource column.
LANE_TELESCOPE, LANE_CATALOG = 0, 1
SHARED_RESOURCE = 2
# Channel ids of the modes start here; the refresh product channels are 90 and 91.
FIRST_CHANNEL_ID = 10
# Seconds in a day, and the ticks in a day of the default 1 ms clock (the clock a
# bracket study runs on is the profile's declared `tick_length_s`).
SECONDS_PER_DAY = 86_400.0
TICKS_PER_DAY = 86_400_000


# Occupancy of the catalog mode, in ticks.
CATALOG_TICKS = 1
# Asset and program rows hold this much; no channel charges them.
UNBOUNDED = 1.0e9


@dataclasses.dataclass(frozen=True)
class Arm:
    """What an arm lets each audience read.

    Attributes:
        name: The arm's label.
        report_audience: Audience flags of every `visit_report` column.
        groups: Catalog groups open to the observer and the policy.
        kernel: The refit kernel the arm runs: `truth`, `posterior` or `shrinkage`.
    """

    name: str
    report_audience: int
    groups: tuple[str, ...]
    kernel: str


ARMS = {
    "oracle": Arm("oracle", EVERYONE, ("existence", "ephemeris", "class"), "truth"),
    "prior_planned": Arm(
        "prior_planned", Audience.OBSERVER | Audience.SCORER, (), "posterior"
    ),
    "replan": Arm("replan", EVERYONE, (), "posterior"),
    "replan+ephemeris": Arm("replan+ephemeris", EVERYONE, ("ephemeris",), "posterior"),
    "replan+class": Arm("replan+class", EVERYONE, ("class",), "posterior"),
    "shrinkage": Arm("shrinkage", EVERYONE, ("ephemeris", "class"), "shrinkage"),
}


def _arm(arm):
    """The `Arm` named or given."""
    if isinstance(arm, Arm):
        return arm
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; known: {sorted(ARMS)}")
    return ARMS[arm]


def bracket_registry(arm, n_slots=4, *, n_bands):
    """The kind registry of `arm`: none, refresh product, visit, block, catalog."""
    arm = _arm(arm)
    return KindRegistry(
        (
            RecordKind("none", ()),
            REFRESH_PRODUCT_KIND,
            visit_report_kind(n_slots, arm.report_audience),
            block_outcome_kind(n_slots),
            catalog_kind(n_slots, n_bands, arm.groups),
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
    """The `ChannelTable` of the observing modes of `channel`.

    A detection mode occupies the settling time plus its exposure; a block mode its
    length (plus the settling time on a survey channel, whose characterization
    observations carry the same overhead as detections); the catalog mode one tick.
    Telescope modes charge their occupancy in days to the shared resource; the
    catalog mode charges nothing.
    """
    built = channel.assets()
    clock = built.context.clock
    settle = int(built.context.settle_ticks)
    per_day = int(clock.ticks_per_day)
    tick_s = float(clock.clock.tick_length_s)
    occupancy, lane, kind = [], [], []
    for exposure in channel.exposures_s:
        occupancy.append(settle + round(exposure / tick_s))
        lane.append(LANE_TELESCOPE)
        kind.append(KIND_VISIT)
    for days in channel.block_days:
        # a survey characterization settles like a detection; a block is its length
        occupancy.append((settle if channel.survey else 0) + round(days * per_day))
        lane.append(LANE_TELESCOPE)
        kind.append(KIND_BLOCK)
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
    cost = cost.at[:, SHARED_RESOURCE].set(jnp.where(on_telescope, days, 0.0))
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
    )


def bracket_study(
    source,
    channel,
    *,
    arm,
    horizon_d,
    budget_d,
    caps=None,
    observer=None,
    policy=None,
    belief_population=None,
    planning_model=None,
):
    """The `Study` of `arm`: the stars of `source` on one telescope and one budget.

    Args:
        source: The world source (stored universes), whose stars are the subjects.
        channel: The `ObservingModeChannel` of those stars.
        arm: An arm name of `ARMS`, or an `Arm`.
        horizon_d: Study length in days.
        budget_d: Telescope days on the shared row.
        caps: Capacities; defaults to `default_capacities` for this horizon, the
            policy's visit and spectrum limits.
        observer: The summary observer. It must be non-incremental and refit the
            visit and catalog kinds.
        policy: The decision policy.
        belief_population: The population decisions and refits start from; `None`
            is the world source's declared one (`Study.decision_view`).
        planning_model: The channel model decisions and refits plan with; `None` is
            `channel`.
    """
    arm = _arm(arm)
    if channel.bands != source.population.bands:
        raise ValueError("the channel and source disagree on the declared bands")
    if observer is None or policy is None:
        raise ValueError("bracket_study needs an observer and a policy")
    built = channel.assets()
    if isinstance(observer, SummaryObserver):
        observer = dataclasses.replace(
            observer, ticks_per_day=built.context.clock.ticks_per_day
        )
    if isinstance(policy, BracketPolicy):
        reaches = bool(arm.report_audience & Audience.POLICY)
        policy = dataclasses.replace(policy, refits_reach_policy=reaches)
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
    caps = bracket_registry(arm, channel.n_slots, n_bands=channel.n_bands).lower(
        caps, observer
    )
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
    )


def bracket_boundary(study, arm, **settings):
    """The refit boundary of `arm` for `study` (`ReportRefit` with the arm's kernel).

    Construction refuses a registry that hides a catalog group the kernel needs.
    `settings` are the kernel's own. The world source is not read: the kernels
    take their prior
    bounds from the study's belief population at refit time.
    """
    arm = _arm(arm)
    return ReportRefit(
        kernel=arm.kernel, truth_groups=arm.groups, caps=study.caps, **settings
    )


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
    by charged days, the kickoff count and a stop flag. `refits_reach_policy` is
    set by `bracket_study` from the arm's report audience.
    """

    budget_d: float = eqx.field(static=True)
    n_slots: int = eqx.field(static=True)
    max_visits: int = eqx.field(default=24, static=True)
    max_search: int = eqx.field(default=8, static=True)
    refits_reach_policy: bool = eqx.field(default=True, static=True)
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
        if not self.refits_reach_policy:
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
        weight = jnp.maximum(cost[:n_detect], 1e-9)

        lod = model.assets().report.lod_arcsec
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
        channel = jnp.where(do_follow, fu_mode[fs], search_mode)
        act = do_follow | do_search
        # Budget: a choice that does not fit falls back to the cheapest visit; when not
        # even that fits, one attempt is made and the policy stops.
        fits = cost[channel] <= remaining
        cheapest_fits = cost[0] <= remaining
        channel = jnp.where(fits, channel, 0)
        attempt_last = act & ~fits & ~cheapest_fits
        use = act & ~jnp.asarray(stopped > 0.5)
        done = (tick >= horizon) | (stopped > 0.5)

        observe_cat = kicking
        kind = jnp.select(
            [done, observe_cat, stale & ~kicking, use],
            [EffectKind.STOP, EffectKind.OBSERVE, EffectKind.WAIT, EffectKind.OBSERVE],
            EffectKind.WAIT,
        )
        cat_channel = caps.n_channels - 1
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
        "status",
    )
)


class SummaryObserver(AbstractObserver):
    """A `DecisionSummary` per star, set only by `install`.

    `prior` is the first-tier summary of each star with no data; it is what the
    observer and the policy hold until the first refit, and it is built from the
    belief's own prior, never from the truth. The study needs `every_ticks > 0` and
    `rows_per_subject > 0`, as every non-incremental observer does.
    """

    subject_leaves = SUMMARY_LEAVES
    prior: DecisionSummary
    claims: tuple[AbstractClaim, ...] = eqx.field(static=True, default=())
    catalog_kind: str = eqx.field(default=CATALOG_KIND_NAME, static=True)
    report_kind: str = eqx.field(default=REPORT_KIND_NAME, static=True)
    block_kind: str = eqx.field(default=BLOCK_KIND_NAME, static=True)
    ticks_per_day: int = eqx.field(default=TICKS_PER_DAY, static=True)

    def capabilities(self):
        """Refit only, from visit reports, the truth catalog and block outcomes."""
        return ObserverCapabilities(
            incremental=False,
            refit=True,
            refit_kinds=(self.report_kind, self.catalog_kind, self.block_kind),
        )

    def init(self, caps):
        """The prior summary of every star."""
        return self.prior

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
        return self.prior

    def layout(self, caps):
        """One type for belief and summary; `n_obs` counts the visits read."""
        shapes = self.prior
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


# -- Outcomes and paired arms ----------------------------------------------------


def candidate_slots(belief, lod_arcsec):
    """The imaging candidates a summary nominates: ``bool[n_stars, n_slots]``.

    A slot is a candidate when it holds at least `CANDIDATE_DETECTIONS` detections,
    a slot status of `SLOT_OK` (a fit
    above the size floor, never a degraded, unsupported or failed one) and a
    predicted-position scatter under one lambda over D (`lod_arcsec`, in arcsec like
    the summary's `scatter`). The policy reads its own summary through this rule
    and the outcome table reads the observer's.
    """
    return (
        (belief.n_det >= CANDIDATE_DETECTIONS)
        & (belief.status == SLOT_OK)
        & (belief.scatter < CANDIDATE_SCATTER_LOD * lod_arcsec)
    )


@dataclasses.dataclass(frozen=True)
class OutcomeTable:
    """What every arm reports, per slot, per star and per world.

    Each field is a dict of NumPy arrays with a leading world axis: `slot` leaves are
    ``[W, n_stars, n_slots]``, `star` leaves ``[W, n_stars]`` and `world` leaves
    ``[W]``.

    Slot leaves: `present`, `real`, `n_det` (visits that detected it), `detected`,
    `detected_again` and `candidate` (the observer's final summary meets
    `candidate_slots`).

    Star leaves: `visited`, `n_visits`, `n_search`, `n_followup`, `t_search` and
    `t_followup` (telescope days), and `dropped` (planets the slots could not hold).

    World leaves: `charged` (telescope days spent), `low_ess` (slots whose last
    refit was degraded, `SLOT_DEGRADED`) and `no_funds` (admissions refused for lack
    of budget).
    """

    slot: dict
    star: dict
    world: dict

    @property
    def n_worlds(self):
        """Worlds in the table."""
        return int(next(iter(self.world.values())).shape[0])

    def to_csv(self, directory, name):
        """Write `<name>_slot.csv`, `<name>_star.csv` and `<name>_world.csv`."""
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for level, leaves in (("slot", self.slot), ("star", self.star)):
            shape = next(iter(leaves.values())).shape
            index = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"))
            labels = ["world", "star", "slot"][: len(shape)]
            frame = {k: v.reshape(-1) for k, v in zip(labels, index, strict=False)}
            frame.update({k: v.reshape(-1) for k, v in leaves.items()})
            _write_columns(directory / f"{name}_{level}.csv", frame)
        frame = {"world": np.arange(self.n_worlds), **self.world}
        _write_columns(directory / f"{name}_world.csv", frame)


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


def outcome_table(
    result,
    truths,
    study,
    *,
    dropped=None,
    check_detection=False,
):
    """The `OutcomeTable` of a `run_worlds` result.

    Args:
        result: The `RunResult` of the batch.
        truths: The `PopulationTruth` batch the run used (`source.take(...)`).
        study: The `Study` that ran.
        dropped: Optional ``[W, n_stars]`` planets the slots could not hold (the
            source's `dropped` rows of those worlds); zeros when not given.
        check_detection: Also fill the slot leaves `det_expected` and `det_variance`:
            the sum over the slot's visits of the channel's detection probability
            at the true elements, and of its Bernoulli variance. It makes one
            channel call per visit, so it is off by default.

    A detection-mode visit is a follow-up when an earlier visit of its star (in
    completion order) detected a slot, a search otherwise.
    """
    carry, trace = result.carry, result.trace
    present = np.asarray(truths.present)
    channel = study.channel_model
    claim_values = claims_on_truth(
        truths,
        study.observer.claims,
        channel.scenes.star_mass_kg / Msun2kg,
        jnp.asarray(channel.scenes.luminosity_lsun),
        channel.epoch_jd,
    )
    real = present & np.asarray(claim_values.all(-1))
    n_w, n, n_slots = present.shape
    log = {
        name: np.asarray(getattr(carry.log, name))
        for name in (
            "kind",
            "subject",
            "valid",
            "payload",
            "start_tick",
            "end_tick",
            "channel_index",
        )
    }
    vis = visit_columns(n_slots)
    det_cols = [column_index(vis, f"detected_{k}") for k in range(n_slots)]
    time_col = column_index(vis, "time_d")
    n_det = np.zeros((n_w, n, n_slots), np.int64)
    names = ("n_visits", "n_search", "n_followup")
    star = {k: np.zeros((n_w, n)) for k in names}
    for k in ("t_search", "t_followup"):
        star[k] = np.zeros((n_w, n))
    expected = np.zeros((n_w, n, n_slots))
    variance = np.zeros((n_w, n, n_slots))
    channel = study.channel_model
    per_day = round(SECONDS_PER_DAY / float(study.clock.tick_length_s))
    for w in range(n_w):
        seen = np.zeros(n, bool)
        if check_detection:
            truth = jax.tree_util.tree_map(lambda x, w=w: x[w], truths)
            true_elements = channel.truth_parameters(truth)
        for r in _rows_of(log, w, KIND_VISIT):
            s = int(log["subject"][w, r])
            days = (log["end_tick"][w, r] - log["start_tick"][w, r]) / per_day
            hit = log["payload"][w, r, det_cols] > 0.5
            star["n_visits"][w, s] += 1
            stage = "followup" if seen[s] else "search"
            star[f"t_{stage}"][w, s] += days
            star[f"n_{stage}"][w, s] += 1
            if check_detection:
                mode = int(log["channel_index"][w, r])
                when = log["payload"][w, r, time_col] - float(channel.epoch_jd)
                p = np.asarray(
                    channel.detect_probability(
                        s, mode, when, jax.tree.map(lambda x, s=s: x[s], true_elements)
                    )
                )
                expected[w, s] += p * present[w, s]
                variance[w, s] += p * (1 - p) * present[w, s]
            n_det[w, s] += hit
            seen[s] |= bool(hit.any())
    belief = carry.belief
    lod = float(study.channel_model.assets().report.lod_arcsec)
    candidate = np.asarray(candidate_slots(belief, lod))
    star["visited"] = (star["n_visits"] > 0).astype(np.float64)
    star["dropped"] = (
        np.zeros((n_w, n)) if dropped is None else np.asarray(dropped, np.float64)
    )
    budget = float(np.asarray(study.budget)[-1, SHARED_RESOURCE])
    balance = np.asarray(carry.ledger.balance)[:, -1, SHARED_RESOURCE]
    world = {
        "charged": budget - balance,
        "low_ess": (np.asarray(belief.status) == SLOT_DEGRADED).sum(axis=(1, 2)),
        "no_funds": (
            np.asarray(trace.admit_code).reshape(n_w, -1) == int(AdmissionCode.NO_FUNDS)
        ).sum(axis=1),
    }
    slot = {
        "present": present,
        "real": real,
        "n_det": n_det,
        "detected": n_det > 0,
        "detected_again": n_det > 1,
        "candidate": candidate,
    }
    if check_detection:
        slot["det_expected"] = expected
        slot["det_variance"] = variance
    return OutcomeTable(
        slot=slot,
        star=star,
        world={k: np.asarray(v, np.float64) for k, v in world.items()},
    )


@dataclasses.dataclass(frozen=True)
class ArmsResult:
    """The outcome tables of the arms of one paired run, and their paired contrasts.

    Every arm ran the same truths and roots, so a contrast is a paired difference per
    world and its standard error is that of the mean of the differences.
    """

    table: dict

    def _endpoint(self, arm, endpoint):
        return np.asarray(self.table[arm].world[endpoint], np.float64)

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
        """Write each arm's three tables under `directory`, named by arm."""
        for name, table in self.table.items():
            table.to_csv(directory, name.replace("+", "_"))


def run_arms(
    arms,
    truths,
    roots,
    n_steps,
    *,
    executor=None,
    dropped=None,
    check_detection=False,
):
    """Run every arm on the same truths and roots; return their `ArmsResult`.

    Args:
        arms: A dict of name to ``(Study, boundary)``.
        truths: The `PopulationTruth` batch, shared by every arm.
        roots: One root key per world, shared by every arm.
        n_steps: Step budget of each run.
        executor: A `CompiledExecutor`; each arm runs with its own boundary through
            it. Defaults to `BoundExecutor`.
        dropped: Passed to `outcome_table`.
        check_detection: Passed to `outcome_table`.
    """
    tables = {}
    for name, (study, boundary) in arms.items():
        base = BoundExecutor() if executor is None else executor
        run = base.run(study, truths, roots, n_steps, refit=boundary)
        tables[name] = outcome_table(
            run,
            truths,
            study,
            dropped=dropped,
            check_detection=check_detection,
        )
    return ArmsResult(table=tables)
