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
- `BracketPolicy` is the one decision rule every arm runs: block, follow-up, search.
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

from spaceodyssey.adapters.jaxedith import (
    block_columns,
    block_outcome_kind,
    column_index,
    truth_catalog_kind,
    visit_columns,
    visit_report_kind,
)
from spaceodyssey.adapters.photomancy import (
    BLOCK_KIND_NAME,
    CATALOG_KIND_NAME,
    CLASS_ONE,
    GRID_SPACING_D,
    N_GRID,
    POSITION_LAG_D,
    REPORT_KIND_NAME,
    REVEAL_OFFSET,
    SLOT_DEGRADED,
    SLOT_KEPT,
    SLOT_OK,
    SUMMARY_LEAVES,
    DecisionSummary,
    ReportRefit,
    population_ln_lam_range,
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
from spaceodyssey.core.study import Clock
from spaceodyssey.studies import toy

# A candidate: three detections, a class probability of at least `CANDIDATE_CLASS`
# (a study parameter), a refit that completed, and a predicted-position scatter under
# one lambda over D.
CANDIDATE_DETECTIONS, CANDIDATE_CLASS, CANDIDATE_SCATTER_LOD = 3, 0.2, 1.0
# Kind indices of `bracket_registry`.
KIND_NONE, KIND_PRODUCT, KIND_VISIT, KIND_BLOCK, KIND_CATALOG = range(5)
# The telescope and the catalog lanes, and the shared budget resource column.
LANE_TELESCOPE, LANE_CATALOG = 0, 1
SHARED_RESOURCE = 2
# Channel ids of the modes start here; the refresh product channels are 90 and 91.
FIRST_CHANNEL_ID = 10
# Characterization knobs of `BracketPolicy`: the probability a block needs to start (the
# first-summary channel's, and the survey channel's, which is the scheduler's 0.95), the
# consecutive failed blocks that retire a slot, and the time held back per remaining
# detection or characterization of an open track on a survey channel (days).
BLOCK_P, SURVEY_BLOCK_P, RETIRE_AFTER, SURVEY_RESERVE_D = 0.8, 0.95, 3, 15.0
# Per-slot block state in `pstate`: attempts started, consecutive failures settled,
# spectra seen when the last attempt started, mode of the last attempt, its ordinal on
# the star, and the first mode a retry may use.
SLOT_FIELDS = 6
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


def bracket_registry(arm, n_slots=4):
    """The kind registry of `arm`: none, refresh product, visit, block, catalog."""
    arm = _arm(arm)
    return KindRegistry(
        (
            RecordKind("none", ()),
            REFRESH_PRODUCT_KIND,
            visit_report_kind(n_slots, arm.report_audience),
            block_outcome_kind(n_slots),
            truth_catalog_kind(n_slots, arm.groups),
        )
    )


def block_log_len(n_slots, n_char=1, retire_after=RETIRE_AFTER):
    """Blocks one star can start: a slot starts at most `retire_after * n_char`."""
    return n_slots * n_char * retire_after


def pstate_size(n_subjects, n_slots, n_char=1, retire_after=RETIRE_AFTER):
    """Length of the `BracketPolicy` state vector.

    Per star: last visit tick, visits, search visits and a first-block ordinal per slot;
    after the stars, the charged days, the kickoff count and a stop flag; then the
    per-slot block state (`SLOT_FIELDS`), the blocks started per star and each star's
    log of the slot of every block it started, in order.
    """
    log = block_log_len(n_slots, n_char, retire_after)
    per_star = SLOT_FIELDS * n_slots + 1 + log
    return n_subjects * (3 + n_slots) + 3 + n_subjects * per_star


# Days between forecast renewals of a summary, and the per-star rows a long study holds.
RENEWAL_D = GRID_SPACING_D * (N_GRID - 1)
# Rows a star typically adds to the log over a study: visits and blocks. The rest are
# refit products, two per renewal and two per visit or block refit.
VISITS_PER_STAR, BLOCKS_PER_STAR = 4, 1
MAX_VISITS = 24


def default_capacities(
    n_subjects,
    n_slots,
    *,
    n_channels,
    horizon_d=None,
    n_char=1,
    retire_after=RETIRE_AFTER,
    max_visits=MAX_VISITS,
):
    """Capacities of the bracket study of `n_subjects` stars.

    `n_pstate` holds the policy state (`pstate_size`). Without `horizon_d` the log,
    counting-row and id capacities are the fixed small-study values (`n_log` 1024,
    `rows_per_subject` 32, `max_sid` 4096). With it they scale with the study:

    - `n_log` holds the catalog, visit and block rows of every star plus the refit
      products (two per renewal per star, one renewal per `RENEWAL_D`, plus two per
      visit or block refit), with a quarter spare, in steps of 256.
    - `rows_per_subject` holds the catalog row, `max_visits` visits and the blocks a
      star can start in that time (at most `retire_after * n_char` per slot, and a few
      per quarter), plus three spare.
    - `max_sid` is the next power of two above two ids per log row, one per day for the
      policy's waits and eight per star; it must leave `max_tick` above the horizon.

    The small studies keep their values: a 120-day, two-star study needs no more than
    the fixed ones.
    """
    n_pstate = pstate_size(n_subjects, n_slots, n_char, retire_after)
    n_log, rows, max_sid = 1024, 32, 4096
    if horizon_d is not None:
        renewals = math.ceil(horizon_d / RENEWAL_D) + 1
        per_star = 1 + VISITS_PER_STAR + BLOCKS_PER_STAR
        refits = renewals + VISITS_PER_STAR + BLOCKS_PER_STAR
        need = n_subjects * (per_star + 2 * refits)
        n_log = max(n_log, 256 * math.ceil(1.25 * need / 256))
        blocks = min(
            block_log_len(n_slots, n_char, retire_after),
            2 + math.ceil(horizon_d / 90.0),
        )
        rows = max(rows, 1 + max_visits + blocks + 3)
        ids = 2 * n_log + 2 * math.ceil(horizon_d) + 8 * n_subjects
        max_sid = max(max_sid, 1 << (ids - 1).bit_length())
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
    if horizon_d is not None and caps.max_tick < horizon_d * TICKS_PER_DAY:
        raise ValueError(
            f"{n_subjects} stars over {horizon_d} d need max_sid {max_sid}, which"
            f" leaves max_tick {caps.max_tick} below the horizon"
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
):
    """The `Study` of `arm`: the stars of `source` on one telescope and one budget.

    Args:
        source: The world source (stored universes), whose stars are the subjects.
        channel: The `ObservingModeChannel` of those stars.
        arm: An arm name of `ARMS`, or an `Arm`.
        horizon_d: Study length in days.
        budget_d: Telescope days on the shared row.
        caps: Capacities; defaults to `default_capacities` for this horizon, the
            policy's `n_char` and its visit limit.
        observer: The summary observer. It must be non-incremental and refit the
            visit and catalog kinds.
        policy: The decision policy.
    """
    arm = _arm(arm)
    if observer is None or policy is None:
        raise ValueError("bracket_study needs an observer and a policy")
    if isinstance(policy, BracketPolicy):
        reaches = bool(arm.report_audience & Audience.POLICY)
        policy = dataclasses.replace(policy, refits_reach_policy=reaches)
    built = channel.assets()
    n_stars = int(channel.catalog_mass_kg.shape[0])
    if len(source.stars) != n_stars:
        raise ValueError("the world source and the channel disagree on the stars")
    shape = {}
    if isinstance(policy, BracketPolicy):
        shape = {"n_char": policy.n_char, "retire_after": policy.retire_after}
    caps = caps or default_capacities(
        n_stars,
        channel.n_slots,
        n_channels=channel.n_channels,
        horizon_d=horizon_d,
        max_visits=getattr(policy, "max_visits", MAX_VISITS),
        **shape,
    )
    if isinstance(policy, BracketPolicy):
        need = pstate_size(n_stars, channel.n_slots, **shape)
        if caps.n_pstate < need:
            raise ValueError(
                f"caps.n_pstate is {caps.n_pstate}; the policy needs {need}"
            )
    caps = bracket_registry(arm, channel.n_slots).lower(caps, observer)
    per_day = int(built.context.clock.ticks_per_day)
    horizon = round(horizon_d * per_day)
    budget = jnp.full((caps.n_balance_rows, caps.n_resources), UNBOUNDED, F64)
    budget = budget.at[-1, SHARED_RESOURCE].set(float(budget_d))
    schedule = RefreshSchedule(per_day, 0, *toy.PRODUCT_CHANNELS, KIND_PRODUCT)
    study = toy.survey_1(
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
    return dataclasses.replace(study, clock=Clock(0.001), world_source=source)


def bracket_boundary(study, arm, **settings):
    """The refit boundary of `arm` for `study` (`ReportRefit` with the arm's kernel).

    Construction refuses a registry that hides a catalog group the kernel needs.
    `settings` are the kernel's own (the posterior and shrinkage kernels need
    `occupancy`; their scale prior `ln_lam_range` is derived from the world source's
    population when not given).
    """
    arm = _arm(arm)
    if arm.kernel != "truth" and "ln_lam_range" not in settings:
        population = getattr(study.world_source, "population", None)
        if population is not None:
            settings["ln_lam_range"] = population_ln_lam_range(population)
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
    """The bracket policy: one rule over the summary, read through any arm's belief.

    The rule is the same in every arm; the arms differ only in what the summary
    holds. One effect per decision.

    State in `pstate` (see `pstate_size`): per star the last visit tick, visits, search
    visits and the ordinal of the first block of each slot; after the stars, the
    charged days, the kickoff count and a stop flag; then, per slot, the block state
    (attempts, consecutive failures, spectra seen, last mode, last ordinal, mode floor),
    per star the blocks started and the slot of each block in order.

    Kickoff observes the catalog mode of one star per decision; the release of its row
    wakes the policy for the next. After a visit the policy waits for the next refresh
    before deciding, then in order:

    1. Block (characterize): a candidate (`candidate_slots`) whose class is not yet
       revealed, or is class 1 with fewer than `n_char` spectra, that is not retired
       and has no block awaiting its refit starts one at its own shortest block mode
       whose `p_block` at the current node is `block_p` or more and is the largest over
       the slot's nodes for that mode (modes are ordered by length). A block that fell
       short (the refit that reads it reveals nothing new) is retried at a longer mode
       than the failed one; `retire_after` consecutive failures retire the slot. A
       candidate no mode serves is not blocked now; it is tried again when a refit
       moves its forecast.
    2. Follow-up: a slot the belief knows (a detection, or class probability at least
       0.5), not a candidate, not revealed, with class probability at least 0.05, is
       observed at the node and mode of largest `p_det` per charged day at least 15
       days after the star's last visit, when that node is the current one and `p_det`
       is `followup_p` or more.
    3. Search: the star not visited in the last 15 days, with fewer than 8 search
       visits, of largest `p_new` per charged day, when `p_new` is 0.02 or more. With a
       reserve (`reserve_d`), a search starts only if the time left covers it plus what
       every open track still needs: for each candidate-class slot with a detection and
       no revealed class or retirement, its detections short of `CANDIDATE_DETECTIONS`
       and its spectra short of `n_char`, each at `reserve_d` plus the mode's overhead.
    4. Otherwise wait a day.

    The wait is for refits to read the report visits (search and follow-up); the
    visits that start a block are not counted, since a refit never reads them.

    No star gets more than 24 visits. A choice that does not fit the budget falls back
    to the cheapest visit; when not even that fits the policy attempts one more visit
    (the refusal is the budget binding) and stops, else it stops at the horizon.

    Attributes:
        candidate_class: The class probability a slot needs to be a candidate.
        budget_d: Telescope days the policy may spend.
        n_slots: Planet slots per star.
        max_visits, max_search: Per-star visit and search limits.
        refits_reach_policy: Whether the policy's summary is ever refit. When false
            (an arm whose report columns the policy may not read) the summary never
            changes, so the policy never waits for a refit to read its visits.
            `bracket_study` sets it from the arm.
        n_char: Spectra a slot needs to count as characterized; the first reveals the
            class.
        retire_after: Consecutive failed blocks that retire a slot (one reproduces a
            single block per slot).
        block_p: Probability a block needs to start; `None` is 0.8 on a first-summary
            channel and 0.95 (the survey scheduler's) on a survey channel.
        followup_p: Detection probability a follow-up needs to start.
        reserve_d: Heuristic days per remaining detection or characterization held
            back before a blind search; `None` is none on a first-summary channel and
            15 (the survey scheduler's) on a survey channel, and zero turns it off.
    """

    budget_d: float = eqx.field(static=True)
    n_slots: int = eqx.field(static=True)
    max_visits: int = eqx.field(default=24, static=True)
    max_search: int = eqx.field(default=8, static=True)
    refits_reach_policy: bool = eqx.field(default=True, static=True)
    candidate_class: float = eqx.field(default=CANDIDATE_CLASS, static=True)
    n_char: int = eqx.field(default=1, static=True)
    retire_after: int = eqx.field(default=RETIRE_AFTER, static=True)
    block_p: float | None = eqx.field(default=None, static=True)
    followup_p: float = eqx.field(default=0.5, static=True)
    reserve_d: float | None = eqx.field(default=None, static=True)

    def declare(self):
        """The campaign drain the policy ends with."""
        return Declaration(stops=((Scope.CAMPAIGN, Disposition.DRAIN),))

    def decide(self, pstate, view, study, key):
        """One decision: kickoff, wait for a fresh summary, or act on it."""
        caps, ch = study.caps, study.channels
        model = study.channel_model
        n, slots = caps.n_subjects, self.n_slots
        stride = 3 + slots
        per_day = jnp.asarray(86_400_000, I64)
        stars = pstate[: n * stride].reshape(n, stride)
        last_tick, n_vis, n_search = stars[:, 0], stars[:, 1], stars[:, 2]
        charged, kick, stopped = pstate[n * stride : n * stride + 3]
        log_len = block_log_len(slots, self.n_char, self.retire_after)
        ext = n * stride + 3
        mid = ext + n * slots * SLOT_FIELDS
        slot_state = pstate[ext:mid].reshape(n, slots, SLOT_FIELDS)
        n_blk = pstate[mid : mid + n]
        att, cons, seen, last_mode, last_ord, floor = (
            slot_state[..., i] for i in range(SLOT_FIELDS)
        )
        belief = view.belief
        survey = bool(model.survey)
        block_p = self.block_p
        if block_p is None:
            block_p = SURVEY_BLOCK_P if survey else BLOCK_P
        reserve_d = self.reserve_d
        if reserve_d is None:
            reserve_d = SURVEY_RESERVE_D if survey else 0.0
        # the mode counts are the summary's own, so any channel's grids fit
        n_detect, n_block = belief.p_det.shape[-1], belief.p_block.shape[-1]
        cost = ch.cost[:, SHARED_RESOURCE]
        remaining = self.budget_d - charged
        tick = view.tick
        horizon = study.horizon_tick

        # Kickoff: the catalog row of star `kick`.
        kicking = kick < n
        # Fresh: every report visit made has been read by a refit. A block produces
        # a block outcome, not a report row, so it is not a visit a refit reads.
        stale = jnp.any(n_vis - n_blk > belief.n_obs.astype(F64))
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

        # Where each slot's characterization stands. The latest attempt is settled once
        # a refit has read its block row: it succeeded when the slot's spectra grew.
        n_read = belief.n_blocks.astype(F64)[:, None]
        spectra = belief.n_spectra.astype(F64)
        settled = (att > 0.5) & (last_ord <= n_read)
        won = spectra > seen
        eff_cons = jnp.where(settled, jnp.where(won, 0.0, cons + 1.0), cons)
        eff_floor = jnp.where(settled, jnp.where(won, 0.0, last_mode + 1.0), floor)
        pending = (att > 0.5) & (last_ord > n_read)
        retired = eff_cons >= self.retire_after - 0.5
        capped = att >= block_log_len(1, self.n_char, self.retire_after) - 0.5
        char_open = belief.revealed == 0
        if self.n_char > 1:
            char_open |= (belief.revealed == CLASS_ONE + REVEAL_OFFSET) & (
                belief.n_spectra < self.n_char
            )
        open_track = char_open & ~retired & ~capped
        tryable = open_track & ~pending

        # 1. Block.
        lod = model.assets().report.lod_arcsec
        candidate = candidate_slots(belief, lod, self.candidate_class)
        want_block = candidate & tryable
        pb_now = belief.p_block[s_ix, :, g_cur, :]
        best_pb = belief.p_block.max(axis=2)
        longer = jnp.arange(n_block)[None, None, :] >= eff_floor[:, :, None] - 0.5
        mode_ok = (pb_now >= block_p) & (pb_now >= best_pb) & longer
        block_ok = want_block & mode_ok.any(axis=-1)
        block_mode = jnp.argmax(mode_ok, axis=-1)
        block_score = jnp.where(block_ok, 1.0, -1.0)
        bs = jnp.argmax(block_score.reshape(-1))
        block_star, block_slot = bs // slots, bs % slots
        do_block = block_score.reshape(-1)[bs] > 0
        block_channel = n_detect + block_mode[block_star, block_slot]

        # 2. Follow-up.
        known = (belief.n_det >= 1) | (belief.p_class >= 0.5)
        eligible = known & ~candidate & (belief.revealed == 0)
        eligible &= belief.p_class >= 0.05
        late = node_ticks >= (last_tick + 15 * per_day)[:, None]
        pd_val = belief.p_det / weight[None, None, None, :]
        pd_val = jnp.where(
            late[:, None, :, None] & eligible[:, :, None, None], pd_val, -1.0
        )
        flat = pd_val.reshape(n, -1)
        fu_best = flat.argmax(axis=1)
        fu_score = flat.max(axis=1)
        fu_slot = fu_best // (N_NODES * n_detect)
        fu_node = (fu_best // n_detect) % N_NODES
        fu_mode = fu_best % n_detect
        fu_p = belief.p_det[s_ix, fu_slot, fu_node, fu_mode]
        fu_ok = (
            (fu_score > 0)
            & (fu_node == g_cur)
            & (fu_p >= self.followup_p)
            & (n_vis < self.max_visits)
        )
        fs = jnp.argmax(jnp.where(fu_ok, fu_score, -1.0))
        do_follow = fu_ok[fs]

        # 3. Search.
        rested = (tick - last_tick) >= 15 * per_day
        sp = belief.p_new[s_ix, g_cur, :]
        sp_val = sp / weight[None, :]
        sp_ok = (
            (rested | (n_vis == 0))[:, None]
            & (n_search < self.max_search)[:, None]
            & (n_vis < self.max_visits)[:, None]
            & (sp >= 0.02)
        )
        if reserve_d > 0:
            reserve = self._reserve(
                belief, model, cost, n_detect, open_track, reserve_d
            )
            sp_ok &= (cost[None, :n_detect] + reserve) <= remaining
        sp_flat = jnp.where(sp_ok, sp_val, -1.0).reshape(-1)
        sb = jnp.argmax(sp_flat)
        do_search = sp_flat[sb] > 0
        search_star, search_mode = sb // n_detect, sb % n_detect

        star = jnp.where(do_block, block_star, jnp.where(do_follow, fs, search_star))
        channel = jnp.where(
            do_block, block_channel, jnp.where(do_follow, fu_mode[fs], search_mode)
        )
        act = do_block | do_follow | do_search
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
        cat_channel = n_detect + n_block
        eff_star = jnp.where(observe_cat, jnp.minimum(kick.astype(I64), n - 1), star)
        eff_channel = jnp.where(observe_cat, cat_channel, channel)
        wake = jnp.where(stale & ~kicking, next_day, tick + per_day)
        eff = toy._effects(
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
        # A block that did not fit ran as the cheapest visit: it starts no block.
        is_block = observing & do_block & fits
        row = star
        is_search = observing & ~is_block & ~do_follow
        new = pstate
        new = new.at[row * stride].set(
            jnp.where(observing, tick.astype(F64), pstate[row * stride])
        )
        new = new.at[row * stride + 1].add(observing.astype(F64))
        new = new.at[row * stride + 2].add(is_search.astype(F64))
        # block bookkeeping: the star's count and log, the slot's state, first ordinal
        ordinal = n_blk[row] + 1.0
        new = new.at[mid + row].add(is_block.astype(F64))
        log_ix = (
            mid + n + row * log_len + jnp.minimum(n_blk[row].astype(I64), log_len - 1)
        )
        new = new.at[log_ix].set(jnp.where(is_block, block_slot + 1.0, new[log_ix]))
        state_ix = ext + (row * slots + block_slot) * SLOT_FIELDS
        mode = block_mode[block_star, block_slot].astype(F64)
        entry = jnp.stack(
            [
                att[row, block_slot] + 1.0,
                eff_cons[row, block_slot],
                spectra[row, block_slot],
                mode,
                ordinal,
                eff_floor[row, block_slot],
            ]
        )
        new = new.at[state_ix + jnp.arange(SLOT_FIELDS)].set(
            jnp.where(is_block, entry, slot_state[row, block_slot])
        )
        slot_ix = row * stride + 3 + block_slot
        first = is_block & (att[row, block_slot] < 0.5)
        new = new.at[slot_ix].set(jnp.where(first, ordinal, pstate[slot_ix]))
        base = n * stride
        spent = jnp.where(observing & (cost[channel] <= remaining), cost[channel], 0.0)
        new = new.at[base].add(spent)
        new = new.at[base + 1].add(observe_cat.astype(F64))
        new = new.at[base + 2].set(
            jnp.maximum(pstate[base + 2], (observing & attempt_last).astype(F64))
        )
        return eff, new, jnp.asarray(0.0, F64)

    def _reserve(self, belief, model, cost, n_detect, open_track, heuristic_d):
        """Telescope days the open tracks still need, at `heuristic_d` plus overhead.

        A track is a slot of the candidate class with a detection whose characterization
        is still open (`open_track`: not revealed against, retired or out of attempts).
        A block awaiting its refit still owes its spectrum.
        """
        exposure_d = jnp.asarray(model.exposures_s[0], F64) / 86_400.0
        char_d = jnp.asarray(model.block_days[0], F64)
        over_det = cost[0] - exposure_d
        over_char = cost[n_detect] - char_d
        track = (belief.n_det >= 1) & (belief.p_class >= self.candidate_class)
        track &= open_track
        det_left = jnp.clip(
            CANDIDATE_DETECTIONS - belief.n_det, 0, CANDIDATE_DETECTIONS
        )
        char_left = jnp.clip(self.n_char - belief.n_spectra, 0, self.n_char)
        need = det_left * (heuristic_d + over_det)
        need += char_left * (heuristic_d + over_char)
        return jnp.sum(jnp.where(track, need, 0.0))


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
        "p_new",
        "status",
        "revealed",
        "n_spectra",
        "n_blocks",
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
    catalog_kind: str = eqx.field(default=CATALOG_KIND_NAME, static=True)
    report_kind: str = eqx.field(default=REPORT_KIND_NAME, static=True)
    block_kind: str = eqx.field(default=BLOCK_KIND_NAME, static=True)

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
        elapsed = (tick - summary.ref_tick) / (NODE_SPACING_D * TICKS_PER_DAY)
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
        the slots that fit are installed. `p_new` is the first tier's: it keeps its
        previous value when an undetected slot (no detection in `fitted`) failed.
        """
        status = fitted.status[subject]
        kept = jnp.isin(status, jnp.asarray(SLOT_KEPT, I64))
        dark_failed = jnp.any(kept & (fitted.n_det[subject] == 0))

        def merged(name):
            new, old = getattr(fitted, name)[subject], getattr(belief, name)[subject]
            if name == "p_new":
                keep = dark_failed
            elif name in ("revealed", "n_spectra"):
                return new
            elif name in _SLOT_LEAVES:
                keep = kept.reshape(kept.shape + (1,) * (new.ndim - 1))
            else:
                return new
            out = jnp.where(keep, old, new)
            if name == "p_class":
                # a revealed slot is decided: 1 for class 1, 0 for any other class
                code = fitted.revealed[subject]
                decided = (code == CLASS_ONE + REVEAL_OFFSET).astype(out.dtype)
                out = jnp.where(code > 0, decided, out)
            return out

        return eqx.tree_at(
            lambda b: tuple(getattr(b, n) for n in SUMMARY_LEAVES),
            belief,
            tuple(
                getattr(belief, n).at[subject].set(merged(n)) for n in SUMMARY_LEAVES
            ),
        )


# -- Outcomes and paired arms ----------------------------------------------------

# The study clock: one tick is a millisecond.
TICKS_PER_DAY = 86_400_000


def candidate_slots(belief, lod_arcsec, candidate_class=CANDIDATE_CLASS):
    """The slots a summary nominates for a block: ``bool[n_stars, n_slots]``.

    A slot is a candidate when it holds at least `CANDIDATE_DETECTIONS` detections, a
    class probability of at least `candidate_class`, a slot status of `SLOT_OK` (a fit
    above the size floor, never a degraded, unsupported or failed one) and a
    predicted-position scatter under one lambda over D (`lod_arcsec`, in arcsec like
    the summary's `scatter`). Imaging nominates; only a delivered block decides the
    class (`confirmed_slots`). The policy reads its own summary through this rule and
    the outcome table reads the observer's, so the two cannot drift apart.
    """
    return (
        (belief.n_det >= CANDIDATE_DETECTIONS)
        & (belief.p_class >= candidate_class)
        & (belief.status == SLOT_OK)
        & (belief.scatter < CANDIDATE_SCATTER_LOD * lod_arcsec)
    )


def confirmed_slots(belief, n_char=1):
    """The slots characterized as class 1: ``bool[n_stars, n_slots]``.

    A delivered block reveals the class; the slot is confirmed once `n_char` delivered
    blocks (spectra) have been read. At `n_char = 1` the reveal alone confirms it.
    """
    out = belief.revealed == CLASS_ONE + REVEAL_OFFSET
    if n_char > 1:
        out = out & (belief.n_spectra >= n_char)
    return out


@dataclasses.dataclass(frozen=True)
class OutcomeTable:
    """What every arm reports, per slot, per star and per world.

    Each field is a dict of NumPy arrays with a leading world axis: `slot` leaves are
    ``[W, n_stars, n_slots]``, `star` leaves ``[W, n_stars]`` and `world` leaves
    ``[W]``.

    Slot leaves: `present`, `klass`, `n_det` (visits that detected it), `detected`,
    `detected_again`, `candidate` (the observer's final summary meets
    `candidate_slots` at the table's `candidate_class`), `revealed` (the class a
    delivered block revealed in the observer's final summary, `-1` for none),
    `claimed` (the slot is confirmed: revealed class 1 with `n_char` spectra read),
    `claim_error_lod` (distance of the claimed position from the truth at the same
    epoch, in lambda over D, NaN when not claimed), `block_attempted`,
    `block_attempts` (blocks started on the slot), `n_spectra` (blocks that met the
    signal-to-noise requirement), `block_delivered` (a block on a present class-1 slot
    that met it), `block_snr` (the largest over the slot's blocks) and `block_days`
    (telescope days of the slots' blocks, NaN for none).

    Star leaves: `visited`, `n_visits`, `t_search`, `t_followup`, `t_block` (telescope
    days) and `dropped` (planets the slots could not hold).

    World leaves: `confirmed` (slots revealed class 1 in the observer's summary at the
    horizon), `false_claims` (claimed slots whose truth is not a present class-1
    planet; zero by construction, so a nonzero value flags a bug), `wasted_blocks`
    and `wasted_days` (blocks, and their telescope days, on slots whose truth is not
    a present class-1 planet, delivered or not), `delivered`, `charged` (telescope
    days spent), `low_ess` (slots whose last refit was degraded, `SLOT_DEGRADED`) and
    `no_funds` (admissions refused for lack of budget).
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
    candidate_class=CANDIDATE_CLASS,
    n_char=None,
):
    """The `OutcomeTable` of a `run_worlds` result.

    Args:
        result: The `RunResult` of the batch.
        truths: The `PopulationTruth` batch the run used (`source.take(...)`).
        study: The `Study` that ran.
        dropped: Optional ``[W, n_stars]`` planets the slots could not hold (the
            source's `dropped` rows of those worlds); zeros when not given.
        candidate_class: The class probability a slot needs to be a candidate; the
            policy's own value for a study run with another one.
        n_char: Spectra a slot needs to count as confirmed; the policy's own when
            `None`.
        check_detection: Also fill the slot leaves `det_expected` and `det_variance`:
            the sum over the slot's visits of the channel's detection probability
            at the true elements, and of its Bernoulli variance. It makes one
            channel call per visit, so it is off by default.

    A detection-mode visit is a follow-up when an earlier visit of its star (in
    completion order) detected a slot, a search otherwise; a block is the telescope
    time of a block mode. Block `k` of a star belongs to the slot the policy's state
    logged for it.
    """
    carry, trace = result.carry, result.trace
    present = np.asarray(truths.present)
    klass = np.asarray(truths.klass)
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
    vis, blk = visit_columns(n_slots), block_columns(n_slots)
    det_cols = [column_index(vis, f"detected_{k}") for k in range(n_slots)]
    time_col = column_index(vis, "time_d")
    snr_cols = [column_index(blk, f"block_snr_{k}") for k in range(n_slots)]
    n_det = np.zeros((n_w, n, n_slots), np.int64)
    last_det_jd = np.full((n_w, n, n_slots), np.nan)
    names = ("n_visits", "n_search", "n_followup", "n_block")
    star = {k: np.zeros((n_w, n)) for k in names}
    for k in ("t_search", "t_followup", "t_block"):
        star[k] = np.zeros((n_w, n))
    expected = np.zeros((n_w, n, n_slots))
    variance = np.zeros((n_w, n, n_slots))
    channel = study.channel_model
    block_snr = np.full((n_w, n, n_slots), np.nan)
    block_days = np.full((n_w, n, n_slots), np.nan)
    attempts = np.zeros((n_w, n, n_slots), np.int64)
    spectra = np.zeros((n_w, n, n_slots), np.int64)
    policy = study.agenda.policies[0]
    if n_char is None:
        n_char = getattr(policy, "n_char", 1)
    retire = getattr(policy, "retire_after", RETIRE_AFTER)
    pstate = np.asarray(carry.pstate)
    mid = n * (3 + n_slots) + 3 + n * n_slots * SLOT_FIELDS
    length = block_log_len(n_slots, getattr(policy, "n_char", 1), retire)
    block_log = pstate[:, mid + n : mid + n + n * length].reshape(n_w, n, length)
    reveal = list(channel.reveal_columns())
    for w in range(n_w):
        seen = np.zeros(n, bool)
        if check_detection:
            truth = jax.tree_util.tree_map(lambda x, w=w: x[w], truths)
            true_elements = channel.truth_elements(truth)
        for r in _rows_of(log, w, KIND_VISIT):
            s = int(log["subject"][w, r])
            days = (log["end_tick"][w, r] - log["start_tick"][w, r]) / TICKS_PER_DAY
            hit = log["payload"][w, r, det_cols] > 0.5
            star["n_visits"][w, s] += 1
            stage = "followup" if seen[s] else "search"
            star[f"t_{stage}"][w, s] += days
            star[f"n_{stage}"][w, s] += 1
            if check_detection:
                mode = int(log["channel_index"][w, r])
                when = log["payload"][w, r, time_col] - float(channel.epoch_jd)
                p = np.asarray(
                    channel.detect_probability(s, mode, when, true_elements[s])
                )
                expected[w, s] += p * present[w, s]
                variance[w, s] += p * (1 - p) * present[w, s]
            n_det[w, s] += hit
            last_det_jd[w, s, hit] = log["payload"][w, r, time_col]
            seen[s] |= bool(hit.any())
        count = np.zeros(n, np.int64)
        for r in _rows_of(log, w, KIND_BLOCK):
            s = int(log["subject"][w, r])
            count[s] += 1
            star["n_block"][w, s] += 1
            days = (log["end_tick"][w, r] - log["start_tick"][w, r]) / TICKS_PER_DAY
            star["t_block"][w, s] += days
            k = int(block_log[w, s, count[s] - 1]) - 1
            if k < 0:
                continue
            attempts[w, s, k] += 1
            spectra[w, s, k] += log["payload"][w, r, reveal[k]] > 0.5
            block_snr[w, s, k] = np.fmax(
                block_snr[w, s, k], log["payload"][w, r, snr_cols[k]]
            )
            block_days[w, s, k] = np.nan_to_num(block_days[w, s, k]) + days
    belief = carry.belief
    lod = float(study.channel_model.assets().report.lod_arcsec)
    candidate = np.asarray(candidate_slots(belief, lod, candidate_class))
    claimed = np.asarray(confirmed_slots(belief, n_char))
    code = np.asarray(belief.revealed)
    real = present & (klass == CLASS_ONE)
    attempted = attempts > 0
    delivered = attempted & (block_snr >= float(channel.block_snr_required)) & real
    error = _claim_error(study, truths, belief, claimed, last_det_jd)
    star["visited"] = (star["n_visits"] > 0).astype(np.float64)
    star["dropped"] = (
        np.zeros((n_w, n)) if dropped is None else np.asarray(dropped, np.float64)
    )
    budget = float(np.asarray(study.budget)[-1, SHARED_RESOURCE])
    balance = np.asarray(carry.ledger.balance)[:, -1, SHARED_RESOURCE]
    wasted = attempted & ~real
    world = {
        "confirmed": claimed.sum(axis=(1, 2)),
        "false_claims": (claimed & ~real).sum(axis=(1, 2)),
        "wasted_blocks": wasted.sum(axis=(1, 2)),
        "wasted_days": np.where(wasted, block_days, 0.0).sum(axis=(1, 2)),
        "delivered": delivered.sum(axis=(1, 2)),
        "charged": budget - balance,
        "low_ess": (np.asarray(belief.status) == SLOT_DEGRADED).sum(axis=(1, 2)),
        "no_funds": (
            np.asarray(trace.admit_code).reshape(n_w, -1) == int(AdmissionCode.NO_FUNDS)
        ).sum(axis=1),
    }
    slot = {
        "present": present,
        "klass": klass,
        "n_det": n_det,
        "detected": n_det > 0,
        "detected_again": n_det > 1,
        "candidate": candidate,
        "revealed": np.where(code > 0, code - REVEAL_OFFSET, -1),
        "claimed": claimed,
        "claim_error_lod": error,
        "block_attempted": attempted,
        "block_attempts": attempts,
        "n_spectra": spectra,
        "block_delivered": delivered,
        "block_snr": block_snr,
        "block_days": block_days,
    }
    if check_detection:
        slot["det_expected"] = expected
        slot["det_variance"] = variance
    return OutcomeTable(
        slot=slot,
        star=star,
        world={k: np.asarray(v, np.float64) for k, v in world.items()},
    )


def _claim_error(study, truths, belief, claimed, last_det_jd):
    """Claimed position against the truth at the predicted epoch, in lambda over D."""
    channel = study.channel_model
    lod = float(channel.assets().report.lod_arcsec)
    out = np.full(claimed.shape, np.nan)
    pos = np.asarray(belief.pos_mean)
    for w, s, k in zip(*np.nonzero(claimed), strict=True):
        truth = jax.tree_util.tree_map(lambda x, w=w: x[w], truths)
        elements = channel.truth_elements(truth)[s, k][None]
        epoch = float(channel.epoch_jd)
        if np.isnan(last_det_jd[w, s, k]):
            continue
        day = last_det_jd[w, s, k] - epoch + POSITION_LAG_D
        true_xy = np.asarray(channel.sky_position(int(s), day, elements))[0]
        out[w, s, k] = float(np.hypot(*(pos[w, s, k] - true_xy))) / lod
    return out


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
    candidate_class=CANDIDATE_CLASS,
    n_char=None,
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
        candidate_class: Passed to `outcome_table`.
        n_char: Passed to `outcome_table`.
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
            candidate_class=candidate_class,
            n_char=n_char,
        )
    return ArmsResult(table=tables)
