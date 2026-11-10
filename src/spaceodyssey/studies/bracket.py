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

import dataclasses

import equinox as eqx
import jax.numpy as jnp

from spaceodyssey.adapters.jaxedith import (
    block_outcome_kind,
    truth_catalog_kind,
    visit_report_kind,
)
from spaceodyssey.adapters.photomancy import (
    CATALOG_KIND_NAME,
    REPORT_KIND_NAME,
    SUMMARY_LEAVES,
    DecisionSummary,
    ReportRefit,
)
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import (
    EVERYONE,
    F64,
    I64,
    PUBLIC,
    Audience,
    CoincidentOrder,
)
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.kinds import AbstractObserver, ObserverCapabilities
from spaceodyssey.core.ledger import ChannelTable
from spaceodyssey.core.records import KindRegistry, RecordKind
from spaceodyssey.core.refresh import REFRESH_PRODUCT_KIND, RefreshSchedule
from spaceodyssey.core.snapshot import BeliefLayout
from spaceodyssey.core.study import Clock
from spaceodyssey.studies import toy

# Kind indices of `bracket_registry`.
KIND_NONE, KIND_PRODUCT, KIND_VISIT, KIND_BLOCK, KIND_CATALOG = range(5)
# The telescope and the catalog lanes, and the shared budget resource column.
LANE_TELESCOPE, LANE_CATALOG = 0, 1
SHARED_RESOURCE = 2
# Channel ids of the modes start here; the refresh product channels are 90 and 91.
FIRST_CHANNEL_ID = 10
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
    "shrinkage": Arm("shrinkage", EVERYONE, ("ephemeris",), "shrinkage"),
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


def default_capacities(n_subjects, n_slots, *, n_channels):
    """Capacities of the bracket study of `n_subjects` stars.

    `rows_per_subject` holds 24 visits, the catalog row and margin; `n_pstate` holds
    three counters per star, a block flag per slot and the charged time.
    """
    return Capacities(
        n_subjects=n_subjects,
        n_assets=2,
        n_lanes=2,
        n_programs=1,
        n_resources=3,
        n_channels=n_channels,
        n_log=1024,
        rows_per_subject=32,
        refit_lanes=2,
        n_pstate=n_subjects * (3 + n_slots) + 3,
        n_an=1,
    )


def channel_table(channel, caps, *, budget_unit_s=None):
    """The `ChannelTable` of the observing modes of `channel`.

    A detection mode occupies the settling time plus its exposure; a block mode its
    length; the catalog mode one tick. Telescope modes charge their occupancy in days
    to the shared resource; the catalog mode charges nothing.
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
        occupancy.append(round(days * per_day))
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
        caps: Capacities; defaults to `default_capacities`.
        observer: The summary observer. It must be non-incremental and refit the
            visit and catalog kinds.
        policy: The decision policy.
    """
    arm = _arm(arm)
    if observer is None or policy is None:
        raise ValueError("bracket_study needs an observer and a policy")
    built = channel.assets()
    n_stars = int(channel.catalog_mass_kg.shape[0])
    if len(source.stars) != n_stars:
        raise ValueError("the world source and the channel disagree on the stars")
    caps = caps or default_capacities(
        n_stars, channel.n_slots, n_channels=channel.n_channels
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


def bracket_boundary(study, arm):
    """The refit boundary of `arm` for `study` (`ReportRefit` with the arm's kernel).

    Construction refuses a registry that hides a catalog group the kernel needs.
    """
    arm = _arm(arm)
    return ReportRefit(kernel=arm.kernel, truth_groups=arm.groups, caps=study.caps)


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


def project_detections(belief):
    """The trace projection: detections per star, and a zero auxiliary vector."""
    return belief.n_det.sum(axis=-1).astype(F64), jnp.zeros_like(belief.aux)


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

    def capabilities(self):
        """Refit only, from visit reports and the truth catalog."""
        return ObserverCapabilities(
            incremental=False,
            refit=True,
            refit_kinds=(self.report_kind, self.catalog_kind),
        )

    def init(self, caps):
        """The prior summary of every star."""
        return self.prior

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
        """Replace one star's leaves with those of the refit belief."""
        return eqx.tree_at(
            lambda b: tuple(getattr(b, n) for n in SUMMARY_LEAVES),
            belief,
            tuple(
                getattr(belief, n).at[subject].set(getattr(fitted, n)[subject])
                for n in SUMMARY_LEAVES
            ),
        )
