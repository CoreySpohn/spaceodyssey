"""Executors base."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.carry import Carry, Trace
from spaceodyssey.core.enums import I64, RunStatus
from spaceodyssey.core.snapshot import BeliefLayout, named_leaves

GO_TOLERANCE = 1e-12
KILL_TOLERANCE = 1e-9

# Every leaf of `Carry` and `Trace` appears here, in `LAYOUT_FIELDS` or in
# `CANONICAL_EXCLUSIONS` (`tests/core/test_canonical_fields.py` enumerates the
# dataclasses independently).
CANONICAL_FIELDS = {
    "carry": (
        "tick",
        "stage",
        "micro",
        "phase",
        "n_log",
        "acquisition_counts",
        "pstate",
        "woke",
        "pending_wake",
        "done",
        "draining",
        "stalled",
        "idle",
        "overflow",
        "overflow_mask",
        "violation",
        "double_settle",
        "dup_fold",
        "n_dec",
        "dec_tick",
        "dec_ordinal",
        "next_sid",
        "release_version",
        "refresh_dirty",
        "refresh_status",
        "query_refused",
    ),
    "carry.ev": (
        "kind",
        "tick",
        "stage",
        "micro",
        "tie",
        "sid",
        "lane",
        "channel_index",
        "channel",
        "subject",
        "acquisition_index",
        "gen",
        "group",
        "audience",
        "slot",
        "scope",
        "disposition",
        "acquisition_id",
        "snapshot_fp",
        "version",
        "decision",
        "valid",
        "program",
        "bundle",
        "config",
        "ops",
    ),
    "carry.log": (
        "kind",
        "subject",
        "channel",
        "channel_index",
        "lane",
        "acquisition_index",
        "start_tick",
        "end_tick",
        "available_tick",
        "status",
        "phase",
        "acquisition_id",
        "revision",
        "released_tick",
        "accepted_tick",
        "superseded_by",
        "audience",
        "source",
        "vintage",
        "config_id",
        "refund_of",
        "cost",
        "charged",
        "payload",
        "valid",
        "product",
        "program",
        "nonrefundable",
        "refund_index",
        "award_gen",
        "award_gens",
        "award_rules",
        "award_tos",
        "bundle",
        "ops",
    ),
    "carry.ledger": (
        "balance",
        "lane_busy_until",
        "lane_gen",
        "lane_closed",
        "valid_from_tick",
        "valid_until_tick",
        "remainder_rule",
        "disruptive_used",
        "debited",
        "refunded",
        "program_closed",
        "refund_ignored",
        "win_state",
        "win_gen",
        "lane_award_gens",
        "lane_award_rules",
        "lane_award_tos",
        "realloc_row",
        "granted",
        "lapsed",
        "cm_state",
        "cm_lane",
        "cm_program",
        "cm_start",
        "cm_end",
        "cm_index",
        "cm_count",
        "cm_config",
        "cm_ops",
        "ops_state",
    ),
    "carry.request": (
        "kind",
        "tick",
        "stage",
        "micro",
        "tie",
        "sid",
        "lane",
        "channel_index",
        "channel",
        "subject",
        "acquisition_index",
        "gen",
        "group",
        "audience",
        "slot",
        "scope",
        "disposition",
        "acquisition_id",
        "snapshot_fp",
        "version",
        "decision",
        "valid",
        "program",
        "bundle",
        "config",
        "ops",
    ),
    "carry.refresh": (
        "kind",
        "tick",
        "stage",
        "micro",
        "tie",
        "sid",
        "lane",
        "channel_index",
        "channel",
        "subject",
        "acquisition_index",
        "gen",
        "group",
        "audience",
        "slot",
        "scope",
        "disposition",
        "acquisition_id",
        "snapshot_fp",
        "version",
        "decision",
        "valid",
        "program",
        "bundle",
        "config",
        "ops",
    ),
    "trace": (
        "handled",
        "kind",
        "tick",
        "stage",
        "micro",
        "tie",
        "sid",
        "lane",
        "channel",
        "subject",
        "acquisition_index",
        "log_slot",
        "audience",
        "n_written",
        "rec_status",
        "wake",
        "pending_wake",
        "n_dec",
        "snapshot_fp",
        "event_fp",
        "release_version",
        "eff_kind",
        "eff_disruptive",
        "admit_code",
        "belief_mean",
        "belief_aux",
        "n_log",
        "pol_u",
        "program",
        "alloc_amount",
        "alloc_from",
        "alloc_until",
        "alloc_recipient",
        "alloc_remainder",
        "alloc_to",
    ),
}

# Carry fields whose leaves are named by the observer's `BeliefLayout`: every
# comparator leaf of the full belief and of the policy summary (see `belief_leaves`).
LAYOUT_FIELDS = ("carry.belief", "carry.pbelief")


def refresh_on(carry):
    """Whether the carry holds the refresh protocol (its status leaf has size)."""
    return np.asarray(carry.refresh_status).size > 0


def programs_on(carry):
    """Whether the carry holds more than one program (the closed flags exist)."""
    return carry.ledger.program_closed is not None


def refunds_on(carry):
    """Whether the carry holds the refund columns and counter."""
    return carry.log.refund_index is not None


def bundles_on(carry):
    """Whether the carry holds the bundle ids (the event table has the column)."""
    return carry.ev.bundle is not None


def config_on(carry):
    """Whether the carry holds configuration stamps (the event rows have the column)."""
    return carry.ev.config is not None


def ops_on(carry):
    """Whether the carry holds asset ops states (the event rows have the column)."""
    return carry.ev.ops is not None


def balances_on(carry):
    """Whether the carry holds the balance snapshot (its refused-request leaf)."""
    return carry.query_refused is not None


def awards_on(carry):
    """Whether the carry holds the award state (the window generations exist)."""
    return carry.ledger.win_gen is not None


def commitments_on(carry):
    """Whether the carry holds the commitment table."""
    return carry.ledger.cm_state is not None


# Fields that exist only where a feature does, by dotted path, each with the predicate
# that says whether the carry holds the feature. A field of a feature that is off has
# zero size and leaves the canonical history, so no history of a study without the
# feature moves; a feature adds its fields here (the guard in `test_slice3_guard.py`
# checks that each is empty while its predicate is false, and `canonical_history`
# drops it). `CONDITIONAL_GROUPS` does the same for a whole carry group.
CONDITIONAL_GROUPS = {"carry.refresh": refresh_on}
CONDITIONAL_FIELDS = {
    "carry.refresh_dirty": refresh_on,
    "carry.refresh_status": refresh_on,
    "carry.log.product": refresh_on,
    "carry.ev.program": programs_on,
    "carry.log.program": programs_on,
    "carry.request.program": programs_on,
    "carry.refresh.program": programs_on,
    "carry.ledger.program_closed": programs_on,
    "carry.log.nonrefundable": refunds_on,
    "carry.log.refund_index": refunds_on,
    "carry.log.award_gen": refunds_on,
    "carry.ledger.refund_ignored": refunds_on,
    "carry.ev.bundle": bundles_on,
    "carry.log.bundle": bundles_on,
    "carry.request.bundle": bundles_on,
    "carry.refresh.bundle": bundles_on,
    "carry.ev.config": config_on,
    "carry.request.config": config_on,
    "carry.refresh.config": config_on,
    "carry.ledger.cm_config": config_on,
    "carry.ev.ops": ops_on,
    "carry.log.ops": ops_on,
    "carry.request.ops": ops_on,
    "carry.refresh.ops": ops_on,
    "carry.ledger.cm_ops": ops_on,
    "carry.ledger.ops_state": ops_on,
    "trace.program": programs_on,
    "carry.query_refused": balances_on,
    "carry.ledger.win_state": awards_on,
    "carry.ledger.win_gen": awards_on,
    "carry.ledger.lane_award_gens": awards_on,
    "carry.log.award_gens": awards_on,
    "carry.log.award_rules": awards_on,
    "carry.log.award_tos": awards_on,
    "carry.ledger.lane_award_rules": awards_on,
    "carry.ledger.lane_award_tos": awards_on,
    "carry.ledger.realloc_row": awards_on,
    "carry.ledger.granted": awards_on,
    "carry.ledger.lapsed": awards_on,
    "carry.ledger.cm_state": commitments_on,
    "carry.ledger.cm_lane": commitments_on,
    "carry.ledger.cm_program": commitments_on,
    "carry.ledger.cm_start": commitments_on,
    "carry.ledger.cm_end": commitments_on,
    "carry.ledger.cm_index": commitments_on,
    "carry.ledger.cm_count": commitments_on,
    "trace.alloc_amount": awards_on,
    "trace.alloc_from": awards_on,
    "trace.alloc_until": awards_on,
    "trace.alloc_recipient": awards_on,
    "trace.alloc_remainder": awards_on,
    "trace.alloc_to": awards_on,
}
REFRESH_GROUPS = tuple(CONDITIONAL_GROUPS)
REFRESH_FIELDS = tuple(CONDITIONAL_FIELDS)

# Whole-field exclusions: dotted output path -> reason. Empty on purpose; invalid
# queue/log rows and unhandled trace rows are masked in `canonical_history`, not
# excluded by field. See docs/executors.md.
CANONICAL_EXCLUSIONS = {}


def run_status(carry):
    """Per-lane `RunStatus` of a final carry (elementwise, so batches work).

    An overflow outranks a stall, a stall outranks a request nobody answered, that
    outranks a run that merely used up its step budget, and only a run that reached its
    own end without any of them is `COMPLETE`.
    """
    s = jnp.where(carry.done, RunStatus.COMPLETE, RunStatus.TRUNCATED)
    s = jnp.where(carry.idle, RunStatus.IDLE, s)
    s = jnp.where(carry.request.valid | carry.refresh.valid, RunStatus.AWAITING, s)
    s = jnp.where(carry.stalled, RunStatus.STALLED, s)
    s = jnp.where(carry.overflow, RunStatus.OVERFLOW, s)
    return s.astype(I64)


def index_world(tree, w):
    """Slice one world out of a batched pytree."""
    return jax.tree_util.tree_map(lambda a: a[w], tree)


class RunHistory(eqx.Module):
    """What parity compares: handled steps in order, valid log rows, belief path."""

    events: tuple  # ((tick, stage, kind), ...) per handled step
    # (tick, stage, kind, lane, subject, acquisition_index, n_written, wake)
    rows: tuple
    lanes: tuple
    n_handled: int
    n_records: int
    double_settle: bool
    flags: tuple  # (done, idle, stalled, overflow, violation)
    ints: dict  # name -> int array
    floats: dict
    status: int  # a RunStatus code
    overflow_mask: int  # Capacity bits that ran out

    @property
    def complete(self):
        """Whether the run reached its own end without stalling or overflowing."""
        return self.status == RunStatus.COMPLETE


class ParityReport(eqx.Module):
    """Integer and floating-point differences between two histories."""

    structure_equal: bool
    max_int_diff: int
    max_abs: float
    max_rel: float
    max_ulps: float
    vacuous: bool

    @property
    def ok(self):
        """Whether the nonempty histories agree below the kill tolerance."""
        return (
            self.structure_equal
            and self.max_int_diff == 0
            and self.max_abs <= KILL_TOLERANCE
            and not self.vacuous
        )

    @property
    def go(self):
        """Whether the histories also agree within the go tolerance."""
        return self.ok and self.max_abs <= GO_TOLERANCE


def belief_leaves(carry, layout):
    """Comparator leaves of both beliefs, as `belief.<leaf>` and `pbelief.<leaf>`.

    The names are the layout's: leaves of nonzero size of the full belief and of the
    policy summary.
    """
    out = {}
    for prefix, tree, names in (
        ("belief", carry.belief, layout.belief_leaves),
        ("pbelief", carry.pbelief, layout.summary_leaves),
    ):
        leaves = named_leaves(tree)
        out.update({f"{prefix}.{name}": np.asarray(leaves[name]) for name in names})
    return out


def _canonical_leaves(carry, trace, handled, layout):
    """Dotted path -> array for every inventoried field, inactive rows removed.

    Queue and log rows compare where `valid` is set, in slot order (the `valid`
    masks themselves compare in full); trace rows compare where `handled` is set.
    The retained request and the scalar carry fields compare in full.
    """
    out = {}
    for group, names in CANONICAL_FIELDS.items():
        if group in CONDITIONAL_GROUPS and not CONDITIONAL_GROUPS[group](carry):
            continue
        obj = carry if group == "carry" else trace if group == "trace" else None
        if obj is None:
            obj = carry
            for part in group.split(".")[1:]:
                obj = getattr(obj, part)
        mask = None
        if group == "trace":
            mask = handled
        elif group in ("carry.ev", "carry.log"):
            mask = np.asarray(obj.valid)
        for name in names:
            gate = CONDITIONAL_FIELDS.get(f"{group}.{name}")
            if gate is not None and not gate(carry):
                continue
            leaf = np.asarray(getattr(obj, name))
            if mask is not None and name != "valid":
                leaf = leaf[mask]
            out[f"{group}.{name}"] = leaf
    for name, leaf in belief_leaves(carry, layout).items():
        out[f"carry.{name}"] = leaf
    return out


def canonical_history(carry, trace, layout):
    """Valid-compressed history of one world: handled steps and valid log rows.

    `layout` is the observer's `BeliefLayout` (`study.observer.layout(study.caps)`);
    it names the belief leaves compared, the count leaf and the projections of the
    final beliefs.
    """
    final_mean, final_aux = layout.project(carry.belief)
    summary_mean, _ = layout.summary_projection(carry.pbelief)
    h = np.asarray(trace.handled)
    pick = lambda a: np.asarray(a)[h]  # noqa: E731
    events = tuple(
        zip(
            pick(trace.tick).tolist(),
            pick(trace.stage).tolist(),
            pick(trace.kind).tolist(),
            strict=False,
        )
    )
    rows = tuple(
        zip(
            *(
                pick(getattr(trace, n)).tolist()
                for n in (
                    "tick",
                    "stage",
                    "kind",
                    "lane",
                    "subject",
                    "acquisition_index",
                    "n_written",
                )
            ),
            pick(trace.wake).astype(np.int64).tolist(),
            strict=False,
        )
    )
    v = np.asarray(carry.log.valid)
    log = carry.log
    ints = {
        "events": np.asarray(events, dtype=np.int64).reshape(-1, 3),
        "lanes": pick(trace.lane),
        "n_written": pick(trace.n_written),
        "rec_status": pick(trace.rec_status),
        "wake": pick(trace.wake),
        "eff_kind": pick(trace.eff_kind),
        "eff_disruptive": pick(trace.eff_disruptive),
        "admit_code": pick(trace.admit_code),
        "n_dec": pick(trace.n_dec),
        "snapshot_fp": pick(trace.snapshot_fp),
        "event_fp": pick(trace.event_fp),
        "release_version": pick(trace.release_version),
        "release_version_final": np.asarray(carry.release_version),
        "micro": pick(trace.micro),
        "sid": pick(trace.sid),
        "tie": pick(trace.tie),
        "pending_wake": pick(trace.pending_wake),
        "pending_wake_final": np.asarray(carry.pending_wake),
        "overflow_mask": np.asarray(carry.overflow_mask),
        "run_status": np.asarray(run_status(carry)),
        "subject": np.asarray(log.subject)[v],
        "acquisition_index": np.asarray(log.acquisition_index)[v],
        "acquisition_id": np.asarray(log.acquisition_id)[v],
        "channel": np.asarray(log.channel)[v],
        "status": np.asarray(log.status)[v],
        "phase": np.asarray(log.phase)[v],
        "vintage": np.asarray(log.vintage)[v],
        "start_tick": np.asarray(log.start_tick)[v],
        "end_tick": np.asarray(log.end_tick)[v],
        "available_tick": np.asarray(log.available_tick)[v],
        "released_tick": np.asarray(log.released_tick)[v],
        "accepted_tick": np.asarray(log.accepted_tick)[v],
        "revision": np.asarray(log.revision)[v],
        "superseded_by": np.asarray(log.superseded_by)[v],
        "audience": np.asarray(log.audience)[v],
        "kind": np.asarray(log.kind)[v],
        "next_sid": np.asarray(carry.next_sid),
        "n_log": np.asarray(carry.n_log),
        "acquisition_counts": np.asarray(carry.acquisition_counts),
        "belief_n": np.asarray(layout.count(carry.belief)),
        "pbelief_n": np.asarray(layout.count(carry.pbelief)),
        "lane_gen": np.asarray(carry.ledger.lane_gen),
        "refund_of": np.asarray(log.refund_of)[v],
        "valid_from_tick": np.asarray(carry.ledger.valid_from_tick),
        "valid_until_tick": np.asarray(carry.ledger.valid_until_tick),
        "remainder_rule": np.asarray(carry.ledger.remainder_rule),
        "disruptive_used": np.asarray(carry.ledger.disruptive_used),
    }
    floats = {
        "payload": np.asarray(log.payload)[v],
        "cost": np.asarray(log.cost)[v],
        "charged": np.asarray(log.charged)[v],
        "belief_mean": pick(trace.belief_mean),
        "belief_aux_path": pick(trace.belief_aux),
        "belief_final": np.asarray(final_mean),
        "pbelief_final": np.asarray(summary_mean),
        "belief_aux": np.asarray(final_aux),
        "balance": np.asarray(carry.ledger.balance),
    }
    ints["meta"] = np.asarray(
        [int(h.sum()), int(v.sum()), int(run_status(carry)), int(carry.overflow_mask)],
        dtype=np.int64,
    )
    ints["lane_tuple"] = np.asarray(pick(trace.lane), dtype=np.int64)
    for path, leaf in _canonical_leaves(carry, trace, h, layout).items():
        (floats if leaf.dtype.kind == "f" else ints)[path] = leaf
    return RunHistory(
        events=events,
        rows=rows,
        lanes=tuple(pick(trace.lane).tolist()),
        n_handled=int(h.sum()),
        n_records=int(v.sum()),
        double_settle=bool(carry.double_settle),
        flags=tuple(
            bool(getattr(carry, n))
            for n in ("done", "idle", "stalled", "overflow", "violation")
        ),
        ints=ints,
        floats=floats,
        status=int(run_status(carry)),
        overflow_mask=int(carry.overflow_mask),
    )


def histories_equal(a, b):
    """Compare nonempty histories with exact integers and bounded float differences."""
    ia = dict(a.ints, events=np.asarray(a.events, dtype=np.int64).reshape(-1, 3))
    ib = dict(b.ints, events=np.asarray(b.events, dtype=np.int64).reshape(-1, 3))
    ia["rows"] = np.asarray(a.rows, dtype=np.int64).reshape(-1, 8)
    ib["rows"] = np.asarray(b.rows, dtype=np.int64).reshape(-1, 8)
    ia["flags"] = np.asarray((*a.flags, a.double_settle), dtype=np.int64)
    ib["flags"] = np.asarray((*b.flags, b.double_settle), dtype=np.int64)
    ia["lanes_history"] = np.asarray(a.lanes, dtype=np.int64)
    ib["lanes_history"] = np.asarray(b.lanes, dtype=np.int64)
    ia["counts"] = np.asarray([a.n_handled, a.n_records, a.status, a.overflow_mask])
    ib["counts"] = np.asarray([b.n_handled, b.n_records, b.status, b.overflow_mask])
    rep = tree_equal_report((ia, a.floats), (ib, b.floats))
    return ParityReport(
        structure_equal=rep["structure_equal"] and rep["shapes_equal"],
        max_int_diff=rep["max_int_diff"],
        max_abs=rep["max_abs"],
        max_rel=rep["max_rel"],
        max_ulps=rep["max_ulps"],
        vacuous=a.n_records == 0 or a.n_handled < 2,
    )


def tree_equal_report(a, b):
    """Leaf-by-leaf comparison by dtype; never calls `jnp.array_equal` on a module.

    Raises `TypeError` for a complex leaf: the float comparison would drop its
    imaginary part and report equal values.
    """
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    rep = {
        "structure_equal": ta == tb,
        "shapes_equal": False,
        "max_int_diff": 0,
        "max_abs": 0.0,
        "max_rel": 0.0,
        "max_ulps": 0.0,
    }
    if ta != tb:
        return rep
    rep["shapes_equal"] = all(
        np.shape(x) == np.shape(y) for x, y in zip(la, lb, strict=False)
    )
    if not rep["shapes_equal"]:
        return rep
    for x, y in zip(la, lb, strict=False):
        x, y = np.asarray(x), np.asarray(y)
        if "c" in (x.dtype.kind, y.dtype.kind):
            raise TypeError("tree_equal_report does not compare complex leaves")
        if x.dtype != y.dtype:
            rep["structure_equal"] = False
            continue
        if x.dtype.kind in "biu":
            if not np.array_equal(x, y):
                # Python integers retain the full difference at either 64-bit limit.
                d = np.abs(x.astype(object) - y.astype(object)).max()
                rep["max_int_diff"] = max(rep["max_int_diff"], int(d))
        else:
            if not x.size:
                continue
            x, y = x.astype(np.float64), y.astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                same_nan = np.isnan(x) & np.isnan(y)
                d = np.where(same_nan | (x == y), 0.0, np.abs(x - y))
                d = np.where(np.isnan(x) != np.isnan(y), np.inf, d)
                scale = np.maximum(np.abs(x), np.abs(y))
                relative_error = np.where(
                    scale > 0, d / np.where(scale > 0, scale, 1.0), 0.0
                )
                ulp = np.where(
                    d > 0, d / np.spacing(np.where(scale > 0, scale, 1.0)), 0.0
                )
            relative_error = np.where(np.isfinite(d), relative_error, np.inf)
            ulp = np.where(np.isfinite(d), ulp, np.inf)
            rep["max_abs"] = max(rep["max_abs"], float(d.max()))
            rep["max_rel"] = max(rep["max_rel"], float(relative_error.max()))
            rep["max_ulps"] = max(rep["max_ulps"], float(ulp.max()))
    return rep


class RunResult(eqx.Module):
    """Final carry and per-step trace; unpackable as a pair.

    `layout` is the observer's `BeliefLayout`, which `history` compares through.
    `roots` holds the world root keys the run used (one per world of a batched run), so
    that a replay can derive the same keys.
    """

    carry: Carry
    trace: Trace
    layout: BeliefLayout = eqx.field(static=True)
    roots: jax.Array | None = None
    # per world: the run saves checkpoints and its budget ended with a refresh
    # checkpoint held (kept, not dropped); absent (no leaf) for every other run
    held_by_budget: jax.Array | None = None

    def __iter__(self):
        """Iterate over the carry and trace."""
        return iter((self.carry, self.trace))

    @property
    def status(self):
        """`RunStatus` codes, one per world of a batched run."""
        status = run_status(self.carry)
        if self.held_by_budget is None:
            return status
        held = jnp.asarray(self.held_by_budget) & (status == RunStatus.AWAITING)
        return jnp.where(held, RunStatus.TRUNCATED_AT_HOLD, status).astype(I64)

    @property
    def complete(self):
        """Boolean per world: reached its own end, no stall, no overflow."""
        return self.status == RunStatus.COMPLETE

    def history(self, world=None):
        """Canonical history of one world, or of a run without a world axis."""
        if world is None:
            return canonical_history(self.carry, self.trace, self.layout)
        return canonical_history(
            index_world(self.carry, world), index_world(self.trace, world), self.layout
        )


class AbstractExecutor(eqx.Module):
    """Executor contract for the shared transaction."""

    version: int = eqx.field(default=1, kw_only=True)

    def run(self, study, truths, roots, n_steps):
        """Run a study for a fixed step budget."""
        raise NotImplementedError

    def run_worlds(self, study, truths, roots, n_steps):
        """Run a batch of worlds (leading axis on `truths` and `roots`).

        Both executors return a `RunResult` with a leading world axis, so the same
        status and history reads work on either.
        """
        raise NotImplementedError
