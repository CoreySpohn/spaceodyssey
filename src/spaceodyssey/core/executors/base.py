"""Executors base."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.carry import Carry, Trace
from spaceodyssey.core.enums import I64, RunStatus

GO_TOLERANCE = 1e-12
KILL_TOLERANCE = 1e-9


def run_status(carry):
    """Per-lane `RunStatus` of a final carry (elementwise, so batches work).

    An overflow outranks a stall, a stall outranks a request nobody answered, that
    outranks a run that merely used up its step budget, and only a run that reached its
    own end without any of them is `COMPLETE`.
    """
    s = jnp.where(carry.done, RunStatus.COMPLETE, RunStatus.TRUNCATED)
    s = jnp.where(carry.idle, RunStatus.IDLE, s)
    s = jnp.where(carry.request.valid, RunStatus.AWAITING, s)
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


def canonical_history(carry, trace):
    """Valid-compressed history of one world: handled steps and valid log rows."""
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
        "n_log": np.asarray(carry.n_log),
        "acquisition_counts": np.asarray(carry.acquisition_counts),
        "belief_n": np.asarray(carry.belief.n),
        "pbelief_n": np.asarray(carry.pbelief.n),
        "lane_gen": np.asarray(carry.ledger.lane_gen),
        "refund_of": np.asarray(log.refund_of)[v],
        "program_charge_rule": np.asarray(carry.ledger.program_charge_rule),
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
        "belief_final": np.asarray(carry.belief.mean),
        "pbelief_final": np.asarray(carry.pbelief.mean),
        "belief_aux": np.asarray(carry.belief.aux),
        "balance": np.asarray(carry.ledger.balance),
    }
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
    """Leaf-by-leaf comparison by dtype; never calls `jnp.array_equal` on a module."""
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
    """Final carry and per-step trace; unpackable as a pair."""

    carry: Carry
    trace: Trace

    def __iter__(self):
        """Iterate over the carry and trace."""
        return iter((self.carry, self.trace))

    @property
    def status(self):
        """`RunStatus` codes, one per world of a batched run."""
        return run_status(self.carry)

    @property
    def complete(self):
        """Boolean per world: reached its own end, no stall, no overflow."""
        return self.status == RunStatus.COMPLETE

    def history(self, world=None):
        """Canonical history of one world, or of a run without a world axis."""
        if world is None:
            return canonical_history(self.carry, self.trace)
        return canonical_history(
            index_world(self.carry, world), index_world(self.trace, world)
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
