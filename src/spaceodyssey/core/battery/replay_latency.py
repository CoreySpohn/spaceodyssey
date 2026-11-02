"""Replay with latency: a release delay moves belief updates and nothing else."""

import equinox as eqx
import numpy as np

from spaceodyssey.core.battery.result import (
    bits_equal,
    default_executor,
    log_index,
    make_result,
    worlds_of,
)
from spaceodyssey.core.enums import EventKind, RunStatus
from spaceodyssey.core.executors.base import run_status

# Log columns a release delay may move, all by exactly the delay.
SHIFTED_LOG_FIELDS = ("available_tick", "released_tick", "accepted_tick")
# Trace fields compared on every step that is not a release. The belief path moves with
# its updates and the policy draw is keyed by its decision tick, which a release-woken
# decision shares with the release, so both are compared separately or not at all.
# The snapshot and event digests hash release ticks, which the delay moves by design.
LATENCY_FREE_SKIP = ("belief_mean", "belief_aux", "pol_u", "snapshot_fp", "event_fp")


def delayed(study, delay_ticks):
    """`study` with every channel's release latency raised by `delay_ticks`."""
    return eqx.tree_at(
        lambda s: s.channels.latency_ticks,
        study,
        study.channels.latency_ticks + delay_ticks,
    )


def delay_in_ticks(study, delay_s):
    """The whole number of ticks nearest `delay_s` seconds on the study clock."""
    return round(delay_s / float(study.clock.tick_length_s))


def _belief_at(trace, tick):
    """Observer belief after the last handled step at or before `tick` (zeros first)."""
    handled = np.asarray(trace.handled) & (np.asarray(trace.tick) <= tick)
    belief = np.asarray(trace.belief_mean)
    idx = np.flatnonzero(handled)
    return belief[idx[-1]] if idx.size else np.zeros_like(belief[0])


def latency_shift(study, truths, roots, n_steps, *, delay_s=2 * 86400.0, executor=None):
    """Delay every release by `delay_s` seconds and require only belief updates to move.

    Assumptions: the study follows a fixed plan, meaning its acquisitions start at
    ticks that do not depend on when records are released (absolute waits), and the
    delay is shorter than the spacing of the plan, so a delayed release never crosses
    the next planned decision. Both runs must finish. Then, after the release steps
    are set aside, every step (event order, admissions, effects) is bitwise equal;
    every release, record column and payload is equal except that the release ticks
    move by exactly the delay; the final belief and ledger are bitwise equal; and the
    belief path is the original one delayed by the delay, with at least one update
    that really moved. Policy draws keyed by the tick of a release-woken decision move
    with it and are not compared.
    """
    delay = delay_in_ticks(study, delay_s)
    if delay < 1:
        raise ValueError("the delay is shorter than one tick")
    runner = default_executor(executor)
    base = runner.run_worlds(study, truths, roots, n_steps)
    late = runner.run_worlds(delayed(study, delay), truths, roots, n_steps)
    failures = []
    moved = 0
    for name, run in (("base", base), ("delayed", late)):
        if bool(np.any(np.asarray(run_status(run.carry)) != RunStatus.COMPLETE)):
            failures.append(f"the {name} run did not complete")
    for w, ((cb, tb), (cl, tl)) in enumerate(
        zip(worlds_of(base), worlds_of(late), strict=True)
    ):
        failures += _compare_world(w, cb, tb, cl, tl, delay)
        moved += _count_moved_updates(tb, tl)
    if moved == 0 and not failures:
        failures.append("the delay moved no belief update, so the check is vacuous")
    return make_result("latency_shift", failures, delay_ticks=delay, moved=moved)


def _non_release(trace):
    keep = np.asarray(trace.handled) & (np.asarray(trace.kind) != EventKind.RELEASE)
    return {
        n: np.asarray(getattr(trace, n))[keep]
        for n in type(trace).__dataclass_fields__
        if n not in LATENCY_FREE_SKIP
    }


def _compare_world(w, cb, tb, cl, tl, delay):
    out = []
    a, b = _non_release(tb), _non_release(tl)
    bad = [n for n in a if not bits_equal(a[n], b[n])]
    if bad:
        out.append(f"world {w}: steps other than releases changed in {', '.join(bad)}")
    rb, rl = log_index(cb), log_index(cl)
    if rb.keys() != rl.keys():
        out.append(f"world {w}: the set of records changed")
        return out
    for key, i in rb.items():
        j = rl[key]
        for name in type(cb.log).__dataclass_fields__:
            x, y = (
                np.asarray(getattr(cb.log, name)[i]),
                np.asarray(getattr(cl.log, name)[j]),
            )
            if name in SHIFTED_LOG_FIELDS:
                want = np.where(x >= 0, x + delay, x)
                if not bits_equal(want, y):
                    out.append(f"world {w}: {name} of {key} did not move by the delay")
            elif not bits_equal(x, y):
                out.append(f"world {w}: {name} of {key} changed")
    for name in ("balance",):
        if not bits_equal(getattr(cb.ledger, name), getattr(cl.ledger, name)):
            out.append(f"world {w}: the ledger {name} changed")
    for name in ("mean", "n", "aux"):
        if not bits_equal(getattr(cb.belief, name), getattr(cl.belief, name)):
            out.append(f"world {w}: the final belief {name} changed")
    ticks = np.unique(np.asarray(tb.tick)[np.asarray(tb.handled)])
    for t in ticks:
        if not bits_equal(_belief_at(tb, t), _belief_at(tl, t + delay)):
            out.append(f"world {w}: the belief path is not the original delayed")
            break
    return out


def _count_moved_updates(tb, tl):
    """Ticks at which the delayed belief lags the original one."""
    ticks = np.unique(np.asarray(tb.tick)[np.asarray(tb.handled)])
    return sum(not bits_equal(_belief_at(tb, t), _belief_at(tl, t)) for t in ticks)
