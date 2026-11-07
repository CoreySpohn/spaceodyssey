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
from spaceodyssey.core.battery.views import DecisionViews, compare_views
from spaceodyssey.core.carry import PROJECTION_FIELDS
from spaceodyssey.core.enums import EventKind, RunStatus
from spaceodyssey.core.executors.base import run_status
from spaceodyssey.core.records import log_columns
from spaceodyssey.core.snapshot import named_leaves

# Log columns a release delay may move, all by exactly the delay.
SHIFTED_LOG_FIELDS = ("available_tick", "released_tick", "accepted_tick")
# Trace fields compared on every step that is not a release. The belief path moves with
# its updates and the policy draw is keyed by its decision tick, which a release-woken
# decision shares with the release, so both are compared separately or not at all.
# The snapshot and event digests hash release ticks, which the delay moves by design.
LATENCY_FREE_SKIP = (*PROJECTION_FIELDS, "pol_u", "snapshot_fp", "event_fp")


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
    """Projected belief after the last handled step at or before `tick`, else zeros.

    Both projection fields count: a layout may record its updates in `belief_aux`.
    """
    handled = np.asarray(trace.handled) & (np.asarray(trace.tick) <= tick)
    belief = np.concatenate(
        [np.asarray(trace.belief_mean), np.asarray(trace.belief_aux)], axis=-1
    )
    idx = np.flatnonzero(handled)
    return belief[idx[-1]] if idx.size else np.zeros_like(belief[0])


def latency_shift(study, truths, roots, n_steps, *, delay_s=2 * 86400.0, executor=None):
    """Delay every release by `delay_s` seconds and require only belief updates to move.

    Assumptions: the study follows a fixed plan, meaning its acquisitions start at
    ticks that do not depend on when records are released (absolute waits), and the
    delay is shorter than the spacing of the plan, so a delayed release never crosses
    the next planned decision. Both runs must finish. Then, after the release steps
    are aligned by their delay, every decision has the same effects and admission
    codes; steps other than releases are bitwise equal;
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
    va, vb = DecisionViews(), DecisionViews()
    moved_study = delayed(study, delay)
    base = runner.run_worlds(va.instrument(study), truths, roots, n_steps)
    late = runner.run_worlds(vb.instrument(moved_study), truths, roots, n_steps)
    layout = study.observer.layout(study.caps)
    failures = []
    moved = 0
    for name, run in (("base", base), ("delayed", late)):
        if bool(np.any(np.asarray(run_status(run.carry)) != RunStatus.COMPLETE)):
            failures.append(f"the {name} run did not complete")
    for w, ((cb, tb), (cl, tl)) in enumerate(
        zip(worlds_of(base), worlds_of(late), strict=True)
    ):
        failures += compare_views(
            va.read(study, roots[w], tb),
            vb.read(moved_study, roots[w], tl),
            cb,
            cl,
            study.caps,
            tb,
            layout=layout,
            delay=delay,
        )
        failures += _compare_world(w, cb, tb, cl, tl, delay, layout)
        moved += _count_moved_updates(tb, tl)
    if moved == 0 and not failures:
        failures.append("the delay moved no belief update, so the check is vacuous")
    return make_result("latency_shift", failures, delay_ticks=delay, moved=moved)


def _non_release(trace):
    keep = np.asarray(trace.handled) & (np.asarray(trace.kind) != EventKind.RELEASE)
    return {
        n: np.asarray(getattr(trace, n))[keep]
        for n in type(trace).__dataclass_fields__
        if n not in LATENCY_FREE_SKIP and (n != "program" or trace.program is not None)
    }


def _compare_world(w, cb, tb, cl, tl, delay, layout):
    out = []
    a, b = _non_release(tb), _non_release(tl)
    bad = [n for n in a if not bits_equal(a[n], b[n])]
    if bad:
        out.append(f"world {w}: steps other than releases changed in {', '.join(bad)}")
    wb = np.asarray(tb.handled) & np.asarray(tb.wake)
    wl = np.asarray(tl.handled) & np.asarray(tl.wake)
    for name in ("eff_kind", "eff_disruptive", "admit_code"):
        if not bits_equal(
            np.asarray(getattr(tb, name))[wb], np.asarray(getattr(tl, name))[wl]
        ):
            out.append(f"world {w}: decision {name} changed under the delay")
    rb, rl = log_index(cb), log_index(cl)
    if rb.keys() != rl.keys():
        out.append(f"world {w}: the set of records changed")
        return out
    for key, i in rb.items():
        j = rl[key]
        cols_l = log_columns(cl.log)
        for name, col in log_columns(cb.log).items():
            x, y = np.asarray(col[i]), np.asarray(cols_l[name][j])
            if name in SHIFTED_LOG_FIELDS:
                want = np.where(x >= 0, x + delay, x)
                if not bits_equal(want, y):
                    out.append(f"world {w}: {name} of {key} did not move by the delay")
            elif not bits_equal(x, y):
                out.append(f"world {w}: {name} of {key} changed")
    for name in ("balance",):
        if not bits_equal(getattr(cb.ledger, name), getattr(cl.ledger, name)):
            out.append(f"world {w}: the ledger {name} changed")
    final_b, final_l = named_leaves(cb.belief), named_leaves(cl.belief)
    for name in layout.belief_leaves:
        if not bits_equal(final_b[name], final_l[name]):
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
