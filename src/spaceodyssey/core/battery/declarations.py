"""Declared stops and needs must match what a run emits."""

import jax
import numpy as np

from spaceodyssey.core.battery.result import make_result
from spaceodyssey.core.declarations import declaration_of
from spaceodyssey.core.enums import Disposition, EffectKind, Scope
from spaceodyssey.core.executors.host import HostExecutor


def _phase_of(study, tick):
    bounds = np.asarray(study.agenda.phase_until[:-1], dtype=np.int64)
    return int((tick >= bounds).sum())


def emitted_stops(study, truth, root, n_steps):
    """Stops each phase's policy asked for in one run, and the run itself.

    Returns ({phase: {(scope, disposition)}}, decision ticks, final carry). Runs the
    host executor so that every requested effect table is seen, including the ones
    admission refuses.
    """
    effects = []
    carry, trace = HostExecutor().run(study, truth, root, n_steps, effects_log=effects)
    woke = np.flatnonzero(np.asarray(trace.wake) & np.asarray(trace.handled))
    ticks = [int(np.asarray(trace.tick)[i]) for i in woke]
    stops = {}
    for tick, table in zip(ticks, effects, strict=True):
        phase = _phase_of(study, tick)
        for k in np.flatnonzero(np.asarray(table.kind) == EffectKind.STOP):
            pair = (Scope(int(table.scope[k])), Disposition(int(table.disposition[k])))
            stops.setdefault(phase, set()).add(pair)
    return stops, ticks, carry


def declared_matches_run(study, truths, roots, n_steps):
    """Check that the stops and needs the policies declare match what a run emits.

    Assumptions: the run is long enough to reach the decisions that matter (a stop
    that is declared and never emitted within `n_steps` is reported in the detail
    and is not a failure, since a declaration is a permission) and the declared
    needs name acquisitions the plan really makes. Required: every stop a policy asks
    for in a world appears in its phase's declaration; every declared need is met in
    at least one world, meaning an acquisition on its channel starts at its start
    tick, a decision runs at its decision tick, and the record is released to the
    policy by then.
    """
    failures = []
    declared = [declaration_of(p) for p in study.agenda.policies]
    emitted_all = {}
    needs_met = {}
    for w in range(int(roots.shape[0])):
        truth = jax.tree_util.tree_map(lambda x, w=w: x[w], truths)
        stops, ticks, carry = emitted_stops(study, truth, roots[w], n_steps)
        for phase, pairs in stops.items():
            emitted_all.setdefault(phase, set()).update(pairs)
            lies = pairs - set(declared[phase].stops)
            if lies:
                shown = ", ".join(f"{s.name}/{d.name}" for s, d in sorted(lies))
                failures.append(
                    f"world {w}: phase {phase} asked for undeclared stop(s) {shown}"
                )
        for phase, decl in enumerate(declared):
            for need in decl.needs:
                needs_met.setdefault((phase, need), False)
                if _need_met(
                    carry, [t for t in ticks if _phase_of(study, t) == phase], need
                ):
                    needs_met[(phase, need)] = True
    for (phase, need), met in needs_met.items():
        if not met:
            failures.append(
                f"phase {phase} declares a need of channel {need.channel_id} "
                f"(start {need.start_tick}, decide {need.decide_tick}) no run meets"
            )
    unused = {
        phase: sorted(
            f"{s.name}/{d.name}"
            for s, d in set(decl.stops) - emitted_all.get(phase, set())
        )
        for phase, decl in enumerate(declared)
    }
    return make_result("declared_matches_run", failures, declared_not_emitted=unused)


def _need_met(carry, ticks, need):
    log = carry.log
    rows = np.flatnonzero(
        np.asarray(log.valid)
        & (np.asarray(log.channel) == need.channel_id)
        & (np.asarray(log.start_tick) == need.start_tick)
    )
    if not rows.size or need.decide_tick not in ticks:
        return False
    released = np.asarray(log.released_tick)[rows]
    return bool(((released >= 0) & (released <= need.decide_tick)).any())
