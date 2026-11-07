"""Truth swap: nothing the policy does may depend on truth it has not been given."""

import numpy as np

from spaceodyssey.core.battery.result import (
    bits_equal,
    default_executor,
    make_result,
    worlds_of,
)
from spaceodyssey.core.carry import PROJECTION_FIELDS
from spaceodyssey.core.enums import Audience, EventKind
from spaceodyssey.core.records import payload_columns

# Trace fields that carry the observer belief (the layout's projection); the rest are
# actions and event order.
BELIEF_FIELDS = PROJECTION_FIELDS


def _release_steps(carry, trace):
    """(step, slot) for every handled release, with its audience, in step order."""
    log = carry.log
    handled = np.asarray(trace.handled) & (np.asarray(trace.kind) == EventKind.RELEASE)
    out = []
    for i in np.flatnonzero(handled):
        hit = (
            np.asarray(log.valid)
            & (np.asarray(log.released_tick) == int(trace.tick[i]))
            & (np.asarray(log.subject) == int(trace.subject[i]))
            & (np.asarray(log.acquisition_index) == int(trace.acquisition_index[i]))
            & (np.asarray(log.channel) == int(trace.channel[i]))
        )
        if not hit.any():
            # an observer-only release does not set the policy release tick
            hit = (
                np.asarray(log.valid)
                & (np.asarray(log.accepted_tick) == int(trace.tick[i]))
                & (np.asarray(log.subject) == int(trace.subject[i]))
                & (np.asarray(log.acquisition_index) == int(trace.acquisition_index[i]))
                & (np.asarray(log.channel) == int(trace.channel[i]))
            )
        slot = int(np.argmax(hit)) if hit.any() else -1
        out.append((int(i), slot, int(trace.audience[i])))
    return out


def first_truth_dependent_release(run_a, run_b, caps):
    """Steps of the first released record whose payload differs between two runs.

    Returns (policy_step, observer_step) for one world: the first step that releases a
    differing masked payload to the policy and the observer, respectively.
    A step is None when no released record differs.
    """
    (ca, ta), (cb, tb) = run_a, run_b
    steps_b = {s: slot for s, slot, _ in _release_steps(cb, tb)}
    policy_step = any_step = None
    for step, slot, audience in _release_steps(ca, ta):
        other = steps_b.get(step, -1)

        def differs(reader, slot=slot, other=other):
            if slot < 0 or other < 0:
                return True
            left = np.where(
                payload_columns(caps, ca.log.kind[slot], reader),
                ca.log.payload[slot],
                0.0,
            )
            right = np.where(
                payload_columns(caps, cb.log.kind[other], reader),
                cb.log.payload[other],
                0.0,
            )
            return not bits_equal(left, right)

        if (
            any_step is None
            and audience & Audience.OBSERVER
            and differs(Audience.OBSERVER)
        ):
            any_step = step
        if (
            policy_step is None
            and audience & Audience.POLICY
            and differs(Audience.POLICY)
        ):
            policy_step = step
        if policy_step is not None and any_step is not None:
            break
    return policy_step, any_step


def _prefix_equal(trace_a, trace_b, steps, skip=()):
    """Names of the trace fields that differ in the first `steps` steps."""
    bad = []
    for name in type(trace_a).__dataclass_fields__:
        if name in skip or (trace_a.program is None and name == "program"):
            continue
        a, b = np.asarray(getattr(trace_a, name)), np.asarray(getattr(trace_b, name))
        if not bits_equal(a[:steps], b[:steps]):
            bad.append(name)
    return bad


def truth_swap(study, truths_a, truths_b, roots, n_steps, *, executor=None):
    """Swap an independent truth in and require identical behavior until it is read.

    Assumptions: `truths_a` and `truths_b` are batches of the same shape and differ
    in a way the records can show, `roots` are shared, and every record that carries
    truth is released at a Release step. Until the first released record that differs
    between the two runs, every trace field (event order, effects, admission codes,
    policy draws) must be bitwise equal, and until the first masked difference reaches
    the observer its belief must be too. The check fails when no record differs,
    since a swap that changes nothing proves nothing.
    """
    runner = default_executor(executor)
    run_a = runner.run_worlds(study, truths_a, roots, n_steps)
    run_b = runner.run_worlds(study, truths_b, roots, n_steps)
    failures = []
    steps = []
    for w, (wa, wb) in enumerate(zip(worlds_of(run_a), worlds_of(run_b), strict=True)):
        policy_step, any_step = first_truth_dependent_release(wa, wb, study.caps)
        if policy_step is None:
            failures.append(
                f"world {w}: no truth-dependent record reached the policy audience"
            )
            continue
        steps.append(policy_step)
        bad = _prefix_equal(wa[1], wb[1], policy_step, skip=BELIEF_FIELDS)
        if bad:
            failures.append(
                f"world {w}: {', '.join(bad)} changed before the first "
                f"truth-dependent release (step {policy_step})"
            )
        bad = _prefix_equal(wa[1], wb[1], any_step)
        if [b for b in bad if b in BELIEF_FIELDS]:
            failures.append(
                f"world {w}: the belief changed before the first "
                f"truth-dependent release (step {any_step})"
            )
    return make_result("truth_swap", failures, first_release_steps=steps)
