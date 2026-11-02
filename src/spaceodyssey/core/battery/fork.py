"""Forks: a counterfactual branch shares a prefix, and later draws only as declared."""

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from spaceodyssey.core.battery.result import (
    bits_equal,
    default_executor,
    log_index,
    make_result,
    worlds_of,
)
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import BIG, F64, EventKind, Purpose, RecordStatus
from spaceodyssey.core.keys import Coupling, Fork, draw_key, policy_key
from spaceodyssey.core.records import Record, record_none

# Purposes addressed by record identity; a policy draw is keyed by decision instead.
RECORD_PURPOSES = (
    Purpose.WORLD,
    Purpose.MEASUREMENT,
    Purpose.INFERENCE,
    Purpose.PROCESS,
)


def forked(study, at, branch=1, coupling=None):
    """`study` with a fork at acquisition index `at` onto `branch` and a coupling."""
    out = dataclasses.replace(study, fork=Fork(from_index=at, branch=branch))
    return out if coupling is None else dataclasses.replace(out, coupling=coupling)


def _shares(coupling, purpose, *coordinates):
    return set(coordinates) <= coupling.shared_for(purpose)


def key_report(coupling, *, record_key=draw_key, decision_key=policy_key, branch=1):
    """Which purposes share a draw between the trunk and a fork, by the keys.

    Returns {purpose name: (observed shared, declared shared)}. A record-addressed
    purpose forks onto `branch` and `lineage` together; a policy draw has no branch
    coordinate, so its fork is a lineage. `record_key` and `decision_key` are the key
    functions under test, with the signatures of `draw_key` and `policy_key`.
    """
    rec = dataclasses.replace(
        record_none(Capacities()),
        subject=jnp.asarray(1),
        channel=jnp.asarray(10),
        acquisition_index=jnp.asarray(3),
    )
    root = jr.PRNGKey(0)
    out = {}
    for purpose in RECORD_PURPOSES:
        trunk = record_key(root, rec, purpose, coupling=coupling)
        fork = record_key(
            root, rec, purpose, branch=branch, lineage=branch, coupling=coupling
        )
        out[Purpose(purpose).name] = (
            bits_equal(trunk, fork),
            _shares(coupling, purpose, "branch", "lineage"),
        )
    trunk = decision_key(root, 0, 130, 0, coupling=coupling)
    fork = decision_key(root, 0, 130, 0, lineage=branch, coupling=coupling)
    out["POLICY"] = (
        bits_equal(trunk, fork),
        _shares(coupling, Purpose.POLICY, "lineage"),
    )
    return out


def _log_record(log, i):
    return Record(
        **{
            f.name: jnp.asarray(getattr(log, f.name)[i])
            for f in dataclasses.fields(Record)
        }
    )


def _keyed_draw(study, truth, root, log, i, branch):
    """The (value, variate) pair a completed record must hold, from its coordinates."""
    rec = _log_record(log, i)
    coords = dataclasses.replace(rec, status=jnp.asarray(RecordStatus.COMPLETED))
    key = draw_key(
        root, coords, Purpose.MEASUREMENT, branch=branch, coupling=study.coupling
    )
    value, variate = study.channel_model.realize(
        truth,
        rec.subject,
        rec.channel_index,
        jnp.asarray(1.0, F64),
        key,
        study.design,
        study.channels,
    )
    return np.asarray([value, variate])


def fork_check(
    study,
    truths,
    roots,
    n_steps,
    *,
    at=2,
    branch=1,
    coupling=None,
    executor=None,
    record_key=draw_key,
    decision_key=policy_key,
):
    """Fork at acquisition index `at` and check prefix identity and declared sharing.

    Assumptions: `study` has no fork of its own, completes at least `at + 2`
    acquisitions of one subject in every world, never cancels or splits one, and its
    channel model draws only from the key it is handed. The trunk runs unforked and
    the fork runs with every acquisition from `at` on branch `branch`, both under
    `coupling` (default: nothing shared). Required: the trace and the records before
    the first fork-side settlement are bitwise equal on both sides (common prefix);
    every trunk record, including those after `at`, equals the draw recomputed from
    its coordinates, so the fork disturbed none of them; the forked records from `at`
    on equal the draws of the forked branch, and equal the trunk's exactly when the
    coupling shares `branch` for the measurement purpose; and `key_report` agrees with
    the coupling for every purpose, policy included.
    """
    coupling = Coupling() if coupling is None else coupling
    runner = default_executor(executor)
    trunk_study = forked(study, BIG, 0, coupling)
    trunk = runner.run_worlds(trunk_study, truths, roots, n_steps)
    side = runner.run_worlds(
        forked(study, at, branch, coupling), truths, roots, n_steps
    )
    failures = []
    counts = {"prefix": 0, "after": 0}
    for w, (a, b) in enumerate(zip(worlds_of(trunk), worlds_of(side), strict=True)):
        truth = jax.tree_util.tree_map(lambda x, w=w: x[w], truths)
        failures += _trace_prefix(w, a[1], b[1], at)
        failures += _records(
            w, trunk_study, truth, roots[w], a[0], b[0], at, branch, counts
        )
    if not counts["prefix"] or not counts["after"]:
        failures.append(
            f"too few records on either side of the fork ({counts}), so the check "
            "is vacuous"
        )
    report = key_report(
        coupling, record_key=record_key, decision_key=decision_key, branch=branch
    )
    for name, (seen, want) in report.items():
        if seen != want:
            failures.append(
                f"{name} draws are {'shared' if seen else 'separate'} across the fork "
                f"but the coupling declares {'shared' if want else 'separate'}"
            )
    return make_result(
        "fork", failures, counts=counts, key_report=report, coupling=coupling
    )


def _trace_prefix(w, trunk, side, at):
    """The trace before the first fork-side settlement is identical on both sides."""
    after = (
        np.asarray(trunk.handled)
        & (np.asarray(trunk.kind) == EventKind.DONE)
        & (np.asarray(trunk.acquisition_index) >= at)
    )
    first = int(np.argmax(after)) if after.any() else np.asarray(trunk.handled).size
    return [
        f"world {w}: trace field {name} differs before the fork"
        for name in type(trunk).__dataclass_fields__
        if not bits_equal(
            np.asarray(getattr(trunk, name))[:first],
            np.asarray(getattr(side, name))[:first],
        )
    ]


def _records(w, study, truth, root, trunk, side, at, branch, counts):
    shares = _shares(study.coupling, Purpose.MEASUREMENT, "branch")
    out = []
    on_side = log_index(side)
    pay_t, pay_s = np.asarray(trunk.log.payload), np.asarray(side.log.payload)
    for key, i in log_index(trunk).items():
        if key[4] != RecordStatus.COMPLETED:
            continue
        index = key[1]
        want = _keyed_draw(study, truth, root, trunk.log, i, 0)
        if not bits_equal(pay_t[i][:2], want):
            out.append(f"world {w}: trunk record {key} is not the draw of its keys")
        j = on_side.get(key)
        if j is None:
            out.append(f"world {w}: record {key} is missing on the fork")
            continue
        same = bits_equal(pay_t[i], pay_s[j])
        if index < at:
            counts["prefix"] += 1
            if not same:
                out.append(f"world {w}: prefix record {key} differs across the fork")
            continue
        counts["after"] += 1
        if same != shares:
            out.append(
                f"world {w}: record {key} is "
                f"{'shared' if same else 'separate'} across the fork but the coupling "
                f"declares {'shared' if shares else 'separate'}"
            )
        want = _keyed_draw(study, truth, root, side.log, j, branch)
        if not bits_equal(pay_s[j][:2], want):
            out.append(f"world {w}: forked record {key} is not its keyed draw")
    return out
