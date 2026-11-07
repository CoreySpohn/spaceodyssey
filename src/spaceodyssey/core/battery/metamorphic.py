"""Metamorphic relations: edits that must change nothing, each with its assumption."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.battery.result import (
    bits_equal,
    default_executor,
    make_result,
    tree_bits_equal,
    worlds_of,
)
from spaceodyssey.core.battery.views import DecisionViews, compare_views, moved_ticks
from spaceodyssey.core.enums import I64
from spaceodyssey.core.executors.base import belief_leaves, canonical_history
from spaceodyssey.core.kinds import AbstractProcess

# Float columns of a canonical history that no draw feeds.
DRAW_FREE_FLOATS = ("cost", "charged", "balance")
# Digests of a snapshot or event. They hash ticks and ids, which a relation moves by
# design, and the fields they summarize are compared directly.
DIGEST_FIELDS = ("snapshot_fp", "event_fp")
# Integer fields that hold a stable channel id, which a relabeling maps.
CHANNEL_ID_FIELDS = ("channel", "source")


def relabel_ids(
    study, relabeled, channel_map, truths, roots, n_steps, *, executor=None
):
    """Relabel stable channel ids and require the same run under the new names.

    Assumptions: `relabeled` is `study` with every channel id replaced through
    `channel_map` (old id to new id) and every reference to an id inside a component,
    such as a plan, replaced the same way; nothing orders or compares ids by value.
    Draws are keyed by stable ids, so the variates, and with them payloads and
    beliefs, move to those of the new names and are not compared. Everything else
    must be equal: event order, admissions, records (channel ids mapped back through
    `channel_map`), costs, charges and the ledger.
    """
    if all(old == new for old, new in channel_map.items()):
        raise ValueError("the relabeling changes no id")
    runner = default_executor(executor)
    layout = study.observer.layout(study.caps)
    va, vb = DecisionViews(), DecisionViews()
    a = runner.run_worlds(va.instrument(study), truths, roots, n_steps)
    b = runner.run_worlds(vb.instrument(relabeled), truths, roots, n_steps)
    failures = []
    lookup = np.vectorize(lambda x: channel_map.get(int(x), int(x)), otypes=[np.int64])
    n_records = 0
    for w, ((ca, ta), (cb, tb)) in enumerate(
        zip(worlds_of(a), worlds_of(b), strict=True)
    ):
        failures += compare_views(
            va.read(study, roots[w], ta),
            vb.read(relabeled, roots[w], tb),
            ca,
            cb,
            study.caps,
            ta,
            layout=layout,
            channel_map=channel_map,
        )
        ha, hb = canonical_history(ca, ta, layout), canonical_history(cb, tb, layout)
        n_records += ha.n_records
        for name, x in ha.ints.items():
            # History keys are bare ("channel") or dotted leaf paths ("trace.channel").
            leaf = name.rsplit(".", 1)[-1]
            if leaf in DIGEST_FIELDS:
                continue
            x = lookup(x) if leaf in CHANNEL_ID_FIELDS else x
            if not bits_equal(x, hb.ints[name]):
                failures.append(f"world {w}: {name} changed under relabeling")
        for name in DRAW_FREE_FLOATS:
            if not bits_equal(ha.floats[name], hb.floats[name]):
                failures.append(f"world {w}: {name} changed under relabeling")
    if not n_records:
        failures.append("no record was logged, so the check is vacuous")
    return make_result("relabel_ids", failures, n_records=n_records)


def shifted_epoch(study, delta_ticks):
    """`study` with its absolute horizon and phase boundaries moved by `delta_ticks`."""
    until = tuple(int(t) + int(delta_ticks) for t in study.agenda.phase_until)
    return dataclasses.replace(
        study,
        horizon_tick=study.horizon_tick + int(delta_ticks),
        agenda=dataclasses.replace(study.agenda, phase_until=until),
    )


def _move_start(carry, delta):
    ev = carry.ev
    tick = jnp.where(ev.valid, ev.tick + delta, ev.tick)
    carry = eqx.tree_at(lambda c: c.ev.tick, carry, tick)
    return eqx.tree_at(lambda c: c.tick, carry, carry.tick + delta)


class _StartedLater(AbstractProcess):
    """The study's process, with the seeded queue and the clock moved by `delta`.

    `init_carry` ends with the process's `seed`, so moving after it moves every event
    queued at the start (the kickoff wait included) and the clock. `delta` is an
    array, so a run and its shifted run share one compiled program per executor.
    """

    inner: AbstractProcess
    delta: jax.Array

    def seed(self, carry, caps):
        """Seed as the inner process does, then move the start."""
        return _move_start(self.inner.seed(carry, caps), self.delta)

    def handle(self, carry, ev, study):
        """Handle as the inner process does."""
        return self.inner.handle(carry, ev, study)


def _started_later(study, delta):
    """`study` with its process wrapped to start `delta` ticks later."""
    later = _StartedLater(inner=study.process, delta=jnp.asarray(delta, I64))
    return dataclasses.replace(study, process=later)


def shift_epoch(study, truths, roots, n_steps, delta_ticks, *, executor=None):
    """Start the whole run `delta_ticks` later and require the same run, shifted.

    Assumptions: no component reads an absolute tick (a channel, process, observer or
    policy that does breaks the relation, and the check is expected to fail for it);
    the study carries its absolute horizon and phase boundaries, which are moved by
    `delta_ticks`; nothing asks the host (no `Query` request, no refresh schedule).
    The shifted run starts with its queue and clock moved, on the named executor (the
    compiled one by default). Every tick in the trace and the records, less
    `delta_ticks`, must equal the original; every other column, belief, ledger and
    payload must be bitwise equal. Tick columns are known by name, not by sign, so an
    observation time before the run starts (negative) moves like any other; only a
    release or acceptance stamp still at -1 (unset) stays. The policy draw is keyed by
    its decision tick, an absolute time by design, and is not compared.

    `delta_ticks` must be positive: a zero shift checks nothing, and a negative one
    could move a stamp onto the unset value -1.
    """
    if int(delta_ticks) <= 0:
        raise ValueError(
            f"the shift must be a positive number of ticks, not {delta_ticks}"
        )
    runner = default_executor(executor)
    layout = study.observer.layout(study.caps)
    va, vb = DecisionViews(), DecisionViews()
    moved = shifted_epoch(study, delta_ticks)
    base = runner.run_worlds(
        _started_later(va.instrument(study), 0), truths, roots, n_steps
    )
    late = runner.run_worlds(
        _started_later(vb.instrument(moved), delta_ticks), truths, roots, n_steps
    )
    failures = []
    n_records = n_before = n_views = 0
    for w, ((cb, tb), (cl, tl)) in enumerate(
        zip(worlds_of(base), worlds_of(late), strict=True)
    ):
        left = va.read(study, roots[w], tb)
        failures += compare_views(
            left,
            vb.read(moved, roots[w], tl),
            cb,
            cl,
            study.caps,
            tb,
            layout=layout,
            epoch=delta_ticks,
        )
        failures += _epoch_world(w, cb, tb, cl, tl, delta_ticks, layout)
        valid = np.asarray(cb.log.valid)
        n_records += int(valid.sum())
        before = (np.asarray(cb.log.start_tick) < 0) | (np.asarray(cb.log.end_tick) < 0)
        n_before += int((valid & before).sum())
        n_views += len(left)
    if not n_records:
        failures.append("no record was logged, so the check is vacuous")
    return make_result(
        "shift_epoch",
        failures,
        n_records=n_records,
        n_before_start=n_before,
        n_views=n_views,
    )


def _epoch_world(w, cb, tb, cl, tl, delta, layout):
    out = []
    handled = np.asarray(tb.handled)
    if not bits_equal(handled, np.asarray(tl.handled)):
        return [f"world {w}: the handled steps changed"]
    for name in type(tb).__dataclass_fields__:
        if name == "pol_u" or name in DIGEST_FIELDS:
            continue
        a, b = np.asarray(getattr(tb, name)), np.asarray(getattr(tl, name))
        if name == "tick":
            ok = bits_equal(a[handled], b[handled] - delta)
        else:
            ok = bits_equal(a, b)
        if not ok:
            out.append(f"world {w}: trace field {name} changed under the shift")
    for name in type(cb.log).__dataclass_fields__:
        a, b = np.asarray(getattr(cb.log, name)), np.asarray(getattr(cl.log, name))
        b = moved_ticks(name, b, cl.log.valid, -delta)
        if not bits_equal(a, b):
            out.append(f"world {w}: log column {name} changed under the shift")
    out += [
        f"world {w}: {name} changed under the shift"
        for name in _changed_beliefs(cb, cl, layout)
    ]
    if not bits_equal(cb.ledger.balance, cl.ledger.balance):
        out.append(f"world {w}: the balance changed under the shift")
    return out


def _changed_beliefs(ca, cb, layout):
    """Comparator leaves of the two beliefs, and the acquisition counts, that differ."""
    la, lb = belief_leaves(ca, layout), belief_leaves(cb, layout)
    changed = [name for name, leaf in la.items() if not bits_equal(leaf, lb[name])]
    if not bits_equal(ca.acquisition_counts, cb.acquisition_counts):
        changed.append("acquisition_counts")
    return changed


def with_zero_time_asset(study):
    """`study` plus one asset that holds no balance, with one idle lane on it.

    The asset takes the index after the existing assets, so program and shared balance
    rows move down by one and gain nothing.
    """
    caps = study.caps
    n_assets = caps.n_assets
    new_caps = dataclasses.replace(
        caps, n_assets=n_assets + 1, n_lanes=caps.n_lanes + 1
    )
    assets = study.assets
    new_assets = type(assets)(
        lane_asset=jnp.concatenate([assets.lane_asset, jnp.asarray([n_assets], I64)]),
        transition_ticks=jnp.concatenate([assets.transition_ticks, jnp.zeros(1, I64)]),
        freeze_ticks=jnp.concatenate([assets.freeze_ticks, jnp.zeros(1, I64)]),
        disruptive_resource=jnp.concatenate(
            [assets.disruptive_resource, jnp.full(1, -1, I64)]
        ),
    )
    budget = jnp.insert(study.budget, n_assets, 0.0, axis=0)
    return dataclasses.replace(study, caps=new_caps, assets=new_assets, budget=budget)


def zero_time_asset(study, truths, roots, n_steps, *, executor=None):
    """Add an asset with no balance and require the same run.

    Assumptions: the policy's value does not count assets (a component that reads the
    number of assets or lanes breaks the relation, and the check is expected to fail
    for it), and the study keeps its balance as asset rows, then program rows, then
    shared rows. The new asset's row is zero and no channel uses its lane. Every trace
    field, log column, belief, count and the balance (with a zero row inserted for
    the new asset) must be bitwise equal.
    """
    runner = default_executor(executor)
    layout = study.observer.layout(study.caps)
    more = with_zero_time_asset(study)
    a = runner.run_worlds(study, truths, roots, n_steps)
    b = runner.run_worlds(more, truths, roots, n_steps)
    failures = []
    n_records = 0
    for w, ((ca, ta), (cb, tb)) in enumerate(
        zip(worlds_of(a), worlds_of(b), strict=True)
    ):
        n_records += int(np.asarray(ca.log.valid).sum())
        if not tree_bits_equal((ta, ca.log), (tb, cb.log)):
            failures.append(f"world {w}: the trace or the records changed")
        failures += [
            f"world {w}: {name} changed" for name in _changed_beliefs(ca, cb, layout)
        ]
        want = np.insert(
            np.asarray(ca.ledger.balance), study.caps.n_assets, 0.0, axis=0
        )
        if not bits_equal(want, cb.ledger.balance):
            failures.append(f"world {w}: the balance changed beyond the new row")
    if not n_records:
        failures.append("no record was logged, so the check is vacuous")
    return make_result("zero_time_asset", failures, n_records=n_records)
