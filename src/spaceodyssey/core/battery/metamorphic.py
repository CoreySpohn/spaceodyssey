"""Metamorphic relations: edits that must change nothing, each with its assumption."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from spaceodyssey.core.battery.result import (
    bits_equal,
    default_executor,
    make_result,
    tree_bits_equal,
    worlds_of,
)
from spaceodyssey.core.carry import init_carry
from spaceodyssey.core.enums import I64
from spaceodyssey.core.executors.base import canonical_history
from spaceodyssey.core.transaction import transition

# Float columns of a canonical history that no draw feeds.
DRAW_FREE_FLOATS = ("cost", "charged", "balance")
# Log columns that hold a tick (-1 marks an unset one).
TICK_LOG_FIELDS = (
    "start_tick",
    "end_tick",
    "available_tick",
    "released_tick",
    "accepted_tick",
)


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
    a = runner.run_worlds(study, truths, roots, n_steps)
    b = runner.run_worlds(relabeled, truths, roots, n_steps)
    failures = []
    lookup = np.vectorize(lambda x: channel_map.get(int(x), int(x)), otypes=[np.int64])
    n_records = 0
    for w, ((ca, ta), (cb, tb)) in enumerate(
        zip(worlds_of(a), worlds_of(b), strict=True)
    ):
        ha, hb = canonical_history(ca, ta), canonical_history(cb, tb)
        n_records += ha.n_records
        for name, x in ha.ints.items():
            x = lookup(x) if name == "channel" else x
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


@eqx.filter_jit
def _scan_from(study, truths, roots, delta, n_steps):
    def one(truth, root):
        c0 = _move_start(init_carry(study, truth), delta)
        return lax.scan(
            lambda c, i: transition(c, i, root, study, truth), c0, jnp.arange(n_steps)
        )

    return jax.vmap(one)(truths, roots)


def shift_epoch(study, truths, roots, n_steps, delta_ticks):
    """Start the whole run `delta_ticks` later and require the same run, shifted.

    Assumptions: no component reads an absolute tick (a channel, process, observer or
    policy that does breaks the relation, and the check is expected to fail for it),
    and the study carries its absolute horizon and phase boundaries, which are moved
    by `delta_ticks`. The shifted run starts with its queue and clock moved. Every
    tick in the trace and the records, less `delta_ticks`, must equal the original;
    every other column, belief, ledger and payload must be bitwise equal. The policy
    draw is keyed by its decision tick, an absolute time by design, and is not
    compared.
    """
    zero = jnp.asarray(0, I64)
    base_carry, base_trace = _scan_from(study, truths, roots, zero, n_steps)
    moved = shifted_epoch(study, delta_ticks)
    late_carry, late_trace = _scan_from(
        moved, truths, roots, jnp.asarray(delta_ticks, I64), n_steps
    )
    failures = []
    n_records = 0
    for w in range(int(roots.shape[0])):
        pick = lambda tree, w=w: jax.tree_util.tree_map(lambda x: x[w], tree)  # noqa: E731
        failures += _epoch_world(
            w,
            pick(base_carry),
            pick(base_trace),
            pick(late_carry),
            pick(late_trace),
            delta_ticks,
        )
        n_records += int(np.asarray(base_carry.log.valid[w]).sum())
    if not n_records:
        failures.append("no record was logged, so the check is vacuous")
    return make_result("shift_epoch", failures, n_records=n_records)


def _epoch_world(w, cb, tb, cl, tl, delta):
    out = []
    handled = np.asarray(tb.handled)
    if not bits_equal(handled, np.asarray(tl.handled)):
        return [f"world {w}: the handled steps changed"]
    for name in type(tb).__dataclass_fields__:
        if name == "pol_u":
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
        if name in TICK_LOG_FIELDS:
            b = np.where(np.asarray(cb.log.valid) & (b >= 0), b - delta, b)
        if not bits_equal(a, b):
            out.append(f"world {w}: log column {name} changed under the shift")
    for name in ("belief", "pbelief", "acquisition_counts"):
        if not tree_bits_equal(getattr(cb, name), getattr(cl, name)):
            out.append(f"world {w}: {name} changed under the shift")
    if not bits_equal(cb.ledger.balance, cl.ledger.balance):
        out.append(f"world {w}: the balance changed under the shift")
    return out


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
        for name in ("belief", "pbelief", "acquisition_counts"):
            if not tree_bits_equal(getattr(ca, name), getattr(cb, name)):
                failures.append(f"world {w}: {name} changed")
        want = np.insert(
            np.asarray(ca.ledger.balance), study.caps.n_assets, 0.0, axis=0
        )
        if not bits_equal(want, cb.ledger.balance):
            failures.append(f"world {w}: the balance changed beyond the new row")
    if not n_records:
        failures.append("no record was logged, so the check is vacuous")
    return make_result("zero_time_asset", failures, n_records=n_records)
