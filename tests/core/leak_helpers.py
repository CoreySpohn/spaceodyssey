import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from spaceodyssey import core
from spaceodyssey import core as sch
from spaceodyssey.core.enums import Audience, EventKind, Purpose
from spaceodyssey.core.executors.base import index_world
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.studies import toy as kinds
from tests.core.corpus import leak_study

ROOT0 = jr.PRNGKey(0)


def make_worlds(n, caps, root=ROOT0):
    roots = jax.vmap(lambda i: jr.fold_in(root, i))(jnp.arange(n))
    truths = jax.vmap(lambda r: kinds.draw_truth(jr.fold_in(r, Purpose.WORLD), caps))(
        roots
    )
    return truths, roots


@dataclasses.dataclass
class LeakResult:
    wakes: int
    wakes_baseline: int
    decisions: int
    n_private_events: int
    coincident_present: bool
    views_equal: bool


def _host_run(process, wake_fn, n_worlds, n_steps, executor):
    study = leak_study(process)
    truths, roots = make_worlds(n_worlds, study.caps)
    out = []
    for w in range(n_worlds):
        views = []
        carry, trace = executor().run(
            study,
            index_world(truths, w),
            roots[w],
            n_steps,
            wake_fn=wake_fn,
            view_log=views,
        )
        out.append((carry, trace, views))
    return out


def _views_match(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=False):
        r = sch.tree_equal_report(x, y)
        if not (r["structure_equal"] and r["shapes_equal"]):
            return False
        if r["max_int_diff"] != 0 or r["max_abs"] != 0.0:
            return False
    return True


def run_leak(process, wake_fn, n_worlds=4, n_steps=60, *, executor=HostExecutor):
    """Compare wakes and views against a no-process baseline on the same executor.

    The optional executor factory lets historical per-event controls use the same
    scenarios without changing the current transaction's coalescing rule.
    """
    base = _host_run(
        kinds.NullProcess(), core.wake_by_audience, n_worlds, n_steps, executor
    )
    runs = _host_run(process, wake_fn, n_worlds, n_steps, executor)
    wakes = sum(int(np.asarray(t.wake).sum()) for _, t, _ in runs)
    wakes_base = sum(int(np.asarray(t.wake).sum()) for _, t, _ in base)
    n_private, coincident = 0, False
    for _, t, _ in runs:
        h = np.asarray(t.handled)
        aud, kind, tick = (np.asarray(x)[h] for x in (t.audience, t.kind, t.tick))
        n_private += int(((aud & Audience.POLICY) == 0).sum())
        proc_ticks = set(tick[kind == EventKind.PROCESS].tolist())
        public_rel = (kind == EventKind.RELEASE) & ((aud & Audience.POLICY) != 0)
        coincident |= bool(proc_ticks & set(tick[public_rel].tolist()))
    return LeakResult(
        wakes=wakes,
        wakes_baseline=wakes_base,
        decisions=sum(len(v) for _, _, v in runs),
        n_private_events=n_private,
        coincident_present=coincident,
        views_equal=all(
            _views_match(v, vb)
            for (_, _, v), (_, _, vb) in zip(runs, base, strict=False)
        ),
    )


def leaky_wake(carry, ev):
    """Negative control: wakes the policy whatever the event's audience."""
    return carry.woke & ~carry.done & ~carry.draining


def historical_leaky_wake(carry, ev):
    """Negative control: every release wakes, including observer-only releases.

    Paired with the per-event executor this restores the historical control before
    coalescing and release-side audience filtering each independently blocked it.
    """
    return (carry.woke | (ev.kind == EventKind.RELEASE)) & ~carry.done & ~carry.draining
