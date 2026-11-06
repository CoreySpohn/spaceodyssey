"""Shared builders for the core and executor tests (not a test file)."""

import dataclasses
import functools
import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from spaceodyssey import core
from spaceodyssey import core as sch
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import Audience, EffectKind, EventKind, Purpose, Stage
from spaceodyssey.core.executors.base import (
    index_world,
)
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.studies import toy as kinds
from tests.core.corpus import HAND_STUDIES, frozen

CAPS = Capacities()
ROOT0 = jr.PRNGKey(0)


@functools.cache
def worlds(n, caps=CAPS, *, seed=0):
    roots = jax.vmap(lambda i: jr.fold_in(jr.PRNGKey(seed), i))(jnp.arange(n))
    truths = jax.vmap(lambda r: kinds.draw_truth(jr.fold_in(r, Purpose.WORLD), caps))(
        roots
    )
    return truths, roots


def one_world(w=0, caps=CAPS):
    truths, roots = worlds(w + 1, caps)
    return index_world(truths, w), roots[w]


def private_channel_study(plan, channel_index=2, **kw):
    """Build a study with one observer-only channel."""
    base = kinds.survey_1(policies=(frozen(plan=plan),), **kw)
    aud = base.channels.audience.at[channel_index].set(Audience.OBSERVER)
    return eqx.tree_at(lambda s: s.channels.audience, base, aud)


def history_of(study, n_steps=40, w=0):
    truths, roots = worlds(w + 1)
    return CompiledExecutor().run(study, truths, roots, n_steps).history(w)


@functools.cache
def hand_run(name):
    """Share hand-study runs; the phase boundary checks also exercise batching."""
    n_worlds = 2 if name == "phase_switch" else 1
    return CompiledExecutor().run(HAND_STUDIES[name](), *worlds(n_worlds), 40)


@functools.cache
def baseline_host_run():
    """The baseline host run and its decision views, shared by record checks."""
    study = kinds.survey_1()
    truth, root = one_world()
    views = []
    run = HostExecutor().run(study, truth, root, 40, view_log=views)
    return study, run, views


transition_j = eqx.filter_jit(core.transition)
advance_j = eqx.filter_jit(core.advance)
admit_j = eqx.filter_jit(core.admit)


def run_until(study, pred, truth=None, root=None, max_steps=60):
    """Step the shared transition until `pred(carry)` holds; returns the carry."""
    if truth is None:
        truth, root = one_world()
    c = eqx.filter_jit(core.init_carry)(study, truth)
    for i in range(max_steps):
        if bool(pred(c)):
            return c
        # Keep the loop index dynamic so all steps reuse one compiled transition.
        c, _ = transition_j(c, jnp.asarray(i), root, study, truth)
    raise AssertionError("predicate never held")


def continue_until(c, study, truth, root, pred, max_steps=60):
    """Continue stepping an existing carry until `pred(carry)` holds."""
    for i in range(max_steps):
        if bool(pred(c)):
            return c
        c, _ = transition_j(c, jnp.asarray(i), root, study, truth)
    raise AssertionError("predicate never held")


@functools.cache
def _default_bundle_state():
    return after_decision_1_bundle(kinds.survey_1(policies=(frozen(bundle_at=1),)))


def after_decision_1_bundle(study=None):
    """State just after the 185 Release: lane 0 running subject 1, lane 2 done."""
    if study is None:
        return _default_bundle_state()
    c = run_until(
        study, lambda c: (c.tick == 185) & (c.stage == Stage.ADMIT) & (c.n_dec == 3)
    )
    return study, c


@eqx.filter_jit
def _handle_event(c, study, truth, root, kind, fields):
    c = sch.enqueue(
        c, True, kind=kind, tick=c.tick, stage=Stage.ADMIT, caps=study.caps, **fields
    )
    c, ev, _ = sch.select_next(c, study.caps)
    c, _, _ = core.step(c, ev, study, truth, root)
    return c


_EVENT_DEFAULTS = {
    name: parameter.default
    for name, parameter in inspect.signature(sch.enqueue).parameters.items()
    if parameter.default is not inspect.Parameter.empty
}


def handle_event(c, study, truth, root, kind, **fields):
    """Enqueue, select and handle one row with a reusable compiled program."""
    # An omitted default and an explicit value must have the same pytree signature.
    fields = {
        name: jnp.asarray(value, jnp.int64)
        for name, value in (_EVENT_DEFAULTS | fields).items()
    }
    return _handle_event(c, study, truth, root, jnp.asarray(kind, jnp.int64), fields)


def pending_done_row(c, lane):
    m = c.ev.valid & (c.ev.kind == EventKind.DONE) & (c.ev.lane == lane)
    i = jnp.argmax(m)
    row = jax.tree_util.tree_map(lambda a: a[i], c.ev)
    return dataclasses.replace(row, valid=row.valid & m.any())


def log_row(c, i):
    return jax.tree_util.tree_map(lambda a: a[i], c.log)


def observe_effect(subject, channel_id, study=None, caps=CAPS):
    channel_index = {10: 0, 11: 1, 12: 2}[channel_id]
    e = sch.effects_none(caps)
    return dataclasses.replace(
        e,
        kind=e.kind.at[0].set(EffectKind.OBSERVE),
        subject=e.subject.at[0].set(subject),
        channel_index=e.channel_index.at[0].set(channel_index),
    )


def effects_of(kind, caps=CAPS):
    e = sch.effects_none(caps)
    return dataclasses.replace(e, kind=e.kind.at[0].set(kind))


def run_until_quiet(c, study, truth, root, max_steps=60):
    for i in range(max_steps):
        c, _ = transition_j(c, jnp.asarray(i), root, study, truth)
    return c


N_STEPS_CORPUS = 40


def service_of(case):
    """The keyword that hands a case its host service, empty when it has none."""
    return {} if case.make_service is None else {"service": case.make_service()}
