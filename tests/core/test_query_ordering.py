"""A request and its result form a barrier before sibling effects run."""

import dataclasses as dc

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey import core
from spaceodyssey.core.events import empty_event_row
from spaceodyssey.studies import oracle, toy
from tests.core.helpers import one_world, worlds


class QueryWithSibling(core.AbstractPolicy):
    stop: bool
    query_first: bool

    def decide(self, pstate, view, study, key):
        effects = core.effects_none(study.caps)
        sibling = jnp.where(self.stop, core.EffectKind.STOP, core.EffectKind.WAIT)
        kinds = jnp.asarray([core.EffectKind.QUERY, sibling])
        kinds = jnp.where(self.query_first, kinds, kinds[::-1])
        effects = dc.replace(
            effects,
            kind=jnp.where(view.n_dec == 0, jnp.asarray(kinds), effects.kind),
            tick=jnp.full(study.caps.n_effects, view.tick),
            scope=jnp.full(study.caps.n_effects, int(core.Scope.CAMPAIGN)),
            disposition=jnp.full(study.caps.n_effects, int(core.Disposition.DRAIN)),
        )
        return effects, pstate, jnp.asarray(0.0)


@pytest.mark.parametrize("stop", [False, True])
@pytest.mark.parametrize("query_first", [False, True])
@pytest.mark.parametrize("chunk", [None, 1, 3])
def test_result_commits_before_sibling_effects(stop, query_first, chunk):
    registry, kind = oracle.oracle_registry()
    study = toy.survey_1(
        caps=registry.lower(), policies=(QueryWithSibling(stop, query_first),)
    )
    truths, roots = worlds(1)
    service = oracle.ToyOracleService(kind)
    if chunk is None:
        run = core.HostExecutor().run(
            study, core.index_world(truths, 0), roots[0], 8, service=service
        )
        carry, trace = run
    else:
        run = core.CompiledExecutor().run(
            study, truths, roots, 8, service=service, chunk=chunk
        )
        carry, trace = core.index_world(run.carry, 0), core.index_world(run.trace, 0)
    handled = np.asarray(trace.handled)
    kinds = np.asarray(trace.kind)[handled].tolist()
    sibling = core.EventKind.STOP if stop else core.EventKind.WAIT
    assert kinds == [
        core.EventKind.WAIT,
        core.EventKind.REQUEST,
        core.EventKind.RESULT,
        sibling,
    ]
    assert int(carry.log.released_tick[0]) == 0
    assert int(core.snapshot_view(carry, core.Audience.POLICY, study.caps).n_rows) == 1
    assert not bool(carry.request.valid)
    assert not bool(carry.violation)
    assert int(core.run_status(carry)) == (
        core.RunStatus.COMPLETE if stop else core.RunStatus.IDLE
    )
    positions = list(
        zip(
            *(
                np.asarray(getattr(trace, name))[handled].tolist()
                for name in ("tick", "stage", "micro", "tie", "sid")
            ),
            strict=True,
        )
    )
    assert positions == sorted(positions)
    assert positions[1][:2] == positions[2][:2]
    assert positions[2][2] == positions[1][2] + 1


@pytest.mark.parametrize("pending", [False, True])
def test_campaign_drain_waits_for_host_answer(pending):
    study, service = oracle.oracle_study(query_at=0, plan=())
    truth, root = one_world()
    carry = eqx.filter_jit(core.init_carry)(study, truth)
    carry = eqx.tree_at(lambda c: c.ev.valid, carry, jnp.zeros_like(carry.ev.valid))
    view = core.snapshot_view(carry, core.Audience.POLICY, study.caps)
    carry = core.enqueue(
        carry,
        True,
        kind=core.EventKind.REQUEST,
        tick=carry.tick,
        stage=core.Stage.ADMIT,
        audience=core.Audience.POLICY,
        snapshot_fp=eqx.filter_jit(core.snapshot_fingerprint)(
            view, study.observer.layout(study.caps)
        ),
        decision=view.n_dec,
        version=view.release_version,
        caps=study.caps,
    )
    carry = dc.replace(carry, draining=jnp.asarray(True))
    advance = eqx.filter_jit(core.advance)
    carry, trace, _ = advance(carry, root, study, truth)
    assert int(trace.kind) == core.EventKind.REQUEST
    assert not bool(carry.done)
    assert int(core.run_status(carry)) == core.RunStatus.AWAITING
    if pending:
        carry, trace, wake = advance(carry, root, study, truth)
        assert not bool(trace.handled) and not bool(wake)
        assert not bool(carry.done)
    carry = core.answer_pending(carry, study, service)
    carry, trace, _ = advance(carry, root, study, truth)
    assert int(trace.kind) == core.EventKind.RESULT
    assert int(carry.log.released_tick[0]) == 0
    assert int(core.run_status(carry)) == core.RunStatus.COMPLETE


def test_campaign_drain_waits_for_queued_request():
    study, _ = oracle.oracle_study(query_at=0, plan=())
    truth, root = one_world()
    carry = eqx.filter_jit(core.init_carry)(study, truth)
    carry = dc.replace(carry, draining=jnp.asarray(True), request=empty_event_row())
    carry = core.enqueue(
        carry,
        True,
        kind=core.EventKind.REQUEST,
        tick=1,
        stage=core.Stage.ADMIT,
        audience=core.Audience.POLICY,
        caps=study.caps,
    )
    carry, trace, _ = eqx.filter_jit(core.advance)(carry, root, study, truth)
    assert int(trace.kind) == core.EventKind.WAIT
    assert not bool(carry.done)


class TwoQueries(core.AbstractPolicy):
    def decide(self, pstate, view, study, key):
        effects = core.effects_none(study.caps)
        effects = dc.replace(
            effects,
            kind=jnp.where(
                view.n_dec == 0,
                core.EffectKind.QUERY,
                effects.kind,
            ),
        )
        return effects, pstate, jnp.asarray(0.0)


@pytest.mark.parametrize("chunk", [None, 1, 3])
def test_only_one_query_is_admitted_per_decision(chunk):
    registry, kind = oracle.oracle_registry()
    study = toy.survey_1(caps=registry.lower(), policies=(TwoQueries(),))
    truth, root = one_world()
    carry = eqx.filter_jit(core.init_carry)(study, truth)
    view = core.snapshot_view(carry, core.Audience.POLICY, study.caps)
    effects, _, _ = study.agenda.policies[0].decide(carry.pstate, view, study, root)
    _, codes = eqx.filter_jit(core.admit)(carry, effects, study)
    assert np.asarray(codes).tolist() == [
        core.AdmissionCode.OK,
        core.AdmissionCode.NO_SLOT,
    ]
    service = oracle.ToyOracleService(kind)
    if chunk is None:
        run = core.HostExecutor().run(study, truth, root, 8, service=service)
        carry, trace = run
    else:
        truths, roots = worlds(1)
        run = core.CompiledExecutor().run(
            study, truths, roots, 8, service=service, chunk=chunk
        )
        carry, trace = core.index_world(run.carry, 0), core.index_world(run.trace, 0)
    kinds = np.asarray(trace.kind)[np.asarray(trace.handled)].tolist()
    assert kinds == [core.EventKind.WAIT, core.EventKind.REQUEST, core.EventKind.RESULT]
    assert len(service.requests) == 1
    assert int(carry.log.released_tick[0]) == 0
    assert int(core.run_status(carry)) == core.RunStatus.IDLE


def test_result_precedes_queued_peers_without_reordering_them():
    study, service = oracle.oracle_study(query_at=0, plan=())
    truth, root = one_world()
    carry = eqx.filter_jit(core.init_carry)(study, truth)
    carry = eqx.tree_at(lambda c: c.ev.valid, carry, jnp.zeros_like(carry.ev.valid))
    view = core.snapshot_view(carry, core.Audience.POLICY, study.caps)
    request = dc.replace(
        empty_event_row(),
        valid=jnp.asarray(True),
        tick=carry.tick,
        stage=jnp.asarray(int(core.Stage.ADMIT)),
        micro=jnp.asarray(1),
        audience=jnp.asarray(int(core.Audience.POLICY)),
        snapshot_fp=eqx.filter_jit(core.snapshot_fingerprint)(
            view, study.observer.layout(study.caps)
        ),
        decision=view.n_dec,
        version=view.release_version,
    )
    carry = dc.replace(carry, stage=request.stage, micro=request.micro, request=request)
    for subject in (1, 2):
        carry = core.enqueue(
            carry,
            True,
            kind=core.EventKind.WAIT,
            tick=carry.tick,
            stage=carry.stage,
            subject=subject,
            audience=core.Audience.POLICY,
            caps=study.caps,
        )
    carry = eqx.tree_at(
        lambda c: c.ev.micro, carry, carry.ev.micro.at[0].set(1).at[1].set(2)
    )
    carry = core.answer_pending(carry, study, service)
    handled = []
    for _ in range(3):
        carry, trace, _ = eqx.filter_jit(core.advance)(carry, root, study, truth)
        handled.append((int(trace.kind), int(trace.subject), int(trace.micro)))
    assert handled == [
        (core.EventKind.RESULT, 0, 2),
        (core.EventKind.WAIT, 1, 3),
        (core.EventKind.WAIT, 2, 4),
    ]
