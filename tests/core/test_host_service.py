"""Host protocol: a Query becomes a request, the host answers, a Result commits."""

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey import core
from spaceodyssey.core import (
    AdmissionCode,
    Audience,
    CompiledExecutor,
    EffectKind,
    EventKind,
    HostExecutor,
    RunStatus,
    Stage,
    canonical_history,
    effect_mask,
    histories_equal,
    index_world,
    replay_belief,
    snapshot_view,
)
from spaceodyssey.core.snapshot import EffectTable
from spaceodyssey.studies import oracle, toy
from tests.core.helpers import worlds

N_STEPS = 40
ADMIT, REQUEST, RESULT = Stage.ADMIT, EventKind.REQUEST, EventKind.RESULT


@functools.cache
def _setup():
    study, service = oracle.oracle_study()
    return study, service.kind


def _service():
    return oracle.ToyOracleService(_setup()[1])


@functools.cache
def _host(w=0):
    study, _ = _setup()
    truths, roots = worlds(w + 1)
    service = _service()
    carry, trace = HostExecutor().run(
        study, index_world(truths, w), roots[w], N_STEPS, service=service
    )
    return carry, trace, service


def _history(carry, trace):
    study, _ = _setup()
    return canonical_history(carry, trace, study.observer.layout(study.caps))


def _query_effects(caps, subject=1, service=0):
    none = jnp.full(caps.n_effects, EffectKind.NONE, jnp.int64)
    z = jnp.zeros(caps.n_effects, jnp.int64)
    return EffectTable(
        kind=none.at[0].set(EffectKind.QUERY),
        subject=z.at[0].set(subject),
        channel_index=z.at[0].set(service),
        group=z,
        tick=z,
        lane=z,
        scope=z,
        disposition=z,
        disruptive=jnp.zeros(caps.n_effects, jnp.bool_),
    )


admit_j = eqx.filter_jit(core.admit)


def _fresh_carry(study):
    truths, _ = worlds(1)
    return eqx.filter_jit(core.init_carry)(study, index_world(truths, 0))


def test_query_admission_queues_a_request_row():
    study, _ = _setup()
    c0 = _fresh_carry(study)
    c1, code = admit_j(c0, _query_effects(study.caps), study, jnp.asarray(77))
    assert int(code[0]) == AdmissionCode.OK
    rows = np.flatnonzero(np.asarray(c1.ev.valid & (c1.ev.kind == REQUEST)))
    assert len(rows) == 1
    ev = jax.tree_util.tree_map(lambda a: a[rows[0]], c1.ev)
    assert int(ev.audience) == Audience.POLICY and int(ev.subject) == 1
    assert int(ev.tick) == int(c0.tick) and int(ev.stage) == ADMIT
    assert int(ev.snapshot_fp) == 77
    # nothing is charged or counted for a question
    assert np.array_equal(c1.ledger.balance, c0.ledger.balance)
    assert np.array_equal(c1.acquisition_counts, c0.acquisition_counts)
    assert not bool(c1.overflow)


def test_query_with_a_full_queue_is_refused_without_overflow():
    study, _ = _setup()
    c0 = _fresh_carry(study)
    full = eqx.tree_at(lambda c: c.ev.valid, c0, jnp.ones_like(c0.ev.valid))
    c1, code = admit_j(full, _query_effects(study.caps), study)
    assert int(code[0]) == AdmissionCode.NO_SLOT
    assert not bool(c1.overflow)


def test_query_forbidden_in_a_phase_that_does_not_permit_it():
    study, _ = oracle.oracle_study(
        permitted_effects=(effect_mask(EffectKind.OBSERVE, EffectKind.WAIT),)
    )
    c0 = _fresh_carry(study)
    c1, code = admit_j(c0, _query_effects(study.caps), study)
    assert int(code[0]) == AdmissionCode.FORBIDDEN
    assert not bool(np.asarray(c1.ev.valid & (c1.ev.kind == REQUEST)).any())


def test_query_result_same_tick():
    carry, trace, service = _host()
    h = _history(carry, trace)
    kinds = [e[2] for e in h.events]
    i = kinds.index(REQUEST)
    # the answer is the very next row, at the request's own tick and stage
    assert h.events[i + 1][2] == RESULT
    assert h.events[i][:2] == h.events[i + 1][:2]
    tick = h.events[i][0]
    assert h.events[i][1] == ADMIT
    assert kinds.count(REQUEST) == 1 and kinds.count(RESULT) == 1
    assert [r.tick for r in service.requests] == [tick]
    # the record is released at that tick, and wakes a decision at it
    log = carry.log
    row = int(np.flatnonzero(np.asarray(log.kind) == _setup()[1])[0])
    assert int(log.released_tick[row]) == tick == int(log.available_tick[row])
    steps = np.asarray(trace.handled) & (np.asarray(trace.kind) == RESULT)
    step = int(np.flatnonzero(steps)[0])
    assert bool(trace.wake[step]) and int(trace.tick[step]) == tick
    # the result and the request name the snapshot of the decision that asked
    asked = int(np.flatnonzero(np.asarray(trace.eff_kind)[:, 0] == EffectKind.QUERY)[0])
    fp = int(trace.snapshot_fp[asked])
    assert fp != 0
    assert int(trace.event_fp[step]) == fp
    assert int(trace.event_fp[step - 1]) == fp
    assert not bool(carry.request.valid)
    # the answer's content is the belief the question was asked on
    assert float(log.payload[row, 1]) == 1.0
    assert not (bool(carry.violation) or bool(carry.dup_fold))
    assert not bool(carry.double_settle)
    assert h.status == RunStatus.COMPLETE


def test_answer_reads_the_snapshot_it_was_asked_on():
    carry, _, service = _host()
    (request,) = service.requests
    first = int(np.flatnonzero(np.asarray(carry.log.channel) == 10)[0])
    # one release had reached the policy when it asked: the answer is that row's value
    row = int(np.flatnonzero(np.asarray(carry.log.kind) == _setup()[1])[0])
    assert float(carry.log.payload[row, 0]) == float(carry.log.payload[first, 0])
    assert float(carry.log.payload[row, 2]) == request.release_version == 1


def test_host_service_parity():
    study, _ = _setup()
    truths, roots = worlds(3)
    for chunk in (1, 32):
        service = _service()
        run = CompiledExecutor().run(
            study, truths, roots, N_STEPS, service=service, chunk=chunk
        )
        assert np.all(np.asarray(run.status) == RunStatus.COMPLETE)
        assert len(service.requests) == 3
        for w in range(3):
            carry, trace, _ = _host(w)
            host, comp = _history(carry, trace), run.history(w)
            assert histories_equal(host, comp).ok, (chunk, w)
            assert host.events == comp.events
            for name in ("snapshot_fp", "event_fp", "release_version"):
                assert np.array_equal(host.ints[name], comp.ints[name]), (chunk, name)
            assert not (comp.flags[4] or comp.double_settle)
            rep = replay_belief(
                study, index_world(run.carry, w), index_world(run.trace, w)
            )
            assert np.array_equal(rep.mean, carry.belief.mean)


def test_each_world_is_answered_from_its_own_snapshot():
    answers = [_host(w)[0] for w in range(3)]
    kind = _setup()[1]
    means = []
    for carry in answers:
        row = int(np.flatnonzero(np.asarray(carry.log.kind) == kind)[0])
        means.append(float(carry.log.payload[row, 0]))
    assert len(set(means)) == 3


@pytest.mark.parametrize("n_steps", [5, 6, 9])
def test_a_truncated_hosted_run_agrees_across_executors(n_steps):
    study, _ = _setup()
    truths, roots = worlds(1)
    carry, trace = HostExecutor().run(
        study, index_world(truths, 0), roots[0], n_steps, service=_service()
    )
    run = CompiledExecutor().run(
        study, truths, roots, n_steps, service=_service(), chunk=4
    )
    host, comp = _history(carry, trace), run.history(0)
    assert histories_equal(host, comp).ok
    assert host.status == comp.status == RunStatus.TRUNCATED
    assert host.n_handled == comp.n_handled == n_steps


def test_an_unanswered_request_holds_the_run_and_says_so():
    study, _ = _setup()
    truths, roots = worlds(1)
    carry, trace = HostExecutor().run(study, index_world(truths, 0), roots[0], N_STEPS)
    run = CompiledExecutor().run(study, truths, roots, N_STEPS)
    for c, t in (
        (carry, trace),
        (index_world(run.carry, 0), index_world(run.trace, 0)),
    ):
        h = _history(c, t)
        assert h.status == RunStatus.AWAITING and not h.complete
        assert bool(c.request.valid)
        assert [e[2] for e in h.events][-1] == REQUEST
        assert int(c.request.audience) == Audience.POLICY
    assert histories_equal(_history(carry, trace), run.history(0)).ok


def test_the_answer_reaches_the_audience_of_its_snapshot_only():
    carry, _, _ = _host()
    study, kind = _setup()[0], _setup()[1]
    row = int(np.flatnonzero(np.asarray(carry.log.kind) == kind)[0])
    assert int(carry.log.audience[row]) == Audience.POLICY
    policy = snapshot_view(carry, Audience.POLICY, study.caps)
    observer = snapshot_view(carry, Audience.OBSERVER, study.caps)
    assert kind in np.asarray(policy.log.kind)[: int(policy.n_rows)]
    assert kind not in np.asarray(observer.log.kind)[: int(observer.n_rows)]
    assert int(policy.n_rows) == int(observer.n_rows) + 1
    # the observer folds nothing of it and its release count does not move
    assert int(carry.release_version[0]) == int(observer.n_rows)
    assert int(carry.release_version[1]) == int(policy.n_rows)


def test_a_service_answer_wider_than_the_payload_is_refused():
    class Wide(oracle.ToyOracleService):
        def answer(self, request, view, layout):
            return np.zeros(99)

    study, kind = _setup()
    truths, roots = worlds(1)
    with pytest.raises(ValueError, match="wider"):
        HostExecutor().run(
            study, index_world(truths, 0), roots[0], N_STEPS, service=Wide(kind)
        )


def test_the_core_does_not_name_the_toy_service():
    from pathlib import Path

    for path in Path(core.__file__).parent.rglob("*.py"):
        assert "oracle" not in path.read_text().lower(), path.name
    assert toy and oracle
