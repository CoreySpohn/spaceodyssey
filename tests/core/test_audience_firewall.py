"""Audience boundaries for records, folds, snapshots and host requests."""

import dataclasses as dc

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey import core
from spaceodyssey.core import handlers
from spaceodyssey.core.events import empty_event_row
from spaceodyssey.core.records import record_none, write_log
from spaceodyssey.studies import oracle, toy
from tests.core.helpers import one_world

PUBLIC = core.Audience.OBSERVER | core.Audience.POLICY
init = eqx.filter_jit(core.init_carry)
release_j = eqx.filter_jit(handlers.h_release)
fp_j = eqx.filter_jit(core.snapshot_fingerprint)


def _study():
    return toy.survey_1(policies=(toy.NullPolicy(),))


def _add(
    carry,
    study,
    audience,
    *,
    acquisition=100,
    revision=0,
    status=core.RecordStatus.COMPLETED,
):
    rec = dc.replace(
        record_none(study.caps),
        kind=jnp.asarray(toy.RECORD_MEASUREMENT),
        valid=jnp.asarray(True),
        acquisition_id=jnp.asarray(acquisition),
        revision=jnp.asarray(revision),
        audience=jnp.asarray(int(audience)),
        released_tick=jnp.asarray(-1),
        accepted_tick=jnp.asarray(-1),
        superseded_by=jnp.asarray(-1),
        status=jnp.asarray(int(status)),
        payload=jnp.zeros(study.caps.payload_width)
        .at[:3]
        .set(jnp.asarray([7.0, 123.0, 1.0])),
    )
    return write_log(carry, rec, study.caps)


def _release(carry, study, slot, audience, tick=10):
    truth, root = one_world()
    event = dc.replace(
        empty_event_row(),
        kind=jnp.asarray(int(core.EventKind.RELEASE)),
        valid=jnp.asarray(True),
        tick=jnp.asarray(tick),
        slot=jnp.asarray(slot),
        audience=jnp.asarray(int(audience)),
    )
    return release_j(dc.replace(carry, tick=event.tick), event, study, truth, root)[0]


class RequestService(oracle.ToyOracleService):
    def answer(self, request, view):
        self.requests.append(request)
        return np.asarray([getattr(request, "micro", 0), 0.0, 0.0])


def test_service_request_is_unchanged_by_private_events():
    truth, root = one_world()
    requests, answers = [], []
    for private in (False, True):
        process = toy.HeartbeatProcess(
            period_ticks=130,
            n_fires=int(private),
            audience=core.Audience.OBSERVER,
            writes=True,
        )
        study, template = oracle.oracle_study(
            query_at=1,
            plan=((0, 10),),
            process=process,
        )
        service = RequestService(template.kind)
        run = core.HostExecutor().run(study, truth, root, 16, service=service)
        (request,) = service.requests
        requests.append(request)
        rows = np.flatnonzero(
            np.asarray(run.carry.log.valid & (run.carry.log.kind == service.kind))
        )
        assert len(rows) == 1
        answers.append(float(run.carry.log.payload[rows[0], 0]))
    assert requests[0].snapshot_fp == requests[1].snapshot_fp
    assert requests[0] == requests[1]
    assert answers[0] == answers[1]


def _changed_revisions():
    study = _study()
    truth, _ = one_world()
    carry = init(study, truth)
    for acquisition in (100, 101, 102):
        carry, slot = _add(carry, study, core.Audience.POLICY, acquisition=acquisition)
        carry = _release(carry, study, slot, core.Audience.POLICY)
    view = core.snapshot_view(carry, core.Audience.POLICY, study.caps)
    # Two nonnegative revisions exchange contributions in an additive digest.
    revisions = carry.log.revision.at[jnp.asarray([0, 2])].set(4868853749383748346)
    changed = dc.replace(carry, log=dc.replace(carry.log, revision=revisions))
    return study, view, changed


def test_fingerprint_distinguishes_exchanged_integer_contributions():
    study, before, changed = _changed_revisions()
    after = core.snapshot_view(changed, core.Audience.POLICY, study.caps)
    assert not np.array_equal(before.log.revision, after.log.revision)
    assert int(fp_j(before)) != int(fp_j(after))


def test_pending_request_rejects_changed_visible_revisions():
    study, before, changed = _changed_revisions()
    req = dc.replace(
        empty_event_row(),
        valid=jnp.asarray(True),
        tick=changed.tick,
        stage=jnp.asarray(int(core.Stage.ADMIT)),
        audience=jnp.asarray(int(core.Audience.POLICY)),
        snapshot_fp=fp_j(before),
        version=before.release_version,
        decision=before.n_dec,
    )
    changed = dc.replace(changed, stage=req.stage, request=req)
    service = oracle.ToyOracleService(toy.RECORD_AUXILIARY)
    with pytest.raises(RuntimeError, match="snapshot changed"):
        core.answer_pending(changed, study, service)
    assert not service.requests


class ProvenanceObserver(toy.ToyRunningMean):
    def fold(self, belief, rec, caps):
        folded = super().fold(belief, rec, caps)
        return dc.replace(folded, aux=folded.aux.at[1].set(rec.acquisition_id))


def test_policy_fold_hides_private_event_identity():
    truth, root = one_world()
    values = []
    for private in (False, True):
        process = toy.HeartbeatProcess(
            period_ticks=1,
            n_fires=int(private),
            audience=core.Audience.OBSERVER,
            writes=False,
        )
        study = toy.survey_1(
            observer=ProvenanceObserver(),
            process=process,
            policies=(toy.FixedPlanPolicy(plan=((0, 10),)),),
        )
        run = core.HostExecutor().run(study, truth, root, 10)
        view = core.snapshot_view(run.carry, core.Audience.POLICY, study.caps)
        assert int(view.n_rows) == 1
        values.append(float(view.belief.aux[1]))
        assert values[-1] == int(view.log.acquisition_id[0]) == 0
    assert values[0] == values[1]


class UnreplacedObserver(toy.ToyRunningMean):
    def fold(self, belief, rec, caps):
        return super().fold(
            belief, dc.replace(rec, valid=rec.valid & (rec.superseded_by < 0)), caps
        )


def test_policy_fold_ignores_observer_only_supersession():
    study = dc.replace(_study(), observer=UnreplacedObserver())
    truth, _ = one_world()
    carry, slot = _add(init(study, truth), study, PUBLIC)
    carry = _release(carry, study, slot, core.Audience.OBSERVER)
    carry, private_slot = _add(carry, study, core.Audience.OBSERVER, revision=1)
    carry = _release(carry, study, private_slot, core.Audience.OBSERVER, tick=20)
    carry = _release(carry, study, slot, core.Audience.POLICY, tick=30)
    view = core.snapshot_view(carry, core.Audience.POLICY, study.caps)
    assert int(view.log.superseded_by[0]) == -1
    assert int(view.belief.n[0]) == 1
    assert float(view.belief.mean[0]) == 7.0


def test_snapshot_uses_the_belief_of_its_audience():
    measurement = core.RecordKind(
        "measurement",
        (
            core.PayloadColumn("value", core.Audience.POLICY),
            core.PayloadColumn("variate", core.Audience.SCORER),
            core.PayloadColumn("fraction", core.Audience.POLICY),
        ),
    )
    registry = core.KindRegistry(
        (core.RecordKind("none", ()), measurement, toy.AUXILIARY_KIND)
    )
    study = _study()
    study = dc.replace(study, caps=registry.lower(study.caps))
    truth, _ = one_world()
    carry, slot = _add(init(study, truth), study, PUBLIC)
    carry = _release(carry, study, slot, PUBLIC)
    observer = core.snapshot_view(carry, core.Audience.OBSERVER, study.caps)
    policy = core.snapshot_view(carry, core.Audience.POLICY, study.caps)
    assert float(observer.log.payload[0, 0]) == float(carry.belief.mean[0]) == 0.0
    assert float(observer.belief.mean[0]) == 0.0
    assert float(policy.belief.mean[0]) == 7.0


@pytest.mark.parametrize(
    "row_audience,event_audience",
    [
        (core.Audience.OBSERVER, core.Audience.POLICY),
        (core.Audience.POLICY, core.Audience.OBSERVER),
        (core.Audience.OBSERVER, PUBLIC),
        (core.Audience.POLICY, PUBLIC),
    ],
)
def test_release_reaches_only_the_intersection_of_audiences(
    row_audience, event_audience
):
    study = _study()
    truth, _ = one_world()
    carry, slot = _add(init(study, truth), study, row_audience)
    carry = _release(carry, study, slot, event_audience)
    reached = row_audience & event_audience
    for audience, belief, column, version in (
        (core.Audience.OBSERVER, carry.belief, "accepted_tick", 0),
        (core.Audience.POLICY, carry.pbelief, "released_tick", 1),
    ):
        allowed = bool(reached & audience)
        assert int(belief.n[0]) == int(allowed)
        assert int(getattr(carry.log, column)[slot]) == (10 if allowed else -1)
        assert int(carry.release_version[version]) == int(allowed)
        assert int(core.snapshot_view(carry, audience, study.caps).n_rows) == int(
            allowed
        )
    assert bool(carry.woke) == bool(reached & core.Audience.POLICY)


@pytest.mark.parametrize(
    "status,audience",
    [
        (core.RecordStatus.UNRELEASED, core.Audience.POLICY),
        (core.RecordStatus.COMPLETED, core.Audience.OBSERVER),
    ],
)
def test_unreadable_release_does_not_wake_policy(status, audience):
    study = _study()
    truth, root = one_world()
    carry, slot = _add(init(study, truth), study, audience, status=status)
    carry = eqx.tree_at(lambda c: c.ev.valid, carry, jnp.zeros_like(carry.ev.valid))
    carry = core.enqueue(
        carry,
        True,
        kind=core.EventKind.RELEASE,
        tick=100,
        stage=core.Stage.RELEASE,
        slot=slot,
        audience=core.Audience.POLICY,
        caps=study.caps,
    )
    carry, trace, wake = eqx.filter_jit(core.advance)(carry, root, study, truth)
    assert bool(trace.handled)
    assert not bool(wake)
    assert not bool(carry.pending_wake)
    assert int(core.snapshot_view(carry, core.Audience.POLICY, study.caps).n_rows) == 0
    assert int(carry.release_version[1]) == 0
