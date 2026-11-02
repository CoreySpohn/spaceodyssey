"""A2: the host log, fed back, reproduces the observer belief bit for bit."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey import core
from spaceodyssey.core import HostExecutor, MeanBelief, index_world, replay_belief
from spaceodyssey.core.records import write_log
from spaceodyssey.studies import toy
from tests.core.corpus import CORPUS, CORPUS_BY_NAME
from tests.core.helpers import N_STEPS_CORPUS, advance_j, one_world, service_of, worlds

RELEASING = [c for c in CORPUS if not c.name.startswith("overflow")]


def _bits(belief):
    return [np.asarray(x).tobytes() for x in (belief.mean, belief.n, belief.aux)]


@pytest.mark.parametrize("case", RELEASING, ids=lambda c: c.name)
def test_replay_reproduces_belief_bitwise(case):
    study = case.make_study()
    truths, roots = worlds(2)
    for w in range(2):
        carry, trace = HostExecutor().run(
            study, index_world(truths, w), roots[w], N_STEPS_CORPUS, **service_of(case)
        )
        replayed = replay_belief(study, carry, trace)
        assert _bits(replayed) == _bits(carry.belief), (case.name, w)
        assert int(jnp.sum(replayed.n)) >= 1


def test_replay_depends_on_the_log():
    study = CORPUS_BY_NAME["baseline"].make_study()
    truths, roots = worlds(1)
    carry, trace = HostExecutor().run(
        study, index_world(truths, 0), roots[0], N_STEPS_CORPUS
    )
    payload = carry.log.payload.at[0, 0].add(1e-9)
    edited = dataclasses.replace(
        carry, log=dataclasses.replace(carry.log, payload=payload)
    )
    assert _bits(replay_belief(study, edited, trace)) != _bits(carry.belief)


def test_replay_returns_a_belief():
    study = CORPUS_BY_NAME["baseline"].make_study()
    truths, roots = worlds(1)
    carry, trace = HostExecutor().run(
        study, index_world(truths, 0), roots[0], N_STEPS_CORPUS
    )
    assert isinstance(replay_belief(study, carry, trace), MeanBelief)


@pytest.fixture
def reordered_releases():
    study = toy.survey_1(policies=(toy.NullPolicy(),))
    truth, root = one_world()
    carry = core.empty_carry(study.caps)
    for revision, value in ((0, 2.0), (1, 9.0)):
        record = dataclasses.replace(
            core.record_none(study.caps),
            kind=jnp.asarray(toy.RECORD_MEASUREMENT, jnp.int64),
            channel=jnp.asarray(10, jnp.int64),
            acquisition_id=jnp.asarray(7001, jnp.int64),
            revision=jnp.asarray(revision, jnp.int64),
            available_tick=jnp.asarray(10, jnp.int64),
            released_tick=jnp.asarray(-1, jnp.int64),
            accepted_tick=jnp.asarray(-1, jnp.int64),
            superseded_by=jnp.asarray(-1, jnp.int64),
            audience=jnp.asarray(core.PUBLIC, jnp.int64),
            valid=jnp.asarray(True),
            payload=jnp.array([value, 0.0, 1.0, 0.0]),
        )
        carry, _ = write_log(carry, record, study.caps)
    for slot in (1, 0):
        carry = core.enqueue(
            carry,
            True,
            kind=core.EventKind.RELEASE,
            tick=10,
            stage=core.Stage.RELEASE,
            slot=slot,
            channel=10,
            audience=core.PUBLIC,
            acquisition_id=7001,
            caps=study.caps,
        )
    traces = []
    for _ in range(2):
        carry, trace, _ = advance_j(carry, root, study, truth)
        traces.append(trace)
    trace = jax.tree.map(lambda *xs: jnp.stack(xs), *traces)
    return study, carry, trace


def test_replay_folds_first_released_revision_instead_of_first_logged(
    reordered_releases,
):
    study, carry, trace = reordered_releases
    assert carry.log.revision[:2].tolist() == [0, 1]
    assert trace.log_slot.tolist() == [1, 0]
    assert float(carry.belief.mean[0]) == 9.0 and int(carry.belief.n[0]) == 1
    assert _bits(replay_belief(study, carry, trace)) == _bits(carry.belief)


@pytest.mark.parametrize(
    "field, value",
    [("channel", 110), ("log_slot", -1), ("log_slot", 2), ("log_slot", 1000)],
)
def test_replay_rejects_unmatched_release(reordered_releases, field, value):
    study, carry, trace = reordered_releases
    trace = dataclasses.replace(
        trace, **{field: jnp.full_like(getattr(trace, field), value)}
    )
    with pytest.raises(ValueError, match="release"):
        replay_belief(study, carry, trace)
