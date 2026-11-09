"""A2: the host log, fed back, reproduces the observer belief bit for bit."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey import core
from spaceodyssey.core import (
    HostExecutor,
    MeanBelief,
    index_world,
    replay_belief,
    replay_run,
)
from spaceodyssey.core.records import write_log
from spaceodyssey.studies import toy
from tests.core.corpus import CORPUS, CORPUS_BY_NAME
from tests.core.helpers import N_STEPS_CORPUS, advance_j, one_world, service_of, worlds
from tests.core.mutants import (
    initial_belief_replay,
    live_belief_replay,
    log_order_replay,
)

# A refresh case's belief comes from installs, not release folds: `replay_belief`
# folds only, so `replay_run` replays those cases (below).
RELEASING = [
    c for c in CORPUS if not c.name.startswith("overflow") and c.make_refit is None
]


REFRESHING = [c for c in CORPUS if c.make_refit is not None]


def _bits(belief):
    return [np.asarray(x).tobytes() for x in (belief.mean, belief.n, belief.aux)]


def replay_matches_live(case, replay=None):
    """True when `replay` reproduces the live belief bit for bit in both worlds."""
    replay = replay_belief if replay is None else replay
    study = case.make_study()
    truths, roots = worlds(2)
    for w in range(2):
        carry, trace = HostExecutor().run(
            study, index_world(truths, w), roots[w], N_STEPS_CORPUS, **service_of(case)
        )
        replayed = replay(study, carry, trace)
        if _bits(replayed) != _bits(carry.belief) or int(jnp.sum(replayed.n)) < 1:
            return False
    return True


def replay_follows_the_log(replay=None):
    """True when editing one logged payload value changes what `replay` returns."""
    replay = replay_belief if replay is None else replay
    study = CORPUS_BY_NAME["baseline"].make_study()
    truths, roots = worlds(1)
    carry, trace = HostExecutor().run(
        study, index_world(truths, 0), roots[0], N_STEPS_CORPUS
    )
    payload = carry.log.payload.at[0, 0].add(1e-9)
    edited = dataclasses.replace(
        carry, log=dataclasses.replace(carry.log, payload=payload)
    )
    return _bits(replay(study, edited, trace)) != _bits(replay(study, carry, trace))


@pytest.mark.parametrize("case", RELEASING, ids=lambda c: c.name)
def test_replay_reproduces_belief_bitwise(case):
    assert replay_matches_live(case), case.name


def test_replay_depends_on_the_log():
    assert replay_follows_the_log()


@pytest.mark.parametrize("name", ["baseline", "revision"])
def test_a_replay_that_ignores_the_run_fails(name):
    """Fails if a replay that folds nothing still matches the live belief."""
    assert not replay_matches_live(CORPUS_BY_NAME[name], initial_belief_replay)


def test_a_replay_that_returns_the_live_belief_fails_the_log_edit():
    """Fails if the log-edit check cannot tell a replay that returns the live belief
    (equal by construction) from one that follows the log.
    """
    assert replay_matches_live(CORPUS_BY_NAME["baseline"], live_belief_replay)
    assert not replay_follows_the_log(live_belief_replay)
    assert not replay_follows_the_log(initial_belief_replay)


def test_a_replay_in_log_order_fails_the_release_order_check():
    """Fails if a replay that folds the first logged revision passes as one that
    folds the first released.
    """
    assert reordered_replay_matches()
    assert not reordered_replay_matches(log_order_replay)


def test_replay_returns_a_belief():
    study = CORPUS_BY_NAME["baseline"].make_study()
    truths, roots = worlds(1)
    carry, trace = HostExecutor().run(
        study, index_world(truths, 0), roots[0], N_STEPS_CORPUS
    )
    assert isinstance(replay_belief(study, carry, trace), MeanBelief)


def build_reordered_releases():
    """A run that logs revision 0 first and releases revision 1 first."""
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


@pytest.fixture
def reordered_releases():
    return build_reordered_releases()


def reordered_replay_matches(replay=None):
    """True when `replay` folds the revision that was released first."""
    replay = replay_belief if replay is None else replay
    study, carry, trace = build_reordered_releases()
    return _bits(replay(study, carry, trace)) == _bits(carry.belief)


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


@pytest.mark.parametrize("case", REFRESHING, ids=lambda c: c.name)
def test_refresh_cases_replay_with_their_checkpoints(case):
    study = case.make_study()
    truths, roots = worlds(2)
    for w in range(2):
        carry, trace = HostExecutor().run(
            study,
            index_world(truths, w),
            roots[w],
            N_STEPS_CORPUS,
            refit=case.make_refit(),
            **service_of(case),
        )
        out = replay_run(study, roots[w], carry, trace, refit=case.make_refit())
        assert out.checkpoints >= 1 and out.products >= 1, case.name
