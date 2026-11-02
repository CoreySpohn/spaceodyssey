"""A2: the host log, fed back, reproduces the observer belief bit for bit."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import HostExecutor, MeanBelief, index_world, replay_belief
from tests.core.corpus import CORPUS, CORPUS_BY_NAME
from tests.core.helpers import N_STEPS_CORPUS, service_of, worlds

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
