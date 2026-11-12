"""Run status: a run that hit a capacity or ran out of steps is never complete."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import (
    Capacities,
    CompiledExecutor,
    HostExecutor,
    RunStatus,
    empty_carry,
    run_status,
)
from spaceodyssey.studies import builder
from tests.core.corpus import CORPUS_BY_NAME
from tests.core.helpers import worlds
from tests.core.status_helpers import OVERFLOW_CASES, overflow_study, run_status_case

EXECUTORS = [CompiledExecutor, HostExecutor]


@pytest.mark.parametrize("executor", EXECUTORS, ids=lambda e: e.__name__)
@pytest.mark.parametrize("name", sorted(OVERFLOW_CASES))
def test_overflow_sets_status(executor, name):
    caps, bit = OVERFLOW_CASES[name]
    res = run_status_case(executor, overflow_study(name, caps), caps)
    assert res.status.tolist() == [RunStatus.OVERFLOW] * 2
    assert not res.complete.any()
    assert res.carry.overflow_mask.tolist() == [int(bit)] * 2
    for world in range(2):
        h = res.history(world)
        assert h.status == RunStatus.OVERFLOW and not h.complete
        assert h.overflow_mask == bit


@pytest.mark.parametrize("executor", EXECUTORS, ids=lambda e: e.__name__)
def test_overflow_is_never_reported_complete_with_more_steps(executor):
    caps, _ = OVERFLOW_CASES["log_rows"]
    res = run_status_case(executor, builder.survey_1(caps=caps), caps, n_steps=120)
    assert res.status.tolist() == [RunStatus.OVERFLOW] * 2


@pytest.mark.parametrize("executor", EXECUTORS, ids=lambda e: e.__name__)
def test_microstep_guard_reports_stalled(executor):
    study = CORPUS_BY_NAME["overflow_stall"].make_study()
    res = run_status_case(executor, study, study.caps)
    assert res.status.tolist() == [RunStatus.STALLED] * 2
    assert not res.complete.any() and res.carry.overflow_mask.tolist() == [0, 0]


@pytest.mark.parametrize("executor", EXECUTORS, ids=lambda e: e.__name__)
def test_truncated_run_not_reported_complete(executor):
    study = builder.survey_1()
    short = run_status_case(executor, study, study.caps, n_steps=5)
    assert short.status.tolist() == [RunStatus.TRUNCATED] * 2
    assert not short.complete.any() and not short.history(0).complete
    full = run_status_case(executor, study, study.caps, n_steps=40)
    assert full.status.tolist() == [RunStatus.COMPLETE] * 2
    assert full.complete.all() and full.history(1).complete


def test_status_agrees_across_executors():
    for name, caps in (
        ("base", Capacities()),
        *((k, c) for k, (c, _) in OVERFLOW_CASES.items()),
    ):
        study = overflow_study(name, caps)
        a = run_status_case(CompiledExecutor, study, caps)
        b = run_status_case(HostExecutor, study, caps)
        assert a.status.tolist() == b.status.tolist()
        assert a.carry.overflow_mask.tolist() == b.carry.overflow_mask.tolist()


def test_zero_step_budget_returns_matching_truncated_results():
    study = builder.survey_1()
    truths, roots = worlds(2)
    compiled = CompiledExecutor().run_worlds(study, truths, roots, 0)
    host = HostExecutor().run_worlds(study, truths, roots, 0)
    assert host.status.tolist() == compiled.status.tolist() == [RunStatus.TRUNCATED] * 2
    assert not host.complete.any()
    for a, b in zip(jax.tree.leaves(host), jax.tree.leaves(compiled), strict=True):
        np.testing.assert_array_equal(a, b, strict=True)
    assert host.trace.handled.shape == (2, 0)
    assert host.history(0).n_handled == host.history(0).n_records == 0


def test_status_priority_and_idle():
    c = empty_carry(Capacities())
    assert int(run_status(c)) == RunStatus.TRUNCATED
    done = dataclasses.replace(c, done=jnp.asarray(True))
    assert int(run_status(done)) == RunStatus.COMPLETE
    idle = dataclasses.replace(done, idle=jnp.asarray(True))
    assert int(run_status(idle)) == RunStatus.IDLE
    stalled = dataclasses.replace(idle, stalled=jnp.asarray(True))
    assert int(run_status(stalled)) == RunStatus.STALLED
    over = dataclasses.replace(stalled, overflow=jnp.asarray(True))
    assert int(run_status(over)) == RunStatus.OVERFLOW
