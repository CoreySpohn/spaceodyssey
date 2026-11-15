"""Run status: a run that hit a capacity or ran out of steps is never complete."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.testbudget.compiles import cache_hit_count, compile_count

from spaceodyssey.core import (
    Capacities,
    CompiledExecutor,
    HostExecutor,
    RunStatus,
    empty_carry,
    run_status,
)
from spaceodyssey.core.executors import host
from spaceodyssey.core.executors.base import index_world
from spaceodyssey.core.lift import lift_traced
from spaceodyssey.core.snapshot import effects_none
from spaceodyssey.core.transaction import mask_effects
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


def test_host_stops_advancing_terminal_worlds(monkeypatch):
    """Fails if the host spends the remaining allowance on completed worlds."""
    calls = []
    advance = HostExecutor.advance

    def counted(self, carry, *args, **kwargs):
        calls.append(bool(carry.done | carry.idle | carry.stalled | carry.overflow))
        return advance(self, carry, *args, **kwargs)

    monkeypatch.setattr(HostExecutor, "advance", counted)
    study = builder.survey_1()
    result = run_status_case(HostExecutor, study, study.caps, n_steps=120)
    assert result.complete.all()
    assert not any(calls)
    assert len(calls) < 240
    assert result.trace.handled.shape[1] == 120


def test_host_batch_pads_worlds_that_stop_at_different_lengths(monkeypatch):
    """Fails if early stopping breaks batching or writes events into padding."""
    study = builder.survey_1()
    truths, roots = worlds(2)
    executor = HostExecutor()
    run = HostExecutor.run
    budgets = iter((5, 120, 40, 120))

    def limited(self, study, truth, root, n_steps, **kwargs):
        return run(self, study, truth, root, next(budgets), **kwargs)

    monkeypatch.setattr(HostExecutor, "run", limited)
    result = executor.run_worlds(study, truths, roots, 120)
    assert result.status.tolist() == [RunStatus.TRUNCATED, RunStatus.COMPLETE]
    assert not result.trace.handled[0, 5:].any()
    assert not result.trace.wake[0, 5:].any()
    assert result.history(1).n_records == 5
    terminal_batch = executor.run_worlds(study, truths, roots, 120)
    assert terminal_batch.complete.all()
    for world in range(2):
        expected = run(executor, study, index_world(truths, world), roots[world], 120)
        actual = index_world(terminal_batch.trace, world)
        for a, b in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected.trace), strict=True
        ):
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def _stepped_host(study, truth, root, n_steps):
    """The host transaction without the early terminal shortcut."""
    study = lift_traced(study)
    carry = host._init(study, truth)
    executor, rows = HostExecutor(), []
    for _ in range(n_steps):
        carry, trace, wake = executor.advance(carry, root, study, truth)
        eff, pstate, u = (
            executor.decide(carry, root, study)
            if bool(wake)
            else (effects_none(study.caps), carry.pstate, jnp.asarray(0.0))
        )
        eff = mask_effects(eff, wake)
        fp = host._fp(carry, study) if bool(wake) else jnp.asarray(0, jnp.int64)
        carry, codes = host._adm(carry, eff, study, fp)
        carry = host._close(carry, eff, pstate, wake, study)
        rows.append(host._fin(trace, eff, codes, u, wake, carry, study, fp))
    return carry, jax.tree.map(lambda *xs: np.stack(xs), *rows)


def test_host_terminal_padding_matches_every_stepped_leaf():
    """Fails if early stopping changes diagnostics, padding bytes or final carry."""
    study = builder.survey_1()
    truths, roots = worlds(1)
    truth = jax.tree.map(lambda x: x[0], truths)
    result = HostExecutor().run(study, truth, roots[0], 40)
    expected = _stepped_host(study, truth, roots[0], 40)
    assert bool(result.carry.done) and int(result.trace.handled.sum()) < 40
    for a, b in zip(
        jax.tree.leaves((result.carry, result.trace)),
        jax.tree.leaves(expected),
        strict=True,
    ):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_trace_row_stacking_has_bounded_program_count():
    """Fails if repeated host trace layouts compile again or alter padding bytes."""
    row = host.empty_trace(Capacities())
    layouts = (([row], row, 4), ([row, row, row], row, 2))
    before = compile_count() + cache_hit_count()
    for _ in range(2):
        for traces, padding, size in layouts:
            result = host._trace_rows(traces, padding, size)
            expected = jax.tree.map(
                lambda *xs: np.stack(xs), *traces, *([padding] * size)
            )
            for a, b in zip(
                jax.tree.leaves(result), jax.tree.leaves(expected), strict=True
            ):
                assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    assert compile_count() + cache_hit_count() - before <= 2
