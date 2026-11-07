"""Refresh protocol on runs: both executors, both boundaries, per-audience install."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import (
    Audience,
    CompiledExecutor,
    HostExecutor,
    RunStatus,
    handlers,
    histories_equal,
)
from spaceodyssey.core import refresh as refresh_module
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.carry import _tree_set
from spaceodyssey.core.conformance import refit_beliefs
from spaceodyssey.core.enums import F64, I64, Capacity, EventKind, RefitStatus, Stage
from spaceodyssey.core.executors import base
from spaceodyssey.studies import toy
from tests.core.corpus import LEAK_PLAN, LEAK_SCENARIOS
from tests.core.helpers import worlds
from tests.core.kit_helpers import RefitSample
from tests.core.leak_helpers import _views_match
from tests.core.refresh_helpers import (
    CorruptInstall,
    FailingFor,
    SpyBoundary,
    handled_steps,
    host_loop,
    masked,
    observer_only_column_study,
    oracle_refresh_study,
    policy_study,
    product_rows,
    refresh_study,
    run_pair,
)
from tests.core.test_canonical_fields import REFRESH_ONLY, _perturb


@functools.cache
def pair(chunk=7):
    return run_pair(refresh_study(100), 60, 2, chunk)


@pytest.mark.parametrize("chunk", [1, 7, 32])
def test_host_and_compiled_histories_equal_at_every_chunk(chunk):
    h, c = pair(chunk) if chunk == 7 else run_pair(refresh_study(100), 60, 2, chunk)
    assert h.status.tolist() == c.status.tolist()
    for w in range(2):
        r = histories_equal(h.history(w), c.history(w))
        assert r.ok and r.go, r


def test_refresh_ran_and_installed_something():
    h, _ = pair()
    kinds = np.asarray(h.trace.kind[0])[np.asarray(h.trace.handled[0])]
    assert (kinds == EventKind.REFRESH).any()
    assert np.asarray(h.carry.belief.n[0]).sum() > 0  # control: the belief moved


def test_host_loop_matches_masked_batch():
    study = refresh_study(100)
    truths, roots = worlds(1)
    a = HostExecutor().run_worlds(study, truths, roots, 60, refit=masked())
    b = HostExecutor().run_worlds(study, truths, roots, 60, refit=host_loop())
    r = histories_equal(a.history(0), b.history(0))
    assert r.ok and r.go, r


def test_awaiting_without_a_boundary_on_both_executors():
    study = refresh_study(100)
    truths, roots = worlds(1)
    h = HostExecutor().run_worlds(study, truths, roots, 60)
    c = CompiledExecutor().run(study, truths, roots, 60)
    assert h.status.tolist() == [RunStatus.AWAITING]
    assert c.status.tolist() == [RunStatus.AWAITING]


def test_no_refresh_study_ignores_a_boundary():
    from spaceodyssey.studies import toy

    study = toy.survey_1()
    truths, roots = worlds(1)
    a = HostExecutor().run_worlds(study, truths, roots, 40)
    b = HostExecutor().run_worlds(study, truths, roots, 40, refit=masked())
    assert histories_equal(a.history(0), b.history(0)).ok


def test_batched_worlds_equal_single_runs():
    study = refresh_study(100)
    truths, roots = worlds(3)
    batch = CompiledExecutor().run(study, truths, roots, 60, refit=masked(), chunk=7)
    for w in range(3):
        t1, _ = worlds(w + 1)
        single = HostExecutor().run_worlds(
            study,
            jax_index(t1, w),
            roots[w : w + 1],
            60,
            refit=masked(),
        )
        r = histories_equal(batch.history(w), single.history(0))
        assert r.ok and r.go, (w, r)


def jax_index(tree, w):
    import jax

    return jax.tree_util.tree_map(lambda x: x[w : w + 1], tree)


def test_corrupted_install_changes_the_history():
    study = refresh_study(100)
    truths, roots = worlds(1)
    good = HostExecutor().run_worlds(study, truths, roots, 60, refit=masked())
    bad = HostExecutor().run_worlds(
        study, truths, roots, 60, refit=CorruptInstall(masked())
    )
    assert not histories_equal(good.history(0), bad.history(0)).ok


def test_failed_status_installs_nothing_for_that_audience():
    study = refresh_study(100)
    truths, roots = worlds(1)
    ok = HostExecutor().run_worlds(study, truths, roots, 60, refit=masked())
    fail = HostExecutor().run_worlds(
        study, truths, roots, 60, refit=FailingFor(masked(), 0, RefitStatus.FAILED)
    )
    # the observer belief stays at its initial value; the control run moved it
    assert np.asarray(ok.carry.belief.n).sum() > 0
    assert np.asarray(fail.carry.belief.n).sum() == 0


def test_dedup_computes_once_when_rows_match():
    study = refresh_study(100)
    truths, roots = worlds(1)
    spy = SpyBoundary(masked())
    HostExecutor().run_worlds(study, truths, roots, 60, refit=spy)
    assert spy.calls
    req = spy.calls[0]
    dirty = req.dirty[0]
    both = dirty[0] & dirty[1]
    assert both.any()  # control: the dedup case is exercised
    n = dirty.shape[1]
    own = np.arange(2 * n).reshape(2, n)
    assert (req.source[0][1][both] == own[0][both]).all()  # policy served by observer
    assert (req.source[0][0][dirty[0]] == own[0][dirty[0]]).all()


# ---- field mutation parity on a refresh study ---------------------------------------


@functools.cache
def _mutation_pair():
    # Two programs, refunds, awards and balances shown, so the program, refund, award
    # and refused-request leaves are held too: every conditional leaf is exercised (a
    # producer naming no program charges program 0).
    caps = Capacities(
        rows_per_subject=8,
        refit_lanes=2,
        n_programs=2,
        refunds=True,
        snapshot_balances=True,
        awards=True,
    )
    # 20 steps: a refresh has run and an event is still pending (60 drains the queue).
    h, c = run_pair(refresh_study(100, caps=caps), 20, 2, 7)
    host = base.RunResult(
        base.index_world(h.carry, 0), base.index_world(h.trace, 0), h.layout
    )
    comp = base.RunResult(
        base.index_world(c.carry, 0), base.index_world(c.trace, 0), c.layout
    )
    return host, comp


@pytest.mark.medium  # The two-program refresh pair compiles the executor cold.
@pytest.mark.parametrize("field", sorted(REFRESH_ONLY))
def test_refresh_leaf_mutation_fails_parity(field):
    """Each conditional refresh leaf is compared; a compiled-only change is caught."""
    host, comp = _mutation_pair()
    assert base.histories_equal(host.history(), comp.history()).go  # control
    mutant = _perturb(comp, field.split("."))
    assert not base.histories_equal(host.history(), mutant.history()).ok, field


# ---- leak scenarios ------------------------------------------------------------------


def _leak_run(process, n_worlds=2, n_steps=60):
    study = toy.survey_refresh(
        100, policies=(toy.ReactivePolicy(plan=LEAK_PLAN),), process=process
    )
    truths, roots = worlds(n_worlds)
    out = []
    for w in range(n_worlds):
        views = []
        run = HostExecutor().run(
            study,
            base.index_world(truths, w),
            roots[w],
            n_steps,
            view_log=views,
            refit=masked(),
        )
        out.append((run, views))
    return out


@functools.cache
def _leak_baseline():
    return _leak_run(toy.NullProcess())


@pytest.mark.parametrize("name", ["heartbeat", "private_writer"])
def test_private_events_leave_wakes_views_and_product_rows_unchanged(name):
    baseline, scenario = _leak_baseline(), _leak_run(LEAK_SCENARIOS[name])
    n_private = 0
    for (b, bviews), (s, sviews) in zip(baseline, scenario, strict=True):
        assert np.asarray(s.trace.wake).sum() == np.asarray(b.trace.wake).sum()
        assert _views_match(sviews, bviews)
        assert product_rows(s.carry) == product_rows(b.carry)
        aud = np.asarray(s.trace.audience)[np.asarray(s.trace.handled)]
        n_private += int(((aud & Audience.POLICY) == 0).sum())
    assert n_private >= 1  # control: private events were handled
    assert product_rows(baseline[0][0].carry)  # control: checkpoints wrote rows
    assert sum(len(v) for _, v in baseline) >= 4  # control: the policy decided


# ---- a public release at a checkpoint followed by a Query ----------------------------


def _oracle_host(process=None, **kw):
    study, service = oracle_refresh_study(130, **kw)
    if process is not None:
        study = dataclasses.replace(study, process=process)
    truths, roots = worlds(1)
    run = HostExecutor().run(
        study,
        base.index_world(truths, 0),
        roots[0],
        40,
        service=service,
        refit=masked(),
    )
    return study, run, service


@pytest.mark.parametrize("coincident", [False, True])
def test_release_at_a_checkpoint_then_query_owes_one_decision_after_refresh(coincident):
    process = (
        toy.HeartbeatProcess(
            period_ticks=130, n_fires=1, audience=Audience.POLICY, writes=False
        )
        if coincident
        else None
    )
    _, run, service = _oracle_host(process)
    carry, trace = run.carry, run.trace
    steps = handled_steps(trace)
    at = [x for x in steps if x[1] == 130 and x[2] <= Stage.DECIDE]
    refresh = [x for x in at if x[0] == EventKind.REFRESH]
    assert len(refresh) == 2  # hold, then completion
    hold, done = refresh
    assert hold[3] == 0 and hold[4] == 0 and done[3] > 0
    # the owed decision runs once, in the completion transition, before the Query
    first_request = [x[0] for x in steps].index(EventKind.REQUEST)
    decisions = [x for x in steps[:first_request] if x[1] == 130 and x[4] != 0]
    assert decisions == [done]
    # it read the post-refresh snapshot: one row counted, though nothing folds
    (request,) = service.requests
    assert request.tick == 130
    row = int(np.flatnonzero(np.asarray(carry.log.kind) == 4)[0])
    assert float(carry.log.payload[row, 1]) == 1.0
    assert not (bool(carry.request.valid) or bool(carry.refresh.valid))
    assert not (bool(carry.violation) or bool(carry.dup_fold))


def test_request_and_refresh_never_wait_together():
    _, run, _ = _oracle_host()
    carry, trace = run.carry, run.trace
    kinds = [x[0] for x in handled_steps(trace)]
    i = kinds.index(EventKind.REQUEST)
    # nothing is handled between a Query and its answer
    assert kinds[i + 1] == EventKind.RESULT
    assert not (bool(carry.request.valid) and bool(carry.refresh.valid))


def test_oracle_refresh_parity_across_executors_and_chunks():
    study, _ = oracle_refresh_study(130)
    truths, roots = worlds(2)
    hosts = []
    for w in range(2):
        _, service = oracle_refresh_study(130)
        hosts.append(
            HostExecutor().run(
                study,
                base.index_world(truths, w),
                roots[w],
                40,
                service=service,
                refit=masked(),
            )
        )
    for chunk in (1, 7):
        _, service = oracle_refresh_study(130)
        c = CompiledExecutor().run(
            study, truths, roots, 40, service=service, refit=masked(), chunk=chunk
        )
        assert c.status.tolist() == [RunStatus.COMPLETE] * 2
        for w in range(2):
            h = hosts[w].history()
            r = histories_equal(h, c.history(w))
            assert r.ok and r.go, (chunk, w, r)


# ---- drain ---------------------------------------------------------------------------


def _both(study, n_steps=200, **kw):
    truths, roots = worlds(1)
    refit = masked()
    host = HostExecutor().run_worlds(study, truths, roots, n_steps, refit=refit, **kw)
    comp = CompiledExecutor().run(
        study, truths, roots, n_steps, refit=refit, chunk=7, **kw
    )
    return host, comp


def test_release_past_the_horizon_schedules_a_later_checkpoint_and_completes():
    # two observations: the second is released at 260, after the horizon, and the
    # campaign then drains through a checkpoint on the unclamped grid
    study = refresh_study(
        100, horizon_tick=240, policies=(toy.FixedPlanPolicy(plan=((0, 10), (1, 10))),)
    )
    host, comp = _both(study)
    assert host.status.tolist() == comp.status.tolist() == [RunStatus.COMPLETE]
    r = histories_equal(host.history(0), comp.history(0))
    assert r.ok and r.go, r
    steps = handled_steps(base.index_world(host.trace, 0))
    last_release = max(x[1] for x in steps if x[0] == EventKind.RELEASE)
    last_refresh = max(x[1] for x in steps if x[0] == EventKind.REFRESH)
    assert last_release > 240  # control: a release arrived after the horizon
    assert last_refresh >= last_release and last_refresh % 100 == 0  # unclamped grid
    # the kit's end-of-run refit of the final log is what the checkpoints installed
    for run in (host, comp):
        _, roots = worlds(1)
        want = refit_beliefs(study, run, roots)
        assert np.array_equal(np.asarray(run.carry.belief.n), np.asarray(want.n))
        assert np.allclose(np.asarray(run.carry.belief.mean), np.asarray(want.mean))
    # the belief holds every accepted row: the late release was refit too
    n = np.asarray(host.carry.belief.n[0])
    assert (
        n.sum() == (np.asarray(host.carry.log.kind[0]) == toy.RECORD_MEASUREMENT).sum()
    )


def test_admit_stage_result_at_a_checkpoint_tick_falls_in_the_next_checkpoint():
    # one observation: the answer is the only row after tick 130, and the plan is spent
    plan = ((0, 10),)
    study, _ = oracle_refresh_study(130, answers_refit=True, plan=plan)
    control, _ = oracle_refresh_study(130, answers_refit=False, plan=plan)
    truths, roots = worlds(1)
    out = {}
    for name, st in (("answers", study), ("control", control)):
        _, service = oracle_refresh_study(
            130, answers_refit=name == "answers", plan=plan
        )
        out[name] = HostExecutor().run(
            st,
            base.index_world(truths, 0),
            roots[0],
            60,
            service=service,
            refit=masked(),
        )
    ticks = {
        k: [x[1] for x in handled_steps(v.trace) if x[0] == EventKind.REFRESH]
        for k, v in out.items()
    }
    assert all(int(v.status) == RunStatus.COMPLETE for v in out.values())
    assert ticks["control"] == [130, 130]  # control: no later row, no later checkpoint
    # the result is accepted at ADMIT of tick 130, after ANALYZE: the next grid tick
    assert ticks["answers"] == [130, 130, 260, 260]
    carry = out["answers"].carry
    assert not bool(carry.refresh.valid) and not bool(carry.request.valid)
    c = CompiledExecutor().run(
        study,
        truths,
        roots,
        60,
        service=oracle_refresh_study(130, answers_refit=True, plan=plan)[1],
        refit=masked(),
        chunk=7,
    )
    assert c.status.tolist() == [RunStatus.COMPLETE]
    assert histories_equal(out["answers"].history(), c.history(0)).ok


def test_checkpoints_do_not_wake_the_policy():
    host, _ = pair()
    steps = handled_steps(base.index_world(host.trace, 0))
    release_ticks = {x[1] for x in steps if x[0] == EventKind.RELEASE}
    decided = {x[1] for x in steps if x[4] != 0}
    assert decided <= release_ticks | {0}  # control: decisions did run
    assert len(decided) >= 2
    refresh_ticks = {x[1] for x in steps if x[0] == EventKind.REFRESH}
    assert refresh_ticks - release_ticks  # a checkpoint fell where nothing released


# ---- schedule mutant -----------------------------------------------------------------


def _refresh_ticks_for(process):
    study = toy.survey_refresh(
        100, policies=(toy.ReactivePolicy(plan=LEAK_PLAN),), process=process
    )
    truths, roots = worlds(1)
    run = HostExecutor().run(
        study, base.index_world(truths, 0), roots[0], 60, refit=masked()
    )
    return [x[1] for x in handled_steps(run.trace) if x[0] == EventKind.REFRESH]


@pytest.fixture
def fresh_compilation():
    import equinox as eqx

    eqx.clear_caches()
    jax.clear_caches()
    yield
    eqx.clear_caches()
    jax.clear_caches()


def test_private_events_do_not_move_checkpoints():
    quiet = _refresh_ticks_for(toy.NullProcess())
    busy = _refresh_ticks_for(LEAK_SCENARIOS["private_writer"])
    assert quiet and quiet == busy


def test_a_transition_count_schedule_is_caught(monkeypatch, fresh_compilation):
    """A grid shifted by the log length (private writes change it) moves checkpoints."""
    real = handlers.schedule_refresh

    def by_count(c, mark, study):
        shifted = _tree_set(c, tick=c.tick + 25 * c.n_log)
        return _tree_set(real(shifted, mark, study), tick=c.tick)

    monkeypatch.setattr(handlers, "schedule_refresh", by_count)
    quiet = _refresh_ticks_for(toy.NullProcess())
    busy = _refresh_ticks_for(LEAK_SCENARIOS["private_writer"])
    assert quiet != busy


# ---- policy-side parity --------------------------------------------------------------


def test_belief_reading_policy_agrees_across_executors_and_reads_the_install():
    study = policy_study(100)
    h, c = run_pair(study, 60, 2, 7)
    for w in range(2):
        r = histories_equal(h.history(w), c.history(w))
        assert r.ok and r.go, (w, r)
    truths, roots = worlds(1)
    bad = HostExecutor().run_worlds(
        study, truths, roots, 60, refit=CorruptInstall(masked())
    )
    # control: the policy's decisions depend on what the refresh installed
    assert not histories_equal(h.history(0), bad.history(0)).ok
    decided = int((np.asarray(h.trace.snapshot_fp[0]) != 0).sum())
    assert decided >= 3


# ---- an order-sensitive observer -----------------------------------------------------


def test_order_sensitive_observer_agrees_across_executors():
    observer = RefitSample()
    assert observer.order_sensitive
    study = toy.survey_refresh(100, observer=observer)
    h, c = run_pair(study, 60, 2, 7, refit=refresh_module.MaskedBatchRefit())
    for w in range(2):
        r = histories_equal(h.history(w), c.history(w))
        assert r.ok and r.go, (w, r)
    assert np.asarray(h.carry.belief.n).sum() > 0
    # the refit sample was installed: every observed subject holds a draw
    seen = np.asarray(h.carry.belief.n[0]) > 0
    assert (np.asarray(h.carry.belief.sample[0])[seen] != 0).all()


# ---- deduplication control for an observer-only column -------------------------------


def _computations(study):
    truths, roots = worlds(1)
    spy = SpyBoundary(masked())
    HostExecutor().run_worlds(study, truths, roots, 60, refit=spy)
    total = 0
    for out in spy.results:
        total += int(out.computed.sum())
    return total, sum(int(r.dirty.sum()) for r in spy.calls)


def test_observer_only_column_is_computed_twice_and_public_rows_once():
    shared, wanted = _computations(refresh_study(100))
    twice, wanted2 = _computations(observer_only_column_study(100))
    assert wanted == wanted2 and wanted > shared  # control: dedup happens at all
    assert twice == wanted2  # no entry shares a computation
    assert shared < twice


# ---- review fixes: budget, capacity, execution, audience ---------------------------


@pytest.mark.parametrize("chunk", [1, 7, 32])
@pytest.mark.parametrize("budget", [1, 2, 3, 4])
def test_a_budget_that_ends_on_the_hold_installs_nothing(budget, chunk):
    """Fails if a checkpoint installs when no step is left for its completion."""
    h, c = run_pair(refresh_study(100), budget, 2, chunk)
    for res in (h, c):
        carry = base.index_world(res.carry, 0)
        assert float(np.asarray(carry.belief.n).sum()) == 0.0
        assert float(np.asarray(carry.pbelief.n).sum()) == 0.0
        assert product_rows(carry) == []
        assert not np.asarray(carry.refresh.valid)
    # control: with a budget for the completion the same checkpoint installs
    full = run_pair(refresh_study(100), 60, 2, chunk)[0]
    assert float(np.asarray(full.carry.belief.n[0]).sum()) > 0


@pytest.mark.parametrize("which", ["ids", "rows"])
def test_capacity_exhaustion_refuses_the_whole_install(monkeypatch, which):
    """Fails if a belief installs while its product rows or ids have no room."""
    study = refresh_study(100)
    caps = study.caps
    seen = []
    real = refresh_module._commit

    def tight(carry, request, result, st):
        field = dict(ids="next_sid", rows="n_log")[which]
        limit = dict(ids=caps.max_sid, rows=caps.n_log)[which]
        squeezed = _tree_set(carry, **{field: jnp.asarray(limit, I64)})
        out = real(squeezed, request, result, st)
        seen.append((squeezed, out))
        return out

    monkeypatch.setattr(refresh_module, "_commit", tight)
    truths, roots = worlds(1)
    HostExecutor().run_worlds(study, truths, roots, 20, refit=masked())
    assert seen
    before, after = seen[0]
    bit = dict(ids=Capacity.STABLE_IDS, rows=Capacity.LOG_ROWS)[which]
    assert bool(after.overflow) and int(after.overflow_mask) & int(bit)
    for a, b in zip(
        jax.tree_util.tree_leaves(before.belief),
        jax.tree_util.tree_leaves(after.belief),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(before.refresh_dirty, after.refresh_dirty)
    assert product_rows(after) == []


def _counting_kernel(counter):
    def kernel(study, entry):
        out = toy.toy_mean_kernel(study, entry)
        jax.debug.callback(lambda: counter.append(1))
        return out

    return kernel


def test_masked_batch_executes_only_unique_dirty_entries():
    """Fails if the kernel runs for padded, clean or deduplicated entries."""
    study = refresh_study(100)
    truths, roots = worlds(1)
    counter = []
    spy = SpyBoundary(refresh_module.MaskedBatchRefit(_counting_kernel(counter)))
    HostExecutor().run_worlds(study, truths, roots, 60, refit=spy)
    jax.effects_barrier()
    wanted = sum(int(out.computed.sum()) for out in spy.results)
    assert wanted > 0
    assert len(counter) == wanted
    # control: dedup shared some entries, so a run of every dirty entry would be more
    assert wanted < sum(int(r.dirty.sum()) for r in spy.calls)


def _fraction_kernel(study, entry):
    """A fit of the observer-only fraction column: the policy copy of it is masked."""
    rows = entry.rows
    take = rows.valid & (rows.kind == study.observer.measurement_kind)
    n = jnp.sum(take).astype(I64)
    total = jnp.sum(jnp.where(take, rows.payload[:, 2], 0.0))
    mean = total / jnp.maximum(n, 1).astype(F64)
    ok = jnp.asarray(RefitStatus.OK, I64)
    return (mean, n), ok, jnp.zeros(1, F64)


def test_each_audience_installs_its_own_slice():
    """Fails if the policy summary takes the observer's slice, or the reverse."""
    study = observer_only_column_study(100)
    refit = refresh_module.MaskedBatchRefit(_fraction_kernel)
    h, c = run_pair(study, 60, 1, 7, refit=refit)
    for res in (h, c):
        carry = base.index_world(res.carry, 0)
        seen = np.asarray(carry.belief.n) > 0
        assert seen.any()
        observer_mean = np.asarray(carry.belief.mean)[seen]
        policy_mean = np.asarray(carry.pbelief.mean)[seen]
        assert (observer_mean > 0).all()  # the observer read its column
        assert (policy_mean == 0).all()  # the policy copy of it is masked
    r = histories_equal(h.history(0), c.history(0))
    assert r.ok and r.go, r
