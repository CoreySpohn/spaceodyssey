"""Boundary checks reject missing evidence and committed-state corruption."""

import dataclasses
import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import EventKind, handlers, transaction
from spaceodyssey.core.conformance import run_case
from spaceodyssey.core.conformance.invariants import (
    balance_conserved,
    dense_acquisition_index,
    no_scorer_column_in_snapshot,
    one_fold_per_acquisition,
)
from spaceodyssey.core.enums import Capacity
from spaceodyssey.core.executors import host as host_module
from spaceodyssey.studies.conformance import CORPUS_BY_NAME
from tests.core.conformance_helpers import runs
from tests.core.test_corpus_invariants import planted


@pytest.fixture
def fresh_compilation():
    # Mutations must reach traced handlers, including after cached positive runs.
    eqx.clear_caches()
    jax.clear_caches()
    yield
    eqx.clear_caches()
    jax.clear_caches()


@pytest.mark.parametrize("executor", ("host", "compiled"))
@pytest.mark.parametrize("decision", (1, 3))
def test_actual_decision_inputs_hide_columns(
    executor, decision, monkeypatch, fresh_compilation
):
    original = transaction.snapshot_view

    def leak(carry, audience, caps):
        view = original(carry, audience, caps)
        return eqx.tree_at(
            lambda v: v.log.payload,
            view,
            jnp.where(carry.n_dec == decision, carry.log.payload, view.log.payload),
        )

    monkeypatch.setattr(transaction, "snapshot_view", leak)
    (run,) = run_case(CORPUS_BY_NAME["baseline"], executor, n_worlds=1, probe=False)
    result = no_scorer_column_in_snapshot(run)
    assert not result.passed, result
    assert "decision" in result.diagnostic


@pytest.mark.parametrize("executor", ("host", "compiled"))
@pytest.mark.parametrize("case", ("baseline", "overflow_log"))
def test_admission_counters_cannot_disappear(executor, case):
    run = runs(case, executor)[0]
    assert np.any(np.asarray(run.carry.log.valid))
    bad = planted(
        run,
        lambda c: c.acquisition_counts,
        jnp.zeros_like(run.carry.acquisition_counts),
    )
    assert not dense_acquisition_index(bad).passed


@pytest.mark.parametrize("executor", ("host", "compiled"))
@pytest.mark.parametrize("index", (-1, -7))
def test_overflow_retained_indices_have_a_lower_bound(executor, index):
    run = runs("overflow_log", executor)[0]
    bad = planted(
        run,
        lambda c: c.log.acquisition_index,
        run.carry.log.acquisition_index.at[0].set(index),
    )
    assert not dense_acquisition_index(bad).passed


@pytest.mark.parametrize("executor", ("host", "compiled"))
@pytest.mark.parametrize("audience", ("observer", "policy"))
def test_each_accepted_acquisition_requires_a_committed_fold(
    executor, audience, monkeypatch, fresh_compilation
):
    original = handlers.h_release
    source = inspect.getsource(original)
    if audience == "observer":
        old = (
            "belief = where_tree(to_observer & ~already & ~accepted.any(), "
            "folded, c.belief)"
        )
        new = "belief = c.belief"
    else:
        old = (
            "pbelief = where_tree(public & ~seen & ~released.any(), pfolded, c.pbelief)"
        )
        new = "pbelief = c.pbelief"
    assert old in source
    namespace = dict(original.__globals__)
    exec(compile(source.replace(old, new), "<missing_commit>", "exec"), namespace)
    monkeypatch.setattr(
        handlers,
        "HANDLERS",
        tuple(
            namespace["h_release"] if h is original else h for h in handlers.HANDLERS
        ),
    )
    (run,) = run_case(CORPUS_BY_NAME["baseline"], executor, n_worlds=1, probe=False)
    assert np.any(np.asarray(run.carry.log.accepted_tick) >= 0)
    assert not bool(run.carry.dup_fold)
    assert not one_fold_per_acquisition(run).passed


@pytest.mark.parametrize("executor", ("host", "compiled"))
def test_overflow_does_not_disable_conservation(executor):
    run = runs("overflow_log", executor)[0]
    assert balance_conserved(run).passed
    bad = planted(run, lambda c: c.ledger.balance, run.carry.ledger.balance + 0.5)
    assert not balance_conserved(bad).passed


@pytest.mark.parametrize("executor", ("host", "compiled"))
def test_missing_decision_capture_cannot_pass(executor):
    run = runs("baseline", executor)[0]
    assert run.wakes > 0
    assert not no_scorer_column_in_snapshot(dataclasses.replace(run, views=())).passed


@pytest.mark.parametrize("executor", ("host", "compiled"))
@pytest.mark.parametrize("audience", (0, 1))
def test_missing_fold_evidence_cannot_pass(executor, audience):
    run = runs("baseline", executor)[0]
    folds = list(run.folds)
    ids, counts = folds[audience]
    changed = counts.copy()
    changed[np.flatnonzero(changed)[0]] = 0
    folds[audience] = (ids, changed)
    assert not one_fold_per_acquisition(
        dataclasses.replace(run, folds=tuple(folds))
    ).passed


@pytest.mark.parametrize("executor", ("host", "compiled"))
@pytest.mark.parametrize("audience", (0, 1))
def test_fold_totals_cannot_substitute_for_acquisition_identity(executor, audience):
    run = runs("baseline", executor)[0]
    folds = list(run.folds)
    ids, counts = folds[audience]
    changed = ids.copy()
    changed[np.flatnonzero(counts)[0]] += 123
    folds[audience] = (changed, counts)
    bad = dataclasses.replace(run, folds=tuple(folds))
    assert not one_fold_per_acquisition(bad).passed


def test_host_python_form_views_are_the_ones_judged(monkeypatch, fresh_compilation):
    original = host_module.snapshot_view

    def leak(carry, audience, caps):
        view = original(carry, audience, caps)
        return eqx.tree_at(lambda v: v.log.payload, view, carry.log.payload)

    monkeypatch.setattr(host_module, "snapshot_view", leak)
    case = CORPUS_BY_NAME["baseline"]
    assert "python" in case.policy_forms
    (run,) = run_case(case, "host", n_worlds=1, probe=False)
    result = no_scorer_column_in_snapshot(run)
    assert not result.passed, result
    assert "python" in result.diagnostic


@pytest.mark.parametrize("executor", ("host", "compiled"))
def test_every_woken_step_has_exactly_one_captured_view(executor):
    run = runs("baseline", executor)[0]
    assert run.wakes > 1 and len(run.views) == run.wakes
    assert all(view is not None for view in run.views)
    forms = run.form_views if executor == "host" else {}
    assert (executor == "host") == ("python" in forms)
    assert all(len(v) == run.wakes for v in forms.values())


@pytest.mark.parametrize("executor", ("host", "compiled"))
def test_extra_admission_in_a_counter_is_caught(executor):
    run = runs("baseline", executor)[0]
    counts = run.carry.acquisition_counts
    bad = planted(run, lambda c: c.acquisition_counts, counts.at[0].add(1))
    assert not dense_acquisition_index(bad).passed


@pytest.mark.parametrize("executor", ("host", "compiled"))
def test_overflow_conservation_matches_an_independent_balance(executor):
    # survey_1 admits full-cost observations only, so every admission is a completion
    # the trace handled or one still queued; each costs its channel's declared cost.
    run = runs("overflow_log", executor)[0]
    study, trace, carry = run.study, run.trace, run.carry
    assert int(carry.overflow_mask) & Capacity.LOG_ROWS and int(carry.n_log) == 4
    ids = np.asarray(study.channels.id).tolist()
    lanes = np.asarray(study.assets.lane_asset)
    cost = np.asarray(study.channels.cost)
    expected = np.asarray(study.budget).copy()
    handled = np.asarray(trace.handled) & (np.asarray(trace.kind) == EventKind.DONE)
    settled = [
        (int(trace.channel[i]), int(trace.lane[i])) for i in np.flatnonzero(handled)
    ]
    ev = carry.ev
    queued = np.asarray(ev.valid) & (np.asarray(ev.kind) == EventKind.DONE)
    settled += [(int(ev.channel[i]), int(ev.lane[i])) for i in np.flatnonzero(queued)]
    assert len(settled) == int(np.asarray(carry.acquisition_counts).sum()) > 4
    rows = (None, study.caps.n_assets, study.caps.n_balance_rows - 1)
    for channel, lane in settled:
        for r in range(3):
            row = int(lanes[lane]) if r == 0 else rows[r]
            expected[row, r] -= cost[ids.index(channel), r]
    np.testing.assert_allclose(carry.ledger.balance, expected, atol=1e-9, rtol=0)
    assert balance_conserved(run).passed


@pytest.mark.parametrize("executor", ("host", "compiled"))
def test_a_consistent_forgery_of_balance_and_counter_is_caught_in_overflow(executor):
    run = runs("overflow_log", executor)[0]
    ledger = run.carry.ledger
    bad = planted(run, lambda c: c.ledger.balance, ledger.balance - 0.5)
    bad = planted(bad, lambda c: c.ledger.debited, ledger.debited + 0.5)
    assert not balance_conserved(bad).passed


@pytest.mark.parametrize("case", ("baseline", "overflow_log"))
def test_ledger_counters_agree_across_executors(case):
    host, compiled = runs(case, "host")[0], runs(case, "compiled")[0]
    for name in ("debited", "refunded"):
        np.testing.assert_allclose(
            getattr(host.carry.ledger, name),
            getattr(compiled.carry.ledger, name),
            atol=1e-12,
            rtol=0,
        )
