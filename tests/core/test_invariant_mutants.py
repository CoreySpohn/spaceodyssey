"""Boundary checks reject missing evidence and committed-state corruption."""

import dataclasses
import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import handlers, transaction
from spaceodyssey.core.conformance import run_case
from spaceodyssey.core.conformance.invariants import (
    balance_conserved,
    dense_acquisition_index,
    no_scorer_column_in_snapshot,
    one_fold_per_acquisition,
)
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
