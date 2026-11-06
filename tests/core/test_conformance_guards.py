"""Conformance checks reject incomplete accounting and changed public rows."""

import dataclasses
import inspect

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import canonical_history, handlers
from spaceodyssey.core.conformance import public_events, run_case
from spaceodyssey.core.conformance.invariants import (
    balance_conserved,
    dense_acquisition_index,
    no_extra_wakes_under_private_events,
    one_fold_per_acquisition,
)
from spaceodyssey.core.enums import Capacity, EventKind
from spaceodyssey.core.events import where_tree
from spaceodyssey.studies import catalog
from spaceodyssey.studies.conformance import CORPUS_BY_NAME, RECORD_AUXILIARY
from tests.core.test_corpus_invariants import planted, runs


@pytest.mark.parametrize("executor", ("host", "compiled"))
def test_positive_count_requires_retained_acquisitions(executor):
    run = runs("bundle_unaffordable", executor)[0]
    bad = planted(
        run, lambda c: c.acquisition_counts, run.carry.acquisition_counts.at[1].set(2)
    )
    assert not dense_acquisition_index(bad)


@pytest.mark.parametrize("executor", ("host", "compiled"))
@pytest.mark.parametrize("check", (balance_conserved, dense_acquisition_index))
def test_stable_id_exhaustion_keeps_accounting_checks(executor, check):
    original = CORPUS_BY_NAME["baseline"]

    def small_ids():
        study = original.make_study()
        return dataclasses.replace(
            study, caps=dataclasses.replace(study.caps, max_sid=2)
        )

    case = dataclasses.replace(original, make_study=small_ids)
    (run,) = run_case(case, executor, n_worlds=1, probe=False)
    assert int(run.carry.overflow_mask) == int(Capacity.STABLE_IDS)
    if check is balance_conserved:
        bad = planted(run, lambda c: c.ledger.balance, run.carry.ledger.balance + 0.5)
    else:
        bad = planted(
            run,
            lambda c: c.acquisition_counts,
            run.carry.acquisition_counts.at[0].add(1),
        )
    assert not check(bad)


@pytest.mark.parametrize("executor", ("host", "compiled"))
def test_private_events_preserve_public_row_identities(executor):
    run = runs("coincident", executor)[0]
    trace = run.trace
    i, j = np.flatnonzero(
        np.asarray(trace.handled)
        & (np.asarray(trace.tick) == 160)
        & (np.asarray(trace.kind) == EventKind.DONE)
    )
    changed = eqx.tree_at(
        lambda t: t.lane,
        trace,
        trace.lane.at[i].set(trace.lane[j]).at[j].set(trace.lane[i]),
    )
    bad = dataclasses.replace(
        run, probe=(run.wakes, run.decisions, public_events(changed))
    )
    assert not no_extra_wakes_under_private_events(bad)


@pytest.mark.parametrize("executor", ("host", "compiled"))
def test_balance_must_be_finite(executor):
    run = runs("baseline", executor)[0]
    bad = planted(
        run, lambda c: c.ledger.balance, run.carry.ledger.balance.at[0, 0].set(jnp.nan)
    )
    assert not balance_conserved(bad)


@pytest.mark.parametrize("executor", ("host", "compiled"))
def test_refused_bundle_preserves_stable_ids(executor):
    run = runs("bundle_unaffordable", executor)[0]
    bad = planted(run, lambda c: c.next_sid, run.carry.next_sid + 2)
    layout = run.study.observer.layout(run.study.caps)
    history = canonical_history(bad.carry, bad.trace, layout)
    assert not all(check(history) for check in run.case.invariants)


@pytest.mark.parametrize("executor", ("host", "compiled"))
def test_catalog_case_requires_its_kind_and_contribution(executor):
    original = CORPUS_BY_NAME["catalog_record"]

    def wrong_kind():
        study = original.make_study()
        return eqx.tree_at(
            lambda s: s.channels.kind,
            study,
            study.channels.kind.at[3].set(RECORD_AUXILIARY),
        )

    (run,) = run_case(
        dataclasses.replace(original, make_study=wrong_kind),
        executor,
        n_worlds=1,
        probe=False,
    )
    assert not all(check(run.history) for check in run.case.invariants)


@pytest.mark.parametrize("executor", ("host", "compiled"))
def test_release_handler_cannot_fold_an_auxiliary_acquisition_twice(
    executor, monkeypatch
):
    def fold_twice(observer, belief, rec, caps):
        once = observer.fold(belief, rec, caps)
        twice = observer.fold(once, rec, caps)
        return where_tree(rec.kind == 3, twice, once)

    original = handlers.h_release
    source = inspect.getsource(original).replace(
        "study.observer.fold(c.belief, obs, caps)",
        "fold_twice(study.observer, c.belief, obs, caps)",
    )
    namespace = dict(vars(handlers), fold_twice=fold_twice)
    exec(compile(source, "<double_fold>", "exec"), namespace)
    monkeypatch.setattr(
        handlers,
        "HANDLERS",
        tuple(
            namespace["h_release"] if h is original else h for h in handlers.HANDLERS
        ),
    )

    class LocalObserver(catalog.CatalogObserver):
        pass

    case = CORPUS_BY_NAME["catalog_record"]

    def study():
        base = case.make_study()
        return dataclasses.replace(
            base, observer=LocalObserver(catalog_kind=base.observer.catalog_kind)
        )

    (run,) = run_case(
        dataclasses.replace(case, make_study=study), executor, n_worlds=1, probe=False
    )
    assert not bool(run.carry.dup_fold)
    assert int(run.carry.belief.n.sum()) == 5
    result = one_fold_per_acquisition(run)
    assert not result
    assert "folded 2 times" in result.diagnostic
