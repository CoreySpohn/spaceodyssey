"""The observer contract: capabilities, refit declarations, keyed folds and refits."""

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from spaceodyssey.core import (
    PUBLIC,
    Audience,
    HostExecutor,
    ObserverCapabilities,
    PayloadColumn,
    Purpose,
    RecordKind,
    RecordStatus,
    draw_key,
    fold_key,
    masked_record,
    replay_belief,
)
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.conformance.folds import CountingObserver, fold_counts, plain
from spaceodyssey.core.executors.replay import _row
from spaceodyssey.studies import toy
from tests.core.helpers import one_world

N_STEPS = 40
REFIT = ObserverCapabilities(refit=True, refit_kinds=("measurement",))


class RefitMean(toy.ToyRunningMean):
    """The running mean, declared refit-capable on measurement rows."""

    def capabilities(self):
        return REFIT


class RefitOnlyMean(toy.ToyRunningMean):
    """A running mean that declares no incremental folds."""

    def capabilities(self):
        return ObserverCapabilities(
            incremental=False, refit=True, refit_kinds=("measurement",)
        )


class IgnoresValid(toy.ToyRunningMean):
    """A running mean whose fold does not check `valid`, so a caller must mask."""

    def fold(self, belief, rec, caps, key=None):
        return super().fold(belief, dataclasses.replace(rec, valid=True), caps)


def _registry_with_product():
    registry = toy.toy_registry()
    registry.register(
        RecordKind("product", (PayloadColumn("value", PUBLIC),), derived=True)
    )
    return registry


@pytest.fixture(scope="module")
def sample_run():
    """One host run of the toy survey through the sampling observer."""
    study = toy.survey_1(observer=toy.ToySampleObserver())
    truth, root = one_world()
    return study, root, HostExecutor().run(study, truth, root, N_STEPS)


def _observer_rows(study, carry, subject):
    """The rows of `subject` the observer accepted, unmasked, in log order."""
    log = carry.log
    take = (
        np.asarray(log.valid)
        & (np.asarray(log.accepted_tick) >= 0)
        & (np.asarray(log.subject) == subject)
        & (np.asarray(log.kind) == toy.RECORD_MEASUREMENT)
    )
    return [_row(log, int(i)) for i in np.flatnonzero(take)]


def _stack(rows):
    return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *rows)


@pytest.mark.parametrize(
    "kw",
    [
        dict(incremental=False),  # never changes
        dict(refit=True),  # refits nothing
        dict(refit_kinds=("measurement",)),  # names kinds without a refit
        dict(refit=True, refit_kinds=("measurement",), refit_statuses=(99,)),
    ],
)
def test_an_inconsistent_capability_declaration_is_refused(kw):
    """Fails if an observer may declare a refit it cannot run or a bogus status."""
    with pytest.raises(ValueError):
        ObserverCapabilities(**kw)
    # control: the consistent declaration is accepted with the default statuses
    assert REFIT.refit_statuses == (RecordStatus.COMPLETED, RecordStatus.PARTIAL)


def test_lowering_stores_refit_kinds_and_refuses_derived_or_unregistered():
    """Fails if a refit kind lowers to the wrong index or a bad name lowers at all."""
    registry = _registry_with_product()
    caps = registry.lower(Capacities(), RefitMean())
    assert caps.refit_kinds == (registry.index("measurement"),)
    assert caps.refit_statuses == (RecordStatus.COMPLETED, RecordStatus.PARTIAL)
    assert caps.derived_kinds == (registry.index("product"),)
    # an observer that does not refit lowers nothing
    assert registry.lower(Capacities()).refit_kinds == ()
    for name, match in (("product", "derived"), ("missing", "not a registered")):

        class Naming(toy.ToyRunningMean):
            def capabilities(self, name=name):
                return ObserverCapabilities(refit=True, refit_kinds=(name,))

        with pytest.raises(ValueError, match=match):
            registry.lower(Capacities(), Naming())


@pytest.mark.parametrize(
    "case, match",
    [
        ("not_lowered", "do not hold the observer's refit"),
        ("derived", "derived"),
        ("unregistered", "not a registered"),
        ("refit_only", "not incremental"),
    ],
)
def test_study_refuses_a_refit_declaration_it_cannot_honour(case, match):
    """Fails if a study runs with refit kinds its capacities do not hold, or bad ones.

    The capacities are hand-built here so the study's own check, not the registry's
    lowering, is what refuses.
    """
    registry = _registry_with_product()
    observer = RefitMean()
    lowered = registry.lower(Capacities(), observer)
    measurement = registry.index("measurement")
    caps = {
        "not_lowered": registry.lower(Capacities()),
        "derived": dataclasses.replace(lowered, derived_kinds=(measurement,)),
        "unregistered": dataclasses.replace(lowered, kind_names=lowered.kind_names[:1]),
        "refit_only": lowered,
    }[case]
    if case == "refit_only":
        observer = RefitOnlyMean()
    with pytest.raises(ValueError, match=match):
        toy.survey_1(caps=caps, observer=observer)
    # control: the properly lowered declaration builds
    toy.survey_1(caps=lowered, observer=RefitMean())


def test_each_fold_draws_from_its_records_inference_key(sample_run):
    """Fails if the core passes `fold` no key, another purpose's key, or a key of
    another record; or if the summary leaves the belief's path.

    The compiled executor's fold variates are checked against the host's in
    `test_draws.py::test_compiled_fold_variates_equal_the_host_variates`.
    """
    study, root, run = sample_run
    carry = run.carry
    draws = np.asarray(carry.belief.draws)
    n_checked = 0
    for subject in range(study.caps.n_subjects):
        for row in _observer_rows(study, carry, subject):
            index = int(row.acquisition_index)
            want = jr.normal(fold_key(root, row, study), dtype=jnp.float64)
            assert draws[subject, index] == float(want)
            other = draw_key(root, row, Purpose.MEASUREMENT)
            assert draws[subject, index] != float(jr.normal(other, dtype=jnp.float64))
            n_checked += 1
    assert n_checked >= 4
    # every row is public, so the summary the policy folded is the belief's summary
    summary = study.observer.summary(carry.belief)
    for got, want in zip(
        jax.tree_util.tree_leaves(carry.pbelief),
        jax.tree_util.tree_leaves(summary),
        strict=True,
    ):
        np.testing.assert_array_equal(got, want)


def test_replay_with_the_world_key_reproduces_a_keyed_belief(sample_run):
    """Fails if replay folds a keyed observer without the run's fold keys."""
    study, root, run = sample_run
    replayed = replay_belief(study, run.carry, run.trace, root)
    for got, want in zip(
        jax.tree_util.tree_leaves(replayed),
        jax.tree_util.tree_leaves(run.carry.belief),
        strict=True,
    ):
        np.testing.assert_array_equal(got, want)
    # control: without the key the sampling observer cannot fold
    with pytest.raises(ValueError, match="fold key"):
        replay_belief(study, run.carry, run.trace)


def test_default_refit_then_install_equals_the_incremental_belief(sample_run):
    """Fails if the default refit folds padding rows, skips a row, or `install`
    writes another subject's values.
    """
    study, _, run = sample_run
    caps, observer = study.caps, IgnoresValid()
    subject = 1
    rows = [
        masked_record(r, caps, Audience.OBSERVER)
        for r in _observer_rows(study, run.carry, subject)
    ]
    assert len(rows) >= 2
    pad = dataclasses.replace(rows[0], valid=jnp.asarray(False))
    zero = observer.init(caps)
    key = jr.PRNGKey(0)

    def refit_installed(rows):
        fitted = observer.refit(zero, _stack(rows), caps, key)
        return observer.analyze(observer.install(zero, fitted, subject))

    got = refit_installed([rows[0], pad, *rows[1:], pad])
    assert float(got.mean[subject]) == float(run.carry.belief.mean[subject])
    assert int(got.n[subject]) == int(run.carry.belief.n[subject])
    others = np.arange(caps.n_subjects) != subject
    assert not np.asarray(got.n)[others].any()
    # control: a refit missing one row does not match
    assert int(refit_installed(rows[:-1]).n[subject]) != int(got.n[subject])


def test_the_fold_counter_delegates_every_observer_method(sample_run):
    """Fails if the counter drops the fold key, counts a refit or an install as a
    fold, or answers a method from the default instead of the inner observer.
    """
    study, root, run = sample_run
    caps = study.caps
    counter = CountingObserver(inner=RefitSample())
    inner = counter.inner
    assert counter.capabilities() == inner.capabilities() == REFIT
    (row, *_) = _observer_rows(study, run.carry, 0)
    rec = masked_record(row, caps, Audience.OBSERVER)
    key = fold_key(root, row, study)

    def same(a, b):
        for x, y in zip(
            jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True
        ):
            np.testing.assert_array_equal(x, y)

    belief = counter.fold(counter.init(caps), rec, caps, key=key)
    same(plain(belief), inner.fold(inner.init(caps), rec, caps, key=key))
    summary = counter.fold_summary(counter.init_summary(caps), rec, caps, key=key)
    same(plain(summary), inner.fold_summary(inner.init_summary(caps), rec, caps))
    same(plain(counter.summary(belief)), inner.summary(plain(belief)))
    counts = fold_counts(belief)
    fitted = counter.refit(counter.init(caps), _stack([rec]), caps, key)
    same(plain(fitted), inner.refit(inner.init(caps), _stack([rec]), caps, key))
    installed = counter.install(belief, fitted, 0)
    same(plain(installed), inner.install(plain(belief), plain(fitted), 0))
    same(fold_counts(installed), counts)
    assert not np.asarray(fold_counts(fitted)[1]).any()
    put = counter.install_summary(summary, counter.summary(fitted), 0)
    same(
        plain(put),
        inner.install_summary(plain(summary), inner.summary(plain(fitted)), 0),
    )
    same(fold_counts(put), fold_counts(summary))


class RefitSample(toy.ToySampleObserver):
    """The sampling observer, declared refit-capable."""

    def capabilities(self):
        return REFIT
