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
    MeanBelief,
    ObserverCapabilities,
    PayloadColumn,
    Purpose,
    RecordKind,
    RecordStatus,
    draw_key,
    empty_log,
    fold_key,
    fold_keys,
    masked_record,
    refit_rows,
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


@pytest.mark.medium(reason="over 3 s cold here, about 5 s on the runner")
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


def _release_order(study, carry, subject):
    """The subject's observer rows in the order the observer accepted them."""
    rows = _observer_rows(study, carry, subject)
    return sorted(rows, key=lambda r: int(r.accepted_tick))


def _pad(row):
    """A padding row: not valid, with a zero payload."""
    return dataclasses.replace(
        row, valid=jnp.asarray(False), payload=jnp.zeros_like(row.payload)
    )


def _refit(observer, study, root, rows, audience=Audience.OBSERVER):
    """`observer.refit` of `rows` from its initial belief, with the run's fold keys."""
    stack = _stack(rows)
    caps = study.caps
    return observer.refit(
        observer.init(caps),
        masked_record(stack, caps, audience),
        caps,
        jr.PRNGKey(1),
        fold_keys(root, stack, study),
    )


def test_keyed_refit_of_the_run_rows_equals_the_incremental_belief(sample_run):
    """Fails if the default refit keys a row other than as its incremental fold, folds
    padding, drops the final valid row, reverses the rows, or `install` writes
    another subject.
    """
    study, root, run = sample_run
    caps, observer = study.caps, study.observer
    subject = 0
    rows = _release_order(study, run.carry, subject)
    assert len(rows) >= 2
    pad = _pad(rows[0])

    def refit_installed(rows):
        fitted = _refit(observer, study, root, rows)
        return observer.analyze(observer.install(observer.init(caps), fitted, subject))

    want = run.carry.belief
    got = refit_installed([pad, rows[0], pad, *rows[1:]])
    for leaf in ("mean", "n", "sample", "draws"):
        assert np.array_equal(getattr(got, leaf)[subject], getattr(want, leaf)[subject])
    others = np.arange(caps.n_subjects) != subject
    assert not np.asarray(got.n)[others].any()
    # controls: a reversed order and a dropped final row each change the belief
    reversed_ = refit_installed([pad, *rows[::-1]])
    assert float(reversed_.sample[subject]) != float(want.sample[subject])
    dropped = refit_installed([pad, rows[0], pad, *rows[1:-1]])
    assert int(dropped.n[subject]) == int(want.n[subject]) - 1


class CountsAnalyses(toy.ToyRunningMean):
    """A running mean whose analysis is not idempotent: it counts its own calls."""

    def analyze(self, belief):
        return MeanBelief(belief.mean, belief.n, belief.aux.at[1].add(1.0))


def test_refit_returns_before_the_final_analysis(sample_run):
    """Fails if the default refit analyzes after the last fold (the caller analyzes
    again) or skips the analysis between folds that the incremental path runs.
    """
    study, root, run = sample_run
    caps, observer = study.caps, CountsAnalyses()
    rows = _release_order(study, run.carry, 0)
    belief = observer.init(caps)
    for row in rows:
        rec = masked_record(row, caps, Audience.OBSERVER)
        belief = observer.analyze(observer.fold(belief, rec, caps))
    got = observer.analyze(_refit(observer, study, root, [*rows, _pad(rows[0])]))
    assert float(got.aux[1]) == float(belief.aux[1]) == len(rows)
    assert float(got.mean[0]) == float(belief.mean[0])


class InverseGain(toy.ToyRunningMean):
    """Folds `gain / value` and ignores `valid`: zero padding divides by zero."""

    gain: jax.Array = 1.0

    def fold(self, belief, rec, caps, key=None):
        value = self.gain / rec.payload[0]
        rec = dataclasses.replace(rec, valid=True, payload=rec.payload.at[0].set(value))
        return super().fold(belief, rec, caps)


def test_padding_rows_neither_count_nor_poison_gradients(sample_run):
    """Fails if a padding row is folded, or if its fold, evaluated and discarded,
    puts a NaN into the gradient of a refit batched over both audiences.
    """
    study, root, run = sample_run
    caps = study.caps
    rows = _release_order(study, run.carry, 0)
    seq = [_pad(rows[0]), *rows, _pad(rows[0])]
    stack = _stack(seq)
    batch = jax.tree_util.tree_map(
        lambda *x: jnp.stack(x),
        masked_record(stack, caps, Audience.OBSERVER),
        masked_record(stack, caps, Audience.POLICY),
    )
    keys = jnp.stack([fold_keys(root, stack, study)] * 2)

    def mean(gain):
        observer = InverseGain(gain=gain)

        def one(rows, row_keys):
            belief = observer.init(caps)
            return observer.refit(belief, rows, caps, jr.PRNGKey(1), row_keys)

        fitted = jax.vmap(one)(batch, keys)
        return fitted.mean[:, 0].sum(), fitted.n[:, 0]

    (value, n), grad = jax.value_and_grad(mean, has_aux=True)(1.0)
    values = np.array([float(r.payload[0]) for r in rows])
    assert np.array_equal(n, [len(rows), len(rows)])
    np.testing.assert_allclose(grad, 2 * np.mean(1 / values), rtol=1e-12)
    np.testing.assert_allclose(value, grad, rtol=1e-12)


def _hand_log(caps, rows):
    """A record log whose rows set the named columns; every row is valid."""
    log = empty_log(caps)
    defaults = dict(accepted_tick=-1, released_tick=-1, kind=toy.RECORD_MEASUREMENT)
    cols = {}
    for name in ("acquisition_id", "subject", "channel", "acquisition_index"):
        cols[name] = [r.get(name, 0) for r in rows]
    for name in ("kind", "status", "revision", "accepted_tick", "released_tick"):
        cols[name] = [r.get(name, defaults.get(name, 0)) for r in rows]
    n = len(rows)
    out = {
        k: getattr(log, k).at[:n].set(jnp.asarray(v, jnp.int64))
        for k, v in cols.items()
    }
    payload = log.payload.at[:n, 0].set(jnp.asarray([r["value"] for r in rows]))
    return dataclasses.replace(
        log, **out, payload=payload, valid=log.valid.at[:n].set(True)
    )


FAILED, COMPLETED = RecordStatus.FAILED, RecordStatus.COMPLETED
# acquisition 1 is corrected after acceptance by a failed revision the policy never
# receives; 3 is not a refit kind; 4 is another subject; 5 is released, not accepted
REVISION_ROWS = [
    dict(acquisition_id=1, channel=10, value=2.0, accepted_tick=5, released_tick=5),
    dict(acquisition_id=1, channel=10, revision=1, value=9.0, accepted_tick=8),
    dict(
        acquisition_id=2,
        channel=11,
        acquisition_index=1,
        value=5.0,
        accepted_tick=6,
        released_tick=6,
    ),
    dict(
        acquisition_id=3,
        channel=10,
        acquisition_index=2,
        value=1.0,
        kind=toy.RECORD_AUXILIARY,
        accepted_tick=6,
        released_tick=6,
    ),
    dict(acquisition_id=4, subject=1, value=3.0, accepted_tick=6, released_tick=6),
    dict(acquisition_id=5, channel=12, acquisition_index=3, value=4.0, released_tick=7),
]


@pytest.mark.parametrize("correction", [FAILED, COMPLETED])
def test_refit_rows_take_the_current_revision_then_eligibility(correction):
    """Fails if selection checks status before taking the current revision (a failed
    correction would leave the corrected value), keeps a superseded revision, or
    reads rows the audience has not received.
    """
    caps = _registry_with_product().lower(Capacities(), RefitMean())
    rows = [dict(r) for r in REVISION_ROWS]
    rows[1]["status"] = correction
    log = _hand_log(caps, rows)
    observer = RefitMean()
    keys = jnp.zeros((caps.n_log, 2), jnp.uint32)

    def selected(audience):
        picked = refit_rows(log, caps, audience, 0)
        fitted = observer.refit(observer.init(caps), picked, caps, None, keys)
        valid = np.asarray(picked.valid)
        return np.asarray(picked.payload[valid, 0]).tolist(), float(fitted.mean[0])

    # the observer: the failed correction retracts acquisition 1; a completed one
    # replaces its value
    want = [5.0] if correction == FAILED else [9.0, 5.0]
    assert selected(Audience.OBSERVER) == (want, float(np.mean(want)))
    # the policy received only the first revision, and acquisition 5
    assert selected(Audience.POLICY) == ([2.0, 5.0, 4.0], 11.0 / 3.0)


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
    row_keys = fold_keys(root, _stack([row]), study)
    fitted = counter.refit(counter.init(caps), _stack([rec]), caps, key, row_keys)
    same(
        plain(fitted),
        inner.refit(inner.init(caps), _stack([rec]), caps, key, row_keys),
    )
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


def test_policy_refit_rows_ignore_corrections_it_never_received():
    """Fails if an observer-only revision still marks the earlier public row for the
    policy, so that a metadata-sensitive refit sees a different input.
    """
    caps = _registry_with_product().lower(Capacities(), RefitMean())
    rows = [dict(r) for r in REVISION_ROWS]
    with_fix = _hand_log(caps, rows)
    without = _hand_log(caps, [r for i, r in enumerate(rows) if i != 1])
    a = refit_rows(with_fix, caps, Audience.POLICY, 0)
    b = refit_rows(without, caps, Audience.POLICY, 0)
    keep = np.asarray(a.valid)
    assert keep.sum() == np.asarray(b.valid).sum() > 0
    for x, y in zip(
        jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(x)[keep], np.asarray(y)[keep])
    assert (np.asarray(a.superseded_by)[keep] == -1).all()
    # control: the observer does see the correction
    seen = refit_rows(with_fix, caps, Audience.OBSERVER, 0)
    assert (np.asarray(seen.revision)[np.asarray(seen.valid)] > 0).any()
