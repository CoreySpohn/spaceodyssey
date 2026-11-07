"""Decision snapshot fingerprints and release versions."""

import dataclasses
import functools
import types

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import spaceodyssey.core.snapshot as snapshot
from spaceodyssey.core import (
    Audience,
    HostExecutor,
    canonical_history,
    index_world,
    init_carry,
    snapshot_fingerprint,
    snapshot_view,
)
from spaceodyssey.studies import catalog, toy
from tests.core.corpus import frozen
from tests.core.helpers import (
    N_STEPS_CORPUS,
    baseline_host_run,
    hand_run,
    one_world,
    worlds,
)

fingerprint_j = eqx.filter_jit(snapshot_fingerprint)
PLAN = ((0, 10), (0, catalog.CATALOG_CHANNEL_ID), (1, 10), (0, 10), (1, 10))


@functools.cache
def _catalog():
    study = catalog.catalog_study(policies=(frozen(plan=PLAN),))
    truths, roots = worlds(1)
    carry, trace = HostExecutor().run(
        study, index_world(truths, 0), roots[0], N_STEPS_CORPUS
    )
    return study, carry, trace


def _fp(carry, study, audience=Audience.POLICY):
    view = snapshot_view(carry, audience, study.caps)
    return int(fingerprint_j(view, study.observer.layout(study.caps)))


def test_snapshot_fingerprint_recorded():
    study, baseline, views = baseline_host_run()
    carry, trace = baseline
    layout = study.observer.layout(study.caps)
    woke = np.asarray(trace.wake)
    fps = np.asarray(trace.snapshot_fp)
    assert woke.sum() == len(views) == 6
    # zero where no decision ran, nonzero and distinct where one did
    assert np.all(fps[~woke] == 0)
    assert np.all(fps[woke] != 0) and len(set(fps[woke].tolist())) == len(views)
    # each recorded value is the fingerprint of the view that decision was handed
    for fp, view in zip(fps[woke], views, strict=True):
        view = jax.tree_util.tree_map(jnp.asarray, view)
        assert int(fingerprint_j(view, layout)) == int(fp)
    # the history carries it, and the compiled executor records the same values
    h = canonical_history(carry, trace, layout)
    assert np.array_equal(h.ints["snapshot_fp"][h.ints["wake"] == 1], fps[woke])
    run = hand_run("baseline")
    assert np.array_equal(run.history(0).ints["snapshot_fp"], h.ints["snapshot_fp"])


@pytest.mark.medium  # Cold catalog execution and fingerprint compilation exceed 5 s.
def test_the_fingerprint_hashes_the_masked_view_not_the_raw_log():
    study, carry, _ = _catalog()
    obs_rows = np.flatnonzero(
        np.asarray(carry.log.channel) == catalog.CATALOG_CHANNEL_ID
    )
    assert len(obs_rows) == 1  # the observer-only row, absent from the policy view
    base = _fp(carry, study)

    def edit(column, rows, value, log=carry.log):
        col = getattr(log, column).at[jnp.asarray(rows)].set(value)
        return dataclasses.replace(carry, log=dataclasses.replace(log, **{column: col}))

    def raw(c):
        view = snapshot_view(c, Audience.POLICY, study.caps)
        layout = study.observer.layout(study.caps)
        return int(fingerprint_j(dataclasses.replace(view, log=c.log), layout))

    # a row the policy cannot see: its integers and its timing do not move it
    hidden = edit("subject", obs_rows, 3)
    assert _fp(hidden, study) == base and raw(hidden) != raw(carry)
    assert _fp(edit("end_tick", obs_rows, 12345), study) == base
    # a scorer-only column of every row, and any float
    n = int(carry.n_log)
    noisy = edit("payload", np.arange(n), carry.log.payload[:n] + 0.25)
    assert _fp(noisy, study) == base
    # a visible row: its identity is part of what the policy read
    seen = int(np.flatnonzero(np.asarray(carry.log.released_tick) >= 0)[0])
    assert _fp(edit("subject", [seen], 2), study) != base
    assert _fp(edit("revision", [seen], 4), study) != base
    # the observer sees the catalog row, so its view of the same edit does move
    obs = _fp(carry, study, Audience.OBSERVER)
    assert _fp(hidden, study, Audience.OBSERVER) != obs


def test_the_fingerprint_follows_the_release_version_and_the_decision():
    study, run, _ = baseline_host_run()
    carry = run.carry
    base = _fp(carry, study)
    for name, value in (("n_dec", carry.n_dec + 1), ("tick", carry.tick + 1)):
        assert _fp(dataclasses.replace(carry, **{name: value}), study) != base
    bumped = dataclasses.replace(carry, release_version=carry.release_version + 1)
    assert _fp(bumped, study) != base


def test_release_versions_count_the_releases_each_audience_can_read():
    study, carry, trace = _catalog()
    log = carry.log
    n = int(carry.n_log)
    policy = int((np.asarray(log.released_tick)[:n] >= 0).sum())
    observer_view = snapshot_view(carry, Audience.OBSERVER, study.caps)
    observer = int(observer_view.n_rows)
    # the observer-only catalog row reaches one audience and not the other
    assert observer == policy + 1 and policy >= 1
    assert int(carry.release_version[1]) == policy
    assert int(carry.release_version[0]) == observer
    # the snapshot of each audience reports its own count
    policy_view = snapshot_view(carry, Audience.POLICY, study.caps)
    assert int(policy_view.release_version) == policy
    assert int(observer_view.release_version) == observer
    # a release moves exactly the versions of the audiences it reaches, by one
    versions = np.asarray(trace.release_version)[np.asarray(trace.handled)]
    steps = np.diff(versions, axis=0, prepend=0)
    assert steps.min() >= 0 and steps.max() <= 1
    kinds = np.asarray(trace.kind)[np.asarray(trace.handled)]
    assert np.all(steps[kinds != 2] == 0)
    assert np.array_equal(steps.sum(axis=0), np.asarray(carry.release_version))


def test_an_empty_leaf_moves_the_fingerprint(monkeypatch):
    """A listed leaf of size zero still changes the digest: its tag and count enter.

    This is why a leaf whose capacity is zero must be left out of the list, as the
    zero-width `product` column is, rather than hashed empty.
    """
    study = toy.survey_1()
    truth, _ = one_world()
    carry = eqx.filter_jit(init_carry)(study, truth)
    view = snapshot_view(carry, Audience.POLICY, study.caps)
    layout = study.observer.layout(study.caps)
    names = (*snapshot.LOG_INT_FIELDS, "valid", "product")
    log = types.SimpleNamespace(
        program=view.log.program, **{n: getattr(view.log, n) for n in names}
    )
    shim = types.SimpleNamespace(
        log=log,
        **{n: getattr(view, n) for n in ("belief", "n_rows", "tick", "n_dec")},
        release_version=view.release_version,
    )
    base = int(snapshot_fingerprint(view, layout))
    assert int(snapshot_fingerprint(shim, layout)) == base
    assert view.log.product.shape[-1] == 0  # the conditional leaf, left out
    log.extra = jnp.zeros(0, jnp.int64)
    monkeypatch.setattr(snapshot, "LOG_INT_FIELDS", (*snapshot.LOG_INT_FIELDS, "extra"))
    assert int(snapshot_fingerprint(shim, layout)) != base
