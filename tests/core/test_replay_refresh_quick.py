"""Replay with refresh on a hand-built log: no executor run, quick tier.

One subject-0 acquisition is released, a checkpoint holds and installs (a boundary that
writes a value derived from the request key), and a correction of the acquisition
arrives at the checkpoint tick's `ADMIT` stage, after the checkpoint. The "run" is the
core's own gather, boundary and install on a hand-written log; the trace is written by
hand. Fails-if lines name what each test guards.
"""

import dataclasses
import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey import core
from spaceodyssey.core import EventKind, replay_run
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.carry import _tree_set, empty_trace
from spaceodyssey.core.enums import F64, I64, Capacity, RecordStatus, RefitStatus, Stage
from spaceodyssey.core.events import empty_event_row
from spaceodyssey.core.executors import replay
from spaceodyssey.core.records import record_none, write_log
from spaceodyssey.core.refresh import AbstractRefitBoundary, RefitResult
from spaceodyssey.studies import toy

ROOT = jnp.asarray([3, 5], jnp.uint32)
ACQUISITION = 1


class KeyedBoundary(AbstractRefitBoundary):
    """Installs a mean that is the request key itself: a refit that reads its key."""

    def refit_batch(self, study, requests):
        mean = requests.keys[..., 0].astype(F64)
        shape = requests.dirty.shape
        return RefitResult(
            slices=(mean, jnp.ones(shape, I64)),
            status=jnp.full(shape, int(RefitStatus.OK), I64),
            diagnostics=jnp.zeros((*shape, 1), F64),
            computed=requests.dirty,
        )


def _measurement(caps, revision, value, tick, subject=0, acquisition=ACQUISITION):
    return dataclasses.replace(
        record_none(caps),
        kind=jnp.asarray(toy.RECORD_MEASUREMENT, I64),
        subject=jnp.asarray(subject, I64),
        channel=jnp.asarray(10, I64),
        status=jnp.asarray(RecordStatus.COMPLETED, I64),
        acquisition_id=jnp.asarray(acquisition, I64),
        revision=jnp.asarray(revision, I64),
        available_tick=jnp.asarray(tick, I64),
        released_tick=jnp.asarray(-1, I64),
        accepted_tick=jnp.asarray(-1, I64),
        superseded_by=jnp.asarray(-1, I64),
        audience=jnp.asarray(core.PUBLIC, I64),
        valid=jnp.asarray(True),
        payload=jnp.array([value, 0.0, 1.0, 0.0]),
    )


def _stamp(c, slot, tick, superseded=None):
    log = c.log
    new = {
        "log.accepted_tick": log.accepted_tick.at[slot].set(tick),
        "log.released_tick": log.released_tick.at[slot].set(tick),
    }
    return _tree_set(c, **new)


def _step(caps, **fields):
    return dataclasses.replace(empty_trace(caps), handled=jnp.asarray(True), **fields)


@functools.cache
def study():
    caps = Capacities(rows_per_subject=1, refit_lanes=1, n_log=4)
    return toy.survey_refresh(100, caps=caps, policies=(toy.NullPolicy(),))


def build(scenario="normal"):
    """`(carry, trace)` of the hand-built run: release, hold, completion, result.

    `scenario` is `normal`, `ids` (the id allocator is exhausted at the checkpoint),
    `rows` (the log has no room for the product rows) or `cut` (the run ends on the
    hold, as a step budget would end it).
    """
    s = study()
    caps = s.caps
    layout = s.observer.layout(caps)
    c = core.empty_carry(caps)
    c = _tree_set(
        c, belief=s.observer.init(caps), pbelief=s.observer.init_summary(caps)
    )
    c, _ = write_log(c, _measurement(caps, 0, 2.0, 10), caps)
    c = _stamp(c, 0, 10)
    if scenario == "rows":  # two unreleased rows of another subject fill the log
        for k in (1, 2):
            c, _ = write_log(c, _measurement(caps, 0, 1.0, 20, 1, 1 + k), caps)
    c = _tree_set(
        c,
        refresh_dirty=c.refresh_dirty.at[:, 0].set(True),
        next_sid=jnp.asarray(caps.max_sid if scenario == "ids" else 7, I64),
    )
    n_hold = int(c.n_log)
    held = dataclasses.replace(
        empty_event_row(),
        tick=jnp.asarray(100, I64),
        acquisition_index=jnp.asarray(1, I64),
        sid=jnp.asarray(-1, I64),
        valid=jnp.asarray(True),
    )
    c = _tree_set(c, refresh=held)
    request = core.refresh._gather(c, ROOT, s)
    batch = jax.tree_util.tree_map(lambda x: x[None], request)
    one = jax.tree_util.tree_map(lambda x: x[0], KeyedBoundary().refit_batch(s, batch))
    done = core.refresh._commit(c, request, one, s)
    mean, aux = layout.project(done.belief)
    rows = [
        _step(
            caps,
            kind=jnp.asarray(EventKind.RELEASE, I64),
            tick=jnp.asarray(10, I64),
            stage=jnp.asarray(Stage.RELEASE, I64),
            channel=jnp.asarray(10, I64),
            audience=jnp.asarray(core.PUBLIC, I64),
            n_log=jnp.asarray(1, I64),
        ),
        _step(
            caps,
            kind=jnp.asarray(EventKind.REFRESH, I64),
            tick=jnp.asarray(100, I64),
            stage=jnp.asarray(Stage.ANALYZE, I64),
            acquisition_index=jnp.asarray(1, I64),
            n_log=jnp.asarray(n_hold, I64),
        ),
    ]
    if scenario == "cut":
        return _tree_set(
            done, refresh=empty_event_row(), belief=c.belief, pbelief=c.pbelief
        ), _stack(rows)
    wrote = int(done.n_log) - n_hold
    rows.append(
        _step(
            caps,
            kind=jnp.asarray(EventKind.REFRESH, I64),
            tick=jnp.asarray(100, I64),
            stage=jnp.asarray(Stage.ANALYZE, I64),
            acquisition_index=jnp.asarray(1, I64),
            n_written=jnp.asarray(wrote, I64),
            n_log=done.n_log,
            belief_mean=mean,
            belief_aux=aux,
        )
    )
    if scenario == "normal":
        correction = _measurement(caps, 1, 9.0, 100)
        done, _ = write_log(done, correction, caps)
        slot = int(done.n_log) - 1
        done = _stamp(done, slot, 100)
        done = _tree_set(
            done, **{"log.superseded_by": done.log.superseded_by.at[0].set(1)}
        )
        rows.append(
            _step(
                caps,
                kind=jnp.asarray(EventKind.RESULT, I64),
                tick=jnp.asarray(100, I64),
                stage=jnp.asarray(Stage.ADMIT, I64),
                channel=jnp.asarray(10, I64),
                log_slot=jnp.asarray(slot, I64),
                audience=jnp.asarray(core.PUBLIC, I64),
                n_log=done.n_log,
            )
        )
    return done, _stack(rows)


def _stack(rows):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *rows)


@functools.cache
def built(scenario="normal"):
    return build(scenario)


def replayed(scenario="normal", carry=None):
    base, trace = built(scenario)
    carry = base if carry is None else carry
    return replay_run(study(), ROOT, carry, trace, refit=KeyedBoundary())


def test_replay_of_the_hand_log_reproduces_the_run():
    """Fails if a hold, install, product row or final belief of the hand log differs."""
    out = replayed()
    assert out.checkpoints == 1 and out.products == 2
    assert not out.refused and not out.unfinished
    # the installed mean is the key, so the replay used the run's key
    assert float(out.belief.mean[0]) > 100.0


def test_a_replayer_that_reads_the_final_superseded_by_is_caught(monkeypatch):
    """Fails if hiding the rows the final log marks superseded still matches."""
    replayed()
    real = replay._visible

    def reads_final(carry, n_log, accepted, released):
        cols = real(carry, n_log, accepted, released)
        final = jnp.asarray(carry.log.superseded_by) < 0
        return {**cols, "log.valid": cols["log.valid"] & final}

    monkeypatch.setattr(replay, "_visible", reads_final)
    with pytest.raises(ValueError, match="checkpoint"):
        replayed()


def test_a_replayer_that_keys_a_refit_by_the_log_length_is_caught(monkeypatch):
    """Fails if a refit key taken from the log length still reproduces the run."""
    real = replay._request

    def by_length(view, root, study):
        request = real(view, root, study)
        count = jnp.asarray(view.n_log, request.keys.dtype)
        keys = jnp.broadcast_to(root + count, request.keys.shape)
        return eqx.tree_at(lambda r: r.keys, request, keys)

    monkeypatch.setattr(replay, "_request", by_length)
    with pytest.raises(ValueError, match="belief_mean"):
        replayed()


@pytest.mark.parametrize(
    "field, rows, message",
    [
        ("acquisition_id", {2: 7}, "checkpoint 1"),
        ("acquisition_id", {1: 1, 2: 2}, "checkpoint 1"),
        ("phase", {1: 1}, "checkpoint 1"),
        ("vintage", {2: 1}, "checkpoint 1"),
    ],
    ids=["duplicate_id", "id_taken_by_a_source_row", "phase", "vintage"],
)
def test_corrupted_product_provenance_is_caught(field, rows, message):
    """Fails if a duplicated or stale reserved id, a wrong phase or a wrong vintage on
    a product row still replays."""
    carry, _ = built()
    column = vars(carry.log)[field]
    for slot, value in rows.items():
        column = column.at[slot].set(value)
    bad = _tree_set(carry, **{f"log.{field}": column})
    with pytest.raises(ValueError, match=message):
        replayed(carry=bad)


@pytest.mark.parametrize(
    "scenario, bits",
    [("ids", Capacity.STABLE_IDS), ("rows", Capacity.LOG_ROWS)],
)
def test_a_refused_install_is_reported_as_a_refusal(scenario, bits):
    """Fails if a refusal (no ids, no log room) replays as a budget cut or silently."""
    out = replayed(scenario)
    assert out.refused == ((1, int(bits)),)
    assert not out.unfinished and out.products == 0
    assert out.overflow_mask & int(bits)
    carry, _ = built(scenario)
    forged = _tree_set(carry, overflow_mask=jnp.asarray(0, I64))
    with pytest.raises(ValueError, match="refused in the replay only"):
        replayed(scenario, carry=forged)


def test_a_run_cut_on_a_hold_is_unfinished_not_refused():
    """Fails if a hold with no completion is reported as a refusal or an install."""
    out = replayed("cut")
    assert out.unfinished and not out.refused
    assert out.checkpoints == 0 and out.products == 0
    assert float(np.abs(np.asarray(out.belief.mean)).sum()) == 0.0
