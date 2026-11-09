"""A version-2 channel model receives a context and returns the whole payload row.

A version-1 model keeps its call and its compiled program; the history of every
version-1 study is pinned by the golden file.
"""

import dataclasses
import json

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from spaceodyssey.codec import dump_study, load_study
from spaceodyssey.core import battery
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import (
    F64,
    I64,
    PUBLIC,
    Disposition,
    EffectKind,
    RecordStatus,
    Scope,
)
from spaceodyssey.core.executors.base import histories_equal
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.core.kinds import RealizeContext, realize_row
from spaceodyssey.core.records import (
    KindRegistry,
    PayloadColumn,
    RecordKind,
    record_none,
)
from spaceodyssey.studies import toy
from tests.codec.test_document import TOY_KINDS
from tests.core.battery_helpers import _table
from tests.core.helpers import one_world, worlds
from tests.doubles import ToyEpochPhotometry

KINDS = {**TOY_KINDS, ("channel_model", "test.epoch_photometry", 2): ToyEpochPhotometry}
PLAN = ((0, 10), (1, 10), (0, 10), (1, 10), (0, 10))
N_STEPS = 40


class _CancelOnWake(toy.NullPolicy):
    """Observe and wake 60 ticks later, then cancel the live occupancy and stop."""

    def declare(self):
        return Declaration(stops=((Scope.CAMPAIGN, Disposition.DRAIN),))

    def decide(self, pstate, view, study, key):
        first = view.n_dec == 0
        kind = jnp.where(
            first,
            EffectKind.OBSERVE,
            jnp.where(view.n_dec == 1, EffectKind.CANCEL, EffectKind.STOP),
        )
        eff = _table(
            study.caps,
            kind,
            jnp.where(first, EffectKind.WAIT, EffectKind.NONE),
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
            channel_index=0,
            subject=0,
        )
        eff = dataclasses.replace(eff, tick=eff.tick.at[1].set(60))
        return eff, pstate, jnp.asarray(0.0, F64)


def _epoch_registry():
    """The toy kinds with the measurement layout widened to the epoch row."""
    cols = tuple(
        PayloadColumn(n, PUBLIC)
        for n in ("start", "end", "acquisition_index", "fraction", "variate")
    )
    return KindRegistry(
        (RecordKind("none", ()), RecordKind("measurement", cols), toy.AUXILIARY_KIND)
    )


def _v2(**kw):
    """The five-acquisition toy study with the version-2 model, optionally delayed."""
    caps = _epoch_registry().lower(Capacities())
    study = toy.survey_1(caps, **kw)
    return dataclasses.replace(study, channel_model=ToyEpochPhotometry())


def _delayed(wait_ticks):
    policy = toy.FixedPlanPolicy(plan=PLAN, wait_at=1, wait_ticks=wait_ticks)
    return _v2(policies=(policy,))


def _completed(history_run):
    """Start tick and payload of each completed record, in log order."""
    log = history_run.carry.log
    keep = np.asarray(log.valid) & (np.asarray(log.status) == RecordStatus.COMPLETED)
    return np.asarray(log.start_tick)[keep], np.asarray(log.payload)[keep]


def _base(study):
    return dataclasses.replace(
        record_none(study.caps),
        subject=jnp.asarray(0, I64),
        channel_index=jnp.asarray(0, I64),
        start_tick=jnp.asarray(7, I64),
        end_tick=jnp.asarray(107, I64),
    )


def test_a_version_two_row_is_the_context_function_of_the_record():
    """Fails if a context field is not the record's, or the row is not padded."""
    study = _v2()
    truth, _ = one_world()
    base = dataclasses.replace(_base(study), acquisition_index=jnp.asarray(4, I64))
    fraction = jnp.asarray(0.5, F64)
    row = realize_row(study, truth, base, fraction, jr.PRNGKey(3))
    assert row.shape == (study.caps.payload_width,)
    assert [float(x) for x in row[:4]] == [7.0, 107.0, 4.0, 0.5]
    ctx = RealizeContext.from_record(base, fraction)
    assert (int(ctx.start_tick), int(ctx.end_tick), int(ctx.acquisition_index)) == (
        7,
        107,
        4,
    )


@pytest.mark.parametrize(
    ("version", "row", "match"),
    [(2, jnp.zeros(99), "1-D row"), (0, None, "version 0"), (3, None, "version 3")],
)
def test_a_wide_row_or_an_unknown_version_is_refused(version, row, match):
    """Fails if a too-wide row is truncated or an unknown version gets the v2 call."""

    class Odd(ToyEpochPhotometry):
        def realize(self, *args, context=None, **kw):
            return row

    study = dataclasses.replace(toy.survey_1(), channel_model=Odd(version=version))
    truth, root = one_world()
    with pytest.raises(ValueError, match=match):
        realize_row(study, truth, _base(study), jnp.asarray(1.0, F64), root)


@pytest.mark.medium(reason="one host and one compiled run of the toy survey, cold")
def test_a_version_two_log_row_carries_its_context_on_both_executors():
    """Fails if the executors build different contexts or a delay does not move it."""
    truths, roots = worlds(1)
    truth, root = one_world()
    runs = {}
    for name, executor in (("host", HostExecutor()), ("compiled", CompiledExecutor())):
        for wait in (0, 50):
            study = _delayed(wait)
            if name == "host":
                runs[name, wait] = executor.run(study, truth, root, N_STEPS)
            else:
                runs[name, wait] = executor.run(study, truths, roots, N_STEPS)
    for wait in (0, 50):
        host, compiled = runs["host", wait], runs["compiled", wait]
        assert histories_equal(host.history(), compiled.history(0)).ok
    start0, pay0 = _completed(runs["host", 0])
    start1, pay1 = _completed(runs["host", 50])
    assert len(start0) == len(start1) == 5
    for start, pay in ((start0, pay0), (start1, pay1)):
        np.testing.assert_array_equal(pay[:, 0], start)
        np.testing.assert_array_equal(pay[:, 3], 1.0)
    shift = start1 - start0
    assert np.all(shift[:1] == 0) and np.all(shift[1:] == 50)
    np.testing.assert_array_equal(pay1[:, 0] - pay0[:, 0], shift)


@pytest.mark.medium(reason="one host run with a cancel, cold")
def test_a_cancelled_version_two_row_carries_its_logged_context():
    """Fails if the cancel path builds a context that differs from the logged row."""
    truth, root = one_world()
    run = HostExecutor().run(_v2(policies=(_CancelOnWake(),)), truth, root, N_STEPS)
    log = run.carry.log
    keep = np.asarray(log.valid) & (np.asarray(log.status) != RecordStatus.COMPLETED)
    assert keep.any()
    pay = np.asarray(log.payload)[keep]
    np.testing.assert_array_equal(pay[:, 0], np.asarray(log.start_tick)[keep])
    np.testing.assert_array_equal(pay[:, 1], np.asarray(log.end_tick)[keep])
    np.testing.assert_array_equal(pay[:, 2], np.asarray(log.acquisition_index)[keep])
    assert np.all(pay[:, 3] < 1.0)


@pytest.mark.medium(reason="two compiled runs of the toy survey, 5.3 s cold")
def test_the_fork_battery_builds_the_same_context_from_the_log():
    """Fails if the fork battery's recomputed row differs from the handler's."""
    truths, roots = worlds(1)
    assert battery.fork_check(_delayed(50), truths, roots, N_STEPS)


def test_a_version_two_study_round_trips_and_a_version_one_reference_migrates():
    """Fails if the version-2 model does not survive its document."""
    study = _v2()
    doc = dump_study(study, KINDS)
    text = doc.to_json()
    again = load_study(text, KINDS)
    assert dump_study(again, KINDS).content_hash == doc.content_hash
    data = json.loads(text)
    del data["content_hash"]
    data["study"]["channel_model"]["$kind"][2] = 1
    old = json.dumps(data)
    assert dump_study(load_study(old, KINDS), KINDS).content_hash == doc.content_hash
