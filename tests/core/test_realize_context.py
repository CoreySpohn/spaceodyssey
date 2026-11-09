"""A version-2 channel model receives a context and returns the whole payload row.

A version-1 model keeps its call and its compiled program; the history of every
version-1 study is pinned by the golden file.
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from spaceodyssey.codec import dump_study, load_study
from spaceodyssey.core import battery, handlers
from spaceodyssey.core.enums import F64, I64, Purpose, RecordStatus
from spaceodyssey.core.executors.base import histories_equal
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.core.keys import asset_coordinate, draw_key
from spaceodyssey.core.kinds import RealizeContext, realize_row
from spaceodyssey.core.records import record_none
from spaceodyssey.studies import toy
from tests.codec.test_document import TOY_KINDS
from tests.core.helpers import one_world, worlds
from tests.doubles import ToyEpochPhotometry

KINDS = {**TOY_KINDS, ("channel_model", "test.epoch_photometry", 2): ToyEpochPhotometry}
PLAN = ((0, 10), (1, 10), (0, 10), (1, 10), (0, 10))
N_STEPS = 40


def _v2(**kw):
    """The five-acquisition toy study with the version-2 model, optionally delayed."""
    study = toy.survey_1(**kw)
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


def _legacy_realize_rec(c, study, truth, root, base, fraction):
    """`_realize_rec` as it was before contexts: the version-1 call, written out."""
    coords = dataclasses.replace(base, status=jnp.asarray(RecordStatus.COMPLETED, I64))
    key = draw_key(
        root,
        coords,
        Purpose.MEASUREMENT,
        asset=asset_coordinate(study, base.channel_index),
        branch=study.fork.branch_of(base.acquisition_index),
        coupling=study.coupling,
    )
    meas, z = study.channel_model.realize(
        truth,
        base.subject,
        base.channel_index,
        fraction,
        key,
        study.design,
        study.channels,
    )
    payload = jnp.zeros(study.caps.payload_width, F64).at[0].set(meas).at[1].set(z)
    return dataclasses.replace(base, payload=payload.at[2].set(fraction))


def test_version_one_realization_lowers_to_the_pre_context_program():
    """Fails if a version-1 model's realization changes its lowered program."""
    study = toy.survey_1()
    truth, root = one_world()
    args = (truth, root, _base(study), jnp.asarray(0.5, F64))

    def text(fn):
        return str(jax.make_jaxpr(lambda *a: fn(None, study, *a))(*args))

    assert text(handlers._realize_rec) == text(_legacy_realize_rec)


@pytest.mark.medium(reason="one cold host run of the toy survey, about 4.5 s")
def test_a_version_one_history_is_unchanged_beside_a_version_two_study():
    """Fails if the version-1 payload columns move when the dispatch is added."""
    study = toy.survey_1()
    truth, root = one_world()
    run = HostExecutor().run(study, truth, root, N_STEPS)
    _, pay = _completed(run)
    assert len(pay) == 5
    assert np.all(pay[:, 2] == 1.0)
    assert np.all(np.isfinite(pay[:, :2]))


def test_a_version_two_row_is_the_context_function_of_the_record():
    """Fails if the context start tick is not the record's, or the row is not padded."""
    study = _v2()
    truth, _ = one_world()
    fraction = jnp.asarray(0.5, F64)
    row = realize_row(study, truth, _base(study), fraction, jr.PRNGKey(3))
    assert row.shape == (study.caps.payload_width,)
    assert (float(row[0]), float(row[1])) == (7.0, 22.0)
    assert np.all(np.asarray(row[3:]) == 0.0)
    ctx = RealizeContext.from_record(_base(study), fraction)
    assert (int(ctx.start_tick), int(ctx.end_tick)) == (7, 107)


def test_a_version_two_row_wider_than_the_payload_is_refused():
    """Fails if a too-wide row is truncated silently."""

    class Wide(ToyEpochPhotometry):
        def realize(self, *args, context, **kw):
            return jnp.zeros(99)

    study = dataclasses.replace(toy.survey_1(), channel_model=Wide())
    truth, root = one_world()
    with pytest.raises(ValueError, match="1-D row"):
        realize_row(study, truth, _base(study), jnp.asarray(1.0, F64), root)


@pytest.mark.medium(reason="one host and one compiled run of the toy survey, cold")
def test_a_version_two_log_row_is_the_function_of_its_start_tick_on_both_executors():
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
        np.testing.assert_array_equal(pay[:, 1], 3 * start + 1)
    shift = start1 - start0
    assert np.all(shift[:1] == 0) and np.all(shift[1:] == 50)
    np.testing.assert_array_equal(pay1[:, 0] - pay0[:, 0], shift)


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
