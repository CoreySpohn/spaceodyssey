"""A world-drawn trajectory table is sampled by a version-2 channel at the start tick.

Studies that do not use a table are pinned by the golden file; here a toy world source
returns a table inside its truth and a version-2 channel samples it.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from spaceodyssey.codec import dump_study, load_study
from spaceodyssey.core import battery, preflight
from spaceodyssey.core.battery.fork import RECORD_PURPOSES
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import F64, PUBLIC, Purpose, RecordStatus
from spaceodyssey.core.executors.base import histories_equal
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.core.kinds import AbstractWorldSource
from spaceodyssey.core.records import KindRegistry, PayloadColumn, RecordKind
from spaceodyssey.core.trajectory import TrajectoryTable, draw_walk, trajectory_key
from spaceodyssey.studies import builder, toy
from tests.codec.test_document import TOY_KINDS
from tests.doubles import ToyEpochPhotometry

TICKS = jnp.arange(0, 1001, 50)
PLAN = ((0, 10),) * 5
N_STEPS = 40


class TableTruth(eqx.Module):
    """Study-defined truth: the toy signal plus one trajectory table."""

    signal: jax.Array
    table: TrajectoryTable


class TableWorldSource(AbstractWorldSource):
    """Draws the signal and the table; the salts move one draw each."""

    other_salt: int = 0
    own_salt: int = 0

    def draw(self, key, capacities):
        """Signal from a fold of the world key, table from the trajectory purpose."""
        other = jr.fold_in(jr.fold_in(key, Purpose.PROCESS), self.other_salt)
        own = jr.fold_in(trajectory_key(key), self.own_salt)
        return TableTruth(
            signal=jr.uniform(other, (capacities.n_subjects,)),
            table=draw_walk(own, TICKS, 1.0),
        )


class TableChannel(ToyEpochPhotometry):
    """Row `[start_tick, table at start, acquisition_index, fraction, z]`."""

    def realize(
        self, truth, subject, channel_index, fraction, key, design, channels, *, context
    ):
        """Sample the world table at the record's start tick."""
        z = jr.normal(key, dtype=F64)
        row = (
            context.start_tick,
            truth.table.at(context.start_tick),
            context.acquisition_index,
            context.fraction,
            z,
        )
        return jnp.stack([jnp.asarray(x, F64) for x in row])


KINDS = {
    **TOY_KINDS,
    ("channel_model", "test.table_channel", 2): TableChannel,
    ("world_source", "test.table_world", 1): TableWorldSource,
}


def _study(wait):
    cols = tuple(
        PayloadColumn(n, PUBLIC)
        for n in ("start", "sample", "acquisition_index", "fraction", "variate")
    )
    caps = KindRegistry(
        (RecordKind("none", ()), RecordKind("measurement", cols), toy.AUXILIARY_KIND)
    ).lower(Capacities())
    policy = toy.FixedPlanPolicy(plan=PLAN, wait_at=1, wait_ticks=wait)
    study = builder.survey_1(caps, policies=(policy,))
    return dataclasses.replace(
        study, channel_model=TableChannel(), world_source=TableWorldSource()
    )


def _worlds(study, n=2):
    roots = jax.vmap(lambda i: jr.fold_in(jr.PRNGKey(0), i))(jnp.arange(n))
    truths = jax.vmap(
        lambda r: study.world_source.draw(jr.fold_in(r, Purpose.WORLD), study.caps)
    )(roots)
    return truths, roots


@pytest.fixture(scope="module")
def runs():
    """Host and compiled runs of the undelayed and the delayed study, one world each."""
    out = {}
    for wait in (0, 50):
        study = _study(wait)
        truths, roots = _worlds(study)
        truth = jax.tree.map(lambda x: x[0], truths)
        out["host", wait] = HostExecutor().run(study, truth, roots[0], N_STEPS)
        out["compiled", wait] = CompiledExecutor().run(study, truths, roots, N_STEPS)
        out["truth", wait] = truth
    return out


def _completed(run):
    log = run.carry.log
    keep = np.asarray(log.valid) & (np.asarray(log.status) == RecordStatus.COMPLETED)
    return np.asarray(log.start_tick)[keep], np.asarray(log.payload)[keep]


def test_the_table_interpolates_between_ticks_and_holds_its_ends():
    """Fails if sampling is off by a row, extrapolates, or reads the wrong axis."""
    table = TrajectoryTable(ticks=jnp.asarray([10, 20]), values=jnp.asarray([1.0, 3.0]))
    got = [float(table.at(t)) for t in (0, 10, 15, 20, 99)]
    assert got == [1.0, 1.0, 2.0, 3.0, 3.0]


def test_the_trajectory_draw_moves_only_with_its_own_purpose_key():
    """Fails if another draw of the world source moves the table, or the table
    shares a stream with the world key or another purpose."""
    study = _study(0)
    key = jr.fold_in(jr.PRNGKey(0), Purpose.WORLD)
    base = study.world_source.draw(key, study.caps)
    other = TableWorldSource(other_salt=1).draw(key, study.caps)
    own = TableWorldSource(own_salt=1).draw(key, study.caps)
    np.testing.assert_array_equal(base.table.values, other.table.values)
    assert not np.array_equal(base.signal, other.signal)
    np.testing.assert_array_equal(base.signal, own.signal)
    assert not np.array_equal(base.table.values, own.table.values)
    for purpose in (Purpose.WORLD, Purpose.MEASUREMENT, Purpose.PROCESS):
        stream = draw_walk(jr.fold_in(key, purpose), TICKS, 1.0).values
        assert not np.array_equal(stream, base.table.values)
    assert Purpose.TRAJECTORY in RECORD_PURPOSES


def test_existing_purposes_keep_their_codes():
    """Fails if adding the purpose renumbered one, moving every pinned key."""
    assert [int(p) for p in (0, 1, 2, 3, 4)] == [
        Purpose.WORLD,
        Purpose.MEASUREMENT,
        Purpose.POLICY,
        Purpose.INFERENCE,
        Purpose.PROCESS,
    ]
    assert int(Purpose.TRAJECTORY) == 5


@pytest.mark.medium(reason="host and compiled runs of the table study, cold")
def test_a_record_samples_the_table_at_its_start_tick_on_both_executors(runs):
    """Fails if the sample is read at another tick (end, admission), a delay does not
    move it, or the executors disagree."""
    for wait in (0, 50):
        host, compiled = runs["host", wait], runs["compiled", wait]
        assert histories_equal(host.history(), compiled.history(0)).ok
        starts, pay = _completed(host)
        assert len(starts) == 5
        truth = runs["truth", wait]
        expect = np.interp(starts, np.asarray(TICKS), np.asarray(truth.table.values))
        np.testing.assert_allclose(pay[:, 1], expect, rtol=0, atol=1e-12)
    s0, p0 = _completed(runs["host", 0])
    s1, p1 = _completed(runs["host", 50])
    assert np.all(s1[1:] > s0[1:]) and s1[0] == s0[0]
    assert not np.allclose(p1[1:, 1], p0[1:, 1])


@pytest.mark.medium(reason="two compiled runs of the delayed table study")
def test_the_fork_battery_builds_the_same_sample():
    """Fails if the fork battery's recomputed row reads another table."""
    study = _study(50)
    truths, roots = _worlds(study, 1)
    assert battery.fork_check(study, truths, roots, N_STEPS)


def test_a_table_study_passes_preflight_and_round_trips():
    """Fails if the table world source or its channel does not survive the document."""
    study = _study(0)
    assert not preflight(study)
    doc = dump_study(study, KINDS)
    again = load_study(doc.to_json(), KINDS)
    assert dump_study(again, KINDS).content_hash == doc.content_hash
    assert again.world_source == study.world_source
