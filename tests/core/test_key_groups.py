"""Optional counters address completed records independently of admissions."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.codec.checkpoint import DirectoryCheckpoint
from spaceodyssey.core.battery.fork import _log_record
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.carry import empty_carry
from spaceodyssey.core.checkpoint import named_leaves, rebuild
from spaceodyssey.core.enums import I64, EventKind, RecordStatus
from spaceodyssey.core.events import empty_event_row, enqueue
from spaceodyssey.core.executors.base import histories_equal, index_world
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.core.executors.replay import _row
from spaceodyssey.core.handlers import h_cancel, h_done
from spaceodyssey.core.kinds import RealizeContext
from spaceodyssey.core.records import record_none
from spaceodyssey.studies import builder, toy
from tests.codec.test_document import TOY_KINDS
from tests.core.helpers import one_world, worlds

done = eqx.filter_jit(h_done)
cancel = eqx.filter_jit(h_cancel)


def _study(K, groups, **settings):
    caps = Capacities(n_key_groups=K)
    study = builder.survey_1(caps, **settings)
    membership = jnp.zeros((caps.n_channels, K), bool)
    for channel, indices in groups.items():
        membership = membership.at[channel, jnp.asarray(indices, I64)].set(True)
    return dataclasses.replace(
        study, channels=dataclasses.replace(study.channels, key_groups=membership)
    )


def _event(carry, study, channel, index, subject=0):
    channels = study.channels
    lane = channels.lane[channel]
    return dataclasses.replace(
        empty_event_row(study.caps),
        kind=jnp.asarray(EventKind.DONE, I64),
        channel_index=jnp.asarray(channel, I64),
        channel=channels.id[channel],
        subject=jnp.asarray(subject, I64),
        lane=lane,
        tick=carry.tick + channels.occupancy_ticks[channel],
        acquisition_index=jnp.asarray(index, I64),
        acquisition_id=jnp.asarray(index + 1, I64),
        gen=carry.ledger.lane_gen[lane],
        valid=jnp.asarray(True),
    )


def _run(study, plan):
    truth, root = one_world()
    carry = empty_carry(study.caps)
    for index, step in enumerate(plan):
        is_cancel = isinstance(step, str)
        channel = int(step.split()[1]) if is_cancel else step
        event = _event(carry, study, channel, index)
        if is_cancel:
            carry = enqueue(
                carry,
                True,
                caps=study.caps,
                kind=event.kind,
                tick=event.tick,
                stage=event.stage,
                lane=event.lane,
                channel_index=event.channel_index,
                channel=event.channel,
                subject=event.subject,
                gen=event.gen,
                acquisition_index=event.acquisition_index,
                acquisition_id=event.acquisition_id,
            )
            carry, _, _ = cancel(
                carry,
                dataclasses.replace(event, kind=jnp.asarray(EventKind.CANCEL, I64)),
                study,
                truth,
                root,
            )
            # The old completion must remain stale after cancellation.
            carry, _, _ = done(carry, event, study, truth, root)
        else:
            carry, _, _ = done(carry, event, study, truth, root)
            # Replaying the same completion must not consume another ordinal.
            carry, _, _ = done(carry, event, study, truth, root)
    return carry


def run_toy_with_key_groups(K, groups, plan):
    """The log after completing or cancelling each acquisition in `plan`."""
    return _run(_study(K, groups), plan).log


def completed_ordinals(log):
    """Group ordinals of completed rows, in write order."""
    completed = np.asarray(log.valid) & (
        np.asarray(log.status) == RecordStatus.COMPLETED
    )
    return np.asarray(log.group_ordinal)[completed].tolist()


@pytest.mark.medium(reason="compiles completion and cancellation handlers")
def test_a_record_carries_its_group_ordinals_before_the_increment():
    """Fails if an ordinal counts the record itself, or a cancelled occupancy."""
    log = run_toy_with_key_groups(
        K=2, groups={0: (0,), 1: (0, 1)}, plan=(0, 1, 1, "cancel 1")
    )
    assert completed_ordinals(log) == [[0, 0], [1, 0], [2, 1]]
    study = _study(2, {0: (0,), 1: (0, 1)})
    carry = _run(study, (0, 1, 1, "cancel 1", 1))
    assert completed_ordinals(carry.log)[-1] == [3, 2]
    np.testing.assert_array_equal(carry.group_counts[0], [4, 3])
    np.testing.assert_array_equal(carry.group_counts[1:], 0)


def test_without_key_groups_the_carry_and_records_are_unchanged():
    """Fails if the default layout gains a leaf (it would move every golden)."""
    caps = Capacities()
    carry, rec = empty_carry(caps), record_none(caps)
    assert carry.group_counts is None and rec.group_ordinal is None
    assert carry.log.group_ordinal is None
    assert len(jax.tree.leaves(carry)) == 133
    assert RealizeContext.from_record(rec, 1.0).group_ordinal.shape == (0,)


def test_only_a_written_completion_advances_its_subject_and_restores():
    """Fails if full logs consume ordinals, subjects mix, or restore loses counters."""
    study = _study(2, {0: (0,), 1: (0, 1)})
    truth, root = one_world()
    carry = empty_carry(study.caps)
    event = _event(carry, study, 1, 0, subject=1)
    carry, _, rec = done(carry, event, study, truth, root)
    np.testing.assert_array_equal(rec.group_ordinal, [0, 0])
    np.testing.assert_array_equal(carry.group_counts[1], [1, 1])
    np.testing.assert_array_equal(carry.group_counts[0], 0)
    restored = rebuild(empty_carry(study.caps), dict(named_leaves(carry)))
    np.testing.assert_array_equal(restored.group_counts, carry.group_counts)
    np.testing.assert_array_equal(restored.log.group_ordinal, carry.log.group_ordinal)
    full = dataclasses.replace(carry, n_log=jnp.asarray(study.caps.n_log, I64))
    full, emit, _ = done(full, _event(full, study, 1, 1), study, truth, root)
    assert not bool(emit.n_written)
    np.testing.assert_array_equal(full.group_counts, carry.group_counts)


@pytest.mark.medium(reason="runs both executors and resumes one checkpointed toy run")
def test_group_ordinals_survive_executors_checkpoints_and_row_readers(tmp_path):
    """Fails if executors, checkpoints, replay, or fork readers lose ordinals."""
    study = _study(
        2,
        {0: (0,), 1: (0, 1)},
        policies=(toy.FixedPlanPolicy(plan=((0, 10), (0, 11), (0, 11))),),
    )
    truth, root = one_world()
    truths, roots = worlds(1)
    host = HostExecutor().run(study, truth, root, 40)
    batch = CompiledExecutor().run(study, truths, roots, 40)
    assert histories_equal(host.history(), batch.history(0)).ok
    assert completed_ordinals(host.carry.log) == [[0, 0], [1, 0], [2, 1]]
    np.testing.assert_array_equal(index_world(batch.carry, 0).group_counts[0], [3, 2])
    checkpoint = DirectoryCheckpoint(tmp_path / "run", kinds=TOY_KINDS)
    prefix = HostExecutor().run(study, truth, root, 12, checkpoint=checkpoint)
    assert bool(jnp.any(prefix.carry.group_counts))
    resumed = HostExecutor().run(
        study, truth, root, 28, checkpoint=checkpoint, resume=True
    )
    assert histories_equal(host.history(), resumed.history()).ok
    for reader in (_log_record, _row):
        rec = reader(resumed.carry.log, 2)
        np.testing.assert_array_equal(rec.group_ordinal, [2, 1])
        np.testing.assert_array_equal(
            RealizeContext.from_record(rec, 1.0).group_ordinal, [2, 1]
        )
