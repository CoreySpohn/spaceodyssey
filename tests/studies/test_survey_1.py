"""A hand-counted fixed search and complete executor-history parity."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from spaceodyssey.core import Capacities, CompiledExecutor, HostExecutor
from spaceodyssey.core.executors.base import histories_equal, index_world
from spaceodyssey.studies.toy import ToyPhotometry, ToyTruth, survey_1


class ToyKnownSignal(ToyPhotometry):
    """Return the known signal; retain the declared uncertainty for scoring."""

    def realize(self, truth, subject, channel_index, fraction, key, design, channels):
        return design.gain * truth.signal[subject], jnp.asarray(0.0, jnp.float64)


def test_survey_1_hand_count_and_executor_parity():
    caps = Capacities(n_assets=1, n_lanes=1, n_channels=1, n_subjects=2, n_an=2)
    study = survey_1(caps=caps, channel_model=ToyKnownSignal(noise=jnp.array([0.5])))
    truth = ToyTruth(signal=jnp.array([1.0, 1.0]))
    root = jr.PRNGKey(0)

    # The plan is (0, 1, 0, 1, 0), all on channel 10: counts (3, 2).
    # Each acquisition costs 1 from balances (10, 100, 100), so all five fit.
    # Transition 10 + occupancy 100 + latency 20 = 130 ticks per release;
    # the fifth release is at 650, before the 1000-tick horizon.
    # Gain 1 and known signals (1, 1) give means (1, 1). With uncertainty
    # 0.5, SNRs are sqrt(3)/0.5 = 2*sqrt(3) > 3 and sqrt(2)/0.5 < 3.
    # Thus exactly one subject is credited, without consulting a run.
    expected_credit = 1
    # Acquisition k starts after the 10-tick transition at 130 k + 10, ends 100
    # ticks later, and is available, released and accepted 20 ticks after that.
    starts = [130 * k + 10 for k in range(5)]
    ends = [s + 100 for s in starts]
    releases = [e + 20 for e in ends]
    plan = np.asarray(study.agenda.policies[0].plan)
    # the plan array is padded with -1 rows; only its real rows are the plan
    np.testing.assert_array_equal(
        plan[plan[:, 0] >= 0], [(0, 10), (1, 10), (0, 10), (1, 10), (0, 10)]
    )
    np.testing.assert_array_equal(study.channels.id, [10])
    np.testing.assert_array_equal(study.channels.occupancy_ticks, [100])
    np.testing.assert_array_equal(study.channels.latency_ticks, [20])
    np.testing.assert_array_equal(study.assets.transition_ticks, [10])
    np.testing.assert_array_equal(study.channels.cost, [[1.0, 1.0, 1.0]])
    assert float(study.objective.threshold) == 3.0

    host = HostExecutor().run(study, truth, root, 40)
    compiled = CompiledExecutor().run_worlds(
        study, jax.tree_util.tree_map(lambda a: a[None], truth), root[None], 40
    )
    assert bool(host.complete) and bool(compiled.complete[0])
    for carry in (host.carry, index_world(compiled.carry, 0)):
        assert int(study.objective.evaluate(carry, study)) == expected_credit
        np.testing.assert_array_equal(carry.belief.n, [3, 2])
        np.testing.assert_allclose(carry.belief.mean, [1.0, 1.0], atol=1e-12, rtol=0)
        valid = np.asarray(carry.log.valid)
        np.testing.assert_array_equal(
            np.asarray(carry.log.subject)[valid], [0, 1, 0, 1, 0]
        )
        np.testing.assert_array_equal(np.asarray(carry.log.start_tick)[valid], starts)
        np.testing.assert_array_equal(np.asarray(carry.log.end_tick)[valid], ends)
        for field in ("available_tick", "released_tick", "accepted_tick"):
            np.testing.assert_array_equal(
                np.asarray(getattr(carry.log, field))[valid], releases
            )
    assert host.history().n_records == 5
    assert histories_equal(host.history(), compiled.history(0)).go
