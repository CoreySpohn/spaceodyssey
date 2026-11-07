"""Plan-as-data loading and replay on both transaction executors."""

import dataclasses
import json
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from planit_py import PlanTable

from spaceodyssey import core
from spaceodyssey.adapters.planit import PlanTablePolicy, plan_table_to_ticks
from spaceodyssey.studies import toy
from tests.core.helpers import worlds

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/planit/greedy_voi.json").read_text()
)


class View(NamedTuple):
    tick: jax.Array | int


def table(subjects=(0,), days=(0.0,), channel_sets=None, **kwargs):
    return PlanTable.from_lists(
        subjects,
        days,
        [(10,)] * len(subjects) if channel_sets is None else channel_sets,
        channel_ids=kwargs.pop("channel_ids", (10, 11)),
        **kwargs,
    )


@pytest.fixture(scope="module")
def study():
    caps = core.Capacities(
        n_subjects=3,
        n_an=3,
        n_assets=1,
        n_lanes=2,
        n_channels=2,
        n_events=8,
        n_log=4,
        n_effects=1,
        max_sid=64,
    )
    base = toy.survey_1(
        caps=caps,
        policies=(toy.NullPolicy(),),
        lane_asset=jnp.asarray([0, 0], jnp.int64),
        transition_ticks=jnp.asarray([0, 0], jnp.int64),
        horizon_tick=24_000_000_000,
        phase_until=(24_000_000_000,),
        process=toy.HeartbeatProcess(
            period_ticks=7_776_000_000, n_fires=3, audience=core.PUBLIC
        ),
    )
    return eqx.tree_at(
        lambda s: (s.clock.tick_length_s, s.channels.occupancy_ticks),
        base,
        (jnp.asarray(0.001), jnp.full(2, 86_400, jnp.int64)),
    )


@pytest.fixture(scope="module")
def replays(study):
    inputs = FIXTURE["inputs"]
    names = inputs["targets"]
    truths, roots = worlds(1, study.caps)
    results = []
    traces = {"compiled": [], "host": []}

    def compiled_wake(carry, event):
        traces["compiled"].append(1)
        return core.wake_by_audience(carry, event)

    def host_wake(carry, event):
        traces["host"].append(1)
        return core.wake_by_audience(carry, event)

    for sequence in (("A", "A", "B", "C"), ("A", "C", "A", "C")):
        plan = table([names.index(n) for n in sequence], inputs["epochs_d"])
        policy = PlanTablePolicy.from_plan_table(plan, study)
        st = eqx.tree_at(lambda s: s.agenda.policies, study, (policy,))
        compiled = core.CompiledExecutor().run(
            st, truths, roots, 20, wake_fn=compiled_wake
        )
        host = core.HostExecutor().run(
            st, core.index_world(truths, 0), roots[0], 20, wake_fn=host_wake
        )
        results.append((st, compiled, host))
        if len(results) == 1:
            first_traces = {name: len(rows) for name, rows in traces.items()}
            assert first_traces == {"compiled": 1, "host": 1}
        else:
            assert {name: len(rows) for name, rows in traces.items()} == first_traces
    return results


def test_b7_sequences_utilities_and_admitted_starts(replays):
    inputs = FIXTURE["inputs"]
    for (study, compiled, host), sequence, utility, policies in zip(
        replays,
        (("A", "A", "B", "C"), ("A", "C", "A", "C")),
        (1.3100000000000003, 2.21),
        (
            ("greedy", "rollout_2"),
            ("rollout_3", "full_horizon", "optimal", "exact_package_selection"),
        ),
        strict=True,
    ):
        counts = list(inputs["initial_state"])
        earned = 0.0
        for slot, name in enumerate(sequence):
            target = inputs["targets"].index(name)
            if inputs["visibility"][target][slot]:
                earned += (
                    inputs["detection_rewards"][target]
                    if counts[target] == 0
                    else inputs["characterization_reward"]
                    if counts[target] == 1
                    else 0
                )
                counts[target] = min(counts[target] + 1, 2)
        assert earned == pytest.approx(utility, abs=1e-12, rel=0)
        for name in policies:
            assert FIXTURE["expected"][name]["sequence"] == list(sequence)
            assert FIXTURE["expected"][name]["utility"] == pytest.approx(
                utility, abs=1e-12, rel=0
            )
        want_subjects = [inputs["targets"].index(n) for n in sequence]
        # 90 days / (1 ms / tick) = 7,776,000,000 ticks.
        want_ticks = [0, 7_776_000_000, 15_552_000_000, 23_328_000_000]
        for carry in (core.index_world(compiled.carry, 0), host.carry):
            valid = np.asarray(carry.log.valid)
            np.testing.assert_array_equal(carry.log.subject[valid], want_subjects)
            np.testing.assert_array_equal(carry.log.start_tick[valid], want_ticks)
            assert int(carry.n_log) == 4
            assert not bool(carry.overflow | carry.violation | carry.stalled)
        assert float(study.clock.tick_length_s) == 0.001


def test_half_tick_at_day_270_is_refused(study):
    bad = table(days=(270 + 0.0005 / 86400,))
    with pytest.raises(ValueError, match=r"row 0.*day.*offset.*ticks"):
        PlanTablePolicy.from_plan_table(bad, study)
    # A relative 1e-9 check would also incorrectly accept this 7.8-tick error.
    with pytest.raises(ValueError, match=r"row 0.*offset.*ticks"):
        plan_table_to_ticks(table(days=(90 + 0.0078 / 86400,)), study.clock)


def test_absolute_tolerance_edge_rounds():
    # At this epoch 4 ULPs reach the absolute cap; 0.0005 ticks is representable.
    clock = core.Clock(0.001)
    day = (1_728_000_000_000 + 0.5e-3) * 0.001 / 86400
    ticks = plan_table_to_ticks(table(days=(day,)), clock)
    assert isinstance(ticks, np.ndarray) and ticks.dtype == np.int64
    np.testing.assert_array_equal(ticks, [1_728_000_000_000])
    with pytest.raises(ValueError, match="offset"):
        plan_table_to_ticks(
            table(days=((1_728_000_000_000 + 2e-3) * 0.001 / 86400,)), clock
        )
    # At small epochs the 1e-9 floor is binding, rather than the 1e-3 cap.
    np.testing.assert_array_equal(
        plan_table_to_ticks(table(days=(0.5e-9 / 86400,)), core.Clock(1)), [0]
    )
    with pytest.raises(ValueError, match="offset"):
        plan_table_to_ticks(table(days=(2e-9 / 86400,)), core.Clock(1))


def test_invalid_plan_rows_are_refused(study):
    cases = [
        (table(subjects=(3,)), "subject"),
        (eqx.tree_at(lambda t: t.target_ids, table(), jnp.array([-1])), "subject"),
        (
            eqx.tree_at(lambda t: t.start_times_d, table(), jnp.array([np.nan])),
            "non-finite",
        ),
        (
            eqx.tree_at(lambda t: t.start_times_d, table(), jnp.array([1e300])),
            "int64",
        ),
        (table((0, 1), (0.0, 90.0), group_ids=(7, 7)), "group"),
        (table(channel_ids=(99,), channel_sets=((99,),)), "channel"),
        (
            eqx.tree_at(lambda t: t.start_times_d, table(), jnp.array([-1.0])),
            "negative",
        ),
        (table(days=(300.0,)), "horizon"),
        (table(asset_ids=(1,)), "asset"),
        (table(transition_kinds=(1,)), "transition"),
    ]
    for plan, reason in cases:
        with pytest.raises(ValueError, match=reason):
            PlanTablePolicy.from_plan_table(plan, study)


def test_decision_cursor_masks_padding_and_channel_order(study):
    study = dataclasses.replace(
        study, caps=dataclasses.replace(study.caps, n_effects=2)
    )
    plan = table(
        (2,), (90.0,), ((11, 10),), channel_ids=(11, 10), group_ids=(7,), capacity=3
    )
    policy = PlanTablePolicy.from_plan_table(plan, study)
    assert policy.version == 1
    key = jax.random.PRNGKey(0)
    for decide in (eqx.filter_jit(policy.decide), policy.decide_py):
        state = jnp.zeros(study.caps.n_pstate)
        for tick, kinds, position in (
            (0, [core.EffectKind.NONE] * 2, 0),
            (7_776_000_000, [core.EffectKind.OBSERVE] * 2, 1),
            (7_776_000_001, [core.EffectKind.NONE] * 2, 1),
        ):
            effects, state, draw = decide(
                state, View(tick=jnp.asarray(tick)), study, key
            )
            np.testing.assert_array_equal(effects.kind, kinds)
            assert state[0] == position and draw == 0
            if position == 1 and tick == 7_776_000_000:
                np.testing.assert_array_equal(effects.subject, [2, 2])
                np.testing.assert_array_equal(effects.channel_index, [0, 1])
                np.testing.assert_array_equal(effects.group, [7, 7])
    empty = PlanTablePolicy.from_plan_table(table((), (), capacity=3), study)
    effects, state, _ = empty.decide_py(
        np.zeros(study.caps.n_pstate), View(tick=0), study, key
    )
    assert not np.any(effects.kind) and not np.any(state)


def test_host_and_compiled_agree_record_for_record(replays):
    for study, compiled, host in replays:
        comparison = core.histories_equal(compiled.history(0), host.history())
        assert comparison.ok, comparison
        for field in dataclasses.fields(host.carry.log):
            if getattr(host.carry.log, field.name) is None:
                continue
            np.testing.assert_array_equal(
                getattr(compiled.carry.log, field.name)[0],
                getattr(host.carry.log, field.name),
            )
        truths, roots = worlds(1, study.caps)
        python = core.HostExecutor().run(
            study, core.index_world(truths, 0), roots[0], 20, policy_form="python"
        )
        assert core.histories_equal(python.history(), host.history()).ok
