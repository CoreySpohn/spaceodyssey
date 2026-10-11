"""Tests for the event-driven mission loop on deterministic doubles."""

import pytest
from planit_py.vocabulary import Action, FixedDuration, Mode, StopOnCertificate

from spaceodyssey.belief import MissionBelief
from spaceodyssey.build import build
from spaceodyssey.loop import (
    DETECTION_CERTIFICATE,
    MissionState,
    observe,
    run_mission,
    step,
)
from spaceodyssey.rng import Purpose, stream
from tests.doubles import (
    PLANET_TARGETS,
    TARGET_IDS,
    build_test_registry,
    make_config,
    make_ledger,
    make_spec,
)


def make_mission(spec=None, config=None):
    """Compile the toy mission."""
    return build(spec or make_spec(), config or make_config(), build_test_registry())


def draw_universe(mission):
    """Draw the toy universe from the mission's scene source."""
    return mission.truth.scene.draw(stream(0, 0, 0, Purpose.UNIVERSE))


def make_action(target_id: int, integration=None) -> Action:
    """An action for direct observe() tests."""
    return Action(
        target_id=target_id,
        mode=Mode.IMAGING,
        bands=("500nm",),
        integration=integration or FixedDuration(duration_d=1.0),
        start_time_d=0.0,
    )


class TestObserve:
    """The single bridge across the truth/belief firewall."""

    def test_returns_dataset_and_realized_duration(self):
        """observe() samples the dataset and reports the duration."""
        mission = make_mission()
        universe = draw_universe(mission)
        obs = observe(mission.truth, universe, make_action(1), stream(0, 1, 0, 1))
        assert float(obs.dataset.snr) == pytest.approx(9.0)
        assert obs.duration_d == 1.0

    def test_adaptive_integration_not_implemented_yet(self):
        """Adaptive stopping raises a clear NotImplementedError."""
        mission = make_mission()
        universe = draw_universe(mission)
        adaptive = StopOnCertificate(threshold=7.0, max_duration_d=5.0)
        with pytest.raises(NotImplementedError, match="FixedDuration"):
            observe(
                mission.truth,
                universe,
                make_action(1, adaptive),
                stream(0, 1, 0, 1),
            )


class TestStep:
    """One propose-observe-update-certify cycle."""

    def test_advances_clock_and_logs(self):
        """A step charges realized time plus overhead and logs the record."""
        mission = make_mission()
        universe = draw_universe(mission)
        state = MissionState(time_d=0.0, ledger=make_ledger(), belief=MissionBelief())
        state = step(mission, universe, state, root=0)
        assert state.time_d == pytest.approx(1.1)
        assert state.epoch_index == 1
        assert len(state.log) == 1
        assert state.ledger.spent["time_d"] == pytest.approx(1.1)

    def test_zero_time_action_rejected(self):
        """A configuration that charges no time is a loud loop error."""
        config = make_config(overhead_d=0.0, policy_duration_d=0.0)
        mission = make_mission(config=config)
        universe = draw_universe(mission)
        state = MissionState(time_d=0.0, ledger=make_ledger(), belief=MissionBelief())
        with pytest.raises(RuntimeError, match="charged no time"):
            step(mission, universe, state, root=0)


class TestRunMission:
    """Termination and certified outcomes."""

    def test_certifies_exactly_the_planet_targets(self):
        """Round-robin looks certify the planet-bearing targets only."""
        mission = make_mission(spec=make_spec(duration_d=6.0))
        state = run_mission(mission, draw_universe(mission), seed=0)
        assert state.belief.certified(DETECTION_CERTIFICATE) == PLANET_TARGETS
        assert state.time_d >= 6.0
        assert state.epoch_index == 6

    def test_budget_exhaustion_terminates(self):
        """The ledger stops the mission before the duration would."""
        mission = make_mission(spec=make_spec(duration_d=100.0, time_budget_d=3.0))
        state = run_mission(mission, draw_universe(mission), seed=0)
        assert state.ledger.exhausted
        assert state.epoch_index == 3

    def test_every_target_visited(self):
        """The log covers all targets in round-robin order."""
        mission = make_mission(spec=make_spec(duration_d=6.0))
        state = run_mission(mission, draw_universe(mission), seed=0)
        visited = [obs.action.target_id for obs in state.log]
        assert visited[:3] == list(TARGET_IDS)
