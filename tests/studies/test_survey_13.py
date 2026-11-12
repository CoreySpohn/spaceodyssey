"""Survey 13, configuration switch: records carry their admission configuration.

One module-scoped host run of the toy study whose asset 0 switches configuration at
tick 100: an observation admitted at 0, a commitment admitted at 0 that starts at 200,
and an observation admitted at 130.
"""

import types

import numpy as np
import pytest

from spaceodyssey.core import HostExecutor
from spaceodyssey.core.preflight import preflight
from tests.core.helpers import one_world
from tests.studies.scenarios.survey_13 import (
    COMMIT_CHANNEL,
    COMMIT_START,
    GAINS,
    OBSERVE_CHANNEL,
    SWITCH_TICK,
    survey_13,
)

N_STEPS = 100


@pytest.fixture(scope="module")
def run():
    """The host run, its truth, and the records as `(channel, start, config, row)`."""
    study = survey_13()
    assert preflight(study) == []
    truth, root = one_world(caps=study.caps)
    result = HostExecutor().run(study, truth, root, N_STEPS)
    log = result.carry.log
    rows = [
        types.SimpleNamespace(
            channel=int(log.channel[i]),
            start=int(log.start_tick[i]),
            config=int(log.config_id[i]),
            payload=np.asarray(log.payload[i]),
        )
        for i in np.flatnonzero(np.asarray(log.valid))
    ]
    return types.SimpleNamespace(study=study, truth=truth, records=rows)


@pytest.mark.slow(reason="one host run of the switch study, 4 s cold")
def test_records_before_and_after_the_switch_carry_different_configurations(run):
    """Fails if admission stamps a constant or the tick of the Done: the observation
    admitted at 0 carries configuration 0 and the one admitted at 130 carries 1
    (both counted), and the realized row uses the stamp, so its value is the gain of
    the stamped configuration times the signal plus the noise draw, and not that of
    the other configuration.
    """
    live = [r for r in run.records if r.channel == OBSERVE_CHANNEL]
    assert [r.config for r in live] == [0, 1]
    assert live[0].start < SWITCH_TICK < live[1].start
    signal = float(run.truth.signal[0])
    ids = np.asarray(run.study.channels.id).tolist()
    noises = np.asarray(run.study.channel_model.noise)
    for r in run.records:
        _, variate, fraction, stamp = r.payload
        assert stamp == r.config
        draw = noises[ids.index(r.channel)] / np.sqrt(max(fraction, 1e-3)) * variate

        def value(config, draw=draw):
            return GAINS[config] * signal + draw

        assert r.payload[0] == pytest.approx(value(r.config), abs=1e-12)
        assert abs(r.payload[0] - value(1 - r.config)) > 0.1


@pytest.mark.slow(reason="shares the host run of the switch study")
def test_a_commitment_admitted_before_the_switch_keeps_its_configuration(run):
    """Fails if a commitment takes the configuration of its start tick: it is admitted
    at 0, before the switch, and starts at 200, after it, so it must carry 0 while an
    observation admitted after the switch carries 1. Counted: one commitment record.
    """
    (commit,) = [r for r in run.records if r.channel == COMMIT_CHANNEL]
    assert commit.start >= COMMIT_START > SWITCH_TICK
    assert commit.config == 0
    assert 1 in [r.config for r in run.records if r.channel == OBSERVE_CHANNEL]
