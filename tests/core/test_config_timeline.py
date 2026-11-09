"""Configuration timeline: each admission is stamped with the asset's configuration.

An asset declares sorted switch ticks; the configuration of an admission is the number
of them at or before the admission tick, and that number follows the occupancy into its
record (completed, cancelled or refused at its start), whatever the tick of the Done or
the Cancel. A commitment keeps the configuration it was admitted under. Every run below
shares one compiled study structure (commitments and a two-slot timeline on); scenarios
differ in script data and in the timeline leaf.
"""

import dataclasses
import json
from types import SimpleNamespace

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from spaceodyssey.codec import dump_study, load_study
from spaceodyssey.core import CompiledExecutor, HostExecutor, battery, histories_equal
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import I64, RecordStatus
from spaceodyssey.core.ledger import CONFIG_PAD
from spaceodyssey.core.preflight import preflight
from spaceodyssey.studies import toy
from tests.codec.test_document import TOY_KINDS
from tests.core.commitment_helpers import ScriptedCommitments, plan
from tests.core.helpers import one_world, worlds
from tests.core.program_helpers import cancel, end, observe, wait
from tests.core.test_realize_context import _epoch_registry
from tests.doubles import ToyEpochPhotometry

CAPS = _epoch_registry().lower(
    Capacities(n_effects=4, n_commitments=2, max_config_switches=2)
)
SWITCH = 100  # the tick asset 0 (lanes 0 and 1) switches; asset 1 (lane 2) never does
PAD = CONFIG_PAD
N_STEPS = 200
COMPLETED, PARTIAL = int(RecordStatus.COMPLETED), int(RecordStatus.PARTIAL)


class ToyConfigPhotometry(ToyEpochPhotometry):
    """A version-2 toy model whose row is `[config_id, start, end, fraction, z]`."""

    def realize(
        self, truth, subject, channel_index, fraction, key, design, channels, *, context
    ):
        """Return the row; its first column is the stamped configuration."""
        fields = (
            context.config_id,
            context.start_tick,
            context.end_tick,
            context.fraction,
        )
        z = jr.normal(key, dtype=jnp.float64)
        return jnp.stack([*(jnp.asarray(f, jnp.float64) for f in fields), z])


def timeline(*ticks):
    """Asset 0 switches at `ticks` (padded); asset 1 never switches."""
    row = [*ticks, *[PAD] * (CAPS.max_config_switches - len(ticks))]
    return jnp.asarray([row, [PAD] * CAPS.max_config_switches], I64)


def config_study(script, plans=(), switches=timeline(SWITCH)):
    """The toy study on `CAPS` with the scripted policy and the given timeline."""
    policy = ScriptedCommitments(script=script, plan=plans)
    study = toy.survey_1(caps=CAPS, policies=(policy,), config_switch_ticks=switches)
    return dataclasses.replace(study, channel_model=ToyConfigPhotometry())


# Lane 1 starts an observation at once that a Cancel after the switch ends; lane 0 holds
# a commitment admitted at 0 that starts at 150; a decision at 120 cancels, then observes
# on asset 1; one at 130 observes on lane 1 again (admitted after the switch) and one at
# 175, when the asset 1 observation ends, cancels that one.
MIXED = (
    observe(0, 0, 11, 0),
    observe(0, 0, 10, 0),
    wait(0, 120),
    cancel(1, 1),
    observe(1, 0, 12, 0),
    wait(1, 10),
    observe(2, 0, 11, 0),
    cancel(3, 1),
)
COMMITTED = (plan(0, 1, 150),)
# The same with a drain at 130: the commitment is refused when it comes to start.
DRAINED = (*MIXED[:-1], end(2))
# One observation on lane 0 after another, each at the Done of the one before.
CHAIN = tuple(observe(n, 0, 10, 0) for n in range(5))


def host(study):
    """A host run of one world, as its carry, trace and study."""
    truth, root = one_world(caps=CAPS)
    run = HostExecutor().run(study, truth, root, N_STEPS)
    return SimpleNamespace(study=study, carry=run.carry, trace=run.trace, run=run)


def records(result):
    """The valid log rows as `(channel, status, start, end, config_id, payload[0])`."""
    log = result.carry.log
    keep = np.flatnonzero(np.asarray(log.valid))
    return [
        (
            int(log.channel[i]),
            int(log.status[i]),
            int(log.start_tick[i]),
            int(log.end_tick[i]),
            int(log.config_id[i]),
            float(log.payload[i][0]),
        )
        for i in keep
    ]


@pytest.fixture(scope="module")
def mixed():
    """Host runs of the mixed script: the commitment starts, and is refused."""
    return SimpleNamespace(
        started=host(config_study(MIXED, COMMITTED)),
        refused=host(config_study(DRAINED, COMMITTED)),
    )


@pytest.fixture(scope="module")
def chain():
    """A compiled run of the chain script, with how often a scan body was traced."""
    study = config_study(CHAIN)
    truths, roots = worlds(1, CAPS)
    counter = SimpleNamespace(n=0)

    class Counting(CompiledExecutor):
        def transition(self, carry, i, root, study, truth, **kw):
            counter.n += 1
            return super().transition(carry, i, root, study, truth, **kw)

    executor = Counting()
    run = executor.run(study, truths, roots, 40)
    return SimpleNamespace(
        study=study, run=run, executor=executor, counter=counter, worlds=(truths, roots)
    )


@pytest.mark.medium(reason="one host run of the mixed script, 9 s cold")
def test_records_before_the_switch_carry_zero_and_after_it_one(mixed):
    """Fails if admission stamps a constant, the Done or Cancel tick's configuration or
    the wrong asset's: a record admitted at 0 on asset 0 and one admitted at 130 on
    asset 0 must differ, an asset without switches stays 0, and the stamp reaches the
    payload a version-2 model realizes.
    """
    rows = {(r[0], r[2]): r for r in records(mixed.started)}
    first = rows[11, 10]  # admitted at 0, cancelled at 120 (after the switch)
    asset_one = rows[12, 125]  # admitted at 120 on the asset that never switches
    second = rows[11, 140]  # admitted at 130 on asset 0, after the switch, cancelled
    assert (first[1], first[4]) == (PARTIAL, 0)
    assert (asset_one[1], asset_one[4]) == (COMPLETED, 0)
    assert (second[1], second[4]) == (PARTIAL, 1)
    configs = [r[4] for r in records(mixed.started)]
    assert configs.count(0) > 0 and configs.count(1) > 0
    assert all(r[5] == r[4] for r in records(mixed.started))  # payload = stamp


@pytest.mark.medium(reason="shares the mixed host run")
def test_a_commitment_keeps_the_configuration_it_was_admitted_under(mixed):
    """Fails if a commitment takes the configuration of its start tick, or loses it
    when it settles (started) or is cancelled at its start (a drain refuses it).
    """
    started = {r[0]: r for r in records(mixed.started)}
    assert started[10][1:5] == (COMPLETED, 160, 260, 0)  # starts after the switch
    refused = {r[0]: r for r in records(mixed.refused)}
    assert refused[10][1:5] == (PARTIAL, 150, 150, 0)


@pytest.mark.medium(reason="shares the chain compiled run")
def test_both_executors_agree_on_a_stamped_run(chain):
    """Fails if the compiled path stamps differently from the host path, or the
    configuration is missing from the compiled log.
    """
    truth, root = one_world(caps=CAPS)
    ref = HostExecutor().run(chain.study, truth, root, 40)
    assert histories_equal(chain.run.history(0), ref.history()).ok
    log = chain.run.carry.log
    keep = np.asarray(log.valid[0])
    assert np.asarray(log.config_id[0])[keep].tolist() == [0, 1, 1, 1, 1]


@pytest.mark.medium(reason="shares the chain compiled run")
def test_the_timeline_is_a_traced_leaf_that_reuses_the_program(chain):
    """Fails if a new timeline recompiles (the ticks became static), or is ignored:
    a switch at tick 0 puts every record in configuration 1 on the same program.
    """
    study = dataclasses.replace(
        chain.study,
        assets=dataclasses.replace(chain.study.assets, config_switch_ticks=timeline(0)),
    )
    before = chain.counter.n
    run = chain.executor.run(study, *chain.worlds, 40)
    assert chain.counter.n == before
    log = run.carry.log
    keep = np.asarray(log.valid[0])
    assert np.asarray(log.config_id[0])[keep].tolist() == [1] * 5


@pytest.mark.medium(reason="two compiled runs of the chain, cold")
def test_the_fork_battery_rebuilds_the_stamped_context_from_the_log():
    """Fails if the fork battery's recomputed row reads another configuration than the
    handler did: records 2 to 4 sit after the switch and their payload holds it.
    """
    assert battery.fork_check(config_study(CHAIN), *worlds(1, CAPS), 40)


def test_the_timeline_declaration_round_trips_and_default_studies_omit_it():
    """Fails if the ticks are dropped or rebuilt with another dtype, or a study without
    a timeline writes the field (every document hash would move).
    """
    study = toy.survey_1(
        caps=Capacities(max_config_switches=2),
        config_switch_ticks=timeline(30, 90),
    )
    assert preflight(study) == []
    text = dump_study(study, TOY_KINDS).to_json()
    loaded = load_study(text, TOY_KINDS)
    assert loaded.assets.config_switch_ticks.dtype == jnp.int64
    assert loaded.assets.config_switch_ticks.tolist() == timeline(30, 90).tolist()
    assert dump_study(loaded, TOY_KINDS).to_json() == text
    plain = json.loads(dump_study(toy.survey_1(), TOY_KINDS).to_json())
    assert "config_switch_ticks" not in json.dumps(plain["study"]["assets"])
