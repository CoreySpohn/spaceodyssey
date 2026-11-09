"""Asset ops state: a process moves it, and admission prices the occupancy by it.

A failure process sets ops state 1 on asset 0 at tick 100 and writes a notice released
25 ticks later. An admission is debited from the cost row of the state at that moment;
that state is stored with the occupancy, so its completion, cancel and refund are priced
from the stored state whatever the asset's state is by then, and the debit is returned
exactly. Every run below shares one compiled study structure (commitments and two ops
states); scenarios are script data and the process leaves.
"""

import dataclasses
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.codec import dump_study, load_study
from spaceodyssey.core import CompiledExecutor, HostExecutor, battery, histories_equal
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import F64, Audience, RecordStatus
from spaceodyssey.core.preflight import preflight
from spaceodyssey.core.snapshot import snapshot_view
from spaceodyssey.studies import toy
from tests.codec.test_document import TOY_KINDS
from tests.core.commitment_helpers import ScriptedCommitments, plan
from tests.core.helpers import one_world, worlds
from tests.core.program_helpers import cancel, observe, wait
from tests.core.test_realize_context import _epoch_registry
from tests.doubles import OpsFailure, ToyOpsPhotometry

CAPS = _epoch_registry().lower(Capacities(n_effects=4, n_commitments=2, n_ops=2))
FAIL = 100  # the tick asset 0 moves to ops state 1
DELAY = 25  # the failure record is released this long after it is written
COSTS = (2.0, 5.0)  # the cost of every channel in ops state 0 and in ops state 1
N_STEPS = 200
FAILURE_CHANNEL = 50


# Decisions: 0 at tick 0 admits A (lane 0, ends 110), B (lane 1) and a commitment CM
# on lane 0 that starts at 250, all in state 0. The failure at 100 moves asset 0 to
# state 1 and its process event wakes decision 1, which cancels B (admitted in state
# 0). The release of B's record at 120 wakes decision 2, which admits C (lane 1) and D
# (lane 0, ends 230) in state 1; the release of the failure notice at 125 wakes
# decision 3 and that of A at 130 wakes decision 4, both with no rows; decision 5 at 145
# cancels C.
SCRIPT = (
    observe(0, 0, 10, 0),
    observe(0, 0, 11, 0),
    observe(0, 0, 10, 0),
    cancel(1, 1),
    observe(2, 0, 11, 0),
    observe(2, 0, 10, 0),
    wait(2, 25),
    cancel(5, 1),
)
PLANS = (plan(0, 2, 250),)
LOW, HIGH = COSTS
COMPLETED, PARTIAL = int(RecordStatus.COMPLETED), int(RecordStatus.PARTIAL)


def ops_study(script, plans=(), *, fail=FAIL, delay=DELAY, costs=COSTS, caps=CAPS):
    """The toy study on `caps`: two ops states, per-state costs and a failing asset 0.

    Every channel costs `costs[s]` per resource in state `s`; the scripted policy runs
    `script` with the commitment `plans`; the failure fires at `fail` and its record is
    released `delay` ticks later. Budgets are large, so no admission is refused.
    """
    base = toy.survey_1(caps=caps)
    shape = (caps.n_channels, caps.n_resources)
    by_ops = jnp.stack([jnp.full(shape, c, F64) for c in costs])
    channels = dataclasses.replace(
        base.channels,
        cost=jnp.full(shape, costs[0], F64),
        charge=jnp.full(shape, costs[0], F64),
        cost_by_ops=by_ops,
    )
    process = OpsFailure(
        ticks=(fail,),
        values=(1.0,),
        ids=(0,),
        revisions=(0,),
        statuses=(int(RecordStatus.FAILED),),
        kinds=(toy.RECORD_AUXILIARY,),
        audience=int(Audience.POLICY),
        delay=delay,
    )
    study = toy.survey_1(
        caps=caps,
        channels=channels,
        budget=jnp.full((caps.n_balance_rows, caps.n_resources), 100.0, F64),
        policies=(ScriptedCommitments(script=script, plan=plans),),
        process=process,
    )
    return dataclasses.replace(study, channel_model=ToyOpsPhotometry())


def host(study):
    """A host run of one world: the study, carry, trace and run."""
    truth, root = one_world(caps=study.caps)
    run = HostExecutor().run(study, truth, root, N_STEPS)
    return SimpleNamespace(study=study, carry=run.carry, trace=run.trace, run=run)


def records(result):
    """The valid log rows as dicts (channel, status, ticks, ops, charge, payload)."""
    log = result.carry.log
    out = []
    for i in np.flatnonzero(np.asarray(log.valid)):
        out.append(
            dict(
                channel=int(log.channel[i]),
                status=int(log.status[i]),
                start=int(log.start_tick[i]),
                end=int(log.end_tick[i]),
                ops=int(log.ops[i]),
                charged=float(log.charged[i][0]),
                released=int(log.released_tick[i]),
                first=float(log.payload[i][0]),
            )
        )
    return out


def by_channel(rows, channel, **match):
    """The one record on `channel` whose fields equal `match`."""
    hits = [r for r in rows if r["channel"] == channel and match.items() <= r.items()]
    assert len(hits) == 1, (channel, match, rows)
    return hits[0]


@pytest.fixture(scope="module")
def run():
    """A host run of the script: A, B, CM in state 0, the failure, then C and D in 1."""
    return host(ops_study(SCRIPT, PLANS))


@pytest.fixture(scope="module")
def compiled():
    """A compiled run of the same study, and how often a scan body was traced."""
    study = ops_study(SCRIPT, PLANS)
    truths, roots = worlds(1, CAPS)
    counter = SimpleNamespace(n=0)

    class Counting(CompiledExecutor):
        def transition(self, carry, i, root, study, truth, **kw):
            counter.n += 1
            return super().transition(carry, i, root, study, truth, **kw)

    executor = Counting()
    result = executor.run(study, truths, roots, N_STEPS)
    return SimpleNamespace(
        study=study,
        run=result,
        executor=executor,
        counter=counter,
        worlds=(truths, roots),
    )


@pytest.mark.medium(reason="one host run of the script, 8 s cold")
def test_an_admission_is_debited_from_the_state_it_finds(run):
    """Fails if admission always reads state 0 (or always the current table), or the
    process does not move the state: three admissions before the failure cost 2 each
    and the two after it cost 5 each, and only asset 0 changes state.
    """
    ledger = run.carry.ledger
    assert np.asarray(ledger.ops_state).tolist() == [1, 0]
    assert float(ledger.debited[0, 0]) == pytest.approx(3 * LOW + 2 * HIGH)
    log = records(run)
    assert by_channel(log, 11, start=130)["ops"] == 1  # C, admitted after the failure
    assert by_channel(log, 10, start=130)["ops"] == 1  # D
    assert [r["ops"] for r in log if r["start"] < FAIL] == [0, 0]  # A and B


@pytest.mark.medium(reason="shares the host run")
def test_completion_and_cancel_charge_the_stored_admission_state(run):
    """Fails if a settlement or a cancel recomputes the price from the asset's state
    at that moment: A completes at 110 and B is cancelled at 100, both after the
    failure but admitted before it, and keep price 2; C is cancelled at 150 and keeps
    price 5 for its 15 elapsed ticks of 150. The refund is the stored price times the
    unused fraction, so the ledger's refund total is exact.
    """
    log = records(run)
    a = by_channel(log, 10, start=10)
    b = by_channel(log, 11, start=10)
    c = by_channel(log, 11, start=130)
    assert (a["status"], a["ops"], a["charged"]) == (COMPLETED, 0, LOW)
    assert (b["status"], b["ops"]) == (PARTIAL, 0)
    assert b["charged"] == pytest.approx(LOW * 90 / 150)
    assert (c["status"], c["ops"]) == (PARTIAL, 1)
    assert c["charged"] == pytest.approx(HIGH * 15 / 150)
    refunded = LOW * 60 / 150 + HIGH * 135 / 150
    assert float(run.carry.ledger.refunded[0, 0]) == pytest.approx(refunded)


@pytest.mark.medium(reason="shares the host run")
def test_a_commitment_keeps_the_state_it_was_admitted_under(run):
    """Fails if a commitment is priced at its start (the asset is in state 1 by then)
    in place of its admission: it is admitted at 0, starts at 250 and completes at 360
    charged 2 in state 0.
    """
    cm = by_channel(records(run), 10, start=260)
    assert (cm["status"], cm["ops"], cm["charged"]) == (COMPLETED, 0, LOW)
    assert np.asarray(run.carry.ledger.ops_state)[0] == 1


@pytest.mark.medium(reason="shares the host run")
def test_a_version_two_model_realizes_with_the_stored_state(run):
    """Fails if the realize context carries the current state or none: the model writes
    the state into its first payload column, and every record shows the state it was
    admitted under, including the commitment realized after the failure.
    """
    rows = [r for r in records(run) if r["channel"] != FAILURE_CHANNEL]
    assert [r["first"] for r in rows] == [float(r["ops"]) for r in rows]
    assert {r["ops"] for r in rows} == {0, 1}


@pytest.mark.medium(reason="shares the host run")
def test_the_failure_record_is_released_after_the_declared_delay(run):
    """Fails if the notice is released at the failure tick, or the delay is ignored:
    the row written at 100 is released at 125, and the policy's view of the log holds
    no ops column.
    """
    notice = by_channel(records(run), FAILURE_CHANNEL)
    assert (notice["start"], notice["released"]) == (FAIL, FAIL + DELAY)
    view = snapshot_view(run.carry, Audience.POLICY, run.study.caps)
    assert view.log.ops is None


@pytest.mark.medium(reason="shares the compiled run")
def test_both_executors_agree_and_the_delay_is_a_leaf(compiled):
    """Fails if the compiled path prices differently from the host path, or the delay
    or the failure tick became static: a later failure on the same program moves the
    release with it and adds no trace.
    """
    truth, root = one_world(caps=CAPS)
    ref = HostExecutor().run(compiled.study, truth, root, N_STEPS)
    assert histories_equal(compiled.run.history(0), ref.history()).ok
    log = compiled.run.carry.log
    keep = np.asarray(log.valid[0])
    assert np.asarray(log.ops[0])[keep].tolist() == [0, 0, 0, 1, 1, 0]
    later = ops_study(SCRIPT, PLANS, fail=FAIL + 5, delay=2 * DELAY)
    before = compiled.counter.n
    result = compiled.executor.run(later, *compiled.worlds, N_STEPS)
    assert compiled.counter.n == before
    rel = np.asarray(result.carry.log.released_tick[0])
    chan = np.asarray(result.carry.log.channel[0])
    assert rel[chan == FAILURE_CHANNEL].tolist() == [FAIL + 5 + 2 * DELAY]


@pytest.mark.medium(reason="two compiled runs, cold")
def test_the_fork_battery_rebuilds_the_context_from_the_log(compiled):
    """Fails if the battery's recomputed row reads another state than the handler did:
    its records straddle the failure and their payload holds the stored state.
    """
    assert battery.fork_check(compiled.study, *compiled.worlds, N_STEPS)


def test_the_cost_table_round_trips_and_default_studies_omit_it():
    """Fails if the table loses its dtype or shape, n_ops is dropped, or a study without
    ops writes either field (every document hash would move).
    """
    base = toy.survey_1(caps=Capacities(n_ops=3))
    table = jnp.arange(27.0).reshape(3, 3, 3)
    channels = dataclasses.replace(base.channels, cost_by_ops=table)
    study = dataclasses.replace(base, channels=channels)
    assert preflight(study) == []
    text = dump_study(study, TOY_KINDS).to_json()
    loaded = load_study(text, TOY_KINDS)
    assert loaded.caps.n_ops == 3
    assert loaded.channels.cost_by_ops.dtype == jnp.float64
    assert loaded.channels.cost_by_ops.tolist() == table.tolist()
    assert dump_study(loaded, TOY_KINDS).to_json() == text
    plain = json.dumps(json.loads(dump_study(toy.survey_1(), TOY_KINDS).to_json()))
    assert "cost_by_ops" not in plain and "n_ops" not in plain


def test_a_study_without_ops_carries_no_ops_leaf():
    """Fails if the carry, the log or an event row holds an ops leaf at the default."""
    study = toy.survey_1()
    assert dataclasses.replace(study).caps.ops is False
    truth, _ = one_world(caps=study.caps)
    from spaceodyssey.core.carry import init_carry

    carry = init_carry(study, truth)
    assert (carry.ev.ops, carry.log.ops) == (None, None)
    assert (carry.ledger.ops_state, carry.ledger.cm_ops) == (None, None)
