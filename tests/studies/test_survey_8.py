"""Survey 8, a second suppression system: fuel runs out on one asset, a policy stops it.

Two assets each own lanes (asset 0 owns lanes 0 and 1, asset 1 owns lane 2) and the
only priced resource is fuel, resource 0, held per asset. The script, keyed by tick so a
different number of releases cannot move it, asks asset 0 for fuel it does not have and
keeps asking asset 1 for work. A level-1 array policy reads the balances and, once asset
0 is dry, emits `Stop(ASSET)` for it. The merge-time guard (medium) runs the study and
its reference on the host executor and checks (a) to (d); the control and the
two-executor parity are slow.
"""

import dataclasses
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import fault_tie
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import (
    F64,
    I64,
    AdmissionCode,
    Disposition,
    EffectKind,
    RecordStatus,
    Scope,
)
from spaceodyssey.core.executors.base import index_world
from spaceodyssey.core.records import log_columns
from spaceodyssey.studies import toy
from tests.core.conformance_helpers import ScriptedFaults
from tests.core.helpers import worlds
from tests.core.program_helpers import ScriptedPrograms, end, observe, wait
from tests.studies.survey_helpers import run_on, same_histories

N_STEPS = 200
NO_FUNDS, CLOSED, BUSY = (
    AdmissionCode.NO_FUNDS,
    AdmissionCode.LANE_CLOSED,
    AdmissionCode.LANE_BUSY,
)
DRY, WET = 0, 1  # the asset that runs out of fuel, the asset that does not
LANES = {10: 0, 11: 1, 12: 2}  # channel id to lane
FUEL_DRY, FUEL_WET, FUEL_FULL = 2.0, 10.0, 10.0
N_EFFECTS = 5  # three scripted rows, the wait, and the slot the policy's Stop takes

# Ticks of the script (a decision runs at each: the waits chain them).
FIRST, SECOND, STOP_SEEN, AFTER, FINISH = 0, 120, 140, 230, 300
FAULT_TICK = 140  # on lane 2, in the settle stage before the tick-140 decision
SCRIPT = (
    # asset 0 and asset 1 each start one acquisition (fuel 2 -> 1 on asset 0)
    observe(FIRST, 0, 10, 0),
    observe(FIRST, 2, 12, 0),
    wait(FIRST, SECOND - FIRST),
    # asset 0 takes its last fuel; lane 1 of the same asset finds none (NO_FUNDS)
    observe(SECOND, 0, 10, 0),
    observe(SECOND, 1, 11, 0),
    observe(SECOND, 2, 12, 0),
    wait(SECOND, STOP_SEEN - SECOND),
    # the fault has failed lane 2's work; asset 1 starts again on that lane
    observe(STOP_SEEN, 2, 12, 0),
    wait(STOP_SEEN, AFTER - STOP_SEEN),
    # after the Stop both lanes of asset 0 are asked; asset 1 keeps working
    observe(AFTER, 0, 10, 0),
    observe(AFTER, 1, 11, 0),
    observe(AFTER, 2, 12, 0),
    wait(AFTER, FINISH - AFTER),
    end(FINISH),
)


def _as_rows(rows):
    """The script as an int table; the shared helper's table holds only eight rows."""
    return jnp.asarray(np.asarray(rows, np.int64), I64)


class FuelWatch(ScriptedPrograms):
    """The script, read by tick, plus a policy that stops an asset that ran dry.

    A level-1 array policy: it reads the balances in the snapshot, finds the row of
    the dry asset and the fuel column, and while that balance is spent it puts one
    `Stop(ASSET)` in the last effect row, once (`pstate[0]` remembers it). With
    `stops=False` it never stops, which is the control.
    """

    script: object = eqx.field(converter=_as_rows)
    stops: object = eqx.field(default=True, converter=lambda x: jnp.asarray(x, bool))

    def declare(self):
        """The script's declaration with the asset stop added when the policy has it."""
        base = super().declare()
        if not bool(self.stops):
            return base
        stops = (*base.stops, (Scope.ASSET, Disposition.CLOSE))
        return Declaration(programs=base.programs, stops=stops)

    def decide(self, pstate, view, study, key):
        """Emit the scripted rows of this tick, then the asset Stop when it is due."""
        by_tick = eqx.tree_at(lambda v: v.n_dec, view, view.tick)
        eff, _, _ = super().decide(pstate, by_tick, study, key)
        rows = view.balance_rows  # kind 0 is an asset row, indexed within the kind
        row = jnp.argmax((rows[:, 0] == 0) & (rows[:, 1] == DRY))
        dry = view.balance[row, 0] <= 0.0
        due = dry & (pstate[0] == 0.0) & self.stops
        last = N_EFFECTS - 1

        def put(column, value):
            return column.at[last].set(jnp.where(due, value, column[last]))

        eff = dataclasses.replace(
            eff,
            kind=put(eff.kind, EffectKind.STOP),
            lane=put(eff.lane, 0),  # a lane of the asset names it
            scope=put(eff.scope, Scope.ASSET),
            disposition=put(eff.disposition, Disposition.CLOSE),
            group=put(eff.group, view.tick * N_EFFECTS + last),
        )
        return eff, pstate.at[0].set(jnp.where(due, 1.0, pstate[0])), jnp.zeros((), F64)


def survey_study(*, fuel=FUEL_DRY, stops=True, faults=((FAULT_TICK, 2, 2, 2),)):
    """The toy study with two assets, fuel only, balances on, and the scripted faults.

    Every channel costs one unit of fuel and nothing else, so the program and shared
    rows never move. A fault row is `(tick, lane, channel index, generation)`.
    """
    caps = Capacities(n_effects=N_EFFECTS, snapshot_balances=True)
    study = toy.survey_1(caps=caps, policies=(FuelWatch(script=SCRIPT, stops=stops),))
    cost = jnp.zeros_like(study.channels.cost).at[:, 0].set(1.0)
    channels = dataclasses.replace(study.channels, cost=cost, charge=cost)
    budget = study.budget.at[DRY, 0].set(fuel).at[WET, 0].set(FUEL_WET)
    study = dataclasses.replace(study, channels=channels, budget=budget)
    process = ScriptedFaults(
        rows=faults,
        tie=int(fault_tie(study.channels, 0)),
        channel_ids=tuple(int(x) for x in np.asarray(study.channels.id)),
    )
    return eqx.tree_at(lambda s: s.process, study, process)


def _run(study, executor="host"):
    """The study on one executor, one world: the batched run."""
    truths, roots = worlds(1, study.caps)
    return run_on(executor, study, truths, roots, N_STEPS)


def _first_world(run):
    """The carry and trace of world 0 of a batched run."""
    carry = jax.tree_util.tree_map(lambda x: x[0], run.carry)
    return SimpleNamespace(carry=carry, trace=index_world(run.trace, 0))


def _outcomes(world):
    """Admission code of each scripted Observe: `{(tick, channel id): code}`.

    A decision is woken once per tick, so the e-th scripted row of a tick is the e-th
    effect row of that tick's decision.
    """
    t = world.trace
    tick, handled = np.asarray(t.tick), np.asarray(t.handled)
    kinds, codes = np.asarray(t.eff_kind), np.asarray(t.admit_code)
    out = {}
    for step in np.flatnonzero(handled & (kinds == EffectKind.OBSERVE).any(axis=1)):
        rows = [r for r in SCRIPT if r[0] == tick[step]]
        for e, r in enumerate(rows):
            if r[1] == EffectKind.OBSERVE:
                assert (int(tick[step]), r[3]) not in out
                out[(int(tick[step]), r[3])] = int(codes[step, e])
    return out


def _stop_ticks(world):
    """Ticks of every Stop effect the run handled, in order."""
    t = world.trace
    steps = np.flatnonzero(
        np.asarray(t.handled) & (np.asarray(t.eff_kind) == EffectKind.STOP).any(axis=1)
    )
    return [int(np.asarray(t.tick)[s]) for s in steps]


def _records(world, study, lane=None):
    """The log rows (all, or one lane's) as columns by name."""
    log, n = world.carry.log, int(world.carry.n_log)
    cols = {
        k: np.asarray(v)[:n]
        for k, v in log_columns(log).items()
        if np.ndim(v) and v.shape[0] == study.caps.n_log
    }
    keep = np.ones(n, bool) if lane is None else cols["lane"] == lane
    return {k: v[keep] for k, v in cols.items()}


def _admit_ticks(study, rows):
    """The tick each record was admitted: its start less its lane's transition."""
    return rows["start_tick"] - np.asarray(study.assets.transition_ticks)[rows["lane"]]


def _assert_a_and_b(world, study):
    """Rows (a) and (b): one policy Stop after the exhaustion tick, and the refusals
    change from `NO_FUNDS` to the closure code across it.
    """
    out = _outcomes(world)
    on_dry = _records(world, study)
    on_dry = _admit_ticks(study, on_dry)[
        np.isin(_records(world, study)["lane"], (0, 1))
    ]
    exhausted = int(np.sort(on_dry)[int(FUEL_DRY) - 1])  # the tick of the last fuel
    stops = _stop_ticks(world)
    asset_stops = [t for t in stops if t < FINISH]
    # (a) the Stop effect exists, once, and comes after the exhaustion tick
    assert exhausted == SECOND
    assert len(asset_stops) == 1
    assert asset_stops[0] > exhausted
    # (b) before it the refusal is NO_FUNDS, after it the closure code
    before = [
        c for (t, _), c in out.items() if t <= asset_stops[0] and c != AdmissionCode.OK
    ]
    after = [
        c for (t, _), c in out.items() if t > asset_stops[0] and c != AdmissionCode.OK
    ]
    assert before == [NO_FUNDS]
    assert after == [CLOSED, CLOSED]


@pytest.fixture(scope="module")
def stopped():
    study = survey_study()
    return study, _run(study)


@pytest.mark.medium  # merge-time guard: the study and its reference, host executor
def test_a_policy_stops_the_dry_asset_and_the_other_asset_is_unaffected(stopped):
    """B7 row 8 (a) to (d). Fails if the policy does not stop the dry asset or stops
    it before it is dry, if the refusal code does not change across the Stop, if the
    other asset's records or balances differ from a run with fuel, or if the fault on
    its lane does anything but fail the work on that lane.
    """
    study, host = stopped
    assert int(np.asarray(host.carry.n_log)[0]) == 6 > 0
    world = _first_world(host)
    _assert_a_and_b(world, study)
    out = _outcomes(world)
    assert sum(c == AdmissionCode.OK for c in out.values()) == 6
    assert sum(c == NO_FUNDS for c in out.values()) == 1
    assert sum(c == CLOSED for c in out.values()) == 2

    # (c) asset 1 is admitted after the exhaustion tick and after the Stop, and its
    # records and balances are those of the run in which asset 0 never ran dry
    rows = _records(world, study, lane=2)
    admitted = _admit_ticks(study, rows)
    assert admitted.tolist() == [0, 120, 140, 230]
    assert (admitted > SECOND).sum() == 2
    ref_host = _run(survey_study(fuel=FUEL_FULL))
    ref_world = _first_world(ref_host)
    ref = _records(ref_world, study, lane=2)
    assert ref.keys() == rows.keys()
    # the acquisition id is the queue's internal id, which asset 0's events move
    for name, column in rows.items():
        if name != "acquisition_id":
            assert np.array_equal(column, ref[name]), name
    balance, ref_balance = (
        np.asarray(world.carry.ledger.balance),
        np.asarray(ref_world.carry.ledger.balance),
    )
    others = [i for i in range(balance.shape[0]) if i != DRY]
    assert np.array_equal(balance[others], ref_balance[others])
    assert balance[WET, 0] == FUEL_WET - 4
    assert (
        balance[DRY, 0] == 0.0 and ref_balance[DRY, 0] == FUEL_FULL - 4
    )  # the dry run really differed

    # (d) the fault failed the work on lane 2 at its tick, charged in full, and the
    # work on lane 0 and its later requests on lane 2 are not affected: FAULT cancels
    # and does not close the lane
    everything = _records(world, study)
    failed = everything["status"] == RecordStatus.FAILED
    assert failed.sum() == 1
    assert everything["lane"][failed].tolist() == [2]
    assert everything["end_tick"][failed].tolist() == [FAULT_TICK]
    assert everything["charged"][failed][0, 0] == 1.0
    on_lane0 = everything["lane"] == 0
    assert on_lane0.sum() == 2
    assert (everything["status"][on_lane0] == RecordStatus.COMPLETED).all()


@pytest.mark.slow  # one more host run of the same compiled shapes (K6)
def test_control_without_the_policy_stop_every_refusal_is_no_funds_and_row_b_fails():
    """Control for (a) and (b). The same study with the Stop removed refuses every
    request on the dry asset `NO_FUNDS`, no Stop effect is emitted before the campaign
    ends, and the assertions that row (a) and row (b) make fail on it.
    """
    study = survey_study(stops=False)
    host = _run(study)
    world = _first_world(host)
    out = _outcomes(world)
    assert [t for t in _stop_ticks(world) if t < FINISH] == []
    assert sum(c == NO_FUNDS for c in out.values()) == 3  # the three asked of asset 0
    assert sum(c == CLOSED for c in out.values()) == 0
    assert out[(AFTER, 10)] == out[(AFTER, 11)] == NO_FUNDS
    with pytest.raises(AssertionError):
        _assert_a_and_b(world, study)


@pytest.mark.slow  # the compiled executor compiles every program again (K6)
def test_both_executors_make_the_same_history_for_the_dry_asset_study(stopped):
    """Parity for (a) to (d): ints exact, floats in the go band, on both executors."""
    study, host = stopped
    compiled = _run(study, "compiled")
    assert int(np.asarray(compiled.carry.n_log)[0]) == 6 > 0
    assert same_histories(compiled, host, 1)
