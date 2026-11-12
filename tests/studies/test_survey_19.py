"""Survey 19, joint planning: a facility that holds no time changes nothing.

Two facilities plan jointly (one admission group spanning both assets, admitted whole);
a third facility with a zero time balance and no channel is added. The runs are compared
row by row: the draw key and payload of each record, each belief leaf, each trace field
of the actions, and each balance row of the first two facilities. The asset-key mutant
(`asset * n_assets`) is run through the same comparison and must fail it.
"""

import dataclasses
import types

import jax
import numpy as np
import pytest

from spaceodyssey.core import HostExecutor, handlers
from spaceodyssey.core.enums import AdmissionCode, EffectKind
from spaceodyssey.core.keys import asset_coordinate
from spaceodyssey.core.preflight import preflight
from tests.core.helpers import one_world
from tests.studies.scenarios.survey_19 import (
    FACILITY_CHANNELS,
    N_ROUNDS,
    counting_asset,
    measurement_keys,
    survey_19,
    with_idle_facility,
)

N_STEPS = 100
OK = int(AdmissionCode.OK)


def _collect(study, asset_of=asset_coordinate):
    """One host run's rows, keyed for the comparison."""
    truth, root = one_world(caps=study.caps)
    run = HostExecutor().run(study, truth, root, N_STEPS)
    log = run.carry.log
    rows = np.flatnonzero(np.asarray(log.valid))
    n_assets = study.caps.n_assets
    return types.SimpleNamespace(
        study=study,
        run=run,
        channel=np.asarray(log.channel)[rows],
        keys=np.asarray(measurement_keys(study, root, log, rows, asset_of)),
        payload=np.asarray(log.payload)[rows],
        beliefs=[np.asarray(x) for x in jax.tree_util.tree_leaves(run.carry.belief)],
        trace={
            f.name: np.asarray(getattr(run.trace, f.name))
            for f in dataclasses.fields(run.trace)
        },
        # facility rows first, then the program and shared rows after the facilities
        balance=np.asarray(run.carry.ledger.balance),
        n_assets=n_assets,
    )


def differences(a, b):
    """The record, belief, action or balance rows of two runs that differ.

    Rows are compared one by one, not through a fingerprint of the whole snapshot. `b`
    is the run with the extra facility, so its balance rows after the facilities
    start one row later.
    """
    found = []
    if a.keys.shape != b.keys.shape:
        return ["the number of records"]
    for i, (ka, kb) in enumerate(zip(a.keys, b.keys, strict=True)):
        if not np.array_equal(ka, kb):
            found.append(f"key of record {i}")
    for i, (pa, pb) in enumerate(zip(a.payload, b.payload, strict=True)):
        if not np.array_equal(pa, pb):
            found.append(f"payload of record {i}")
    for i, (ba, bb) in enumerate(zip(a.beliefs, b.beliefs, strict=True)):
        if not np.array_equal(ba, bb):
            found.append(f"belief leaf {i}")
    for name in a.trace:
        if not np.array_equal(a.trace[name], b.trace[name]):
            found.append(f"action field {name}")
    for r in range(a.n_assets):
        if not np.array_equal(a.balance[r], b.balance[r]):
            found.append(f"balance row {r}")
    rest = zip(a.balance[a.n_assets :], b.balance[a.n_assets + 1 :], strict=True)
    for r, (ra, rb) in enumerate(rest):
        if not np.array_equal(ra, rb):
            found.append(f"balance row {a.n_assets + r}")
    return found


@pytest.fixture(scope="module")
def runs():
    """The two facility run and the three facility run."""
    study = survey_19()
    more = with_idle_facility(study)
    assert preflight(study) == [] and preflight(more) == []
    assert float(more.budget[study.caps.n_assets].sum()) == 0.0
    return types.SimpleNamespace(small=_collect(study), big=_collect(more))


@pytest.mark.slow(reason="two host runs of the joint study, 6 s cold")
def test_the_pair_of_facilities_is_admitted_whole(runs):
    """Fails if the joint group is split or refused: each round's two requests share one
    admission group and are admitted together (counted: N_ROUNDS rounds, each with
    both members admitted), and the records land on both assets.
    """
    trace = runs.small.trace
    pair = (
        (trace["eff_kind"][:, :2] == EffectKind.OBSERVE).all(axis=1)
        & trace["handled"]
        & trace["wake"]
    )
    codes = trace["admit_code"][pair][:, :2]
    assert len(codes) == N_ROUNDS and (codes == OK).all()
    lanes = np.asarray(runs.small.study.channels.lane)
    ids = np.asarray(runs.small.study.channels.id).tolist()
    assets = {
        int(runs.small.study.assets.lane_asset[lanes[ids.index(c)]])
        for c in FACILITY_CHANNELS
    }
    assert assets == {0, 1}
    assert sorted(set(runs.small.channel.tolist())) == sorted(FACILITY_CHANNELS)
    assert len(runs.small.channel) == len(FACILITY_CHANNELS) * N_ROUNDS


@pytest.mark.slow(reason="shares the two host runs")
def test_an_idle_facility_changes_no_key_payload_belief_action_or_balance_row(runs):
    """Fails if a facility with a zero balance moves any of the first two facilities'
    rows. Counted: four records (two per facility) with their keys and payloads, the
    belief leaves, the action fields of the trace and the balance rows are all
    compared, and the rows compared are not empty.
    """
    small, big = runs.small, runs.big
    assert small.keys.shape == (len(FACILITY_CHANNELS) * N_ROUNDS, 2)
    assert len(small.beliefs) > 0 and len(small.trace) > 0
    assert big.balance.shape[0] == small.balance.shape[0] + 1
    assert differences(small, big) == []


@pytest.mark.slow(reason="four more host runs after a recompile, 8 s cold")
def test_the_asset_key_mutant_fails_the_key_and_payload_comparison(runs, monkeypatch):
    """Fails if the comparison cannot see a lookup that counts facilities
    (`asset * n_assets`): with it, adding a facility must move the key and payload of
    every asset-1 record (counted: two) and none of asset 0's.
    """
    small_study, big_study = runs.small.study, runs.big.study
    monkeypatch.setattr(handlers, "asset_coordinate", counting_asset)
    jax.clear_caches()  # compiled programs closed over the real lookup
    try:
        small = _collect(small_study, counting_asset)
        big = _collect(big_study, counting_asset)
    finally:
        monkeypatch.undo()
        jax.clear_caches()
    found = differences(small, big)
    moved = {int(i) for i, c in enumerate(small.channel) if c == FACILITY_CHANNELS[1]}
    assert len(moved) == N_ROUNDS
    assert {f for f in found if f.startswith("key")} == {
        f"key of record {i}" for i in moved
    }
    assert {f for f in found if f.startswith("payload")} == {
        f"payload of record {i}" for i in moved
    }
