"""Golden content hashes of every conformance corpus history, on both executors.

`golden_histories.json` maps `case/executor/world` to the SHA-256 digests
(`spaceodyssey.inputs.digest`) of the integer and floating-point parts of that world's
canonical history (`canonical_history`, as the shared corpus run store records it).
Any change to any history, by one bit, fails here. A deliberate change regenerates
the manifest with `SPACEODYSSEY_WRITE_GOLDEN=1` and names every moved entry in a
change record under `docs/changes/`.

`golden_pin.json` freezes the manifest as it stood at a recorded commit, with the
snapshot fingerprints of every corpus run and the history digests of a small fault
fixture (the corpus never handles a `FAULT`). Nothing regenerates the pin: a pinned
entry that differs fails, documented or not. It is written once, by running this file
with `SPACEODYSSEY_CAPTURE_PIN=<full commit sha>` while no pin file exists. Float
digests depend on the platform; the pin was taken on macOS arm64.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from spaceodyssey.core.conformance import EXECUTORS
from spaceodyssey.core.enums import EventKind, RecordStatus
from spaceodyssey.inputs import digest
from tests.core.conformance_helpers import (
    N_WORLDS,
    fault_run,
    run_fingerprints,
    runs,
)
from tests.core.corpus import CORPUS

pytestmark = pytest.mark.clear_caches_between_tests

MANIFEST = Path(__file__).with_name("golden_histories.json")
PIN = Path(__file__).with_name("golden_pin.json")
WRITE = os.environ.get("SPACEODYSSEY_WRITE_GOLDEN", "0") == "1"
CAPTURE = os.environ.get("SPACEODYSSEY_CAPTURE_PIN", "")
N_PINNED = 116


def entry_key(case, executor, world):
    """Manifest key of one world of one case on one executor."""
    return f"{case}/{executor}/{world}"


def history_digests(history):
    """Digests of the integer and float arrays of a canonical history."""
    return {
        "ints": digest(history.ints, "ints"),
        "floats": digest(history.floats, "floats"),
    }


def case_digests(name, executor):
    """Manifest entries of one case on one executor, from the shared run store."""
    return {
        entry_key(name, executor, run.world): history_digests(run.history)
        for run in runs(name, executor)
    }


def expected_keys():
    """Every key the manifest must hold: each case, executor and world."""
    return {
        entry_key(case.name, executor, w)
        for case in CORPUS
        for executor in EXECUTORS
        for w in range(N_WORLDS)
    }


def test_manifest_covers_every_case_executor_and_world():
    """Fails if a corpus case, executor or world has no golden entry, or one is stale.

    With `SPACEODYSSEY_WRITE_GOLDEN=1` the manifest is first rewritten from the
    current runs.
    """
    if WRITE:
        current = {}
        for case in CORPUS:
            for executor in EXECUTORS:
                current.update(case_digests(case.name, executor))
        MANIFEST.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
    manifest = json.loads(MANIFEST.read_text())
    assert set(manifest) == expected_keys()


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.name)
def test_history_matches_its_golden_digest(case, executor):
    """Fails if any integer or float of a corpus history changed, by even one bit."""
    manifest = json.loads(MANIFEST.read_text())
    got = case_digests(case.name, executor)
    moved = {k: v for k, v in got.items() if manifest.get(k) != v}
    assert not moved, sorted(moved)


def test_a_one_bit_change_moves_the_digest():
    """Fails if the digest ignores a float bit or an integer of the history."""
    run = runs("baseline", "host")[0]
    base = history_digests(run.history)
    floats = dict(run.history.floats)
    payload = floats["payload"].copy()
    payload.view("u8").flat[0] ^= 1
    floats["payload"] = payload
    ints = dict(run.history.ints)
    ints["carry.tick"] = ints["carry.tick"] + 1
    assert digest(floats, "floats") != base["floats"]
    assert digest(ints, "ints") != base["ints"]


# ---- the pin -----------------------------------------------------------------------


def fault_digests(executor, close_lane=False):
    """History digests of each world of the fault fixture on one executor."""
    result = fault_run(executor, close_lane)
    return {
        f"{executor}/{w}": history_digests(result.history(w)) for w in range(N_WORLDS)
    }


def _capture(commit):
    """Write the pin from the current runs; refuses to replace an existing pin."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("SPACEODYSSEY_CAPTURE_PIN names a full commit sha")
    if PIN.exists():
        raise FileExistsError(f"{PIN.name} exists; a pin is never rewritten")
    histories, fingerprints, fault = {}, {}, {}
    for case in CORPUS:
        for executor in EXECUTORS:
            histories.update(case_digests(case.name, executor))
            for run in runs(case.name, executor):
                key = entry_key(case.name, executor, run.world)
                fingerprints[key] = run_fingerprints(run)
    for executor in EXECUTORS:
        fault.update(fault_digests(executor))
    pin = {
        "source_commit": commit,
        "platform": "macOS arm64; float digests differ on other platforms",
        "histories": histories,
        "fingerprints": fingerprints,
        "fault": fault,
    }
    PIN.write_text(json.dumps(pin, indent=1, sort_keys=True) + "\n")


def _pin():
    if CAPTURE and not PIN.exists():
        _capture(CAPTURE)
    return json.loads(PIN.read_text())


def test_the_pin_names_its_commit_and_covers_every_entry():
    """Fails if the pin lost its source commit or an entry, or holds a stale one."""
    pin = _pin()
    assert re.fullmatch(r"[0-9a-f]{40}", pin["source_commit"])
    assert len(pin["histories"]) == N_PINNED
    assert set(pin["histories"]) == expected_keys()
    assert set(pin["fingerprints"]) == expected_keys()
    assert set(pin["fault"]) == {
        f"{executor}/{w}" for executor in EXECUTORS for w in range(N_WORLDS)
    }


def test_the_manifest_equals_its_pin():
    """Fails if any manifest entry moved since the pin, even with a change record."""
    manifest = json.loads(MANIFEST.read_text())
    pinned = _pin()["histories"]
    moved = sorted(k for k in pinned if manifest.get(k) != pinned[k])
    assert not moved, moved
    assert set(manifest) == set(pinned)


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.name)
def test_fingerprints_match_their_pin(case, executor):
    """Fails if a decision or final view fingerprint of a corpus run moved."""
    pinned = _pin()["fingerprints"]
    got = {
        entry_key(case.name, executor, run.world): run_fingerprints(run)
        for run in runs(case.name, executor)
    }
    moved = {k: v for k, v in got.items() if pinned[k] != v}
    assert not moved, sorted(moved)


def test_the_pinned_fingerprints_carry_evidence():
    """Fails if a pinned run made no decision, or no run tells the audiences apart."""
    pinned = _pin()["fingerprints"].values()
    assert all(fp["n_decisions"] > 0 for fp in pinned)
    assert all(fp["policy"] != 0 and fp["observer"] != 0 for fp in pinned)
    assert any(fp["policy"] != fp["observer"] for fp in pinned)


@pytest.mark.parametrize("executor", EXECUTORS)
def test_the_fault_fixture_matches_its_pin(executor):
    """Fails if a fault's default behavior changed; the fixture must fail one row."""
    result = fault_run(executor)
    for w in range(N_WORLDS):
        ints = result.history(w).ints
        # evidence: two faults handled, the live one failed the first acquisition
        assert (ints["events"][:, 2] == EventKind.FAULT).sum() == 2
        assert (ints["status"] == RecordStatus.FAILED).sum() == 1
        assert (ints["status"] == RecordStatus.COMPLETED).sum() > 0
    pinned = _pin()["fault"]
    got = fault_digests(executor)
    assert got == {k: pinned[k] for k in got}


def test_a_lane_closing_fault_breaks_the_fault_pin():
    """Control: a fault that also closes its lane must move the pinned digests."""
    pinned = _pin()["fault"]
    got = fault_digests("host", close_lane=True)
    assert all(got[k]["ints"] != pinned[k]["ints"] for k in got)
    ints = fault_run("host", True).history(0).ints
    assert np.count_nonzero(ints["status"] == RecordStatus.COMPLETED) == 0
