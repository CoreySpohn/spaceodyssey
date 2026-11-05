"""Golden content hashes of every conformance corpus history, on both executors.

`golden_histories.json` maps `case/executor/world` to the SHA-256 digests
(`spaceodyssey.inputs.digest`) of the integer and floating-point parts of that world's
canonical history (`canonical_history`, as the shared corpus run store records it).
Any change to any history, by one bit, fails here. A deliberate change regenerates
the manifest with `SPACEODYSSEY_WRITE_GOLDEN=1` and names every moved entry in a
change record under `docs/changes/`.
"""

import json
import os
from pathlib import Path

import pytest

from spaceodyssey.core.conformance import EXECUTORS
from spaceodyssey.inputs import digest
from tests.core.conformance_helpers import N_WORLDS, runs
from tests.core.corpus import CORPUS

MANIFEST = Path(__file__).with_name("golden_histories.json")
WRITE = os.environ.get("SPACEODYSSEY_WRITE_GOLDEN", "0") == "1"


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
