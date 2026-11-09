"""The checkpoint directory: generations, the pointer, chunks and interrupted writes.

These tests run the store on small arrays with no study and no compiled program. A
write is stopped at each durable step by raising from the store's hook, which stands
in for a crash: the state on disk afterwards is what a killed process would leave.
"""

import json
import os

import numpy as np
import pytest

from spaceodyssey.codec.checkpoint import (
    CARRY_FILE,
    CURRENT,
    MANIFEST,
    GenerationStore,
)
from spaceodyssey.core.checkpoint import (
    CheckpointCode,
    CheckpointError,
    array_digest,
    concat_rows,
    differing_parts,
    identity_digest,
    rebuild,
    tree_digest,
)

OPS = (
    "payload",
    "manifest",
    "generation-sync",
    "pointer-write",
    "pointer-rename",
    "pointer-sync",
)


class Crash(BaseException):
    """Stands for a killed process: nothing in the store catches it."""


def stop_at(op):
    def hook(name):
        if name == op:
            raise Crash(op)

    return hook


def items(n):
    return [
        (".tick", np.asarray(n, np.int64)),
        (".balance", np.arange(6, dtype=np.float64).reshape(2, 3) + n),
    ]


def put(store, n, rows=2):
    ref = store.put_chunk(
        {"trace/.tick": np.arange(rows, dtype=np.int64) + 10 * n}, rows
    )
    return ref


def publish(store, n, chunks):
    return store.publish(
        {"boundary": "final", "steps": n}, {CARRY_FILE: items(n)}, chunks
    )


def test_a_published_generation_loads_back_exactly(tmp_path):
    store = GenerationStore(tmp_path / "run")
    assert store.load() is None
    ref = put(store, 1)
    assert publish(store, 1, [ref]) == 1
    got = store.load()
    assert got.generation == 1 and got.manifest["steps"] == 1
    for (path, a), loaded in zip(items(1), got.arrays[CARRY_FILE].items(), strict=True):
        assert loaded[0] == path and np.array_equal(loaded[1], a)
        assert loaded[1].dtype == a.dtype
    assert np.array_equal(got.chunks[0]["trace/.tick"], [10, 11])


def test_the_last_two_generations_are_kept_and_their_chunks(tmp_path):
    store = GenerationStore(tmp_path / "run")
    refs = []
    for n in range(1, 5):
        refs.append(put(store, n))
        publish(store, n, list(refs))
    root = tmp_path / "run"
    assert sorted(p.name for p in root.glob("gen-*")) == ["gen-000003", "gen-000004"]
    # the chunks of the oldest generation are still named by the kept ones
    assert len(list((root / "chunks").glob("*.npz"))) == 4
    assert store.load().generation == 4


def test_a_chunk_no_kept_generation_names_is_removed(tmp_path):
    store = GenerationStore(tmp_path / "run")
    first = put(store, 1)
    publish(store, 1, [first])
    second = put(store, 2)
    publish(store, 2, [second])
    third = put(store, 3)
    publish(store, 3, [second, third])
    names = {p.name for p in (tmp_path / "run" / "chunks").glob("*.npz")}
    assert first["name"] not in names and {second["name"], third["name"]} <= names


def test_an_identical_chunk_is_written_once(tmp_path):
    store = GenerationStore(tmp_path / "run")
    a, b = put(store, 1), put(store, 1)
    assert a["name"] == b["name"]
    assert len(list((tmp_path / "run" / "chunks").glob("*.npz"))) == 1


@pytest.mark.parametrize("op", OPS)
def test_a_crash_at_each_write_step_leaves_the_previous_generation(tmp_path, op):
    store = GenerationStore(tmp_path / "run")
    publish(store, 1, [put(store, 1)])
    store.hook = stop_at(op)
    with pytest.raises(Crash):
        publish(store, 2, [put(store, 2)])
    store.hook = None
    got = store.load()
    # the pointer is renamed only after "pointer-rename"; "pointer-sync" is after it
    assert got.generation == (2 if op == "pointer-sync" else 1)
    # the next save works and numbers past anything half written
    assert publish(store, 3, [put(store, 3)]) >= 2
    assert store.load().manifest["steps"] == 3


@pytest.mark.parametrize("op", OPS[:2])
def test_a_crash_before_the_manifest_in_the_first_save_leaves_an_empty_run(
    tmp_path, op
):
    store = GenerationStore(tmp_path / "run")
    store.hook = stop_at(op)
    with pytest.raises(Crash):
        publish(store, 1, [])
    store.hook = None
    assert store.load() is None


@pytest.mark.parametrize("op", OPS[2:5])
def test_a_crash_after_the_manifest_in_the_first_save_needs_the_recovery_flag(
    tmp_path, op
):
    # a manifest with no pointer cannot be told from a deleted pointer
    store = GenerationStore(tmp_path / "run")
    store.hook = stop_at(op)
    with pytest.raises(Crash):
        publish(store, 1, [])
    store.hook = None
    with pytest.raises(CheckpointError) as err:
        store.load()
    assert err.value.code is CheckpointCode.C005
    assert store.load(recover=True).generation == 1


def test_a_generation_without_a_manifest_is_never_loaded(tmp_path):
    store = GenerationStore(tmp_path / "run")
    publish(store, 1, [put(store, 1)])
    publish(store, 2, [put(store, 2)])
    (tmp_path / "run" / "gen-000002" / MANIFEST).unlink()
    with pytest.raises(CheckpointError) as err:
        store.load()
    assert err.value.code is CheckpointCode.C005
    got = GenerationStore(tmp_path / "run").load(recover=True)
    assert got.generation == 1
    assert any("generation" in f for f in got.findings)


@pytest.mark.parametrize("cut", [0, 1, 17, -1])
def test_a_truncated_payload_is_refused_not_loaded_in_part(tmp_path, cut):
    store = GenerationStore(tmp_path / "run")
    publish(store, 1, [put(store, 1)])
    publish(store, 2, [put(store, 2)])
    path = tmp_path / "run" / "gen-000002" / CARRY_FILE
    path.write_bytes(path.read_bytes()[:cut])
    with pytest.raises(CheckpointError) as err:
        store.load()
    assert err.value.code is CheckpointCode.C005 and "unusable" in err.value.message
    assert store.load(recover=True).generation == 1


def test_an_altered_payload_of_the_same_size_is_refused(tmp_path):
    store = GenerationStore(tmp_path / "run")
    publish(store, 1, [])
    path = tmp_path / "run" / "gen-000001" / CARRY_FILE
    raw = bytearray(path.read_bytes())
    raw[-30] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(CheckpointError):
        store.load()


def test_a_truncated_or_missing_chunk_is_refused(tmp_path):
    store = GenerationStore(tmp_path / "run")
    ref = put(store, 1)
    publish(store, 1, [ref])
    chunk = tmp_path / "run" / "chunks" / ref["name"]
    chunk.write_bytes(chunk.read_bytes()[:-5])
    with pytest.raises(CheckpointError) as err:
        store.load()
    assert ".npz" in err.value.message
    chunk.unlink()
    with pytest.raises(CheckpointError):
        store.load()


@pytest.mark.parametrize(
    "pointer",
    [
        b"",
        b"not json",
        b"[]",
        b'{"generation": 1}',
        b'{"generation": "1", "manifest_sha256": "a"}',
    ],
)
def test_a_corrupt_pointer_is_refused_and_the_recovery_flag_loads_the_newest(
    tmp_path, pointer
):
    store = GenerationStore(tmp_path / "run")
    publish(store, 1, [])
    publish(store, 2, [])
    (tmp_path / "run" / CURRENT).write_bytes(pointer)
    with pytest.raises(CheckpointError) as err:
        store.load()
    assert err.value.code is CheckpointCode.C005
    assert store.load(recover=True).generation == 2


def test_a_pointer_naming_a_missing_generation_is_refused(tmp_path):
    store = GenerationStore(tmp_path / "run")
    publish(store, 1, [])
    publish(store, 2, [])
    ptr = json.loads((tmp_path / "run" / CURRENT).read_text())
    ptr["generation"] = 9
    (tmp_path / "run" / CURRENT).write_text(json.dumps(ptr))
    with pytest.raises(CheckpointError):
        store.load()
    assert store.load(recover=True).generation == 2


def test_a_pointer_with_the_wrong_manifest_digest_is_refused(tmp_path):
    store = GenerationStore(tmp_path / "run")
    publish(store, 1, [])
    publish(store, 2, [])
    ptr = json.loads((tmp_path / "run" / CURRENT).read_text())
    ptr["manifest_sha256"] = "0" * 64
    (tmp_path / "run" / CURRENT).write_text(json.dumps(ptr))
    with pytest.raises(CheckpointError):
        store.load()
    assert store.load(recover=True).generation == 1


def test_a_missing_pointer_beside_generations_is_refused(tmp_path):
    store = GenerationStore(tmp_path / "run")
    publish(store, 1, [])
    (tmp_path / "run" / CURRENT).unlink()
    with pytest.raises(CheckpointError) as err:
        store.load()
    assert err.value.code is CheckpointCode.C005
    assert store.load(recover=True).generation == 1


def test_nothing_verifies_when_every_generation_is_damaged(tmp_path):
    store = GenerationStore(tmp_path / "run")
    publish(store, 1, [])
    (tmp_path / "run" / "gen-000001" / MANIFEST).write_bytes(b"{")
    (tmp_path / "run" / CURRENT).unlink()
    with pytest.raises(CheckpointError) as err:
        store.load(recover=True)
    assert err.value.findings


def test_a_symlinked_payload_or_pointer_is_refused(tmp_path):
    store = GenerationStore(tmp_path / "run")
    publish(store, 1, [])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_bytes((tmp_path / "run" / "gen-000001" / CARRY_FILE).read_bytes())
    link = tmp_path / "run" / "gen-000001" / CARRY_FILE
    link.unlink()
    os.symlink(elsewhere, link)
    with pytest.raises(CheckpointError):
        store.load()
    link.unlink()
    elsewhere.rename(link)
    pointer = tmp_path / "run" / CURRENT
    target = tmp_path / "pointer-copy"
    target.write_bytes(pointer.read_bytes())
    pointer.unlink()
    os.symlink(target, pointer)
    with pytest.raises(CheckpointError):
        store.load()


def test_a_symlinked_run_directory_is_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    os.symlink(real, tmp_path / "run")
    with pytest.raises(CheckpointError) as err:
        GenerationStore(tmp_path / "run").publish({}, {CARRY_FILE: items(1)}, [])
    assert err.value.code is CheckpointCode.C009


def test_the_keep_count_cannot_drop_the_previous_generation(tmp_path):
    with pytest.raises(ValueError):
        GenerationStore(tmp_path / "run", keep=1)


# ---- rebuilding on a template ----------------------------------------------------


TEMPLATE = {"a": np.zeros(3, np.int64), "b": np.zeros((2, 2), np.float64)}


def saved(**over):
    arrays = {"['a']": np.arange(3, dtype=np.int64), "['b']": np.ones((2, 2))}
    arrays.update(over)
    return arrays


def test_rebuild_returns_the_template_structure():
    out = rebuild(TEMPLATE, saved())
    assert set(out) == {"a", "b"} and out["a"].tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    "bad",
    [
        {"['a']": np.arange(3, dtype=np.int32)},
        {"['a']": np.arange(4, dtype=np.int64)},
        {"['b']": np.ones(4)},
    ],
)
def test_rebuild_refuses_a_saved_array_that_does_not_fit(bad):
    with pytest.raises(CheckpointError) as err:
        rebuild(TEMPLATE, saved(**bad))
    assert err.value.code is CheckpointCode.C007


def test_rebuild_refuses_a_missing_or_extra_leaf():
    for arrays in (
        {"['a']": np.zeros(3, np.int64)},
        {**saved(), "['c']": np.zeros(1)},
    ):
        with pytest.raises(CheckpointError):
            rebuild(TEMPLATE, arrays)


def test_rebuild_places_rows_on_the_row_axis():
    row = {"a": np.zeros((4,), np.int64)}
    host = rebuild(row, {"['a']": np.zeros((7, 4), np.int64)}, row_axis=0)
    assert host["a"].shape == (7, 4)
    batched = {"a": np.zeros((3, 4), np.int64)}
    out = rebuild(batched, {"['a']": np.zeros((3, 9, 4), np.int64)}, row_axis=1)
    assert out["a"].shape == (3, 9, 4)
    with pytest.raises(CheckpointError):
        rebuild(batched, {"['a']": np.zeros((9, 3, 4), np.int64)}, row_axis=1)


def test_digests_see_dtype_shape_and_value():
    a = np.arange(6, dtype=np.int64)
    assert array_digest(a) == array_digest(a.copy())
    assert (
        len(
            {
                array_digest(a),
                array_digest(a.astype(np.int32)),
                array_digest(a.reshape(2, 3)),
                array_digest(a + 1),
            }
        )
        == 4
    )
    assert tree_digest({"x": a}) != tree_digest({"y": a})


def test_identity_names_what_differs():
    base = {"study": "s", "roots": "r", "truth": "t"}
    assert identity_digest(base) == identity_digest(dict(reversed(base.items())))
    assert identity_digest(base) != identity_digest({**base, "roots": "r2"})
    assert differing_parts(base, {**base, "roots": "r2", "extra": 1}) == [
        "extra",
        "roots",
    ]


def test_rows_join_along_the_row_axis():
    a, b = {"x": np.ones((2, 3))}, {"x": np.zeros((2, 4))}
    assert concat_rows(a, b, 1)["x"].shape == (2, 7)
    assert concat_rows(None, b, 1) is b and concat_rows(a, None, 1) is a
