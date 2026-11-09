"""Checkpoints: what an executor saves at a boundary and what a resume needs back.

The executors never touch a file. They hand a `CheckpointSink` the carry, the trace rows
produced since the last save and the step count at a boundary, and ask it, once, for
the state to continue from. `spaceodyssey.codec.checkpoint` holds the sink that writes
generations to a directory; this module holds what both sides agree on: the boundary
names, the digests, the run identity and the errors.

A carry and a trace row are pytrees of arrays. Each is saved as a flat list of named
arrays (the tree path of every leaf) and rebuilt on a template the executor builds
from the study and the truth, so the structure of a restored carry always comes from
the code, never from the file: a file can only supply array contents, and each array is
checked against the template's dtype and trailing shape before it is used.
"""

import dataclasses
import enum
import hashlib
import json

import jax
import numpy as np

FORMAT = "spaceodyssey-checkpoint"
FORMAT_VERSION = 1

# Boundaries an executor saves at. After acquisition exists only on the host executor
# (a compiled scan has no per-event boundary); the others exist on both.
AFTER_ACQUISITION = "acquisition"
AFTER_INFERENCE = "inference"
AT_HOLD = "hold"  # the step budget ended with a refresh checkpoint held
AT_CHUNK = "chunk"  # a compiled chunk ended
FINAL = "final"
BOUNDARIES = (AFTER_ACQUISITION, AFTER_INFERENCE, AT_HOLD, AT_CHUNK, FINAL)


class CheckpointCode(enum.StrEnum):
    """Findings a checkpoint refuses or reports with."""

    C001 = "C001"  # the study cannot be described, so its identity cannot be taken
    C002 = "C002"  # the run identity differs from the one saved
    C003 = "C003"  # the dependency fingerprint differs from the one saved
    C004 = "C004"  # a payload or chunk is missing, truncated or does not match
    C005 = "C005"  # the pointer is missing, malformed or names an unusable generation
    C006 = "C006"  # a generation has no verified manifest
    C007 = "C007"  # a saved array does not fit the structure the study builds
    C008 = "C008"  # a checkpoint exists and the call did not ask to resume
    C009 = "C009"  # the directory or a file in it is not safe to use


class CheckpointError(Exception):
    """A checkpoint was refused: `code` is a `CheckpointCode`, `findings` more text."""

    def __init__(self, code, message, findings=()):
        """Keep the code, the message and any further findings."""
        super().__init__(f"{code}: {message}")
        self.code = CheckpointCode(code)
        self.message = message
        self.findings = tuple(findings)


@dataclasses.dataclass
class Restored:
    """The state a resume continues from.

    `carry` has the structure of the carry the executor builds (batched on a leading
    world axis for the compiled executor). `trace` holds the rows already produced,
    stacked on the row axis, or `None` when there are none. `view_rows` and
    `effect_rows` are the host loop's optional per-decision logs.
    """

    carry: object
    trace: object
    steps: int
    boundary: str
    generation: int
    view_rows: tuple = ()
    effect_rows: tuple = ()


class CheckpointSink:
    """Where an executor saves and from where it resumes."""

    def begin(self, study, truths, roots, *, config, templates, resume):
        """Open the run: check its identity and return the `Restored` state or `None`.

        With `resume` false the call must start a new run and a directory that already
        holds a checkpoint is refused. `config` names what the run does that its
        arguments do not (the executor, the chunk size, the bound boundary and
        service); `templates` maps `carry`, `trace`, and when asked `view` and
        `effects` to a pytree with the structure to rebuild.
        """
        raise NotImplementedError

    def wants(self, boundary):
        """Whether to save at `boundary` (a name in `BOUNDARIES`)."""
        return True

    def save(self, boundary, carry, trace, steps, *, view_rows=(), effect_rows=()):
        """Publish a generation of the carry and the trace rows since the last save."""
        raise NotImplementedError


# ---- digests ---------------------------------------------------------------------


def array_digest(a):
    """Hex sha256 of an array's dtype, shape and C-order bytes."""
    a = np.asarray(a)
    if a.dtype.hasobject:
        raise ValueError("an object array has no digest")
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{a.shape}|".encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def named_leaves(tree):
    """`[(path, array)]` of every array leaf of `tree`, in flatten order.

    A path is the tree path as text; a leaf that is not an array (and so cannot be
    saved) is a `TypeError`.
    """
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    out = []
    for path, leaf in flat:
        name = jax.tree_util.keystr(path)
        try:
            arr = np.asarray(leaf)
        except Exception as exc:
            raise TypeError(f"leaf {name} is not an array") from exc
        if arr.dtype.hasobject:
            raise TypeError(f"leaf {name} is an object array")
        out.append((name, arr))
    return out


def tree_digest(tree):
    """Hex sha256 over the paths and array digests of a pytree."""
    h = hashlib.sha256()
    for name, arr in named_leaves(tree):
        h.update(f"{name}={array_digest(arr)};".encode("ascii"))
    return h.hexdigest()


def rebuild(template, arrays, *, row_axis=None):
    """A tree with `template`'s structure and the saved `arrays` (`{path: array}`).

    Each saved array must carry the template leaf's path, dtype and trailing shape. With
    `row_axis` (0 or 1) the template is one trace row, with the world axis in front of
    it for a batched run: the saved array has one more axis, of any length, at
    `row_axis`. Anything else is `CheckpointCode.C007`.
    """
    flat, treedef = jax.tree_util.tree_flatten_with_path(template)
    names = [jax.tree_util.keystr(p) for p, _ in flat]
    if set(arrays) != set(names):
        extra = sorted(set(arrays) ^ set(names))[:3]
        raise CheckpointError(
            CheckpointCode.C007, f"the saved leaves do not match the structure: {extra}"
        )
    leaves = []
    for (_, want), name in zip(flat, names, strict=True):
        got = np.asarray(arrays[name])
        want = np.asarray(want)
        if got.dtype != want.dtype:
            raise CheckpointError(
                CheckpointCode.C007,
                f"leaf {name} is {got.dtype}, the study builds {want.dtype}",
            )
        shape = want.shape
        if row_axis is None:
            ok = got.shape == shape
        else:
            ok = got.ndim == want.ndim + 1 and (
                got.shape[:row_axis] + got.shape[row_axis + 1 :] == want.shape
            )
        if not ok:
            raise CheckpointError(
                CheckpointCode.C007,
                f"leaf {name} has shape {got.shape}, the study builds {shape}",
            )
        leaves.append(got)
    return jax.tree_util.tree_unflatten(treedef, leaves)


# ---- run identity ----------------------------------------------------------------


def canonical_json(value):
    """Canonical JSON text: sorted keys, no whitespace, ASCII, finite numbers only."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def identity_digest(parts):
    """The run identity: a sha256 over the canonical form of its named parts."""
    return hashlib.sha256(canonical_json(parts).encode("ascii")).hexdigest()


def differing_parts(saved, now):
    """Names of the identity parts that differ between two dicts of parts."""
    return sorted(k for k in set(saved) | set(now) if saved.get(k) != now.get(k))


def callable_name(obj):
    """A stable name for a function or class, for the run configuration."""
    return f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', repr(obj))}"


def binding_record(boundary):
    """The configuration a refit boundary adds to a run identity: its `binding()`."""
    out = dict(boundary.binding())
    out["model_hash"] = int(out["model_hash"])
    return {"class": callable_name(type(boundary)), **out}


def concat_rows(prefix, rows, axis):
    """Join two stacked trace pytrees along the row axis; either may be `None`."""
    if prefix is None:
        return rows
    if rows is None:
        return prefix
    return jax.tree_util.tree_map(
        lambda a, b: np.concatenate([np.asarray(a), np.asarray(b)], axis=axis),
        prefix,
        rows,
    )
