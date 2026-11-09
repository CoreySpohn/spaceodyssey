"""Checkpoint directories: immutable generations, an atomic pointer, trace chunks.

A run directory holds

- `gen-<n>/`: one immutable generation, its payload files and a `manifest.json`;
- `chunks/<sha256>.npz`: append-only trace chunks named by the digest of their content;
- `CURRENT`: the pointer, the generation number and the digest of its manifest.

Publishing a generation writes and syncs every payload file, then the manifest, then
the generation directory, and only then replaces `CURRENT` (write to a temporary name,
sync, rename, sync the directory). A crash at any step leaves `CURRENT` naming an intact
earlier generation; a generation directory with no verified manifest is ignored and
reported. The last two generations are kept, and a chunk is kept while a kept
generation names it.

Everything is opened relative to a directory descriptor without following a link, and a
file is read only when it is a regular file of the size the manifest says. A manifest
carries a digest of every array and of every file, so a truncated or altered payload is
refused, never loaded in part.
"""

import contextlib
import dataclasses
import hashlib
import io
import json
import os
import platform
import re
import secrets
import stat
from importlib import metadata

import jax
import numpy as np

from spaceodyssey.codec.document import SCHEMA_VERSION, DocumentError
from spaceodyssey.codec.dump import dump_study
from spaceodyssey.codec.files import DEFAULT_MAX_BYTES, FileStore, UnsafeDirectory
from spaceodyssey.core.checkpoint import (
    BOUNDARIES,
    FORMAT,
    FORMAT_VERSION,
    CheckpointCode,
    CheckpointError,
    CheckpointSink,
    Restored,
    array_digest,
    canonical_json,
    concat_rows,
    differing_parts,
    identity_digest,
    named_leaves,
    rebuild,
    tree_digest,
)

CURRENT = "CURRENT"
MANIFEST = "manifest.json"
CARRY_FILE = "carry.npz"
CHUNKS = "chunks"
_GEN = re.compile(r"gen-(\d{6})")
_CHUNK = re.compile(r"[0-9a-f]{64}\.npz")
_FILE = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _unsafe(message):
    return CheckpointError(CheckpointCode.C009, message)


def _gen_name(n):
    return f"gen-{n:06d}"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def content_digest(arrays):
    """Digest of a `{name: array}` dict by names and array digests, not file bytes."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        h.update(f"{name}={array_digest(arrays[name])};".encode("ascii"))
    return h.hexdigest()


def _npz_bytes(arrays):
    """Uncompressed `.npz` bytes of arrays under index keys, with their records."""
    keys = {f"a{i:05d}": (name, np.asarray(a)) for i, (name, a) in enumerate(arrays)}
    buf = io.BytesIO()
    np.savez(buf, **{k: a for k, (_, a) in keys.items()})
    records = [
        {
            "path": name,
            "dtype": a.dtype.str,
            "shape": list(a.shape),
            "sha256": array_digest(a),
        }
        for name, a in keys.values()
    ]
    return buf.getvalue(), records


def _load_npz(raw, records, where):
    """The arrays of an `.npz` checked against their records, as `{path: array}`."""
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            if sorted(z.files) != [f"a{i:05d}" for i in range(len(records))]:
                raise ValueError("array keys differ from the manifest")
            arrays = {}
            for i, rec in enumerate(records):
                a = z[f"a{i:05d}"]
                if (
                    a.dtype.str != rec["dtype"]
                    or list(a.shape) != rec["shape"]
                    or array_digest(a) != rec["sha256"]
                ):
                    raise ValueError(f"array {rec['path']} does not match its digest")
                arrays[rec["path"]] = a
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(
            CheckpointCode.C004, f"{where} cannot be loaded: {exc}"
        ) from exc
    return arrays


def _open_read(fd, name, *, size=None, max_bytes=DEFAULT_MAX_BYTES):
    """The bytes of a regular file below `fd`, never through a link or special file."""
    try:
        f = os.open(name, _FILE, dir_fd=fd)
    except FileNotFoundError:
        raise CheckpointError(CheckpointCode.C004, f"{name} is missing") from None
    except OSError as exc:
        raise _unsafe(f"{name} cannot be opened safely: {exc}") from exc
    try:
        info = os.fstat(f)
        if not stat.S_ISREG(info.st_mode):
            raise _unsafe(f"{name} is not a regular file")
        if info.st_size > max_bytes or (size is not None and info.st_size != size):
            raise CheckpointError(
                CheckpointCode.C004,
                f"{name} has {info.st_size} bytes, expected "
                f"{size if size is not None else 'at most ' + str(max_bytes)}",
            )
        parts = []
        got = 0
        while got <= info.st_size:
            block = os.read(f, min(1 << 20, info.st_size - got + 1))
            if not block:
                break
            parts.append(block)
            got += len(block)
        raw = b"".join(parts)
        if len(raw) != info.st_size or os.fstat(f).st_size != info.st_size:
            raise CheckpointError(CheckpointCode.C004, f"{name} changed while read")
        return raw
    finally:
        os.close(f)


def _write_new(fd, name, raw):
    """Create `name` exclusively below `fd`, write, sync, close."""
    f = os.open(
        name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd
    )
    try:
        view = memoryview(raw)
        while view:
            view = view[os.write(f, view) :]
        os.fsync(f)
    finally:
        os.close(f)


def _rmtree(fd, name):
    """Remove a generation directory and its files below `fd`."""
    try:
        sub = os.open(name, _DIR, dir_fd=fd)
    except OSError:
        return
    try:
        for entry in os.listdir(sub):
            with contextlib.suppress(OSError):
                os.unlink(entry, dir_fd=sub)
    finally:
        os.close(sub)
    with contextlib.suppress(OSError):
        os.rmdir(name, dir_fd=fd)


@dataclasses.dataclass
class Loaded:
    """A verified generation: its manifest, payload arrays and trace chunks."""

    manifest: dict
    arrays: dict  # file name -> {path: array}
    chunks: list  # per chunk, {path: array}
    generation: int
    findings: list


class GenerationStore:
    """The directory of generations, with its crash-safe publication.

    `hook`, when given, is called as `hook(op)` before each durable step (`payload`,
    `manifest`, `generation-sync`, `pointer-write`, `pointer-rename`, `pointer-sync`,
    `chunk`), so a test can stop the write at any point.
    """

    def __init__(self, directory, *, keep=2, max_bytes=DEFAULT_MAX_BYTES, hook=None):
        """Open (without creating) the run directory `directory`."""
        if keep < 2:
            raise ValueError("keep at least the last two generations")
        self.directory = directory
        self.keep = keep
        self.max_bytes = max_bytes
        self.hook = hook
        self._files = FileStore(directory)

    def _at(self, op):
        if self.hook is not None:
            self.hook(op)

    def _root(self, create=False):
        try:
            return self._files._open(create=create)
        except UnsafeDirectory as exc:
            raise _unsafe(str(exc)) from exc
        except FileNotFoundError:
            return None

    def _generations(self, fd):
        found = []
        for name in os.listdir(fd):
            m = _GEN.fullmatch(name)
            if m:
                found.append(int(m.group(1)))
        return sorted(found)

    def _published(self, root, gens):
        """Whether any generation directory holds a manifest."""
        for g in gens:
            try:
                gfd = os.open(_gen_name(g), _DIR, dir_fd=root)
            except OSError:
                continue
            try:
                os.stat(MANIFEST, dir_fd=gfd, follow_symlinks=False)
                return True
            except OSError:
                continue
            finally:
                os.close(gfd)
        return False

    def _chunks_fd(self, fd, create):
        flags = _DIR
        try:
            return os.open(CHUNKS, flags, dir_fd=fd)
        except FileNotFoundError:
            if not create:
                return None
            os.mkdir(CHUNKS, dir_fd=fd)
            os.fsync(fd)
            return os.open(CHUNKS, flags, dir_fd=fd)
        except OSError as exc:
            raise _unsafe(f"{CHUNKS} cannot be opened safely: {exc}") from exc

    # ---- chunks -----------------------------------------------------------------

    def put_chunk(self, arrays, rows):
        """Write a trace chunk once, under the digest of its content; return its ref."""
        items = sorted(arrays.items())
        digest = content_digest(arrays)
        name = f"{digest}.npz"
        raw, records = _npz_bytes(items)
        root = self._root(create=True)
        try:
            cfd = self._chunks_fd(root, create=True)
            try:
                with contextlib.suppress(CheckpointError):
                    have = _open_read(cfd, name, max_bytes=self.max_bytes)
                    _load_chunk(have, digest, records)  # an intact chunk is kept
                    return _chunk_ref(name, digest, have, rows, records)
                self._at("chunk")
                tmp = f".{name}.{secrets.token_hex(8)}.tmp"
                _write_new(cfd, tmp, raw)
                os.rename(tmp, name, src_dir_fd=cfd, dst_dir_fd=cfd)
                os.fsync(cfd)
            finally:
                os.close(cfd)
        finally:
            os.close(root)
        return _chunk_ref(name, digest, raw, rows, records)

    # ---- publication ------------------------------------------------------------

    def publish(self, meta, payloads, chunk_refs):
        """Write a new generation and make it current; return its number.

        `payloads` maps a file name to a list of `(path, array)`; `meta` is merged into
        the manifest, which also lists the files, the chunks and their digests.
        """
        root = self._root(create=True)
        try:
            previous = self._generations(root)
            n = (previous[-1] + 1) if previous else 1
            self._at("payload")
            os.mkdir(_gen_name(n), dir_fd=root)
            gfd = os.open(_gen_name(n), _DIR, dir_fd=root)
            try:
                files = {}
                for name, items in payloads.items():
                    raw, records = _npz_bytes(items)
                    _write_new(gfd, name, raw)
                    files[name] = {
                        "size": len(raw),
                        "sha256": _sha(raw),
                        "arrays": records,
                    }
                body = {
                    **meta,
                    "format": FORMAT,
                    "format_version": FORMAT_VERSION,
                    "generation": n,
                    "files": files,
                    "chunks": list(chunk_refs),
                }
                raw = canonical_json(body).encode("ascii")
                self._at("manifest")
                _write_new(gfd, MANIFEST, raw)
                self._at("generation-sync")
                os.fsync(gfd)
            finally:
                os.close(gfd)
            os.fsync(root)
            pointer = canonical_json(
                {"generation": n, "manifest_sha256": _sha(raw)}
            ).encode("ascii")
            self._at("pointer-write")
            tmp = f".{CURRENT}.{secrets.token_hex(8)}.tmp"
            _write_new(root, tmp, pointer)
            self._at("pointer-rename")
            os.rename(tmp, CURRENT, src_dir_fd=root, dst_dir_fd=root)
            self._at("pointer-sync")
            os.fsync(root)
            self._prune(root, n)
        finally:
            os.close(root)
        return n

    def _prune(self, root, current):
        """Keep the newest `keep` generations up to `current`, and their chunks."""
        gens = [g for g in self._generations(root) if g <= current]
        keep = set(gens[-self.keep :])
        for g in self._generations(root):
            if g not in keep:
                _rmtree(root, _gen_name(g))
        named = set()
        for g in keep:
            with contextlib.suppress(CheckpointError, ValueError, KeyError):
                gfd = os.open(_gen_name(g), _DIR, dir_fd=root)
                try:
                    body = json.loads(
                        _open_read(gfd, MANIFEST, max_bytes=self.max_bytes)
                    )
                finally:
                    os.close(gfd)
                named.update(c["name"] for c in body["chunks"])
        cfd = self._chunks_fd(root, create=False)
        if cfd is None:
            return
        try:
            for name in os.listdir(cfd):
                if name not in named and (
                    _CHUNK.fullmatch(name) or name.endswith(".tmp")
                ):
                    with contextlib.suppress(OSError):
                        os.unlink(name, dir_fd=cfd)
        finally:
            os.close(cfd)

    # ---- loading ----------------------------------------------------------------

    def load(self, *, recover=False):
        """The current generation, verified, or `None` for an empty directory.

        A pointer that names a missing or unusable generation is refused, unless
        `recover` selects the newest generation that verifies; what was skipped is in
        `Loaded.findings`.
        """
        root = self._root(create=False)
        if root is None:
            return None
        try:
            gens = self._generations(root)
            try:
                raw = _open_read(root, CURRENT, max_bytes=1 << 16)
            except CheckpointError as exc:
                if exc.code is CheckpointCode.C004 and not self._published(root, gens):
                    # nothing was ever committed: a first save that never finished
                    return None
                if not recover:
                    raise CheckpointError(
                        CheckpointCode.C005, f"the pointer is unusable: {exc.message}"
                    ) from exc
                return self._newest(root, gens, [exc.message])
            try:
                ptr = json.loads(raw)
                if (
                    set(ptr) != {"generation", "manifest_sha256"}
                    or type(ptr["generation"]) is not int
                    or type(ptr["manifest_sha256"]) is not str
                ):
                    raise ValueError("pointer fields")
            except (ValueError, TypeError) as exc:
                if not recover:
                    raise CheckpointError(
                        CheckpointCode.C005, "the pointer is malformed"
                    ) from exc
                return self._newest(root, gens, ["the pointer is malformed"])
            try:
                return self._verify(root, ptr["generation"], ptr["manifest_sha256"], [])
            except CheckpointError as exc:
                if not recover:
                    raise CheckpointError(
                        CheckpointCode.C005,
                        f"the pointer names generation {ptr['generation']}, "
                        f"which is unusable: {exc.message}",
                    ) from exc
                return self._newest(
                    root, [g for g in gens if g != ptr["generation"]], [exc.message]
                )
        finally:
            os.close(root)

    def _newest(self, root, gens, findings):
        for g in sorted(gens, reverse=True):
            try:
                return self._verify(root, g, None, findings)
            except CheckpointError as exc:
                findings = [*findings, f"generation {g}: {exc.message}"]
        raise CheckpointError(
            CheckpointCode.C005, "no generation verifies", findings=findings
        )

    def _verify(self, root, g, manifest_sha, findings):
        try:
            gfd = os.open(_gen_name(g), _DIR, dir_fd=root)
        except OSError as exc:
            raise CheckpointError(
                CheckpointCode.C006, f"generation {g} is missing"
            ) from exc
        try:
            try:
                raw = _open_read(gfd, MANIFEST, max_bytes=self.max_bytes)
            except CheckpointError as exc:
                raise CheckpointError(
                    CheckpointCode.C006, f"generation {g} has no manifest"
                ) from exc
            if manifest_sha is not None and _sha(raw) != manifest_sha:
                raise CheckpointError(
                    CheckpointCode.C004,
                    f"the manifest of generation {g} does not match",
                )
            try:
                body = json.loads(raw)
                if body["format"] != FORMAT or body["generation"] != g:
                    raise ValueError("format or generation")
                files, chunks = body["files"], body["chunks"]
            except (ValueError, KeyError, TypeError) as exc:
                raise CheckpointError(
                    CheckpointCode.C006, f"the manifest of generation {g} is malformed"
                ) from exc
            arrays = {}
            for name, rec in files.items():
                data = _open_read(gfd, name, size=rec["size"], max_bytes=self.max_bytes)
                if _sha(data) != rec["sha256"]:
                    raise CheckpointError(
                        CheckpointCode.C004, f"{name} of generation {g} does not match"
                    )
                arrays[name] = _load_npz(
                    data, rec["arrays"], f"{name} of generation {g}"
                )
        finally:
            os.close(gfd)
        cfd = self._chunks_fd(root, create=False)
        loaded = []
        try:
            for ref in chunks:
                if cfd is None:
                    raise CheckpointError(CheckpointCode.C004, "the chunks are missing")
                data = _open_read(
                    cfd, ref["name"], size=ref["size"], max_bytes=self.max_bytes
                )
                if _sha(data) != ref["file_sha256"]:
                    raise CheckpointError(
                        CheckpointCode.C004, f"chunk {ref['name']} does not match"
                    )
                loaded.append(_load_chunk(data, ref["sha256"], ref["arrays"]))
        finally:
            if cfd is not None:
                os.close(cfd)
        return Loaded(body, arrays, loaded, g, list(findings))


def _chunk_ref(name, digest, raw, rows, records):
    return {
        "name": name,
        "sha256": digest,
        "file_sha256": _sha(raw),
        "size": len(raw),
        "rows": rows,
        "arrays": records,
    }


def _load_chunk(raw, digest, records):
    """The arrays of a chunk file, checked against the digest of its content."""
    arrays = _load_npz(raw, records, "a chunk")
    if content_digest(arrays) != digest:
        raise CheckpointError(CheckpointCode.C004, "a chunk does not match its digest")
    return arrays


# ---- identity --------------------------------------------------------------------


def dependency_fingerprint():
    """What numerical results depend on: library versions, precision and backend."""

    def version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "unknown"

    return {
        "jax": version("jax"),
        "jaxlib": version("jaxlib"),
        "numpy": version("numpy"),
        "spaceodyssey": version("spaceodyssey"),
        "python": ".".join(platform.python_version_tuple()[:2]),
        "x64": bool(jax.config.jax_enable_x64),
        "backend": jax.default_backend(),
    }


def study_digest(study, kinds=None):
    """Hex sha256 of the study's canonical document bytes, or `CheckpointCode.C001`."""
    try:
        return _sha(dump_study(study, kinds).canonical_bytes())
    except (DocumentError, TypeError, ValueError) as exc:
        raise CheckpointError(
            CheckpointCode.C001, f"the study cannot be described: {exc}"
        ) from exc


def identity_parts(study, truths, roots, config, kinds=None):
    """The named parts of a run's identity (everything but the dependencies)."""
    return {
        "study": study_digest(study, kinds),
        "roots": array_digest(roots),
        "truth": tree_digest(truths),
        "key_version": int(study.key_version),
        "codec": {"format": FORMAT_VERSION, "schema": SCHEMA_VERSION},
        "config": config,
    }


# ---- the sink --------------------------------------------------------------------


def _stack_rows(prefix, rows):
    """`{prefix/path: array}` of a list of same-structure pytrees stacked on axis 0."""
    out = {}
    if not rows:
        return out
    per_row = [dict(named_leaves(r)) for r in rows]
    for name in per_row[0]:
        out[f"{prefix}/{name}"] = np.stack([r[name] for r in per_row])
    return out


def _unstack_rows(prefix, arrays, template):
    """The rows saved by `_stack_rows`, rebuilt on `template`, as a tuple of pytrees."""
    mine = {
        k[len(prefix) + 1 :]: v for k, v in arrays.items() if k.startswith(prefix + "/")
    }
    if not mine:
        return ()
    n = len(next(iter(mine.values())))
    return tuple(
        rebuild(template, {k: v[i] for k, v in mine.items()}) for i in range(n)
    )


class DirectoryCheckpoint(CheckpointSink):
    """Save to and resume from a run directory.

    `kinds` is the `KindResolver` or mapping the study needs to be described (as for
    `dump_study`). `recover` loads the newest generation that verifies when the
    pointer is unusable. `allow_dependency_change` accepts a changed dependency
    fingerprint; the override and both fingerprints are recorded in every later
    manifest. `boundaries` lists where to save.
    """

    def __init__(
        self,
        directory,
        *,
        kinds=None,
        keep=2,
        recover=False,
        allow_dependency_change=False,
        boundaries=BOUNDARIES,
        store=None,
    ):
        """Name the run directory and how to treat what is already in it."""
        self.store = store or GenerationStore(directory, keep=keep)
        self.kinds = kinds
        self.recover = recover
        self.allow_dependency_change = allow_dependency_change
        self.boundaries = tuple(boundaries)
        self._parts = None
        self._deps = None
        self._overrides = []
        self._chunks = []
        self._rows = 0
        self._row_axis = 0

    def wants(self, boundary):
        """Whether `boundary` is one this sink saves at."""
        return boundary in self.boundaries

    def begin(self, study, truths, roots, *, config, templates, resume):
        """Check the identity against the directory and restore its newest state."""
        parts = identity_parts(study, truths, roots, config, self.kinds)
        deps = dependency_fingerprint()
        self._parts, self._deps = parts, deps
        self._row_axis = config["row_axis"]
        loaded = self.store.load(recover=self.recover)
        if loaded is None:
            return None
        if not resume:
            raise CheckpointError(
                CheckpointCode.C008,
                "the directory holds a checkpoint; resume it or use another directory",
            )
        saved = loaded.manifest["identity"]
        diff = differing_parts(saved["parts"], parts)
        if diff:
            raise CheckpointError(
                CheckpointCode.C002,
                f"the run identity differs from the checkpoint in: {', '.join(diff)}",
            )
        self._overrides = list(loaded.manifest.get("overrides", []))
        if saved["dependencies"] != deps:
            if not self.allow_dependency_change:
                diff = differing_parts(saved["dependencies"], deps)
                raise CheckpointError(
                    CheckpointCode.C003,
                    f"the dependencies differ from the saved: {', '.join(diff)}",
                )
            self._overrides.append(
                {"dependencies": {"saved": saved["dependencies"], "now": deps}}
            )
        carry = rebuild(templates["carry"], dict(loaded.arrays[CARRY_FILE]))
        if tree_digest(carry.log) != loaded.manifest["records_sha256"]:
            raise CheckpointError(
                CheckpointCode.C004, "the record log does not match its digest"
            )
        trace = None
        view_rows, effect_rows = [], []
        for arrays in loaded.chunks:
            ours = {k[6:]: v for k, v in arrays.items() if k.startswith("trace/")}
            if ours:
                trace = concat_rows(
                    trace,
                    rebuild(templates["trace"], ours, row_axis=self._row_axis),
                    self._row_axis,
                )
            if "view" in templates:
                view_rows.extend(_unstack_rows("view", arrays, templates["view"]))
            if "effects" in templates:
                effect_rows.extend(
                    _unstack_rows("effects", arrays, templates["effects"])
                )
        rows = (
            0
            if trace is None
            else int(jax.tree_util.tree_leaves(trace)[0].shape[self._row_axis])
        )
        if rows != loaded.manifest["rows"]:
            raise CheckpointError(
                CheckpointCode.C004,
                "the trace chunks do not hold the rows the manifest names",
            )
        self._chunks = list(loaded.manifest["chunks"])
        self._rows = rows
        return Restored(
            carry,
            trace,
            int(loaded.manifest["steps"]),
            loaded.manifest["boundary"],
            loaded.generation,
            tuple(view_rows),
            tuple(effect_rows),
        )

    def save(self, boundary, carry, trace, steps, *, view_rows=(), effect_rows=()):
        """Publish a generation of `carry` and the trace rows since the last save."""
        arrays = {}
        n_new = 0
        if trace is not None:
            leaves = named_leaves(trace)
            n_new = int(leaves[0][1].shape[self._row_axis]) if leaves else 0
            if n_new:
                arrays.update({f"trace/{name}": a for name, a in leaves})
        arrays.update(_stack_rows("view", list(view_rows)))
        arrays.update(_stack_rows("effects", list(effect_rows)))
        chunks = list(self._chunks)
        if arrays:
            chunks.append(self.store.put_chunk(arrays, n_new))
        meta = {
            "boundary": boundary,
            "steps": int(steps),
            "rows": self._rows + n_new,
            "records_sha256": tree_digest(carry.log),
            "identity": {
                "digest": identity_digest(self._parts),
                "parts": self._parts,
                "dependencies": self._deps,
            },
            "overrides": self._overrides,
        }
        payload = {CARRY_FILE: named_leaves(carry)}
        n = self.store.publish(meta, payload, chunks)
        self._chunks, self._rows = chunks, self._rows + n_new
        return n
