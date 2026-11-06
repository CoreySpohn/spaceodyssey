"""Codec-owned files for large arrays: written by the dumper, checked by the loader.

An array with more elements than the inline limit is stored as its raw bytes (C
order, native byte order) in a file named by the sha256 of those bytes, in a
directory the caller names. The document keeps the dtype and shape and a reference,
`{"name": "<sha256>.bin", "sha256": "<hex>", "bytes": n}`, and the reference is part
of the document hash, so the content of every file is covered by it. A reference
names a file in the directory, never a path: a name with a separator is refused.
"""

import contextlib
import contextvars
import dataclasses
import hashlib
import os
import re
from pathlib import Path

from spaceodyssey.codec.document import DocumentError, pointer

DEFAULT_MAX_INLINE = 1024
_NAME = re.compile(r"[0-9a-f]{64}\.bin")
_REF_KEYS = frozenset({"name", "sha256", "bytes"})


@dataclasses.dataclass(frozen=True)
class FileStore:
    """A directory of array files and the size above which an array goes there."""

    directory: Path
    max_inline: int = DEFAULT_MAX_INLINE

    def wants(self, size):
        """Whether an array of `size` elements is written to a file."""
        return size > self.max_inline

    def put(self, raw):
        """Write `raw` under the name of its hash; return the document reference."""
        digest = hashlib.sha256(raw).hexdigest()
        name = f"{digest}.bin"
        target = self.directory / name
        if not (target.is_file() and target.stat().st_size == len(raw)):
            self.directory.mkdir(parents=True, exist_ok=True)
            tmp = self.directory / f".{name}.{os.getpid()}.tmp"
            tmp.write_bytes(raw)
            os.replace(tmp, target)
        return {"name": name, "sha256": digest, "bytes": len(raw)}

    def get(self, ref, path):
        """The bytes a reference names, refusing a bad name, a gap or a wrong hash."""
        if not isinstance(ref, dict) or set(ref) != _REF_KEYS:
            raise DocumentError(
                "a file reference is {name, sha256, bytes}", pointer(path)
            )
        name, digest, size = ref["name"], ref["sha256"], ref["bytes"]
        if type(name) is not str or not _NAME.fullmatch(name):
            raise DocumentError(
                "a file name is <sha256>.bin, with no path", pointer((*path, "name"))
            )
        if type(digest) is not str or name != f"{digest}.bin":
            raise DocumentError(
                "the file name does not match its sha256", pointer((*path, "sha256"))
            )
        if type(size) is not int or size < 0:
            raise DocumentError("bytes is a size", pointer((*path, "bytes")))
        target = self.directory / name
        try:
            raw = target.read_bytes()
        except OSError as err:
            raise DocumentError(
                f"array file {name} is missing from {self.directory} ({err.strerror})",
                pointer((*path, "name")),
            ) from err
        if len(raw) != size:
            raise DocumentError(
                f"array file {name} holds {len(raw)} bytes, the document says {size}",
                pointer((*path, "bytes")),
            )
        if hashlib.sha256(raw).hexdigest() != digest:
            raise DocumentError(
                f"array file {name} does not match its sha256",
                pointer((*path, "sha256")),
            )
        return raw


_STORE = contextvars.ContextVar("file_store", default=None)


@contextlib.contextmanager
def using_store(store):
    """Make `store` (or `None`) the file store of the dump or load that runs inside."""
    token = _STORE.set(store)
    try:
        yield
    finally:
        _STORE.reset(token)


def current_store():
    """The file store of the running dump or load, or `None`."""
    return _STORE.get()


def make_store(files, max_inline):
    """A `FileStore` for a directory argument, or `None` when there is none."""
    if files is None:
        return None
    if type(max_inline) is not int or max_inline < 0:
        raise ValueError("max_inline is a non-negative element count")
    return FileStore(Path(files), max_inline)
