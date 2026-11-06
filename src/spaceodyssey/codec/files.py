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
import stat
import tempfile
from pathlib import Path

from spaceodyssey.codec.document import DocumentError, pointer

DEFAULT_MAX_INLINE = 1024
DEFAULT_MAX_BYTES = 1 << 30
_NAME = re.compile(r"[0-9a-f]{64}\.bin")
_REF_KEYS = frozenset({"name", "sha256", "bytes"})


@dataclasses.dataclass(frozen=True)
class FileStore:
    """A directory of array files and the size above which an array goes there.

    Reads never follow a symlink and accept only regular files, and never read more
    than `max_bytes` (default `DEFAULT_MAX_BYTES`) or more than the size the
    document's dtype and shape imply. Writes go through an exclusively created
    temporary file with an unpredictable name and are moved into place atomically.
    """

    directory: Path
    max_inline: int = DEFAULT_MAX_INLINE
    max_bytes: int = DEFAULT_MAX_BYTES

    def wants(self, size):
        """Whether an array of `size` elements is written to a file."""
        return size > self.max_inline

    def put(self, raw):
        """Write `raw` under the name of its hash; return the document reference.

        A file already there is kept only when it is a regular file whose bytes hash
        to the name; anything else (a wrong size, a same-length corruption, a
        symlink, a special file) is replaced atomically, and a replaced symlink
        loses only the link, never its target.
        """
        digest = hashlib.sha256(raw).hexdigest()
        name = f"{digest}.bin"
        ref = {"name": name, "sha256": digest, "bytes": len(raw)}
        if self._intact(name, digest, len(raw)):
            return ref
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
            os.replace(tmp, self.directory / name)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        return ref

    def _intact(self, name, digest, size):
        """Whether `name` is a regular file of `size` bytes that hashes to `digest`."""
        try:
            raw = self._read(name, size, size)
        except (OSError, ValueError):
            return False
        return hashlib.sha256(raw).hexdigest() == digest

    def _read(self, name, size, limit):
        """Exactly `size` bytes of a regular file, refusing a link or another size."""
        if size > limit:
            raise ValueError("file is larger than the limit")
        dir_fd = os.open(self.directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(name, flags, dir_fd=dir_fd)
        finally:
            os.close(dir_fd)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("file is not a regular file")
            if info.st_size != size:
                raise ValueError(f"file holds {info.st_size} bytes")
            chunks, left = [], size
            while left:
                chunk = os.read(fd, min(left, 1 << 20))
                if not chunk:
                    raise ValueError("file ended early")
                chunks.append(chunk)
                left -= len(chunk)
            return b"".join(chunks)
        finally:
            os.close(fd)

    def get(self, ref, path, expected=None):
        """The bytes a reference names, refusing a bad name, a gap or a wrong hash.

        `expected` is the byte count the array's dtype and shape imply; it is checked
        against the reference before any file is opened, and the file is read only
        when its size is exactly that.
        """
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
        if expected is not None and size != expected:
            raise DocumentError(
                f"the reference says {size} bytes, the dtype and shape need {expected}",
                pointer((*path, "bytes")),
            )
        if size > self.max_bytes:
            raise DocumentError(
                f"array file {name} is {size} bytes, over the limit {self.max_bytes}",
                pointer((*path, "bytes")),
            )
        try:
            raw = self._read(name, size, self.max_bytes)
        except FileNotFoundError as err:
            raise DocumentError(
                f"array file {name} is missing from {self.directory}",
                pointer((*path, "name")),
            ) from err
        except OSError as err:
            raise DocumentError(
                f"array file {name} cannot be read as a regular file ({err.strerror}); "
                "a symlink is refused",
                pointer((*path, "name")),
            ) from err
        except ValueError as err:
            where = "name" if "regular" in str(err) else "bytes"
            raise DocumentError(
                f"array file {name}: {err}; the document says {size} bytes",
                pointer((*path, where)),
            ) from err
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


def make_store(files, max_inline, max_bytes=DEFAULT_MAX_BYTES):
    """A `FileStore` for a directory argument, or `None` when there is none."""
    if files is None:
        return None
    if type(max_inline) is not int or max_inline < 0:
        raise ValueError("max_inline is a non-negative element count")
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("max_bytes is a non-negative byte count")
    return FileStore(Path(files), max_inline, max_bytes)
