"""Service and refit-boundary kinds a run document may name.

A `BindingKind` says how to describe an instance as a configuration and how to build
one from it; `Bindings` is the table of them that a caller passes in, as study kinds
are passed to the loader. Code is never stored in a document.
"""

import dataclasses
from collections.abc import Callable

from spaceodyssey.codec.document import CallableRefused, DocumentError, pointer

_HOOK_ATTRIBUTES = ("fit", "kernel")


@dataclasses.dataclass(frozen=True)
class BindingKind:
    """A service or refit boundary the codec can describe and rebuild.

    `kind` and `version` name it in a document. `describe(obj, registry)` returns
    its configuration, a plain JSON object, and `build(configuration, registry)`
    returns an instance from one, where `registry` is the run's `KindRegistry` (a
    service names the record kind it writes by name there). `matches(obj)` says
    whether an instance is of this kind; it defaults to `type(obj) is cls`.
    `migrate(from_version, configuration)` reads a configuration stored at an older
    version. A refit boundary's `model_hash` is read from the instance and checked
    against the one the document stores when it is rebuilt.
    """

    kind: str
    version: int
    describe: Callable
    build: Callable
    cls: type | None = None
    matches: Callable | None = None
    migrate: Callable | None = None

    def __post_init__(self):
        """Require a name, a positive version and a way to match instances."""
        if not self.kind or type(self.version) is not int or self.version < 1:
            raise ValueError("a binding kind needs a name and a positive version")
        if self.cls is None and self.matches is None:
            raise ValueError(f"binding kind {self.kind!r} needs cls or matches")

    def covers(self, obj):
        """Whether `obj` is an instance of this kind."""
        if self.matches is not None:
            return bool(self.matches(obj))
        return type(obj) is self.cls


class Bindings:
    """The service and refit kinds a run document may name, passed in explicitly."""

    def __init__(self, kinds=()):
        """Register `kinds`; one kind name has one (current) version."""
        self._kinds = {}
        for kind in kinds:
            if kind.kind in self._kinds:
                raise ValueError(f"binding kind {kind.kind!r} is registered twice")
            self._kinds[kind.kind] = kind

    def resolve(self, kind, version, path=()):
        """The `BindingKind` a document names; `D001` if unknown or too new."""
        entry = self._kinds.get(kind)
        if entry is None:
            raise DocumentError(
                f"unknown binding kind {kind!r}", pointer((*path, "kind")), code="D001"
            )
        if version > entry.version:
            raise DocumentError(
                f"binding kind {kind!r} version {version} is newer than the "
                f"registered version {entry.version}",
                pointer((*path, "version")),
                code="D001",
            )
        if version < entry.version and entry.migrate is None:
            raise DocumentError(
                f"binding kind {kind!r} version {version} is older than "
                f"{entry.version} and defines no migrate",
                pointer((*path, "version")),
                code="D001",
            )
        return entry

    def of(self, obj, path):
        """The `BindingKind` an instance belongs to, or a refusal at `path`."""
        for entry in self._kinds.values():
            if entry.covers(obj):
                return entry
        name = type(obj).__name__
        for attr in _HOOK_ATTRIBUTES:
            hook = getattr(obj, attr, None)
            if callable(hook):
                label = getattr(hook, "__qualname__", type(hook).__name__)
                raise CallableRefused(f"{name}.{attr} ({label})", pointer(path))
        raise DocumentError(
            f"no binding kind is registered for a {name}", pointer(path)
        )


def build_binding(entry, record, registry, path):
    """An instance from a binding record, through `migrate` for an older version."""
    config = record["configuration"]
    if record["version"] < entry.version:
        try:
            config = entry.migrate(record["version"], dict(config))
        except (ValueError, TypeError, KeyError) as err:
            raise DocumentError(
                f"migrate of {entry.kind!r} from version {record['version']}: {err}",
                pointer((*path, "configuration")),
            ) from err
    try:
        return entry.build(config, registry)
    except (ValueError, TypeError, KeyError) as err:
        raise DocumentError(
            f"cannot build {entry.kind!r}: {err}", pointer((*path, "configuration"))
        ) from err
