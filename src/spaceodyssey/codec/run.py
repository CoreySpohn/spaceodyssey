"""Run documents: a study plus what a run binds, dumped and loaded together.

A `RunDocument` (`spaceodyssey.codec.document`) holds the study, the run record the
JSON pre-flight reads (executor, `Query` service, refit boundary on each executor,
each as a kind, a version and a configuration, a boundary also with its inference
model hash) and the record-kind registry. The refresh schedule is a field of the
study. `dump_run` writes one and `load_run` rebuilds a `Run`, which executes the
study on its bound executor with a fresh service and the bound boundary.

Code is never stored. A service or a boundary is named by a kind registered in a
`Bindings` table the caller passes in, as study kinds are passed to the loader; the
table says how to describe an instance as a configuration and how to build one from
it. A hook or function that no registered kind covers (`wake_fn`, a refit `fit`, an
activation) is refused by name.
"""

import dataclasses
import math
from collections.abc import Callable

from spaceodyssey.codec.document import (
    RUN_FORMAT,
    CallableRefused,
    DocumentError,
    RunDocument,
    StudyDocument,
    check_schema_version,
    pointer,
)
from spaceodyssey.codec.dump import dump_study
from spaceodyssey.codec.files import DEFAULT_MAX_INLINE
from spaceodyssey.codec.load import load_study
from spaceodyssey.codec.preflight import (
    CALLABLE_MEMBERS,
    RUN_KEY,
    record_findings,
    service_findings,
)
from spaceodyssey.codec.records import check_lowering, decode_records, encode_records
from spaceodyssey.core.declarations import EXECUTORS
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor

_SIDES = ("host", "compiled")
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


def _plain(value, path):
    """A configuration value, refusing anything JSON cannot hold."""
    if isinstance(value, dict):
        if not all(type(k) is str for k in value):
            raise DocumentError("a configuration needs string keys", pointer(path))
        return {k: _plain(v, (*path, k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v, (*path, i)) for i, v in enumerate(value)]
    if type(value) is float and not math.isfinite(value):
        raise DocumentError("a configuration number is not finite", pointer(path))
    if value is None or type(value) in (bool, int, float, str):
        return value
    if callable(value):
        raise CallableRefused(type(value).__name__, pointer(path))
    raise DocumentError(f"{type(value).__name__} is not a JSON value", pointer(path))


def _binding(obj, registry, bindings, path, *, boundary):
    """The record of one bound instance."""
    entry = bindings.of(obj, path)
    config = entry.describe(obj, registry)
    if not isinstance(config, dict):
        raise DocumentError(
            "a configuration is an object", pointer((*path, "configuration"))
        )
    out = {
        "kind": entry.kind,
        "version": entry.version,
        "configuration": _plain(config, (*path, "configuration")),
    }
    if boundary:
        out["model_hash"] = obj.model_hash
    return out


def dump_run(
    study,
    registry,
    *,
    executor,
    bindings,
    service=None,
    refit=None,
    kinds=None,
    files=None,
    max_inline=DEFAULT_MAX_INLINE,
    wake_fn=None,
):
    """Encode a study and its run bindings as a `RunDocument`.

    `registry` is the `KindRegistry` the study's capacities were lowered from; it
    must lower, with the study's observer, to exactly those capacities. `executor` is
    `"host"` or `"compiled"`. `service` is the `Query` service instance, if the study
    asks one. `refit` is a refit boundary, bound on both executors, or a dict with a
    `host` and a `compiled` boundary. `bindings` is the `Bindings` table that names
    them. `kinds`, `files` and `max_inline` are `dump_study`'s. A `wake_fn` is code
    and is refused (`CallableRefused`), as is a boundary or service no binding kind
    covers.
    """
    if wake_fn is not None:
        raise CallableRefused("wake_fn", "/run/wake_fn")
    if executor not in EXECUTORS:
        raise ValueError(f"executor is one of {sorted(EXECUTORS)}, not {executor!r}")
    check_lowering(registry, study)
    document = dump_study(study, kinds, files=files, max_inline=max_inline)
    run = {"executor": executor}
    if service is not None:
        run["service"] = _binding(
            service, registry, bindings, (RUN_KEY, "service"), boundary=False
        )
    if refit is not None:
        sides = refit if isinstance(refit, dict) else dict.fromkeys(_SIDES, refit)
        if set(sides) != set(_SIDES):
            raise ValueError("refit is a boundary or a dict with host and compiled")
        run["refit"] = {
            side: _binding(
                sides[side], registry, bindings, (RUN_KEY, "refit", side), boundary=True
            )
            for side in _SIDES
        }
    found = (*record_findings(run, strict=True), *service_findings(study, run))
    problems = [f for f in found if f.severity == "error"]
    if problems:
        first = problems[0]
        raise DocumentError(first.message, first.pointer, code=str(first.code))
    return RunDocument(
        study=document.study,
        schema_version=document.schema_version,
        run=run,
        records=encode_records(registry),
    )


@dataclasses.dataclass(frozen=True)
class Run:
    """A loaded run document: the study, its registry and what it is bound to.

    `service` is the record of the `Query` service (or `None`), `refit` maps each
    executor to its rebuilt boundary (or is empty), and `bindings` rebuilds a
    service on demand.
    """

    study: object
    registry: object
    executor: str
    service: dict | None
    refit: dict
    bindings: Bindings

    def make_service(self):
        """A fresh instance of the bound `Query` service, or `None`."""
        if self.service is None:
            return None
        entry = self.bindings.resolve(
            self.service["kind"], self.service["version"], (RUN_KEY, "service")
        )
        return _build(entry, self.service, self.registry, (RUN_KEY, "service"))

    def execute(self, truths, roots, n_steps, *, executor=None, **kw):
        """Run the bound study on `executor` (default: the document's executor)."""
        name = self.executor if executor is None else executor
        if name not in EXECUTORS:
            raise ValueError(f"executor is one of {sorted(EXECUTORS)}, not {name!r}")
        exe = HostExecutor() if name == "host" else CompiledExecutor()
        return exe.run_worlds(
            self.study,
            truths,
            roots,
            n_steps,
            service=self.make_service(),
            refit=self.refit.get(name),
            **kw,
        )


def _build(entry, record, registry, path):
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


def load_run(document, bindings, kinds=None, *, files=None):
    """Rebuild the `Run` a `RunDocument` (or its JSON text) describes.

    Every failure is a `DocumentError` with a pointer: the study's own (see
    `load_study`), a run record whose members are missing, ill-typed or unknown
    (`D003`), a member that is a callable (`CallableRefused`), a registry that does
    not decode or does not lower to the study's capacities (`/records`), a binding
    kind unknown or newer than the table (`D001`), and a rebuilt boundary whose
    `model_hash` is not the stored one.
    """
    if isinstance(document, str):
        document = RunDocument.from_json(document)
    elif not isinstance(document, RunDocument):
        raise TypeError("load_run needs a RunDocument or its JSON text")
    check_schema_version(document.schema_version)
    run = document.run
    _refuse_callables(run)
    study = load_study(
        StudyDocument(study=document.study, schema_version=document.schema_version),
        kinds,
        files=files,
    )
    problems = [
        f
        for f in (*record_findings(run, strict=True), *service_findings(study, run))
        if f.severity == "error"
    ]
    if problems:
        first = problems[0]
        raise DocumentError(first.message, first.pointer, code=str(first.code))
    if "executor" not in run:
        raise DocumentError("the run record has no 'executor'", "/run/executor")
    registry = decode_records(document.records)
    check_lowering(registry, study)
    service = run.get("service")
    if service is not None:
        entry = bindings.resolve(
            service["kind"], service["version"], (RUN_KEY, "service")
        )
        _build(entry, service, registry, (RUN_KEY, "service"))
    boundaries = {}
    for side in _SIDES:
        if side not in run.get("refit", {}):
            continue
        record = run["refit"][side]
        where = (RUN_KEY, "refit", side)
        entry = bindings.resolve(record["kind"], record["version"], where)
        boundary = _build(entry, record, registry, where)
        if boundary.model_hash != record["model_hash"]:
            raise DocumentError(
                f"the rebuilt boundary has model_hash {boundary.model_hash!r}, "
                f"the document says {record['model_hash']!r}",
                pointer((*where, "model_hash")),
            )
        boundaries[side] = boundary
    return Run(study, registry, run["executor"], service, boundaries, bindings)


def _refuse_callables(run):
    """Name a callable member of the run record before anything else is read."""
    if isinstance(run, dict):
        for key in sorted(set(run) & CALLABLE_MEMBERS):
            raise CallableRefused(key, pointer((RUN_KEY, key)))


__all__ = ["RUN_FORMAT", "BindingKind", "Bindings", "Run", "dump_run", "load_run"]
