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

from spaceodyssey.codec.bindings import (
    BindingKind as BindingKind,
)
from spaceodyssey.codec.bindings import Bindings as Bindings
from spaceodyssey.codec.bindings import build_binding
from spaceodyssey.codec.document import (
    RUN_FORMAT,
    CallableRefused,
    DocumentError,
    RunDocument,
    StudyDocument,
    pointer,
)
from spaceodyssey.codec.dump import dump_study
from spaceodyssey.codec.files import DEFAULT_MAX_BYTES, DEFAULT_MAX_INLINE
from spaceodyssey.codec.load import load_study
from spaceodyssey.codec.preflight import (
    CALLABLE_MEMBERS,
    RUN_KEY,
    DocumentCode,
    preflight_document,
    record_findings,
    service_findings,
)
from spaceodyssey.codec.records import check_lowering, decode_records, encode_records
from spaceodyssey.core.declarations import EXECUTORS
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor

_SIDES = ("host", "compiled")


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
        return build_binding(entry, self.service, self.registry, (RUN_KEY, "service"))

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


def load_run(
    document, bindings, kinds=None, *, files=None, max_file_bytes=DEFAULT_MAX_BYTES
):
    """Rebuild the `Run` a `RunDocument` (or its JSON text) describes.

    The document is validated by `preflight_document` with the same arguments, and
    the first error among its document-level findings (`D001`, `D002`, `D003`,
    `L001`) is raised as a `DocumentError` with that code (`None` for `L001`) and
    pointer, so `load_run` and pre-flight agree on what is wrong first. A callable
    member of the run record is `CallableRefused` at the same pointer. The study is
    then loaded strictly (see `load_study`).
    """
    if not isinstance(document, (str, RunDocument)):
        raise TypeError("load_run needs a RunDocument or its JSON text")
    findings = preflight_document(
        document,
        kinds,
        bindings=bindings,
        files=files,
        max_file_bytes=max_file_bytes,
    )
    for finding in findings:
        if finding.severity == "error" and isinstance(finding.code, DocumentCode):
            _raise(finding)
    if isinstance(document, str):
        document = RunDocument.from_json(document)
    run = document.run
    study = load_study(
        StudyDocument(study=document.study, schema_version=document.schema_version),
        kinds,
        files=files,
        max_file_bytes=max_file_bytes,
    )
    registry = decode_records(document.records)
    check_lowering(registry, study)
    boundaries = {}
    for side in _SIDES:
        if side in run.get("refit", {}):
            record = run["refit"][side]
            where = (RUN_KEY, "refit", side)
            entry = bindings.resolve(record["kind"], record["version"], where)
            boundaries[side] = build_binding(entry, record, registry, where)
    return Run(
        study, registry, run["executor"], run.get("service"), boundaries, bindings
    )


def _raise(finding):
    """Raise the error a document-level finding stands for."""
    text = finding.message
    if text.startswith("callable refused: "):
        raise CallableRefused(text[18:].split(" is code", 1)[0], finding.pointer)
    if finding.code == DocumentCode.D003 and any(
        f"member {name!r} is a callable" in text for name in CALLABLE_MEMBERS
    ):
        raise CallableRefused(finding.pointer.rsplit("/", 1)[-1], finding.pointer)
    code = None if finding.code == DocumentCode.L001 else str(finding.code)
    raise DocumentError(text, finding.pointer, code=code)


__all__ = ["RUN_FORMAT", "BindingKind", "Bindings", "Run", "dump_run", "load_run"]
