"""Pre-flight of a study document: every finding, each with a pointer.

`preflight_document` runs the structural checks on a JSON document without running
the study. It reports all thirteen finding codes:

- `D001`, `D002`: a kind id or version the resolver does not know, and a schema
  version newer than the loader. Every `$kind` and `$enum` reference is checked, so
  all unknown references are reported, each at its own pointer.
- `D003`: a service binding missing from the run record, or host and compiled refit
  bindings that carry different inference-model hashes.
- the ten in-memory codes (`FindingCode`): the document is loaded and
  `spaceodyssey.core.preflight.preflight` runs on the study; each finding keeps its
  code and gains the JSON pointer of the offending value in the document.

Unknown kinds and a newer schema stop the load, so for such a document only the `D`
findings can be reported (the in-memory checks need a study). A document that
fails to load for any other reason (malformed JSON, a wrong shape, a hand-edited
hash, a module whose validation fails) raises `DocumentError` with its pointer, as
`load_study` does: those are not findings of the thirteen codes.

The run record is not part of the study document. It is a JSON object, passed as
`run` or stored under the key `run` of the document object, with these members:

- `executor`: `"host"` or `"compiled"`, used when `executor` is not passed;
- `service`: `{"kind", "version", "configuration"}`, the `Query` service;
- `refit`: `{"host": binding, "compiled": binding}`, each binding
  `{"kind", "version", "configuration", "model_hash"}`.

Pointers into the record start with `/run`. Without a record, `D003` is not checked.
"""

import dataclasses
import json
from enum import StrEnum

from spaceodyssey.codec.document import (
    DocumentError,
    StudyDocument,
    canonical_json,
    check_schema_version,
    pointer,
)
from spaceodyssey.codec.kinds import as_resolver
from spaceodyssey.codec.load import load_study
from spaceodyssey.core.declarations import declaration_of
from spaceodyssey.core.enums import EffectKind
from spaceodyssey.core.preflight import Finding, FindingCode, Severity
from spaceodyssey.core.preflight import preflight as preflight_study

RUN_KEY = "run"
_SERVICE_FIELDS = ("kind", "version", "configuration")
_REFIT_FIELDS = ("kind", "version", "configuration", "model_hash")
_COMPONENTS = frozenset(
    {"world_source", "process", "channel_model", "observer", "objective", "design"}
)


class DocumentCode(StrEnum):
    """Findings that only a document can have."""

    D001 = "D001"  # an unknown kind id or version
    D002 = "D002"  # a schema version newer than the loader
    D003 = "D003"  # a service binding missing, or refit model hashes that differ


@dataclasses.dataclass(frozen=True)
class DocumentFinding:
    """One finding on a document: a code, the pointer of the offending value, a hint.

    `in_memory` is the finding the core pre-flight made on the loaded study, for the
    ten in-memory codes, and `None` for `D001` to `D003`.
    """

    code: FindingCode | DocumentCode
    severity: Severity
    pointer: str
    message: str
    fix: str
    in_memory: Finding | None = None

    def __str__(self) -> str:
        """One line plus the fix hint."""
        return (
            f"[{self.code}] {self.severity} at {self.pointer or '/'}: {self.message}\n"
            f"        fix: {self.fix}"
        )


def resolve_pointer(root, ptr):
    """The value a JSON pointer (RFC 6901) names in parsed JSON; `KeyError` if none."""
    node = root
    for token in ptr.split("/")[1:] if ptr else ():
        token = token.replace("~1", "/").replace("~0", "~")
        try:
            node = node[int(token)] if isinstance(node, list) else node[token]
        except (KeyError, IndexError, ValueError, TypeError):
            raise KeyError(f"{ptr} does not resolve in the document") from None
    return node


def preflight_document(document, kinds=None, *, run=None, executor=None):
    """Every pre-flight finding of a study document, errors first.

    `document` is JSON text, a parsed object or a `StudyDocument`. `kinds` is a
    `KindResolver` or the mapping `load_study` takes; `None` means the core kinds
    only, as in `dump_study` and `load_study`, and a caller that wants installed
    plugin kinds passes `spaceodyssey.plugins.installed_resolver()`. `run` is the run
    record (see the module docstring); `executor` selects the executor the in-memory
    capability check uses and defaults to the record's. A broken installation is a
    `RegistrationError` when the resolver is built, never a finding.
    """
    resolver = as_resolver(kinds)
    data = _parsed(document)
    if RUN_KEY in data:
        if run is not None:
            raise ValueError("pass the run record once: as `run` or in the document")
        run = data.pop(RUN_KEY)
    try:
        check_schema_version(data.get("schema_version"))
    except DocumentError as err:
        if err.code is None:
            raise
        return [_document_finding(err, "write a schema version this loader reads")]
    found = _unknown_kinds(data.get("study"), resolver)
    if found:
        return _ordered(found)
    study = load_study(canonical_json(data), resolver)
    if executor is None and isinstance(run, dict):
        executor = run.get("executor")
    for item in preflight_study(study, executor):
        found.append(
            DocumentFinding(
                item.code,
                item.severity,
                _locate(item),
                item.message,
                item.fix,
                in_memory=item,
            )
        )
    if run is not None:
        found += _bindings(study, run)
    return _ordered(found)


def _ordered(found):
    """Errors first, then by code and pointer."""
    return sorted(
        found, key=lambda f: (f.severity != Severity.ERROR, f.code, f.pointer)
    )


def _parsed(document):
    """The document as a parsed JSON object that is safe to edit."""
    if isinstance(document, StudyDocument):
        document = document.to_json()
    elif not isinstance(document, str):
        document = canonical_json(document)
    try:
        data = json.loads(document)
    except json.JSONDecodeError as err:
        raise DocumentError(f"malformed JSON: {err}", "") from err
    if not isinstance(data, dict):
        raise DocumentError("a document is a JSON object", "")
    return data


def _document_finding(err, fix):
    """The `D` finding a coded `DocumentError` stands for."""
    return DocumentFinding(
        DocumentCode(err.code), Severity.ERROR, err.pointer, str(err), fix
    )


def _unknown_kinds(node, resolver, path=("study",)):
    """A `D001` for every `$kind` and `$enum` reference the resolver cannot resolve."""
    out = []
    if isinstance(node, list):
        for i, item in enumerate(node):
            out += _unknown_kinds(item, resolver, (*path, i))
    elif isinstance(node, dict):
        for tag in ("$kind", "$enum"):
            ref = node.get(tag)
            if _is_ref(ref):
                try:
                    resolver.resolve(tuple(ref), (*path, tag))
                except DocumentError as err:
                    out.append(
                        _document_finding(
                            err, "name a kind the installed plugins register"
                        )
                    )
        if "$array" not in node:
            for key, item in node.items():
                out += _unknown_kinds(item, resolver, (*path, key))
    return out


def _is_ref(ref):
    """Whether `ref` has the shape `[group, id, version]`."""
    return (
        type(ref) is list
        and len(ref) == 3
        and type(ref[0]) is str
        and type(ref[1]) is str
        and type(ref[2]) is int
    )


def _locate(item):
    """The pointer of the document value an in-memory finding is about."""
    head = item.path.split(" <- ")[0]
    study = ("study",)
    if head.startswith("policy["):
        return pointer((*study, "agenda", "policies", _index(head)))
    if head in _COMPONENTS:
        return pointer((*study, head))
    if item.code == FindingCode.K001:
        name = {"world": "world_source", "process": "process"}.get(head)
        return pointer((*study, name or "channel_model"))
    if item.code == FindingCode.K002:
        return pointer((*study, "channels", "id", "data", _index(head)))
    if item.code == FindingCode.R001:
        return pointer((*study, "channels", "kind", "data", _index(head)))
    return pointer((*study, *head.split(".")))


def _index(name):
    """The integer in `name[3]`."""
    return int(name[name.index("[") + 1 : -1])


def _bindings(study, run):
    """`D003`: a missing service binding, or refit model hashes that differ."""
    out = []

    def flag(ptr, message, fix):
        out.append(
            DocumentFinding(DocumentCode.D003, Severity.ERROR, ptr, message, fix)
        )

    if not isinstance(run, dict):
        flag(pointer((RUN_KEY,)), "the run record is not an object", "write an object")
        return out
    asking = [
        name
        for name, component in _service_askers(study)
        if EffectKind.QUERY in declaration_of(component).effects
    ]
    service = run.get("service")
    if asking and service is None:
        flag(
            pointer((RUN_KEY,)),
            f"{', '.join(asking)} asks for a Query and the run record binds no service",
            "bind the Query service (kind, version, configuration) under 'service'",
        )
    elif service is not None:
        _check_binding(service, ("service",), _SERVICE_FIELDS, flag)
    refit = run.get("refit")
    if refit is None:
        return out
    hashes = {}
    for side in ("host", "compiled"):
        binding = refit.get(side) if isinstance(refit, dict) else None
        where = ("refit", side)
        if binding is None:
            flag(
                pointer((RUN_KEY, "refit")),
                f"the refit record has no {side} binding",
                f"bind the {side} refit boundary",
            )
        elif not _check_binding(binding, where, _REFIT_FIELDS, flag):
            hashes[side] = binding["model_hash"]
    if len(hashes) == 2 and hashes["host"] != hashes["compiled"]:
        flag(
            pointer((RUN_KEY, "refit", "compiled", "model_hash")),
            f"host and compiled refit bindings carry different model hashes "
            f"({hashes['host']!r} and {hashes['compiled']!r})",
            "bind both executors to boundaries that fit the same inference model",
        )
    return out


def _check_binding(binding, where, required, flag):
    """Flag a binding that is not an object or lacks a member; return what is wrong."""
    ptr = pointer((RUN_KEY, *where))
    wrong = []
    if not isinstance(binding, dict):
        flag(ptr, f"the {where[-1]} binding is not an object", "write an object")
        return [True]
    for name in required:
        if name not in binding:
            flag(ptr, f"the {where[-1]} binding has no {name!r}", f"add {name!r}")
            wrong.append(name)
    return wrong


def _service_askers(study):
    """(name, component) pairs whose declarations can ask a Query."""
    pairs = [("observer", study.observer), ("process", study.process)]
    pairs += [(f"policy[{i}]", p) for i, p in enumerate(study.agenda.policies)]
    return pairs
