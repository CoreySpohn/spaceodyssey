"""Pre-flight of a study document: every finding, each with a pointer.

`preflight_document` runs the structural checks on a JSON document without running
the study. It reports all thirteen finding codes:

- `D001`, `D002`: a kind id or version the resolver does not know, and a schema
  version newer than the loader. Every `$kind` and `$enum` reference is checked, so
  all unknown references are reported, each at its own pointer.
- `D003`: a run record that binds no service where a component asks a `Query`, a
  binding with a missing or ill-typed member, or host and compiled refit bindings
  that carry different inference-model hashes.
- the ten in-memory codes (`FindingCode`): the document is loaded and
  `spaceodyssey.core.preflight.preflight` runs on the study; each finding keeps its
  code and gains the JSON pointer of the offending value in the document.

Pointers name the value, not its module. A finding about a declaration points at the
field or element the declaration reads (`/study/observer/reads/0` for a read of
truth, `/study/agenda/policies/0/needs/0` for a need), and one about a table at the
element (`/study/channels/id/data/1`). The value is found in the decoded component and
located through the document's own encoding, so a field written as `{"$list": [...]}`
or `{"$dict": [...]}` gets the pointer that really holds it. A declaration that a
class computes rather than stores has no such field, and its finding points at the
component. The pointer of every finding resolves in the document.

Independent checks always run: the schema version, the kind references, the content
hash and the run record. The in-memory checks need a decoded study, so they run only
when the document has no `D001` or `D002` and decodes. A document that does not decode
for another reason (malformed JSON, a repeated key, a non-finite number, a wrong
shape, a module whose validation fails, a wrong envelope or hash) is reported as `L001`
at the pointer of the failure, alongside the independent findings; `L001` is not one of
the thirteen finding codes but the in-band form of a `DocumentError`. One field-level
failure is reported with its own code instead: a `null` coincident order loads, so
`T002` is reported with the other in-memory findings.

The run record is not part of the study document. It is a JSON object, passed as
`run` or stored under the key `run` of the document object, with these members:

- `executor`: `"host"` or `"compiled"`, used when `executor` is not passed;
- `service`: `{"kind", "version", "configuration"}`, the `Query` service;
- `refit`: `{"host": binding, "compiled": binding}`, each binding
  `{"kind", "version", "configuration", "model_hash"}`.

`kind` is a non-empty string, `version` a positive integer, `configuration` an object
and `model_hash` an integer or a non-empty string. Pointers into the record start with
`/run`. Without a record, `D003` is not checked.
"""

import dataclasses
import math
from enum import StrEnum

from spaceodyssey.codec.bindings import build_binding
from spaceodyssey.codec.document import (
    DOCUMENT_KEYS,
    FORMAT,
    RUN_FORMAT,
    RUN_KEYS,
    DocumentError,
    RunDocument,
    StudyDocument,
    check_schema_version,
    parse_text,
    pointer,
)
from spaceodyssey.codec.files import DEFAULT_MAX_BYTES
from spaceodyssey.codec.kinds import RegistrationError, as_resolver
from spaceodyssey.codec.load import load_study, tolerating_null_arrays
from spaceodyssey.codec.records import check_lowering, decode_records
from spaceodyssey.core.declarations import EXECUTORS, declaration_of
from spaceodyssey.core.enums import EffectKind
from spaceodyssey.core.ledger import ChannelTable
from spaceodyssey.core.preflight import Finding, FindingCode, Severity
from spaceodyssey.core.preflight import preflight as preflight_study
from spaceodyssey.core.preflight.checks import _components

RUN_KEY = "run"
_RUN_MEMBERS = frozenset({"executor", "service", "refit"})
CALLABLE_MEMBERS = frozenset({"wake_fn", "decide_py", "activation", "kernel", "fit"})


def _text(value):
    return type(value) is str and value != ""


def _version(value):
    return type(value) is int and value >= 1


def _object(value):
    return isinstance(value, dict)


def _is_hash(value):
    return type(value) is int or _text(value)


def _executor(value):
    return isinstance(value, str) and value in EXECUTORS


_SERVICE_SCHEMA = (
    ("kind", _text, "a non-empty string"),
    ("version", _version, "a positive integer"),
    ("configuration", _object, "an object"),
)
_REFIT_SCHEMA = (*_SERVICE_SCHEMA, ("model_hash", _is_hash, "an integer or a string"))


class DocumentCode(StrEnum):
    """Findings that only a document can have, and the in-band load failure."""

    D001 = "D001"  # an unknown kind id or version
    D002 = "D002"  # a schema version newer than the loader
    D003 = "D003"  # a binding missing, ill-typed, or refit model hashes that differ
    L001 = "L001"  # the document does not decode as a study (not one of the thirteen)


@dataclasses.dataclass(frozen=True)
class DocumentFinding:
    """One finding on a document: a code, the pointer of the offending value, a hint.

    `in_memory` is the finding the core pre-flight made on the loaded study, for the
    ten in-memory codes, and `None` for the document codes.
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


def preflight_document(
    document,
    kinds=None,
    *,
    run=None,
    executor=None,
    bindings=None,
    files=None,
    max_file_bytes=DEFAULT_MAX_BYTES,
):
    """Every pre-flight finding of a study document, errors first.

    `document` is JSON text, a parsed object or a `StudyDocument`. `kinds` is a
    `KindResolver` or the mapping `load_study` takes; `None` means the core kinds
    only, as in `dump_study` and `load_study`, and a caller that wants installed
    plugin kinds passes `spaceodyssey.plugins.installed_resolver()`. `run` is the run
    record (see the module docstring); `executor` selects the executor the in-memory
    capability check uses and defaults to the record's. A broken registration is a
    `RegistrationError` when the resolver is built, never a finding; document content
    never raises.

    A `RunDocument` (or its text, or a parsed object whose `format` is
    `spaceodyssey.run_document`) carries the run record as `run` and the record-kind
    registry as `records`, both hashed with the study: the record is checked as
    above, unknown members of it are `D003`, and the registry must decode and lower
    to the study's capacities (`L001` at `/records` otherwise). `bindings` is the
    `Bindings` table of service and refit kinds; with it, a binding that names an
    unknown kind or a newer version is `D001`; and, for a run document, each service
    and boundary is rebuilt as `load_run` rebuilds it: a configuration its kind rejects
    and a rebuilt `model_hash` that is not the stored one are `D003`. `files` is the
    directory of array files a dump wrote, forwarded to the loader with
    `max_file_bytes`, so a file-backed document is checked, integrity included.
    `load_run` raises the first error this returns.
    """
    resolver = _resolver(kinds)
    try:
        data = _parsed(document)
    except DocumentError as err:
        return [_from_error(err)]
    is_run = data.get("format") == RUN_FORMAT
    if is_run:
        if run is not None:
            raise ValueError("pass the run record once: as `run` or in the document")
        run = data.get(RUN_KEY)
    elif RUN_KEY in data:
        if run is not None:
            raise ValueError("pass the run record once: as `run` or in the document")
        run = data.pop(RUN_KEY)
    found, decodable = _envelope(data, is_run)
    version = data.get("schema_version")
    if version is not None:
        try:
            check_schema_version(version)
        except DocumentError as err:
            found.append(_from_error(err))
            decodable = False
    if "study" in data:
        refs = _unknown_kinds(data["study"], resolver)
        found += refs
        decodable = decodable and not refs
    if run is not None or is_run:
        found += record_findings(run, strict=is_run)
        if bindings is not None:
            found += _unknown_bindings(run, bindings)
        if executor is None and isinstance(run, dict):
            chosen = run.get("executor")
            executor = chosen if _executor(chosen) else None
    found += _check_hash(data)
    study = None
    if decodable:
        study, failure = _decoded(data, resolver, files, max_file_bytes)
        found += failure
    if study is not None:
        found += _in_memory(study, data["study"], executor)
        if run is not None:
            found += service_findings(study, run)
        if is_run and "records" in data:
            bad = _records(data["records"], study)
            found += bad
            if bindings is not None and not bad:
                found += _rebuilt(run, data["records"], bindings)
    return _ordered(found)


def _resolver(kinds):
    """A `KindResolver`, with a broken registration as a `RegistrationError`."""
    try:
        return as_resolver(kinds)
    except RegistrationError:
        raise
    except ValueError as err:
        raise RegistrationError(str(err)) from err


def _ordered(found):
    """Errors first, then by code and pointer."""
    return sorted(
        found, key=lambda f: (f.severity != Severity.ERROR, f.code, f.pointer)
    )


def _parsed(document):
    """The document as a JSON object, validated as the loader validates it.

    Text goes through the loader's parser: malformed JSON, a repeated key and a
    non-finite number are refused with a pointer. A `StudyDocument` is read as it is,
    as `load_study` reads it, with no stored hash. A parsed object is copied, and a
    value JSON cannot hold (a set, a non-string key, a non-finite number) is refused
    with a pointer.
    """
    if isinstance(document, RunDocument):
        return {
            "format": RUN_FORMAT,
            "schema_version": document.schema_version,
            "study": document.study,
            "run": document.run,
            "records": document.records,
        }
    if isinstance(document, StudyDocument):
        return {
            "format": FORMAT,
            "schema_version": document.schema_version,
            "study": document.study,
        }
    data = parse_text(document) if isinstance(document, str) else _json(document)
    if not isinstance(data, dict):
        raise DocumentError("a document is a JSON object", "")
    return data


def _json(value, path=()):
    """A copy of a parsed JSON value, refusing what JSON cannot hold."""
    if isinstance(value, dict):
        if not all(type(k) is str for k in value):
            raise DocumentError("an object needs string keys", pointer(path))
        return {k: _json(v, (*path, k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_json(v, (*path, i)) for i, v in enumerate(value)]
    if type(value) is float and not math.isfinite(value):
        raise DocumentError("a number is not finite; write a $float tag", pointer(path))
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise DocumentError(f"{type(value).__name__} is not a JSON value", pointer(path))


def _from_error(err, fix=None):
    """The finding a `DocumentError` stands for: its own code, or `L001`."""
    code = DocumentCode(err.code) if err.code else DocumentCode.L001
    fix = fix or "correct the value at the pointer so the document decodes"
    return DocumentFinding(code, Severity.ERROR, err.pointer or "", err.detail, fix)


def _envelope(data, is_run=False):
    """`L001` for a wrong envelope, and whether the study can be decoded."""
    out = []
    extra = sorted(set(data) - (RUN_KEYS if is_run else DOCUMENT_KEYS))
    if extra:
        err = DocumentError(f"unknown document key {extra[0]!r}", pointer(extra[:1]))
        out.append(_from_error(err))
    needed = ("format", "schema_version", "study") + (
        ("run", "records") if is_run else ()
    )
    for key in needed:
        if key not in data:
            out.append(_from_error(DocumentError(f"the document has no {key!r}", "")))
    want = RUN_FORMAT if is_run else FORMAT
    if "format" in data and data["format"] != want:
        out.append(_from_error(DocumentError(f"format is not {want!r}", "/format")))
    return out, not out


def _check_hash(data):
    """`L001` when a stored content hash differs from the document's canonical hash."""
    stored = data.get("content_hash")
    version = data["schema_version"]
    if stored is None or type(version) is not int:
        return []
    if data.get("format") == RUN_FORMAT and "run" in data and "records" in data:
        doc = RunDocument(
            study=data["study"],
            schema_version=version,
            run=data["run"],
            records=data["records"],
        )
    else:
        doc = StudyDocument(study=data["study"], schema_version=version)
    if stored == doc.content_hash:
        return []
    err = DocumentError(
        "content_hash does not match the document's canonical bytes", "/content_hash"
    )
    return [_from_error(err, "recompute the hash, or drop it after a hand edit")]


def _decoded(data, resolver, files=None, max_file_bytes=DEFAULT_MAX_BYTES):
    """The loaded study, or the `L001` of the failure; a null order loads."""
    doc = StudyDocument(study=data["study"], schema_version=data["schema_version"])
    try:
        with tolerating_null_arrays([(ChannelTable, "coincident_order")]):
            return load_study(
                doc, resolver, files=files, max_file_bytes=max_file_bytes
            ), []
    except DocumentError as err:
        return None, [_from_error(err)]


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
                        _from_error(err, "name a kind the installed plugins register")
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


# ---- in-memory findings, located in the document --------------------------------


def _in_memory(study, doc_study, executor):
    """The core findings on `study`, each at the pointer of its offending value."""
    return [
        DocumentFinding(
            item.code,
            item.severity,
            pointer(_follow(doc_study, _tokens(item, study), ("study",))),
            item.message,
            item.fix,
            in_memory=item,
        )
        for item in preflight_study(study, executor)
    ]


def _tokens(item, study):
    """The logical path (field names and indices) a finding's locator names.

    The slot is a field of the study (the policy slot is `agenda.policies`). The member
    and entry are followed only when the slot has a field of that name, which is how a
    component that stores what it declares is located; a component that computes its
    declaration is located at the slot.
    """
    if not item.subject:
        return ()
    if item.subject == "policy":
        tokens = ("agenda", "policies", item.index)
        slot = study.agenda.policies[item.index]
    else:
        tokens = (item.subject,)
        slot = getattr(study, item.subject)
    names = (
        {f.name for f in dataclasses.fields(slot)}
        if dataclasses.is_dataclass(slot)
        else ()
    )
    if item.member in names:
        tokens += (item.member, *item.entry)
    return tokens


def _step(node, token, path):
    """The child of an encoded node under a field name or index, with its path."""
    if isinstance(node, list):
        if type(token) is int and 0 <= token < len(node):
            return node[token], (*path, token)
        return None
    if not isinstance(node, dict):
        return None
    if "$kind" in node:
        if type(token) is str and token in node and token != "$kind":
            return node[token], (*path, token)
    elif "$list" in node:
        return _step(node["$list"], token, (*path, "$list"))
    elif "$dict" in node:
        for i, pair in enumerate(node["$dict"]):
            if pair[0] == token:
                return pair[1], (*path, "$dict", i, 1)
    elif "$array" in node:
        return _step(node.get("data"), token, (*path, "data"))
    return None


def _follow(node, tokens, path):
    """The path of the value `tokens` name, through the document's own encoding.

    Stops at the deepest value the encoding holds, so the result always resolves.
    """
    for token in tokens:
        nxt = _step(node, token, path)
        if nxt is None:
            break
        node, path = nxt
    return path


# ---- the run record ----------------------------


def _records(node, study):
    """`L001` when the registry does not decode or does not lower to the study."""
    try:
        check_lowering(decode_records(node), study)
    except DocumentError as err:
        return [
            _from_error(err, "write the registry the study's capacities lower from")
        ]
    return []


def _rebuilt(run, records, bindings):
    """`D003` for a service or boundary that does not rebuild, or another model."""
    try:
        registry = decode_records(records)
    except DocumentError:
        return []
    out = []
    if not isinstance(run, dict):
        return out
    refit = run.get("refit")
    items = [(("service",), run.get("service"), False)]
    if isinstance(refit, dict):
        items += [(("refit", s), refit.get(s), True) for s in ("host", "compiled")]
    for where, record, boundary in items:
        ok = (
            isinstance(record, dict)
            and _text(record.get("kind"))
            and _version(record.get("version"))
            and _object(record.get("configuration"))
        )
        if not ok:
            continue
        ptr = (RUN_KEY, *where)
        try:
            entry = bindings.resolve(record["kind"], record["version"], ptr)
            built = build_binding(entry, record, registry, ptr)
        except DocumentError as err:
            if err.code != "D001":
                out.append(
                    DocumentFinding(
                        DocumentCode.D003,
                        Severity.ERROR,
                        err.pointer or "",
                        err.detail,
                        "write a configuration the kind accepts",
                    )
                )
            continue
        if boundary and _is_hash(record.get("model_hash")):
            if built.model_hash != record["model_hash"]:
                out.append(
                    DocumentFinding(
                        DocumentCode.D003,
                        Severity.ERROR,
                        pointer((*ptr, "model_hash")),
                        f"the rebuilt boundary has model_hash {built.model_hash!r}, "
                        f"the document says {record['model_hash']!r}",
                        "store the model hash the boundary's kind reports",
                    )
                )
    return out


def _unknown_bindings(run, bindings):
    """`D001` for a service or refit binding that names a kind the table lacks."""
    found = []
    if not isinstance(run, dict):
        return found
    refit = run.get("refit")
    items = [(("service",), run.get("service"))]
    if isinstance(refit, dict):
        items += [(("refit", side), refit.get(side)) for side in ("host", "compiled")]
    for where, binding in items:
        if not (isinstance(binding, dict) and _text(binding.get("kind"))):
            continue
        if not _version(binding.get("version")):
            continue
        try:
            bindings.resolve(binding["kind"], binding["version"], (RUN_KEY, *where))
        except DocumentError as err:
            found.append(
                _from_error(err, "bind a kind the installed bindings register")
            )
    return found


def record_findings(run, strict=False):
    """`D003` for a run record whose members are missing, ill-typed or inconsistent.

    With `strict` (a run document), a member other than `executor`, `service` and
    `refit` is `D003` too; a callable such as `wake_fn` is named as one.
    """
    out = []

    def flag(ptr, message, fix):
        out.append(
            DocumentFinding(DocumentCode.D003, Severity.ERROR, ptr, message, fix)
        )

    if not isinstance(run, dict):
        flag(pointer((RUN_KEY,)), "the run record is not an object", "write an object")
        return out
    if strict:
        for key in sorted(set(run) - _RUN_MEMBERS):
            what = (
                "a callable, which a document cannot hold"
                if key in CALLABLE_MEMBERS
                else "unknown"
            )
            flag(
                pointer((RUN_KEY, key)),
                f"run member {key!r} is {what}",
                "remove it; bind code by kind in service or refit",
            )
        if "executor" not in run:
            flag(pointer((RUN_KEY,)), "the run record has no 'executor'", "add it")
    if "executor" in run and not _executor(run["executor"]):
        flag(
            pointer((RUN_KEY, "executor")),
            f"executor {run['executor']!r} is not one of {sorted(EXECUTORS)}",
            "bind 'host' or 'compiled'",
        )
    if "service" in run:
        _binding(run["service"], ("service",), _SERVICE_SCHEMA, flag)
    if "refit" not in run:
        return out
    refit = run["refit"]
    if not isinstance(refit, dict):
        flag(
            pointer((RUN_KEY, "refit")),
            "the refit record is not an object",
            "write an object",
        )
        return out
    hashes = {}
    for side in ("host", "compiled"):
        if side not in refit:
            flag(
                pointer((RUN_KEY, "refit")),
                f"the refit record has no {side} binding",
                f"bind the {side} refit boundary",
            )
            continue
        valid = _binding(refit[side], ("refit", side), _REFIT_SCHEMA, flag)
        if "model_hash" in valid:
            hashes[side] = refit[side]["model_hash"]
    if len(hashes) == 2 and hashes["host"] != hashes["compiled"]:
        flag(
            pointer((RUN_KEY, "refit", "compiled", "model_hash")),
            "host and compiled refit bindings carry different model hashes "
            f"({hashes['host']!r} and {hashes['compiled']!r})",
            "bind both executors to boundaries that fit the same inference model",
        )
    return out


def _binding(binding, where, schema, flag):
    """Flag each member a binding lacks or holds in the wrong form; return the valid."""
    ptr = pointer((RUN_KEY, *where))
    if not isinstance(binding, dict):
        flag(ptr, f"the {where[-1]} binding is not an object", "write an object")
        return set()
    valid = set()
    for name, ok, form in schema:
        if name not in binding:
            flag(ptr, f"the {where[-1]} binding has no {name!r}", f"add {name!r}")
        elif not ok(binding[name]):
            flag(
                pointer((RUN_KEY, *where, name)),
                f"{name!r} of the {where[-1]} binding is not {form}",
                f"write {name!r} as {form}",
            )
        else:
            valid.add(name)
    return valid


def service_findings(study, run):
    """`D003` when a component asks a `Query` and the record binds no service."""
    asking = [
        name
        for name, component in _components(study)
        if EffectKind.QUERY in declaration_of(component).effects
    ]
    if not asking or (isinstance(run, dict) and "service" in run):
        return []
    message = (
        f"{', '.join(asking)} asks for a Query and the run record binds no service"
    )
    fix = "bind the Query service (kind, version, configuration) under 'service'"
    return [
        DocumentFinding(
            DocumentCode.D003, Severity.ERROR, pointer((RUN_KEY,)), message, fix
        )
    ]
