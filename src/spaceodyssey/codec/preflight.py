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
import enum
import re
from enum import StrEnum

import numpy as np

from spaceodyssey.codec.document import (
    DOCUMENT_KEYS,
    FORMAT,
    DocumentError,
    StudyDocument,
    canonical_json,
    check_schema_version,
    parse_text,
    pointer,
)
from spaceodyssey.codec.kinds import RegistrationError, as_resolver, is_version_field
from spaceodyssey.codec.load import load_study, tolerating_null_arrays
from spaceodyssey.core.declarations import (
    EXECUTORS,
    SUPPORTED_EFFECTS,
    declaration_of,
)
from spaceodyssey.core.enums import CoincidentOrder, Disposition, EffectKind, Scope
from spaceodyssey.core.preflight import Finding, FindingCode, Severity
from spaceodyssey.core.preflight import preflight as preflight_study
from spaceodyssey.core.preflight.checks import _components

RUN_KEY = "run"


def _text(value):
    return type(value) is str and value != ""


def _version(value):
    return type(value) is int and value >= 1


def _object(value):
    return isinstance(value, dict)


def _hash(value):
    return type(value) is int or _text(value)


_SERVICE_SCHEMA = (
    ("kind", _text, "a non-empty string"),
    ("version", _version, "a positive integer"),
    ("configuration", _object, "an object"),
)
_REFIT_SCHEMA = (*_SERVICE_SCHEMA, ("model_hash", _hash, "an integer or a string"))
_COMPONENTS = frozenset(
    {"world_source", "process", "channel_model", "observer", "objective", "design"}
)


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


def preflight_document(document, kinds=None, *, run=None, executor=None):
    """Every pre-flight finding of a study document, errors first.

    `document` is JSON text, a parsed object or a `StudyDocument`. `kinds` is a
    `KindResolver` or the mapping `load_study` takes; `None` means the core kinds
    only, as in `dump_study` and `load_study`, and a caller that wants installed
    plugin kinds passes `spaceodyssey.plugins.installed_resolver()`. `run` is the run
    record (see the module docstring); `executor` selects the executor the in-memory
    capability check uses and defaults to the record's. A broken registration is a
    `RegistrationError` when the resolver is built, never a finding; document content
    never raises.
    """
    resolver = _resolver(kinds)
    try:
        data = _parsed(document)
    except DocumentError as err:
        return [_from_error(err)]
    if RUN_KEY in data:
        if run is not None:
            raise ValueError("pass the run record once: as `run` or in the document")
        run = data.pop(RUN_KEY)
    found, decodable = _envelope(data)
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
    if run is not None:
        found += _record(run)
        if executor is None and isinstance(run, dict):
            chosen = run.get("executor")
            executor = chosen if chosen in EXECUTORS else None
    study = None
    if decodable:
        found += _hash(data)
        study, failure = _decoded(data, resolver)
        found += failure
    if study is not None:
        found += _in_memory(study, data["study"], executor)
        if run is not None:
            found += _service(study, run)
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
    """The document as a parsed JSON object.

    Duplicate keys and non-finite numbers are refused as the loader refuses them.
    """
    if isinstance(document, StudyDocument):
        document = document.to_json()
    elif not isinstance(document, str):
        document = canonical_json(document)
    data = parse_text(document)
    if not isinstance(data, dict):
        raise DocumentError("a document is a JSON object", "")
    return data


def _from_error(err, fix=None):
    """The finding a `DocumentError` stands for: its own code, or `L001`."""
    code = DocumentCode(err.code) if err.code else DocumentCode.L001
    fix = fix or "correct the value at the pointer so the document decodes"
    return DocumentFinding(code, Severity.ERROR, err.pointer or "", str(err), fix)


def _envelope(data):
    """`L001` for a wrong envelope, and whether the study can be decoded."""
    out = []
    extra = sorted(set(data) - DOCUMENT_KEYS)
    if extra:
        err = DocumentError(f"unknown document key {extra[0]!r}", pointer(extra[:1]))
        out.append(_from_error(err))
    for key in ("format", "schema_version", "study"):
        if key not in data:
            out.append(_from_error(DocumentError(f"the document has no {key!r}", "")))
    if "format" in data and data["format"] != FORMAT:
        out.append(_from_error(DocumentError(f"format is not {FORMAT!r}", "/format")))
    return out, not out


def _hash(data):
    """`L001` when a stored content hash differs from the document's canonical hash."""
    stored = data.get("content_hash")
    version = data["schema_version"]
    if stored is None or type(version) is not int:
        return []
    doc = StudyDocument(study=data["study"], schema_version=version)
    if stored == doc.content_hash:
        return []
    err = DocumentError(
        "content_hash does not match the document's canonical bytes", "/content_hash"
    )
    return [_from_error(err, "recompute the hash, or drop it after a hand edit")]


def _decoded(data, resolver):
    """The loaded study, or the `L001` of the failure; a null order loads."""
    doc = StudyDocument(study=data["study"], schema_version=data["schema_version"])
    try:
        with tolerating_null_arrays(("coincident_order",)):
            return load_study(doc, resolver), []
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
    seen = {}
    out = []
    for item in preflight_study(study, executor):
        tokens = _logical(item, study, seen)
        out.append(
            DocumentFinding(
                item.code,
                item.severity,
                pointer(_follow(doc_study, tokens, ("study",))),
                item.message,
                item.fix,
                in_memory=item,
            )
        )
    return out


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


def _walk(value, path=()):
    """(path, node) for every field, element and entry below `value`, depth first."""
    if dataclasses.is_dataclass(value) and not isinstance(value, (type, enum.Enum)):
        for field in dataclasses.fields(value):
            if is_version_field(type(value), field.name):
                continue
            child = getattr(value, field.name)
            yield (*path, field.name), child
            yield from _walk(child, (*path, field.name))
    elif type(value) in (tuple, list):
        for i, child in enumerate(value):
            yield (*path, i), child
            yield from _walk(child, (*path, i))
    elif type(value) is dict:
        for key, child in value.items():
            yield (*path, key), child
            yield from _walk(child, (*path, key))


def _where(study, name):
    """(logical path, component) of a component name used by the core findings."""
    if name.startswith("policy["):
        i = _index(name)
        return ("agenda", "policies", i), study.agenda.policies[i]
    return (name,), getattr(study, name)


def _inside(study, name, test, in_sequence=False):
    """The path of the first value under a component that satisfies `test`.

    With `in_sequence`, a match that is an element of a tuple or list is preferred.
    Without any match the component's own path is returned.
    """
    base, component = _where(study, name)
    first = None
    for path, node in _walk(component, base):
        if test(node):
            if not in_sequence or type(path[-1]) is int:
                return path
            first = first or path
    return first or base


def _is_seq(value):
    return type(value) in (tuple, list)


def _int_like(value, number):
    return isinstance(value, int) and not isinstance(value, bool) and value == number


def _next(seen, key):
    """How many findings of this kind came before: the position of this one."""
    seen[key] = seen.get(key, -1) + 1
    return seen[key]


def _index(name):
    """The integer in `name[3]`."""
    return int(name[name.index("[") + 1 : -1])


def _logical(item, study, seen):
    """The logical path (field names and indices) of the value a finding is about."""
    code, head = item.code, item.path.split(" <- ")[0]
    if code in (FindingCode.F001, FindingCode.F002):
        needle = "truth" if code == FindingCode.F001 else "scorer"
        return _inside(study, head, lambda v: v == needle and type(v) is str, True)
    if code == FindingCode.C001:
        return _need(item, head, study, seen)
    if code == FindingCode.K001:
        name = {"world": "world_source", "process": "process"}.get(head)
        name = name or "channel_model"
        coords = tuple(declaration_of(_where(study, name)[1]).key_coords)
        return _inside(study, name, lambda v: _is_seq(v) and tuple(v) == coords)
    if code == FindingCode.K002:
        return ("channels", "id", _index(head))
    if code == FindingCode.R001:
        return ("channels", "kind", _index(head))
    if code == FindingCode.X001:
        return _capability(item, head, study, seen)
    if code == FindingCode.X002:
        return _reserved(head, study)
    if code == FindingCode.T001:
        return _drain(head, study, seen)
    if code == FindingCode.T002:
        return _order(study)
    return _where(study, head)[0]


def _need(item, head, study, seen):
    """The `needs` entry a C001 is about, found by its channel id and position."""
    cid = int(item.path.rsplit(" ", 1)[1])
    policy = _where(study, head)[1]
    needs = [n for n in declaration_of(policy).needs if n.channel_id == cid]
    k = _next(seen, (head, "need", cid))
    if k >= len(needs):
        return _where(study, head)[0]
    want = (needs[k].channel_id, needs[k].start_tick, needs[k].decide_tick)

    def triple(v):
        return _is_seq(v) and len(v) == 3 and tuple(int(x) for x in v) == want

    return _inside(study, head, triple)


def _capability(item, head, study, seen):
    """The effect, requirement or executor set an X001 is about."""
    decl = declaration_of(_where(study, head)[1])
    message = item.message
    if message.startswith("asks for"):
        bad = [
            k
            for k in sorted(decl.effects, key=int)
            if EffectKind(k) not in SUPPORTED_EFFECTS
        ]
        k = _next(seen, (head, "effect"))
        if k < len(bad):
            return _inside(study, head, lambda v: _int_like(v, int(bad[k])), True)
    elif message.startswith("requires"):
        found = re.match(r"requires (['\"])(.*?)\1", message)
        if found:
            name = found.group(2)
            return _inside(study, head, lambda v: v == name and type(v) is str, True)
    elif message.startswith("runs only on"):
        names = set(decl.executors)
        return _inside(
            study,
            head,
            lambda v: _is_seq(v) and v and set(v) == names and all(map(_is_text, v)),
        )
    return _where(study, head)[0]


def _is_text(value):
    return type(value) is str


def _reserved(head, study):
    """The first element of a reserved column that breaks its rule, or the field."""
    if head.startswith("policy["):
        return _inside(study, head, lambda v: v is True)
    table, column = head.split(".")
    ch, assets = study.channels, study.assets
    if head == "channels.charge":
        a, b = np.asarray(ch.charge), np.asarray(ch.cost)
        bad = np.argwhere(a != b) if a.shape == b.shape else []
    else:
        value = 0 if column == "freeze_ticks" else -1
        bad = np.argwhere(np.asarray(getattr(assets, column)) != value)
    tokens = (table, column)
    return (*tokens, *map(int, bad[0])) if len(bad) else tokens


def _drain(head, study, seen):
    """The `stops` pair a T001 is about, found by its position among the drains."""
    policy = _where(study, head)[1]
    pairs = [
        (int(scope), int(disposition))
        for scope, disposition in declaration_of(policy).stops
        if disposition == Disposition.DRAIN and scope != Scope.CAMPAIGN
    ]
    k = _next(seen, (head, "drain"))
    if k >= len(pairs):
        return _where(study, head)[0]

    def pair(v):
        ok = _is_seq(v) and len(v) == 2 and all(isinstance(x, int) for x in v)
        return ok and (int(v[0]), int(v[1])) == pairs[k]

    return _inside(study, head, pair)


def _order(study):
    """The first coincident order outside the enum, or the field."""
    table = study.channels
    field = ("channels", "coincident_order")
    if table.coincident_order is None:
        return field
    values = np.asarray(table.coincident_order)
    n = np.asarray(table.id).shape[0]
    if values.shape != (n,):
        return field
    bad = np.flatnonzero(~np.isin(values, [int(v) for v in CoincidentOrder]))
    return (*field, int(bad[0])) if len(bad) else field


# ---- the run record -------------------------------------------------------------


def _record(run):
    """`D003` for a run record whose members are missing, ill-typed or inconsistent."""
    out = []

    def flag(ptr, message, fix):
        out.append(
            DocumentFinding(DocumentCode.D003, Severity.ERROR, ptr, message, fix)
        )

    if not isinstance(run, dict):
        flag(pointer((RUN_KEY,)), "the run record is not an object", "write an object")
        return out
    if "executor" in run and run["executor"] not in EXECUTORS:
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


def _service(study, run):
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
