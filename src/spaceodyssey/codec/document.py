"""The study document: canonical JSON, its content hash, and kind references.

A document is a JSON object with four keys:

- `format`: the string `"spaceodyssey.study"`;
- `schema_version`: the integer version of the encoding rules below;
- `study`: the encoded study value;
- `content_hash`: the sha256 hex digest of the canonical bytes of the other three
  keys. It is written by `StudyDocument.to_json` and checked by `from_json`.

Values are encoded by type:

- `None`, `bool`, `int`, `str`: the JSON literal
- finite `float`: a JSON number written with `repr`, so it round-trips (`-0.0` too)
- NaN, +inf, -inf: `{"$float": "nan"}`, `{"$float": "inf"}`, `{"$float": "-inf"}`
- `tuple`: a JSON array
- `list`: `{"$list": [...]}`
- `dict` with string keys: `{"$dict": [[key, value], ...]}`, in insertion order
- enum member: `{"$enum": [group, id, version], "value": ...}`
- JAX or NumPy array: `{"$array": dtype, "shape": [...], "data": nested lists}`
- `eqx.Module` or dataclass: `{"$kind": [group, id, version], <field>: <value>, ...}`

An array also carries `"weak": true` when it is a weakly typed JAX array, and
`"type": "numpy"` or `"type": "numpy_scalar"` when it is a NumPy array or scalar
rather than a JAX array. Float elements use the same non-finite tags; a NaN is
written as the one tag whatever its sign or payload bits. Floats narrower than 64
bits are written as the exact float64 value of each element. An array in a
non-native byte order is refused rather than silently converted.

A kind reference names a class by `(group, id, version)` through a
`KindResolver` (`spaceodyssey.codec.kinds`). A class field named `version` is the
kind version: it is not written among the fields, and loading sets it to the
registered version, after migrating the fields of an older one.

Canonical JSON sorts the keys of every JSON object, uses no whitespace, escapes
non-ASCII characters and refuses non-finite numbers (bare NaN or Infinity, or a
number such as `1e999` that overflows), so equal documents give equal bytes in
any process. Dict entries are a list, not an object, so their insertion order is
part of the content and of the hash.
"""

import dataclasses
import hashlib
import json
import math

FORMAT = "spaceodyssey.study"
SCHEMA_VERSION = 1

DOCUMENT_KEYS = frozenset({"format", "schema_version", "study", "content_hash"})


class DocumentError(ValueError):
    """A study or document the codec cannot encode or decode.

    `pointer` is the JSON pointer (RFC 6901) of the offending value in the
    document, or `None` when no single value is at fault. `code` is the pre-flight
    code the error corresponds to, when there is one: `D001` for an unknown kind
    id or version, `D002` for a schema version newer than this loader.
    """

    def __init__(self, message, pointer=None, code=None):
        """Store the pointer and code and put the pointer in the message."""
        self.pointer = pointer
        self.code = code
        where = "" if pointer is None else f" at {pointer or '/'}"
        prefix = "" if code is None else f"{code}: "
        super().__init__(f"{prefix}{message}{where}")


def pointer(path):
    """The JSON pointer of a path of keys and indices."""
    tokens = (str(p).replace("~", "~0").replace("/", "~1") for p in path)
    return "".join("/" + t for t in tokens)


def canonical_json(value):
    """Canonical JSON text: sorted keys, no whitespace, ASCII, finite numbers only."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class StudyDocument:
    """Canonical JSON form of a study, built by `dump_study` and read by `load_study`.

    `study` is the encoded study value (plain JSON types only) and
    `schema_version` the version of the encoding rules it follows.
    """

    study: dict
    schema_version: int = SCHEMA_VERSION

    def body(self):
        """The document without its hash."""
        return {
            "format": FORMAT,
            "schema_version": self.schema_version,
            "study": self.study,
        }

    def canonical_bytes(self):
        """The canonical JSON bytes the content hash is taken over."""
        body = self.body()
        check_parsed(body)
        return canonical_json(body).encode("ascii")

    @property
    def content_hash(self):
        """Hex sha256 digest of `canonical_bytes()`."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def to_json(self):
        """Canonical JSON text of the document with its `content_hash`."""
        return canonical_json({**self.body(), "content_hash": self.content_hash})

    @classmethod
    def from_json(cls, text):
        """Parse document text, checking its format, schema version and hash.

        The text need not be canonical: the hash is checked against the canonical
        bytes of what it parses to. A document without `content_hash` is accepted.
        """
        data = parse_text(text)
        if not isinstance(data, dict):
            raise DocumentError("a document is a JSON object", "")
        extra = sorted(set(data) - DOCUMENT_KEYS)
        if extra:
            raise DocumentError(
                f"unknown document key {extra[0]!r}", pointer(extra[:1])
            )
        for key in ("format", "schema_version", "study"):
            if key not in data:
                raise DocumentError(f"the document has no {key!r}", "")
        if data["format"] != FORMAT:
            raise DocumentError(f"format is not {FORMAT!r}", "/format")
        version = data["schema_version"]
        check_schema_version(version)
        doc = cls(study=data["study"], schema_version=version)
        stored = data.get("content_hash")
        if stored is not None and stored != doc.content_hash:
            raise DocumentError(
                "content_hash does not match the document's canonical bytes",
                "/content_hash",
            )
        return doc


def parse_text(text):
    """Parsed document text as plain dicts and lists.

    Malformed JSON, a key written twice and a non-finite number (bare NaN or
    Infinity, or an overflowing number) are each a `DocumentError` with a pointer.
    """
    try:
        data = json.loads(text, object_pairs_hook=_Pairs, parse_constant=float)
    except json.JSONDecodeError as err:
        raise DocumentError(f"malformed JSON: {err}", "") from err
    check_parsed(data)
    return _plain(data)


def check_schema_version(version):
    """Refuse a schema version this loader does not read; newer ones are `D002`."""
    if type(version) is not int or version < 1:
        raise DocumentError("schema_version is a positive integer", "/schema_version")
    if version > SCHEMA_VERSION:
        raise DocumentError(
            f"schema_version {version} is newer than this loader ({SCHEMA_VERSION})",
            "/schema_version",
            code="D002",
        )


class _Pairs(dict):
    """A parsed JSON object that remembers the first key it was given twice."""

    def __init__(self, pairs):
        """Keep the last value of each key, as `json` does, and note repeats."""
        super().__init__()
        self.repeated = None
        for key, value in pairs:
            if key in self and self.repeated is None:
                self.repeated = key
            self[key] = value


def check_parsed(value, path=()):
    """Refuse a repeated key or a non-finite number anywhere in parsed JSON.

    Bare NaN and Infinity parse to floats, and so does a number such as `1e999`
    that overflows, so all three are caught here, at their pointer, before any
    hash is taken.
    """
    if isinstance(value, dict):
        repeated = getattr(value, "repeated", None)
        if repeated is not None:
            raise DocumentError(
                f"key {repeated!r} appears twice", pointer((*path, repeated))
            )
        for key, item in value.items():
            check_parsed(item, (*path, key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            check_parsed(item, (*path, i))
    elif type(value) is float and not math.isfinite(value):
        raise DocumentError("a number is not finite; write a $float tag", pointer(path))


def _plain(value):
    """Parsed JSON with every object as a plain dict."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value
