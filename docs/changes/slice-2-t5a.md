---
orphan: true
---

# Slice 2 T5a: the study codec core

A new subpackage, `spaceodyssey.codec`, turns a `Study` into canonical JSON and
back. It walks the fields of every module generically, so no kind writes its own
`to_params` or `from_params`. The codec sits outside `spaceodyssey.core`, because
its walk needs `type(...)` and a non-literal `getattr`, which the core lint
refuses; the core imports none of it, and the core import and domain-word lints
pass unchanged. No file under `core/` changed, and no run behavior changed: the
golden manifest is untouched.

## Document format

A document is a JSON object with four keys: `format` (`"spaceodyssey.study"`),
`schema_version` (1), `study` (the encoded study) and `content_hash`. Values are
encoded by type:

| Python value | JSON |
|---|---|
| `None`, `bool`, `int`, `str` | the JSON literal |
| finite `float` | a JSON number written with `repr`, so it round-trips, `-0.0` included |
| NaN, +inf, -inf | `{"$float": "nan"}`, `{"$float": "inf"}`, `{"$float": "-inf"}` |
| `tuple` | a JSON array |
| `list` | `{"$list": [...]}` |
| `dict` with string keys | `{"$dict": {...}}` |
| enum member | `{"$enum": [group, id, version], "value": ...}` |
| JAX or NumPy array | `{"$array": dtype, "shape": [...], "data": nested lists}` |
| `eqx.Module` or dataclass | `{"$kind": [group, id, version], <field>: <value>, ...}` |

Tuples are the common immutable container in modules, so they take the plain JSON
array and lists carry the tag. An array also carries `"weak": true` when it is a
weakly typed JAX array (the toy channel table holds two), and `"type": "numpy"` or
`"type": "numpy_scalar"` when it is not a JAX array. Float elements use the same
non-finite tags. A NaN is written as the one tag whatever its sign or payload
bits; every other float value, `-0.0` included, comes back bit for bit. Floats
narrower than 64 bits are written as the exact float64 value of each element.
Complex arrays, sets, functions and any other type are refused with the pointer of
the value.

Canonical JSON sorts keys, uses no whitespace, escapes non-ASCII characters and
refuses bare `NaN` and `Infinity`. The content hash is the sha256 hex digest of
the canonical bytes of `format`, `schema_version` and `study`. Integers are
written exactly, including values above 2^53 such as the default fork index; a
JSON reader that parses integers as doubles would lose them, and Python's does
not.

The codec requires x64 at both ends (`core.require_x64`).

## Kind references

Every module and dataclass records its class as `(group, id, version)`. The
caller passes the mapping from references to classes (`kinds`) to both
`dump_study` and `load_study`; `CORE_KINDS` (group `core`: the study, clock,
agenda, channel and asset tables, capacities, coupling, fork, effect table and
the core enums) is always included, and a caller key that rebinds a core key is
refused. A class field named `version`, which every abstract kind declares, is the
kind version: it is not written among the fields, the dumper picks the reference
whose version equals it, and the loader sets it from the reference. The mapping
is an argument, not a global, so `KindResolver` can replace it.

## Loading

`load_study` rebuilds each module from its stored field values as they are, the
way a pytree is unflattened (`object.__new__` and the stored fields): neither
`__init__` nor a field converter runs, so Python types, tuples, dtypes and weak
types come back exactly as written. Validation does run: each class's
`__check_init__` (an `eqx.Module`) or `__post_init__` (a plain dataclass), with an
error reported at the module's pointer. A reloaded toy study has an equal treedef,
an equal static part and bitwise-equal leaves, and the host executor runs it with
no new trace or compile.

Every load error is a `DocumentError` (a `ValueError`) with `pointer`, the RFC
6901 JSON pointer of the offending value, and `code`: `D001` for an unknown kind
reference and `D002` for a schema version newer than the loader, ready for the
JSON pre-flight. The loader refuses unknown and missing fields, unknown tags, an
array whose data does not match its declared shape or dtype (a float in an
integer array, for example), a weak flag on anything but an int64 or float64 JAX
array, a key given twice, and a stored hash that does not match.

## Public API

Added, in `spaceodyssey.codec` (not re-exported from the top-level package):

- `dump_study(study, kinds=None) -> StudyDocument`
- `load_study(document_or_text, kinds=None) -> Study`
- `StudyDocument(study, schema_version=1)`: `body()`, `canonical_bytes()`,
  `content_hash`, `to_json()`, `StudyDocument.from_json(text)`
- `DocumentError` (`pointer`, `code`), `KindTable`, `CORE_KINDS`,
  `canonical_json`, `FORMAT`, `SCHEMA_VERSION`

## Left for T5b and T5c

- T5b: the record-kind registry (`Capacities` keeps the lowered tables but not the
  `KindRegistry`), file references for large arrays, the named finding for a
  callable, `RunDocument` with its service bindings and schedule, and
  `{"no_switch": true}`. A per-class escape hook, if one is needed, goes in with
  the first kind whose fields the generic walk cannot hold.
- T5c: `KindResolver` and entry points replace the explicit mapping; `migrate`
  handles an older kind version, which T5a refuses as unknown.
- Three corpus studies use kinds the T5a test mapping does not name
  (`catalog_record`, `host_service`, `revision`); the other 25 round-trip with
  equal treedefs and hashes in a probe. The round trip on every corpus study is
  T5b's acceptance check.

## Tests

Added, in `tests/codec/test_document.py` (unit tier, 5 functions, 8 cases):

- the toy round trip is exact: treedef, static part, every leaf's type, dtype,
  shape, weak type and bytes, and the dumped text of the reload equals the
  original's (a two-phase study with a static process field, a bool, a Python
  float, a coupling holding an enum member and weak arrays);
- the reloaded baseline study runs on the host executor with zero trace or compile
  events and the same canonical history (checked by hand against a control with a
  changed static field, which traces 16 programs);
- the content hash computed in a subprocess with another `PYTHONHASHSEED` equals
  the in-process hash;
- NaN, +inf, -inf and `-0.0` survive in a float32 array, in float64 scalars and
  in Python floats, with no bare `NaN` or `Infinity` in the text, and the float32
  dtype is kept;
- a bad document fails with the pointer of the offending value: data that does
  not match its declared shape, a field that breaks its module's check, an unknown
  kind (`D001`) and a hand-edited hash.

Retired: none.
