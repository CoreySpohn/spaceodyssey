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
| `dict` with string keys | `{"$dict": [[key, value], ...]}`, in insertion order |
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
Complex arrays, arrays in a non-native byte order, sets, functions and any other
type are refused with the pointer of the value, as is a field annotated
`jax.Array` that holds something other than an array.

Canonical JSON sorts the keys of every JSON object, uses no whitespace, escapes
non-ASCII characters and refuses non-finite numbers. Dict entries are an ordered
list of pairs, not a JSON object, so a dict's insertion order is part of the
content and of the hash. The content hash is the sha256 hex digest of
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
way a pytree is unflattened (`object.__new__` and the stored fields), so Python
types, tuples, dtypes and weak types come back exactly as written; `__init__` does
not run. The guarantees construction gives are checked instead, never applied, so
a valid stored value is not coerced and an invalid one is refused:

- a field with a converter must hold a value the converter returns unchanged
  (type, dtype, shape, weak flag and bytes): a clock of `"hello"` or an `int64`
  tick stored as the float `10.75` is refused, not converted or truncated;
- a field annotated `jax.Array` must hold an array, so a tuple there is refused
  before any check reads it;
- a class's `__post_init__`, where it defines one (an `eqx.Module` or a plain
  dataclass), runs and must leave every field as stored; then each
  `__check_init__` runs.

A tag must name the right category of class: `$enum` an `Enum` (with a `value`),
`$kind` a module or dataclass, never an enum, so no document can call a mapped
module's constructor. A reloaded toy study has an equal treedef, an equal static
part and bitwise-equal leaves, and the host executor runs it with no new trace or
compile. All 28 corpus studies round-trip exactly; in a probe, every one also ran
again on both executors after reloading with zero new traces or compiles.

Every load error is a `DocumentError` (a `ValueError`) with `pointer`, the RFC
6901 JSON pointer of the offending value, and `code`: `D001` for an unknown kind
reference and `D002` for a schema version newer than the loader, ready for the
JSON pre-flight. The loader refuses unknown and missing fields, unknown tags, an
array whose data does not match its declared shape or dtype (a float in an
integer array, for example), a weak flag on anything but an int64 or float64 JAX
array, a key given twice in any object (reported at that key's path), a number
that is not finite (bare `NaN` or `Infinity`, or a value such as `1e999` that
overflows; reported at its pointer before any hash is taken), malformed JSON
(pointer `""`), and a stored hash that does not match.

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
- The test mapping names every kind the 28 corpus studies hold, and each study
  round-trips exactly. B3, the round trip of a run document with an equal
  canonical history on both executors, remains T5b's acceptance check.

## Tests

Added, in `tests/codec/test_document.py` (unit tier, 9 functions, 22 cases):

- the toy round trip is exact: treedef, static part, every leaf's type, dtype,
  shape, weak type and bytes, and the dumped text of the reload equals the
  original's (a two-phase study with a static process field, a bool, a Python
  float, a coupling holding an enum member and weak arrays);
- every corpus study round-trips exactly, so restore validation accepts what
  construction made;
- the reloaded baseline study runs on the host executor with zero trace or compile
  events and the same canonical history (checked by hand against a control with a
  changed static field, which traces 16 programs);
- the content hash computed in a subprocess with another `PYTHONHASHSEED` equals
  the in-process hash;
- NaN, +inf, -inf and `-0.0` survive in a float32 array, in float64 scalars and
  in Python floats, with no bare `NaN` or `Infinity` in the text, and the float32
  dtype is kept;
- dict insertion order is content: opposite orders hash differently and reload in
  their own order, and a stored field that `__post_init__` derives differently is
  refused at that field;
- `>f8` and `>i8` arrays are refused at their pointers;
- a bad document fails with the pointer of the offending value (10 cases): data
  that does not match its declared shape, a field that breaks its module's check,
  an unknown kind (`D001`), a hand-edited hash, a string clock, a fractional
  integer tick, a tuple where an array is expected, `$enum` naming a module,
  `$kind` naming an enum, and `$enum` with no value;
- malformed text fails with a pointer (4 cases): `1e999` and bare `NaN` in a
  hashed document, a key repeated inside a nested object, and truncated JSON.

Retired: none. Net: 9 functions, 22 cases.

## Review fixes

A fresh review found six defects, each fixed in its own commit with its test:
restored fields were not checked against their converters or `__post_init__`; a
dict's insertion order was lost and absent from the hash; `$enum` could name a
module and `$kind` an enum; byte order was dropped silently; a number that
overflows to infinity bypassed the non-finite tags; and malformed input (a tuple
for an array, bad JSON, a nested repeated key) escaped as a raw error or without a
pointer. The review confirmed that no document reaches a class outside the
mapping, and that float formatting, `-0.0`, large integers and canonical output
were correct.

## Timing

Local runs on the four-thread compute launcher, `-n 2 --dist loadfile`:

| Run | Result |
|---|---|
| Codec file alone, persistent cache off | 8 passed, 6.6 s wall; slowest 3.9 s (program reuse, which compiles the host programs once), 1.6 s (cross-process hash), 0.2 s (round trip) |
| Quick tier with the codec tests, warm cache | 1,736 passed, 50.5 s wall, peak RSS 3.2 GB; the codec tests sum to 3.4 s |
| Quick tier without them, warm cache, beside a full-suite run | 1,728 passed, 63.9 s wall |
| Full suite | 2,798 passed, 1,260 s wall, peak RSS 4.1 GB |
| Codec file after the review fixes | 22 passed, 5.0 s wall in one process; slowest 1.8 s (program reuse), 1.7 s (cross-process hash), 0.2 s (all 28 corpus round trips) |
| Quick tier after the review fixes | 1,750 passed, 67.3 s wall, peak RSS 2.9 GB (the machine was busier than for the 50.5 s run) |

The three slowest quick-tier tests are unchanged and outside the codec:
`test_every_hand_study_reproduces_its_history`,
`test_statuses_agree_between_executors_and_replay` and
`test_the_fingerprint_hashes_the_masked_view_not_the_raw_log`; the first two were
over the 5 s unit budget before this change as well.
