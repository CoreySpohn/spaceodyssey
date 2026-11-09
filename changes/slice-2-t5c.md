---
orphan: true
---

# Slice 2 T5c: kind resolution, entry points and version migration

The codec now resolves kinds through `KindResolver`, which replaces T5a's
`KindTable` behind the same `kinds=` argument of `dump_study` and `load_study`. A
new module outside the core, `spaceodyssey.plugins`, reads entry points in groups
`spaceodyssey.<family>` and builds a resolver; the package registers its own
kinds there. Every kind has one registered version: a document at a newer kind
version is refused with `D001`, a newer schema with `D002`, and an older kind
version is rebuilt through the class's `migrate(from_version, params)`. No file
under `core/` changed and no run behavior changed: the golden manifest is
untouched.

## Commits

1. `feat(codec): resolve kinds through KindResolver and migrate older kind versions`
2. `feat(plugins): discover kinds from spaceodyssey entry points`
3. `test(core): add lint negative controls for plugins and the codec`
4. `docs(changes): record Slice 2 T5c`
5. `fix(codec): require a positive integer version default for every registered kind`
6. `fix(plugins): refuse one class registered under two entry-point ids`
7. `test(codec): check migrated fields against annotations, converters and hooks`
8. `docs(changes): record the T5c review fixes`

## KindResolver

`spaceodyssey.codec.kinds` (exported as `spaceodyssey.codec.KindResolver`; `CORE_KINDS`
moved here from `document.py` and is still exported from `spaceodyssey.codec`).

- `KindResolver(kinds=None)`: `kinds` maps `(group, id, version)` to a class and
  is merged over `CORE_KINDS`. Construction raises `ValueError` for a key that
  rebinds a core key to another class, one `(group, id)` at two versions, one
  class under two references, a class whose `version` field has no positive
  integer default (a factory, a missing default, a value of another type), and a
  reference whose version differs from that default. The resolver is a read-only mapping of
  every registered reference.
- `current_version(group, kind_id)`: the registered version, or `None`.
- `resolve(ref, path=())`: the class a document reference names. An unknown
  `(group, id)`, a version below 1 or above the registered one, and an older
  version whose class defines no `migrate` are each a `DocumentError` with code
  `D001` at the reference's pointer. It never imports anything a document names.
- `migrate(ref, params, path=())`: `params` (decoded field values) unchanged at
  the current version; at an older one, the result of
  `cls.migrate(from_version, params)`, which must be a dict of field names. An
  error inside `migrate` is a `DocumentError` at the module's pointer.
- `ref_of(value, path=())`: the reference of an instance's class; an instance
  whose `version` field differs from the registered version is refused.

`dump_study` and `load_study` accept a `KindResolver` or a mapping, which they
wrap. `None` still means the core kinds only: the codec takes its kinds as an
argument and never reads the installed entry points itself.

## Versions and migration

A kind's version is the default of its class's `version` field, which every
abstract kind declares (default 1), and its entry point names the same version.
When a class bumps its version it keeps one registration, at the new version, and
defines a classmethod `migrate(cls, from_version, params)` that returns the
fields of the current version from those stored at `from_version`. `params` holds the stored field values already decoded (arrays, tuples, nested
modules, which have been resolved and migrated themselves). The loader checks the
returned fields against the class exactly as it checks stored ones (field names,
converters, array annotations, `__post_init__`, `__check_init__`), and sets the
`version` field to the registered version, so a migrated module dumps at the new
version. A migration that returns other field names is refused at the module's
pointer.

## Entry points

`pyproject.toml` now has an entry-points table. The group is
`spaceodyssey.<family>`, the name `<id>@<version>` and the object the class.
`spaceodyssey.plugins`:

- `FAMILIES`: `core` and the seven module families, each with the abstract base
  its classes must subclass: `observer`, `policy`, `process`, `world_source`,
  `channel_model`, `design`, `objective`. The family is the document group.
- `kinds_from_entry_points(entries)`: the `(group, id, version)` to class mapping
  a set of entry points registers; entries outside the `spaceodyssey.` groups are
  ignored. A group with no family, a malformed name, an object that fails to load,
  a class of another family, a class whose `version` field has no positive integer
  default, a version the class does not declare, a core entry that differs from
  `CORE_KINDS`, one `(group, id)` registered twice, and one class registered under
  two ids each raise `RegistrationError` (a `ValueError`). Both duplicate errors
  name both entries and both distributions, so an installation that builds a
  mapping here never fails later in `KindResolver` with a plain `ValueError`.
- `installed_kinds()` and `installed_resolver()`: the same over
  `importlib.metadata.entry_points()`.

Registered by this package (37 entries, all at version 1):

| Group | Ids |
|---|---|
| `spaceodyssey.core` | the 23 `CORE_KINDS`: `study`, `clock`, `agenda`, `channel_table`, `asset_table`, `capacities`, `coupling`, `fork`, `effect_table` and the 14 core enums |
| `spaceodyssey.observer` | `toy.running_mean`, `catalog.observer` |
| `spaceodyssey.policy` | `toy.fixed_plan`, `toy.threshold_stop`, `toy.spin_wait`, `toy.null`, `oracle.query` |
| `spaceodyssey.process` | `toy.null`, `toy.heartbeat`, `scripted.records` |
| `spaceodyssey.world_source` | `toy.uniform_signal` |
| `spaceodyssey.channel_model` | `toy.photometry` |
| `spaceodyssey.design` | `toy.gain` |
| `spaceodyssey.objective` | `toy.threshold` |

These are the kinds the 28 corpus studies and the codec's rich toy study hold,
the same set as T5a's test mapping less its test-only design. Not registered: the
toy policies no corpus study holds (`ReactivePolicy`, `StochasticPolicy`,
`ReplayPolicy`), which therefore have no round trip to pass, and survey 25's
`NetworkPolicy`, whose `eqx.nn.MLP` holds activation functions the codec refuses.

Helper types a plugin kind may hold in its fields (its own enums or nested
dataclasses) have no family, so a plugin cannot register them yet; the first
such kind adds one.

## Tests

Added, in the unit tier:

- `tests/codec/test_plugins.py` (6 functions, 22 cases):
  - `test_load_by_id_without_path_import`: entry points build a resolver that
    round-trips survey 1 by id; an entry in another group is never loaded; a
    document whose kind id is an importable module path fails with `D001` at the
    reference and the module stays unimported.
  - `test_older_kind_version_migrates_to_the_registered_one`: a hand-written
    version-1 `toy.gain` design loads into a version-2 kind that stores the log of
    the gain, with the migrated bits and version 2, and dumps back at version 2.
  - `test_unreadable_kind_version_fails_with_a_pointer` (6 cases): a newer kind
    version and an older one with no `migrate` are `D001` at the reference; a
    migration returning the old field names, a tuple in an array field, a value its
    converter would change, or a value the class's `__check_init__` refuses is
    refused at the module.
  - `test_bad_registration_is_refused` (6 cases): one id from two distributions
    and one class under two ids (each error names both entries and both
    distributions), an unknown family, a name without a version, a class of
    another family, a core entry bound to another class.
  - `test_version_must_be_a_matching_integer_default` (6 cases): through both the
    resolver and the entry points, a version its class does not declare, a
    `version` field with a factory, and one with a string default are refused.
  - `test_resolver_refuses_a_double_binding` (2 cases): one id at two versions,
    one class under two ids.
- `tests/codec/test_entry_points_installed.py` (1 function): the installed
  `spaceodyssey` entry points equal the pyproject table (stale metadata fails with
  a message to reinstall); every installed entry loads and validates; the core
  group equals `CORE_KINDS`; the rich toy study and every corpus study round-trip
  exactly through the installed resolver; and every registered kind appears in one
  of those documents, so each is round-tripped.
- `tests/codec/test_document.py`: a `newer-schema` case (`D002` at
  `/schema_version`) joins `test_bad_document_fails_with_a_pointer`.
- `tests/core/test_lints.py`: four cases join
  `test_core_imports_rejects_outside_boundary` (a core import of `plugins`, of the
  codec by absolute and relative path, and of an adapter), and
  `test_plugins_and_codec_are_allowed_only_outside_the_core` copies the real
  `plugins.py` into a core tree, where the import lint refuses its entry-point
  loader (`importlib.metadata`), and the codec, where the dispatch gate refuses its
  field walk, while the repository core lint passes with both outside the core.

Retired: none. Net: 8 functions, 29 cases.

The installed test runs the round trip only. The observer kit
(`core/conformance/observer_kit.py`, T3a) is on its own branch and not yet on
main; once it is, the installed test runs it on each installed kind of the
`observer` family, the only family with a kit.

### The installed test in a worktree

Entry points are read from the installed distribution metadata, not from the
source tree. A library worktree runs on the shared environment, whose metadata
belongs to the main checkout, so before this branch merges the installed test
there fails with "installed metadata is stale; reinstall the package". It was run
here with the branch's own metadata first on the path: build a wheel of the
worktree (`uv build --wheel`), unpack only its `.dist-info` directory into a
scratch directory, and put that directory on `PYTHONPATH`. In CI the package is
installed from the branch, and after the merge `uv sync` reinstalls it because
`pyproject.toml` changed.

## Review fixes

A fresh review found three gaps, each fixed in its own commit with its test:

- A `version` field with a factory or a non-integer default was read as no
  version, so such a class registered at any version (a factory giving 2 was
  accepted at version 7) and its default instance could not be dumped. Both
  registration paths now require a positive integer default; the new test failed
  on the four factory and string cases before the fix.
- Two entry-point ids naming one class passed `kinds_from_entry_points`, and
  `installed_resolver` then failed with a plain `ValueError` and no distributions.
  Discovery now refuses it with a `RegistrationError` naming both entries and
  distributions; the new case failed before the fix.
- The checks on migrated fields (array annotation, converter, `__post_init__` and
  `__check_init__`) were right but untested: a mutant skipping them for migrated
  modules passed every codec test. Three migration cases now fail against that
  mutant, checked by applying it.

## What later tasks need from this

- T6 (JSON pre-flight): build the resolver once (`installed_resolver()`, or an
  explicit mapping in tests) and pass it in; walk the document's `$kind` and
  `$enum` references and call `resolve(ref, path)` on each to collect every `D001`
  with its pointer, rather than stopping at the first, as `load_study` does;
  `check_schema_version` gives `D002`. A missing id can never surface as
  `ImportError`, because the resolver imports nothing. A `RegistrationError` is a
  broken installation, raised when the resolver is built, not a document finding.
- T9 (legacy deletion): the resolver and entry points replace `registry.py` and
  the qualified-name aliases for documents; no new code imports any legacy module.
  The public-API test can import `spaceodyssey.codec.KindResolver`,
  `spaceodyssey.plugins.installed_resolver`, `kinds_from_entry_points`,
  `RegistrationError` and `FAMILIES`.

## Timing

Local runs on the four-thread compute launcher, `-n 2 --dist loadfile`, with the
branch's metadata on the path as above:

| Run | Result |
|---|---|
| Quick tier on main (dbef55a), back to back with the next row | 1,750 passed, 62.2 s wall, peak RSS 3.27 GB |
| Quick tier on this branch | 1,770 passed, 55.0 s wall, peak RSS 3.35 GB |
| Quick tier on this branch, earlier, beside two other jobs | 1,770 passed, 121.8 s wall, peak RSS 3.08 GB |
| Full suite | 2,832 passed, 1,553 s wall, peak RSS 4.11 GB |
| Quick tier after the review fixes, beside a quick-tier run in the other slot | 1,779 passed, 63.7 s wall, peak RSS 2.79 GB |

The quick tier is flat: the first 20 new cases sum to about 0.9 s (the installed round
trip 0.49 s, the lint control 0.27 s, everything else under 0.1 s each). The
three slowest quick-tier tests are unchanged and outside this change:
`test_every_hand_study_reproduces_its_history`,
`test_statuses_agree_between_executors_and_replay` and `test_gain_enters_measurement`.
