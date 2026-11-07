"""Kind references: `(group, id, version)` to a class and back, and migration.

A document names every module, dataclass and enum it holds by a kind reference.
`KindResolver` maps references to classes from a mapping it is given; it never
imports anything a document names. Discovery of installed kinds (entry points)
lives in `spaceodyssey.plugins`, which builds a resolver.

Each `(group, id)` has one registered version, its current one. A document
reference at that version resolves directly. A newer version is refused as
`D001`. An older version resolves to the same class and its stored fields pass
through the class's `migrate(from_version, params)` classmethod, which returns
the fields of the current version; a class without `migrate` cannot read an older
version, which is also `D001`.
"""

import dataclasses
import enum
from collections.abc import Mapping

from spaceodyssey.codec.document import DocumentError, pointer
from spaceodyssey.core import enums
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.keys import Coupling, Fork
from spaceodyssey.core.ledger import AssetTable, ChannelTable
from spaceodyssey.core.refresh import RefreshSchedule
from spaceodyssey.core.snapshot import EffectTable
from spaceodyssey.core.study import Agenda, Clock, Study

KindRef = tuple[str, str, int]

CORE_KINDS: Mapping[KindRef, type] = {
    ("core", "study", 1): Study,
    ("core", "clock", 1): Clock,
    ("core", "agenda", 1): Agenda,
    ("core", "channel_table", 1): ChannelTable,
    ("core", "asset_table", 1): AssetTable,
    ("core", "capacities", 1): Capacities,
    ("core", "coupling", 1): Coupling,
    ("core", "fork", 1): Fork,
    ("core", "refresh_schedule", 1): RefreshSchedule,
    ("core", "effect_table", 1): EffectTable,
    ("core", "stage", 1): enums.Stage,
    ("core", "event_kind", 1): enums.EventKind,
    ("core", "effect_kind", 1): enums.EffectKind,
    ("core", "scope", 1): enums.Scope,
    ("core", "disposition", 1): enums.Disposition,
    ("core", "record_status", 1): enums.RecordStatus,
    ("core", "purpose", 1): enums.Purpose,
    ("core", "admission_code", 1): enums.AdmissionCode,
    ("core", "coincident_order", 1): enums.CoincidentOrder,
    ("core", "charge_rule", 1): enums.ChargeRule,
    ("core", "remainder_rule", 1): enums.RemainderRule,
    ("core", "audience", 1): enums.Audience,
    ("core", "capacity", 1): enums.Capacity,
    ("core", "run_status", 1): enums.RunStatus,
}


class RegistrationError(ValueError):
    """A kind registration that is broken: the installation, not a document."""


def is_kind_class(cls):
    """Whether `$kind` may name `cls`: a dataclass (or module) type, not an enum."""
    return (
        isinstance(cls, type)
        and dataclasses.is_dataclass(cls)
        and not issubclass(cls, enum.Enum)
    )


def is_version_field(cls, name):
    """Whether field `name` of a kind class is the kind version."""
    return (
        name == "version" and isinstance(cls, type) and not issubclass(cls, enum.Enum)
    )


def is_optional_field(field):
    """Whether a dataclass field is optional in a document (`OPTIONAL` metadata).

    The encoder omits an optional field at its default and the loader fills it when
    it is missing, so a document written before the field existed keeps its hash.
    The default must be a plain value (a bool, int, float, str, or a tuple of them),
    so that "at its default" has one meaning.
    """
    return bool(field.metadata.get("optional"))


def optional_default(field, cls=None):
    """The default of an optional field; a registration error if it is not plain."""
    default = field.default
    if default is dataclasses.MISSING or not _plain(default):
        raise RegistrationError(
            f"optional field {field.name!r}"
            f"{'' if cls is None else ' of ' + _name(cls)} needs a plain default"
        )
    return default


def _plain(value):
    if type(value) is tuple:
        return all(map(_plain, value))
    return type(value) in (bool, int, float, str)


def at_default(field, value, cls=None):
    """Whether `value` is the default of an optional field, in type and content."""
    default = optional_default(field, cls)
    if type(value) is not type(default):
        return False
    if type(value) is float:
        return repr(value) == repr(default)
    return value == default


def declared_version(cls):
    """The default of a kind class's `version` field, or `None` if it has none.

    A `version` field without a positive integer default (a factory, a missing
    default, a value of another type) raises `ValueError`: a default instance of
    such a class has no version a registration can match.
    """
    if not is_kind_class(cls):
        return None
    for field in dataclasses.fields(cls):
        if field.name == "version":
            if type(field.default) is not int or field.default < 1:
                raise ValueError(
                    f"the version field of {_name(cls)} needs a positive integer "
                    "default"
                )
            return field.default
    return None


def check_ref(ref):
    """A kind reference as a `(str, str, int)` tuple with a positive version."""
    ok = (
        isinstance(ref, (tuple, list))
        and len(ref) == 3
        and isinstance(ref[0], str)
        and isinstance(ref[1], str)
        and type(ref[2]) is int
        and ref[2] >= 1
    )
    if not ok:
        raise ValueError(f"a kind reference is (group, id, version >= 1), not {ref!r}")
    return tuple(ref)


def _name(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


class KindResolver(Mapping):
    """Kind references both ways, from `CORE_KINDS` plus an injected mapping.

    `kinds` maps `(group, id, version)` to a class, each `(group, id)` at its
    current version. The resolver refuses, with `ValueError`, a key that rebinds a
    core key to another class, one `(group, id)` at two versions, one class under
    two references, and a reference whose version differs from the default of the
    class's `version` field. As a mapping it holds every registered reference.
    """

    def __init__(self, kinds=None):
        """Merge `kinds` over the core kinds and check the registrations."""
        table = dict(CORE_KINDS)
        for ref, cls in (kinds or {}).items():
            ref = check_ref(ref)
            if table.get(ref, cls) is not cls:
                raise ValueError(f"kind {list(ref)} is already bound to another class")
            table[ref] = cls
        current = {}
        bound = {}
        for ref, cls in table.items():
            key = ref[:2]
            if key in current:
                raise ValueError(
                    f"kind {list(key)} is registered at versions "
                    f"{current[key]} and {ref[2]}; register only the current one"
                )
            current[key] = ref[2]
            if cls in bound:
                raise ValueError(
                    f"{_name(cls)} is bound to both {list(bound[cls])} and {list(ref)}"
                )
            bound[cls] = ref
            version = declared_version(cls)
            if version is not None and version != ref[2]:
                raise ValueError(
                    f"kind {list(ref)} is bound to {_name(cls)}, whose version "
                    f"field defaults to {version}"
                )
        self._by_ref = table
        self._current = current

    def __getitem__(self, ref):
        """The class registered under exactly `ref`."""
        return self._by_ref[tuple(ref)]

    def __iter__(self):
        """Every registered reference."""
        return iter(self._by_ref)

    def __len__(self):
        """The number of registered references."""
        return len(self._by_ref)

    def current_version(self, group, kind_id):
        """The registered version of `(group, kind_id)`, or `None` if unknown."""
        return self._current.get((group, kind_id))

    def resolve(self, ref, path=()):
        """The class a document reference names, at its version or an older one.

        An unknown `(group, id)`, a version newer than the registered one, and an
        older version whose class defines no `migrate` are `D001` at `path`.
        """
        current = self.current_version(ref[0], ref[1])
        if current is None or ref[2] < 1:
            raise DocumentError(f"unknown kind {list(ref)}", pointer(path), code="D001")
        cls = self._by_ref[(ref[0], ref[1], current)]
        if ref[2] > current:
            raise DocumentError(
                f"kind {list(ref)} is newer than the registered version {current}",
                pointer(path),
                code="D001",
            )
        if ref[2] < current and not callable(getattr(cls, "migrate", None)):
            raise DocumentError(
                f"kind {list(ref)} is older than the registered version {current} "
                f"and {_name(cls)} defines no migrate",
                pointer(path),
                code="D001",
            )
        return cls

    def migrate(self, ref, params, path=()):
        """The fields of the current version, from the fields stored at `ref`.

        `params` maps field names to decoded values. At the current version they
        are returned unchanged; at an older one the class's
        `migrate(from_version, params)` returns them. A failure inside `migrate`,
        or a result that is not a dict of field names, is a `DocumentError` at
        `path`.
        """
        cls = self.resolve(ref, path)
        if ref[2] == self.current_version(ref[0], ref[1]):
            return params
        try:
            out = cls.migrate(ref[2], dict(params))
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as err:
            raise DocumentError(
                f"{cls.__qualname__}.migrate from version {ref[2]}: {err}",
                pointer(path),
            ) from err
        if type(out) is not dict or not all(type(k) is str for k in out):
            raise DocumentError(
                f"{cls.__qualname__}.migrate from version {ref[2]} returns a "
                f"{type(out).__name__}, not a dict of fields",
                pointer(path),
            )
        return out

    def ref_of(self, value, path=()):
        """The reference of an instance's class, checked against its version field."""
        cls = type(value)
        refs = [ref for ref, c in self._by_ref.items() if c is cls]
        if not refs:
            raise DocumentError(f"no kind is bound to {_name(cls)}", pointer(path))
        version = _version_of(value)
        if version is not None and version != refs[0][2]:
            raise DocumentError(
                f"{_name(cls)} has version {version}, but its kind "
                f"{list(refs[0])} is registered at version {refs[0][2]}",
                pointer(path),
            )
        return refs[0]


def as_resolver(kinds):
    """A `KindResolver` from a resolver, a mapping of references, or `None`."""
    if isinstance(kinds, KindResolver):
        return kinds
    return KindResolver(kinds)


def _version_of(value):
    """The `version` field of a dataclass instance, or `None` if it has none."""
    if isinstance(value, enum.Enum) or not dataclasses.is_dataclass(value):
        return None
    if not any(f.name == "version" for f in dataclasses.fields(value)):
        return None
    return value.version
