"""Installed kinds: entry points in groups `spaceodyssey.<family>`.

A distribution registers each kind it ships as one entry point. The group is
`spaceodyssey.<family>`, the name is `<id>@<version>` and the object is the
class::

    [project.entry-points."spaceodyssey.policy"]
    "toy.fixed_plan@1" = "spaceodyssey.studies.toy:FixedPlanPolicy"

The family is the document group of the kind and names the abstract base its
class must subclass (`FAMILIES`); the `core` group holds only the core's own
kinds (`CORE_KINDS`). An id is lowercase words joined by dots, and the version is
a positive integer equal to the default of the class's `version` field when it
has one. `installed_resolver()` reads the installed entry points and builds the
`KindResolver` the codec takes; the codec itself never imports by name.

An entry point of an optional extra is declared with extras syntax
(`"module:Class [physical]"`). In an install without that extra the entry is skipped
rather than failing the whole table: `installed_missing()` names it, and the
installed resolver refuses it only when a document asks for that kind.

This module sits outside the core, which may not load code dynamically.
"""

import re
from importlib.metadata import PackageNotFoundError, entry_points, version
from importlib.metadata import (
    requires as dist_requires,
)

from spaceodyssey.codec.document import DocumentError, pointer
from spaceodyssey.codec.kinds import (
    CORE_KINDS,
    KindResolver,
    declared_version,
)
from spaceodyssey.codec.kinds import (
    RegistrationError as RegistrationError,
)
from spaceodyssey.core.kinds import (
    AbstractChannelModel,
    AbstractDesign,
    AbstractObjective,
    AbstractObserver,
    AbstractPolicy,
    AbstractProcess,
    AbstractWorldSource,
)

GROUP_PREFIX = "spaceodyssey."

FAMILIES = {
    "core": None,
    "observer": AbstractObserver,
    "policy": AbstractPolicy,
    "process": AbstractProcess,
    "world_source": AbstractWorldSource,
    "channel_model": AbstractChannelModel,
    "design": AbstractDesign,
    "objective": AbstractObjective,
}

_NAME = re.compile(
    r"(?P<id>[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*)@(?P<version>[1-9][0-9]*)"
)


_EXTRA = re.compile(r"""extra\s*==\s*["']([^"']+)["']""")
_REQ_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _origin(entry):
    """The entry point and the distribution it comes from, for messages."""
    dist = getattr(entry.dist, "name", None) or "an unknown distribution"
    return f"{entry.group} {entry.name} = {entry.value} (from {dist})"


def _extras_unmet(entry):
    """The requirements of an entry's declared extras that are not installed."""
    extras = tuple(getattr(entry, "extras", None) or ())
    dist = getattr(entry.dist, "name", None)
    if not extras or not dist:
        return []
    try:
        lines = dist_requires(dist) or []
    except PackageNotFoundError:
        return []
    unmet = []
    for line in lines:
        head, _, marker = line.partition(";")
        if not set(_EXTRA.findall(marker)) & set(extras):
            continue
        name = _REQ_NAME.match(head.strip())
        try:
            version(name[0])
        except PackageNotFoundError:
            unmet.append(name[0])
    return unmet


def scan_entry_points(entries):
    """The registered kinds and the kinds skipped for a missing extra.

    Returns `(kinds, missing)`: `kinds` as `kinds_from_entry_points` documents, and
    `missing` mapping `(group, id, version)` to the extra names that would provide it.
    An entry that declares extras is skipped, not raised, when a requirement of those
    extras is not installed or its load fails with `ModuleNotFoundError` for a module
    outside spaceodyssey; every other defect raises `RegistrationError`.
    """
    kinds = {}
    missing = {}
    seen = {}
    bound = {}
    for entry in entries:
        if not entry.group.startswith(GROUP_PREFIX):
            continue
        family = entry.group.removeprefix(GROUP_PREFIX)
        if family not in FAMILIES:
            raise RegistrationError(
                f"{_origin(entry)}: no family {family!r}; "
                f"families are {sorted(FAMILIES)}"
            )
        match = _NAME.fullmatch(entry.name)
        if match is None:
            raise RegistrationError(
                f"{_origin(entry)}: the name is <id>@<version>, with lowercase "
                "dotted words for the id and a positive integer version"
            )
        ref = (family, match["id"], int(match["version"]))
        if ref[:2] in seen:
            raise RegistrationError(
                f"kind {list(ref[:2])} is registered twice: "
                f"{_origin(seen[ref[:2]])} and {_origin(entry)}"
            )
        seen[ref[:2]] = entry
        extras = tuple(getattr(entry, "extras", None) or ())
        if _extras_unmet(entry):
            missing[ref] = extras
            continue
        try:
            cls = entry.load()
        except ModuleNotFoundError as err:
            top = (err.name or "").partition(".")[0]
            if extras and top != "spaceodyssey":
                missing[ref] = extras
                continue
            raise RegistrationError(f"{_origin(entry)}: cannot load: {err}") from err
        except (ImportError, AttributeError) as err:
            raise RegistrationError(f"{_origin(entry)}: cannot load: {err}") from err
        cls = _checked(entry, ref, cls)
        if cls in bound:
            raise RegistrationError(
                f"{cls.__module__}.{cls.__qualname__} is registered under two ids: "
                f"{_origin(bound[cls])} and {_origin(entry)}"
            )
        bound[cls] = entry
        kinds[ref] = cls
    return kinds, missing


def kinds_from_entry_points(entries):
    """The `(group, id, version)` to class mapping a set of entry points registers.

    Entries outside the `spaceodyssey.` groups are ignored, and entries of an extra
    that is not installed are skipped (see `scan_entry_points`). A group naming no
    family, a malformed name, an object that fails to load, a class of the wrong
    family or version, a core entry that differs from `CORE_KINDS`, and one
    `(group, id)` registered twice or one class registered under two ids (naming
    both entries and their distributions) each raise `RegistrationError`.
    """
    return scan_entry_points(entries)[0]


def _checked(entry, ref, cls):
    """The class of one entry, checked against its family and version."""
    base = FAMILIES[ref[0]]
    if base is None:
        if CORE_KINDS.get(ref) is not cls:
            raise RegistrationError(
                f"{_origin(entry)}: the core group holds only the core kinds"
            )
        return cls
    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise RegistrationError(
            f"{_origin(entry)}: a {ref[0]} kind subclasses {base.__name__}"
        )
    try:
        version = declared_version(cls)
    except ValueError as err:
        raise RegistrationError(f"{_origin(entry)}: {err}") from err
    if version is not None and version != ref[2]:
        raise RegistrationError(
            f"{_origin(entry)}: the class's version field defaults to {version}"
        )
    return cls


def installed_kinds():
    """The kinds the installed distributions register through entry points."""
    return kinds_from_entry_points(entry_points())


def installed_missing():
    """The kinds skipped for a missing extra, as `(group, id, version)` to extras."""
    return scan_entry_points(entry_points())[1]


class InstalledResolver(KindResolver):
    """A `KindResolver` that names the extra when a skipped kind is requested."""

    def __init__(self, kinds=None, missing=None):
        """Resolve over `kinds`; `missing` maps skipped references to their extras."""
        super().__init__(kinds)
        self._missing = {
            ref[:2]: (ref, extras) for ref, extras in (missing or {}).items()
        }

    def resolve(self, ref, path=()):
        """The class a reference names; a kind of a missing extra says to install it."""
        hit = self._missing.get(tuple(ref[:2]))
        if hit is not None and self.current_version(*ref[:2]) is None:
            extras = ",".join(hit[1])
            raise DocumentError(
                f"kind {list(ref)} needs the optional install `spaceodyssey[{extras}]`",
                pointer(path),
                code="D001",
            )
        return super().resolve(ref, path)


def installed_resolver():
    """A `KindResolver` over the installed kinds."""
    kinds, missing = scan_entry_points(entry_points())
    return InstalledResolver(kinds, missing)
