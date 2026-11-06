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

This module sits outside the core, which may not load code dynamically.
"""

import re
from importlib.metadata import entry_points

from spaceodyssey.codec.kinds import CORE_KINDS, KindResolver, declared_version
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


class RegistrationError(ValueError):
    """An entry point that does not register a kind correctly."""


def _origin(entry):
    """The entry point and the distribution it comes from, for messages."""
    dist = getattr(entry.dist, "name", None) or "an unknown distribution"
    return f"{entry.group} {entry.name} = {entry.value} (from {dist})"


def kinds_from_entry_points(entries):
    """The `(group, id, version)` to class mapping a set of entry points registers.

    Entries outside the `spaceodyssey.` groups are ignored. A group naming no
    family, a malformed name, an object that fails to load, a class of the wrong
    family or version, a core entry that differs from `CORE_KINDS`, and one
    `(group, id)` registered twice (naming both entries and their distributions)
    each raise `RegistrationError`.
    """
    kinds = {}
    seen = {}
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
        try:
            cls = entry.load()
        except (ImportError, AttributeError) as err:
            raise RegistrationError(f"{_origin(entry)}: cannot load: {err}") from err
        kinds[ref] = _checked(entry, ref, cls)
    return kinds


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


def installed_resolver():
    """A `KindResolver` over the installed kinds."""
    return KindResolver(installed_kinds())
