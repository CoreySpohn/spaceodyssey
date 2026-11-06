"""The record-kind registry in a run document.

`Study` keeps only the tables a registry lowers into `Capacities` (kind names,
payload audiences, derived kinds), and `KindRegistry.lower` drops the column names.
A run document therefore carries the registry itself, passed to the codec explicitly,
as a list in registry order: `{"name", "derived", "columns": [[name, audience]]}`,
with each audience an integer made of `Audience` flags. The loader rebuilds the
registry and checks that lowering it with the study's observer gives exactly the
study's capacities, so the registry and the study cannot drift apart.
"""

from spaceodyssey.codec.document import DocumentError, pointer
from spaceodyssey.core.records import KindRegistry, PayloadColumn, RecordKind

_KIND_KEYS = frozenset({"name", "derived", "columns"})


def encode_records(registry):
    """The JSON form of a registry, in registry order."""
    return [
        {
            "name": kind.name,
            "derived": kind.derived,
            "columns": [[c.name, int(c.audience)] for c in kind.columns],
        }
        for kind in (registry[i] for i in range(len(registry)))
    ]


def decode_records(node, path=("records",)):
    """A `KindRegistry` from its JSON form; a malformed entry fails at its pointer."""
    if type(node) is not list:
        raise DocumentError("records is a list of record kinds", pointer(path))
    registry = KindRegistry()
    for i, entry in enumerate(node):
        where = (*path, i)
        if type(entry) is not dict or set(entry) != _KIND_KEYS:
            raise DocumentError(
                "a record kind is {name, derived, columns}", pointer(where)
            )
        if type(entry["name"]) is not str or type(entry["derived"]) is not bool:
            raise DocumentError(
                "a record kind has a string name and a bool derived flag",
                pointer(where),
            )
        columns = entry["columns"]
        ok = type(columns) is list and all(
            type(c) is list and len(c) == 2 and type(c[0]) is str and type(c[1]) is int
            for c in columns
        )
        if not ok:
            raise DocumentError(
                "columns is a list of [name, audience]", pointer((*where, "columns"))
            )
        try:
            kind = RecordKind(
                entry["name"],
                tuple(PayloadColumn(n, a) for n, a in columns),
                derived=entry["derived"],
            )
            registry.register(kind)
        except ValueError as err:
            raise DocumentError(str(err), pointer(where)) from err
    return registry


def check_lowering(registry, study, path=("records",)):
    """Require that the registry lowers, with the study's observer, to its caps."""
    try:
        lowered = registry.lower(study.caps, study.observer)
    except ValueError as err:
        raise DocumentError(
            f"the registry does not lower: {err}", pointer(path)
        ) from err
    if lowered != study.caps:
        raise DocumentError(
            "the record-kind registry does not lower to the study's capacities "
            "(kind names, payload audiences, derived kinds or refit kinds differ)",
            pointer(path),
        )
