"""The installed entry points: every registered kind loads and round-trips."""

import tomllib
from importlib.metadata import entry_points
from pathlib import Path

from spaceodyssey.codec import CORE_KINDS, dump_study, load_study
from spaceodyssey.plugins import GROUP_PREFIX, installed_kinds, installed_resolver
from spaceodyssey.studies.conformance import CORPUS
from tests.codec.test_document import assert_same_pytree, rich_study

PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"


def kind_refs(node):
    """Every `$kind` and `$enum` reference in an encoded value."""
    if isinstance(node, list):
        return set().union(*map(kind_refs, node)) if node else set()
    if not isinstance(node, dict):
        return set()
    refs = {tuple(node[tag]) for tag in ("$kind", "$enum") if tag in node}
    return refs.union(*map(kind_refs, node.values()))


def test_installed_kinds_round_trip_every_corpus_study():
    """Fails if the installed entry points differ from the pyproject table (stale
    metadata), if an installed entry fails to load or validate, if the core group
    differs from CORE_KINDS, if a corpus study (or the codec's rich toy study) holds a
    kind with no entry point or does not round-trip through the installed resolver,
    or if a registered kind is held by none of them and so is never round-tripped.
    """
    table = tomllib.loads(PYPROJECT.read_text())["project"]["entry-points"]
    declared = {
        (group, name, value)
        for group, items in table.items()
        if group.startswith(GROUP_PREFIX)
        for name, value in items.items()
    }
    installed = {
        (entry.group, entry.name, entry.value)
        for entry in entry_points()
        if entry.group.startswith(GROUP_PREFIX) and entry.dist.name == "spaceodyssey"
    }
    assert installed == declared, "installed metadata is stale; reinstall the package"

    kinds = installed_kinds()
    assert {ref: cls for ref, cls in kinds.items() if ref[0] == "core"} == CORE_KINDS
    resolver = installed_resolver()
    used = set()
    for study in (rich_study(), *(case.make_study() for case in CORPUS)):
        doc = dump_study(study, resolver)
        assert_same_pytree(study, load_study(doc.to_json(), resolver))
        used |= kind_refs(doc.study)
    assert set(kinds) - set(CORE_KINDS) <= used
