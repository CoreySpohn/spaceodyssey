"""The installed entry points: every registered kind loads and round-trips."""

import dataclasses
import tomllib
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from spaceodyssey import plugins
from spaceodyssey.codec import (
    CORE_KINDS,
    DocumentError,
    KindResolver,
    dump_study,
    load_study,
)
from spaceodyssey.plugins import (
    GROUP_PREFIX,
    InstalledResolver,
    RegistrationError,
    installed_kinds,
    installed_resolver,
    scan_entry_points,
)
from spaceodyssey.studies.conformance import CORPUS
from tests.codec.test_document import assert_same_pytree, rich_study
from tests.codec.test_plugins import Dist, Entry

PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"
PHYSICAL = ("channel_model", "jaxedith.summary_channel", 2)


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
    metadata), if an installed entry fails to load or validate, if a corpus study
    (or the codec's rich toy study) holds a kind with no entry point or does not
    round-trip through the installed resolver, or if a registered kind is held by none
    of them and so is never round-tripped.
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
    resolver = installed_resolver()
    used = set()
    studies = [rich_study(), *(case.make_study() for case in CORPUS)]
    if PHYSICAL in kinds:
        # The physical extra is present: its channel kind round-trips by id as well.
        from spaceodyssey.adapters.jaxedith import DetectorSpec, JaxedithSummaryChannel

        channel = JaxedithSummaryChannel(
            profile="first-summary-v1",
            target=0,
            detector=dataclasses.astuple(DetectorSpec.fixture()),
            n_time_nodes=8,
            n_exozodi=3.0,
            law=(5.0, 1.0, 0.0),
            windows=((0, 10, 0),),
            calibration=1,
        )
        studies.append(dataclasses.replace(rich_study(), channel_model=channel))
    for study in studies:
        doc = dump_study(study, resolver)
        assert_same_pytree(study, load_study(doc.to_json(), resolver))
        used |= kind_refs(doc.study)
    assert set(kinds) <= used


@dataclasses.dataclass(frozen=True)
class ExtraEntry(Entry):
    """An entry point declared with extras, as a distribution's metadata gives it."""

    extras: tuple = ("physical",)


def installed_entries():
    """The spaceodyssey entries as installed, extras and all."""
    return [
        entry
        for entry in entry_points()
        if entry.group.startswith(GROUP_PREFIX) and entry.dist.name == "spaceodyssey"
    ]


def test_the_physical_entry_declares_its_extra():
    """Fails if the pyproject entry of the physical channel drops the extras syntax,
    which is what lets a plain install skip it instead of failing every kind."""
    table = tomllib.loads(PYPROJECT.read_text())["project"]["entry-points"]
    value = table["spaceodyssey.channel_model"]["jaxedith.summary_channel@2"]
    assert value.endswith("[physical]")


def test_a_plain_install_skips_a_kind_of_a_missing_extra(monkeypatch):
    """Fails if a kind whose extra is not installed (its module missing, or a
    requirement of the extra absent) makes `installed_kinds()` raise instead of being
    skipped, if the resolver of such an install is not told to install the extra only
    when that kind is requested, if the other kinds stop resolving, or if an entry
    without extras that fails to import is skipped instead of raising."""
    entries = [e for e in installed_entries() if e.name != "jaxedith.summary_channel@2"]
    core = scan_entry_points(entries)[0]
    gone = ModuleNotFoundError("no skyscapes", name="skyscapes")
    by_module = ExtraEntry(
        "spaceodyssey.channel_model", "jaxedith.summary_channel@2", gone
    )
    kinds, missing = scan_entry_points([*entries, by_module])
    assert kinds == core and missing == {PHYSICAL: ("physical",)}

    # A requirement of the declared extra that is not installed skips it unloaded.
    monkeypatch.setattr(
        plugins, "dist_requires", lambda dist: ["not-a-package>=1; extra == 'physical'"]
    )
    never = ExtraEntry(
        "spaceodyssey.channel_model",
        "jaxedith.summary_channel@2",
        AssertionError("loaded"),
    )
    assert scan_entry_points([*entries, never]) == (core, missing)
    monkeypatch.undo()

    resolver = InstalledResolver(kinds, missing)
    assert isinstance(resolver, KindResolver) and PHYSICAL not in resolver
    assert resolver.resolve(("core", "clock", 1)) is CORE_KINDS[("core", "clock", 1)]
    with pytest.raises(DocumentError, match=r"spaceodyssey\[physical\]") as err:
        resolver.resolve(PHYSICAL)
    assert err.value.code == "D001"
    with pytest.raises(DocumentError, match="unknown kind") as other:
        resolver.resolve(("channel_model", "no.such", 1))
    assert "spaceodyssey[" not in str(other.value)

    plain = Entry(
        "spaceodyssey.channel_model",
        "jaxedith.summary_channel@2",
        gone,
        Dist("spaceodyssey"),
    )
    with pytest.raises(RegistrationError, match="cannot load"):
        scan_entry_points([*entries, plain])
