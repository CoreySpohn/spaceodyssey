"""The in-process record-kind registry and its lowering into capacities."""

import pytest

from spaceodyssey.core import (
    Audience,
    Capacities,
    KindRegistry,
    PayloadColumn,
    RecordKind,
    payload_columns,
)
from spaceodyssey.studies import toy


def _kind(name, width, audience=Audience.OBSERVER | Audience.POLICY):
    return RecordKind(
        name, tuple(PayloadColumn(f"c{i}", audience) for i in range(width))
    )


def test_two_kinds_of_different_widths_lower_to_one_width():
    registry = KindRegistry((_kind("narrow", 2), _kind("wide", 6)))
    caps = registry.lower(Capacities())
    assert caps.payload_width == 6
    assert [len(row) for row in caps.payload_audience] == [6, 6]
    # the narrow kind declares nothing past its own columns: those are unreadable
    assert caps.payload_audience[0][2:] == (0, 0, 0, 0)
    assert registry.width == 6 and len(registry) == 2


def test_lowering_never_narrows_the_declared_width():
    caps = KindRegistry((_kind("narrow", 2),)).lower(Capacities(payload_width=5))
    assert caps.payload_width == 5


@pytest.mark.parametrize("audience", [None, 0, 8, 16])
def test_a_column_without_a_declared_audience_fails_at_construction(audience):
    with pytest.raises(ValueError):
        PayloadColumn("value", audience)
    with pytest.raises(TypeError):
        PayloadColumn("value")


def test_kind_construction_checks_its_columns():
    with pytest.raises(ValueError):
        RecordKind("bad", ("value",))
    with pytest.raises(ValueError):
        RecordKind("bad", (PayloadColumn("a", 1), PayloadColumn("a", 2)))


def test_registry_indices_and_duplicates():
    registry = KindRegistry()
    assert registry.register(_kind("a", 1)) == 0
    assert registry.register(_kind("b", 1)) == 1
    assert registry.index("b") == 1 and registry[0].name == "a"
    with pytest.raises(ValueError):
        registry.register(_kind("a", 3))
    with pytest.raises(TypeError):
        registry.register("a")
    with pytest.raises(KeyError):
        registry.index("missing")


def test_toy_registry_matches_the_record_numbers():
    registry = toy.toy_registry()
    assert registry.index("measurement") == toy.RECORD_MEASUREMENT
    assert registry.index("auxiliary") == toy.RECORD_AUXILIARY
    caps = registry.lower(Capacities())
    assert caps.payload_width == 4


def test_column_readability_by_audience():
    caps = toy.toy_registry().lower(Capacities())
    kind = toy.RECORD_MEASUREMENT
    for audience, expected in (
        (Audience.POLICY, [True, False, True, False]),
        (Audience.OBSERVER, [True, False, True, False]),
        (Audience.SCORER, [False, True, False, False]),
    ):
        assert list(map(bool, payload_columns(caps, kind, audience))) == expected
    # a kind the registry lacks exposes nothing, and an unlowered caps everything
    assert not payload_columns(caps, 9, Audience.POLICY).any()
    assert payload_columns(Capacities(), 9, Audience.POLICY).all()
