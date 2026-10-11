"""Tests for contract-type resolution and port compatibility."""

import pytest

from spaceodyssey.contracts import compatible, resolve_contract
from spaceodyssey.errors import WiringError


class Base:
    """Toy contract base type."""


class Derived(Base):
    """Toy contract subtype."""


class Other:
    """Toy contract type unrelated to Base."""


def qual(cls) -> str:
    """Module-qualified contract name for a class defined in this module."""
    return f"{__name__}.{cls.__name__}"


class TestResolveContract:
    """resolve_contract maps qualified names to classes with clear errors."""

    def test_resolves_public_type(self):
        """A real public type resolves to the class object."""
        from planit_py.vocabulary import Action

        assert resolve_contract("planit_py.Action") is Action

    def test_unqualified_name_rejected(self):
        """Bare names are rejected with guidance."""
        with pytest.raises(WiringError, match="module-qualified"):
            resolve_contract("SummaryDataset")

    def test_missing_package(self):
        """A missing package is reported as unavailable, naming the module."""
        with pytest.raises(WiringError, match="no_such_pkg_qq"):
            resolve_contract("no_such_pkg_qq.Thing")

    def test_missing_attribute(self):
        """A missing attribute on a real module is reported."""
        with pytest.raises(WiringError, match="NoSuchThing"):
            resolve_contract("planit_py.NoSuchThing")

    def test_non_class_rejected(self):
        """Resolving to a non-class object is an error."""
        with pytest.raises(WiringError, match="not a class"):
            resolve_contract("spaceodyssey.seams.SEAMS")


class TestCompatible:
    """Port compatibility is nominal subtyping over resolved types."""

    def test_subclass_passes(self):
        """A produced subtype satisfies a consumed base type."""
        assert compatible((qual(Derived),), (qual(Base),))

    def test_unrelated_fails(self):
        """Unrelated types do not satisfy each other."""
        assert not compatible((qual(Other),), (qual(Base),))

    def test_empty_consumes_accepts_anything(self):
        """A consumer that declares nothing accepts any producer."""
        assert compatible((qual(Other),), ())

    def test_no_produces_fails_a_requirement(self):
        """A producer that declares nothing cannot satisfy a requirement."""
        assert not compatible((), (qual(Base),))
