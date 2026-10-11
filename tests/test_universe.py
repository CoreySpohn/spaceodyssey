"""Tests for truth containers and universe sources."""

import jax.random as jr
import pytest

from spaceodyssey.universe import AbstractUniverseSource, FixedUniverse, Universe


class TestUniverse:
    """The frozen truth container."""

    def test_defaults(self):
        """An empty universe has no systems, exozodi, or instrument."""
        universe = Universe()
        assert universe.systems == {}
        assert universe.exozodi == {}
        assert universe.instrument is None

    def test_fixed_source_ignores_key(self):
        """FixedUniverse returns its universe for any key."""
        universe = Universe(systems={1: "system-one"})
        source = FixedUniverse(universe=universe)
        assert source.draw(jr.PRNGKey(0)) is universe
        assert source.draw(jr.PRNGKey(1)) is universe

    def test_abstract_source_not_instantiable(self):
        """The source interface cannot be constructed directly."""
        with pytest.raises(TypeError):
            AbstractUniverseSource()
