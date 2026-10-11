"""Tests for the observing-context seam."""

import jax.numpy as jnp
import pytest
from planit_py.vocabulary import ObservingContext

from spaceodyssey.context import AbstractContextProvider, AlwaysObservable
from spaceodyssey.registry import REGISTRY


class TestAlwaysObservable:
    """The null context provider."""

    def test_reports_all_targets_observable(self):
        """Every configured target is observable with the constant slew."""
        provider = AlwaysObservable(target_ids=(4, 7, 9), slew_d=0.25)
        ctx = provider.context(catalog=None, time_d=12.0)
        assert isinstance(ctx, ObservingContext)
        assert ctx.time_d == 12.0
        assert jnp.array_equal(ctx.target_ids, jnp.asarray([4, 7, 9]))
        assert bool(jnp.all(ctx.observable))
        assert bool(jnp.all(ctx.slew_d == 0.25))

    def test_abstract_provider_not_instantiable(self):
        """The seam interface cannot be constructed directly."""
        with pytest.raises(TypeError):
            AbstractContextProvider()

    def test_registered_by_default(self):
        """The null provider is discoverable in the registry."""
        entry = REGISTRY.get("context", "always_observable")
        assert entry.cls is AlwaysObservable
        assert "planit_py.ObservingContext" in entry.info.produces
