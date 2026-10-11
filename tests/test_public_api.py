"""Tests for the public package surface."""

import importlib
import inspect

import spaceodyssey


def test_all_names_importable():
    """Every advertised name resolves on the top-level package."""
    for name in spaceodyssey.__all__:
        assert getattr(spaceodyssey, name) is not None


def test_prepared_build_exports():
    """Prepared construction objects are exported alongside build."""
    module = importlib.import_module("spaceodyssey.build")
    for name in ("BuildPlan", "ResolvedImpl", "prepare", "construct"):
        assert name in spaceodyssey.__all__
        assert getattr(spaceodyssey, name) is getattr(module, name)


def test_all_names_are_public_objects():
    """No advertised name resolves to a submodule."""
    for name in spaceodyssey.__all__:
        assert not inspect.ismodule(getattr(spaceodyssey, name)), name


def test_certified_yield_is_reporting_function():
    """The root entry point shares the reporting implementation."""
    assert spaceodyssey.certified_yield is spaceodyssey.reporting.certified_yield


def test_observe_is_loop_function():
    """The public observe entry point is the loop function."""
    assert spaceodyssey.observe is spaceodyssey.loop.observe


def test_observe_submodule_remains_importable():
    """The measurement interface remains available through its module path."""
    module = importlib.import_module("spaceodyssey.observe")
    assert module.AbstractMeasurement is spaceodyssey.AbstractMeasurement


def test_default_registry_entries_available():
    """The in-tree reference implementations are discoverable and usable."""
    for seam, name in (
        ("certification", "snr_threshold"),
        ("certification", "toy_gaussian"),
        ("context", "always_observable"),
        ("cost", "constant_overheads"),
    ):
        entry = spaceodyssey.REGISTRY.get(seam, name)
        assert entry.available, f"{seam}:{name} unavailable: {entry.reason}"


def test_sample_engine_available():
    """The sampling engine resolves through the public entry point."""
    assert callable(spaceodyssey.get_engine("sample"))


def test_integrate_engine_available():
    """The analytic integrate engine resolves through the public entry point."""
    assert callable(spaceodyssey.get_engine("integrate"))
