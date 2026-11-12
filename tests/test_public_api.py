"""Tests for the public package surface."""

import inspect

import spaceodyssey
from spaceodyssey import codec, plugins


def test_all_names_importable():
    """Every advertised name resolves on the top-level package."""
    for name in spaceodyssey.__all__:
        assert getattr(spaceodyssey, name) is not None


def test_all_names_are_public_objects():
    """No advertised name resolves to a submodule."""
    for name in spaceodyssey.__all__:
        assert not inspect.ismodule(getattr(spaceodyssey, name)), name


def test_document_codec_surface():
    """The codec and plugin entry points that replace the spec and registry."""
    assert inspect.isclass(codec.KindResolver)
    assert callable(codec.dump_study)
    assert callable(codec.load_study)
    assert callable(plugins.installed_resolver)
    assert callable(plugins.kinds_from_entry_points)
    assert issubclass(plugins.RegistrationError, Exception)
    assert plugins.FAMILIES
