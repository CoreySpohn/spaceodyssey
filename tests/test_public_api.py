"""Tests for the public package surface."""

import inspect

import pytest

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


@pytest.mark.parametrize(
    "name",
    [
        "REGISTRY",
        "Registry",
        "MissionSpec",
        "FidelityConfig",
        "FileRef",
        "SeamRef",
        "WiringError",
        "RegistryError",
        "build",
        "run",
        "run_mission",
        "get_engine",
        "register_engine",
        "to_json",
        "spec_from_json",
        "AbstractCertificate",
        "AbstractProfileCertificate",
        "CertificateState",
        "GaussianProfileCertifier",
        "SnrThreshold",
        "ToyGaussianCertifier",
        "certified_yield",
        "CertifiedYield",
        "PlanSummary",
        "Truth",
        "Profile",
        "Universe",
    ],
)
def test_legacy_names_are_gone(name):
    """Fails if a name of a removed API is exported again."""
    assert not hasattr(spaceodyssey, name)
    assert name not in spaceodyssey.__all__


def test_document_codec_surface():
    """The codec and plugin entry points that replace the spec and registry."""
    assert inspect.isclass(codec.KindResolver)
    assert callable(codec.dump_study)
    assert callable(codec.load_study)
    assert callable(plugins.installed_resolver)
    assert callable(plugins.kinds_from_entry_points)
    assert issubclass(plugins.RegistrationError, Exception)
    assert plugins.FAMILIES
