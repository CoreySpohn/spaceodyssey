"""Smoke test for the package skeleton."""

import spaceodyssey


def test_package_imports():
    """The package imports.

    Fails if the package skeleton or its build metadata is broken.
    """
    assert spaceodyssey.__doc__
