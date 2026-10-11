"""Tests for the policy vocabulary shim."""

import planit_py.vocabulary as vocabulary

from spaceodyssey import policy


def test_shim_re_exports_identical_objects():
    """The shim exposes the scheduling library's objects, not copies."""
    for name in policy.__all__:
        assert getattr(policy, name) is getattr(vocabulary, name)
