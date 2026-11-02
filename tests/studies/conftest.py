"""Scope 64-bit arithmetic to the transaction tests."""

import jax
import pytest


class _X64Module(pytest.Module):
    def _getobj(self):
        with jax.enable_x64():
            return super()._getobj()


def pytest_pycollect_makemodule(module_path, parent):
    return _X64Module.from_parent(parent, path=module_path)


@pytest.fixture(scope="module", autouse=True)
def enable_x64():
    with jax.enable_x64():
        yield
