"""Shared test configuration: float64 is set once, here."""

import importlib.util
import os

import jax
from hypothesis import settings

jax.config.update("jax_enable_x64", True)

# Released hwoutils versions may not include the optional plugins yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))
