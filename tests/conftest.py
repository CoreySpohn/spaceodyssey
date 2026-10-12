"""Shared optional artifacts for plotting smoke tests."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def save_viz_png():
    """Save a small smoke-test image only when explicitly requested."""

    def save(result, name):
        directory = os.environ.get("SPACEODYSSEY_VIZ_PNG_DIR")
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            result.fig.savefig(path / f"{name}.png", dpi=70)

    return save
