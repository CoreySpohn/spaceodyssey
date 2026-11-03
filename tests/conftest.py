"""Shared optional artifacts for plotting smoke tests, and the suite's compile budget.

The suite is dominated by XLA compilation. Two settings keep it fast and small:

- a persistent compilation cache, so separate processes and later runs reuse compiled
  programs (``SPACEODYSSEY_JAX_CACHE`` names another directory, ``off`` disables it);
- a bound on what a long-lived process keeps: when its resident memory passes
  ``SPACEODYSSEY_RSS_LIMIT_MB`` (default 4500) and has grown by more than 1000 MB since
  the last clear, the next test teardown drops the compiled programs.
"""

import ctypes
import gc
import os
import sys
from pathlib import Path

import jax
import pytest

_cache = os.environ.get(
    "SPACEODYSSEY_JAX_CACHE", str(Path(__file__).parents[1] / ".jax-cache")
)
if _cache != "off":
    jax.config.update("jax_compilation_cache_dir", _cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

RSS_LIMIT = int(os.environ.get("SPACEODYSSEY_RSS_LIMIT_MB", "4500")) * 2**20
RSS_GROWTH = 1000 * 2**20
_after_clear = [0]


class _TaskInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("rest", ctypes.c_uint8 * 80),
    ]


def _rss_bytes():
    """Resident memory of this process now, or 0 where it cannot be read."""
    try:
        if sys.platform == "darwin":
            info = _TaskInfo()
            libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
            size = ctypes.sizeof(info)
            got = libproc.proc_pidinfo(os.getpid(), 4, 0, ctypes.byref(info), size)
            return info.resident_size if got > 0 else 0
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, AttributeError, IndexError, ValueError):
        return 0


def _release_freed_memory():
    """Hand freed allocator pages back to the system (macOS keeps them otherwise)."""
    if sys.platform == "darwin":
        try:
            ctypes.CDLL("/usr/lib/libSystem.B.dylib").malloc_zone_pressure_relief(0, 0)
        except (OSError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def bound_resident_memory():
    """Drop compiled programs between tests when the process has grown too large."""
    yield
    rss = _rss_bytes()
    if rss > RSS_LIMIT and rss - _after_clear[0] > RSS_GROWTH:
        jax.clear_caches()
        gc.collect()
        _release_freed_memory()
        _after_clear[0] = _rss_bytes()


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
