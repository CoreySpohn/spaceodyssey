"""Shared optional artifacts for plotting smoke tests, and the suite's compile budget.

The suite is dominated by XLA compilation. Two settings keep it fast and small:

- a persistent compilation cache per xdist worker (or ``main`` for serial runs), so
  later runs reuse compiled programs without concurrent readers seeing partial writes
  (``SPACEODYSSEY_JAX_CACHE`` names another cache root, ``off`` disables it);
- fixed cache watermarks: module teardown clears when RSS exceeds
  ``SPACEODYSSEY_RSS_LIMIT_MB`` (default 3500 MiB), or live heap allocations exceed
  half that limit. On macOS, malloc's live bytes exclude freed pages it still keeps
  resident; elsewhere RSS is the fallback. This retains useful executables without
  letting the cache budget grow with allocator retention. Cleanup and the next
  compile both need headroom below the 6 GB process budget. Release freed allocator
  pages before rechecking either watermark, so retention alone does not evict
  programs when live heap accounting falls back to RSS.
"""

import ctypes
import gc
import os
import sys
import warnings
from pathlib import Path

import equinox as eqx
import jax
import pytest
from hwoutils.testbudget.config import load_config
from hypothesis import settings

jax.config.update("jax_enable_x64", True)

# The budget plugin is mandatory and enforces the tiers set in [tool.testbudget]
# (unit 5 s, medium 30 s, slow 300 s per test, measured on a laptop). A slower machine
# scales every tier budget by SPACEODYSSEY_BUDGET_SCALE (CI sets the runner's factor).
pytest_plugins = ["hwoutils.testbudget.plugin"]
BUDGET_SCALE = float(os.environ.get("SPACEODYSSEY_BUDGET_SCALE", "1"))

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))

_cache = os.environ.get(
    "SPACEODYSSEY_JAX_CACHE", str(Path(__file__).parents[1] / ".jax-cache")
)
if _cache != "off":
    # JAX writes entries in place; workers must not read each other's partial files.
    _worker_cache = Path(_cache) / os.environ.get("PYTEST_XDIST_WORKER", "main")
    jax.config.update("jax_compilation_cache_dir", str(_worker_cache))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

RSS_LIMIT = int(os.environ.get("SPACEODYSSEY_RSS_LIMIT_MB", "3500")) * 2**20
RSS_EMERGENCY = int(os.environ.get("SPACEODYSSEY_RSS_EMERGENCY_MB", "5500")) * 2**20


class _TaskInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("rest", ctypes.c_uint8 * 80),
    ]


class _MallocStatistics(ctypes.Structure):
    _fields_ = [
        ("blocks_in_use", ctypes.c_uint),
        ("size_in_use", ctypes.c_size_t),
        ("max_size_in_use", ctypes.c_size_t),
        ("size_allocated", ctypes.c_size_t),
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
    """Request release of freed allocator pages where supported."""
    if sys.platform == "darwin":
        try:
            ctypes.CDLL("/usr/lib/libSystem.B.dylib").malloc_zone_pressure_relief(0, 0)
        except (OSError, AttributeError):
            pass
    elif sys.platform.startswith("linux"):
        try:
            trim = ctypes.CDLL(None).malloc_trim
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            trim(0)
        except (OSError, AttributeError):
            pass


def _heap_bytes(fallback):
    """Live malloc bytes across all macOS zones, excluding retained free pages."""
    if sys.platform == "darwin":
        try:
            stats = _MallocStatistics()
            read = ctypes.CDLL("/usr/lib/libSystem.B.dylib").malloc_zone_statistics
            read.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MallocStatistics)]
            read.restype = None
            read(None, ctypes.byref(stats))
            return stats.size_in_use or fallback
        except (OSError, AttributeError):
            pass
    return fallback


def _over_watermark():
    """True when resident or live heap memory exceeds the clear watermark."""
    rss = _rss_bytes()
    if rss > RSS_LIMIT or _heap_bytes(rss) > RSS_LIMIT // 2:
        _release_freed_memory()
        rss = _rss_bytes()
    return rss > RSS_LIMIT or _heap_bytes(rss) > RSS_LIMIT // 2


@pytest.fixture(autouse=True)
def warn_on_runaway_memory():
    """Fail loudly when one file grows past the emergency limit mid-file.

    Caches are never cleared between tests: a mid-file clear invalidates the compiled
    programs that module-scoped fixtures share and changes trace counts. The limit sits
    between the clear watermark and the 6 GB process budget; per-file peaks were about
    4.6 GB on the Mac and 2.2 GB on the runner, so crossing it means a file needs
    splitting or sharing more programs.
    """
    yield
    rss = _rss_bytes()
    if rss > RSS_EMERGENCY:
        _release_freed_memory()
        rss = _rss_bytes()
    if rss > RSS_EMERGENCY:
        warnings.warn(
            f"resident memory {rss / 2**20:.0f} MiB exceeds the emergency limit "
            f"{RSS_EMERGENCY / 2**20:.0f} MiB inside one file",
            ResourceWarning,
            stacklevel=1,
        )


@pytest.fixture(autouse=True, scope="module")
def drop_shared_runs():
    """Release memoized runs, then clear compiled programs if memory is high.

    Clearing happens only at a file boundary, where no shared fixture is alive.
    """
    yield
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("tests."):
            continue
        for value in vars(module).values():
            clear = getattr(value, "cache_clear", None)
            if callable(clear):
                clear()
    gc.collect()
    if _over_watermark():
        eqx.clear_caches()
        jax.clear_caches()
        gc.collect()
        _release_freed_memory()


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


# Files that exercise expensive program variants. Use one pytest process per file
# when enforcing the memory budget; `-m "not slow"` excludes them. The default
# collection keeps every check.
SLOW_FILES = frozenset(
    {
        "test_slice1_gate.py",
        # the Slice 2b gate reruns the golden histories and reads the commit history
        "test_slice2b_gate.py",
        # observer kit on runs: one executor compile alone exceeds the unit budget
        "test_observer_kit_runs.py",
        # refresh runs compile the chunked driver and the masked kernel per executor
        "test_refresh_runs.py",
        # replay of refresh runs on both executors: the refresh runs' compiles
        "test_replay_refresh.py",
        "test_observer_kit_statistics.py",
        # correction checks on scripted refresh runs: one executor compile per study
        "test_observer_kit_corrections.py",
        # a reloaded run executes on both executors: compiles the chunked driver and
        # the masked kernel, like the refresh runs
        "test_run_document_runs.py",
        "test_corpus_invariants.py",
        "test_parity_corpus.py",
        "test_corpus.py",
        "test_scorer_columns.py",
        "test_acquisition_index.py",
        "test_query_ordering.py",
        "test_executors.py",
        "test_revisions.py",
        # runs: pedigree rows, producer-local indices and an index mutant on the host
        "test_pedigree.py",
        # runs on both executors and the release handler with derived kinds
        "test_derived_corrections.py",
        "test_metamorphic.py",
        "test_host_service.py",
        "test_draws.py",
        "test_replay_log.py",
        "test_visible_relations.py",
        "test_battery_guards.py",
        "test_invariant_mutants.py",
        "test_run_status.py",
        "test_conformance_guards.py",
        "test_gradients.py",
        "test_fork.py",
        "test_latency_shift.py",
        "test_truth_swap.py",
        "test_null_reference.py",
        "test_contracts.py",
        "test_audience_firewall.py",
        "test_admission_rules.py",
        "test_leak.py",
        "test_second_kind.py",
        "test_canonical_fields.py",
        "test_golden_histories.py",
        "test_survey_25.py",
    }
)


def _tier(item):
    """The budget tier of a test, by the rule the budget plugin applies."""
    if item.get_closest_marker("slow"):
        return "slow"
    return "medium" if item.get_closest_marker("medium") else "unit"


def pytest_collection_modifyitems(config, items):
    budget = load_config(config.rootpath)
    for item in items:
        if item.path.name in SLOW_FILES:
            item.add_marker(pytest.mark.slow)
        tier = _tier(item)
        # the junit file carries the tier, so the CI summary counts medium tests
        item.user_properties.append(("tier", tier))
        if BUDGET_SCALE != 1 and not item.get_closest_marker("time_budget"):
            seconds = BUDGET_SCALE * budget.tier_budget_s(tier)
            item.add_marker(pytest.mark.time_budget(seconds))
        if item.path.name == "test_slice1_gate.py":
            # with `-n 2 --dist loadgroup` each worker compiles one executor family
            family = "compiled" if "compiled" in item.name else "host"
            item.add_marker(pytest.mark.xdist_group(family))
