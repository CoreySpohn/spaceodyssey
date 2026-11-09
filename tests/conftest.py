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


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "clear_caches_between_tests: clear compiled programs after any test that "
        "leaves memory above the clear watermark; for files that share no compiled "
        "program or trace count across tests",
    )


@pytest.fixture(autouse=True)
def bound_resident_memory(request):
    """Keep one file's memory bounded without clearing under shared fixtures.

    By default nothing is cleared between tests: a mid-file clear invalidates the
    compiled programs that module-scoped fixtures share and changes trace counts. Tests
    that clear caches on purpose (cache-state controls) do so themselves. Past the
    emergency limit a warning fails the test loudly; it is measured on live heap where
    the platform reports it, because macOS keeps freed pages resident and RSS stays
    high after a clear. The limit sits between the clear
    watermark and the 6 GB process budget; per-file peaks were about 4.6 GB on the Mac
    and 2.2 GB on the runner.

    A file marked `clear_caches_between_tests` opts back into the watermark clear after
    each test. Only files that compile many unrelated programs and read no compiled
    fixture or trace count across tests carry it.
    """
    yield
    if request.node.get_closest_marker("clear_caches_between_tests"):
        if _over_watermark():
            eqx.clear_caches()
            jax.clear_caches()
            gc.collect()
            _release_freed_memory()
        return
    rss = _rss_bytes()
    if rss > RSS_EMERGENCY:
        _release_freed_memory()
        rss = _rss_bytes()
    live = _heap_bytes(rss)
    if live > RSS_EMERGENCY:
        warnings.warn(
            f"live memory {live / 2**20:.0f} MiB exceeds the emergency limit "
            f"{RSS_EMERGENCY / 2**20:.0f} MiB inside one file",
            ResourceWarning,
            stacklevel=1,
        )


@pytest.fixture(autouse=True, scope="module")
def drop_shared_runs():
    """Release memoized runs, then clear compiled programs if memory is high.

    The budget plugin also clears JAX caches after every file (clear_caches in
    pyproject.toml), so each file's program count is independent of the files
    before it; this fixture adds the equinox caches and freed-memory release when
    over the watermark. Files marked clear_caches_between_tests also clear after
    each test. Session-scoped fixtures survive either clear and recompile on use.
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


# Files that exercise expensive program variants, with the reason each is slow. Use one
# pytest process per file when enforcing the memory budget; `-m "not slow"` excludes
# them. The default collection keeps every check.
GATE = "gate or corpus file"
SLOW_FILES = {
    "test_slice3a_gate.py": "reruns goldens and four mutant runs, reads the history",
    "test_observer_kit_runs.py": "observer kit on runs, one executor compile alone",
    "test_refresh_runs.py": "refresh runs compile the chunked driver per executor",
    "test_replay_refresh.py": "replay of refresh runs repeats the refresh compiles",
    "test_observer_kit_statistics.py": GATE,
    "test_observer_kit_corrections.py": "scripted refresh runs, one compile per study",
    "test_run_document_runs.py": "a reloaded run executes on both executors",
    "test_corpus_invariants.py": GATE,
    "test_parity_corpus.py": GATE,
    "test_corpus.py": GATE,
    "test_scorer_columns.py": GATE,
    "test_acquisition_index.py": GATE,
    "test_query_ordering.py": GATE,
    "test_executors.py": GATE,
    "test_revisions.py": GATE,
    "test_pedigree.py": "pedigree rows and an index mutant run on the host",
    "test_derived_corrections.py": "runs on both executors with derived kinds",
    "test_metamorphic.py": GATE,
    "test_host_service.py": GATE,
    "test_draws.py": GATE,
    "test_replay_log.py": GATE,
    "test_visible_relations.py": GATE,
    "test_battery_guards.py": GATE,
    "test_invariant_mutants.py": GATE,
    "test_run_status.py": GATE,
    "test_conformance_guards.py": GATE,
    "test_gradients.py": GATE,
    "test_fork.py": GATE,
    "test_latency_shift.py": GATE,
    "test_truth_swap.py": GATE,
    "test_null_reference.py": GATE,
    "test_contracts.py": GATE,
    "test_audience_firewall.py": GATE,
    "test_admission_rules.py": GATE,
    "test_leak.py": GATE,
    "test_second_kind.py": GATE,
    "test_canonical_fields.py": GATE,
    "test_golden_histories.py": GATE,
    "test_survey_25.py": GATE,
}


def _tier(item):
    """The budget tier of a test, by the rule the budget plugin applies."""
    if item.get_closest_marker("slow"):
        return "slow"
    return "medium" if item.get_closest_marker("medium") else "unit"


def pytest_collection_modifyitems(config, items):
    budget = load_config(config.rootpath)
    for item in items:
        if item.path.name in SLOW_FILES:
            reason = f"whole file in SLOW_FILES: {SLOW_FILES[item.path.name]}"
            item.add_marker(pytest.mark.slow(reason=reason))
        tier = _tier(item)
        # the junit file carries the tier, so the CI summary counts medium tests
        item.user_properties.append(("tier", tier))
        if BUDGET_SCALE != 1 and not item.get_closest_marker("time_budget"):
            seconds = BUDGET_SCALE * budget.tier_budget_s(tier)
            item.add_marker(pytest.mark.time_budget(seconds))
