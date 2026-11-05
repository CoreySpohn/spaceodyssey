"""The memory guard releases allocator retention before discarding programs."""

from types import SimpleNamespace

import pytest

from tests import conftest


@pytest.mark.parametrize(
    "rss, retained_rss, heap, expected",
    [
        (0.5, 0.5, 0.25, []),
        (1.1, 0.8, 0.25, ["release"]),
        (1.1, 1.1, 0.25, ["release", "clear", "collect", "release"]),
        (0.8, 0.8, 0.6, ["release", "clear", "collect", "release"]),
        (1.1, 0.8, 0.6, ["release", "clear", "collect", "release"]),
        (0.8, 0.4, None, ["release"]),
        (0.8, 0.8, None, ["release", "clear", "collect", "release"]),
    ],
)
def test_memory_guard_keeps_fixed_watermarks(
    monkeypatch, rss, retained_rss, heap, expected
):
    readings = iter([rss, retained_rss])
    actions = []

    def heap_bytes(fallback):
        return fallback if heap is None else heap * 100

    monkeypatch.setattr(conftest, "RSS_LIMIT", 100)
    monkeypatch.setattr(conftest, "_rss_bytes", lambda: next(readings) * 100)
    monkeypatch.setattr(conftest, "_heap_bytes", heap_bytes)
    monkeypatch.setattr(
        conftest, "_release_freed_memory", lambda: actions.append("release")
    )
    monkeypatch.setattr(conftest.jax, "clear_caches", lambda: actions.append("clear"))
    monkeypatch.setattr(conftest.gc, "collect", lambda: actions.append("collect"))
    guard = conftest.bound_resident_memory.__wrapped__()
    next(guard)
    with pytest.raises(StopIteration):
        next(guard)
    assert actions == expected


@pytest.mark.parametrize("available", [False, True])
def test_linux_allocator_release_is_optional(monkeypatch, available):
    calls = []

    def trim(padding):
        calls.append(padding)
        return 1

    library = SimpleNamespace(malloc_trim=trim) if available else SimpleNamespace()
    monkeypatch.setattr(conftest, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(conftest.ctypes, "CDLL", lambda name: library)
    conftest._release_freed_memory()
    assert calls == ([0] if available else [])
    if available:
        assert trim.argtypes == [conftest.ctypes.c_size_t]
        assert trim.restype is conftest.ctypes.c_int
