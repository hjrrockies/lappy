"""Tests for lappy.cache — instance-stored cache decorators."""

import gc
import weakref

import pytest

from lappy.cache import instance_cache, instance_lru_cache


# ── Helper classes ────────────────────────────────────────────────────────────

class _Cached:
    """Tracks call count for instance_cache."""
    def __init__(self):
        self.calls = 0

    @instance_cache
    def compute(self, x):
        self.calls += 1
        return x * 2


class _LRUCached:
    """Tracks call count for instance_lru_cache(maxsize=2)."""
    def __init__(self):
        self.calls = 0

    @instance_lru_cache(maxsize=2)
    def compute(self, x):
        self.calls += 1
        return x * 3


# ── instance_cache ────────────────────────────────────────────────────────────

def test_instance_cache_correct_result():
    assert _Cached().compute(5) == 10


def test_instance_cache_caches():
    obj = _Cached()
    obj.compute(5)
    obj.compute(5)
    assert obj.calls == 1


def test_instance_cache_different_args_computed_separately():
    obj = _Cached()
    obj.compute(1)
    obj.compute(2)
    assert obj.calls == 2


def test_instance_cache_per_instance():
    """Two instances must not share a cache."""
    a, b = _Cached(), _Cached()
    a.compute(7)
    b.compute(7)
    assert a.calls == 1 and b.calls == 1


def test_instance_cache_freed_on_gc():
    """Cache must not prevent the instance from being garbage-collected."""
    ref = None

    def _make():
        nonlocal ref
        obj = _Cached()
        obj.compute(1)
        ref = weakref.ref(obj)

    _make()
    gc.collect()
    assert ref() is None


# ── instance_lru_cache ────────────────────────────────────────────────────────

def test_instance_lru_cache_correct_result():
    assert _LRUCached().compute(4) == 12


def test_instance_lru_cache_caches():
    obj = _LRUCached()
    obj.compute(4)
    obj.compute(4)
    assert obj.calls == 1


def test_instance_lru_cache_per_instance():
    """Two instances must not share a cache."""
    a, b = _LRUCached(), _LRUCached()
    a.compute(9)
    b.compute(9)
    assert a.calls == 1 and b.calls == 1


def test_instance_lru_cache_maxsize_evicts():
    """With maxsize=2, a third distinct arg evicts the LRU entry."""
    obj = _LRUCached()
    obj.compute(1)   # slot 1
    obj.compute(2)   # slot 2
    obj.compute(3)   # evicts 1, fills slot
    calls_before = obj.calls
    obj.compute(1)   # must recompute
    assert obj.calls == calls_before + 1


def test_instance_lru_cache_freed_on_gc():
    """Cache must not prevent the instance from being garbage-collected."""
    ref = None

    def _make():
        nonlocal ref
        obj = _LRUCached()
        obj.compute(1)
        ref = weakref.ref(obj)

    _make()
    gc.collect()
    assert ref() is None
