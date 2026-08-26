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


# ── The byte budget, prompt reclamation, and the in-flight guard ──────────────
#
# These cover the rewrite of 2026-08-26. The two "freed on gc" tests above pass on EITHER design,
# because they call `gc.collect()`; the point of `test_freed_by_refcount_with_the_collector_off` is
# that it does not, and so fails on the old one.

import threading                                                     # noqa: E402
import time                                                          # noqa: E402

import numpy as np                                                   # noqa: E402

from lappy import cache as C                                         # noqa: E402
from lappy.cache import cache_bytes, cache_stats, clear_instance_caches, set_cache_budget  # noqa: E402


class _Big:
    """Returns megabyte-scale arrays, like the Vandermondes the real caches hold."""
    def __init__(self):
        self.calls = 0

    @instance_lru_cache(maxsize=64)
    def block(self, n):
        self.calls += 1
        return np.zeros((n, 1000))            # n * 8 kB


@pytest.fixture(autouse=True)
def _isolated_ledger():
    """`cache_bytes()` is PROCESS-wide, so a test asserting an absolute figure would otherwise be
    reading whatever objects the previous test left alive. Reset the budget and drop every live
    cache around each test so the numbers below mean what they say."""
    def _reset():
        set_cache_budget(C.DEFAULT_MAX_BYTES)
        for cache in list(C._LIVE):
            cache.cache_clear()
    _reset()
    yield
    _reset()


def test_freed_by_refcount_with_the_collector_off():
    """THE regression test for the rewrite.

    The previous design stored a per-instance closure that captured `self`, giving
    `self -> lru_cache -> closure -> cell -> self`. That cycle is collectable, but Python's cyclic
    GC triggers on container allocation counts, not bytes, and numpy memory is invisible to those
    counters -- so hundreds of megabytes sat uncollected until a gen-2 pass happened to run.
    Measured: 958 MB peak with the collector on against 1542 MB with it off, for identical work.

    With no cycle, refcounting alone must reclaim. Note the deliberate absence of `gc.collect()`.
    """
    gc.collect()
    gc.disable()
    try:
        obj = _Big()
        obj.block(200)
        assert cache_bytes() > 0
        ref = weakref.ref(obj)
        del obj
        assert ref() is None, 'the instance survived its last reference: a cycle is back'
        assert cache_bytes() == 0, 'bytes were not returned to the ledger'
    finally:
        gc.enable()


def test_a_key_is_computed_once_even_when_threads_race_it():
    """`mps.solve_interval` dispatches the tension grid across a ThreadPoolExecutor, so two
    threads can arrive on one lambda together. `functools.lru_cache` computes it twice; a
    `regularize_pencil` plus a GSVD is far too expensive to do on a race."""
    class Slow:
        def __init__(self):
            self.calls = 0

        @instance_lru_cache(maxsize=4)
        def f(self, n):
            self.calls += 1
            time.sleep(0.05)
            return np.zeros(n)

    s = Slow()
    barrier = threading.Barrier(12)

    def hit():
        barrier.wait()
        s.f(7)

    threads = [threading.Thread(target=hit) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert s.calls == 1, f'{s.calls} computations for one key'


def test_the_unbounded_decorator_is_still_bounded_in_bytes():
    """`instance_cache` has no entry limit -- it holds lambda-independent geometry, where the
    number of distinct point sets is small and known. The byte budget is what stops a caller that
    defeats the keying from taking the machine down with it."""
    class Geo:
        @instance_cache
        def g(self, n):
            return np.zeros((n, 1000))

    set_cache_budget(8 * 1024**2)
    obj = Geo()
    for i in range(60):
        obj.g(300 + i)
    assert cache_bytes() <= 8 * 1024**2


def test_eviction_never_changes_a_value():
    """The load-bearing property: caching is an optimisation, so a value recomputed after eviction
    must equal the one that was evicted, bit for bit."""
    obj = _Big()
    set_cache_budget(2**40)
    first = obj.block(300).copy()
    set_cache_budget(1)                       # force eviction on every put
    for i in range(10):
        obj.block(300 + i)
    again = obj.block(300)
    assert np.array_equal(first, again)


def test_the_min_entries_floor_beats_the_budget_and_says_so():
    """A budget too small to hold the working set must be EXCEEDED, loudly, rather than honoured
    by recomputing Tier-1 work. Never duplicating that work outranks the bound."""
    set_cache_budget(1024)
    obj = _Big()
    with pytest.warns(ResourceWarning, match='smaller than the working set'):
        obj.block(400)
        obj.block(401)
    cache = obj.__dict__['_icache__Big_block']
    assert len(cache.data) == cache.min_entries


def test_an_exception_is_not_cached_and_reaches_a_waiting_thread():
    """On failure the pending marker must be cleared and the event set, or a second thread waiting
    on that key would block forever."""
    class Boom:
        def __init__(self):
            self.calls = 0

        @instance_lru_cache(maxsize=2)
        def f(self, x):
            self.calls += 1
            time.sleep(0.02)
            raise ValueError('boom')

    b = Boom()
    errors = []

    def call():
        try:
            b.f(1)
        except ValueError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert not any(t.is_alive() for t in threads), 'a waiter hung on a failed computation'
    assert len(errors) == 4
    assert b.calls >= 1


def test_budget_parsing_accepts_suffixes_and_rejects_nonsense():
    assert C._parse_budget('512M', 0) == 512 * 1024**2
    assert C._parse_budget('2G', 0) == 2 * 1024**3
    assert C._parse_budget('1024', 0) == 1024
    with pytest.warns(UserWarning, match='not a byte size'):
        assert C._parse_budget('banana', 4242) == 4242


def test_set_cache_budget_trims_immediately():
    """Entry size is chosen so the new budget holds comfortably more than `min_entries`; a budget
    below two entries is the floor's job, tested separately above."""
    obj = _Big()
    set_cache_budget(2**40)
    for i in range(16):
        obj.block(100 + i)                    # ~0.8 MB each
    assert cache_bytes() > 4 * 1024**2
    cache = obj.__dict__['_icache__Big_block']
    set_cache_budget(4 * 1024**2)
    assert cache_bytes() <= 4 * 1024**2
    assert len(cache.data) > cache.min_entries, 'trimmed further than the budget required'


# ── clear_instance_caches: previously untested in full ────────────────────────

def test_clear_instance_caches_counts_recurses_and_releases():
    """It had no tests at all, while `benchmarks/reference/common.py:136` records an A/B-verified
    SIGKILL from removing one of its call sites."""
    class Leaf:
        @instance_lru_cache(maxsize=4)
        def f(self, n):
            return np.zeros((n, 100))

    class Holder:
        def __init__(self, leaf):
            self.basis = leaf

        @instance_lru_cache(maxsize=4)
        def g(self, n):
            return np.zeros((n, 100))

    leaf = Leaf()
    holder = Holder(leaf)
    leaf.f(50)
    holder.g(50)
    assert cache_bytes() > 0

    n = clear_instance_caches(holder)
    assert n == 2, f'expected to clear holder.g and basis.f, cleared {n}'
    assert cache_bytes() == 0
    assert clear_instance_caches(holder) == 0, 'not idempotent'


def test_clear_instance_caches_tolerates_an_object_without_a_dict_and_a_cycle():
    class Node:
        def __init__(self):
            self.basis = None

        @instance_lru_cache(maxsize=2)
        def f(self, n):
            return n

    a, b = Node(), Node()
    a.basis, b.basis = b, a               # a cycle in the recursion graph
    a.f(1)
    b.f(1)
    assert clear_instance_caches(a) == 2  # visits each node once, does not recurse forever
    assert clear_instance_caches(object()) == 0


def test_cache_stats_reports_entries_bytes_hits_and_misses():
    obj = _Big()
    obj.block(100)
    obj.block(100)
    obj.block(101)
    st = cache_stats()['_Big.block']
    assert st['entries'] >= 2 and st['bytes'] > 0
    assert st['hits'] >= 1 and st['misses'] >= 2
