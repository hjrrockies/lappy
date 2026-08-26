"""Instance-stored caches with a process-wide byte budget.

Drop-in replacements for `@cache` / `@lru_cache` on bound methods. Caches live in `self.__dict__`,
so they are freed with the instance -- by REFCOUNT, not by the cyclic collector.

WHY NOT `functools.lru_cache`. On a method it pins `self` in the cache key, so nothing is ever
freed. The obvious fix -- store a per-instance `lru_cache`-wrapped closure on the instance -- was
what this module did, and it trades the leak for a reference cycle: `self -> lru_cache -> closure
-> cell -> self`. That cycle IS collectable, but Python's cyclic collector triggers on container
allocation COUNTS, not bytes, and numpy array memory is invisible to those counters. So a solver
holding hundreds of megabytes sits uncollected until a gen-2 pass happens to run.

Measured, 14 shape iterates at n_eigs=6 (`benchmarks/memory/cache_probe.py`): peak RSS 958 MB with
the collector on, 1542 MB with it off, while the LIVE solver's caches were only 27.6 MB. The rest
was dead solvers waiting for a collection. In a five-worker `douse` grid that took a 16 GB machine
to 15.4 GB of swap in forty minutes.

So the wrapper here is defined ONCE at decoration time and closes over `attr`, `name`, `maxsize`,
`min_entries` and `method` -- never over `self`. `self.__dict__[attr]` is an ordinary value
reference, and the cache dies with its owner the moment the last reference goes.

WHY BYTES AND NOT ENTRY COUNTS. Seven caches at 64 entries each of an unknown array size is an
unknown total; a per-method entry limit cannot express "this solver may use at most X MB", which
is the only bound a user can reason about. Entry limits are kept as a hit-rate knob -- they encode
real measurements, e.g. `maxsize=8` on `_raw_eval` holds four lambdas of a two-component build --
and the byte budget is a ceiling over all of them that only binds in the runaway case.

THE BUDGET IS PER PROCESS, AND THAT MULTIPLIES. One solve keeps a `NormalizedBasis`, its wrapped
basis, often a `FundamentalBasis` and the `MPSEigensolver` alive -- four cache groups -- so a
per-instance cap generous enough for the largest is several times too generous in aggregate. A
process-wide ledger also evicts the right things: the previous shape iterate's entries are strictly
older than the current one's. But note that N worker processes get N budgets. Five workers at the
512 MiB default is 2.5 GB of cache before anything else, which is precisely the arithmetic that
produced the swap event above.

NEVER DUPLICATE TIER-1 WORK. Two guarantees beyond what `lru_cache` gives:
  * an in-flight guard -- a second thread arriving on a key already being computed waits for the
    first rather than computing it again. `mps.solve_interval` dispatches the tension grid across
    a `ThreadPoolExecutor`, so without this a `regularize_pencil` + GSVD at one lambda could run
    on several threads at once. `lru_cache` duplicates on that race.
  * a `min_entries` floor -- a cache is never trimmed below its most recent few entries, and a key
    being computed is never evictable. If the budget cannot honour the floor, the budget loses and
    a `ResourceWarning` is emitted: recomputing a Vandermonde to respect a number is the wrong
    trade.
"""
import itertools
import os
import sys
import threading
import warnings
import weakref
from collections import OrderedDict

import numpy as np

# One lock for all cache bookkeeping. Every critical section here is a handful of dict operations;
# the expensive part (the wrapped method) always runs OUTSIDE it, so this never serialises real
# work. A per-cache lock plus a ledger lock would invite lock-order inversion for no measurable
# gain.
_LOCK = threading.RLock()
_STAMP = itertools.count()          # monotonic; `next` is one atomic C call
_LIVE = weakref.WeakSet()           # every live _Cache, for the global eviction scan
_TOTAL = [0]                        # bytes currently held, process-wide

_KW_SEP = object()                  # separates positional args from kwargs in the flat key

DEFAULT_MAX_BYTES = 512 * 1024**2
MIN_ENTRIES = 2


def _parse_budget(text, default):
    try:
        t = str(text).strip().upper()
        mult = {'K': 1024, 'M': 1024**2, 'G': 1024**3}.get(t[-1:])
        return int(float(t[:-1]) * mult) if mult else int(t)
    except (TypeError, ValueError, IndexError):
        warnings.warn(f'LAPPY_CACHE_BYTES={text!r} is not a byte size; using the default')
        return default


_BUDGET = [_parse_budget(os.environ['LAPPY_CACHE_BYTES'], DEFAULT_MAX_BYTES)
           if 'LAPPY_CACHE_BYTES' in os.environ else DEFAULT_MAX_BYTES]


def _sizeof(obj, _depth=0, _seen=None):
    """Bytes held by a cached value. Runs on a MISS only, never on the hit path.

    A view is charged its base buffer's size, since holding the view pins the whole thing. Shared
    arrays inside one value are charged once. `mpmath.matrix` (`bases.ExPrecFBBasis`) is
    deliberately over-estimated -- `sys.getsizeof` would report it as nearly free, and being wrong
    by a factor of two is much better than being wrong by a factor of a thousand.
    """
    _seen = set() if _seen is None else _seen
    if isinstance(obj, np.ndarray):
        root = obj
        while isinstance(root.base, np.ndarray):
            root = root.base
        if id(root) in _seen:
            return 0
        _seen.add(id(root))
        return root.nbytes
    if isinstance(obj, (tuple, list)) and _depth < 2:
        return sum(_sizeof(o, _depth + 1, _seen) for o in obj[:64])
    if type(obj).__name__ == 'matrix' and hasattr(obj, 'rows'):        # mpmath, without importing it
        try:
            from mpmath import mp
            return int(obj.rows) * int(obj.cols) * (48 + mp.prec // 8)
        except Exception:                                             # noqa: BLE001
            return int(obj.rows) * int(obj.cols) * 64
    try:
        return sys.getsizeof(obj)
    except TypeError:
        return 0


class _Cache:
    """One method's cache on one instance: `key -> (value, nbytes, stamp)`, newest last."""
    __slots__ = ('name', 'maxsize', 'min_entries', 'data', 'pending', 'hits', 'misses',
                 '_n', '__weakref__')

    def __init__(self, name, maxsize, min_entries):
        self.name = name
        self.maxsize = maxsize
        self.min_entries = min_entries
        self.data = OrderedDict()
        self.pending = {}
        self.hits = 0
        self.misses = 0
        # Byte count lives in a list so the finalizer below can read it WITHOUT holding a
        # reference to self -- a finalizer that captured self would resurrect the cycle this
        # module exists to remove.
        self._n = [0]
        with _LOCK:
            _LIVE.add(self)
        weakref.finalize(self, _release, self._n)

    def cache_clear(self):
        """Named for `functools.lru_cache` compatibility: `clear_instance_caches` calls it."""
        with _LOCK:
            self.data.clear()
            _TOTAL[0] -= self._n[0]
            self._n[0] = 0

    @property
    def nbytes(self):
        return self._n[0]


def _release(n):
    """Give a dead cache's bytes back. Prompt, because there is no cycle to wait on."""
    with _LOCK:
        _TOTAL[0] -= n[0]
        n[0] = 0


def _store(cache, key, value):
    """Insert under `_LOCK`, then trim to the entry limit and the global byte budget."""
    nb = _sizeof(value)
    cache.data[key] = (value, nb, next(_STAMP))
    cache._n[0] += nb
    _TOTAL[0] += nb

    if cache.maxsize is not None:
        while len(cache.data) > max(cache.maxsize, cache.min_entries):
            _drop(cache, next(iter(cache.data)))

    if _TOTAL[0] > _BUDGET[0]:
        _evict_to_budget()


def _drop(cache, key):
    value_nb = cache.data.pop(key, None)
    if value_nb is None:
        return 0
    cache._n[0] -= value_nb[1]
    _TOTAL[0] -= value_nb[1]
    return value_nb[1]


def _evict_to_budget():
    """Globally oldest first, across every live cache. Called under `_LOCK`, and rarely.

    Approximate LRU by stamp rather than a real global order: the scan is O(total entries), which
    is in the hundreds, against the megabytes being freed. A cache is never trimmed below
    `min_entries`, and a key with a computation in flight is never evictable -- so a budget too
    small to hold the working set is exceeded, loudly, rather than honoured by recomputing.
    """
    while _TOTAL[0] > _BUDGET[0]:
        oldest, victim = None, None
        for c in _LIVE:
            if len(c.data) <= c.min_entries:
                continue
            for key, (_v, _nb, stamp) in c.data.items():
                if key in c.pending:
                    continue
                if oldest is None or stamp < oldest:
                    oldest, victim = stamp, (c, key)
                break                      # entries are newest-last, so the first is the oldest
        if victim is None:
            warnings.warn(
                f'lappy cache budget of {_BUDGET[0]/1024**2:.0f} MiB is smaller than the working '
                f'set ({_TOTAL[0]/1024**2:.0f} MiB held, nothing evictable above the '
                f'min_entries floor). Exceeding it rather than recomputing; raise the budget with '
                f'lappy.cache.set_cache_budget() or LAPPY_CACHE_BYTES.', ResourceWarning)
            return
        _drop(*victim)


def _make_key(args, kwargs):
    return args + (_KW_SEP,) + tuple(sorted(kwargs.items())) if kwargs else args


def _cached_call(cache, key, method, self, args, kwargs):
    """Return `method(self, *args, **kwargs)`, computed at most once per key across threads."""
    # HIT PATH, DELIBERATELY LOCK-FREE. `dict.get` is atomic under the GIL, and the value is in
    # hand before any concurrent eviction could drop the key, so a race costs nothing. Recency is
    # tracked by insertion order alone (`_store` appends, eviction takes the front) rather than by
    # re-stamping on every hit: the measured working sets here are 3-8 entries, so the difference
    # between LRU and FIFO is nil, and it saves a lock, a tuple allocation and a dict write on the
    # hottest path in the module. `hits` is a statistic, so a lost increment under a race is fine.
    hit = cache.data.get(key)
    if hit is not None:
        cache.hits += 1
        return hit[0]

    with _LOCK:
        hit = cache.data.get(key)
        if hit is not None:
            cache.hits += 1
            return hit[0]
        event = cache.pending.get(key)
        owner = event is None
        if owner:
            event = cache.pending[key] = threading.Event()

    if not owner:
        event.wait()
        with _LOCK:
            hit = cache.data.get(key)
            if hit is not None:
                cache.hits += 1
                return hit[0]
        # The owner raised. Nothing was cached (matching lru_cache), so compute it ourselves and
        # let the exception surface here too if it is deterministic.
        return method(self, *args, **kwargs)

    try:
        value = method(self, *args, **kwargs)
    except BaseException:
        with _LOCK:
            cache.pending.pop(key, None)
        event.set()
        raise
    with _LOCK:
        cache.misses += 1
        _store(cache, key, value)
        cache.pending.pop(key, None)
    event.set()
    return value


def _decorate(method, maxsize, min_entries):
    attr = f'_icache_{method.__qualname__.replace(".", "_")}'
    name = method.__qualname__

    def wrapper(self, *args, **kwargs):
        d = self.__dict__
        cache = d.get(attr)
        if cache is None:
            cache = d.setdefault(attr, _Cache(name, maxsize, min_entries))
        return _cached_call(cache, _make_key(args, kwargs), method, self, args, kwargs)

    wrapper.__wrapped__ = method
    wrapper.__name__ = method.__name__
    wrapper.__qualname__ = method.__qualname__
    wrapper.__doc__ = method.__doc__
    return wrapper


def instance_cache(method):
    """Replacement for `@cache` on bound methods: unbounded in ENTRIES, bounded in bytes.

    Used for the lambda-independent geometry (`bases.FourierBesselBasis._theta` and friends), where
    the number of distinct point sets a solver sees is small and known -- so an entry limit would
    only ever evict something it should have kept.
    """
    return _decorate(method, None, MIN_ENTRIES)


def instance_lru_cache(maxsize=128, min_entries=MIN_ENTRIES):
    """Replacement for `@lru_cache(maxsize=N)` on bound methods, plus the byte budget.

    `maxsize` stays a hit-rate knob: it encodes real measurements at each call site (see
    `bases._raw_eval`, sized so `tensions(n_workers>2)` does not thrash) and is not a memory bound.
    """
    def decorator(method):
        return _decorate(method, maxsize, min_entries)
    return decorator


def clear_instance_caches(obj, recurse=('basis', 'bases', 'solver'), _seen=None):
    """Drop every instance cache on `obj`, and on the basis/solver objects it holds.

    NO LONGER A MEMORY NECESSITY. Caches are now bounded in bytes and freed with their owner by
    refcount, so nothing needs to call this to stay within memory. It survives as a determinism and
    tuning tool -- `benchmarks/reference/common.py:126` records that removing it once got
    `reg_ngon_6` SIGKILLed at n_basis=320 (A/B verified), which is the regime it was written for,
    while `common.py:205` records the same call having no measurable effect at another site.

    Returns the number of caches cleared.
    """
    _seen = set() if _seen is None else _seen
    if id(obj) in _seen:
        return 0
    _seen.add(id(obj))
    n = 0
    for key, val in list(getattr(obj, '__dict__', {}).items()):
        if not key.startswith('_icache_'):
            continue
        try:
            val.cache_clear()
        except AttributeError:
            pass
        del obj.__dict__[key]
        n += 1
    for attr in recurse:
        child = getattr(obj, attr, None)
        if child is None:
            continue
        for c in (child if isinstance(child, (list, tuple)) else [child]):
            if hasattr(c, '__dict__'):
                n += clear_instance_caches(c, recurse=recurse, _seen=_seen)
    return n


def set_cache_budget(nbytes):
    """Set the process-wide cache budget in bytes, trimming immediately if now over.

    Remember this is PER PROCESS: a pool of N workers gets N budgets.
    """
    with _LOCK:
        _BUDGET[0] = int(nbytes)
        _evict_to_budget()
    return _BUDGET[0]


def cache_budget():
    return _BUDGET[0]


def cache_bytes():
    """Bytes currently held by every live instance cache in this process."""
    return _TOTAL[0]


def cache_stats():
    """`{qualname: {'entries', 'bytes', 'hits', 'misses'}}`, summed over live instances.

    Nothing exposed cache statistics before; the byte attribution in
    `benchmarks/memory/cache_probe.py` had to walk `gc.get_referents` to weigh an `lru_cache`.
    """
    out = {}
    with _LOCK:
        for c in _LIVE:
            e = out.setdefault(c.name, {'entries': 0, 'bytes': 0, 'hits': 0, 'misses': 0})
            e['entries'] += len(c.data)
            e['bytes'] += c._n[0]
            e['hits'] += c.hits
            e['misses'] += c.misses
    return out
