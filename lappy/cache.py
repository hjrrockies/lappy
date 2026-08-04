"""Instance-stored cache decorators.

Drop-in replacements for @cache / @lru_cache on bound methods.
Caches are stored in self.__dict__ so they are freed with the instance.
"""
import threading
from functools import lru_cache

# Sentinel used to separate positional args from kwargs in the flat cache key.
_KW_SEP = object()


def instance_cache(method):
    """Replacement for @cache on bound methods (unlimited, instance-stored)."""
    attr      = f'_icache_{method.__qualname__.replace(".", "_")}'
    lock_attr = attr + '_lock'

    def wrapper(self, *args, **kwargs):
        key = args + (_KW_SEP,) + tuple(sorted(kwargs.items())) if kwargs else args
        d = self.__dict__
        try:
            cache = d[attr]
        except KeyError:
            d.setdefault(lock_attr, threading.Lock())
            d.setdefault(attr, {})
            cache = d[attr]
        if key not in cache:
            with d[lock_attr]:
                if key not in cache:          # double-checked locking
                    cache[key] = method(self, *args, **kwargs)
        return cache[key]

    wrapper.__wrapped__  = method
    wrapper.__name__     = method.__name__
    wrapper.__qualname__ = method.__qualname__
    return wrapper


def instance_lru_cache(maxsize=128):
    """Replacement for @lru_cache(maxsize=N) on bound methods (bounded LRU, instance-stored).

    lru_cache is internally thread-safe; no additional locking is needed.
    The cycle (self → lru_cache → lambda → self) is handled by Python's cyclic GC.

    kwargs are folded into the flat cache key via a sentinel separator so that
    callers using keyword arguments get correct cache hits/misses.
    """
    def decorator(method):
        attr = f'_icache_{method.__qualname__.replace(".", "_")}'

        def wrapper(self, *args, **kwargs):
            flat = args + (_KW_SEP,) + tuple(sorted(kwargs.items())) if kwargs else args
            try:
                bound = self.__dict__[attr]
            except KeyError:
                def bound(*flat_args):
                    idx = next((i for i, x in enumerate(flat_args) if x is _KW_SEP), None)
                    if idx is not None:
                        return method(self, *flat_args[:idx], **dict(flat_args[idx+1:]))
                    return method(self, *flat_args)
                bound = lru_cache(maxsize=maxsize)(bound)
                self.__dict__[attr] = bound
            return bound(*flat)

        wrapper.__wrapped__  = method
        wrapper.__name__     = method.__name__
        wrapper.__qualname__ = method.__qualname__
        return wrapper
    return decorator


def clear_instance_caches(obj, recurse=('basis', 'bases', 'solver')):
    """Drop every ``instance_lru_cache`` on ``obj`` (and, by default, on the
    basis/solver objects it holds).

    The caches are sized in *entries*, not bytes, which is the right trade for
    repeated scalar evaluations at the same lambda but not for evaluations over
    a large point set: one Vandermonde over a degree-10 cubature mesh is
    megabytes, and ``NormalizedBasis.norms`` alone keeps 128 of them. Certifying
    ten eigenvalues across four symmetry sectors that way is enough to exhaust
    memory on a 16GB machine.

    This does not change any default -- nothing calls it unless asked. Use it
    between eigenvalues in a long certification loop to bound peak memory to
    roughly one evaluation.

    Returns the number of caches cleared.
    """
    n = 0
    for key, val in list(getattr(obj, '__dict__', {}).items()):
        if key.startswith('_icache_'):
            try:
                val.cache_clear()
            except AttributeError:
                pass
            del obj.__dict__[key]
            n += 1
    for name in recurse:
        child = getattr(obj, name, None)
        if child is None:
            continue
        for c in (child if isinstance(child, (list, tuple)) else [child]):
            if hasattr(c, '__dict__'):
                n += clear_instance_caches(c, recurse=recurse)
    return n
