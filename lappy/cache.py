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
