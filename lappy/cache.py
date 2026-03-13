"""Instance-stored cache decorators.

Drop-in replacements for @cache / @lru_cache on bound methods.
Caches are stored in self.__dict__ so they are freed with the instance.
"""
import threading
from functools import lru_cache


def instance_cache(method):
    """Replacement for @cache on bound methods (unlimited, instance-stored)."""
    attr      = f'_icache_{method.__qualname__.replace(".", "_")}'
    lock_attr = attr + '_lock'

    def wrapper(self, *args):
        d = self.__dict__
        try:
            cache = d[attr]
        except KeyError:
            d.setdefault(lock_attr, threading.Lock())
            d.setdefault(attr, {})
            cache = d[attr]
        if args not in cache:
            with d[lock_attr]:
                if args not in cache:          # double-checked locking
                    cache[args] = method(self, *args)
        return cache[args]

    wrapper.__wrapped__  = method
    wrapper.__name__     = method.__name__
    wrapper.__qualname__ = method.__qualname__
    return wrapper


def instance_lru_cache(maxsize=128):
    """Replacement for @lru_cache(maxsize=N) on bound methods (bounded LRU, instance-stored).

    lru_cache is internally thread-safe; no additional locking is needed.
    The cycle (self → lru_cache → lambda → self) is handled by Python's cyclic GC.
    """
    def decorator(method):
        attr = f'_icache_{method.__qualname__.replace(".", "_")}'

        def wrapper(self, *args):
            try:
                bound = self.__dict__[attr]
            except KeyError:
                bound = lru_cache(maxsize=maxsize)(lambda *a: method(self, *a))
                self.__dict__[attr] = bound
            return bound(*args)

        wrapper.__wrapped__  = method
        wrapper.__name__     = method.__name__
        wrapper.__qualname__ = method.__qualname__
        return wrapper
    return decorator
