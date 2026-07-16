from __future__ import annotations

import weakref
from functools import wraps

from typing_extensions import Callable, TypeVar

_Key = TypeVar("_Key")
_Value = TypeVar("_Value")

_MISSING = object()
"""
Sentinel distinguishing a cache miss from a cached ``None``/falsy value.
"""


def weak_key_cache(
    function: Callable[[_Key], _Value],
) -> Callable[[_Key], _Value]:
    """
    Memoize a single-argument function whose argument is a weak-referenceable key (e.g.
    a class).

    The cache holds the key *weakly*, so an entry is evicted as soon as the key is no longer
    referenced elsewhere. Unlike :func:`functools.lru_cache` (which keeps a strong reference to every
    key for the process lifetime), this does not pin transient keys — most importantly the
    dynamically created classes a long-running test session produces — so memory does not grow
    without bound.

    .. note::
        The key must be weak-referenceable (classes are; plain ``tuple``/``int`` are not). The cached
        value is returned as-is (no copy).

    :param function: A one-argument function keyed on a weak-referenceable value.
    :return: The memoized function, with a ``cache_clear`` attribute.
    """
    cache: "weakref.WeakKeyDictionary[_Key, _Value]" = weakref.WeakKeyDictionary()

    @wraps(function)
    def wrapper(key: _Key) -> _Value:
        cached = cache.get(key, _MISSING)
        if cached is not _MISSING:
            return cached
        result = function(key)
        cache[key] = result
        return result

    wrapper.cache_clear = cache.clear
    return wrapper
