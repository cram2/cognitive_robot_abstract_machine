from __future__ import annotations

import threading
import weakref
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial, wraps

from typing_extensions import Any, Callable, ClassVar, TypeVar

from krrood.patterns.exceptions import UnmemoizableOwnerError

_Key = TypeVar("_Key")
_Value = TypeVar("_Value")

_MISSING = object()
"""
Sentinel distinguishing a cache miss from a cached ``None``/falsy value.
"""

# %% caching keyed on a weakly held key


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
    cache: weakref.WeakKeyDictionary[_Key, _Value] = weakref.WeakKeyDictionary()

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


# %% per-owner memoization

TCallable = TypeVar("TCallable", bound=Callable[..., Any])


@dataclass(frozen=True)
class MemoizationKey:
    """
    Identifies one memoized call within the cache of a single owner.
    """

    function: Callable[..., Any]
    """
    The undecorated function whose result is cached.
    """

    arguments: tuple[Any, ...]
    """
    The positional arguments the call was made with, excluding the owner.
    """

    keyword_arguments: frozenset[tuple[str, Any]]
    """
    The keyword arguments the call was made with, as name-value pairs.
    """


@dataclass
class MemoizedValue:
    """
    The result of one memoized call, together with the lock that keeps its computation
    and its copies single-threaded.
    """

    lock: threading.RLock = field(default_factory=threading.RLock)
    """
    Guards this one result: at most one thread computes it, and at most one thread
    copies it, at a time.

    Reentrant so a memoized call that re-enters itself fails with a recursion error
    rather than silently deadlocking on its own frame.
    """

    value: Any = _MISSING
    """
    The computed result, or the sentinel while it has not been computed yet.
    """

    def resolve(self, computation: Callable[[], Any]) -> Any:
        """
        Returns the result, running the computation if it has not run yet.

        :param computation: The call that produces the result on the first request.
        """
        with self.lock:
            if self.value is _MISSING:
                self.value = computation()
            return self.value

    def resolve_copy(self, computation: Callable[[], Any]) -> Any:
        """
        Returns an independent copy of the result, running the computation if it has not
        run yet.

        Copying happens under the lock so one cached value is never copied by two
        threads at once.

        :param computation: The call that produces the result on the first request.
        """
        with self.lock:
            if self.value is _MISSING:
                self.value = computation()
            return deepcopy(self.value)


@dataclass
class MemoizationCache:
    """
    The memoized results of one owner.

    An owner keeps its cache as an attribute of its own, never one inherited from its
    class, so a class never shares a cache with its subclasses or with its instances.
    The cache copies and pickles as an empty one, so memoizing a call does not stop an
    owner from being copied.
    """

    attribute_name: ClassVar[str] = "__memoization_cache__"
    """
    The attribute an owner keeps its cache under.

    Spelled as a dunder because a parametrized generic forwards the assignment of every
    other name to the class it parametrizes, which would make all of its
    parametrizations share one cache.
    """

    registration_lock: ClassVar[threading.Lock] = threading.Lock()
    """
    Serialises the creation of an owner's cache, so two threads racing to memoize a call
    on the same owner cannot end up with a cache each.
    """

    entries: dict[MemoizationKey, MemoizedValue] = field(default_factory=dict)
    """
    The result of every memoized call made on the owner so far.
    """

    entries_lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Guards the entry table itself.

    Never held while a memoized call runs, so unrelated keys do not wait for each other.
    """

    @classmethod
    def of(cls, owner: Any, function_name: str) -> MemoizationCache:
        """
        Returns the owner's cache, creating it on first use.

        :param owner: The receiver of the memoized call.
        :param function_name: The name of the memoized function, for error reporting.
        :raises UnmemoizableOwnerError: If the owner cannot hold a cache.
        """
        cache = cls.existing_for(owner)
        if cache is not None:
            return cache
        if not cls.can_own_a_cache(owner):
            raise UnmemoizableOwnerError(owner, function_name)
        with cls.registration_lock:
            cache = cls.existing_for(owner)
            if cache is None:
                cache = cls()
                setattr(owner, cls.attribute_name, cache)
            return cache

    @classmethod
    def existing_for(cls, owner: Any) -> MemoizationCache | None:
        """
        Looks up an owner's own cache, ignoring the cache of its class or of any class
        it inherits from.

        :param owner: The object to look up the cache of.
        :return: The owner's cache, or None if it has never memoized a call.
        """
        if not cls.can_own_a_cache(owner):
            return None
        return vars(owner).get(cls.attribute_name)

    @classmethod
    def can_own_a_cache(cls, owner: Any) -> bool:
        """
        :param owner: The receiver of a memoized call.
        :return: Whether the owner can store attributes of its own.
        """
        return type(owner).__dictoffset__ != 0

    def entry_for(self, key: MemoizationKey) -> MemoizedValue:
        """
        Returns the entry holding the result of one call, creating it on first use.

        :param key: The call to look up.
        """
        entry = self.entries.get(key)
        if entry is not None:
            return entry
        with self.entries_lock:
            return self.entries.setdefault(key, MemoizedValue())

    def clear(self) -> None:
        """
        Forgets every memoized result.

        A computation that is still running keeps the entry it started on, so its result
        is discarded rather than resurrecting the cleared cache.
        """
        with self.entries_lock:
            self.entries.clear()

    def __len__(self) -> int:
        """
        :return: The number of calls whose result the owner has cached.
        """
        return len(self.entries)

    def __deepcopy__(self, memo: dict[int, Any]) -> MemoizationCache:
        """
        Returns an empty cache, so a copy of an owner starts out with nothing cached
        instead of inheriting results computed for the original.
        """
        return MemoizationCache()

    def __reduce__(self) -> tuple[Callable[[], MemoizationCache], tuple[()]]:
        """
        Pickles as an empty cache, since a lock cannot be pickled and cached results do
        not survive the owner they were computed for.
        """
        return MemoizationCache, ()


def _wrap_with_memoization(
    function: TCallable,
    resolve: Callable[[MemoizedValue, Callable[[], Any]], Any],
) -> TCallable:
    """
    Wraps a function so its result is looked up in, and computed into, the cache of the
    owner it is called on.

    :param function: The function whose result is cached.
    :param resolve: How the entry hands out its result, which is what separates
        :func:`memoize` from :func:`copy_memoize`.
    """

    @wraps(function)
    def wrapper(owner: Any, *arguments: Any, **keyword_arguments: Any) -> Any:
        key = MemoizationKey(function, arguments, frozenset(keyword_arguments.items()))
        entry = MemoizationCache.of(owner, function.__name__).entry_for(key)
        return resolve(entry, partial(function, owner, *arguments, **keyword_arguments))

    return wrapper  # type: ignore


def memoize(function: TCallable) -> TCallable:
    """
    Caches the return value of a function call for the owner it was called on.

    Thread-safe: each cached result is computed once, however many threads ask for it at
    the same time, and results of different calls are computed independently.
    """
    return _wrap_with_memoization(function, MemoizedValue.resolve)


def copy_memoize(function: TCallable) -> TCallable:
    """
    Caches the return value of a function call for the owner it was called on, but
    returns a copy of it.

    Thread-safe like :func:`memoize`, and additionally never copies one cached value
    from two threads at once.
    """
    return _wrap_with_memoization(function, MemoizedValue.resolve_copy)


def clear_memoization_cache(owner: Any) -> None:
    """
    Forgets every result memoized for an owner.

    :param owner: The object whose memoized results are no longer valid.
    """
    cache = MemoizationCache.existing_for(owner)
    if cache is None:
        return
    cache.clear()
