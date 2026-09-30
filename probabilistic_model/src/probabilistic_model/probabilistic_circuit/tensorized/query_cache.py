from __future__ import annotations

import functools
from dataclasses import dataclass, field

from typing_extensions import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class QueryCacheKey:
    """
    Which query, evaluated for which layer, an entry of a :class:`QueryCache` is the
    result of.
    """

    query: Callable
    """
    The method that evaluated the query.
    """

    layer_id: int
    """
    The :func:`id` of the layer it was evaluated for.

    Layers are keyed by identity rather than by value: a pass has to evaluate a layer
    that is the child of several parents once, and two layers that happen to hold equal
    parameters are still two layers with two results.
    """


@dataclass
class QueryCache:
    """
    The results one pass over the layers has computed so far.

    Layers form a directed acyclic graph, not a tree: a layer that is the child of
    several parents must only be evaluated once per pass. Every method that walks the
    graph takes this cache as a ``cache`` keyword argument and hands it down to the
    calls it makes on its own children; the top level caller may omit it and gets a
    fresh one.
    """

    results: Dict[QueryCacheKey, Any] = field(default_factory=dict)
    """
    The result of every query evaluated so far, per layer.
    """


def memoized(method: Callable) -> Callable:
    """
    Memoize a query of a layer by the method and the identity of the layer.

    :param method: The method that evaluates the query.
    :return: The memoized method.
    """

    @functools.wraps(method)
    def wrapper(self, *args, cache: Optional[QueryCache] = None, **kwargs):
        if cache is None:
            cache = QueryCache()
        key = QueryCacheKey(method, id(self))
        if key not in cache.results:
            cache.results[key] = method(self, *args, cache=cache, **kwargs)
        return cache.results[key]

    return wrapper
