"""
The results of the queries one pass over the layers of a circuit has evaluated, which
the memoized queries of the layers read and store.
"""

from __future__ import annotations

# %% imports
import functools
from dataclasses import dataclass, field

from typing_extensions import Any, Callable

# %% cache


@dataclass(frozen=True, kw_only=True)
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

    @classmethod
    def from_query(cls, *, query: Callable, layer: Any) -> QueryCacheKey:
        """
        :param query: A memoized query of a layer type.
        :param layer: A layer the query is evaluated for.
        :return: The key of the result of the query for the layer.
        """
        return cls(query=query.__wrapped__, layer_id=id(layer))


@dataclass(kw_only=True)
class QueryCache:
    """
    The results one pass over the layers has computed so far.

    Layers form a directed acyclic graph, not a tree: a layer that is the child of
    several parents must only be evaluated once per pass. Every method that walks the
    graph takes this cache as a ``cache`` keyword argument and hands it down to the
    calls it makes on its own children; the top level caller may omit it and gets a
    fresh one.
    """

    results: dict[QueryCacheKey, Any] = field(default_factory=dict)
    """
    The result of every query evaluated so far, per layer.
    """

    def result_of(self, *, query: Callable, layer: Any) -> Any:
        """
        :param query: A memoized query of a layer type, such as
            :meth:`~probabilistic_model.probabilistic_circuit.tensorized.inner_layer.product_layer.ProductLayer.marginal`.
        :param layer: A layer the query was evaluated for in this pass.
        :return: What the query returned for the layer.
        """
        return self.results[QueryCacheKey.from_query(query=query, layer=layer)]

    def stored_result(self, *, key: QueryCacheKey, result: Any) -> Any:
        """
        :param key: Which query, evaluated for which layer, gave the result.
        :param result: What the query computed.
        :return: What the pass stores and returns as the result, the result itself; a
            cache that changes results on their way up the circuit overrides it.
        """
        return result


# %% memoized queries


def memoized(method: Callable) -> Callable:
    """
    Memoize a query of a layer by the method and the identity of the layer.

    :param method: The method that evaluates the query.
    :return: The memoized method.
    """

    @functools.wraps(method)
    def wrapper(
        self: Any, *args: Any, cache: QueryCache | None = None, **kwargs: Any
    ) -> Any:
        """
        :return: The result of the query for this layer in the pass of the cache,
            evaluated once per pass.
        """
        if cache is None:
            cache = QueryCache()
        key = QueryCacheKey(query=method, layer_id=id(self))
        if key not in cache.results:
            cache.results[key] = cache.stored_result(
                key=key, result=method(self, *args, cache=cache, **kwargs)
            )
        return cache.results[key]

    return wrapper
