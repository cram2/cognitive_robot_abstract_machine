"""
Tests for the cache of a pass over the layers: a memoized query stores what its cache
makes of the result.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

from typing_extensions import Any

from probabilistic_model.probabilistic_circuit.tensorized.query_cache import (
    QueryCache,
    QueryCacheKey,
    memoized,
)

# %% mimics


@dataclass(kw_only=True)
class DoublingQueryCache(QueryCache):
    """
    A cache that stores twice the result of every query.
    """

    def stored_result(self, *, key: QueryCacheKey, result: Any) -> Any:
        """
        Twice the result, see :meth:`QueryCache.stored_result`.
        """
        return 2 * result


@dataclass(kw_only=True)
class FixedAnswer:
    """
    A layer with one memoized query that answers a fixed number.
    """

    answer_value: int = 21
    """
    What the query answers.
    """

    @memoized
    def answer(self, *, cache: QueryCache | None = None) -> int:
        """
        :param cache: The cache of the current pass.
        :return: :attr:`answer_value`.
        """
        return self.answer_value


# %% stored results


def test_a_query_returns_and_caches_its_result_unchanged_by_default():
    """
    A plain cache stores the result itself.
    """
    layer = FixedAnswer()
    cache = QueryCache()

    assert layer.answer(cache=cache) == layer.answer_value
    assert cache.result_of(query=FixedAnswer.answer, layer=layer) == layer.answer_value


def test_a_query_returns_and_caches_what_its_cache_stores():
    """
    A cache that changes results changes what the query returns and what later calls in
    the same pass read.
    """
    layer = FixedAnswer()
    cache = DoublingQueryCache()

    assert layer.answer(cache=cache) == 2 * layer.answer_value
    assert layer.answer(cache=cache) == 2 * layer.answer_value


def test_the_key_of_a_query_names_the_query_and_the_layer():
    """
    The key a cache sees for a query of a layer is the one
    :meth:`QueryCacheKey.from_query` makes.
    """
    layer = FixedAnswer()
    keys = []

    @dataclass(kw_only=True)
    class RecordingQueryCache(QueryCache):
        """
        A cache that records the key of every stored result.
        """

        def stored_result(self, *, key: QueryCacheKey, result: Any) -> Any:
            keys.append(key)
            return result

    layer.answer(cache=RecordingQueryCache())

    assert keys == [QueryCacheKey.from_query(query=FixedAnswer.answer, layer=layer)]
