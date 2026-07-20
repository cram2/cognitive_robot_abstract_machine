"""Safety tests for the per-evaluation subquery result cache: re-executability, no retained state
across evaluations, and the time/memory profile of caching an uncorrelated subquery.

The cache lives in the per-evaluation context, so it is created fresh for each top-level evaluation
and released when that evaluation ends. These tests pin that lifecycle so the speed-up never costs
correctness or leaks memory.
"""

from __future__ import annotations

import gc
import time
import tracemalloc

import krrood.entity_query_language.factories as eql
from krrood.entity_query_language.factories import entity, variable
from krrood.entity_query_language.evaluation_context import get_evaluation_context
from krrood.entity_query_language.query import query as query_module
from krrood.entity_query_language.query.query import CachedResultStream


def _query_with_constant_subquery(outer_size: int, subquery_domain_size: int):
    outer = variable(int, list(range(outer_size)))
    constant_subquery = entity(
        eql.max(variable(int, list(range(subquery_domain_size))))
    )
    return entity(outer).where(outer < constant_subquery)


def test_query_with_cached_subquery_is_re_executable():
    """Evaluating the same query repeatedly yields the same results each time; the cache is scoped to
    a single evaluation and never leaks stale results into the next."""
    query = _query_with_constant_subquery(outer_size=5, subquery_domain_size=4)
    first = query.tolist()
    assert query.tolist() == first
    assert query.tolist() == first
    assert first == [0, 1, 2]


def test_editing_then_re_evaluating_reflects_the_change():
    """A cached evaluation does not pin the query: editing and re-evaluating recomputes."""
    value = variable(int, [1, 2, 3, 4])
    query = entity(value).where(value < 3)
    assert query.tolist() == [1, 2]
    query.where(value > 1)
    assert query.tolist() == [2]


def test_evaluation_context_is_released_after_evaluation():
    """The evaluation context (which holds the cache) does not outlive the evaluation."""
    query = _query_with_constant_subquery(outer_size=5, subquery_domain_size=4)
    query.tolist()
    assert get_evaluation_context() is None


def test_no_cached_streams_are_retained_after_evaluation():
    """Every cached stream created during an evaluation is released once it ends, so the cache cannot
    accumulate across evaluations."""
    query = _query_with_constant_subquery(outer_size=5, subquery_domain_size=4)

    gc.collect()
    streams_before = sum(
        1 for obj in gc.get_objects() if isinstance(obj, CachedResultStream)
    )
    query.tolist()
    gc.collect()
    streams_after = sum(
        1 for obj in gc.get_objects() if isinstance(obj, CachedResultStream)
    )

    assert streams_after == streams_before


def test_repeated_evaluations_do_not_accumulate_memory():
    """Re-evaluating in a loop does not grow retained memory: the cache from each evaluation is
    released before the next, so there is no leak."""
    query = _query_with_constant_subquery(outer_size=50, subquery_domain_size=200)

    tracemalloc.start()
    query.tolist()  # warm up one-off allocations (builders, caches)
    gc.collect()
    current_before, _ = tracemalloc.get_traced_memory()
    for _ in range(25):
        query.tolist()
    gc.collect()
    current_after, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert current_after - current_before < 200_000


def test_caching_makes_an_expensive_uncorrelated_subquery_faster():
    """Caching computes the expensive subquery once instead of once per outer row, so the cached
    evaluation is substantially faster than the same evaluation with caching disabled.
    """

    def evaluate_once():
        return _query_with_constant_subquery(
            outer_size=120, subquery_domain_size=1500
        ).tolist()

    start = time.perf_counter()
    evaluate_once()
    cached_seconds = time.perf_counter() - start

    # Disable the cache by hiding the evaluation context from the query's caching branch only.
    original_get_context = query_module.get_evaluation_context
    query_module.get_evaluation_context = lambda: None
    try:
        start = time.perf_counter()
        evaluate_once()
        uncached_seconds = time.perf_counter() - start
    finally:
        query_module.get_evaluation_context = original_get_context

    assert uncached_seconds > cached_seconds * 3
