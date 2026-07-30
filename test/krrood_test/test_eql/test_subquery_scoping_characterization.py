"""Characterization tests pinning the *current* subquery-scoping behaviour, as the baseline for the
scoping spike.

An embedded subquery is isolated from its enclosing query by the result quantifier, which drops the
incoming source bindings. These tests document, without changing anything, the three consequences of
that design so the spike can preserve the correctness and improve on the rest:

- the isolation is *correct* for aggregation: an aggregating subquery ranges over its variable's full
  domain rather than the outer row's single binding;
- the isolation is *total*: a subquery never correlates with the outer row;
- the isolation is *uncached*: a constant (uncorrelated) subquery is recomputed once per outer row.
"""

from __future__ import annotations

import pytest

import krrood.entity_query_language.factories as eql
from krrood.entity_query_language.factories import entity, variable
from krrood.entity_query_language.query.query import Query


def test_shared_leaf_aggregation_subquery_ranges_over_the_full_domain():
    """An aggregating subquery that shares its range variable with the outer query aggregates over
    the variable's full domain, not the outer row's single binding."""
    variable_shared_with_outer = variable(int, [1, 2, 3])
    query = entity(variable_shared_with_outer).where(
        variable_shared_with_outer == entity(eql.max(variable_shared_with_outer))
    )
    assert query.tolist() == [3]


def test_embedded_subquery_does_not_correlate_with_the_outer_row():
    """The outer row's binding does not flow into an embedded subquery. ``max(o)`` is the global
    maximum (3) for every outer row, so ``o < max(o)`` keeps ``[1, 2]``; a correlated ``max(o) == o``
    would instead keep nothing."""
    outer = variable(int, [1, 2, 3])
    query = entity(outer).where(outer < entity(eql.max(outer)))
    assert query.tolist() == [1, 2]


def test_isolation_is_owned_by_the_query_scope_not_the_quantifier(monkeypatch):
    """The result quantifier forwards the outer row's bindings into the nested query rather than
    dropping them; the query scope is what isolates the subquery, which still ranges over its full
    domain."""
    sources_seen_by_compiled_queries = []
    original_evaluate = Query._evaluate__

    def recording_evaluate(self, sources):
        if self._is_compiled_product_:
            sources_seen_by_compiled_queries.append(dict(sources.bindings))
        yield from original_evaluate(self, sources)

    monkeypatch.setattr(Query, "_evaluate__", recording_evaluate)

    outer = variable(int, [1, 2, 3])
    query = entity(outer).where(outer < entity(eql.max(outer)))

    assert query.tolist() == [1, 2]
    # The nested subquery received forwarded outer bindings (non-empty), proving the quantifier no
    # longer performs the isolating source-cut.
    assert any(bindings for bindings in sources_seen_by_compiled_queries)


def test_constant_subquery_is_computed_once_per_evaluation(monkeypatch):
    """A constant (uncorrelated) subquery is computed a single time per top-level evaluation and its
    results reused across outer rows, rather than recomputed once per row."""
    product_computations = 0
    original_product = Query._evaluate_product_

    def counting_product(self, sources):
        nonlocal product_computations
        product_computations += 1
        return original_product(self, sources)

    monkeypatch.setattr(Query, "_evaluate_product_", counting_product)

    outer = variable(int, [1, 2, 3, 4, 5])
    constant_subquery = entity(eql.max(variable(int, [10, 20, 30])))
    query = entity(outer).where(outer < constant_subquery)

    assert query.tolist() == [1, 2, 3, 4, 5]
    # One computation of the outer product plus one of the subquery product, reused across rows.
    assert product_computations == 1 + 1
