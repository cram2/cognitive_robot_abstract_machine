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


def test_constant_subquery_is_recomputed_once_per_outer_row(monkeypatch):
    """A constant (uncorrelated) subquery is re-evaluated once for every outer row rather than being
    computed a single time and cached. Pins the current cost the scoping change should remove.
    """
    product_evaluations = 0
    original_evaluate = Query._evaluate__

    def counting_evaluate(self, sources):
        nonlocal product_evaluations
        if self._is_compiled_product_:
            product_evaluations += 1
        return original_evaluate(self, sources)

    monkeypatch.setattr(Query, "_evaluate__", counting_evaluate)

    outer = variable(int, [1, 2, 3, 4, 5])
    constant_subquery = entity(eql.max(variable(int, [10, 20, 30])))
    query = entity(outer).where(outer < constant_subquery)

    assert query.tolist() == [1, 2, 3, 4, 5]
    # One evaluation of the outer product plus one of the subquery product per outer row.
    assert product_evaluations == 1 + 5
