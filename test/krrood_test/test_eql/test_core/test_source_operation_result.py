"""
Tests for previous_operation_result chain behavior.

Each expression links its incoming sources
directly as its own previous_operation_result, making the chain purely linear.

Key invariants:
- Variable._evaluate__(sources) links sources as previous_operation_result.
- Query._evaluate__(sources) links the internal child_result as previous_operation_result;
  sources is embedded deeper in the chain (child_result.previous = ... = sources).
- all_bindings traverses the full previous_operation_result chain.
"""

from dataclasses import dataclass

import pytest

from krrood.entity_query_language.core.base_expressions import OperationResult
from krrood.entity_query_language.factories import (
    and_,
    entity,
    inference,
    variable_from,
)


@dataclass(frozen=True)
class Item:
    value: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_result(expr, sources=None):
    """Return the first OperationResult from _evaluate_()."""
    return next(expr._evaluate_(sources))


def _all_results(expr, sources=None):
    return list(expr._evaluate_(sources))


# ---------------------------------------------------------------------------
# Variable: sources becomes previous_operation_result
# ---------------------------------------------------------------------------


def test_variable_previous_is_none_on_top_level_evaluate():
    """Top-level Variable evaluation: previous chain has no real bindings from another stage."""
    var = variable_from([1, 2])
    for result in var._evaluate_():
        # previous_operation_result may be the empty sentinel; no external stage bindings appear
        assert result.bindings[var._id_] in {1, 2}
        # all_bindings does not include bindings from a prior stage
        other_var = variable_from([99])
        assert other_var._id_ not in result.all_bindings


def test_variable_previous_is_sources_when_operation_result_passed():
    """Variable links incoming OperationResult as its previous_operation_result."""
    source_var = variable_from([99])
    incoming = _first_result(source_var)

    target_var = variable_from([1, 2, 3])
    for result in _all_results(target_var, incoming):
        assert result.previous_operation_result is incoming


def test_variable_previous_is_sources_when_empty_operation_result_passed():
    """Variable links an empty OperationResult as previous (sentinel case is still OperationResult)."""
    var = variable_from([1, 2])
    sentinel = OperationResult({})
    for result in _all_results(var, sentinel):
        assert result.previous_operation_result is sentinel


# ---------------------------------------------------------------------------
# Query: sources is embedded deeper in the chain
# ---------------------------------------------------------------------------


def test_query_sources_reachable_via_all_bindings():
    """Query results: incoming sources are reachable via all_bindings even though
    the query's previous_operation_result points at the internal child_result."""
    source_var = variable_from([99])
    incoming = _first_result(source_var)

    val = variable_from([10, 20])
    query = entity(val)
    query.build()
    for result in _all_results(query, incoming):
        assert source_var._id_ in result.all_bindings
        assert result.all_bindings[source_var._id_] == 99


def test_query_with_where_sources_reachable_via_all_bindings():
    """Query+where results: incoming sources propagate into all_bindings."""
    source_var = variable_from([99])
    incoming = _first_result(source_var)

    val = variable_from([1, 6, 11])
    query = entity(val).where(val > 5)
    query.build()
    true_results = [r for r in _all_results(query, incoming) if r.is_true]
    assert len(true_results) == 2
    for result in true_results:
        assert source_var._id_ in result.all_bindings
        assert result.all_bindings[source_var._id_] == 99


# ---------------------------------------------------------------------------
# Short-circuit path: previous_operation_result is the incoming result
# ---------------------------------------------------------------------------


def test_short_circuit_previous_is_incoming():
    """Short-circuit path (ID already in bindings) links incoming as previous."""
    var = variable_from([42])
    incoming = _first_result(var)
    assert var._id_ in incoming.bindings

    results = _all_results(var, incoming)
    assert len(results) == 1
    assert results[0].previous_operation_result is incoming


# ---------------------------------------------------------------------------
# Chain traversal
# ---------------------------------------------------------------------------


def test_variable_chain_is_traversable():
    """previous_operation_result forms a traversable chain across evaluation stages."""
    v1 = variable_from([1])
    r1 = _first_result(v1)

    v2 = variable_from([10, 20])
    r2_list = _all_results(v2, r1)
    assert len(r2_list) == 2
    for r2 in r2_list:
        assert r2.previous_operation_result is r1


def test_chain_two_hops():
    """Three-stage Variable pipeline: each result links to previous stage."""
    v1 = variable_from([1])
    r1 = _first_result(v1)

    v2 = variable_from([10])
    r2 = _first_result(v2, r1)
    assert r2.previous_operation_result is r1

    v3 = variable_from([100])
    r3 = _first_result(v3, r2)
    assert r3.previous_operation_result is r2
    assert r3.previous_operation_result.previous_operation_result is r1


# ---------------------------------------------------------------------------
# all_bindings traverses the full chain
# ---------------------------------------------------------------------------


def test_all_bindings_reaches_upstream_variable():
    """all_bindings traverses the full previous chain to expose upstream variable values."""
    source_var = variable_from([99])
    incoming = _first_result(source_var)

    var = variable_from([7])
    result = _first_result(var, incoming)

    assert var._id_ in result.all_bindings
    assert source_var._id_ in result.all_bindings
    assert result.all_bindings[var._id_] == 7
    assert result.all_bindings[source_var._id_] == 99


def test_all_bindings_does_not_affect_own_bindings():
    """Passing an OperationResult as sources does not change own binding values."""
    source_var = variable_from([99])
    incoming = _first_result(source_var)

    var = variable_from([7])
    result_with_source = _first_result(var, incoming)
    result_without_source = _first_result(var)

    assert result_with_source.bindings[var._id_] == result_without_source.bindings[var._id_]
    assert result_with_source.is_true == result_without_source.is_true


# ---------------------------------------------------------------------------
# __eq__ does not include source_operation_result (field removed)
# ---------------------------------------------------------------------------


def test_results_with_same_content_are_equal_regardless_of_chain():
    """Two Variable results from the same domain with the same previous chain are equal."""
    var = variable_from([7])
    r1 = _first_result(var)
    r2 = _first_result(var)
    assert r1 == r2


# ---------------------------------------------------------------------------
# Inference (InstantiatedVariable) sources propagate via chain
# ---------------------------------------------------------------------------


def test_inference_sources_reachable_via_all_bindings():
    """InstantiatedVariable results: incoming sources reachable via all_bindings."""
    source_var = variable_from([0])
    incoming = _first_result(source_var)

    val = variable_from([3, 7])
    item_inf = inference(Item)
    query = entity(item_inf(value=val))
    query.build()
    results = [r for r in _all_results(query, incoming) if r.is_true]
    assert len(results) == 2
    for r in results:
        assert source_var._id_ in r.all_bindings
        assert r.all_bindings[source_var._id_] == 0
