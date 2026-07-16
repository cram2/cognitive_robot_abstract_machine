"""
Tests that demonstrate binding loss when an InstantiatedVariable (inference) uses a
Query (entity(...).where(...)) as a child variable.

Root cause:
  cartesian_product_while_passing_the_bindings_around calls result.update(bindings) which
  only copies the flat .bindings dict. When a child is a Query, the Query produces SLIM
  bindings ({query._id_: value}), so the inner variable's _id_ is never accumulated.
  The outer InstantiatedVariable then builds its OperationResult from those slim bindings,
  and all_bindings (which merges only one level back) cannot reach the inner variable's ID.

This exactly mirrors the failing semantic annotation test:
  condition.left._child_._id_ (e.g. FixedConnection variable) is missing from
  explanation.operation_result.all_bindings because the Drawer inference used a
  Query-wrapped child variable.
"""

from dataclasses import dataclass

import pytest

from krrood.entity_query_language.factories import entity, inference, variable_from


@dataclass(frozen=True)
class Inner:
    value: int


@dataclass(frozen=True)
class Outer:
    inner: Inner


# ---------------------------------------------------------------------------
# Baseline: no intermediate Query — inner var IS in all_bindings
# ---------------------------------------------------------------------------


def test_inner_var_reachable_without_intermediate_query():
    """
    Without a Query wrapper, the inner variable's _id_ is in all_bindings.
    """
    val = variable_from([2, 3])
    inner_expr = inference(Inner)(value=val)  # InstantiatedVariable, has _id_

    outer_query = entity(inference(Outer)(inner=inner_expr))
    outer_query.build()

    results = [r for r in outer_query._evaluate_() if r.is_true]
    assert len(results) == 2

    for result in results:
        # With no Query wrapper, val._id_ accumulates through the cartesian product
        assert (
            val._id_ in result.all_bindings
        ), "val._id_ should be reachable via all_bindings when no Query sits in between"
        assert (
            inner_expr._id_ in result.all_bindings or inner_expr._id_ in result.bindings
        )


# ---------------------------------------------------------------------------
# Bug: intermediate Query strips inner bindings from all_bindings
# ---------------------------------------------------------------------------


def test_inner_var_reachable_through_query_wrapper_after_fix():
    """
    After fixing Variable to link sources as previous_operation_result and all_bindings
    to traverse the full previous chain, val._id_ and inner_expr._id_ are now reachable
    even when an intermediate Query (with slim bindings) sits between them and the outer
    result.

    This fixes the failure in test_explain_inferred_semantic_annotations where
    condition.left._child_._id_ (FixedConnection variable) was missing from
    explanation.operation_result.all_bindings.
    """
    val = variable_from([2, 3])
    inner_expr = inference(Inner)(value=val)  # InstantiatedVariable

    # Wrap the inference in a Query — this produces SLIM bindings
    inner_query = entity(inner_expr).where(val > 1)
    inner_query.build()

    outer_query = entity(inference(Outer)(inner=inner_query))
    outer_query.build()

    results = [r for r in outer_query._evaluate_() if r.is_true]
    assert len(results) == 2

    for result in results:
        # inner_query._id_ IS present (slim binding from the Query)
        assert inner_query._id_ in result.all_bindings

        # After the fix: val._id_ and inner_expr._id_ are now reachable via deep traversal
        assert (
            val._id_ in result.all_bindings
        ), "val._id_ should now be reachable via all_bindings deep traversal"
        assert (
            inner_expr._id_ in result.all_bindings
        ), "inner_expr._id_ should now be reachable via all_bindings deep traversal"


def test_slim_query_binding_and_deep_binding_after_fix():
    """
    After fixing Variable to link sources as previous_operation_result, all bindings are now
    reachable via all_bindings even when an intermediate Query (with slim bindings) sits in the chain.

    Previously: all_bindings = previous_operation_result.bindings | self.bindings
    (only one level back, so val._id_ was lost when Query produced slim bindings)

    Now: all_bindings traverses the full previous_operation_result chain linearly,
    so val._id_ is always accessible.
    """
    val = variable_from([5])
    inner_expr = inference(Inner)(value=val)  # InstantiatedVariable

    inner_query = entity(inner_expr).where(val > 1)
    inner_query.build()

    outer_query = entity(inference(Outer)(inner=inner_query))
    outer_query.build()

    result = next(r for r in outer_query._evaluate_() if r.is_true)

    # Slim query binding IS present
    assert inner_query._id_ in result.all_bindings
    assert result.all_bindings[inner_query._id_] == Inner(value=5)

    # After the fix: val and inner_expr ARE now in all_bindings via deep traversal
    assert (
        val._id_ in result.all_bindings
    ), "val._id_ should now be reachable via all_bindings deep graph traversal"
    assert (
        inner_expr._id_ in result.all_bindings
    ), "inner_expr._id_ should now be reachable via all_bindings deep graph traversal"
    assert result.all_bindings[val._id_] == 5


# ---------------------------------------------------------------------------
# The fix surface: all_bindings should walk the full previous chain
# ---------------------------------------------------------------------------


def test_deep_all_bindings_now_works():
    """
    After the fix, all_bindings traverses the entire previous_operation_result chain
    linearly, so val._id_ is now directly accessible via result.all_bindings.
    """
    val = variable_from([7])
    inner_expr = inference(Inner)(value=val)  # InstantiatedVariable

    inner_query = entity(inner_expr).where(val > 1)
    inner_query.build()

    outer_query = entity(inference(Outer)(inner=inner_query))
    outer_query.build()

    result = next(r for r in outer_query._evaluate_() if r.is_true)

    # After the fix: all_bindings directly exposes val._id_ via full graph traversal
    assert val._id_ in result.all_bindings, (
        "val._id_ should be directly reachable via result.all_bindings "
        "after the deep traversal fix"
    )
    assert result.all_bindings[val._id_] == 7
