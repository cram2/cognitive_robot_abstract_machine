"""
Tests that verify full chain traversal from a final nested-query result back through
all intermediate sub-query results, including results produced inside sub-query evaluation.

After removing source_operation_result, all navigation is via previous_operation_result only.

Chain structure for a two-stage pipeline built with entity(a).where(a > 1):

  r2  (stage-2 Variable result)
    .previous_operation_result
  → r1  (stage-1 Query result, slim bindings {a._id_: a_val})
      .previous_operation_result
    → product_result  (short-circuit Variable result carrying full bindings)
        .previous_operation_result
      → comparator_result  (= where_result, Where yields child results directly)
          .previous_operation_result
        → literal_result  (Literal(1) in cartesian product inside Comparator)
            .previous_operation_result
          → a_result  (Variable a evaluated inside Comparator)
"""

from dataclasses import dataclass

import pytest

from krrood.entity_query_language.factories import (
    and_,
    entity,
    inference,
    or_,
    variable_from,
)


@dataclass(frozen=True)
class Item:
    value: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first(expr, sources=None):
    return next(expr._evaluate_(sources))


def _all(expr, sources=None):
    return list(expr._evaluate_(sources))


def _chain(result):
    """Walk the previous_operation_result chain and return all nodes."""
    nodes = []
    node = result
    while node is not None:
        nodes.append(node)
        node = node.previous_operation_result
    return nodes


def _any_binding(result, key):
    """Return True if key appears in any node of the previous chain."""
    return any(key in node.bindings for node in _chain(result))


# ---------------------------------------------------------------------------
# 1. Basic pipeline: previous links stage-1 result to stage-2 result
# ---------------------------------------------------------------------------


def test_pipeline_previous_links_stage_results():
    """previous_operation_result on stage-2 Variable result points exactly at the stage-1 result."""
    a = variable_from([2, 3])
    q1 = entity(a).where(a > 1)
    q1.build()

    b = variable_from([10, 20])

    for r1 in q1._evaluate_():
        for r2 in b._evaluate_(r1):
            assert r2.previous_operation_result is r1


def test_pipeline_stage1_values_accessible_from_stage2():
    """Stage-1 variable binding is readable from the stage-2 result via all_bindings.

    Note: Query slim bindings store only the query's own _id_, not the original variable's
    _id_.  Use all_bindings to reach values from the internal evaluation chain.
    """
    a = variable_from([2, 3])
    q1 = entity(a).where(a > 1)
    q1.build()

    b = variable_from([10, 20])

    pairs = []
    for r1 in q1._evaluate_():
        for r2 in b._evaluate_(r1):
            a_val = r2.previous_operation_result.all_bindings[a._id_]
            b_val = r2.bindings[b._id_]
            pairs.append((a_val, b_val))

    assert len(pairs) == 4
    assert {p[0] for p in pairs} == {2, 3}
    assert {p[1] for p in pairs} == {10, 20}


# ---------------------------------------------------------------------------
# 2. Stage-1 internal chain reachable from stage-2
# ---------------------------------------------------------------------------


def test_stage1_product_result_reachable_from_stage2():
    """The product result inside stage-1 (which carries full bindings) is reachable."""
    a = variable_from([2])
    q1 = entity(a).where(a > 1)
    q1.build()

    b = variable_from([10])

    r1 = _first(q1)
    r2 = _first(b, r1)

    # r2.previous = r1; r1.previous = product_result inside q1
    product_result = r2.previous_operation_result.previous_operation_result
    assert product_result is not None
    assert a._id_ in product_result.bindings


def test_stage1_comparator_result_reachable_from_stage2():
    """
    The comparator/Where result from stage-1's internal evaluation is reachable
    via the previous chain from the stage-2 result.
    """
    a = variable_from([2])
    q1 = entity(a).where(a > 1)
    q1.build()

    b = variable_from([10])

    r1 = _first(q1)
    r2 = _first(b, r1)

    # r2.prev = r1; r1.prev = product_result; product_result.prev = comparator_result
    comparator_result = (
        r2.previous_operation_result.previous_operation_result.previous_operation_result  # r1  # product_result  # comparator/where result
    )
    assert comparator_result is not None
    assert a._id_ in comparator_result.bindings


def test_stage1_literal_result_reachable_from_stage2():
    """
    The literal result (Literal(1) evaluated inside the Comparator's cartesian product)
    is reachable in the previous chain from the stage-2 result.
    """
    a = variable_from([2])
    q1 = entity(a).where(a > 1)
    q1.build()

    b = variable_from([10])

    r1 = _first(q1)
    r2 = _first(b, r1)

    literal_result = (
        r2.previous_operation_result.previous_operation_result.previous_operation_result.previous_operation_result  # r1  # product_result  # comparator_result  # literal_result
    )
    assert literal_result is not None
    assert a._id_ in literal_result.bindings


def test_stage1_variable_result_reachable_via_previous_on_literal():
    """
    The innermost Variable-a result (produced inside the Comparator's sub-cartesian-product)
    is reachable via previous_operation_result on the literal result.
    """
    a = variable_from([2])
    q1 = entity(a).where(a > 1)
    q1.build()

    b = variable_from([10])

    r1 = _first(q1)
    r2 = _first(b, r1)

    literal_result = (
        r2.previous_operation_result.previous_operation_result.previous_operation_result.previous_operation_result
    )
    # Literal links the Variable-a result as its previous_operation_result
    a_result = literal_result.previous_operation_result
    assert a_result is not None
    assert a._id_ in a_result.bindings
    assert a_result.bindings[a._id_] == 2


# ---------------------------------------------------------------------------
# 3. Multi-stage pipeline
# ---------------------------------------------------------------------------


def test_three_stage_pipeline_previous_chain():
    """Three-stage pipeline: previous chain includes r2 directly, r1 via internal b_result."""
    a = variable_from([5])
    q1 = entity(a).where(a > 1)
    q1.build()

    b = variable_from([10])
    q2 = entity(b)
    q2.build()

    c = variable_from([100])

    r1 = _first(q1)
    r2 = _first(q2, r1)
    r3 = _first(c, r2)

    # c links r2 directly as its previous
    assert r3.previous_operation_result is r2

    # q2 links its internal b_result as previous; b_result links r1 as previous
    b_result_in_q2 = r3.previous_operation_result.previous_operation_result
    assert b_result_in_q2 is not None
    assert b_result_in_q2.previous_operation_result is r1

    # a's value is reachable via all_bindings (traverses full chain)
    assert r3.all_bindings[a._id_] == 5


def test_three_stage_pipeline_intermediate_reachable():
    """From stage-3, stage-1's internal (comparator) result is reachable via the linear chain."""
    a = variable_from([5])
    q1 = entity(a).where(a > 1)
    q1.build()

    b = variable_from([10])
    q2 = entity(b)
    q2.build()

    c = variable_from([100])

    r1 = _first(q1)
    r2 = _first(q2, r1)
    r3 = _first(c, r2)

    # Navigate: r3 → r2 → b_result_in_q2 → r1 → product_result_in_q1 → comparator_result
    comparator_result = (
        r3.previous_operation_result.previous_operation_result.previous_operation_result.previous_operation_result.previous_operation_result  # r2  # b_result in q2  # r1  # product_result in q1  # comparator/where result
    )
    assert comparator_result is not None
    assert a._id_ in comparator_result.bindings


# ---------------------------------------------------------------------------
# 4. AND condition: intermediate AND result reachable
# ---------------------------------------------------------------------------


def test_and_condition_chain_reachable():
    """With an AND condition, the AND result is present in the chain."""
    a = variable_from([6])
    q1 = entity(a).where(and_(a > 5, a < 10))
    q1.build()

    b = variable_from([1])

    r1 = _first(q1)
    r2 = _first(b, r1)

    # r2 links r1 as previous (b is a Variable)
    assert r2.previous_operation_result is r1
    assert _any_binding(r1.previous_operation_result, a._id_)


# ---------------------------------------------------------------------------
# 5. Inference (InstantiatedVariable) chain reachable
# ---------------------------------------------------------------------------


def test_inference_chain_reachable_from_pipeline():
    """InstantiatedVariable results carry previous chain across a pipeline."""
    val = variable_from([3, 7])
    item_inf = inference(Item)
    q1 = entity(item_inf(value=val))
    q1.build()

    b = variable_from([42])

    for r1 in q1._evaluate_():
        for r2 in b._evaluate_(r1):
            assert r2.previous_operation_result is r1
            # r1 must have a non-trivial previous chain (InstantiatedVariable → child vars)
            assert r1.previous_operation_result is not None


# ---------------------------------------------------------------------------
# 6. all_bindings exposes all stage values
# ---------------------------------------------------------------------------


def test_all_bindings_contains_all_stage_variable_values():
    """All variable values set in every stage are accessible via all_bindings.

    Query slim bindings only store the query's own _id_, so a._id_ lives in the
    previous_operation_result chain.  all_bindings exposes it.
    """
    a = variable_from([2])
    q1 = entity(a).where(a > 1)
    q1.build()

    b = variable_from([99])

    r1 = _first(q1)
    r2 = _first(b, r1)

    # b's value in r2's own bindings
    assert r2.bindings[b._id_] == 99

    # a's value reachable via all_bindings (traverses the full previous chain)
    assert a._id_ in r2.all_bindings
    assert r2.all_bindings[a._id_] == 2


def test_query_slim_bindings_vs_all_bindings():
    """Documents the slim-bindings characteristic: Query.bindings stores the query's own _id_,
    not the original variable's _id_.  all_bindings includes both via the previous chain.
    """
    a = variable_from([7])
    q1 = entity(a)
    q1.build()

    r1 = _first(q1)

    # Slim bindings: only the query's own ID
    assert q1._id_ in r1.bindings
    assert a._id_ not in r1.bindings

    # all_bindings: includes the original variable via previous_operation_result
    assert a._id_ in r1.all_bindings
    assert r1.all_bindings[a._id_] == 7
