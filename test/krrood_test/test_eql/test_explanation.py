import gc
import operator
from dataclasses import dataclass

import pytest

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.evaluation import is_condition_participant
from krrood.entity_query_language._stack import CallStack
from krrood.entity_query_language._monitoring import monitored
from krrood.entity_query_language.explanation.explanation import (
    explain_inference,
    register_inference,
)
from krrood.entity_query_language.factories import (
    inference,
    entity,
    variable_from,
    and_,
    or_,
    not_,
    variable,
    a,
    an,
)
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import AND, OR
from krrood.entity_query_language.query.query import Query
from krrood.entity_query_language.query_graph import QueryGraph
from krrood.symbol_graph.symbol_graph import Symbol
from ..dataset.semantic_world_like_classes import (
    Handle,
    PrismaticConnection,
    FixedConnection,
    Drawer,
)


@dataclass(unsafe_hash=True)
class Person(Symbol):
    name: str


@dataclass(unsafe_hash=True)
class Item(Symbol):
    value: int


def test_explain_inference_basic():
    """
    Test that explain_inference correctly records and retrieves the stack for a simple
    inference.
    """
    # 1. Define the query
    # The stack captured should point here
    person_factory = inference(Person)
    query = entity(person_factory(name="John"))

    # 2. Evaluate the query to trigger instance creation
    results = list(query.evaluate())
    assert len(results) == 1
    john = results[0]

    # 3. Check explanation
    explanation_obj = explain_inference(john)
    assert explanation_obj is not None
    explanation = explanation_obj.as_string()

    # function name appears in the "Query source (function_name, ...)" context line
    assert "Person" in explanation
    assert "test_explain_inference_basic" in explanation
    assert 'person_factory(name="John")' in explanation


def test_explain_inference_nested():
    """
    Test that explain_inference correctly records and retrieves the stack through nested
    function calls.
    """

    def create_person_query(name):
        return person_factory_helper(name)

    def person_factory_helper(name):
        person_inf = inference(Person)
        return person_inf(name=name)

    # Define query through nested calls
    p_var = create_person_query("Alice")
    query = entity(p_var)

    results = list(query.evaluate())
    assert len(results) == 1
    alice = results[0]

    explanation_obj = explain_inference(alice)
    assert explanation_obj is not None
    explanation = explanation_obj.as_string()

    # innermost user frame is person_factory_helper — appears in query source context
    assert "Person" in explanation
    assert "person_factory_helper" in explanation
    assert "person_inf(name=name)" in explanation
    # calling chain only visible in stack trace
    explanation_with_trace = explanation_obj.as_string(show_trace=True)
    assert "test_explain_inference_nested" in explanation_with_trace
    assert "create_person_query" in explanation_with_trace


def test_explain_inference_multiple_instances():
    """
    Test that different instances from the same inference variable have the same stack
    in their explanation.
    """
    from krrood.entity_query_language.factories import variable_from

    names = variable_from(["Bob", "Charlie"])
    person_inf = inference(Person)
    query = entity(person_inf(name=names))

    results = list(query.evaluate())
    assert len(results) == 2

    bob = next(r for r in results if r.name == "Bob")
    charlie = next(r for r in results if r.name == "Charlie")

    expl_bob_obj = explain_inference(bob)
    expl_charlie_obj = explain_inference(charlie)
    assert expl_bob_obj is not None
    assert expl_charlie_obj is not None
    expl_bob = expl_bob_obj.as_string()
    expl_charlie = expl_charlie_obj.as_string()

    # function name appears in query source context; source code contains the call
    assert "test_explain_inference_multiple_instances" in expl_bob
    assert "test_explain_inference_multiple_instances" in expl_charlie
    assert "person_inf(name=names)" in expl_bob
    assert "person_inf(name=names)" in expl_charlie


def test_explain_inference_deeply_nested():
    """
    Test that explain_inference correctly records and retrieves the stack through deeply
    nested function calls.
    """

    def level_4(name):
        person_inf = inference(Person)
        return person_inf(name=name)

    def level_3(name):
        return level_4(name)

    def level_2(name):
        return level_3(name)

    def level_1(name):
        return level_2(name)

    # Define query through deeply nested calls
    p_var = level_1("Dave")
    query = entity(p_var)

    results = list(query.evaluate())
    assert len(results) == 1
    dave = results[0]

    explanation_obj = explain_inference(dave)
    assert explanation_obj is not None
    explanation = explanation_obj.as_string()

    # innermost user frame is level_4 — appears in query source context
    assert "Person" in explanation
    assert "level_4" in explanation
    assert "person_inf(name=name)" in explanation
    # calling chain only visible in stack trace
    explanation_with_trace = explanation_obj.as_string(show_trace=True)
    assert "test_explain_inference_deeply_nested" in explanation_with_trace
    assert "level_1" in explanation_with_trace
    assert "level_2" in explanation_with_trace
    assert "level_3" in explanation_with_trace


def test_query_root_resolves_to_the_evaluating_query_for_a_variable_reused_across_two_queries():
    """
    An inference variable reused as the selected variable of two independent queries
    must have its ``InferenceExplanation.query_root`` resolved to whichever query is
    actually evaluating, not to whichever query first attached the variable to the DAG.

    ``register_inference`` filled in ``query_root`` from ``variable_node._root_`` — a
    walk up the variable's structural, first-attachment-wins ``_parent_`` chain. A
    variable reused as the selected variable of a second, independent query gains an
    extra parent, but its primary ``_parent_`` still points at whichever query embedded
    it first, so the recorded ``query_root`` silently pointed at the stale, unrelated
    first query instead of the query whose own evaluation actually produced the
    instance.
    """
    person_var = inference(Person)(name="Alice")

    first_query = entity(person_var)
    first_results = list(first_query.evaluate())
    assert len(first_results) == 1
    assert (
        len(person_var._parents_) >= 2
    ), "person_var must already be multi-parented after the first query for this to exercise the bug"

    second_query = entity(person_var)
    second_results = list(second_query.evaluate())
    assert len(second_results) == 1
    alice = second_results[0]

    explanation = explain_inference(alice)
    assert explanation is not None
    assert explanation.query_root is second_query._root_, (
        "query_root must be resolved from the query that is actually evaluating "
        "(second_query), not from whichever query first attached person_var to the DAG "
        "(first_query)"
    )


def test_query_stack_tracking():
    """
    Test that Query objects automatically record their creation stack.
    """
    person_inf = inference(Person)
    query = entity(person_inf(name="Eve"))

    assert hasattr(query, "_creation_stack")
    assert isinstance(query._creation_stack, CallStack)
    # The stack should contain this test function
    filenames = [f.filename for f in query._creation_stack]
    assert any("test_explanation.py" in f for f in filenames)
    functions = [f.function_name for f in query._creation_stack]
    assert "test_query_stack_tracking" in functions


def test_explain_inference_focus_package():
    """
    Test that explain_inference correctly filters by focus_package.
    """
    person_inf = inference(Person)
    query = entity(person_inf(name="Frank"))

    results = list(query.evaluate())
    frank = results[0]

    explanation_obj = explain_inference(frank)
    assert explanation_obj is not None
    # filename appears in the "Query source (..., test_explanation.py:N)" context line
    explanation_full = explanation_obj.as_string()
    assert "test_explanation.py" in explanation_full
    # focus_package only filters when show_trace=True
    explanation_filtered = explanation_obj.as_string(
        show_trace=True, focus_package="test_explanation.py"
    )
    assert "test_explanation.py" in explanation_filtered


def test_variable_stack_tracking():
    """
    Test that Variable objects automatically record their creation stack.
    """
    from krrood.entity_query_language.factories import variable_from

    v = variable_from([1, 2, 3])

    assert monitored.is_monitored(v)
    assert isinstance(monitored.get_stack(v), CallStack)
    filenames = [f.filename for f in monitored.get_stack(v)]
    assert any("test_explanation.py" in f for f in filenames)


def test_robust_monitoring_check():
    """
    Test that register_inference safely ignores non-monitored variables.
    """

    # Create a dummy non-monitored variable-like object
    class NonMonitoredVariable:
        def __init__(self):
            self._id_ = "dummy-id"
            self._root_ = self

    dummy_var = NonMonitoredVariable()
    dummy_instance = "dummy-instance"

    # Should NOT raise AttributeError or any other error
    register_inference(dummy_instance, dummy_var)

    # Check that it was NOT recorded
    assert explain_inference(dummy_instance) is None


# ============================================================
# Tests for satisfied condition tracking and condition graph
# ============================================================


def _get_true_results(query: Query):
    """
    Build *query* and return its raw true ``OperationResult``s (via
    :meth:`SymbolicExpression._true_results_`), which carries evaluation metadata
    (``satisfied_condition_ids``) that ``evaluate()``'s processed output does not
    expose.
    """
    query.build()
    return list(query._true_results_())


def _get_satisfied_names(ids, condition_root):
    """
    Get expression names from satisfied condition IDs by traversing the condition tree.
    """
    return {
        expression._name_
        for expression in condition_root._subtree_expressions_with_ids_(ids)
    }


def test_satisfied_conditions_simple():
    """
    A single Comparator condition tracks its ID as satisfied.
    """
    val = variable_from([6, 3])
    query = entity(val).where(val > 5)

    true_results = _get_true_results(query)
    assert len(true_results) == 1
    result = true_results[0]

    assert result.satisfied_condition_ids is not None
    assert len(result.satisfied_condition_ids) > 0


def test_satisfied_condition_ids_for_a_condition_first_used_in_a_filterless_query():
    """
    A condition first used only as a selected/output expression of a Filter-less query,
    then reused as a different query's where-condition, must still have its satisfied
    conditions tracked for the second query.

    The condition's primary parent is fixed by its first attachment (the Filter-less
    query), so a naive walk from its structural root never reaches the second query's
    Where filter and the pass is wrongly treated as having no where-clause at all.

    ..note:: The shared node is a ``Comparator`` rather than a bare variable so that it
        is a condition participant (see :func:`is_condition_participant`) and the
        expected satisfied set is exactly it, rather than an empty set that could not
        distinguish tracking the right ids from tracking none.
    """
    value = variable_from([6])
    condition = value > 5
    where_less_query = entity(condition)
    where_less_query.build()

    target = variable_from([1])
    query = entity(target).where(condition)

    true_results = _get_true_results(query)
    assert len(true_results) == 1
    assert set(true_results[0].satisfied_condition_ids) == {condition._id_}


def test_satisfied_conditions_and_both_true():
    """
    AND with both children true: AND and both comparators are satisfied.
    """
    val = variable_from([6])
    greater = val > 5
    less = val < 10
    condition = and_(greater, less)
    query = entity(val).where(condition)

    true_results = _get_true_results(query)
    assert len(true_results) == 1
    result = true_results[0]

    assert set(result.satisfied_condition_ids) == {
        condition._id_,
        greater._id_,
        less._id_,
    }


def test_satisfied_conditions_and_short_circuit():
    """
    AND short-circuits when left is false: only the false comparator recorded.
    """
    val = variable_from([3])
    query = entity(val).where(and_(val > 5, val < 10))

    true_results = _get_true_results(query)
    # AND is false, so no true results pass the Where filter
    assert len(true_results) == 0


def test_satisfied_conditions_or_first_true():
    """
    OR with first child true: short-circuits, right never evaluated.
    """
    val = variable_from([6])
    greater = val > 5
    short_circuited = val < 0
    condition = or_(greater, short_circuited)
    query = entity(val).where(condition)

    true_results = _get_true_results(query)
    assert len(true_results) == 1
    result = true_results[0]

    # Exact set: the short-circuited right side is absent rather than satisfied.
    assert set(result.satisfied_condition_ids) == {condition._id_, greater._id_}


def test_satisfied_conditions_exclude_a_short_circuited_operator():
    """
    A whole operator skipped by a short-circuit is not satisfied.

    The operand-level cases above only pin a skipped comparator; an operator that was
    never evaluated must be excluded on the same grounds, since it made no truth claim
    for this evaluation at all.
    """
    val = variable_from([6])
    query = entity(val).where(or_(val > 5, and_(val < 10, val != 0)))

    true_results = _get_true_results(query)
    assert len(true_results) == 1

    satisfied = val._conditions_root_._subtree_expressions_with_ids_(
        true_results[0].satisfied_condition_ids
    )
    assert any(isinstance(expression, OR) for expression in satisfied)
    assert not any(isinstance(expression, AND) for expression in satisfied)


def test_satisfied_conditions_or_fallback():
    """
    OR with first false, second true: both children evaluated, OR satisfied.
    """
    val = variable_from([3])
    false_side = val > 5
    true_side = val < 10
    condition = or_(false_side, true_side)
    query = entity(val).where(condition)

    true_results = _get_true_results(query)
    assert len(true_results) == 1
    result = true_results[0]

    # Exact set: the evaluated-but-false left side is absent rather than satisfied.
    assert set(result.satisfied_condition_ids) == {condition._id_, true_side._id_}


def test_satisfied_conditions_not():
    """
    Not inverts satisfaction: Not is satisfied when its child is false.
    """
    val = variable_from([3])
    negated_comparator = val > 5
    condition = not_(negated_comparator)
    query = entity(val).where(condition)

    true_results = _get_true_results(query)
    assert len(true_results) == 1
    result = true_results[0]

    # Exact set: Not is satisfied, its false inner comparator is not.
    assert set(result.satisfied_condition_ids) == {condition._id_}


def test_satisfied_conditions_nested_and_or():
    """
    Nested and_(x > 5, or_(x < 2, x == 3)) with x=3: test tree structure.
    """
    val = variable_from([3])
    query = entity(val).where(and_(val > 5, or_(val < 2, val == 3)))

    # val=3: val > 5 is False → AND short-circuits, no true results
    true_results = _get_true_results(query)
    assert len(true_results) == 0


def test_satisfied_conditions_nested_and_or_satisfied():
    """
    Nested and_(x > 5, or_(x < 10, x == -1)) with x=6: AND and OR satisfied.
    """
    val = variable_from([6])
    greater = val > 5
    less = val < 10
    short_circuited = val == -1
    inner_or = or_(less, short_circuited)
    condition = and_(greater, inner_or)
    query = entity(val).where(condition)

    true_results = _get_true_results(query)
    assert len(true_results) == 1
    result = true_results[0]

    # Exact set: the equality short-circuited by OR is absent rather than satisfied.
    assert set(result.satisfied_condition_ids) == {
        condition._id_,
        inner_or._id_,
        greater._id_,
        less._id_,
    }


def test_satisfied_conditions_no_where():
    """
    Query without where clause: satisfied_condition_ids is None.
    """
    val = variable_from([1, 2])
    query = entity(val)

    true_results = _get_true_results(query)
    assert len(true_results) == 2
    for result in true_results:
        assert result.satisfied_condition_ids is None


def test_satisfied_conditions_for_bare_condition_shared_with_an_unrelated_query():
    """
    A bare (non-Comparator/Predicate/LogicalOperator) condition value that was first
    attached as a Comparator operand in one query, then reused as the direct where-
    condition of a second, unrelated query, must still be recorded as satisfied by the
    second query's own evaluation.

    ``is_condition_participant`` must not rely on the shared node's structural, first-
    attachment-wins ``_parent_``: that pointer keeps referencing the first (Comparator)
    parent even after the node gains a second, unrelated parent, so a check based on it
    answers a question about construction history instead of about the evaluation that
    is currently running.
    """
    flag = variable_from([True])
    sink = variable_from([1])

    # Attaches `flag` as a Comparator operand first, so its structural primary parent is
    # the Comparator, not a TruthValueOperator.
    unrelated_query = entity(sink).where(flag == True)
    unrelated_query.build()

    # Reuses the same `flag` node as the direct where-condition of a second, independent
    # query. Structurally `flag` now has two parents, but only the Comparator is primary.
    target = variable_from([1])
    query = entity(target).where(flag)
    query.build()
    assert (
        len(flag._parents_) == 2
    ), "flag must be a genuinely shared DAG node for this test to exercise the bug"

    true_results = _get_true_results(query)
    assert len(true_results) == 1
    result = true_results[0]

    assert result.satisfied_condition_ids is not None
    assert flag._id_ in result.satisfied_condition_ids, (
        "flag is this query's own where-condition and evaluated true, so it must be "
        "recorded as satisfied regardless of which query attached it to the DAG first"
    )


# ============================================================
# Tests for condition_graph via explain_inference pipeline
# ============================================================


def test_condition_graph_pipeline_simple():
    """
    explain_inference → condition_graph() for a simple satisfied condition.
    """
    val = variable_from([6])
    query = entity(inference(Item)(value=val)).where(val > 5)
    results = list(query.evaluate())
    assert len(results) == 1

    explanation = explain_inference(results[0])
    assert explanation is not None

    qg = explanation.condition_graph()
    assert qg is not None
    assert len(qg.expression_node_map) > 0

    comp_nodes = [node for node in qg.expression_node_map.values() if node.name == ">"]
    assert len(comp_nodes) == 1
    assert comp_nodes[0].is_satisfied is True


def test_condition_graph_pipeline_nested_and_or():
    """
    explain_inference → condition_graph() with nested AND/OR tree.
    """
    val = variable_from([6])
    query = entity(inference(Item)(value=val)).where(
        and_(val > 5, or_(val < 10, val == -1))
    )
    results = list(query.evaluate())
    assert len(results) == 1

    explanation = explain_inference(results[0])
    assert explanation is not None

    qg = explanation.condition_graph()
    assert qg is not None

    nodes_by_name = {node.name: node for node in qg.expression_node_map.values()}
    assert "AND" in nodes_by_name
    assert "OR" in nodes_by_name
    assert nodes_by_name["AND"].is_satisfied is True
    assert nodes_by_name["OR"].is_satisfied is True
    assert nodes_by_name[">"].is_satisfied is True
    assert nodes_by_name["<"].is_satisfied is True
    assert nodes_by_name["=="].is_satisfied is False


def test_condition_graph_pipeline_not():
    """
    explain_inference → condition_graph() with Not condition.
    """
    val = variable_from([3])
    query = entity(inference(Item)(value=val)).where(not_(val > 5))
    results = list(query.evaluate())
    assert len(results) == 1

    explanation = explain_inference(results[0])
    assert explanation is not None

    qg = explanation.condition_graph()
    assert qg is not None

    nodes_by_name = {node.name: node for node in qg.expression_node_map.values()}
    assert "Not" in nodes_by_name
    assert nodes_by_name["Not"].is_satisfied is True
    assert nodes_by_name[">"].is_satisfied is False


def test_condition_graph_pipeline_no_conditions():
    """
    explain_inference → condition_graph() returns None when no conditions exist.
    """
    val = variable_from([1, 2])
    query = entity(inference(Item)(value=val))
    results = list(query.evaluate())
    assert len(results) == 2

    for item in results:
        explanation = explain_inference(item)
        assert explanation is not None
        assert explanation.condition_graph() is None


def test_condition_graph_pipeline_or_short_circuit():
    """
    OR short-circuits: satisfied comparator recorded, short-circuited one is not.
    """
    val = variable_from([6])
    query = entity(inference(Item)(value=val)).where(or_(val > 5, val < 10))
    results = list(query.evaluate())
    assert len(results) == 1

    explanation = explain_inference(results[0])
    assert explanation is not None

    qg = explanation.condition_graph()
    assert qg is not None

    nodes_by_name = {node.name: node for node in qg.expression_node_map.values()}
    assert "OR" in nodes_by_name
    assert nodes_by_name["OR"].is_satisfied is True
    assert nodes_by_name[">"].is_satisfied is True
    assert nodes_by_name["<"].is_satisfied is False


def test_condition_graph_pipeline_complex():
    """
    Deeply nested AND/OR/NOT with val=5: inner AND short-circuited by OR.
    """
    val = variable_from([5])
    query = entity(inference(Item)(value=val)).where(
        and_(val > 0, or_(not_(val == 2), and_(val < 10, val > 1)))
    )
    results = list(query.evaluate())
    assert len(results) == 1

    explanation = explain_inference(results[0])
    assert explanation is not None

    qg = explanation.condition_graph()
    assert qg is not None

    nodes_by_name = {node.name: node for node in qg.expression_node_map.values()}
    # Look up AND nodes directly from expression_node_map using the actual expressions
    and_nodes = [
        (expr, node)
        for expr, node in qg.expression_node_map.items()
        if node.name == "AND"
    ]
    assert len(and_nodes) == 2
    # Root AND: and_(val > 0, ...), the condition root — evaluated and satisfied
    # Inner AND: and_(val < 10, val > 1), OR's right branch — short-circuited, not satisfied
    root_and = next(node for _, node in and_nodes if node.is_satisfied)
    inner_and = next(node for _, node in and_nodes if not node.is_satisfied)
    assert root_and.is_satisfied is True
    assert inner_and.is_satisfied is False
    assert nodes_by_name["OR"].is_satisfied is True
    assert nodes_by_name["Not"].is_satisfied is True
    assert nodes_by_name["=="].is_satisfied is False
    # Two ">" nodes: val > 0 (satisfied) and val > 1 (short-circuited, not satisfied)
    gt_nodes = [node for node in qg.expression_node_map.values() if node.name == ">"]
    assert len(gt_nodes) == 2
    assert sum(1 for n in gt_nodes if n.is_satisfied) == 1


def test_condition_graph_pipeline_multiple_results():
    """
    Each result from val=[1,6,11] with val>5 has correct satisfaction.
    """
    val = variable_from([1, 6, 11])
    query = entity(inference(Item)(value=val)).where(val > 5)
    results = list(query.evaluate())
    assert len(results) == 2  # 6 and 11

    items_by_value = {r.value: r for r in results}
    assert 6 in items_by_value
    assert 11 in items_by_value

    for item in results:
        explanation = explain_inference(item)
        assert explanation is not None
        qg = explanation.condition_graph()
        assert qg is not None
        comp_nodes = [
            node for node in qg.expression_node_map.values() if node.name == ">"
        ]
        assert len(comp_nodes) == 1
        assert comp_nodes[0].is_satisfied is True


def test_condition_graph_pipeline_non_symbol():
    """
    explain_inference returns None for non-Symbol values (e.g. plain int).
    """
    val = variable_from([6])
    query = entity(val).where(val > 5)
    results = list(query.evaluate())
    assert len(results) == 1
    # Plain integers are not Symbol instances, so register_inference silently skips them
    assert explain_inference(results[0]) is None


# ============================================================
# Tests for QueryGraph satisfaction color overlay
# ============================================================


def test_query_graph_marks_a_shared_bare_condition_satisfied_from_its_own_query():
    """
    A bare condition value reused across two unrelated queries must be classified as a
    condition participant by whichever query's own ``QueryGraph`` is being built, not by
    whichever query happened to attach it to the DAG first.

    Mirrors ``test_satisfied_conditions_for_bare_condition_shared_with_an_unrelated_query``
    for the post-hoc ``QueryGraph`` visualization path: ``construct_graph`` already knows
    the edge it is visiting (it recurses via each expression's own ``_children_``), so it
    must not re-derive a possibly-unrelated parent from the shared node's structural
    ``_parent_``.
    """
    flag = variable_from([True])
    sink = variable_from([1])

    unrelated_query = entity(sink).where(flag == True)
    unrelated_query.build()

    target = variable_from([1])
    query = entity(target).where(flag)
    query.build()
    assert (
        len(flag._parents_) == 2
    ), "flag must be a genuinely shared DAG node for this test to exercise the bug"

    true_results = _get_true_results(query)
    result = true_results[0]

    query_graph = QueryGraph(
        query, satisfied_condition_ids=result.satisfied_condition_ids
    )
    flag_node = query_graph.expression_node_map[flag]
    assert flag_node.is_satisfied, (
        "flag is this query's own where-condition and evaluated true, so its QueryNode "
        "must be marked satisfied regardless of which query attached it to the DAG first"
    )
    assert not flag_node.faded


def test_query_graph_marks_a_node_reused_at_two_positions_satisfied_regardless_of_visit_order():
    """
    A node reused at two positions within one query's own tree -- once as a bare
    ``TruthValueOperator`` child, once as a ``Comparator`` operand -- is marked
    satisfied whenever it is in ``satisfied_condition_ids``, whichever of the two
    positions ``construct_graph`` visits first.

    ``construct_graph`` memoizes exactly one ``QueryNode`` per expression
    (``expression_node_map``), so any position-dependent term in the satisfaction check
    would be computed at the first-visited position and then reused for every other one.
    ``and_(flag == True, flag)`` reaches the ``Comparator`` operand position before the
    bare ``AND`` child position, so it pins that the classification stays independent of
    visit order.
    """
    flag = variable_from([True])
    sink = variable_from([1])

    query = entity(sink).where(and_(flag == True, flag))
    assert (
        len(flag._parents_) == 2
    ), "flag must be a genuinely shared DAG node for this to exercise the behaviour"

    true_results = _get_true_results(query)
    result = true_results[0]
    assert flag._id_ in result.satisfied_condition_ids, (
        "flag is directly evaluated as a bare AND condition and is true, so it must be "
        "recorded as satisfied"
    )

    query_graph = QueryGraph(
        query, satisfied_condition_ids=result.satisfied_condition_ids
    )
    flag_node = query_graph.expression_node_map[flag]
    assert flag_node.is_satisfied, (
        "flag's own bare-condition position under AND must be classified as satisfied, "
        "regardless of whichever position (Comparator operand or bare AND child) "
        "construct_graph happened to visit first"
    )


def test_query_graph_satisfaction_colors():
    """
    Unsatisfied nodes get red borders; satisfied nodes keep full color and no border.
    """
    from krrood.entity_query_language.query_graph import ColorLegend

    val = variable_from([6])
    query = entity(inference(Item)(value=val)).where(or_(val > 5, val < 10))
    results = list(query.evaluate())
    assert len(results) == 1

    explanation = explain_inference(results[0])
    assert explanation is not None

    qg = QueryGraph(
        explanation.query_root,
        satisfied_condition_ids=explanation.satisfied_condition_ids,
    )

    original_colors = {
        id(node.data): ColorLegend.from_expression(node.data).color
        for node in qg.expression_node_map.values()
    }

    for node in qg.expression_node_map.values():
        if isinstance(node.data, Comparator) and node.data._name_ == ">":
            # val > 5 is satisfied: full fill color retained, not faded, no red border
            assert node.color.color == original_colors[id(node.data)]
            assert not node.faded
            assert node.border_color is None
        if isinstance(node.data, Comparator) and node.data._name_ == "<":
            # val < 10 is short-circuited (unsatisfied): fill color unchanged,
            # faded flag set, red border applied instead of transparency
            assert node.color.color == original_colors[id(node.data)]
            assert node.faded
            assert node.border_color == "red"

    for node in qg.expression_node_map.values():
        if not is_condition_participant(node.data):
            # Non-condition nodes may be faded if they are exclusive descendants
            # of an unsatisfied node (e.g. Literal(10) under <); they get red borders too.
            # Shared nodes (Variable reachable via a satisfied path) stay unfaded.
            if not node.faded:
                assert node.border_color is None


def test_query_graph_faded_subtree_propagation():
    """
    Descendants only reachable through an unsatisfied node are also faded.
    """
    val = variable_from([6])
    # or_(val > 5, val < 10): val > 5 satisfied, val < 10 short-circuited
    query = entity(inference(Item)(value=val)).where(or_(val > 5, val < 10))
    results = list(query.evaluate())
    explanation = explain_inference(results[0])

    qg = QueryGraph(
        explanation.query_root,
        satisfied_condition_ids=explanation.satisfied_condition_ids,
    )

    # The unsatisfied "<" Comparator should be faded
    unsatisfied_lt = next(
        n
        for n in qg.expression_node_map.values()
        if isinstance(n.data, Comparator) and n.data._name_ == "<"
    )
    assert unsatisfied_lt.faded

    # Literal(10) is only reachable through the unsatisfied "<" → faded
    # (Variable(val) is shared with the satisfied ">" path → not faded)
    faded_children = [c for c in unsatisfied_lt.children if c.faded]
    assert len(faded_children) >= 1, "At least one exclusive child should be faded"
    # The Literal specifically should be faded
    literal_10 = next((c for c in unsatisfied_lt.children if "Literal" in c.name), None)
    if literal_10:
        assert literal_10.faded


def test_query_graph_satisfaction_colors_all_satisfied():
    """
    When all condition nodes are satisfied, none should be faded.
    """
    from krrood.entity_query_language.query_graph import ColorLegend

    val = variable_from([6])
    query = entity(inference(Item)(value=val)).where(and_(val > 5, val < 10))
    results = list(query.evaluate())
    assert len(results) == 1

    explanation = explain_inference(results[0])
    assert explanation is not None

    qg = QueryGraph(
        explanation.query_root,
        satisfied_condition_ids=explanation.satisfied_condition_ids,
    )

    original_colors = {
        id(node.data): ColorLegend.from_expression(node.data).color
        for node in qg.expression_node_map.values()
    }

    for node in qg.expression_node_map.values():
        assert not node.faded
        if is_condition_participant(node.data):
            assert node.color.color == original_colors[id(node.data)]


def test_explanation_condition_graph_and_visualize():
    """
    InferenceExplanation.condition_graph() creates correct QueryGraph.
    """
    val = variable_from([6])
    query = entity(inference(Item)(value=val)).where(
        or_(and_(val > 5, val < 10), val == 11)
    )
    results = list(query.evaluate())
    assert len(results) == 1

    explanation = explain_inference(results[0])
    assert explanation is not None

    qg = explanation.condition_graph()
    qg.visualize()
    assert qg is not None
    assert qg.satisfied_condition_ids == explanation.satisfied_condition_ids

    for node in qg.expression_node_map.values():
        if node.data._name_ == "AND":
            assert "not satisfied" not in node.color.name.lower()

    fig, ax = explanation.condition_graph().visualize()
    assert fig is not None
    assert ax is not None


def test_nested_rule_explanation(drawer_rule):
    found_drawers = drawer_rule.tolist()
    explanation = explain_inference(found_drawers[0])
    assert explanation is not None
    assert isinstance(explanation.query_root, SymbolicExpression)
    for i, condition_and_bindings in enumerate(
        explanation.get_satisfied_conditions_and_their_bindings()
    ):
        condition = condition_and_bindings.condition
        bindings = condition_and_bindings.bindings
        assert isinstance(condition, Comparator)
        assert condition.operation is operator.eq
        if i == 0:
            assert isinstance(condition.left, Attribute)
            assert condition.left._attribute_name_ == "parent"
            assert condition.left._child_._type_ is FixedConnection
            assert condition.right._child_._type_ is PrismaticConnection
            assert isinstance(
                bindings[condition.right._child_._id_], PrismaticConnection
            )
        else:
            assert isinstance(condition.left, Attribute)
            assert condition.left._attribute_name_ == "child"
            assert condition.left._child_._type_ is FixedConnection
            assert isinstance(bindings[condition.left._child_._id_], FixedConnection)
    assert explanation.get_satisfied_conditions_as_string() == (
        "(FixedConnection.parent == PrismaticConnection.child)"
        "\nAND (FixedConnection.child == Handle)"
    )
    explanation.condition_graph().visualize(filename="drawer_explanation.pdf")


def test_nested_rule_meta_queries(drawer_rule):
    explanation = explain_inference(drawer_rule)


def test_explanation_lifecycle_tied_to_instance():
    """
    InferenceExplanation must not keep the inferred instance (or its World) alive after
    all external references are released.

    The world is created directly inside a helper closure so that no pytest fixture
    machinery holds an external strong reference — pytest keeps fixture return-values
    alive for the duration of the test function even after an explicit ``del``.
    """
    import weakref
    from ..test_eql.conf.world.doors_and_drawers import DoorsAndDrawersWorld

    world_ref: weakref.ref
    drawer_ref: weakref.ref

    def _run():
        nonlocal world_ref, drawer_ref
        world = DoorsAndDrawersWorld().create()

        handle = variable(Handle, world.bodies)
        prismatic_connection = variable(PrismaticConnection, world.connections)
        fixed_connection = a(FixedConnection)(
            parent=prismatic_connection.child, child=handle
        ).from_(world.connections)
        drawers = inference(Drawer)(
            container=fixed_connection.expression.parent,
            handle=fixed_connection.expression.child,
        ).tolist()
        assert drawers, "Need at least one inferred Drawer for this test"

        drawer = drawers[0]
        explanation = explain_inference(drawer)
        assert explanation is not None, "Drawer should have an inference explanation"

        world_ref = weakref.ref(world)
        drawer_ref = weakref.ref(drawer)

    _run()
    gc.collect()

    # All strong references inside _run() are now gone.  InferenceExplanation is
    # owned by the drawer (not a global registry), so the entire cluster — drawer,
    # explanation, world entities — must be collectable.
    assert world_ref() is None, (
        "World should be garbage-collected after all local references are released. "
        "If this fails, something outside the World cluster is keeping it alive — "
        "likely an InferenceExplanation or a class-level cache holding a strong "
        "reference to a WorldEntity."
    )
    assert (
        drawer_ref() is None
    ), "Drawer should have been garbage-collected together with its World."


@pytest.fixture
def drawer_rule(doors_and_drawers_world):
    world = doors_and_drawers_world
    handle = variable(Handle, world.bodies)
    prismatic_connection = variable(PrismaticConnection, world.connections)
    fixed_connection = a(FixedConnection)(
        parent=prismatic_connection.child, child=handle
    ).from_(world.connections)
    return inference(Drawer)(
        container=fixed_connection.expression.parent,
        handle=fixed_connection.expression.child,
    )
