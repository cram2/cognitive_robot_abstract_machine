"""
Tests for the typed per-pass state collaborators on EvaluationContext.
"""

import uuid
from dataclasses import dataclass, field

from ordered_set import OrderedSet

from krrood.entity_query_language.evaluation_context import (
    ActiveConditionsRoot,
    EvaluatedExpressionIds,
    EvaluationContext,
)


@dataclass
class _NodeStub:
    """
    Minimal stand-in for a SymbolicExpression: only ``_id_`` is needed by these
    collaborators.
    """

    _id_: uuid.UUID = field(default_factory=uuid.uuid4, init=False)
    """
    Identifier for this stub node.
    """


def test_active_conditions_root_keeps_the_first_node_set_and_ignores_later_ones():
    tracking = ActiveConditionsRoot()
    first = _NodeStub()
    second = _NodeStub()

    tracking.set_active_root_if_not_set(first, has_condition=True)
    tracking.set_active_root_if_not_set(second, has_condition=True)

    assert tracking.is_active_root(first)
    assert not tracking.is_active_root(second)


def test_active_conditions_root_resolves_by_what_was_set_not_by_construction_order():
    """
    The whole point of this class: whichever node is set first for the pass is the
    active root, regardless of any other node's structural/construction history.
    """
    tracking = ActiveConditionsRoot()
    node = _NodeStub()

    assert not tracking.is_active_root(node), "an unset pass must not match any node"
    tracking.set_active_root_if_not_set(node, has_condition=True)
    assert tracking.is_active_root(node)


def test_active_conditions_root_has_condition_when_set_with_a_genuine_filter():
    tracking = ActiveConditionsRoot()
    filter_condition = _NodeStub()

    tracking.set_active_root_if_not_set(filter_condition, has_condition=True)

    assert tracking.has_condition


def test_active_conditions_root_has_no_condition_when_set_without_a_filter():
    """
    A Filter-less evaluation's own _conditions_root_ falls back to its plain _root_, so
    setting it with ``has_condition=False`` must record "no real condition" for this
    pass.
    """
    tracking = ActiveConditionsRoot()
    node = _NodeStub()

    tracking.set_active_root_if_not_set(node, has_condition=False)

    assert not tracking.has_condition


def test_active_conditions_root_has_condition_defaults_false_before_anything_is_set():
    tracking = ActiveConditionsRoot()

    assert not tracking.has_condition


def test_is_child_of_truth_value_operator_reflects_only_recorded_nodes():
    context = EvaluationContext()
    recorded = _NodeStub()
    unrecorded = _NodeStub()

    context.truth_value_operator_children.record(recorded._id_)

    assert context.is_child_of_truth_value_operator(recorded)
    assert not context.is_child_of_truth_value_operator(unrecorded)


def test_evaluated_expression_ids_records_and_iterates():
    tracked = EvaluatedExpressionIds()
    first, second = uuid.uuid4(), uuid.uuid4()

    tracked.record(first)
    tracked.record(second)

    assert set(tracked) == {first, second}


def test_evaluated_expression_ids_merges_ids_from_another_set():
    tracked = EvaluatedExpressionIds()
    tracked.record(uuid.uuid4())
    other_ids = OrderedSet([uuid.uuid4(), uuid.uuid4()])

    tracked.merge(other_ids)

    assert other_ids.issubset(set(tracked))


def test_evaluated_expression_ids_snapshot_reflects_recorded_ids():
    tracked = EvaluatedExpressionIds()
    first = uuid.uuid4()
    tracked.record(first)

    snapshot = tracked.snapshot()

    assert set(snapshot) == {first}


def test_evaluated_expression_ids_snapshot_is_reused_while_set_is_unchanged():
    """
    The id set only grows, so its length is a valid version key: two snapshots taken
    without an intervening record() share the same cached object.
    """
    tracked = EvaluatedExpressionIds()
    tracked.record(uuid.uuid4())

    first_snapshot = tracked.snapshot()
    second_snapshot = tracked.snapshot()

    assert first_snapshot is second_snapshot


def test_evaluated_expression_ids_snapshot_refreshes_after_growth():
    tracked = EvaluatedExpressionIds()
    tracked.record(uuid.uuid4())
    stale_snapshot = tracked.snapshot()

    tracked.record(uuid.uuid4())
    fresh_snapshot = tracked.snapshot()

    assert fresh_snapshot is not stale_snapshot
    assert len(fresh_snapshot) == 2
