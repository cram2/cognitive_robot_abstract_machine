"""
Evaluation context and observer system for the Entity Query Language.

This module provides an aspect-oriented mechanism for hooking into the
evaluation pipeline without polluting the core evaluation methods.
"""

from __future__ import annotations

from ordered_set import OrderedSet
from typing_extensions import Any, Dict, Optional

from krrood.entity_query_language._monitoring import monitored
from krrood.entity_query_language.core.base_expressions import (
    OperationResult,
    TruthValueOperator,
)
from krrood.entity_query_language.core.variable import InstantiatedVariable
from krrood.entity_query_language.enums import EvaluationContextKey
from krrood.entity_query_language.evaluation_context import (
    EvaluationContext,
    EvaluationObserver,
    _evaluation_context_var,
    get_evaluation_context,
    set_evaluation_context,
)
from krrood.entity_query_language.exceptions import NoExpressionFoundForGivenID
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import (
    LogicalOperator,
)
from krrood.entity_query_language.predicate import Predicate
from krrood.entity_query_language.query.query import Query


def is_condition_participant(expr: OperationResult) -> bool:
    """
    Check whether the expression participates in condition evaluation.

    :param expr: The symbolic expression to test.
    :return: ``True`` if *expr* is a :class:`~krrood.entity_query_language.operators.comparator.Comparator`,
        :class:`~krrood.entity_query_language.predicate.Predicate`, or
        :class:`~krrood.entity_query_language.operators.core_logical_operators.LogicalOperator`,
        or if its direct parent is a
        :class:`~krrood.entity_query_language.core.base_expressions.TruthValueOperator`.
    """
    if isinstance(expr, (Comparator, Predicate, LogicalOperator)):
        return True
    parent = expr._parent_
    if parent is not None and isinstance(parent, TruthValueOperator):
        return True
    return False


class EvaluationTracker(EvaluationObserver):
    """Observer that tracks which expressions were evaluated and stamps the cumulative set on each OperationResult.

    Maintains a cumulative set of expression IDs in the evaluation context, adding each expression's ID
    on :meth:`on_evaluate_enter`. On :meth:`on_result_yielded`, snapshots the current set onto the result
    as ``evaluated_expression_ids``.

    This tracking is the foundation for distinguishing evaluated-from-skipped logical operators (for example,
    short-circuited OR/AND branches) in inference explanations.
    """

    def on_evaluate_enter(self, expression, sources):
        evaluation_context = get_evaluation_context()
        if evaluation_context is None:
            return
        evaluated = evaluation_context.data.setdefault(
            EvaluationContextKey.EVALUATED_IDS_KEY, OrderedSet()
        )
        evaluated.add(expression._id_)

        if isinstance(sources, OperationResult) and sources.evaluated_expression_ids:
            evaluated.update(sources.evaluated_expression_ids)

    def on_result_yielded(self, expression, result):
        evaluation_context = get_evaluation_context()
        if evaluation_context is None:
            return
        evaluated = evaluation_context.data.get(EvaluationContextKey.EVALUATED_IDS_KEY)
        if evaluated is not None and result.evaluated_expression_ids is None:
            result.evaluated_expression_ids = OrderedSet(evaluated)


class SatisfiedConditionTracker(EvaluationObserver):
    """Observer that tracks which condition expressions were satisfied during a single evaluation pass.

    Records truth values on :meth:`on_result_yielded` and populates
    ``result.satisfied_condition_ids`` at the conditions root after all conditions have been evaluated.
    """

    def on_evaluate_enter(self, expression, sources):
        evaluation_context = get_evaluation_context()
        if evaluation_context is None:
            return

        satisfied = None
        if isinstance(sources, OperationResult):
            satisfied = sources.satisfied_condition_ids
        if satisfied is not None:
            evaluation_context.data[EvaluationContextKey.SATISFIED_IDS_KEY] = satisfied

    def on_result_yielded(self, expression, result):
        evaluation_context = get_evaluation_context()
        if evaluation_context is None:
            return
        satisfied = evaluation_context.data.get(EvaluationContextKey.SATISFIED_IDS_KEY)
        if satisfied is not None and result.satisfied_condition_ids is None:
            result.satisfied_condition_ids = satisfied

    def on_conclusions_processed(self, expression, result):

        if expression._conditions_root_ is not expression:
            return
        if result.is_false:
            return
        if expression._conditions_root_ is expression._root_:
            return

        evaluation_context = get_evaluation_context()
        evaluated = (
            evaluation_context.data.get(EvaluationContextKey.EVALUATED_IDS_KEY)
            if evaluation_context is not None
            else None
        )
        if evaluated is None:
            return

        # Build a truth map from the OperationResult chain: operand_id -> is_false.
        # This reflects the actual truth values from this specific evaluation path,
        # with no risk of stale state from previous passes.
        chain_truth_map: Dict = {}
        node = result
        seen: set = set()
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            if node.operand is not None:
                chain_truth_map[node.operand._id_] = node.is_false
            node = node.previous_operation_result

        satisfied = OrderedSet()
        for expr_id in evaluated:
            try:
                expr = expression._get_expression_by_id_(expr_id)
            except NoExpressionFoundForGivenID:
                continue
            if not is_condition_participant(expr):
                continue
            if isinstance(expr, LogicalOperator):
                # An operator not present in the chain was short-circuited: not satisfied.
                if not chain_truth_map.get(expr_id, True):
                    satisfied.add(expr_id)
            elif expr_id in result.bindings:
                if result.bindings[expr_id]:
                    satisfied.add(expr_id)

        result.satisfied_condition_ids = satisfied
        if evaluation_context is not None:
            evaluation_context.data[EvaluationContextKey.SATISFIED_IDS_KEY] = satisfied


class InferenceRecorder(EvaluationObserver):
    """Observer that records inferred instances for later explanation.

    Attaches an :class:`~krrood.entity_query_language.explanation.explanation.InferenceExplanation`
    to each newly inferred :class:`~krrood.symbol_graph.symbol_graph.Symbol` instance so that
    callers can retrieve it via
    :func:`~krrood.entity_query_language.explanation.explanation.explain_inference`.
    """

    def on_result_yielded(self, expression, result):
        if not monitored.is_monitored(type(expression)):
            return
        if expression._id_ not in result.bindings:
            return
        # Only record for InstantiatedVariable subclasses whose _evaluate__
        # delegates to _instantiate_using_child_vars_and_yield_results_ (that is,
        # those that actually create new instances).  Query and its subclasses
        # (Entity, SetOf) override _evaluate__ and merely remap bindings
        # without creating new inferred instances.
        if not isinstance(expression, InstantiatedVariable):
            return
        if isinstance(expression, Query):
            return
        # Inline import justified: explanation.py → query_graph.py → evaluation.py
        # creates a load-time cycle that prevents a top-level import here.
        from krrood.entity_query_language.explanation.explanation import (
            register_inference,
        )

        register_inference(result.bindings[expression._id_], expression, result)


def create_default_evaluation_context() -> EvaluationContext:
    """
    Create an :class:`EvaluationContext` populated with the standard set of observers.

    This is the authoritative factory for evaluation contexts used during normal
    query evaluation.  Callers that need custom observer configurations should
    construct an :class:`EvaluationContext` directly rather than calling this function.

    :return: A new :class:`EvaluationContext` with :class:`EvaluationTracker`,
        :class:`SatisfiedConditionTracker`, and :class:`InferenceRecorder` observers attached.
    """
    return EvaluationContext(
        observers=[
            EvaluationTracker(),
            SatisfiedConditionTracker(),
            InferenceRecorder(),
        ]
    )
