"""
Evaluation context and observer system for the Entity Query Language.

This module provides an aspect-oriented mechanism for hooking into the evaluation
pipeline without polluting the core evaluation methods.
"""

from __future__ import annotations

from ordered_set import OrderedSet
from typing_extensions import Any, Optional

from krrood.entity_query_language._monitoring import monitored
from krrood.entity_query_language.core.base_expressions import (
    OperationResult,
    SymbolicExpression,
    TruthValueOperator,
)
from krrood.entity_query_language.core.variable import InstantiatedVariable
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


def is_condition_participant(
    expression: SymbolicExpression,
    parent: Optional[SymbolicExpression] = None,
) -> bool:
    """
    Check whether the expression participates in condition evaluation.

    :param expression: The symbolic expression to test.
    :param parent: The parent relevant to the caller's own traversal, when the caller
        already knows it (for example a graph walk that reached *expression* through one
        of its own children edges). Takes precedence over both of the fallbacks below.
    :return: ``True`` if *expression* is a :class:`~krrood.entity_query_language.operators.comparator.Comparator`,
        :class:`~krrood.entity_query_language.predicate.Predicate`, or
        :class:`~krrood.entity_query_language.operators.core_logical_operators.LogicalOperator`,
        or if it was evaluated (or, per *parent*, reached) as a direct child of a
        :class:`~krrood.entity_query_language.core.base_expressions.TruthValueOperator`.
    """
    if isinstance(expression, (Comparator, Predicate, LogicalOperator)):
        return True
    if parent is not None:
        return isinstance(parent, TruthValueOperator)
    evaluation_context = get_evaluation_context()
    if evaluation_context is not None:
        return evaluation_context.is_child_of_truth_value_operator(expression)
    structural_parent = expression._parent_
    return structural_parent is not None and isinstance(
        structural_parent, TruthValueOperator
    )


class EvaluationTracker(EvaluationObserver):
    """
    Observer that tracks which expressions were evaluated and stamps the cumulative set
    on each OperationResult.

    Maintains a cumulative set of expression IDs in the evaluation context, adding each
    expression's ID on :meth:`on_evaluate_enter`. On :meth:`on_result_yielded`,
    snapshots the current set onto the result as ``evaluated_expression_ids``.

    This tracking is the foundation for distinguishing evaluated-from-skipped logical
    operators (for example, short-circuited OR/AND branches) in inference explanations.
    """

    def on_evaluate_enter(self, expression, sources):
        evaluation_context = get_evaluation_context()
        if evaluation_context is None:
            return
        evaluation_context.evaluated_expression_ids.record(expression._id_)

        if isinstance(sources, OperationResult) and sources.evaluated_expression_ids:
            evaluation_context.evaluated_expression_ids.merge(
                sources.evaluated_expression_ids
            )

    def on_result_yielded(self, expression, result):
        evaluation_context = get_evaluation_context()
        if evaluation_context is None:
            return
        if result.evaluated_expression_ids is None:
            result.evaluated_expression_ids = (
                evaluation_context.evaluated_expression_ids.snapshot()
            )


class SatisfiedConditionTracker(EvaluationObserver):
    """
    Observer that tracks which condition expressions were satisfied during a single
    evaluation pass.

    Records truth values on :meth:`on_result_yielded` and populates
    ``result.satisfied_condition_ids`` at the conditions root after all conditions have
    been evaluated.
    """

    def on_evaluate_enter(self, expression, sources):
        evaluation_context = get_evaluation_context()
        if evaluation_context is None:
            return

        satisfied = None
        if isinstance(sources, OperationResult):
            satisfied = sources.satisfied_condition_ids
        if satisfied is not None:
            evaluation_context.satisfied_condition_ids = satisfied

    def on_result_yielded(self, expression, result):
        evaluation_context = get_evaluation_context()
        if evaluation_context is None:
            return
        satisfied = evaluation_context.satisfied_condition_ids
        if satisfied is not None and result.satisfied_condition_ids is None:
            result.satisfied_condition_ids = satisfied

    def on_conclusions_processed(self, expression, result):
        """
        Record on *result* which of this pass's conditions were satisfied.

        :param expression: The pass's active conditions root.
        :param result: The result whose conclusions were just processed.
        """
        # The structural check comes first because reading a result's truth can be expensive
        # (a bound predicate evaluates itself), and an evaluation with no conditions to track
        # is dismissed without needing the truth at all.
        evaluation_context = get_evaluation_context()
        if not evaluation_context.active_conditions_root.has_condition:
            return
        if result.is_false:
            return

        evaluated = evaluation_context.evaluated_expression_ids

        # Every truth-bearing expression records its truth in the bindings of the result
        # it yields, so one uniform lookup covers operators and value-bearing expressions
        # alike. An expression short-circuited by an operator recorded nothing, and is
        # therefore not satisfied.
        satisfied = OrderedSet()
        for evaluated_id in evaluated:
            try:
                evaluated_expression = expression._get_expression_by_id_(evaluated_id)
            except NoExpressionFoundForGivenID:
                continue
            if not is_condition_participant(evaluated_expression):
                continue
            if result.bindings.get(evaluated_id):
                satisfied.add(evaluated_id)

        result.satisfied_condition_ids = satisfied
        evaluation_context.satisfied_condition_ids = satisfied


class InferenceRecorder(EvaluationObserver):
    """
    Observer that records inferred instances for later explanation.

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

    This is the authoritative factory for evaluation contexts used during normal query
    evaluation.  Callers that need custom observer configurations should construct an
    :class:`EvaluationContext` directly rather than calling this function.

    :return: A new :class:`EvaluationContext` with :class:`EvaluationTracker`,
        :class:`SatisfiedConditionTracker`, and :class:`InferenceRecorder` observers
        attached.
    """
    return EvaluationContext(
        observers=[
            EvaluationTracker(),
            SatisfiedConditionTracker(),
            InferenceRecorder(),
        ]
    )
