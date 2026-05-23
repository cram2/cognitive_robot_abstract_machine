"""
Evaluation context and observer system for the Entity Query Language.

This module provides an aspect-oriented mechanism for hooking into the
evaluation pipeline without polluting the core evaluation methods.
"""

from __future__ import annotations

from abc import ABC
from contextvars import ContextVar
from dataclasses import dataclass, field

from ordered_set import OrderedSet
from typing_extensions import Any, Dict, List, Optional

from typing_extensions import TYPE_CHECKING

from krrood.entity_query_language.enums import EvaluationContextKey
if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import (
        Bindings,
        OperationResult,
        SymbolicExpression,
    )

_evaluation_context_var: ContextVar[Optional[EvaluationContext]] = ContextVar(
    "_evaluation_context", default=None
)


def get_evaluation_context() -> Optional[EvaluationContext]:
    """Return the current evaluation context, or None if outside evaluation."""
    return _evaluation_context_var.get()


def set_evaluation_context(ctx: Optional[EvaluationContext]) -> None:
    """Set or clear the current evaluation context."""
    _evaluation_context_var.set(ctx)


class EvaluationObserver(ABC):
    """Observer for evaluation events in the EQL evaluation pipeline."""

    def on_evaluate_enter(
            self, expression: SymbolicExpression, sources: Bindings
    ) -> None:
        """Called when entering an expression's _evaluate_ method."""

    def on_evaluate_exit(self, expression: SymbolicExpression) -> None:
        """Called when exiting an expression's _evaluate_ method."""

    def on_result_yielded(
            self, expression: SymbolicExpression, result: OperationResult
    ) -> None:
        """Called for each OperationResult yielded from _evaluate__."""

    def on_conclusions_processed(
            self, expression: SymbolicExpression, result: OperationResult
    ) -> None:
        """Called after _evaluate_conclusions_and_update_bindings_ completes."""


@dataclass
class EvaluationContext:
    """Carries observer state through the evaluation pipeline."""

    observers: List[EvaluationObserver] = field(default_factory=list)
    """
    List of observers to notify of evaluation events.
    """
    data: Dict[EvaluationContextKey, Any] = field(default_factory=dict)
    """
    Arbitrary data storage for observers to share information across events during evaluation.
     Observers should use well-known keys defined in EvaluationContextKey to avoid collisions.
     This is the primary mechanism for observers to maintain state across the evaluation of an expression
     and its sub-expressions without needing to modify the expression classes or the core evaluation logic.
     For example, the EvaluationTracker observer uses the EVALUATED_IDS_KEY to track which expressions have been 
     evaluated in the current context, and the SatisfiedConditionTracker uses the SATISFIED_IDS_KEY to track which 
     condition expressions have been satisfied.
    """

    def on_evaluate_enter(self, *, expression, sources):
        """Notify all observers that an expression is about to be evaluated."""
        for obs in self.observers:
            obs.on_evaluate_enter(expression, sources)

    def on_evaluate_exit(self, *, expression):
        """Notify all observers that an expression has finished evaluating."""
        for obs in self.observers:
            obs.on_evaluate_exit(expression)

    def on_result_yielded(self, *, expression, result):
        """Notify all observers that a result has been yielded from an expression."""
        for obs in self.observers:
            obs.on_result_yielded(expression, result)

    def on_conclusions_processed(self, *, expression, result):
        """Notify all observers that conclusions have been processed for an expression."""
        for obs in self.observers:
            obs.on_conclusions_processed(expression, result)


def is_condition_participant(expr) -> bool:
    """Return True if the expression participates in condition evaluation."""
    from krrood.entity_query_language.operators.comparator import Comparator
    from krrood.entity_query_language.predicate import Predicate
    from krrood.entity_query_language.operators.core_logical_operators import (
        LogicalOperator,
    )
    from krrood.entity_query_language.core.base_expressions import (
        TruthValueOperator,
    )

    _condition_types = (Comparator, Predicate, LogicalOperator)
    if isinstance(expr, _condition_types):
        return True
    if isinstance(expr._parent_, TruthValueOperator):
        return True
    return False


class EvaluationTracker(EvaluationObserver):
    """Observer that tracks which expressions were evaluated and stamps the cumulative set on each OperationResult.

    Maintains a cumulative set of expression IDs in the evaluation context, adding each expression's ID
    on :meth:`on_evaluate_enter`. On :meth:`on_result_yielded`, snapshots the current set onto the result
    as ``evaluated_expression_ids``.

    This tracking is the foundation for distinguishing evaluated-from-skipped logical operators (e.g.
    short-circuited OR/AND branches) in inference explanations.
    """

    def on_evaluate_enter(self, expression, sources):
        from krrood.entity_query_language.core.base_expressions import (
            OperationResult,
        )

        ctx = get_evaluation_context()
        if ctx is None:
            return
        evaluated = ctx.data.setdefault(EvaluationContextKey.EVALUATED_IDS_KEY, OrderedSet())
        evaluated.add(expression._id_)

        if isinstance(sources, OperationResult) and sources.evaluated_expression_ids:
            evaluated.update(sources.evaluated_expression_ids)

    def on_result_yielded(self, expression, result):
        ctx = get_evaluation_context()
        if ctx is None:
            return
        evaluated = ctx.data.get(EvaluationContextKey.EVALUATED_IDS_KEY)
        if evaluated is not None and result.evaluated_expression_ids is None:
            result.evaluated_expression_ids = OrderedSet(evaluated)


class SatisfiedConditionTracker(EvaluationObserver):
    """Observer that tracks which condition expressions were satisfied during evaluation.

    Replaces the ad-hoc ``_carried_satisfied_ids_``, ``@captures_satisfied_conditions``,
    and inline propagation that was previously scattered across the evaluation pipeline.
    """

    def on_evaluate_enter(self, expression, sources):
        from krrood.entity_query_language.core.base_expressions import (
            OperationResult,
        )

        ctx = get_evaluation_context()
        if ctx is None:
            return

        satisfied = None
        if isinstance(sources, OperationResult):
            satisfied = sources.satisfied_condition_ids
        if satisfied is not None:
            ctx.data[EvaluationContextKey.SATISFIED_IDS_KEY] = satisfied

    def on_result_yielded(self, expression, result):
        ctx = get_evaluation_context()
        if ctx is None:
            return
        satisfied = ctx.data.get(EvaluationContextKey.SATISFIED_IDS_KEY)
        if satisfied is not None and result.satisfied_condition_ids is None:
            result.satisfied_condition_ids = satisfied

    def on_conclusions_processed(self, expression, result):

        if expression._conditions_root_ is not expression:
            return
        if result.is_false:
            return
        if expression._conditions_root_ is expression._root_:
            return

        ctx = get_evaluation_context()
        evaluated = ctx.data.get(EvaluationContextKey.EVALUATED_IDS_KEY) if ctx is not None else None
        if evaluated is None:
            return

        from krrood.entity_query_language.operators.core_logical_operators import (
            LogicalOperator,
        )
        from krrood.entity_query_language.exceptions import (
            NoExpressionFoundForGivenID,
        )

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
        if ctx is not None:
            ctx.data[EvaluationContextKey.SATISFIED_IDS_KEY] = satisfied


class InferenceRecorder(EvaluationObserver):
    """Observer that records inferred instances for later explanation.

    Replaces the ``@record_inferences`` decorator that was previously applied to
    ``InstantiatedVariable._instantiate_using_child_vars_and_yield_results_``.
    """

    def on_result_yielded(self, expression, result):
        from krrood.entity_query_language._monitoring import monitored
        if not monitored.is_monitored(type(expression)):
            return
        if expression._id_ not in result.bindings:
            return
        # Only record for InstantiatedVariable subclasses whose _evaluate__
        # delegates to _instantiate_using_child_vars_and_yield_results_ (i.e.
        # those that actually create new instances).  Query and its subclasses
        # (Entity, SetOf) override _evaluate__ and merely remap bindings
        # without creating new inferred instances.
        from krrood.entity_query_language.core.variable import (
            InstantiatedVariable,
        )
        from krrood.entity_query_language.query.query import Query

        if not isinstance(expression, InstantiatedVariable):
            return
        if isinstance(expression, Query):
            return
        from krrood.entity_query_language.explanation.explanation import register_inference

        register_inference(result.bindings[expression._id_], expression, result)
