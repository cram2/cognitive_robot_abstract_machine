"""
Evaluation context and observer protocol for the Entity Query Language.

Extracted into its own module so that :mod:`core.base_expressions` can import the
context infrastructure without pulling in the full :mod:`evaluation` module and the
circular dependency chain it carries.
"""

from __future__ import annotations

from abc import ABC
from contextvars import ContextVar
from dataclasses import dataclass, field

from typing_extensions import Any, Dict, List, Optional, TYPE_CHECKING

from krrood.entity_query_language.enums import EvaluationContextKey

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import (
        OperationResult,
        SymbolicExpression,
    )

_evaluation_context_var: ContextVar[Optional[EvaluationContext]] = ContextVar(
    "_evaluation_context", default=None
)


def get_evaluation_context() -> Optional[EvaluationContext]:
    """
    :return: The current :class:`EvaluationContext`, or ``None`` if called outside an active evaluation.
    """
    return _evaluation_context_var.get()


def set_evaluation_context(
    evaluation_context: Optional[EvaluationContext],
):
    """
    Set or clear the current evaluation context and return the reset token.

    :param evaluation_context: The context to set, or ``None`` to clear.
    :return: A :class:`contextvars.Token` that can be passed to
        :meth:`contextvars.ContextVar.reset` to restore the previous value.
    """
    return _evaluation_context_var.set(evaluation_context)


class EvaluationObserver(ABC):
    """Observer for evaluation events in the EQL evaluation pipeline."""

    def on_evaluate_enter(
        self, expression: SymbolicExpression, sources: Optional[OperationResult] = None
    ) -> None:
        """Called when entering an expression's _evaluate_ method."""

    def on_evaluate_exit(self, expression: SymbolicExpression) -> None:
        """Called when exiting an expression's _evaluate_ method."""

    def on_result_yielded(
        self, expression: SymbolicExpression, result: OperationResult
    ) -> None:
        """Called for each OperationResult yielded from _evaluate_."""

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
    Observers should use well-known keys defined in :class:`~krrood.entity_query_language.enums.EvaluationContextKey`
    to avoid collisions.
    """

    def on_evaluate_enter(
        self,
        *,
        expression: SymbolicExpression,
        sources: Optional[OperationResult] = None,
    ) -> None:
        """
        Notify all observers that evaluation of *expression* is about to begin.

        :param expression: The expression being entered.
        :param sources: The incoming :class:`OperationResult` carrying bindings, or ``None``.
        """
        for observer in self.observers:
            observer.on_evaluate_enter(expression, sources)

    def on_evaluate_exit(self, *, expression: SymbolicExpression) -> None:
        """
        Notify all observers that evaluation of *expression* has finished.

        :param expression: The expression that just finished evaluating.
        """
        for observer in self.observers:
            observer.on_evaluate_exit(expression)

    def on_result_yielded(
        self,
        *,
        expression: SymbolicExpression,
        result: OperationResult,
    ) -> None:
        """
        Notify all observers that *expression* has yielded *result*.

        :param expression: The expression that produced the result.
        :param result: The :class:`OperationResult` that was yielded.
        """
        for observer in self.observers:
            observer.on_result_yielded(expression, result)

    def on_conclusions_processed(
        self,
        *,
        expression: SymbolicExpression,
        result: OperationResult,
    ) -> None:
        """
        Notify all observers that conclusions have been processed for *expression*.

        :param expression: The expression whose conclusions were processed.
        :param result: The :class:`OperationResult` after conclusion processing.
        """
        for observer in self.observers:
            observer.on_conclusions_processed(expression, result)
