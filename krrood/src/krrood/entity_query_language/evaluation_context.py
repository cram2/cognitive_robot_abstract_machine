"""
Evaluation context and observer protocol for the Entity Query Language.

Extracted into its own module so that :mod:`core.base_expressions` can import the
context infrastructure without pulling in the full :mod:`evaluation` module and the
circular dependency chain it carries.
"""

from __future__ import annotations

import uuid
from abc import ABC
from contextvars import ContextVar
from dataclasses import dataclass, field

from ordered_set import OrderedSet
from typing_extensions import Iterator, List, Optional, Tuple, TYPE_CHECKING

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
    """
    Observer for evaluation events in the EQL evaluation pipeline.
    """

    def on_evaluate_enter(
        self, expression: SymbolicExpression, sources: Optional[OperationResult] = None
    ) -> None:
        """
        Called when entering an expression's _evaluate_ method.
        """

    def on_evaluate_exit(self, expression: SymbolicExpression) -> None:
        """
        Called when exiting an expression's _evaluate_ method.
        """

    def on_result_yielded(
        self, expression: SymbolicExpression, result: OperationResult
    ) -> None:
        """
        Called for each OperationResult yielded from _evaluate_.
        """

    def on_conclusions_processed(
        self, expression: SymbolicExpression, result: OperationResult
    ) -> None:
        """
        Called after _evaluate_conclusions_and_update_bindings_ completes.
        """


@dataclass
class ActiveConditionsRoot:
    """
    Tracks which node is the active conditions root for the current evaluation pass.

    A node reused as the condition of more than one ``Filter`` has no single correct
    "root" — the right answer depends on which evaluation is currently running, not on
    the node's construction history. The first node to claim this during a pass wins;
    nested evaluations within the same pass never reassign it.
    """

    _root_id: Optional[uuid.UUID] = field(default=None, init=False)
    """
    Identifier of the claimed root, or ``None`` before any node has claimed one this
    pass.
    """

    def claim(self, root: SymbolicExpression) -> None:
        """
        Claim *root* as the active conditions root for this pass, if none is claimed
        yet.

        :param root: The node to claim, normally ``self._conditions_root_`` of whichever
            expression is starting a fresh (context-less) evaluation.
        """
        if self._root_id is None:
            self._root_id = root._id_

    def is_active_root(self, node: SymbolicExpression) -> bool:
        """:return: ``True`` if *node* is the active conditions root for this pass."""
        return self._root_id == node._id_


@dataclass
class EvaluatedExpressionIds:
    """
    Tracks every expression id evaluated so far during the current evaluation pass.

    Used to distinguish evaluated-from-skipped logical operators (for example, short-
    circuited OR/AND branches) when building inference explanations.
    """

    _ids: OrderedSet = field(default_factory=OrderedSet, init=False)
    """
    The expression ids recorded as evaluated so far this pass.
    """

    _snapshot: Optional[Tuple[int, OrderedSet]] = field(default=None, init=False)
    """
    Cached ``(length, snapshot)`` pair.

    The id set is append-only, so its length is a valid version key: a snapshot taken
    while the set has a given length is reused instead of copying the whole set again.
    """

    def record(self, expression_id: uuid.UUID) -> None:
        """
        Record *expression_id* as evaluated during the current pass.
        """
        self._ids.add(expression_id)

    def merge(self, other: OrderedSet) -> None:
        """
        Merge *other* into the recorded ids (for example, ids evaluated by an earlier
        stage of the same result chain).
        """
        self._ids.update(other)

    def __iter__(self) -> Iterator[uuid.UUID]:
        return iter(self._ids)

    def snapshot(self) -> OrderedSet:
        """:return: An immutable snapshot of the ids recorded so far, reusing the cached one
        when the set has not grown since it was last taken."""
        current_length = len(self._ids)
        if self._snapshot is None or self._snapshot[0] != current_length:
            self._snapshot = (current_length, OrderedSet(self._ids))
        return self._snapshot[1]


@dataclass
class EvaluationContext:
    """
    Carries observer state through the evaluation pipeline.
    """

    observers: List[EvaluationObserver] = field(default_factory=list)
    """
    List of observers to notify of evaluation events.
    """

    active_conditions_root: ActiveConditionsRoot = field(
        default_factory=ActiveConditionsRoot
    )
    """
    Tracks which node is the active conditions root for the current evaluation pass.
    """

    evaluated_expression_ids: EvaluatedExpressionIds = field(
        default_factory=EvaluatedExpressionIds
    )
    """
    Tracks every expression id evaluated so far during the current evaluation pass.
    """

    satisfied_condition_ids: Optional[OrderedSet] = field(default=None)
    """
    The satisfied condition-expression ids for the current evaluation iteration, or
    ``None`` if unset.
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
        :param sources: The incoming :class:`OperationResult` carrying bindings, or
            ``None``.
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
